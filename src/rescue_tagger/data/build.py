"""Build the training files: raw sources -> data/processed/{train,dev,test}.jsonl.

  python -m rescue_tagger.data.build --config configs/data.yaml

Steps: download public sets (pinned revisions) -> map each to unified records -> assign fixed
splits for synthetic rows (hash of id, never random) -> drop training rows that leak eval text ->
validate every row -> write files + manifest.json (counts and sha256 per file).
The Harvey hold-out is written to its own file and is never part of train or dev.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from rescue_tagger.data import humaid, munro
from rescue_tagger.data.leakcheck import LeakIndex
from rescue_tagger.records import read_jsonl, validate_record, write_jsonl


def synthetic_split(rec_id: str, dev_pct: int, test_pct: int) -> str:
    bucket = int(hashlib.sha256(rec_id.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + dev_pct:
        return "dev"
    return "train"


def protected_texts(paths: list[str]) -> list[str]:
    """Texts from eval files: JSONL rows with raw_text or text."""
    out: list[str] = []
    for pattern in paths:
        expanded = os.path.expandvars(pattern)
        if "$" in expanded:
            raise SystemExit(
                f"protected_texts entry {pattern!r} needs an unset environment variable. Set "
                "EVAL_REPO to a checkout of Gauntlet-Hackathon/gauntletvshackathon so its eval "
                "sets are kept out of training (or remove the entry deliberately)."
            )
        for p in sorted(glob.glob(expanded)):
            for row in read_jsonl(Path(p)):
                t = row.get("raw_text") or row.get("text")
                if t:
                    out.append(t)
    return out


def build(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    raw_dir = root / cfg["raw_dir"]
    out_dir = root / cfg["out_dir"]
    src_cfg = cfg["sources"]
    rows: list[dict[str, Any]] = []

    if src_cfg["munro"]["enabled"]:
        munro.download(raw_dir)
        rows += list(munro.load(raw_dir))
    if src_cfg["humaid"]["enabled"]:
        humaid.download(raw_dir, token=os.environ.get("HF_TOKEN"))
        rows += list(humaid.load(raw_dir))
    if src_cfg["synthetic"]["enabled"]:
        split_cfg = src_cfg["synthetic"]["split_pct"]
        for p in sorted(glob.glob(str(root / src_cfg["synthetic"]["batches"]))):
            if p.endswith(".rejected.jsonl"):
                continue
            for rec in read_jsonl(Path(p)):
                rec["split"] = synthetic_split(rec["id"], split_cfg["dev"], split_cfg["test"])
                rows.append(rec)

    for rec in rows:
        w = src_cfg[rec["source"]].get("weight", 1.0)
        if w != 1.0:
            rec["weight"] = w

    # Everything that is evaluation-only is protected, plus configured external eval files.
    eval_only = [r for r in rows if r["split"] == "test"]
    index = LeakIndex(
        [r["text"] for r in eval_only] + protected_texts(cfg.get("protected_texts", [])),
        threshold=cfg.get("leak_threshold", 0.5),
    )
    dropped: list[str] = []
    kept: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rec in rows:
        if rec["id"] in seen_ids:
            raise ValueError(f"duplicate id {rec['id']}")
        seen_ids.add(rec["id"])
        if rec["split"] != "test" and index.leaks(rec["text"]):
            dropped.append(rec["id"])
            continue
        validate_record(rec)
        kept.append(rec)

    files: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": [], "harvey": []}
    for rec in kept:
        key = "harvey" if rec.get("meta", {}).get("harvey_holdout") else rec["split"]
        files[key].append(rec)

    manifest: dict[str, Any] = {"files": {}, "leak_dropped": len(dropped)}
    for name, recs in files.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, recs)
        manifest["files"][name] = {
            "rows": len(recs),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_source": dict(Counter(r["source"] for r in recs)),
            "by_lang": dict(Counter(r["lang"] for r in recs)),
        }
    (out_dir / "leak_dropped.txt").write_text("\n".join(dropped) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(cfg, args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
