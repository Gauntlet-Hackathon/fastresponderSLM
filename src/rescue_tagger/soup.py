"""Weight averaging ("model soup") of fine-tuned runs that share a backbone and head layout.

  python -m rescue_tagger.soup --out outputs/soup hf:paawandesai/rescue-tagger@<sha> <dir> ...
  python -m rescue_tagger.soup --out outputs/soup --greedy <best> <next> ...

Every run starts from the same pretrained backbone, so their fine-tuned weights sit in one basin
and their average is a single model with the same speed (Wortsman et al., 2022). Uniform: average
all given runs. Greedy: start from the first run (list them best first, by dev score) and keep
each next run only if the averaged model's dev score does not drop; the dev score is the
mean of the real-text score (is_request accuracy, needs F1, request_kind accuracy) on the
select half of real dev and the product dev rows' is_request accuracy and P1-kept rate.

Dev rows only: test, Harvey and edge never pick ingredients. The soup is re-calibrated on the
calib half of dev (the averaged logits are not calibrated by any ingredient's temperatures).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def resolve(spec: str) -> Path:
    if spec.startswith("hf:"):
        from huggingface_hub import snapshot_download

        repo, _, rev = spec[3:].partition("@")
        return Path(snapshot_download(repo, revision=rev or None))
    return Path(spec)


def _tensors(d: Path) -> dict[str, dict[str, torch.Tensor]]:
    return {"encoder": load_file(d / "encoder" / "model.safetensors"),
            "heads": load_file(d / "heads.safetensors")}


def average(dirs: list[Path], out: Path) -> Path:
    """Uniform average of the encoder and head weights of dirs, written as a model dir."""
    cfg0 = json.loads((dirs[0] / "tagger_config.json").read_text())
    for d in dirs[1:]:
        if json.loads((d / "tagger_config.json").read_text()) != cfg0:
            raise ValueError(f"{d} has a different tagger config; only same-layout runs mix")
    acc: dict[str, dict[str, torch.Tensor]] = {}
    for d in dirs:
        for part, ts in _tensors(d).items():
            tgt = acc.setdefault(part, {})
            for k, v in ts.items():
                if k in tgt and tgt[k].shape != v.shape:
                    raise ValueError(f"{d}: {part}/{k} shape {v.shape} vs {tgt[k].shape}")
                tgt[k] = tgt[k] + v.float() if k in tgt else v.float().clone()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dirs[0], out, ignore=shutil.ignore_patterns("calibration.json", "*.md"))
    first = _tensors(dirs[0])
    for part, ts in acc.items():
        avg = {k: (v / len(dirs)).to(first[part][k].dtype).contiguous() for k, v in ts.items()}
        path = (out / "encoder" / "model.safetensors" if part == "encoder"
                else out / "heads.safetensors")
        save_file(avg, path, metadata={"format": "pt"})
    (out / "soup.json").write_text(json.dumps({"ingredients": [str(d) for d in dirs]}, indent=2))
    return out


def dev_score(model_dir: Path, real_rows: int = 800) -> dict[str, Any]:
    """Dev-only selection score (see the module docstring)."""
    import random

    from rescue_tagger.calibrate import calibrate
    from rescue_tagger.infer import Extractor
    from rescue_tagger.product_eval import score
    from rescue_tagger.records import dev_half, read_jsonl

    calibrate(model_dir, Path("data/processed/dev.jsonl"), half="calib")
    ex = Extractor.load(model_dir)
    real = [r for r in read_jsonl(Path("data/processed/dev.jsonl"))
            if r["source"] in ("munro", "humaid") and dev_half(r["id"]) == "select"]
    random.Random(0).shuffle(real)
    real = real[:real_rows]
    ok = n = 0
    tp = fp = fn = 0
    for r in real:
        p = ex.extract(r["text"])
        lab = r["labels"]
        if "is_request" in lab:
            ok += p["is_request"]["value"] == lab["is_request"]
            n += 1
        if "needs" in lab:
            gold = {k for k, v in lab["needs"].items() if v}
            pred = {i["kind"] for i in p["needs"]}
            tp, fp, fn = tp + len(gold & pred), fp + len(pred - gold), fn + len(gold - pred)
    prod_rows = [r for r in read_jsonl(Path("data/product/rescue_eval_train_dev.jsonl"))
                 if r["split"] == "dev"]
    prod = score(prod_rows, [ex.extract(r["raw_text"]) for r in prod_rows])
    parts = {"real_is_request": ok / max(n, 1),
             "real_needs_f1": 2 * tp / max(2 * tp + fp + fn, 1),
             "product_is_request": prod.get("is_request", 0.0),
             "product_p1_kept": prod.get("p1_critical_kept", 0.0)}
    return {"score": round(sum(parts.values()) / len(parts), 4),
            **{k: round(v, 4) for k, v in parts.items()}}


def greedy(dirs: list[Path], out: Path) -> dict[str, Any]:
    kept = [dirs[0]]
    best = dev_score(average(kept, out))
    log = [{"add": str(dirs[0]), "kept": True, **best}]
    for d in dirs[1:]:
        s = dev_score(average(kept + [d], out))
        keep = s["score"] >= best["score"]
        log.append({"add": str(d), "kept": keep, **s})
        print(json.dumps(log[-1]), file=sys.stderr)
        if keep:
            kept, best = kept + [d], s
    average(kept, out)
    dev_score(out)  # re-calibrate the final soup
    return {"kept": [str(d) for d in kept], "score": best, "log": log}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("models", nargs="+", help="model dirs or hf:<repo>@<sha>, best first")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args(argv)
    dirs = [resolve(m) for m in args.models]
    res = greedy(dirs, args.out) if args.greedy else {"out": str(average(dirs, args.out))}
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
