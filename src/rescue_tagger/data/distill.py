"""Rubric labels for real training tweets from Opus (distillation; training time only).

  python -m rescue_tagger.data.distill --n 2500 --out data/distill/d0001-opus.jsonl

HumAID and Munro give real phrasing with labels that follow their own conventions (donation
appeals count as requests; needs are read loosely), and the synthetic batches follow our rubric
but are not real text. This asks Opus (the structured condition of llm_baseline, the same rubric
prompt) to label real tweets drawn from the TRAIN split of Munro and HumAID only, never dev,
test, Harvey, edge or product rows, and writes them as records with source "distill" and split
"train". The build drops the original Munro/HumAID row of each distilled text, so a text is never
trained on with two different labels. The deployed model stays offline: Opus is a teacher only.

Labels: is_request (soft 0.5 when Opus is below 0.6 confident, the rubric's "unsure"), request_kind
(requests only), hazards and vulnerable kinds at confidence >= 0.5, needs, location_type,
is_stale_post and people_count. Spans are left unknown (Opus quotes only some of them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rescue_tagger import labels as L
from rescue_tagger.records import read_jsonl, validate_record

SOFT_BELOW = 0.6     # Opus is_request confidence under this -> soft 0.5 target
KIND_ON = 0.5


def pick(train: Path, n: int, salt: str = "distill-v1") -> list[dict[str, Any]]:
    """n train rows from Munro and HumAID, half labelled request and half not (by the source's
    own label), ordered by a hash of the id so the sample is fixed and reproducible."""
    rows = [r for r in read_jsonl(train)
            if r["source"] in ("munro", "humaid") and r["split"] == "train" and r["text"].strip()]
    key = lambda r: hashlib.sha256(f"{salt}:{r['id']}".encode()).hexdigest()  # noqa: E731
    req = sorted((r for r in rows if r["labels"].get("is_request") is True), key=key)
    other = sorted((r for r in rows if r["labels"].get("is_request") is not True), key=key)
    return req[: n // 2] + other[: n - n // 2]


def to_record(src: dict[str, Any], pay: dict[str, Any]) -> dict[str, Any]:
    conf = float(pay["is_request"]["confidence"])
    is_req = bool(pay["is_request"]["value"])
    labels: dict[str, Any] = {
        "hazards": {k: any(i["kind"] == k and i["confidence"] >= KIND_ON for i in pay["hazards"])
                    for k in L.HAZARD_KINDS},
        "vulnerable": {k: any(i["kind"] == k and i["confidence"] >= KIND_ON
                              for i in pay["vulnerable"]) for k in L.VULNERABLE_KINDS},
        "needs": {k: any(i["kind"] == k for i in pay["needs"]) for k in L.NEED_KINDS},
        "location_type": pay["location"]["type"],
        "is_stale_post": bool(pay["time_ref"]["is_stale_post"]),
    }
    rec: dict[str, Any] = {
        "id": f"distill:{src['id']}", "text": src["text"], "lang": src.get("lang", "en"),
        "source": "distill", "split": "train", "labels": labels, "spans": None,
        "meta": {"teacher": "opus-structured", "original_id": src["id"],
                 "is_request_confidence": conf},
    }
    if conf < SOFT_BELOW:
        rec["soft_labels"] = {"is_request": 0.5}
    else:
        labels["is_request"] = is_req
        if is_req:
            labels["request_kind"] = pay["request_kind"]
    if pay["people_count"]["value"] is not None:
        rec["people_count"] = pay["people_count"]["value"]
    validate_record(rec)
    return rec


def main(argv: list[str] | None = None) -> int:
    from rescue_tagger import llm_baseline as B

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    todo = pick(args.train, args.n)
    done = ({r["meta"]["original_id"] for r in read_jsonl(args.out)} if args.out.exists()
            else set())
    todo = [r for r in todo if r["id"] not in done]
    print(f"{len(todo)} rows to label ({len(done)} done)", flush=True)
    prompts, opus, lock = B.build_prompts(), B.Opus(), threading.Lock()
    usage: dict[str, int] = {}
    errors = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def work(src: dict[str, Any]) -> None:
        nonlocal errors
        spent: dict[str, Any] = {}
        try:
            res = B.run_one(opus, "structured", src["text"], prompts, spent)
            rec = to_record(src, B.to_payload(res["answer"], src["text"]))
        except Exception as e:  # noqa: BLE001 - a failed row is skipped and counted
            with lock:
                errors += 1
                print(f"  skip {src['id']}: {type(e).__name__}: {e}"[:200], flush=True)
            rec = None
        with lock:
            for k, v in spent.get("usage", {}).items():
                usage[k] = usage.get(k, 0) + v
            if rec:
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s) usage={usage}", flush=True)
    print(json.dumps({"labelled": len(todo) - errors, "errors": errors, "usage": usage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
