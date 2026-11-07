"""Deep check of the request detector on one labeled file, without exposing eval text.

  python -m rescue_tagger.diagnose <model_dir> --data data/processed/harvey.jsonl [--show-errors 0]

Accuracy alone hides a lot when requests are rare (Harvey: 3.5% of posts), so this reports:
  - the majority baseline, precision / recall / F1 on the request class, and the confusion matrix
  - the same at each engine threshold (0.25 / 0.5 / 0.6 / 0.75)
  - how often each upstream category (HumAID class_label) is predicted a request
  - reliability bins for is_request confidence, and results by message length
`--show-errors N` prints N false negatives and N false positives. Only use it on train/dev
files: reading test or Harvey errors and then changing data would tune the model to the test set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rescue_tagger.infer import Extractor
from rescue_tagger.records import read_jsonl

THRESHOLDS = (0.25, 0.5, 0.6, 0.75)
EVAL_ONLY = ("harvey", "test")


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def diagnose(ex: Extractor, rows: list[dict[str, Any]], show_errors: int = 0) -> dict[str, Any]:
    scored = []
    for rec in rows:
        if "is_request" not in rec["labels"]:
            continue
        p = ex.raw(rec["text"])["probs"]["is_request"]
        scored.append((rec, p, bool(rec["labels"]["is_request"])))
    n = len(scored)
    pos = sum(y for _, _, y in scored)
    out: dict[str, Any] = {"n": n, "requests": pos,
                           "majority_baseline_accuracy": round(max(pos, n - pos) / max(n, 1), 4)}
    for t in THRESHOLDS:
        tp = sum(1 for _, p, y in scored if p >= t and y)
        fp = sum(1 for _, p, y in scored if p >= t and not y)
        fn = sum(1 for _, p, y in scored if p < t and y)
        tn = n - tp - fp - fn
        out[f"at_{t}"] = {"accuracy": round((tp + tn) / max(n, 1), 4), **prf(tp, fp, fn),
                          "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # rows, predicted req, gold req
    for rec, p, y in scored:
        c = by_cat[rec.get("meta", {}).get("class_label") or rec["source"]]
        c[0] += 1
        c[1] += p >= 0.5
        c[2] += y
    out["by_category"] = {k: {"rows": v[0], "predicted_request_pct": round(100 * v[1] / v[0], 1),
                              "gold_request_pct": round(100 * v[2] / v[0], 1)}
                          for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1][0])}

    bins: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for _, p, y in scored:
        conf, ok = max(p, 1 - p), (p >= 0.5) == y
        b = min(int(conf * 10), 9)
        bins[b][0] += 1
        bins[b][1] += ok
    out["reliability"] = {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {
        "rows": v[0], "accuracy": round(v[1] / v[0], 3)} for b, v in sorted(bins.items())}

    by_len: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # tp fp fn rows
    for rec, p, y in scored:
        words = len(rec["text"].split())
        k = "<10 words" if words < 10 else "10-25 words" if words <= 25 else ">25 words"
        s = by_len[k]
        s[0] += p >= 0.5 and y
        s[1] += p >= 0.5 and not y
        s[2] += p < 0.5 and y
        s[3] += 1
    out["by_length"] = {k: {"rows": v[3], **prf(v[0], v[1], v[2])} for k, v in by_len.items()}
    out["by_lang"] = dict(Counter(rec["lang"] for rec, _, _ in scored))

    if show_errors:
        fns = sorted((x for x in scored if x[2] and x[1] < 0.5), key=lambda x: x[1])
        fps = sorted((x for x in scored if not x[2] and x[1] >= 0.5), key=lambda x: -x[1])
        out["false_negatives"] = [{"p": round(p, 3), "source": r["source"],
                                   "category": r.get("meta", {}).get("class_label"),
                                   "text": r["text"][:280]} for r, p, _ in fns[:show_errors]]
        out["false_positives"] = [{"p": round(p, 3), "source": r["source"],
                                   "category": r.get("meta", {}).get("class_label"),
                                   "text": r["text"][:280]} for r, p, _ in fps[:show_errors]]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--show-errors", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.show_errors and any(s in args.data.name for s in EVAL_ONLY):
        raise SystemExit("--show-errors is for train/dev files only; eval text stays unread")
    rows = [r for r in read_jsonl(args.data) if args.source in (None, r["source"])]
    if args.show_errors and any(r["split"] not in ("train", "dev")
                                or r.get("meta", {}).get("harvey_holdout") for r in rows):
        raise SystemExit("--show-errors found eval-only rows; eval text stays unread")
    report = diagnose(Extractor.load(args.model_dir), rows, args.show_errors)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
