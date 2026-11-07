"""Model-only evaluation on a labeled JSONL split: field accuracy, calibration, speed, consistency.

  python -m rescue_tagger.evaluate outputs/laya/best --data data/processed/test.jsonl

Scores every head a row has labels for (so Munro rows score is_request/needs, synthetic rows score
everything), reports ECE for the confidence-carrying fields, median CPU latency per request, time
per 1,000 requests, and the share of cases whose JSON changes across repeated runs (expected 0).
The decision-level scores (P1 misses, abstain precision/recall) live in the product repo's eval
runner, which calls this model through the field adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from rescue_tagger import labels as L
from rescue_tagger.calibrate import ece
from rescue_tagger.infer import Extractor
from rescue_tagger.records import dev_half, read_jsonl


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def span_text(rec: dict[str, Any], label: str) -> str | None:
    for s in rec["spans"] or []:
        if s["label"] == label:
            return rec["text"][s["start"]:s["end"]]
    return None


LATENCY_ROWS = 200     # fixed benchmark: the first rows of the slice
LATENCY_PASSES = 2     # extra timing passes; the reported latency is the min of the medians


def cpu_name() -> str:
    """CPU model and count, so a latency number can be read against the container it came from."""
    name = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{name} x{os.cpu_count()}"


def evaluate(ex: Extractor, rows: list[dict[str, Any]], repeats: int = 10,
             consistency_n: int = 50, latency_passes: int = LATENCY_PASSES) -> dict[str, Any]:
    stats: dict[str, Any] = {h: {"correct": 0, "n": 0} for h in
                             ("is_request", "is_stale_post", "request_kind", "location_type",
                              "people_count")}
    multi = {h: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for h in L.MULTI_LABEL_HEADS}
    loc = {"exact": 0, "partial": 0, "n": 0}
    conf: dict[str, list[tuple[float, bool]]] = {h: [] for h in ("is_request", "location",
                                                                  "hazards", "vulnerable")}
    gold_counts: dict[str, Counter] = {h: Counter() for h in stats}
    latencies = []
    vague: list[bool] = []  # rows too vague to call: flagged as unsure (is_request conf < 0.75)?
    for rec in rows:
        t0 = time.perf_counter()
        pay = ex.extract(rec["text"])
        latencies.append(time.perf_counter() - t0)
        lab = rec["labels"]
        if "is_request" in (rec.get("soft_labels") or {}):
            vague.append(pay["is_request"]["confidence"] < 0.75)
        if "is_request" in lab:
            ok = pay["is_request"]["value"] == lab["is_request"]
            stats["is_request"]["correct"] += ok
            stats["is_request"]["n"] += 1
            gold_counts["is_request"][lab["is_request"]] += 1
            conf["is_request"].append((pay["is_request"]["confidence"], ok))
        if "is_stale_post" in lab:
            stats["is_stale_post"]["correct"] += pay["time_ref"]["is_stale_post"] == lab[
                "is_stale_post"]
            stats["is_stale_post"]["n"] += 1
            gold_counts["is_stale_post"][lab["is_stale_post"]] += 1
        if "request_kind" in lab:
            stats["request_kind"]["correct"] += pay["request_kind"] == lab["request_kind"]
            stats["request_kind"]["n"] += 1
            gold_counts["request_kind"][lab["request_kind"]] += 1
        if "location_type" in lab:
            ok = pay["location"]["type"] == lab["location_type"]
            stats["location_type"]["correct"] += ok
            stats["location_type"]["n"] += 1
            gold_counts["location_type"][lab["location_type"]] += 1
            conf["location"].append((pay["location"]["confidence"], ok))
        if rec["spans"] is not None and "location_type" in lab and lab["location_type"] != "none":
            gold, got = span_text(rec, "LOCATION"), pay["location"]["text"]
            loc["n"] += 1
            if gold and got:
                loc["exact"] += got == gold
                loc["partial"] += bool(set(got.lower().split()) & set(gold.lower().split()))
        if "people_count" in rec:
            stats["people_count"]["correct"] += pay["people_count"]["value"] == rec["people_count"]
            stats["people_count"]["n"] += 1
            gold_counts["people_count"][rec["people_count"]] += 1
        for head in L.MULTI_LABEL_HEADS:
            if head not in lab:
                continue
            gold = {k for k, v in lab[head].items() if v}
            items = pay[head]
            if head == "needs":
                pred = {i["kind"] for i in items}
            else:
                pred = {i["kind"] for i in items if i["confidence"] >= 0.5}
                for i in items:
                    conf[head].append((i["confidence"], i["kind"] in gold))
            multi[head]["n"] += 1
            multi[head]["tp"] += len(pred & gold)
            multi[head]["fp"] += len(pred - gold)
            multi[head]["fn"] += len(gold - pred)

    # Shared CPUs are noisy: one busy neighbour can add 50% to a pass. Time the same fixed rows a
    # few more times and keep the fastest median; contention only ever adds time.
    medians = [statistics.median(latencies[:LATENCY_ROWS])] if latencies else []
    for _ in range(latency_passes if latencies else 0):
        t = []
        for rec in rows[:LATENCY_ROWS]:
            t0 = time.perf_counter()
            ex.extract(rec["text"])
            t.append(time.perf_counter() - t0)
        medians.append(statistics.median(t))

    changed = 0
    for rec in rows[:consistency_n]:
        first = json.dumps(ex.extract(rec["text"]), sort_keys=True)
        if any(json.dumps(ex.extract(rec["text"]), sort_keys=True) != first
               for _ in range(repeats - 1)):
            changed += 1

    report: dict[str, Any] = {"n_rows": len(rows)}
    for h, s in stats.items():
        if s["n"]:
            report[f"{h}_accuracy"] = round(s["correct"] / s["n"], 4)
            report[f"{h}_n"] = s["n"]
            # "Always guess the most common answer": a model at or below this has learned nothing.
            report[f"{h}_majority_baseline"] = round(
                gold_counts[h].most_common(1)[0][1] / s["n"], 4)
    for h, s in multi.items():
        if s["tp"] + s["fp"] + s["fn"]:
            report[f"{h}_f1"] = round(f1(s["tp"], s["fp"], s["fn"]), 4)
            report[f"{h}_n"] = s["n"]  # rows labeled for this head
            # Gold-positive items behind the F1 (labels only, the same for every model). Reported
            # for reading; the gates do not use it.
            report[f"{h}_pos"] = s["tp"] + s["fn"]
    if loc["n"]:
        report["location_exact"] = round(loc["exact"] / loc["n"], 4)
        report["location_partial"] = round(loc["partial"] / loc["n"], 4)
    for h, pairs in conf.items():
        if len(pairs) >= 20:
            c = torch.tensor([p for p, _ in pairs])
            ok = torch.tensor([o for _, o in pairs])
            report[f"{h}_ece"] = round(ece(c, ok)[0], 4)
    # Since 2026-11-07: min of the medians over the first 200 rows (earlier: one pass, all rows).
    if vague:  # reported for reading, not a selection gate
        report["vague_flagged_unsure"] = round(sum(vague) / len(vague), 4)
        report["vague_n"] = len(vague)
    report["latency_median_ms"] = round(min(medians) * 1000, 2)
    report["latency_first_pass_ms"] = round(statistics.median(latencies) * 1000, 2)
    report["cpu"] = cpu_name()
    report["seconds_per_1000"] = round(sum(latencies) / len(latencies) * 1000, 2)
    report["consistency_changed_pct"] = round(100 * changed / max(min(consistency_n, len(rows)),
                                                                  1), 2)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--source", default=None, help="only rows from this source")
    ap.add_argument("--exclude-source", default=None, help="drop rows from this source")
    ap.add_argument("--half", choices=["calib", "select"], default=None,
                    help="only rows in this fixed half of dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = [r for r in read_jsonl(args.data) if args.source in (None, r["source"])
            and r["source"] != args.exclude_source
            and (args.half is None or dev_half(r["id"]) == args.half)]
    rows = rows[: args.limit] if args.limit else rows
    report = evaluate(Extractor.load(args.model_dir), rows)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
