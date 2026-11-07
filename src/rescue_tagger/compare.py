"""One table: our model against Opus 5.5 under three prompt conditions, on the same rows.

    python -m rescue_tagger.compare [--slm outputs/laya/best] [--out compare.json]

Rows: the fixed sample in evals/compare_sample.json (balanced requests / non-requests per source
from test and Harvey, picked by a hash of the id) plus every edge case in evals/edge_cases.jsonl.
Test and Harvey are scored twice: upstream labels (primary) and rubric labels (secondary), the same
split the morning check reports. Opus outputs come from results/llm/<condition>.jsonl
(rescue_tagger.llm_baseline); a missing or invalid output scores as "found nothing".

This is a report, never a selection signal: nothing here may pick a run, a threshold or a prompt.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rescue_tagger import labels as L
from rescue_tagger.evaluate import evaluate, f1
from rescue_tagger.llm_baseline import (
    CONDITIONS,
    RESULTS,
    SAMPLE_FILE,
    read_results,
    to_payload,
)
from rescue_tagger.records import read_jsonl

# $ per million tokens, Opus 5.5 standard tier.
PRICE = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_read_input_tokens": 0.2,
         "cache_creation_input_tokens": 5.0}
UNSURE = 0.75  # an is_request confidence below this counts as "flagged as unsure"


def wrong_payload(rec: dict[str, Any]) -> dict[str, Any]:
    """What a missing or failed output scores as: wrong on every labeled field, and confident,
    so a failure never earns credit for "not a request" or "unsure"."""
    lab = rec["labels"]
    gold_kind = lab.get("request_kind")
    gold_loc = lab.get("location_type")
    pc = rec.get("people_count")
    flip = {head: [k for k, v in lab[head].items() if not v] for head in L.MULTI_LABEL_HEADS
            if head in lab}
    return to_payload({
        "is_request": not lab.get("is_request", False), "is_request_confidence": 1.0,
        "request_kind": next(k for k in L.REQUEST_KINDS if k != gold_kind),
        "is_stale_post": not lab.get("is_stale_post", False),
        "location_type": next(k for k in L.LOCATION_TYPES if k not in (gold_loc, "none")),
        "location_text": "?", "location_confidence": 1.0,
        "people_count": (pc or 0) + 1, "people_count_confidence": 1.0,
        "hazards": [{"kind": k, "confidence": 1.0} for k in flip.get("hazards", [])],
        "vulnerable": [{"kind": k, "confidence": 1.0} for k in flip.get("vulnerable", [])],
        "needs": [{"kind": k, "hours_without": None} for k in flip.get("needs", [])],
    }, rec["text"])


class Stored:
    """Looks like an Extractor; returns pre-computed payloads by text (texts are unique across
    the comparison rows). Rows with no usable output get wrong_payload()."""

    def __init__(self, by_text: dict[str, dict[str, Any]],
                 rows: list[dict[str, Any]] | None = None) -> None:
        self.by_text = dict(by_text)
        for r in rows or []:
            if not self.by_text.get(r["text"]):
                self.by_text[r["text"]] = wrong_payload(r)

    def extract(self, text: str) -> dict[str, Any]:
        return self.by_text[text]


def load_sets(processed: Path) -> dict[str, list[dict[str, Any]]]:
    ids = json.loads(SAMPLE_FILE.read_text())
    out: dict[str, list[dict[str, Any]]] = {}
    for name, files in (("test", ("test.jsonl", "test_rubric.jsonl")),
                        ("harvey", ("harvey.jsonl", "harvey_rubric.jsonl"))):
        want = set(ids[name])
        for suffix, fname in zip(("", "_rubric"), files, strict=True):
            path = processed / fname
            if path.exists():
                out[name + suffix] = [r for r in read_jsonl(path) if r["id"] in want]
    out["edge"] = list(read_jsonl(Path("evals/edge_cases.jsonl")))
    return out


def request_prf(ex: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = 0
    for r in rows:
        if "is_request" not in r["labels"]:
            continue
        pred, gold = ex.extract(r["text"])["is_request"]["value"], r["labels"]["is_request"]
        tp += pred and gold
        fp += pred and not gold
        fn += gold and not pred
    return {"request_precision": round(tp / max(tp + fp, 1), 4),
            "request_recall": round(tp / max(tp + fn, 1), 4),
            "request_f1": round(f1(tp, fp, fn), 4)}


def edge_checks(rec: dict[str, Any], pay: dict[str, Any]) -> bool:
    """A case passes when every field it labels is exactly right."""
    lab = rec["labels"]
    if "is_request" in lab and pay["is_request"]["value"] != lab["is_request"]:
        return False
    if "is_stale_post" in lab and pay["time_ref"]["is_stale_post"] != lab["is_stale_post"]:
        return False
    if "request_kind" in lab and pay["request_kind"] != lab["request_kind"]:
        return False
    if "location_type" in lab and pay["location"]["type"] != lab["location_type"]:
        return False
    for head in L.MULTI_LABEL_HEADS:
        if head in lab:
            gold = {k for k, v in lab[head].items() if v}
            items = pay[head]
            pred = {i["kind"] for i in items if head == "needs" or i["confidence"] >= 0.5}
            if pred != gold:
                return False
    if "people_count" in rec and pay["people_count"]["value"] != rec["people_count"]:
        return False
    hours = {n["kind"]: n["hours_without"] for n in pay["needs"]}
    for kind, h in (rec.get("need_hours") or {}).items():
        if h is not None and (hours.get(kind) is None or abs(hours[kind] - h) > 0.5):
            return False
    return True


def edge_report(ex: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    unsure = []
    for r in rows:
        pay = ex.extract(r["text"])
        if r.get("ambiguous"):
            unsure.append(pay["is_request"]["confidence"] < UNSURE)
        else:
            by_cat[r["category"]].append(edge_checks(r, pay))
    allc = [ok for v in by_cat.values() for ok in v]
    return {"edge_pass_rate": round(sum(allc) / max(len(allc), 1), 4),
            "edge_flags_unsure": round(sum(unsure) / max(len(unsure), 1), 4),
            "edge_by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())}}


def llm_systems(rows_by_set: dict[str, list[dict[str, Any]]],
                allow_partial: bool = False) -> dict[str, dict[str, Any]]:
    text_of = {r["id"]: r["text"] for rows in rows_by_set.values() for r in rows}
    out = {}
    for cond in CONDITIONS:
        res = read_results(RESULTS / f"{cond}.jsonl")
        missing = set(text_of) - set(res)
        if missing and not allow_partial:
            raise SystemExit(f"results/llm/{cond}.jsonl covers {len(text_of) - len(missing)} of "
                             f"{len(text_of)} comparison rows; finish the run or pass "
                             "--allow-partial (missing rows then score as wrong)")
        if not res:
            continue
        by_text = {text_of[i]: r["extracted"] for i, r in res.items()
                   if r["valid"] and i in text_of}
        usage: dict[str, float] = defaultdict(float)
        for r in res.values():
            for k, v in (r.get("usage") or {}).items():
                usage[k] += v
        cost = sum(usage[k] * p for k, p in PRICE.items()) / 1e6
        n = len(res)
        out[f"opus_{cond}"] = {
            "ex": Stored(by_text),
            "meta": {"coverage": f"{len(text_of) - len(missing)}/{len(text_of)}",
                     "invalid_or_failed": sum(not r["valid"] for r in res.values()),
                     "latency_median_ms": round(statistics.median(
                         r["latency_ms"] for r in res.values()), 1),
                     "usd_per_1000": round(1000 * cost / max(n, 1), 2),
                     "usd_total": round(cost, 2), "works_offline": False},
        }
    return out


def run(slm: Path | None, processed: Path, allow_partial: bool = False) -> dict[str, Any]:
    sets = load_sets(processed)
    systems = llm_systems(sets, allow_partial)
    if slm is not None:
        from rescue_tagger.infer import Extractor

        ex = Extractor.load(slm)
        by_text = {r["text"]: ex.extract(r["text"]) for rows in sets.values() for r in rows}
        lat = evaluate(ex, sets["edge"], repeats=10, consistency_n=20)
        systems = {"ours": {"ex": Stored(by_text), "meta": {
            "coverage": f"{len(by_text)}/{len(by_text)}", "invalid_or_failed": 0,
            "latency_median_ms": lat["latency_median_ms"],
            "consistency_changed_pct": lat["consistency_changed_pct"],
            "usd_per_1000": 0.0, "works_offline": True, "model": str(slm)}}, **systems}
    report: dict[str, Any] = {"sets": {k: len(v) for k, v in sets.items()}, "systems": {}}
    for name, sysd in systems.items():
        rep: dict[str, Any] = {"meta": sysd["meta"]}
        for set_name, rows in sets.items():
            ex = Stored(sysd["ex"].by_text, rows)
            if set_name == "edge":
                ev = evaluate(ex, rows, consistency_n=0, latency_passes=0)
                rep[set_name] = {**edge_report(ex, rows),
                                 **{k: v for k, v in ev.items()
                                    if k.endswith(("_f1", "_accuracy", "location_partial"))}}
            else:
                ev = evaluate(ex, rows, consistency_n=0, latency_passes=0)
                rep[set_name] = {**request_prf(ex, rows),
                                 **{k: v for k, v in ev.items()
                                    if k.endswith(("_f1", "_accuracy"))}}
        report["systems"][name] = rep
    return report


ROWS = [("test", "request_f1"), ("test", "request_recall"), ("test", "request_precision"),
        ("test", "needs_f1"), ("test", "request_kind_accuracy"),
        ("test_rubric", "request_f1"), ("harvey", "request_f1"), ("harvey", "request_recall"),
        ("harvey_rubric", "request_f1"), ("edge", "edge_pass_rate"),
        ("edge", "edge_flags_unsure"), ("edge", "hazards_f1"), ("edge", "vulnerable_f1"),
        ("edge", "needs_f1"), ("edge", "location_type_accuracy"), ("edge", "location_partial"),
        ("meta", "coverage"), ("meta", "invalid_or_failed"),
        ("meta", "latency_median_ms"), ("meta", "usd_per_1000"), ("meta", "works_offline")]


def markdown(report: dict[str, Any]) -> str:
    names = list(report["systems"])
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for set_name, key in ROWS:
        cells = []
        for n in names:
            v = report["systems"][n].get(set_name, {}).get(key, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) and key != "latency_median_ms"
                         and not key.startswith("usd") else str(v))
        lines.append(f"| {set_name} {key} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Edge cases by category (passed/total):")
    for n in names:
        lines.append(f"- {n}: " + ", ".join(f"{c} {v}" for c, v in
                                            report["systems"][n]["edge"]["edge_by_category"].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--slm", type=Path, default=None, help="our checkpoint dir (best/)")
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score incomplete Opus runs (missing rows count as wrong)")
    args = ap.parse_args(argv)
    report = run(args.slm, args.processed, args.allow_partial)
    md = markdown(report)
    print(md)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        args.out.with_suffix(".md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
