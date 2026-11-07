"""Score a checkpoint on the product team's dev rows (sets B, C and D), head by head.

  python -m rescue_tagger.product_eval --model hf:paawandesai/rescue-tagger@<sha> [--model ...]

Dev rows pick checkpoints; this is the report the product team asked for. Gold values come from the
row's gold `extracted` payload, read the same way the trainer reads it (rescue_tagger.data.product):
- is_request: accuracy where the labeller was sure; on unsure rows (pranks, unclear posts, gold
  confidence < 0.5) the right answer is an unsure output (confidence < 0.75), and a confident
  "true" is counted separately as a joke taken seriously.
- location_type: accuracy against the trainer's target ("none" when the engine finds no usable
  location); `invented_location` counts rows with no usable location where we gave one.
- hazards, vulnerable, needs: micro F1 over kinds at confidence >= 0.5; a hazard the labeller was
  unsure of is left out of both sides.
- location_span: on rows with a labelled LOCATION span (set B), the location text we return is
  exactly that span (case-insensitive).
- p1_critical_kept: on gold P1 rows, every hazard and vulnerable kind the labeller marked is
  reported at >= 0.5 (a dropped one can cost the P1 rank).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rescue_tagger.data.product import UNSURE, to_record
from rescue_tagger.records import read_jsonl

ON = 0.5        # a kind counts as reported at this confidence
UNSURE_OUT = 0.75  # an is_request confidence below this flags the row as unsure (as in evaluate)


def _kinds(items: list[dict[str, Any]], floor: float = ON) -> set[str]:
    return {i["kind"] for i in items if i.get("confidence", 1.0) >= floor}


def _f1(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    return round(2 * tp / (2 * tp + fp + fn), 4)


def score(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, list[bool]] = {k: [] for k in ("is_request", "unsure_flagged", "location_type",
                                                  "request_kind", "people_count", "is_stale_post",
                                                  "p1_critical_kept", "location_span")}
    f1c = {h: [0, 0, 0] for h in ("hazards", "vulnerable", "needs")}
    invented = joke_true = 0
    for row, pay in zip(rows, preds, strict=True):
        g, rec = row["gold_extracted"], to_record(row)
        lab = rec["labels"]
        if "is_request" in lab:
            acc["is_request"].append(pay["is_request"]["value"] == lab["is_request"])
        else:
            unsure = pay["is_request"]["confidence"] < UNSURE_OUT
            acc["unsure_flagged"].append(unsure)
            joke_true += (not unsure) and pay["is_request"]["value"]
        acc["location_type"].append(pay["location"]["type"] == lab["location_type"])
        invented += lab["location_type"] == "none" and pay["location"]["type"] != "none"
        gold_loc = [s for s in (rec["spans"] or []) if s["label"] == "LOCATION"]
        if gold_loc and lab["location_type"] != "none":  # set B: labelled location span
            s = gold_loc[0]
            acc["location_span"].append(
                (pay["location"]["text"] or "").strip().lower()
                == row["raw_text"][s["start"]:s["end"]].strip().lower())
        acc["request_kind"].append(pay["request_kind"] == lab["request_kind"])
        acc["people_count"].append(pay["people_count"]["value"] == (rec["people_count"] or None))
        acc["is_stale_post"].append(pay["time_ref"]["is_stale_post"] == lab["is_stale_post"])
        unsure_h = {h["kind"] for h in g["hazards"] if h["confidence"] < UNSURE}
        gold = {"hazards": _kinds(g["hazards"], UNSURE) - unsure_h,
                "vulnerable": _kinds(g["vulnerable"], UNSURE),
                "needs": {n["kind"] for n in g["needs"]}}
        pred = {"hazards": _kinds(pay["hazards"]) - unsure_h,
                "vulnerable": _kinds(pay["vulnerable"]),
                "needs": {n["kind"] for n in pay["needs"]}}
        for h, c in f1c.items():
            c[0] += len(gold[h] & pred[h])
            c[1] += len(pred[h] - gold[h])
            c[2] += len(gold[h] - pred[h])
        if row["gold_decision"].get("priority_tier") == "P1":
            acc["p1_critical_kept"].append(gold["hazards"] <= pred["hazards"]
                                           and gold["vulnerable"] <= pred["vulnerable"])
    out: dict[str, Any] = {"n": len(rows)}
    for k, v in acc.items():
        if v:
            out[k] = round(sum(v) / len(v), 4)
            out[f"{k}_n"] = len(v)
    for h, (tp, fp, fn) in f1c.items():
        out[f"{h}_f1"] = _f1(tp, fp, fn)
    out["invented_location"] = invented
    out["joke_taken_as_confident_request"] = joke_true
    return out


def report(rows: list[dict[str, Any]], extract: Callable[[str], dict[str, Any]]
           ) -> dict[str, Any]:
    preds = [extract(r["raw_text"]) for r in rows]
    out = {"all": score(rows, preds)}
    for s in sorted({r["set"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["set"] == s]
        out[f"set_{s}"] = score([rows[i] for i in idx], [preds[i] for i in idx])
    return out


def load_model(spec: str):
    from rescue_tagger.infer import Extractor

    if spec.startswith("hf:"):
        from huggingface_hub import snapshot_download

        repo, _, rev = spec[3:].partition("@")
        return Extractor.load(snapshot_download(repo, revision=rev or None), model_version=spec)
    return Extractor.load(spec, model_version=spec)


def stored_extract(path: Path, rows: list[dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    """Stored LLM outputs (results/llm/<condition>.jsonl) as an extractor. A missing or failed
    output scores as compare.wrong_payload: wrong on every labelled field, and confident. Every
    row must be covered, so a run on the wrong split fails instead of scoring fallbacks."""
    from rescue_tagger.compare import wrong_payload

    stored = {r["id"]: r for r in read_jsonl(path) if r.get("set") == "product"}
    by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        rec = to_record(row)
        if rec["id"] not in stored:
            raise SystemExit(f"{path} has no output for {rec['id']}; run llm_baseline --sets "
                             "product first (dev rows only)")
        r = stored[rec["id"]]
        by_text[row["raw_text"]] = (r["extracted"] if r.get("valid") and r.get("extracted")
                                    else wrong_payload(rec))
    return by_text.__getitem__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--llm", action="append", default=[],
                    help="stored LLM outputs (results/llm/<condition>.jsonl); repeatable")
    ap.add_argument("--model", action="append", default=[],
                    help="model dir or hf:<repo>@<sha>; repeat to score several")
    ap.add_argument("--data", type=Path, default=Path("data/product/rescue_eval_train_dev.jsonl"))
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = [r for r in read_jsonl(args.data) if r["split"] == args.split]
    res = {m: report(rows, load_model(m).extract) for m in args.model}
    res.update({f"llm:{p}": report(rows, stored_extract(Path(p), rows)) for p in args.llm})
    text = json.dumps(res, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
