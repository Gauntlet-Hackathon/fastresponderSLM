"""Product team's hand-reviewed eval rows (sets B, C and D, train and dev splits only).

  data/product/rescue_eval_train_dev.jsonl  ->  training records (source "product")

The product team holds back the test split of these sets; it is never in this repo and must never
be recreated. Train rows are gradient data; dev rows only pick checkpoints and are scored by
`rescue_tagger.product_eval`. All text was written by hand for the project (no tweets).

Mapping from the row's gold `extracted` payload (the same schema infer.py emits):
- Confidences are labelling conventions (0.9 clear, 0.5-0.6 unsure, 0.1 absent), not probabilities.
  Value heads train on values.
- is_request below 0.5 confidence (pranks, unclear posts) becomes a soft 0.5 target, as for the
  synthetic vague rows: the payload then reports confidence near 0.5, the lowest it can express
  (confidence is max(p, 1 - p)), never confidently true for a joke and never a confident "no".
- A low-confidence location the engine rejects as unusable ("the church with the green roof",
  abstain reason no_usable_location) is "none", so the engine asks instead of sending a boat.
  A low-confidence address on a prank keeps its type: the address is real, the request is not.
- A hazard below 0.5 confidence ("maybe smoke?") is unsure: the whole hazards head is left unknown
  on that row rather than taught as present or absent.
- people_count null (absent or conflicting counts) is the "not stated" class.
- Rows that share a household (duplicate group) with a held-back test row are dropped at build
  time; a dev row sharing one with a train row is dropped here.
- Set B rows have labelled spans and train the token head. Sets C and D were not labelled, so
  their spans are unknown (null): the token head learns nothing from them, never "no spans".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rescue_tagger import labels as L
from rescue_tagger.records import read_jsonl

UNSURE = 0.5  # gold confidence below this marks a value the labeller was not sure of
SPAN_SETS = {"B"}  # sets whose rows have labelled spans

_ES = re.compile(r"[ñáéíóú¿¡]|\b(estamos|ayuda|necesit\w*|somos|mi|por favor|hay|casa|agua)\b",
                 re.I)
_VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)
_EN = re.compile(r"\b(we|for|need|my|the|and|is|ok)\b", re.I)


def detect_lang(text: str) -> str:
    vi, es = bool(_VI.search(text)), bool(_ES.search(text))
    if vi:
        return "mixed" if _EN.search(text) else "vi"
    if es:
        return "mixed" if _EN.search(text) else "es"
    return "en"


def to_record(row: dict[str, Any]) -> dict[str, Any]:
    g = row["gold_extracted"]
    labels: dict[str, Any] = {
        "is_stale_post": bool(g["time_ref"]["is_stale_post"]),
        "request_kind": g["request_kind"],
        "vulnerable": {k: any(v["kind"] == k for v in g["vulnerable"])
                       for k in L.VULNERABLE_KINDS},
        "needs": {k: any(n["kind"] == k for n in g["needs"]) for k in L.NEED_KINDS},
    }
    soft: dict[str, float] = {}
    if g["is_request"]["confidence"] < UNSURE:
        soft["is_request"] = 0.5
    else:
        labels["is_request"] = bool(g["is_request"]["value"])
    loc = g["location"]
    unusable = (loc["confidence"] < UNSURE
                and row["gold_decision"].get("abstain_reason") == "no_usable_location")
    labels["location_type"] = "none" if unusable else loc["type"]
    if all(h["confidence"] >= UNSURE for h in g["hazards"]):
        labels["hazards"] = {k: any(h["kind"] == k for h in g["hazards"])
                             for k in L.HAZARD_KINDS}
    rec: dict[str, Any] = {
        "id": f"product:{row['id']}",
        "text": row["raw_text"],
        "lang": detect_lang(row["raw_text"]),
        "source": "product",
        "split": row["split"],
        "labels": labels,
        "spans": None,
        "people_count": g["people_count"]["value"] or None,
        "need_hours": {n["kind"]: n.get("hours_without") for n in g["needs"]},
        "meta": {"set": row["set"], "product_id": row["id"],
                 "duplicate_group": row.get("gold_duplicate_group")},
    }
    if soft:
        rec["soft_labels"] = soft
    if row["set"] in SPAN_SETS:
        rec["spans"], rec["span_unknown"] = _labelled_spans(row["raw_text"], g["spans"],
                                                            labels["location_type"])
    return rec


# Set B labels LOCATION, COUNT and TIME, but not every count or time, and never NEED_DURATION.
# Unlabelled numbers and time phrases are left unknown to the token head (not taught as "O").
_UNLABELLED = re.compile(
    r"(?:\b(?:since|for|been out)\s+)?\b\d+\s*(?:am|pm|hours?|hrs?|days?|months?|years?(?: old)?"
    r"|percent|kids?|adults?|people|toddlers?|of us)?\b"
    r"|\b(?:this|last)\s+(?:morning|night|evening|afternoon)\b|\b(?:yesterday|tonight|today)\b", re.I)


def _labelled_spans(text: str, gold: list[dict[str, Any]], location_type: str
                    ) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Labelled spans to train on, plus character ranges the token head should not learn from."""
    spans = [{k: s[k] for k in ("start", "end", "label")} for s in gold]
    unknown: list[dict[str, int]] = []
    if location_type == "none":
        # The engine finds no usable location here, so its words are neither taught as a
        # location nor as "O".
        unknown += [{"start": s["start"], "end": s["end"]} for s in spans
                    if s["label"] == "LOCATION"]
        spans = [s for s in spans if s["label"] != "LOCATION"]
    for m in _UNLABELLED.finditer(text):
        if not any(s["start"] < m.end() and m.start() < s["end"] for s in spans):
            unknown.append({"start": m.start(), "end": m.end()})
    return spans, unknown


def load(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    bad = [r["id"] for r in rows if r.get("split") not in ("train", "dev")]
    if bad:
        raise ValueError(f"{path}: only train/dev product rows may be in the repo, got {bad}")
    # Rows sharing a duplicate group describe the same household. A dev row whose household is
    # also a train row would reward memorizing that train row at checkpoint time: leave it out.
    train_groups = {r["gold_duplicate_group"] for r in rows
                    if r["split"] == "train" and r.get("gold_duplicate_group")}
    return [to_record(r) for r in rows
            if not (r["split"] == "dev" and r.get("gold_duplicate_group") in train_groups)]


def held_back_groups(eval_rows: list[dict[str, Any]], ours: set[str]) -> set[str]:
    """Duplicate groups of the product's eval rows that are not ours (the held-back test rows).
    A row of ours in one of these groups is the same household as a test row, reworded."""
    return {r["gold_duplicate_group"] for r in eval_rows
            if r.get("gold_duplicate_group") and (r.get("raw_text") or r.get("text")) not in ours}
