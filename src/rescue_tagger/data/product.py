"""Product team's hand-reviewed eval rows (sets C and D, train and dev splits only).

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
- Spans were not labelled, so spans are unknown (null): the token head learns nothing from these
  rows, never "no spans".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rescue_tagger import labels as L
from rescue_tagger.records import read_jsonl

UNSURE = 0.5  # gold confidence below this marks a value the labeller was not sure of

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
        "meta": {"set": row["set"], "product_id": row["id"]},
    }
    if soft:
        rec["soft_labels"] = soft
    return rec


def load(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    bad = [r["id"] for r in rows if r.get("split") not in ("train", "dev")]
    if bad:
        raise ValueError(f"{path}: only train/dev product rows may be in the repo, got {bad}")
    return [to_record(r) for r in rows]
