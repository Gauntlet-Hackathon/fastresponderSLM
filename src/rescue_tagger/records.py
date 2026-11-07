"""The unified training record and the `extracted` schema validator.

Every source (public datasets, synthetic generation) is converted to this one JSONL shape so the
trainer never needs to know where a row came from. A row only carries the labels its source can
vouch for; anything absent is *unknown* and is masked out of the loss (never treated as negative).

Record keys:
  id              unique string, prefixed by source (e.g. "munro:train:2")
  source          one of labels.SOURCES
  lang            one of labels.LANGS
  text            the raw message, exactly as the model will see it
  split           train | dev | test (fixed once published; see data/splits)
  labels          partial dict, per head:
                    binary heads       -> bool
                    multi-label heads  -> {kind: bool} covering EVERY kind of that head
                    single-label heads -> one allowed string
  spans           list of {start, end, label} (code-point offsets, label in SPAN_LABELS),
                  or null when the source has no token annotation (tagger loss masked)
  people_count    int | null (null = no count stated); absent = unknown
  need_hours      {need_kind: hours | null}; absent = unknown
  gold_extracted  optional full `extracted` payload (synthetic rows), schema-valid
  weight          optional float loss weight (weak sources < 1.0)
  meta            free-form provenance (original labels, generator spec, ...)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from rescue_tagger import labels as L

SCHEMA_DIR = Path(__file__).parent / "schema"
REQUIRED_KEYS = ("id", "source", "lang", "text", "split", "labels", "spans")


@cache
def extracted_validator() -> Draft202012Validator:
    common = json.loads((SCHEMA_DIR / "common.schema.json").read_text(encoding="utf-8"))
    extracted = json.loads(
        (SCHEMA_DIR / "payloads" / "extracted.schema.json").read_text(encoding="utf-8")
    )
    registry = Registry().with_resources(
        [
            (common["$id"], Resource.from_contents(common)),
            (extracted["$id"], Resource.from_contents(extracted)),
        ]
    )
    return Draft202012Validator(extracted, registry=registry)


def validate_extracted(payload: dict[str, Any]) -> None:
    errors = sorted(extracted_validator().iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        raise ValueError(f"extracted payload invalid at {list(e.path)}: {e.message}")


def validate_record(rec: dict[str, Any]) -> None:
    """Raise ValueError with a precise message if `rec` is not a valid training record."""
    rid = rec.get("id", "<no id>")
    for k in REQUIRED_KEYS:
        if k not in rec:
            raise ValueError(f"{rid}: missing key {k!r}")
    if rec["source"] not in L.SOURCES:
        raise ValueError(f"{rid}: unknown source {rec['source']!r}")
    if rec["lang"] not in L.LANGS:
        raise ValueError(f"{rid}: unknown lang {rec['lang']!r}")
    if rec["split"] not in L.SPLITS:
        raise ValueError(f"{rid}: unknown split {rec['split']!r}")
    text = rec["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{rid}: empty text")

    labels = rec["labels"]
    if not isinstance(labels, dict):
        raise ValueError(f"{rid}: labels must be a dict")
    for head, value in labels.items():
        if head in L.BINARY_HEADS:
            if not isinstance(value, bool):
                raise ValueError(f"{rid}: {head} must be bool")
        elif head in L.MULTI_LABEL_HEADS:
            kinds = L.MULTI_LABEL_HEADS[head]
            if not isinstance(value, dict) or set(value) != set(kinds):
                raise ValueError(f"{rid}: {head} must map every kind of {kinds} to a bool")
            if not all(isinstance(v, bool) for v in value.values()):
                raise ValueError(f"{rid}: {head} values must be bool")
        elif head in L.SINGLE_LABEL_HEADS:
            if value not in L.SINGLE_LABEL_HEADS[head]:
                raise ValueError(f"{rid}: {head}={value!r} not in {L.SINGLE_LABEL_HEADS[head]}")
        else:
            raise ValueError(f"{rid}: unknown head {head!r}")

    spans = rec["spans"]
    if spans is not None:
        if not isinstance(spans, list):
            raise ValueError(f"{rid}: spans must be a list or null")
        last_end = -1
        for s in sorted(spans, key=lambda s: s["start"]):
            if s["label"] not in L.SPAN_LABELS:
                raise ValueError(f"{rid}: unknown span label {s['label']!r}")
            if not (0 <= s["start"] < s["end"] <= len(text)):
                raise ValueError(f"{rid}: span {s} out of bounds for text of length {len(text)}")
            if s["start"] < last_end:
                raise ValueError(f"{rid}: overlapping spans")
            if text[s["start"] : s["end"]] != text[s["start"] : s["end"]].strip():
                raise ValueError(f"{rid}: span {s} has leading/trailing whitespace")
            last_end = s["end"]

    if "people_count" in rec:
        pc = rec["people_count"]
        if pc is not None and (not isinstance(pc, int) or isinstance(pc, bool) or pc < 0):
            raise ValueError(f"{rid}: people_count must be a non-negative int or null")
    if "need_hours" in rec:
        for kind, hours in rec["need_hours"].items():
            if kind not in L.NEED_KINDS:
                raise ValueError(f"{rid}: unknown need kind {kind!r}")
            if hours is not None and (not isinstance(hours, int | float) or hours < 0):
                raise ValueError(f"{rid}: need_hours[{kind}] must be >= 0 or null")
    if "weight" in rec and not (0 < rec["weight"] <= 1):
        raise ValueError(f"{rid}: weight must be in (0, 1]")
    if rec.get("gold_extracted") is not None:
        validate_extracted(rec["gold_extracted"])


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
