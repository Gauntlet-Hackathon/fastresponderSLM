"""Single source of truth for every label the model predicts.

The enums mirror `schema/payloads/extracted.schema.json`;
`tests/test_labels.py` fails if they drift.
Training data, the model heads, inference and the eval code all import from here.
"""

from __future__ import annotations

# --- Classification heads -------------------------------------------------------------------
# Multi-label heads: one sigmoid per kind; the payload lists every kind above a reporting floor.
VULNERABLE_KINDS: tuple[str, ...] = ("child", "elderly", "disabled", "medical", "pregnant")
HAZARD_KINDS: tuple[str, ...] = (
    "rising_water",
    "on_roof",
    "injury",
    "medical_device_no_power",
    "fire",
    "trapped",
)
NEED_KINDS: tuple[str, ...] = ("food", "water", "meds")

# Single-label heads: softmax.
REQUEST_KINDS: tuple[str, ...] = ("rescue", "evacuation", "welfare_check", "supply", "other")
LOCATION_TYPES: tuple[str, ...] = ("landmark", "address", "coords", "none")

# Binary heads: one sigmoid.
BINARY_HEADS: tuple[str, ...] = ("is_request", "is_stale_post")

MULTI_LABEL_HEADS: dict[str, tuple[str, ...]] = {
    "vulnerable": VULNERABLE_KINDS,
    "hazards": HAZARD_KINDS,
    "needs": NEED_KINDS,
}
SINGLE_LABEL_HEADS: dict[str, tuple[str, ...]] = {
    "request_kind": REQUEST_KINDS,
    "location_type": LOCATION_TYPES,
}
ALL_HEADS: tuple[str, ...] = (
    *BINARY_HEADS,
    *MULTI_LABEL_HEADS,
    *SINGLE_LABEL_HEADS,
)

# --- Token tagger (BIO) ---------------------------------------------------------------------
# LOCATION fills location.text and spans[]; COUNT feeds people_count; TIME fills time_ref.text;
# NEED_DURATION ("for 30 hours", "desde ayer") is parsed into needs[].hours_without by code;
# CONTACT is tagged so the UI can highlight it (it never enters the extracted payload).
SPAN_LABELS: tuple[str, ...] = ("LOCATION", "COUNT", "TIME", "NEED_DURATION", "CONTACT")
BIO_TAGS: tuple[str, ...] = ("O", *(f"{p}-{lab}" for lab in SPAN_LABELS for p in ("B", "I")))
TAG2ID: dict[str, int] = {t: i for i, t in enumerate(BIO_TAGS)}
ID2TAG: dict[int, str] = dict(enumerate(BIO_TAGS))

# --- Data provenance ------------------------------------------------------------------------
SOURCES: tuple[str, ...] = ("munro", "humaid", "synthetic", "product")
# "mixed" is English mixed with Spanish or Vietnamese; "vi" comes from the product rows only.
LANGS: tuple[str, ...] = ("en", "es", "mixed", "vi")
SYNTH_LANGS: tuple[str, ...] = ("en", "es", "mixed", "vi")  # what the synthetic generator writes
SPLITS: tuple[str, ...] = ("train", "dev", "test")
