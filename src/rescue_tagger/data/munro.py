"""Munro disaster response messages (CC BY 4.0) -> unified records.

~24k messages from Haiti 2010, Pakistan 2010, Sandy 2012 and news, each labeled by people with 36
binary categories. Source: https://github.com/rmunro/disaster_response_messages (pinned below).

What the labels can vouch for (everything else is left unknown and masked):
  is_request    <- `request` (a request for help/aid). Offers and news are not requests.
  needs         <- water / food / medical_products, but only on requests: an offer of water or a
                   news story about food shortages is not a need of the sender.
  request_kind  <- "other" for non-requests (matches the contract fixtures); for requests,
                   search_and_rescue -> rescue, missing_people -> welfare_check, supplies -> supply;
                   a request with none of those is left unknown.
Rows with related == 2 (annotators could not tell) are dropped. The upstream train / validation /
test split is kept as train / dev / test, and the test split is only ever used for evaluation.
"""

from __future__ import annotations

import csv
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rescue_tagger import labels as L

COMMIT = "1657e1c4e11acbfa56241d2b5cd627adbd989ca2"
BASE_URL = f"https://raw.githubusercontent.com/rmunro/disaster_response_messages/{COMMIT}"
FILES = {
    "train": "disaster_response_training.csv",
    "dev": "disaster_response_validation.csv",
    "test": "disaster_response_test.csv",
}
SUPPLY_COLUMNS = ("water", "food", "medical_products", "shelter", "clothing")


def download(raw_dir: Path) -> Path:
    out = raw_dir / "munro"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = out / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)  # noqa: S310 (pinned https)
    return out


def _flag(row: dict[str, str], col: str) -> bool:
    return row.get(col, "0").strip() == "1"


def map_row(row: dict[str, str], split: str) -> dict[str, Any] | None:
    text = (row.get("message") or "").strip()
    if not text or row.get("related", "").strip() == "2":
        return None
    is_request = _flag(row, "request")
    labels: dict[str, Any] = {
        "is_request": is_request,
        "needs": {
            "water": is_request and _flag(row, "water"),
            "food": is_request and _flag(row, "food"),
            "meds": is_request and _flag(row, "medical_products"),
        },
    }
    if not is_request:
        labels["request_kind"] = "other"
    elif _flag(row, "search_and_rescue"):
        labels["request_kind"] = "rescue"
    elif _flag(row, "missing_people"):
        labels["request_kind"] = "welfare_check"
    elif any(_flag(row, c) for c in SUPPLY_COLUMNS):
        labels["request_kind"] = "supply"
    assert set(labels["needs"]) == set(L.NEED_KINDS)
    return {
        "id": f"munro:{split}:{row['id']}",
        "source": "munro",
        "lang": "en",
        "text": text,
        "split": split,
        "labels": labels,
        "spans": None,
        "meta": {"genre": row.get("genre"), "original": (row.get("original") or "")[:500]},
    }


def load(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Upstream repeats ~50 rows verbatim (same id); the first copy wins."""
    src = raw_dir / "munro"
    seen: set[str] = set()
    for split, name in FILES.items():
        with (src / name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rec = map_row(row, split)
                if rec is not None and rec["id"] not in seen:
                    seen.add(rec["id"])
                    yield rec
