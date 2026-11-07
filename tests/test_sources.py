"""Label mapping for the public sources, on small committed samples (no network)."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from rescue_tagger.data import humaid, munro
from rescue_tagger.records import validate_record

FIXTURES = Path(__file__).parent / "fixtures"


def _munro_rows():
    with (FIXTURES / "munro_sample.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_munro_mapping():
    mapped = [munro.map_row(r, "train") for r in _munro_rows()]
    kept = [m for m in mapped if m is not None]
    assert len(kept) == len(mapped) - 1, "the related==2 row is dropped"
    for rec in kept:
        validate_record(rec)
    rescue = [r for r in kept if r["labels"].get("request_kind") == "rescue"]
    assert len(rescue) == 2 and all(r["labels"]["is_request"] for r in rescue)
    supply = [r for r in kept if r["labels"].get("request_kind") == "supply"]
    assert supply and all(r["labels"]["needs"]["water"] for r in supply)
    for rec in kept:
        if not rec["labels"]["is_request"]:
            assert rec["labels"]["request_kind"] == "other"
            assert not any(rec["labels"]["needs"].values()), "needs only count on requests"
        assert rec["spans"] is None, "Munro has no token labels"


def _tweet_id_at(ts: datetime) -> int:
    ms = int(ts.timestamp() * 1000) - humaid.TWITTER_EPOCH_MS
    return ms << 22


def test_tweet_time_roundtrip():
    ts = datetime(2017, 8, 27, 12, 0, tzinfo=UTC)
    assert abs((humaid.tweet_time(_tweet_id_at(ts)) - ts).total_seconds()) < 1


def test_harvey_holdout():
    in_window = _tweet_id_at(datetime(2017, 8, 28, tzinfo=UTC))
    irma = _tweet_id_at(datetime(2017, 9, 10, tzinfo=UTC))
    assert humaid.is_harvey("hurricane", in_window, "water everywhere")
    assert not humaid.is_harvey("hurricane", irma, "water everywhere in Florida")
    assert humaid.is_harvey("flood", irma, "Still flooded in Houston")
    assert not humaid.is_harvey("flood", in_window, "monsoon in Kerala"), "only hurricanes by date"

    row = {"tweet_id": in_window, "tweet_text": "We need water at the shelter",
           "class_label": "requests_or_urgent_needs"}
    rec = humaid.map_row(row, "hurricane", "train")
    validate_record(rec)
    assert rec["split"] == "test" and rec["meta"]["harvey_holdout"]


def test_humaid_mapping():
    base = {"tweet_id": _tweet_id_at(datetime(2019, 3, 20, tzinfo=UTC))}
    req = humaid.map_row(base | {"tweet_text": "Need food", "class_label":
                                 "requests_or_urgent_needs"}, "flood", "train")
    assert req["labels"] == {"is_request": True} and req["split"] == "train"
    other = humaid.map_row(base | {"tweet_text": "Praying", "class_label":
                                   "sympathy_and_support"}, "flood", "dev")
    assert other["labels"] == {"is_request": False, "request_kind": "other"}
    assert humaid.map_row(base | {"tweet_text": "??", "class_label": "dont_know_cant_judge"},
                          "flood", "train") is None
