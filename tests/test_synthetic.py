"""Synthetic generator logic, without calling any API."""

import json
from pathlib import Path

import pytest

from rescue_tagger import labels as L
from rescue_tagger.data import synthetic as S
from rescue_tagger.records import validate_record

FIXTURES = Path(__file__).parent / "fixtures"


def test_spec_is_deterministic():
    assert S.sample_spec(42) == S.sample_spec(42)
    assert S.sample_spec(42) != S.sample_spec(43)


def test_specs_cover_every_label():
    specs = [S.sample_spec(s) for s in range(3000)]
    assert {s.lang for s in specs} == set(L.LANGS)
    assert {s.request_kind for s in specs} == set(L.REQUEST_KINDS)
    assert {s.location_type for s in specs} == set(L.LOCATION_TYPES)
    assert {h for s in specs for h in s.hazards} == set(L.HAZARD_KINDS)
    assert {v for s in specs for v in s.vulnerable} == set(L.VULNERABLE_KINDS)
    assert {n for s in specs for n in s.needs} == set(L.NEED_KINDS)
    assert any(not s.is_request for s in specs) and any(s.is_stale_post for s in specs)
    for s in specs:
        if not s.is_request:
            assert s.request_kind == "other" and not s.hazards and not s.needs


def _gen(text, **quotes):
    out = {"text": text, "LOCATION": None, "COUNT": None, "TIME": None, "CONTACT": None,
           "NEED_DURATION": dict.fromkeys(L.NEED_KINDS)}
    out.update(quotes)
    return out


def test_build_record_aligns_spans_and_builds_gold():
    spec = S.Spec(seed=1, lang="en", style="x", is_request=True, non_request_type=None,
                  is_stale_post=False, request_kind="supply", needs={"water": 30},
                  location_type="address", location_hint="Telephone Road", people_count=3)
    text = "No water for 30 hours at 620 Telephone Rd, 3 of us"
    gen = _gen(text, LOCATION="620 Telephone Rd", COUNT="3 of us",
               NEED_DURATION={"food": None, "water": "for 30 hours", "meds": None})
    rec = S.build_record(spec, gen, "gpt-test")
    validate_record(rec)
    got = {s["label"]: text[s["start"]:s["end"]] for s in rec["spans"]}
    assert got == {"LOCATION": "620 Telephone Rd", "COUNT": "3 of us",
                   "NEED_DURATION": "for 30 hours"}
    gold = rec["gold_extracted"]
    assert gold["location"]["text"] == "620 Telephone Rd"
    assert gold["needs"] == [{"kind": "water", "hours_without": 30}]
    assert rec["labels"]["needs"] == {"food": False, "water": True, "meds": False}


def test_build_record_rejects_inconsistent_output():
    spec = S.Spec(seed=2, lang="en", style="x", is_request=True, non_request_type=None,
                  is_stale_post=False, request_kind="rescue", location_type="none",
                  people_count=None)
    with pytest.raises(ValueError, match="not found"):
        S.build_record(spec, _gen("help us", COUNT="two of us"), "m")
    with pytest.raises(ValueError, match="location presence"):
        S.build_record(spec, _gen("help at Kirby", LOCATION="Kirby"), "m")


def test_disagreements():
    rec = {"labels": {"is_request": True, "is_stale_post": False, "request_kind": "rescue",
                      "location_type": "address",
                      "hazards": {k: k == "fire" for k in L.HAZARD_KINDS},
                      "vulnerable": dict.fromkeys(L.VULNERABLE_KINDS, False),
                      "needs": dict.fromkeys(L.NEED_KINDS, False)},
           "people_count": 2}
    agree = {"is_request": True, "is_stale_post": False, "request_kind": "rescue",
             "location_type": "address", "hazards": ["fire"], "vulnerable": [], "needs": [],
             "people_count": 2}
    assert S.disagreements(rec, agree) == []
    assert S.disagreements(rec, agree | {"hazards": []}) == ["hazards: spec=['fire'] verify=[]"]


def test_committed_sample_rows_are_valid():
    rows = [json.loads(x) for x in (FIXTURES / "synthetic_sample.jsonl").read_text().splitlines()]
    assert rows
    for rec in rows:
        validate_record(rec)
        assert rec["source"] == "synthetic" and rec["meta"]["generator"].startswith("gpt")
