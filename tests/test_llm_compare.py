"""The Opus baseline converter, the edge-case slice and the comparison scorer (no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rescue_tagger import labels as L
from rescue_tagger import llm_baseline as B
from rescue_tagger.compare import Stored, edge_checks, edge_report
from rescue_tagger.records import validate_extracted, validate_record

ROOT = Path(__file__).resolve().parent.parent
EDGE = [json.loads(ln) for ln in (ROOT / "evals/edge_cases.jsonl").read_text().splitlines()]


def test_naive_shapes_convert_to_valid_payloads():
    text = "3 of us on the roof at 1200 Main St since last night, no water for 2 days"
    ans = {"is_request": "true", "request_kind": "rescue", "hazards": ["on_roof", "lava"],
           "vulnerable": {"child": True, "elderly": False}, "needs": {"water": 48},
           "people_count": 3, "location_text": "1200 main st", "location_type": "address",
           "time_text": "since last night"}
    pay = B.to_payload(ans, text)
    validate_extracted(pay)
    assert pay["is_request"]["value"] is True
    assert [h["kind"] for h in pay["hazards"]] == ["on_roof"]
    assert [v["kind"] for v in pay["vulnerable"]] == ["child"]
    assert pay["needs"] == [{"kind": "water", "hours_without": 48.0}]
    assert {s["label"] for s in pay["spans"]} == {"LOCATION", "TIME"}
    loc = next(s for s in pay["spans"] if s["label"] == "LOCATION")
    assert text[loc["start"]:loc["end"]] == "1200 Main St"


def test_bad_values_fall_back_safely():
    pay = B.to_payload({"is_request": True, "request_kind": "boat", "people_count": -2,
                        "location_type": "planet", "location_text": None,
                        "is_request_confidence": 7}, "help")
    assert pay["request_kind"] == "other"
    assert pay["people_count"]["value"] is None
    assert pay["location"] == {"text": None, "type": "none", "confidence": 0.9}
    assert pay["is_request"]["confidence"] == 1.0
    validate_extracted(B.empty_payload())


def test_parse_json_takes_the_first_object():
    assert B.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert B.parse_json('Here you go: {"a": 1}\n{"b": 2} hope it helps') == {"a": 1}
    with pytest.raises(ValueError):
        B.parse_json("no json here")


def test_schemas_cover_every_key_once():
    assert sorted(B.ALL_KEYS) == sorted(B.PROPS)
    assert not set(B.TRIAGE_KEYS) & set(B.RISK_KEYS) | set(B.TRIAGE_KEYS) & set(B.LOC_KEYS)
    s = B.schema(B.ALL_KEYS)
    assert s["required"] == B.ALL_KEYS and s["additionalProperties"] is False


def test_sample_is_balanced_and_deterministic(tmp_path):
    rows = [{"id": f"x:{i}", "source": "munro" if i % 2 else "humaid",
             "labels": {"is_request": i % 3 == 0}} for i in range(200)]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    a, b = B.sample_ids(p, 10), B.sample_ids(p, 10)
    assert a == b and len(a) == 40 and len(set(a)) == 40
    assert B.sample_ids(p, 10, salt="other") != a


def test_edge_cases_are_valid_and_span_quotes_match():
    assert len(EDGE) >= 40
    assert len({r["id"] for r in EDGE}) == len(EDGE)
    for r in EDGE:
        validate_record({**r, "source": "synthetic"})
        assert r["split"] == "test" and r["source"] == "edge"
        for s in r["spans"]:
            assert r["text"][s["start"]:s["end"]].strip()


def test_edge_cases_are_protected_from_training():
    assert "evals/edge_cases.jsonl" in (ROOT / "configs/data.yaml").read_text()


def test_opus_prompts_share_no_phrases_with_edge_cases():
    """The prompt's illustrations must not hand Opus the edge-case wording."""
    prompts = B.NAIVE + B.ROLE + B.RUBRIC_TRIAGE + B.RUBRIC_RISK + B.RUBRIC_LOC

    def grams(t: str) -> set[str]:
        w = re.findall(r"\w+", t.lower())
        return {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}

    p = grams(prompts)
    leaks = {r["id"]: grams(r["text"]) & p for r in EDGE if grams(r["text"]) & p}
    assert not leaks


def gold_payload(rec: dict) -> dict:
    lab = rec["labels"]
    hours = rec.get("need_hours") or {}
    return {
        "location": {"text": None, "type": lab.get("location_type", "none"), "confidence": 0.9},
        "people_count": {"value": rec.get("people_count"), "confidence": 0.9},
        "vulnerable": [{"kind": k, "confidence": 0.9} for k, v in
                       lab.get("vulnerable", {}).items() if v],
        "hazards": [{"kind": k, "confidence": 0.9} for k, v in lab.get("hazards", {}).items() if v],
        "time_ref": {"text": None, "is_stale_post": lab["is_stale_post"], "confidence": 0.9},
        "is_request": {"value": lab.get("is_request", False),
                       "confidence": 0.5 if rec.get("ambiguous") else 0.95},
        "spans": [],
        "needs": [{"kind": k, "hours_without": hours.get(k)} for k, v in
                  lab.get("needs", {}).items() if v],
        "request_kind": lab.get("request_kind", "other"),
    }


def test_gold_answers_pass_every_edge_case():
    ex = Stored({r["text"]: gold_payload(r) for r in EDGE})
    rep = edge_report(ex, EDGE)
    assert rep["edge_pass_rate"] == 1.0 and rep["edge_flags_unsure"] == 1.0


def test_edge_check_catches_a_wrong_duration_and_a_missing_hazard():
    rec = next(r for r in EDGE if r.get("need_hours") and r["labels"].get("hazards")
               and any(r["labels"]["hazards"].values()))
    pay = gold_payload(rec)
    assert edge_checks(rec, pay)
    bad = json.loads(json.dumps(pay))
    bad["needs"][0]["hours_without"] = 1.0
    assert not edge_checks(rec, bad)
    bad = json.loads(json.dumps(pay))
    bad["hazards"] = []
    assert not edge_checks(rec, bad)


def test_edge_categories_cover_the_judge_challenge():
    cats = {r["category"] for r in EDGE}
    assert {"ambiguous", "not_request", "prompt_injection", "medical_device",
            "multi_need_duration", "stale"} <= cats
    assert all(k in L.NEED_KINDS for r in EDGE for k in (r.get("need_hours") or {}))


def test_a_missing_output_earns_no_credit():
    ex = Stored({}, EDGE)
    rep = edge_report(ex, EDGE)
    assert rep["edge_pass_rate"] == 0.0 and rep["edge_flags_unsure"] == 0.0


def test_odd_shapes_and_nan_do_not_crash_or_leak():
    pay = B.to_payload({"is_request": True, "hazards": True, "vulnerable": 5, "needs": "water",
                        "is_request_confidence": float("nan"),
                        "location_confidence": float("inf")}, "help")
    assert pay["hazards"] == [] and pay["vulnerable"] == [] and pay["needs"] == []
    assert pay["is_request"]["confidence"] == 0.9 and pay["location"]["confidence"] == 0.9
    pay = B.to_payload({"needs": [{"kind": "water", "hours_without": float("inf")}]}, "x")
    assert pay["needs"] == [{"kind": "water", "hours_without": None}]


def test_examples_carry_no_confidences():
    shots = [{"gold_extracted": {
        "is_request": {"value": True, "confidence": 1.0}, "request_kind": "rescue",
        "time_ref": {"text": None, "is_stale_post": False, "confidence": 1.0},
        "hazards": [{"kind": "fire", "confidence": 1.0}], "vulnerable": [], "needs": [],
        "people_count": {"value": 2, "confidence": 1.0},
        "location": {"text": None, "type": "none", "confidence": 1.0}}, "text": "fire, 2 of us"}]
    block = B.examples_block(shots, B.ALL_KEYS)
    assert "confidence\": " not in block and '"hazards": ["fire"]' in block
