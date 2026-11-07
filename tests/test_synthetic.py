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
    assert {s.lang for s in specs} == set(L.SYNTH_LANGS)
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


def test_focus_specs_follow_the_rubric():
    from rescue_tagger.data.synthetic import build_prompt, sample_focus_spec

    specs = [sample_focus_spec(s) for s in range(300000, 300400)]
    assert [sample_focus_spec(300007)] == [specs[7]]
    dev = [s for s in specs if s.device]
    assert dev and all("medical_device_no_power" in s.hazards and "medical" in s.vulnerable
                       for s in dev)
    shared = [s for s in specs if s.shared_duration]
    assert shared and all(len(s.needs) >= 2 and len(set(s.needs.values())) == 1 for s in shared)
    assert "State the duration ONCE" in build_prompt(shared[0])
    assert all(s.is_request for s in specs)
    assert all(s.needs for s in specs if s.request_kind == "supply")
    assert any(s.is_stale_post for s in specs)


def test_verify_only_mode_keeps_agreeing_rows_and_rejects_others(monkeypatch):
    from rescue_tagger.data import synthetic as S

    spec = S.sample_focus_spec(300001)
    gen = {"text": "x", "LOCATION": None, "COUNT": None, "TIME": None, "CONTACT": None,
           "NEED_DURATION": dict.fromkeys(S.L.NEED_KINDS)}
    spec.location_type, spec.people_count = "none", None
    rec = S.build_record(spec, gen, "Qwen/Qwen2.5-32B-Instruct")
    assert rec["id"].startswith("synthetic:Qwen2.5-32B-Instruct:")
    lab = rec["labels"]
    agree = {"is_request": lab["is_request"], "is_stale_post": lab["is_stale_post"],
             "request_kind": lab["request_kind"], "location_type": lab["location_type"],
             "hazards": [k for k, v in lab["hazards"].items() if v],
             "vulnerable": [k for k, v in lab["vulnerable"].items() if v],
             "needs": [k for k, v in lab["needs"].items() if v], "people_count": None}
    monkeypatch.setattr(S, "_call_json", lambda *a, **k: agree)
    assert S.verify_record(None, rec, "microsoft/phi-4")["record"]["meta"]["verifier"]
    monkeypatch.setattr(S, "_call_json", lambda *a, **k: {**agree, "is_request": False})
    assert "verify:" in S.verify_record(None, rec, "microsoft/phi-4")["error"]


def test_vague_specs_become_soft_labeled_records():
    from rescue_tagger.data import synthetic as S

    spec = S.sample_vague_spec(320001)
    spec.location_type, spec.location_hint = "none", None
    assert spec.vague and not (spec.hazards or spec.vulnerable or spec.needs or spec.people_count)
    assert "TOO VAGUE" in S.build_prompt(spec)
    spec.location_type = "none"
    gen = {"text": "idk whats going on out here", "LOCATION": None, "COUNT": None, "TIME": None,
           "CONTACT": None, "NEED_DURATION": dict.fromkeys(S.L.NEED_KINDS)}
    rec = S.build_record(spec, gen, "Qwen/Qwen2.5-32B-Instruct")
    assert "is_request" not in rec["labels"] and "request_kind" not in rec["labels"]
    assert rec["soft_labels"] == {"is_request": 0.5}
    v = {"is_request": True, "is_stale_post": False, "request_kind": "rescue",
         "location_type": "none", "hazards": [], "vulnerable": [], "needs": [],
         "people_count": None}
    assert S.disagreements(rec, v) == []  # the verifier's call on is_request is not checked
    assert S.disagreements(rec, {**v, "hazards": ["fire"]})


def test_a_few_disagreeing_fields_are_masked_not_rejected(monkeypatch):
    from rescue_tagger.data import synthetic as S
    from rescue_tagger.records import validate_record

    spec = S.sample_focus_spec(300001)
    gen = {"text": "x", "LOCATION": None, "COUNT": None, "TIME": None, "CONTACT": None,
           "NEED_DURATION": dict.fromkeys(S.L.NEED_KINDS)}
    spec.location_type, spec.people_count = "none", None

    def fresh():
        return S.build_record(spec, gen, "google/gemma-4-31B-it")

    lab = fresh()["labels"]
    agree = {"is_request": lab["is_request"], "is_stale_post": lab["is_stale_post"],
             "request_kind": lab["request_kind"], "location_type": lab["location_type"],
             "hazards": [k for k, v in lab["hazards"].items() if v],
             "vulnerable": [k for k, v in lab["vulnerable"].items() if v],
             "needs": [k for k, v in lab["needs"].items() if v], "people_count": None}
    wrong_needs = [k for k in S.L.NEED_KINDS if k not in agree["needs"]][:1]
    two_off = {**agree, "location_type": "landmark", "needs": wrong_needs, "people_count": 4}

    out = S.apply_verdict(fresh(), {**agree, "location_type": "landmark",
                                    "needs": wrong_needs}, "microsoft/phi-4")
    rec = out["record"]
    assert "location_type" not in rec["labels"] and "needs" not in rec["labels"]
    assert "need_hours" not in rec and "gold_extracted" not in rec
    assert set(rec["meta"]["masked"]) == {"location_type", "needs"}
    assert rec["labels"]["hazards"] == lab["hazards"]  # agreed heads keep their labels
    validate_record(rec)

    assert "error" in S.apply_verdict(fresh(), two_off, "microsoft/phi-4")  # three heads off
    assert "error" in S.apply_verdict(fresh(), {**agree, "is_request": not agree["is_request"]},
                                      "microsoft/phi-4")
    rec = S.apply_verdict(fresh(), {**agree, "people_count": 4}, "microsoft/phi-4")["record"]
    assert "people_count" not in rec and "gold_extracted" not in rec
    clean = S.apply_verdict(fresh(), agree, "microsoft/phi-4")["record"]
    assert "masked" not in clean["meta"] and "gold_extracted" in clean


def _agreeing_verdict(rec):
    lab = rec["labels"]
    return {"is_request": lab.get("is_request", True), "is_stale_post": lab["is_stale_post"],
            "request_kind": lab.get("request_kind", "other"),
            "location_type": lab["location_type"],
            "hazards": [k for k, v in lab["hazards"].items() if v],
            "vulnerable": [k for k, v in lab["vulnerable"].items() if v],
            "needs": [k for k, v in lab["needs"].items() if v],
            "people_count": rec.get("people_count")}


def test_masked_count_drops_spans_so_the_token_head_learns_nothing_from_it():
    from rescue_tagger.data import synthetic as S

    spec = S.sample_focus_spec(300001)
    spec.location_type, spec.people_count, spec.location_hint = "none", 2, None
    text = "There are 2 of us here and we need help"
    gen = {"text": text, "LOCATION": None, "COUNT": "2 of us", "TIME": None, "CONTACT": None,
           "NEED_DURATION": dict.fromkeys(S.L.NEED_KINDS)}
    rec = S.build_record(spec, gen, "google/gemma-4-31B-it")
    assert any(sp["label"] == "COUNT" for sp in rec["spans"])
    v = {**_agreeing_verdict(rec), "people_count": 3}
    out = S.apply_verdict(rec, v, "microsoft/phi-4")["record"]
    assert out["spans"] is None and "people_count" not in out

    rec = S.build_record(spec, gen, "google/gemma-4-31B-it")
    wrong = [k for k in S.L.LOCATION_TYPES if k != "none"][0]
    out = S.apply_verdict(rec, {**_agreeing_verdict(rec), "location_type": wrong},
                          "microsoft/phi-4")["record"]
    assert out["spans"]  # a head without spans keeps them


def test_vague_rows_are_kept_only_when_the_verifier_cannot_tell():
    from rescue_tagger.data import synthetic as S

    spec = S.sample_vague_spec(320001)
    spec.location_type, spec.location_hint = "none", None
    gen = {"text": "idk whats going on by the bayou", "LOCATION": None, "COUNT": None,
           "TIME": None, "CONTACT": None, "NEED_DURATION": dict.fromkeys(S.L.NEED_KINDS)}
    for answer, kept in (("cannot_tell", True), ("needs_help", False), ("no_help", False)):
        rec = S.build_record(spec, gen, "google/gemma-4-31B-it")
        v = {**_agreeing_verdict(rec), "help": answer}
        assert ("record" in S.apply_verdict(rec, v, "microsoft/phi-4")) is kept
    rec = S.build_record(spec, gen, "google/gemma-4-31B-it")
    assert "error" in S.apply_verdict(rec, _agreeing_verdict(rec), "microsoft/phi-4")


def test_hoax_spec_is_a_request_with_soft_label():
    from rescue_tagger.data.synthetic import build_prompt, sample_hoax_spec

    sp = sample_hoax_spec(400000)
    assert sp.hoax and sp.is_request and not sp.vague and "hoax sign" in sp.style
    assert "prank" in build_prompt(sp)


def test_mix_spec_covers_code_switch_stale_and_third_party():
    from rescue_tagger.data.synthetic import build_prompt, sample_mix_spec

    specs = [sample_mix_spec(350000 + i) for i in range(300)]
    assert all(sp.is_request and not sp.prompt_injection for sp in specs)
    assert sum(sp.lang in ("mixed", "vi") and "mid-sentence" in sp.style for sp in specs) > 100
    stale = [sp for sp in specs if sp.is_stale_post]
    assert stale and all("OLD post" in build_prompt(sp) for sp in stale[:5])
    third = [sp for sp in specs if sp.third_party]
    assert third and all("NOT with the writer" in sp.style for sp in third)
    assert sample_mix_spec(350001) == sample_mix_spec(350001)


def test_implied_spec_names_p1_hazards_only_through_the_situation():
    from rescue_tagger.data.synthetic import build_prompt, sample_implied_spec

    specs = [sample_implied_spec(360000 + i) for i in range(200)]
    assert all(sp.is_request and sp.hazards and sp.hazard_implied for sp in specs)
    assert sum(sp.vulnerable_indirect for sp in specs) > 80
    assert sum("typed fast" in sp.style or "terse" in sp.style or "hedged" in sp.style
               for sp in specs) > 120
    prompt = build_prompt(specs[0])
    assert "ONLY through the situation" in prompt
    assert sample_implied_spec(360001) == sample_implied_spec(360001)
