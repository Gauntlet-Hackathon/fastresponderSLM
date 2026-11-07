import json
from pathlib import Path

import pytest

from rescue_tagger.data.product import load, to_record
from rescue_tagger.product_eval import score
from rescue_tagger.records import validate_record

DATA = Path(__file__).resolve().parent.parent / "data/product/rescue_eval_train_dev.jsonl"
ROWS = {r["id"]: r for r in map(json.loads, DATA.read_text().splitlines())}


def test_every_product_row_maps_to_a_valid_record_and_no_test_rows_are_in_the_repo():
    recs = load(DATA)
    for r in recs:
        validate_record(r)
        if r["meta"]["set"] == "B":
            assert isinstance(r["spans"], list)  # labelled spans train the token head
        else:
            assert r["spans"] is None  # not labelled: unknown, never "no spans"
    assert {r["split"] for r in recs} == {"train", "dev"}


def test_unsure_and_unusable_values_are_not_taught_as_confident():
    prank = to_record(ROWS["D-043"])
    assert "is_request" not in prank["labels"] and prank["soft_labels"] == {"is_request": 0.5}
    assert prank["labels"]["location_type"] == "address"  # a real address on a prank stays
    assert to_record(ROWS["D-028"])["labels"]["location_type"] == "none"  # no usable landmark
    assert "hazards" not in to_record(ROWS["D-064"])["labels"]  # "might be rising??"
    assert to_record(ROWS["D-033"])["people_count"] is None  # conflicting counts
    assert to_record(ROWS["C-028"])["lang"] == "vi"
    oxygen = to_record(ROWS["C-004"])["labels"]
    assert oxygen["hazards"]["medical_device_no_power"] and oxygen["vulnerable"]["elderly"]


def _payload(row, **over):
    g = json.loads(json.dumps(row["gold_extracted"]))
    g.update(over)
    return g


def test_score_rewards_unsure_on_pranks_and_counts_dropped_p1_hazards():
    rows = [ROWS["D-043"], ROWS["C-049"]]
    perfect = [_payload(ROWS["D-043"], is_request={"value": True, "confidence": 0.5}),
               _payload(ROWS["C-049"])]
    s = score(rows, perfect)
    assert s["unsure_flagged"] == 1.0 and s["p1_critical_kept"] == 1.0
    assert s["joke_taken_as_confident_request"] == 0
    bad = [_payload(ROWS["D-043"], is_request={"value": True, "confidence": 0.95}),
           _payload(ROWS["C-049"], hazards=[])]
    s = score(rows, bad)
    assert s["unsure_flagged"] == 0.0 and s["joke_taken_as_confident_request"] == 1
    assert s["p1_critical_kept"] == 0.0 and s["hazards_f1"] == pytest.approx(0.5)


def test_rows_sharing_a_household_with_train_or_held_back_rows_are_left_out():
    from rescue_tagger.data.product import held_back_groups

    ids = {r["meta"]["product_id"] for r in load(DATA)}
    assert "C-053" not in ids and "C-054" in ids  # dev row of a train row's household
    ours = {r["raw_text"] for r in ROWS.values()}
    held = [{"raw_text": "a reworded test message", "gold_duplicate_group": "C-dup-6"},
            {"raw_text": ROWS["C-054"]["raw_text"], "gold_duplicate_group": "C-dup-3"}]
    assert held_back_groups(held, ours) == {"C-dup-6"}


def test_set_b_unlabelled_counts_and_times_carry_no_token_loss():
    from rescue_tagger.features import IGNORE
    from rescue_tagger.labels import TAG2ID

    rec = to_record(ROWS["B-014"])  # "been out 10 hours" is a duration nobody labelled
    assert [rec["text"][u["start"]:u["end"]] for u in rec["span_unknown"]] == ["been out 10 hours"]

    class Tok:  # whitespace tokenizer with offsets, enough for encode()
        def __call__(self, text, **_):
            import re
            offs = [(0, 0)] + [m.span() for m in re.finditer(r"\S+", text)] + [(0, 0)]
            return {"input_ids": list(range(len(offs))), "attention_mask": [1] * len(offs),
                    "offset_mapping": offs}

    from rescue_tagger.features import encode
    item = encode(rec, Tok(), 256)
    words = [None] + rec["text"].split() + [None]
    tags = dict(zip(words, item["token_labels"], strict=True))
    assert tags["hours."] == IGNORE and tags["10"] == IGNORE
    assert tags["Dialysis"] == TAG2ID["O"] and tags["Wycliffe"] == TAG2ID["I-LOCATION"]
    none_loc = to_record(ROWS["B-059"])  # no usable location: its words are not taught
    assert all(s["label"] != "LOCATION" for s in none_loc["spans"])
