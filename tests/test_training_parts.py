"""Training/inference building blocks that need no model download."""

import pytest
import torch

from rescue_tagger import labels as L
from rescue_tagger.durations import nearest_need, parse_hours
from rescue_tagger.features import IGNORE, char_spans_to_bio, count_class
from rescue_tagger.infer import decode_spans
from rescue_tagger.train import compute_loss

T = L.TAG2ID


def test_bio_from_char_spans_with_sentencepiece_offsets():
    text = "en 8120 Fondren Rd ya"
    # mmBERT-style offsets: leading-space tokens, and a lone "▁" before digits
    offsets = [(0, 0), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 12), (12, 15),
               (15, 18), (18, 21), (0, 0)]
    spans = [{"start": 3, "end": 18, "label": "LOCATION"}]
    tags = char_spans_to_bio(text, offsets, spans)
    assert tags[0] == IGNORE and tags[-1] == IGNORE
    assert tags[1] == T["O"]
    assert tags[2] == IGNORE, "whitespace-only token carries no label"
    assert tags[3] == T["B-LOCATION"]
    assert tags[4:10] == [T["I-LOCATION"]] * 6
    assert tags[10] == T["O"]


def test_decode_spans_roundtrip_skips_whitespace_tokens():
    text = "en 8120 Fondren Rd ya"
    offsets = [(0, 0), (0, 2), (2, 3), (3, 4), (4, 7), (7, 15), (15, 18), (18, 21), (0, 0)]
    tags = [0, T["O"], T["O"], T["B-LOCATION"], T["I-LOCATION"], T["I-LOCATION"],
            T["I-LOCATION"], T["O"], 0]
    probs = [1.0] * len(tags)
    spans = decode_spans(text, offsets, tags, probs)
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [
        ("8120 Fondren Rd", "LOCATION")]


@pytest.mark.parametrize("phrase, hours", [
    ("for 30 hours", 30), ("2 days", 48), ("dos días", 48), ("desde ayer", 24),
    ("since last night", 12), ("a day", 24), ("hace 6 horas", 6), ("36hrs", 36),
    ("since the storm", None), ("for a while", None),
    ("for two days", 48), ("for three days", 72), ("sin agua por dos días", 48),
    ("a couple of days", 48), ("1,000 hours", 1000), ("on dialysis", None), ("minimum", None),
])
def test_parse_hours(phrase, hours):
    assert parse_hours(phrase) == hours


def test_nearest_need():
    text = "no water for 2 days and no insulin since yesterday"
    assert nearest_need(text, 9, 19, ["water", "meds"]) == "water"
    assert nearest_need(text, 35, 50, ["water", "meds"]) == "meds"
    assert nearest_need(text, 9, 19, ["food"]) is None, "no food keyword nearby"
    assert nearest_need(text, 9, 19, []) is None
    far = "no power for 3 days" + " " * 60 + "need water"
    assert nearest_need(far, 13, 19, ["water"]) is None, "a power outage is not a water need"


def test_plain_keeps_indices():
    from rescue_tagger.durations import plain
    for t in ("Mi mamá ½ ﬁne İstanbul", "AGUA desde AYER"):
        assert len(plain(t)) == len(t)


def test_count_class():
    assert [count_class(x) for x in (None, 0, 1, 7, 20, 300)] == [0, 0, 1, 7, 20, 20]


def _batch_and_out(b=2, t=5):
    out = {"token_logits": torch.zeros(b, t, len(L.BIO_TAGS), requires_grad=True)}
    batch = {"weight": torch.ones(b), "token_labels": torch.full((b, t), IGNORE),
             "people_count": torch.full((b,), IGNORE)}
    for h in L.BINARY_HEADS:
        out[h] = torch.zeros(b, requires_grad=True)
        batch[h], batch[f"{h}_mask"] = torch.zeros(b), torch.zeros(b)
    for h, kinds in L.MULTI_LABEL_HEADS.items():
        out[h] = torch.zeros(b, len(kinds), requires_grad=True)
        batch[h], batch[f"{h}_mask"] = torch.zeros(b, len(kinds)), torch.zeros(b)
    for h, kinds in L.SINGLE_LABEL_HEADS.items():
        out[h] = torch.zeros(b, len(kinds), requires_grad=True)
        batch[h] = torch.full((b,), IGNORE)
    out["people_count"] = torch.zeros(b, 21, requires_grad=True)
    return out, batch


def test_loss_masks_unknown_labels():
    out, batch = _batch_and_out()
    batch["is_request"] = torch.tensor([1.0, 0.0])
    batch["is_request_mask"] = torch.tensor([1.0, 0.0])  # second row: unknown
    loss, parts = compute_loss(out, batch, {})
    assert set(parts) == {"is_request"}, "heads with no labels contribute nothing"
    loss.backward()
    assert out["is_request"].grad[1] == 0, "a masked row gets no gradient"
    assert out["hazards"].grad is None or out["hazards"].grad.abs().sum() == 0


def test_decode_snaps_to_whole_words_and_merges():
    text = "behind the Kroger on Westheimer"
    offsets = [(0, 6), (6, 10), (10, 17), (17, 20), (20, 25), (25, 31)]
    tags = [T["O"], T["O"], T["B-LOCATION"], T["O"], T["B-LOCATION"], T["O"]]  # "West" | "heimer"
    spans = decode_spans(text, offsets, tags, [1.0] * 6)
    assert [text[s["start"]:s["end"]] for s in spans] == ["Kroger", "Westheimer"]


def test_confidence_truncates_below_thresholds():
    from rescue_tagger.infer import _r
    assert _r(0.5996) == 0.599 and _r(0.2499) == 0.249 and _r(0.75) == 0.75


def test_one_duration_covers_a_listed_group_of_needs():
    from rescue_tagger.durations import needs_for_span

    def kinds(text, phrase, ks=("food", "water", "meds")):
        i = text.index(phrase)
        return sorted(k for k, _ in needs_for_span(text, i, i + len(phrase), list(ks)))

    assert kinds("no food or water for 2 days", "for 2 days") == ["food", "water"]
    assert kinds("sin comida ni agua desde ayer", "desde ayer") == ["food", "water"]
    assert kinds("out of food, water and medicine for 3 days", "for 3 days") == [
        "food", "meds", "water"]
    assert kinds("no food and no water since last night", "since last night") == ["food", "water"]
    # Separate durations stay separate: the list stops at the other duration.
    t = "no food for 2 days and no water since this morning"
    assert kinds(t, "for 2 days") == ["food"]
    assert kinds(t, "since this morning") == ["water"]
    # A need not predicted is never added; a far keyword is not a list member.
    assert kinds("no food or water for 2 days", "for 2 days", ("food",)) == ["food"]
    assert kinds("we have plenty of water at the shelter. no food for a day", "for a day") == [
        "food"]
    # "med" inside "immediately" is not a meds keyword.
    assert kinds("need insulin immediately, no food or water for 2 days", "for 2 days") == [
        "food", "water"]


def test_soft_labels_train_toward_unsure_and_are_validated():
    from rescue_tagger.features import encode
    from rescue_tagger.records import validate_record

    class Tok:
        def __call__(self, text, truncation, max_length, return_offsets_mapping):
            return {"input_ids": [1, 2], "attention_mask": [1, 1],
                    "offset_mapping": [(0, 2), (3, 5)]}

    rec = {"id": "synthetic:x:1", "source": "synthetic", "lang": "en", "text": "it's bad here",
           "split": "train", "labels": {"is_stale_post": False}, "spans": [],
           "soft_labels": {"is_request": 0.5}}
    validate_record(rec)
    item = encode(rec, Tok(), 16)
    assert item["is_request"] == 0.5 and item["is_request_mask"] == 1.0
    with pytest.raises(ValueError):
        validate_record({**rec, "labels": {"is_request": True}})
    with pytest.raises(ValueError):
        validate_record({**rec, "soft_labels": {"is_request": 1.5}})



def test_calibration_ignores_soft_targets():
    from rescue_tagger.calibrate import fit_mask

    d = {"target:is_request_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
         "target:is_request": torch.tensor([1.0, 0.0, 0.5, 0.0])}
    assert fit_mask(d, "is_request").tolist() == [True, True, False, False]
