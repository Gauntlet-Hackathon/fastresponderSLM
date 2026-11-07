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
])
def test_parse_hours(phrase, hours):
    assert parse_hours(phrase) == hours


def test_nearest_need():
    text = "no water for 2 days and no insulin since yesterday"
    assert nearest_need(text, 9, 19, ["water", "meds"]) == "water"
    assert nearest_need(text, 35, 50, ["water", "meds"]) == "meds"
    assert nearest_need(text, 9, 19, ["food"]) == "food"
    assert nearest_need(text, 9, 19, []) is None


def test_count_class():
    assert [count_class(x) for x in (None, 0, 1, 7, 20, 300)] == [0, 1, 1, 7, 20, 20]


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
