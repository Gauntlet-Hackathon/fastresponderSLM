"""Records -> model tensors: BIO token labels from character spans, head targets with masks.

Unknown labels become ignore values (-100 for token/class targets, mask 0 for binary/multi-label),
so a Munro row with no spans trains the heads it has and nothing else.
"""

from __future__ import annotations

from typing import Any

import torch

from rescue_tagger import labels as L

IGNORE = -100
MAX_COUNT = 20  # people_count head classes: 0 = not stated, 1..20 (20 = "20 or more")


def char_spans_to_bio(text: str, offsets: list[tuple[int, int]], spans: list[dict]) -> list[int]:
    """Tag each token by the span covering its first non-space character."""
    tags = [L.TAG2ID["O"]] * len(offsets)
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special token
            tags[i] = IGNORE
            continue
        while s < e and text[s].isspace():
            s += 1
        if s >= e:  # whitespace-only token (e.g. "▁" before digits): no loss, decoder skips it
            tags[i] = IGNORE
            continue
        for sp in spans:
            if sp["start"] <= s < sp["end"]:
                j = i - 1
                while j >= 0 and tags[j] == IGNORE and offsets[j][0] != offsets[j][1]:
                    j -= 1  # look past whitespace-only tokens
                prev_inside = j >= 0 and offsets[j][1] > sp["start"] and tags[j] in (
                    L.TAG2ID[f"B-{sp['label']}"], L.TAG2ID[f"I-{sp['label']}"])
                tags[i] = L.TAG2ID[f"{'I' if prev_inside else 'B'}-{sp['label']}"]
                break
    return tags


def count_class(people_count: int | None) -> int:
    if people_count is None:
        return 0
    return max(1, min(people_count, MAX_COUNT))


def encode(rec: dict[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    enc = tokenizer(rec["text"], truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if rec["spans"] is None:
        token_labels = [IGNORE] * len(offsets)
    else:
        token_labels = char_spans_to_bio(rec["text"], offsets, rec["spans"])
    lab = rec["labels"]
    item: dict[str, Any] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_labels": token_labels,
        "weight": float(rec.get("weight", 1.0)),
        "people_count": count_class(rec["people_count"]) if "people_count" in rec else IGNORE,
    }
    for head in L.BINARY_HEADS:
        item[head] = float(lab[head]) if head in lab else 0.0
        item[f"{head}_mask"] = 1.0 if head in lab else 0.0
    for head, kinds in L.MULTI_LABEL_HEADS.items():
        item[head] = [float(lab[head][k]) for k in kinds] if head in lab else [0.0] * len(kinds)
        item[f"{head}_mask"] = 1.0 if head in lab else 0.0
    for head, kinds in L.SINGLE_LABEL_HEADS.items():
        item[head] = kinds.index(lab[head]) if head in lab else IGNORE
    return item


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(it["input_ids"]) for it in items)
    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([it["input_ids"] + [pad_id] * (n - len(it["input_ids"]))
                                   for it in items]),
        "attention_mask": torch.tensor([it["attention_mask"] + [0] * (n - len(it["attention_mask"]))
                                        for it in items]),
        "token_labels": torch.tensor([it["token_labels"] + [IGNORE] * (n - len(it["token_labels"]))
                                      for it in items]),
        "weight": torch.tensor([it["weight"] for it in items]),
        "people_count": torch.tensor([it["people_count"] for it in items]),
    }
    for head in (*L.BINARY_HEADS, *L.MULTI_LABEL_HEADS):
        batch[head] = torch.tensor([it[head] for it in items])
        batch[f"{head}_mask"] = torch.tensor([it[f"{head}_mask"] for it in items])
    for head in L.SINGLE_LABEL_HEADS:
        batch[head] = torch.tensor([it[head] for it in items])
    return batch
