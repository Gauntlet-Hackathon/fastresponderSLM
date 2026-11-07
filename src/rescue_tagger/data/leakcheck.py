"""Keep evaluation text out of training data (brief, data rules 1, 2 and 4).

A training row is dropped when its text matches any protected text:
  - exactly, after normalization (case, punctuation, whitespace, accents), or
  - near-verbatim: at least `threshold` of its word 8-grams appear in one protected text.
Protected texts are every eval-only row (Munro test, HumAID test, the Harvey hold-out, synthetic
test), the contract fixtures, and any eval set files passed in (e.g. the main repo's evals/sets/).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

N = 8
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(_WORD.findall(text))


def shingles(norm: str, n: int = N) -> set[int]:
    words = norm.split()
    return {hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


class LeakIndex:
    def __init__(self, protected: Iterable[str], threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.exact: set[str] = set()
        self.by_shingle: dict[int, set[int]] = defaultdict(set)
        for doc_id, text in enumerate(protected):
            norm = normalize(text)
            if not norm:
                continue
            self.exact.add(norm)
            for sh in shingles(norm):
                self.by_shingle[sh].add(doc_id)

    def __len__(self) -> int:
        return len(self.exact)

    def leaks(self, text: str) -> bool:
        norm = normalize(text)
        if norm in self.exact:
            return True
        mine = shingles(norm)
        if not mine:
            return False
        hits: dict[int, int] = defaultdict(int)
        for sh in mine:
            for doc_id in self.by_shingle.get(sh, ()):
                hits[doc_id] += 1
        return any(h / len(mine) >= self.threshold for h in hits.values())
