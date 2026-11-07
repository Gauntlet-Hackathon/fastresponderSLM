"""HumAID (CC BY-NC-SA 4.0) -> unified records, with Hurricane Harvey held out for evaluation.

~77k English tweets from 19 disasters (2016-2019), one humanitarian class each, labeled by
qualified crowd workers (2 of 3 agreement). We use the QCRI/HumAID-event-type copy on Hugging Face
because it keeps `tweet_id`, which lets us recover the date (tweet ids are snowflakes) and hold out
Hurricane Harvey (Houston, Aug 2017) as real raw text for golden set B and out-of-domain tests.

Label mapping (weak: HumAID's "requests_or_urgent_needs" also covers donation appeals, so these rows
get a reduced loss weight from the data config):
  is_request    <- class == requests_or_urgent_needs
  request_kind  <- "other" for non-requests; unknown for requests
  dont_know_cant_judge rows are dropped.

Harvey hold-out: hurricane tweets dated 2017-08-17 .. 2017-09-03 (UTC), plus any tweet anywhere
that mentions Harvey or Houston. They go to a separate eval-only file and never reach training.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = "QCRI/HumAID-event-type"
REVISION = "388eefc83ff1f70c9680e9e8afad03eacd9ac7b9"
EVENT_TYPES = ("earthquake", "fire", "flood", "hurricane")
SPLIT_FILES = {"train": "train.json", "dev": "dev.json", "test": "test.json"}
HARVEY_START = datetime(2017, 8, 17, tzinfo=UTC)
HARVEY_END = datetime(2017, 9, 4, tzinfo=UTC)
HARVEY_WORDS = re.compile(r"harvey|houston", re.IGNORECASE)
TWITTER_EPOCH_MS = 1288834974657


def fetch(get, waits: tuple[float, ...] = (5, 15, 45)):
    """get() with retries on network errors (a reset connection killed whole training jobs)."""
    import http.client
    import time

    for w in (*waits, None):
        try:
            return get()
        except (OSError, http.client.HTTPException):   # resets, timeouts, cut-off reads
            if w is None:
                raise
            time.sleep(w)


def _read(req: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read()


def download(raw_dir: Path, token: str | None = None) -> Path:
    out = raw_dir / "humaid"
    out.mkdir(parents=True, exist_ok=True)
    for etype in EVENT_TYPES:
        for fname in SPLIT_FILES.values():
            target = out / f"{etype}_{fname}"
            if target.exists():
                continue
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{etype}/{fname}"
            req = urllib.request.Request(url)
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            target.write_bytes(fetch(lambda r=req: _read(r)))
    return out


def tweet_time(tweet_id: int) -> datetime:
    return datetime.fromtimestamp(((tweet_id >> 22) + TWITTER_EPOCH_MS) / 1000, tz=UTC)


def is_harvey(event_type: str, tweet_id: int, text: str) -> bool:
    if HARVEY_WORDS.search(text):
        return True
    return event_type == "hurricane" and HARVEY_START <= tweet_time(tweet_id) < HARVEY_END


def map_row(row: dict[str, Any], event_type: str, split: str) -> dict[str, Any] | None:
    cls = row["class_label"]
    text = (row.get("tweet_text") or "").strip()
    if not text or cls == "dont_know_cant_judge":
        return None
    is_request = cls == "requests_or_urgent_needs"
    labels: dict[str, Any] = {"is_request": is_request}
    if not is_request:
        labels["request_kind"] = "other"
    harvey = is_harvey(event_type, int(row["tweet_id"]), text)
    return {
        "id": f"humaid:{row['tweet_id']}",
        "source": "humaid",
        "lang": "en",
        "text": text,
        # Harvey rows are evaluation-only, whatever their upstream split was.
        "split": "test" if harvey else split,
        "labels": labels,
        "spans": None,
        "meta": {
            "class_label": cls,
            "event_type": event_type,
            "upstream_split": split,
            "harvey_holdout": harvey,
        },
    }


def load(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield every usable HumAID row; rows with meta.harvey_holdout are eval-only."""
    src = raw_dir / "humaid"
    seen: set[str] = set()
    for etype in EVENT_TYPES:
        for split, fname in SPLIT_FILES.items():
            for row in json.loads((src / f"{etype}_{fname}").read_text(encoding="utf-8")):
                rec = map_row(row, etype, split)
                if rec is None or rec["id"] in seen:
                    continue
                seen.add(rec["id"])
                yield rec
