"""One-off fixes to the synthetic batches for data v3 (2026-11-07). Labels change only in fix 2.

1. Storm names. Every old-post synthetic row named "Harvey 2017", and no other row
named Harvey, which teaches "Harvey means old post". The Harvey hold-out is made of Harvey tweets,
so that shortcut would hit it directly. This swaps the storm for another past Houston storm
picked by a hash of the row id (and any 2017 for that storm's year), and moves every span to match.
2. Powered medical devices. The rubric counts dependence on treatment as "medical", so a row with
   medical_device_no_power also gets the medical vulnerable label (the earlier generator prompt
   forbade it).

Idempotent.

    python data/synthetic/fix_v3.py data/synthetic/batches/b0001-pilot.jsonl ...
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

STORMS = [("Ike", "2008"), ("Allison", "2001"), ("Imelda", "2019"), ("Beryl", "2024"),
          ("Rita", "2005"), ("Alicia", "1983")]
NAME = re.compile(r"harvey", re.I)
YEAR = re.compile(r"\b2017\b")


def case_like(src: str, new: str) -> str:
    if src.isupper():
        return new.upper()
    if src.islower():
        return new.lower()
    return new


def rename(rec: dict) -> bool:
    if not rec["labels"].get("is_stale_post") or rec["meta"].get("storm_renamed"):
        return False
    text = rec["text"]
    name, year = STORMS[int(hashlib.sha1(rec["id"].encode()).hexdigest(), 16) % len(STORMS)]
    edits = []  # (start, end, replacement)
    edits += [(m.start(), m.end(), case_like(m.group(), name)) for m in NAME.finditer(text)]
    if not edits:
        return False
    edits += [(m.start(), m.end(), year) for m in YEAR.finditer(text)]  # Harvey's year
    edits.sort()

    def shift(pos: int) -> int:
        out = pos
        for s, e, new in edits:
            if e <= pos:
                out += len(new) - (e - s)
            elif s < pos < e:
                raise ValueError(f"span boundary inside an edit in {rec['id']}")
        return out

    new_text, last = [], 0
    for s, e, new in edits:
        new_text += [text[last:s], new]
        last = e
    new_text.append(text[last:])
    rec["text"] = "".join(new_text)
    for sp in rec["spans"]:
        sp["start"], sp["end"] = shift(sp["start"]), shift(sp["end"])
    g = rec["gold_extracted"]
    g["spans"] = [s for s in rec["spans"] if s["label"] != "CONTACT"]
    t = next((s for s in rec["spans"] if s["label"] == "TIME"), None)
    g["time_ref"]["text"] = rec["text"][t["start"]:t["end"]] if t else None
    loc = next((s for s in rec["spans"] if s["label"] == "LOCATION"), None)
    g["location"]["text"] = rec["text"][loc["start"]:loc["end"]] if loc else None
    rec["meta"]["storm_renamed"] = f"Harvey -> {name}"
    return True


def device_is_medical(rec: dict) -> bool:
    if not rec["labels"]["hazards"].get("medical_device_no_power") or \
            rec["labels"]["vulnerable"].get("medical"):
        return False
    rec["labels"]["vulnerable"]["medical"] = True
    rec["gold_extracted"]["vulnerable"].append({"kind": "medical", "confidence": 1.0})
    rec["meta"]["relabeled"] = "medical_device_no_power implies medical (rubric)"
    return True


def main(paths: list[str]) -> None:
    for p in map(Path, paths):
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        n = sum(rename(r) for r in rows)
        m = sum(device_is_medical(r) for r in rows)
        p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                             for r in rows))
        print(f"{p}: renamed {n}, device rows relabeled medical {m}")


if __name__ == "__main__":
    main(sys.argv[1:])
