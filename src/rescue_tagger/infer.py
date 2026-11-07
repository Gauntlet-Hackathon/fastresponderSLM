"""Deterministic CPU inference: raw text -> schema-valid `extracted` payload.

This is the module the field adapter (ticket F11) wraps:

    from rescue_tagger.infer import Extractor
    ex = Extractor.load("path/or/downloaded/snapshot")
    payload = ex.extract("HELP! roof, 3 kids, water rising, behind the Kroger on Westheimer")

Determinism: eval mode (no dropout), fixed thread count, no sampling, and every confidence rounded
to 3 decimals, so the same text gives byte-identical JSON across runs (the evals check 10 runs).
Calibration: per-head temperatures from calibration.json (rescue_tagger.calibrate) divide the
logits before the sigmoid/softmax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from rescue_tagger import labels as L
from rescue_tagger.durations import nearest_need, parse_hours
from rescue_tagger.model import RescueTagger
from rescue_tagger.records import validate_extracted

REPORT_FLOOR = 0.2   # list a hazard/vulnerable kind at or above this (engine ignores < 0.25)
NEED_MIN = 0.5       # needs carry no confidence in the schema, so only list confident ones
DIGITS = 3


def _r(x: float) -> float:
    return round(float(x), DIGITS)


def decode_spans(text: str, offsets: list[tuple[int, int]], tag_ids: list[int],
                 tag_probs: list[float]) -> list[dict[str, Any]]:
    """BIO tags -> character spans with a confidence (mean token prob), whitespace-trimmed."""
    spans: list[dict[str, Any]] = []
    cur = None
    for (s, e), tid, p in zip(offsets, tag_ids, tag_probs, strict=True):
        if s == e or not text[s:e].strip():  # special or whitespace-only token: no decision
            continue
        tag = L.ID2TAG[tid]
        if tag == "O":
            cur = None
            continue
        prefix, label = tag.split("-", 1)
        if cur is not None and cur["label"] == label and prefix == "I":
            cur["end"] = e
            cur["probs"].append(p)
        else:
            cur = {"start": s, "end": e, "label": label, "probs": [p]}
            spans.append(cur)
    out = []
    for sp in spans:
        s, e = sp["start"], sp["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append({"start": s, "end": e, "label": sp["label"],
                        "confidence": sum(sp["probs"]) / len(sp["probs"])})
    return out


class Extractor:
    def __init__(self, model: RescueTagger, tokenizer, temperatures: dict[str, float],
                 model_version: str = "local", threads: int = 4) -> None:
        torch.set_num_threads(threads)
        self.model = model.eval()
        self.tok = tokenizer
        self.t = temperatures
        self.model_version = model_version

    @classmethod
    def load(cls, model_dir: str | Path, model_version: str = "local", threads: int = 4):
        model_dir = Path(model_dir)
        model, tok = RescueTagger.load(model_dir)
        cal = model_dir / "calibration.json"
        temps = json.loads(cal.read_text())["temperatures"] if cal.exists() else {}
        return cls(model, tok, temps, model_version, threads)

    def _T(self, head: str) -> float:
        return float(self.t.get(head, 1.0))

    @torch.inference_mode()
    def raw(self, text: str) -> dict[str, Any]:
        """Calibrated probabilities for every head plus decoded spans (also used by evals)."""
        enc = self.tok(text, truncation=True, max_length=self.model.cfg.max_length,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        out = self.model(enc["input_ids"], enc["attention_mask"])
        tok_probs = torch.softmax(out["token_logits"][0] / self._T("tokens"), -1)
        tag_p, tag_id = tok_probs.max(-1)
        probs: dict[str, Any] = {}
        for h in L.BINARY_HEADS:
            probs[h] = float(torch.sigmoid(out[h][0] / self._T(h)))
        for h, kinds in L.MULTI_LABEL_HEADS.items():
            p = torch.sigmoid(out[h][0] / self._T(h)).tolist()
            probs[h] = dict(zip(kinds, p, strict=True))
        for h, kinds in L.SINGLE_LABEL_HEADS.items():
            p = torch.softmax(out[h][0] / self._T(h), -1).tolist()
            probs[h] = dict(zip(kinds, p, strict=True))
        probs["people_count"] = torch.softmax(
            out["people_count"][0] / self._T("people_count"), -1).tolist()
        spans = decode_spans(text, offsets, tag_id.tolist(), tag_p.tolist())
        return {"probs": probs, "spans": spans}

    def extract(self, text: str) -> dict[str, Any]:
        raw = self.raw(text)
        p, spans = raw["probs"], raw["spans"]

        def best(label: str) -> dict[str, Any] | None:
            cands = [s for s in spans if s["label"] == label]
            return max(cands, key=lambda s: (s["confidence"], -s["start"])) if cands else None

        loc_span = best("LOCATION")
        loc_type = max(L.LOCATION_TYPES, key=lambda k: p["location_type"][k])
        if loc_span is None or loc_type == "none":
            # No usable span: report "none" with the model's own probability of no location.
            location = {"text": None, "type": "none",
                        "confidence": _r(p["location_type"]["none"])}
        else:
            location = {"text": text[loc_span["start"]:loc_span["end"]], "type": loc_type,
                        "confidence": _r(p["location_type"][loc_type] * loc_span["confidence"])}

        pc = p["people_count"]
        k = max(range(len(pc)), key=lambda i: pc[i])
        people_count = {"value": None if k == 0 else k, "confidence": _r(pc[k])}

        def listed(head: str) -> list[dict[str, Any]]:
            return [{"kind": kind, "confidence": _r(v)} for kind, v in p[head].items()
                    if v >= REPORT_FLOOR]

        need_kinds = [kind for kind, v in p["needs"].items() if v >= NEED_MIN]
        hours: dict[str, float | None] = dict.fromkeys(need_kinds)
        for sp in (s for s in spans if s["label"] == "NEED_DURATION"):
            kind = nearest_need(text, sp["start"], sp["end"], need_kinds)
            h = parse_hours(text[sp["start"]:sp["end"]])
            if kind is not None and h is not None:
                hours[kind] = max(h, hours[kind] or 0)

        time_span = best("TIME")
        stale = p["is_stale_post"]
        is_req = p["is_request"]
        payload = {
            "location": location,
            "people_count": people_count,
            "vulnerable": listed("vulnerable"),
            "hazards": listed("hazards"),
            "time_ref": {
                "text": text[time_span["start"]:time_span["end"]] if time_span else None,
                "is_stale_post": stale >= 0.5,
                "confidence": _r(max(stale, 1 - stale)),
            },
            "is_request": {"value": is_req >= 0.5, "confidence": _r(max(is_req, 1 - is_req))},
            "spans": [{"start": s["start"], "end": s["end"], "label": s["label"]}
                      for s in spans if s["label"] != "CONTACT"],
            "needs": [{"kind": kind, "hours_without": hours[kind]} for kind in need_kinds],
            "request_kind": max(L.REQUEST_KINDS, key=lambda kk: p["request_kind"][kk]),
        }
        validate_extracted(payload)
        return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract fields from one message.")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("text")
    args = ap.parse_args(argv)
    print(json.dumps(Extractor.load(args.model_dir).extract(args.text), ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
