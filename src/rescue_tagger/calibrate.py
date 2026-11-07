"""Per-head temperature scaling on the dev split, plus expected calibration error (ECE).

  python -m rescue_tagger.calibrate outputs/laya/best --data data/processed/dev.jsonl

The engine branches on fixed thresholds (0.25 / 0.5 / 0.6 / 0.75), so a confidence of 0.8 must mean
right about 80% of the time. One temperature per head is fitted by minimizing negative
log-likelihood on dev rows that carry that head's label, then written to calibration.json in the
model directory (infer.Extractor applies it). Never fit on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rescue_tagger import labels as L
from rescue_tagger.features import IGNORE, collate, encode
from rescue_tagger.model import RescueTagger
from rescue_tagger.records import dev_half, read_jsonl

N_BINS = 10


def ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = N_BINS) -> tuple[float, list]:
    """Expected calibration error and the reliability-diagram bins."""
    bins, total, err = [], len(conf), 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, avg = correct[m].float().mean().item(), conf[m].mean().item()
        err += m.sum().item() / total * abs(acc - avg)
        bins.append({"lo": round(lo.item(), 2), "hi": round(hi.item(), 2), "n": int(m.sum()),
                     "accuracy": round(acc, 4), "confidence": round(avg, 4)})
    return err, bins


def fit_temperature(nll) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


@torch.no_grad()
def collect(model: RescueTagger, tok, rows: list[dict[str, Any]], bs: int = 64):
    """Logits and targets for every head over `rows`."""
    model.eval()
    items = [encode(r, tok, model.cfg.max_length) for r in rows]
    acc: dict[str, list[torch.Tensor]] = {}
    for i in range(0, len(items), bs):
        b = collate(items[i:i + bs], tok.pad_token_id)
        out = model(b["input_ids"], b["attention_mask"])
        for k, v in out.items():
            acc.setdefault(f"logits:{k}", []).append(v)
        for k, v in b.items():
            if k not in ("input_ids", "attention_mask"):
                acc.setdefault(f"target:{k}", []).append(v)
    return {k: torch.cat(v) if not k.endswith("token_logits") and not k.endswith("token_labels")
            else torch.cat([x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1)
                            for x in v])
            for k, v in acc.items()}


DEFAULT_EXCLUDE = ("humaid",)  # weak labels: donation appeals count as "requests" there


def calibrate(model_dir: Path, data: Path,
              exclude_sources: tuple[str, ...] = DEFAULT_EXCLUDE,
              half: str | None = None) -> dict[str, Any]:
    """Fit on dev rows from sources whose labels match the rubric (not weak-label sources).
    `half="calib"` fits on one fixed half of dev so the other half can score runs honestly."""
    model, tok = RescueTagger.load(model_dir)
    rows = [r for r in read_jsonl(data) if r["source"] not in exclude_sources
            and (half is None or dev_half(r["id"]) == half)]
    if any(r["split"] == "test" for r in rows):
        raise RuntimeError("refusing to calibrate on test rows")
    d = collect(model, tok, rows)
    temps: dict[str, float] = {}
    report: dict[str, Any] = {}

    for h in (*L.BINARY_HEADS, *L.MULTI_LABEL_HEADS):
        m = d[f"target:{h}_mask"] > 0
        if m.sum() < 20:
            continue
        logits, y = d[f"logits:{h}"][m], d[f"target:{h}"][m]
        temps[h] = fit_temperature(
            lambda t, z=logits, y=y: F.binary_cross_entropy_with_logits(z / t, y))
        p_before, p_after = torch.sigmoid(logits), torch.sigmoid(logits / temps[h])
        # Reliability of the positive-class probability itself, per kind: the engine bands on
        # p(hazard), so p = 0.7 must mean present about 70% of the time.
        conf_b, conf_a = p_before, p_after
        correct = y.bool()
        report[h] = {"T": temps[h], "n": int(m.sum()),
                     "ece_before": ece(conf_b.flatten(), correct.flatten())[0],
                     "ece_after": ece(conf_a.flatten(), correct.flatten())[0],
                     "bins_after": ece(conf_a.flatten(), correct.flatten())[1]}

    singles = [*L.SINGLE_LABEL_HEADS, "people_count"]
    for h in singles:
        y = d[f"target:{h}"]
        m = y != IGNORE
        if m.sum() < 20:
            continue
        logits, y = d[f"logits:{h}"][m], y[m]
        temps[h] = fit_temperature(lambda t, z=logits, y=y: F.cross_entropy(z / t, y))
        pa = torch.softmax(logits / temps[h], -1)
        conf, pred = pa.max(-1)
        e, bins = ece(conf, pred == y)
        report[h] = {"T": temps[h], "n": int(m.sum()), "ece_after": e, "bins_after": bins}

    y = d["target:token_labels"]
    m = y != IGNORE
    if m.sum() >= 20:
        z = d["logits:token_logits"][m]
        temps["tokens"] = fit_temperature(lambda t, z=z, y=y[m]: F.cross_entropy(z / t, y))

    (model_dir / "calibration.json").write_text(
        json.dumps({"temperatures": temps, "fitted_on": str(data), "half": half,
                    "report": report}, indent=2)
        + "\n")
    return {"temperatures": temps,
            "ece_after": {h: round(r["ece_after"], 4) for h, r in report.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/processed/dev.jsonl"))
    ap.add_argument("--exclude-sources", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--half", choices=["calib", "select"], default=None,
                    help="fit on one fixed half of dev only")
    args = ap.parse_args(argv)
    print(json.dumps(calibrate(args.model_dir, args.data, tuple(args.exclude_sources), args.half),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
