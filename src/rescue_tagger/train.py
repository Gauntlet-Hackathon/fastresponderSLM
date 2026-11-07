"""Multi-task fine-tuning: masked losses per head, per-source sampling, best checkpoint on dev.

  python -m rescue_tagger.train --config configs/train_mmbert_small.yaml

Reads data/processed/{train,dev}.jsonl (built by rescue_tagger.data.build). Never reads test or
harvey. Writes <out_dir>/best/ (loadable with RescueTagger.load) and <out_dir>/history.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from rescue_tagger import labels as L
from rescue_tagger.features import IGNORE, collate, encode
from rescue_tagger.model import RescueTagger
from rescue_tagger.records import read_jsonl

FORBIDDEN_SPLITS = {"test"}


def load_split(data_dir: Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """All rows of a split; `limit` keeps a seeded sample of that many per source (smoke tests)."""
    rows = []
    for rec in read_jsonl(data_dir / f"{split}.jsonl"):
        if rec["split"] in FORBIDDEN_SPLITS or rec.get("meta", {}).get("harvey_holdout"):
            raise RuntimeError(f"eval-only row {rec['id']} found in {split}.jsonl")
        rows.append(rec)
    if limit:
        rng = random.Random(0)
        by_source: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append(r)
        rows = [r for group in by_source.values()
                for r in rng.sample(group, min(limit, len(group)))]
    return rows


def sampler_for(rows: list[dict[str, Any]], source_share: dict[str, float], n: int):
    """Sample so each source contributes its configured share of every epoch."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    present = {s: source_share.get(s, 0.0) for s in counts}
    total = sum(present.values()) or 1.0
    weights = [present[r["source"]] / total / counts[r["source"]] for r in rows]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def compute_loss(out: dict[str, torch.Tensor], b: dict[str, torch.Tensor],
                 head_weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    w = b["weight"]
    parts: dict[str, torch.Tensor] = {}

    tok = F.cross_entropy(out["token_logits"].transpose(1, 2), b["token_labels"],
                          ignore_index=IGNORE, reduction="none")  # (B, T)
    tok_mask = (b["token_labels"] != IGNORE).float()
    if tok_mask.sum() > 0:
        per_row = (tok * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1)
        has = (tok_mask.sum(1) > 0).float()
        parts["tokens"] = (per_row * w * has).sum() / (w * has).sum()

    for head in L.BINARY_HEADS:
        m = b[f"{head}_mask"] * w
        if m.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(out[head], b[head], reduction="none")
            parts[head] = (loss * m).sum() / m.sum()
    for head in L.MULTI_LABEL_HEADS:
        m = b[f"{head}_mask"] * w
        if m.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(out[head], b[head], reduction="none").mean(1)
            parts[head] = (loss * m).sum() / m.sum()
    for head in (*L.SINGLE_LABEL_HEADS, "people_count"):
        valid = (b[head] != IGNORE).float() * w
        if valid.sum() > 0:
            loss = F.cross_entropy(out[head], b[head].clamp(min=0), reduction="none")
            parts[head] = (loss * valid).sum() / valid.sum()

    total = sum(head_weights.get(k, 1.0) * v for k, v in parts.items())
    return total, {k: v.item() for k, v in parts.items()}


@torch.no_grad()
def dev_loss(model, loader, head_weights, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss, _ = compute_loss(model(b["input_ids"], b["attention_mask"]), b, head_weights)
        tot += float(loss) * len(b["weight"])
        n += len(b["weight"])
    model.train()
    return tot / max(n, 1)


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg.get("seed", 13)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = RescueTagger.from_backbone(cfg["backbone"], dropout=cfg.get("dropout", 0.1),
                                            max_length=cfg.get("max_length", 256))
    model.to(device)
    limit = cfg.get("limit_rows")
    train_rows = load_split(data_dir, "train", limit)
    dev_rows = load_split(data_dir, "dev", limit)
    enc = [encode(r, tok, model.cfg.max_length) for r in train_rows]
    dev_enc = [encode(r, tok, model.cfg.max_length) for r in dev_rows]
    pad = tok.pad_token_id

    steps_per_epoch = cfg.get("steps_per_epoch") or math.ceil(len(enc) / cfg["batch_size"])
    sampler = sampler_for(train_rows, cfg["source_share"], steps_per_epoch * cfg["batch_size"])
    loader = DataLoader(enc, batch_size=cfg["batch_size"], sampler=sampler,
                        collate_fn=lambda x: collate(x, pad))
    dev_loader = DataLoader(dev_enc, batch_size=cfg["batch_size"] * 2,
                            collate_fn=lambda x: collate(x, pad))

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["lr_encoder"]},
        {"params": head_params, "lr": cfg["lr_heads"]},
    ], weight_decay=cfg.get("weight_decay", 0.01))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = int(total_steps * cfg.get("warmup_frac", 0.06))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warmup, 1))
        * max(0.0, (total_steps - s) / max(total_steps - warmup, 1)))
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    head_weights = cfg.get("head_weights", {})

    history, best, step = [], float("inf"), 0
    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(b["input_ids"], b["attention_mask"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = compute_loss(out, b, head_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 50) == 0:
                print(json.dumps({"step": step, "loss": round(float(loss), 4),
                                  **{k: round(v, 4) for k, v in parts.items()}}), flush=True)
        dl = dev_loss(model, dev_loader, head_weights, device)
        history.append({"epoch": epoch + 1, "step": step, "dev_loss": dl,
                        "minutes": round((time.time() - t0) / 60, 2)})
        print(json.dumps(history[-1]), flush=True)
        if dl < best:
            best = dl
            model.save(out_dir / "best", tok)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"best_dev_loss": best, "history": history}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=1 limit_rows=200")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    print(json.dumps(train(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
