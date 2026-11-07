"""Multi-task fine-tuning: masked losses per head, per-source sampling, best checkpoint on dev.

  python -m rescue_tagger.train --config configs/train_mmbert_small.yaml

Reads data/processed/{train,dev}.jsonl (built by rescue_tagger.data.build). Never reads test or
harvey. Writes <out_dir>/best/ (loadable with RescueTagger.load) and <out_dir>/history.json.

Guards: a dev check every `eval_every_steps` (default once per epoch) logs training loss next to
dev loss (all, real text, synthetic) so over- and underfitting are visible; the run stops after
`early_stop_patience` checks without a new best, and a non-finite loss fails the run at once.
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
from rescue_tagger.records import dev_half, read_jsonl

FORBIDDEN_SPLITS = {"test"}


def parse_override(kv: str) -> tuple[str, Any]:
    """`key=value` -> (key, value). JSON first (so 2e-05 is a float, which YAML 1.1 reads as a
    string), YAML as the fallback for bare words."""
    k, v = kv.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, yaml.safe_load(v)


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
    missing = set(counts) - set(source_share)
    if missing:
        raise ValueError(f"source_share has no entry for {sorted(missing)}; set 0 to drop them")
    present = {s: source_share[s] for s in counts}
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
def dev_loss(model, loader, head_weights, device) -> tuple[float, dict[str, float]]:
    """Macro average over heads, so checkpoint choice is not decided by whichever head has the
    most dev rows (is_request from Munro/HumAID); spans, hazards and location count equally."""
    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        _, parts = compute_loss(model(b["input_ids"], b["attention_mask"]), b, head_weights)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    model.train()
    per_head = {k: sums[k] / counts[k] for k in sums}
    return sum(per_head.values()) / max(len(per_head), 1), per_head


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg.get("seed", 13)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = RescueTagger.from_backbone(cfg["backbone"], revision=cfg.get("backbone_revision"),
                                            dropout=cfg.get("dropout", 0.1),
                                            max_length=cfg.get("max_length", 256))
    model.to(device)
    limit = cfg.get("limit_rows")
    train_rows = load_split(data_dir, "train", limit)
    # Checkpointing and early stopping use only the "calib" half of dev; the "select" half is
    # kept untouched for ranking runs against each other. The product team's dev rows (sets C
    # and D) are given to us for picking checkpoints, so all of them join the checkpoint check.
    dev_rows = [r for r in load_split(data_dir, "dev", limit)
                if r["source"] == "product" or dev_half(r["id"]) == "calib"]
    enc = [encode(r, tok, model.cfg.max_length) for r in train_rows]
    random.Random(seed).shuffle(dev_rows)  # mix sources within dev batches
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

    # Dev checks: real text and synthetic are tracked apart, so a run that only learns the
    # generator's style shows up as a widening gap; the checkpoint is picked on all of dev.
    real_dev = [e for e, r in zip(dev_enc, dev_rows, strict=True) if r["source"] != "synthetic"]
    product_dev = [e for e, r in zip(dev_enc, dev_rows, strict=True) if r["source"] == "product"]
    synth_dev = [e for e, r in zip(dev_enc, dev_rows, strict=True) if r["source"] == "synthetic"]
    subset_loaders = {
        name: DataLoader(sub, batch_size=cfg["batch_size"] * 2,
                         collate_fn=lambda x: collate(x, pad))
        for name, sub in (("real", real_dev), ("synthetic", synth_dev), ("product", product_dev))
        if sub}
    # The product rows are ~0.3% of dev, so on their own they could never move the checkpoint;
    # they get a fixed share of the checkpoint score instead (0 = plain macro dev loss).
    product_w = cfg.get("checkpoint_product_weight", 0.0) if product_dev else 0.0
    eval_every = cfg.get("eval_every_steps") or steps_per_epoch
    patience = cfg.get("early_stop_patience", 2)  # dev checks without improvement before stopping

    history: list[dict[str, Any]] = []
    best, step, since_best, stopped_early = float("inf"), 0, 0, False
    # Training loss is kept as the same macro average over heads as dev loss, so the two curves
    # are directly comparable.
    train_parts: dict[str, list[float]] = {}
    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(b["input_ids"], b["attention_mask"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = compute_loss(out, b, head_weights)
            if not parts:
                continue  # nothing labeled in this batch (rare); no gradient to take
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}: {parts}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            for k, v in parts.items():
                train_parts.setdefault(k, []).append(v)
            if step % cfg.get("log_every", 50) == 0:
                print(json.dumps({"step": step, "loss": round(float(loss), 4),
                                  **{k: round(v, 4) for k, v in parts.items()}}), flush=True)
            if step % eval_every and step != total_steps:
                continue
            dl, per_head = dev_loss(model, dev_loader, head_weights, device)
            entry = {"epoch": epoch + 1, "step": step,
                     "train_loss": round(sum(sum(v) / len(v) for v in train_parts.values())
                                         / max(len(train_parts), 1), 4),
                     "dev_loss": round(dl, 4),
                     "dev_per_head": {k: round(v, 4) for k, v in per_head.items()},
                     "minutes": round((time.time() - t0) / 60, 2)}
            for name, sub_loader in subset_loaders.items():
                entry[f"dev_loss_{name}"] = round(dev_loss(model, sub_loader, head_weights,
                                                           device)[0], 4)
            train_parts = {}
            if product_w:
                dl = (1 - product_w) * dl + product_w * entry["dev_loss_product"]
                entry["checkpoint_score"] = round(dl, 4)
            if dl < best:
                best, since_best = dl, 0
                model.save(out_dir / "best", tok)
                entry["saved"] = True
            else:
                since_best += 1
            history.append(entry)
            print(json.dumps(entry), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if since_best >= patience:
                stopped_early = True
                break
        if stopped_early:
            print(json.dumps({"early_stop": True, "step": step, "best_dev_loss": best}), flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"best_dev_loss": best, "stopped_early": stopped_early, "steps": step,
            "history": history}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=1 limit_rows=200")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for kv in args.set:
        k, v = parse_override(kv)
        cfg[k] = v
    print(json.dumps(train(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
