"""The tagger: one shared encoder, a BIO token head, and one classification head per field.

Backbones (configs/train_*.yaml pick one):
  jhu-clsp/mmBERT-small                 140M, fastest on CPU
  convaiinnovations/laya-multilingual   mmBERT-base (322M with its own head), trained on typed
                                        decisions; we keep only its encoder weights
Both use the same tokenizer (mmBERT / Gemma vocabulary).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from rescue_tagger import labels as L

LAYA_PREFIX = "convaiinnovations/laya"


@dataclass
class TaggerConfig:
    backbone: str
    hidden_size: int
    dropout: float = 0.1
    bio_tags: list[str] = field(default_factory=lambda: list(L.BIO_TAGS))
    binary_heads: list[str] = field(default_factory=lambda: list(L.BINARY_HEADS))
    multi_label_heads: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in L.MULTI_LABEL_HEADS.items()}
    )
    single_label_heads: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in L.SINGLE_LABEL_HEADS.items()}
    )
    count_classes: int = 21  # 0 = not stated, 1..20 (see features.MAX_COUNT)
    max_length: int = 256


def load_encoder(backbone: str) -> tuple[nn.Module, object]:
    """Returns (encoder, tokenizer) for a backbone name or a local directory."""
    from transformers import AutoModel, AutoTokenizer

    if backbone.startswith(LAYA_PREFIX):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import ModernBertConfig, ModernBertModel

        cfg = ModernBertConfig.from_pretrained(backbone, subfolder="encoder")
        encoder = ModernBertModel(cfg)
        state = load_file(hf_hub_download(backbone, "model.safetensors"))
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Laya encoder mismatch: missing={missing} unexpected={unexpected}")
        tokenizer = AutoTokenizer.from_pretrained(backbone, subfolder="tokenizer")
        return encoder, tokenizer
    return AutoModel.from_pretrained(backbone), AutoTokenizer.from_pretrained(backbone)


class RescueTagger(nn.Module):
    def __init__(self, cfg: TaggerConfig, encoder: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        h = cfg.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.token_head = nn.Linear(h, len(cfg.bio_tags))
        self.heads = nn.ModuleDict()
        for name in cfg.binary_heads:
            self.heads[name] = nn.Linear(h, 1)
        for name, kinds in {**cfg.multi_label_heads, **cfg.single_label_heads}.items():
            self.heads[name] = nn.Linear(h, len(kinds))
        self.heads["people_count"] = nn.Linear(h, cfg.count_classes)

    @classmethod
    def from_backbone(cls, backbone: str, **kw) -> tuple[RescueTagger, object]:
        encoder, tokenizer = load_encoder(backbone)
        cfg = TaggerConfig(backbone=backbone, hidden_size=encoder.config.hidden_size, **kw)
        return cls(cfg, encoder), tokenizer

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out = {"token_logits": self.token_head(hidden)}
        for name, head in self.heads.items():
            logits = head(pooled)
            out[name] = logits.squeeze(-1) if name in self.cfg.binary_heads else logits
        return out

    # --- persistence: encoder via save_pretrained, heads + config alongside -----------------
    def save(self, out_dir: Path, tokenizer) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out_dir / "encoder")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        heads = {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in heads.items()}, out_dir / "heads.safetensors")
        (out_dir / "tagger_config.json").write_text(json.dumps(asdict(self.cfg), indent=2) + "\n")

    @classmethod
    def load(cls, model_dir: Path) -> tuple[RescueTagger, object]:
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        cfg = TaggerConfig(**json.loads((model_dir / "tagger_config.json").read_text()))
        encoder = AutoModel.from_pretrained(model_dir / "encoder")
        model = cls(cfg, encoder)
        missing, unexpected = model.load_state_dict(
            load_file(model_dir / "heads.safetensors"), strict=False
        )
        if unexpected or any(not k.startswith("encoder.") for k in missing):
            raise RuntimeError(f"head mismatch: missing={missing} unexpected={unexpected}")
        return model, AutoTokenizer.from_pretrained(model_dir / "tokenizer")
