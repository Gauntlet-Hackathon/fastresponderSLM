"""Publish a trained model directory to the Hugging Face Hub and return its pinned revision.

  python -m rescue_tagger.publish outputs/laya/best --repo paawandesai/rescue-tagger

Laptops and evals must load `hf:<repo>@<sha>`, never `main`; this prints that version string.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_REPO = "paawandesai/rescue-tagger"


def model_card(model_dir: Path, eval_reports: dict[str, Any] | None) -> str:
    cfg = json.loads((model_dir / "tagger_config.json").read_text())
    cal = model_dir / "calibration.json"
    temps = json.loads(cal.read_text())["temperatures"] if cal.exists() else {}
    evals = json.dumps(eval_reports or {}, indent=2)
    return f"""---
license: cc-by-nc-sa-4.0
language: [en, es]
extra_gated_prompt: >-
  This model is for non-commercial research use only (CC BY-NC-SA 4.0, inherited from its HumAID
  training data). By requesting access you agree to use it only for non-commercial research and
  to share any derivative under the same license.
extra_gated_fields:
  I will use this model only for non-commercial research: checkbox
tags: [token-classification, text-classification, disaster-response, calibrated]
base_model: {cfg["backbone"]}
---
# rescue-tagger

Reads one flood-rescue request (English, Spanish or mixed) and returns the `extracted` payload of
the rescue triage product: location, people count, vulnerable people, hazards, needs, request kind,
time reference and whether it is a request, each with a calibrated confidence. It never decides
priority; a deterministic rules engine does. Runs offline on CPU; outputs are deterministic.

- Backbone: `{cfg["backbone"]}` + BIO span tagger + one head per field.
- Calibration: per-head temperature scaling on the dev split: `{json.dumps(temps)}`.
- Code: https://github.com/Gauntlet-Hackathon/fastresponderSLM

## License and intended use
**CC BY-NC-SA 4.0: non-commercial, research use only.** The license is inherited from HumAID,
which is part of the training data. Access is gated: accept the terms on this page to download.
Derivatives must be shared under the same license.

## Training data
- **HumAID** (CC BY-NC-SA 4.0): crowd-labeled disaster tweets, 2016-2019. Hurricane Harvey
  tweets are held out of training and used only for evaluation. Some classes are relabeled or
  masked for `is_request`; see `docs/DATA.md` in the code repository.
- **Munro disaster response messages** (CC BY 4.0): human-labeled messages from Haiti 2010,
  Pakistan 2010, Superstorm Sandy 2012 and news, via https://github.com/rmunro/disaster_response_messages.
- **Synthetic requests**: English, Spanish and mixed messages generated label-first with an OpenAI
  model and blind-verified.
- **Backbone**: `{cfg["backbone"]}`, under its own license (see its model card).

## Citation
If you use this model, please cite HumAID:

```bibtex
@inproceedings{{alam2021humaid,
  title     = {{HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning
               Benchmarks}},
  author    = {{Alam, Firoj and Qazi, Umair and Imran, Muhammad and Ofli, Ferda}},
  booktitle = {{Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)}},
  year      = {{2021}}
}}
```

```python
from rescue_tagger.infer import Extractor
ex = Extractor.load(snapshot_dir)  # huggingface_hub.snapshot_download(repo, revision=<sha>)
ex.extract("HELP! roof, 3 kids, water rising, behind the Kroger on Westheimer")
```

## Model-only evaluation
```json
{evals}
```
"""


def publish(model_dir: Path, repo: str = DEFAULT_REPO, run_name: str = "",
            eval_reports: dict[str, Any] | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    # Non-commercial research only: downloads require accepting the card's terms while logged in.
    api.update_repo_settings(repo, gated="auto")
    (model_dir / "README.md").write_text(model_card(model_dir, eval_reports))
    info = api.upload_folder(repo_id=repo, folder_path=str(model_dir),
                             commit_message=f"Upload {run_name or model_dir.name}")
    sha = info.oid
    return {"repo": repo, "sha": sha, "model_version": f"hf:{repo}@{sha}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args(argv)
    print(json.dumps(publish(args.model_dir, args.repo), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
