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
- Training data: Munro disaster messages (CC BY 4.0), HumAID (CC BY-NC-SA 4.0, Harvey held out),
  and OpenAI-generated synthetic requests. The license follows HumAID's share-alike terms.
- Code: https://github.com/Gauntlet-Hackathon/fastresponderSLM

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
