"""Run the full training pipeline on a Modal GPU.

  modal run modal_jobs/train.py --config configs/train_laya.yaml [--publish]

Launched by .github/workflows/train.yml (repo secrets MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN).
Steps inside the container: build data -> train -> calibrate on dev -> evaluate on test splits ->
optionally publish to the Hugging Face Hub. Outputs persist in the `rescue-tagger-runs` volume.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.15.0",
        "transformers==5.22.2",
        "safetensors",
        "huggingface-hub>=0.26",
        "jsonschema>=4.23",
        "referencing>=0.35",
        "pyyaml>=6.0",
        "pandas>=2.2",
        "pyarrow>=17",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src", "TOKENIZERS_PARALLELISM": "false",
          "EVAL_REPO": "/tmp/eval_repo"})
    .add_local_dir(REPO_ROOT, REMOTE_ROOT, ignore=[".venv", ".git", "outputs", "data/raw",
                                                     "data/processed", "**/__pycache__",
                                                     "result.txt"])
)
runs = modal.Volume.from_name("rescue-tagger-runs", create_if_missing=True)
app = modal.App("rescue-tagger", image=image)


@app.function(
    gpu="L4",
    timeout=4 * 60 * 60,
    volumes={"/runs": runs},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def pipeline(config: str, run_name: str, publish: bool, overrides: list[str]) -> dict:
    import json
    import subprocess

    import yaml

    os.chdir(REMOTE_ROOT)
    out_dir = f"/runs/{run_name}"

    def sh(*cmd: str) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    # Fresh clone every run (not an image layer, which Modal caches), so eval sets added to the
    # product repo since the last run are protected from training too.
    sh("git", "clone", "--depth", "1",
       "https://github.com/Gauntlet-Hackathon/gauntletvshackathon", "/tmp/eval_repo")
    sh("python", "-m", "rescue_tagger.data.build", "--config", "configs/data.yaml")
    sh("python", "-m", "rescue_tagger.train", "--config", config, "--set",
       f"out_dir={out_dir}", *overrides)
    best = f"{out_dir}/best"
    sh("python", "-m", "rescue_tagger.calibrate", best, "--data", "data/processed/dev.jsonl")
    reports = {}
    for name, extra in {"test_all": [], "test_synthetic": ["--source", "synthetic"],
                        "harvey": []}.items():
        data = "data/processed/harvey.jsonl" if name == "harvey" else "data/processed/test.jsonl"
        sh("python", "-m", "rescue_tagger.evaluate", best, "--data", data, *extra,
           "--out", f"{out_dir}/eval_{name}.json")
        reports[name] = json.loads(Path(f"{out_dir}/eval_{name}.json").read_text())
    result = {"run": run_name, "config": yaml.safe_load(Path(config).read_text()),
              "eval": reports}
    if publish:
        from rescue_tagger.publish import publish as push

        result["published"] = push(Path(best), run_name=run_name, eval_reports=reports)
    Path(f"{out_dir}/result.json").write_text(json.dumps(result, indent=2) + "\n")
    runs.commit()
    return result


@app.local_entrypoint()
def main(config: str, run_name: str = "", publish: bool = False, overrides: str = "") -> None:
    import json
    import time

    name = run_name or f"{Path(config).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    result = pipeline.remote(config, name, publish, overrides.split() if overrides else [])
    print(json.dumps(result, indent=2))
