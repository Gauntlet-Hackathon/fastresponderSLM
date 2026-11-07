"""Run the full training pipeline on a Modal GPU.

  modal run modal_jobs/train.py::main --config configs/train_laya.yaml [--publish]
  modal run modal_jobs/train.py::diag --run <run name>

Launched by .github/workflows/train.yml (repo secrets MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN).
Steps inside the container: build data -> train -> calibrate on one half of dev -> score on the
other half -> compare with the current best model (/runs/champion.json) -> optionally score the
test sets (--final) and publish. Outputs persist in the `rescue-tagger-runs` volume.
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
    cpu=4,  # the latency gate times CPU inference with 4 threads; reserve them
    timeout=4 * 60 * 60,
    volumes={"/runs": runs},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def pipeline(config: str, run_name: str, publish: bool, overrides: list[str],
             final: bool = False, reuse: str = "") -> dict:
    """Build -> train -> calibrate (dev "calib" half) -> score on the dev "select" half ->
    compare with the current best model. Test sets and Harvey are scored only when `final` is set,
    and every such scoring is logged to /runs/eval_reads.jsonl. `reuse` names an earlier run whose
    trained weights are re-calibrated and re-scored instead of training again."""
    import json
    import shutil
    import subprocess

    import yaml

    from rescue_tagger.loop import compare, log_eval_read

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
    best = f"{out_dir}/best"
    if reuse:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(f"/runs/{reuse}/best", best, dirs_exist_ok=True)
    else:
        sh("python", "-m", "rescue_tagger.train", "--config", config, "--set",
           f"out_dir={out_dir}", *overrides)
    sh("python", "-m", "rescue_tagger.calibrate", best, "--data", "data/processed/dev.jsonl",
       "--half", "calib")

    def score(name: str, data: str, *extra: str) -> None:
        sh("python", "-m", "rescue_tagger.evaluate", best, "--data", data, *extra,
           "--out", f"{out_dir}/eval_{name}.json")
        reports[name] = json.loads(Path(f"{out_dir}/eval_{name}.json").read_text())

    reports: dict = {}
    dev = "data/processed/dev.jsonl"
    score("dev_real", dev, "--half", "select", "--exclude-source", "synthetic")
    score("dev_synthetic", dev, "--half", "select", "--source", "synthetic")
    score("dev_rubric", dev, "--half", "select", "--exclude-source", "humaid")
    if final:
        reads = Path("/runs/eval_reads.jsonl")
        for name, data, extra in (("test_real", "data/processed/test.jsonl",
                                   ["--exclude-source", "synthetic"]),
                                  ("test_synthetic", "data/processed/test.jsonl",
                                   ["--source", "synthetic"]),
                                  ("harvey", "data/processed/harvey.jsonl", [])):
            score(name, data, *extra)
            log_eval_read(reads, run_name, name)

    from rescue_tagger.train import parse_override

    cfg = yaml.safe_load(Path(config).read_text())
    for kv in overrides:
        k, v = parse_override(kv)
        cfg[k] = v
    result = {"run": run_name, "config": cfg, "reused_weights_from": reuse or None,
              "eval": reports}
    history = Path(f"{out_dir}/history.json")
    if history.exists():
        result["history"] = json.loads(history.read_text())
    Path(f"{out_dir}/result.json").write_text(json.dumps(result, indent=2) + "\n")
    # See runs that finished while this one trained (the champion and seed repeats), decide,
    # and commit at once so the next run compares against this decision.
    runs.commit()
    runs.reload()
    result["decision"] = compare(Path(f"{out_dir}/result.json"), Path("/runs/champion.json"),
                                 Path("/runs"), promote=True)
    Path(f"{out_dir}/decision.json").write_text(json.dumps(result["decision"], indent=2) + "\n")
    runs.commit()
    if publish:
        from rescue_tagger.publish import publish as push

        result["published"] = push(Path(best), run_name=run_name, eval_reports=reports)
    Path(f"{out_dir}/result.json").write_text(json.dumps(result, indent=2) + "\n")
    runs.commit()
    return result


@app.local_entrypoint()
def main(config: str, run_name: str = "", publish: bool = False, overrides: str = "",
         final: bool = False, reuse: str = "") -> None:
    import json
    import time

    name = run_name or f"{Path(config).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    result = pipeline.remote(config, name, publish, overrides.split() if overrides else [],
                             final, reuse)
    print(json.dumps(result, indent=2))


@app.function(cpu=8, timeout=2 * 60 * 60, volumes={"/runs": runs},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def diagnose_run(run: str) -> dict:
    """Request-detector deep check for a finished run: aggregates only on test and Harvey (logged
    as eval reads), plus example errors from dev, which is safe to read."""
    import json
    import subprocess

    from rescue_tagger.loop import log_eval_read

    os.chdir(REMOTE_ROOT)

    def sh(*cmd: str) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    sh("git", "clone", "--depth", "1",
       "https://github.com/Gauntlet-Hackathon/gauntletvshackathon", "/tmp/eval_repo")
    sh("python", "-m", "rescue_tagger.data.build", "--config", "configs/data.yaml")
    best, out = f"/runs/{run}/best", f"/runs/{run}"
    report = {}
    for name, data, extra in (("harvey", "data/processed/harvey.jsonl", []),
                              ("test_munro", "data/processed/test.jsonl", ["--source", "munro"]),
                              ("test_humaid", "data/processed/test.jsonl", ["--source", "humaid"]),
                              ("dev_munro", "data/processed/dev.jsonl",
                               ["--source", "munro", "--show-errors", "40"]),
                              ("dev_humaid", "data/processed/dev.jsonl",
                               ["--source", "humaid", "--show-errors", "40"])):
        sh("python", "-m", "rescue_tagger.diagnose", best, "--data", data, *extra,
           "--out", f"{out}/diagnose_{name}.json")
        report[name] = json.loads(Path(f"{out}/diagnose_{name}.json").read_text())
        if name.startswith(("harvey", "test")):
            log_eval_read(Path("/runs/eval_reads.jsonl"), f"diagnose:{run}", name)
    runs.commit()
    return report


@app.local_entrypoint()
def diag(run: str) -> None:
    import json

    print(json.dumps(diagnose_run.remote(run), indent=2, ensure_ascii=False))
