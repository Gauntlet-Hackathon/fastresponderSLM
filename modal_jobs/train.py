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
    # Faster GPUs first (2-3x an L4 for about the same cost per run); a fallback list because one
    # run waited 97 minutes for an L4 before its container started.
    gpu=["H100", "L40S", "A100-40GB", "A10G", "L4"],
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
    import time as _time

    # First line in the log, so a long wait for a GPU is not mistaken for a hung run.
    print(f"container started {_time.strftime('%H:%M:%S', _time.gmtime())} UTC", flush=True)
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
    # Product dev rows pick checkpoints (all of them), so they stay out of the select half that
    # ranks runs; they have their own report below.
    nop = ("--exclude-source", "product")
    score("dev_real", dev, "--half", "select", "--exclude-source", "synthetic", *nop)
    score("dev_synthetic", dev, "--half", "select", "--source", "synthetic")
    score("dev_rubric", dev, "--half", "select", "--exclude-source", "humaid", *nop)
    if Path("data/product/rescue_eval_train_dev.jsonl").exists():
        # Product team's dev rows (sets C and D), head by head: their requested report.
        sh("python", "-m", "rescue_tagger.product_eval", "--model", best,
           "--out", f"{out_dir}/eval_product_dev.json")
        reports["product_dev"] = json.loads(
            Path(f"{out_dir}/eval_product_dev.json").read_text())[best]
    if final:
        reads = Path("/runs/eval_reads.jsonl")
        for name, data, extra in (("test_real", "data/processed/test.jsonl",
                                   ["--exclude-source", "synthetic"]),
                                  ("test_synthetic", "data/processed/test.jsonl",
                                   ["--source", "synthetic"]),
                                  ("harvey", "data/processed/harvey.jsonl", []),
                                  # Secondary: same sets with the train/dev category rules applied
                                  # (docs/DATA.md); the upstream-label scores above stay primary.
                                  ("test_real_rubric_labels", "data/processed/test_rubric.jsonl",
                                   ["--exclude-source", "synthetic"]),
                                  ("harvey_rubric_labels", "data/processed/harvey_rubric.jsonl",
                                   [])):
            score(name, data, *extra)
            log_eval_read(reads, run_name, name)

    from rescue_tagger.train import parse_override

    cfg = yaml.safe_load(Path(config).read_text())
    for kv in overrides:
        k, v = parse_override(kv)
        cfg[k] = v
    if reuse:
        # Reused weights were trained on the source run's data, whatever the config says now.
        src = Path(f"/runs/{reuse}/result.json")
        if src.exists():
            src_cfg = json.loads(src.read_text()).get("config", {})
            cfg["data_version"] = src_cfg.get("data_version", cfg.get("data_version"))
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
    if final:
        # The morning read re-scores the champion's own weights: never a candidate. Promoting it
        # would crown the same model under a new name (across data versions, as a new baseline).
        result["decision"] = {"promote": False, "champion": None,
                              "reasons": ["final read of the champion, not a candidate"]}
    else:
        result["decision"] = compare(Path(f"{out_dir}/result.json"),
                                     Path("/runs/champion.json"), Path("/runs"), promote=True)
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


@app.function(cpu=4, timeout=1800, volumes={"/runs": runs})
def extract_sample(run: str, text: str) -> dict:
    """One extracted payload from a finished run's calibrated model (CPU, as in the product)."""
    from rescue_tagger.infer import Extractor

    return Extractor.load(f"/runs/{run}/best", model_version=run).extract(text)


@app.local_entrypoint()
def sample(run: str, text: str) -> None:
    import json

    print(json.dumps(extract_sample.remote(run, text), indent=2, ensure_ascii=False))


@app.function(cpu=8, timeout=2 * 60 * 60, volumes={"/runs": runs},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def compare_run(run: str) -> str:
    """Our model against the stored Opus outputs (results/llm/) on the fixed comparison rows.
    Reads test and Harvey, so each call is logged as an eval read; it is a report, never a
    selection signal."""
    import subprocess

    from rescue_tagger.loop import log_eval_read

    os.chdir(REMOTE_ROOT)

    def sh(*cmd: str) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    sh("git", "clone", "--depth", "1",
       "https://github.com/Gauntlet-Hackathon/gauntletvshackathon", "/tmp/eval_repo")
    sh("python", "-m", "rescue_tagger.data.build", "--config", "configs/data.yaml")
    for name in ("test", "test_rubric", "harvey", "harvey_rubric"):  # logged before reading
        log_eval_read(Path("/runs/eval_reads.jsonl"), f"compare:{run}", name)
    runs.commit()
    sh("python", "-m", "rescue_tagger.compare", "--slm", f"/runs/{run}/best",
       "--out", f"/runs/{run}/compare.json")
    runs.commit()
    return Path(f"/runs/{run}/compare.md").read_text()


@app.local_entrypoint()
def compare(run: str) -> None:
    print(compare_run.remote(run))


@app.function(cpu=2, timeout=1800, volumes={"/runs": runs},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def publish_run(run: str, final_run: str = "") -> dict:
    """Publish a finished run's calibrated model (gated, CC BY-NC-SA) to the Hugging Face Hub as a
    new revision. The card carries the run's dev reports, plus the test-set reads of `final_run`
    (the morning check of these same weights) when given. Reads nothing new."""
    import json

    from rescue_tagger.publish import publish

    reports = {p.stem.removeprefix("eval_"): json.loads(p.read_text())
               for p in sorted(Path(f"/runs/{run}").glob("eval_*.json"))}
    if final_run:
        res = json.loads(Path(f"/runs/{final_run}/result.json").read_text())
        if res.get("reused_weights_from") != run:
            raise ValueError(f"{final_run} did not score {run}'s weights")
        tests = {f"final:{k}": v for k, v in res["eval"].items() if not k.startswith("dev_")}
        if not tests:
            raise ValueError(f"{final_run} has no test-set reads (not a final run)")
        reports.update(tests)
    # With a final run, publish its copy: same weights, and the calibration the card's test
    # numbers were measured with.
    src = Path(f"/runs/{final_run or run}/best")
    out = publish(src, run_name=run, eval_reports=reports)
    runs.commit()
    return out


@app.local_entrypoint()
def publish_existing(run: str, final_run: str = "") -> None:
    import json

    print(json.dumps(publish_run.remote(run, final_run), indent=2))
