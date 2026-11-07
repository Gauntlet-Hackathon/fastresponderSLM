"""Generate synthetic training data with open-weights models on a Modal GPU (no Claude, no OpenAI).

  modal run modal_jobs/synth.py --name b0004-focus-qwen --n 30 --seed 300000 --focus

A vLLM server (OpenAI-compatible) holds one model at a time. Phase 1: the generator writes messages
for label-first specs (rescue_tagger.data.synthetic). Phase 2: a model from another family labels
each message blind; rows where it disagrees with the spec are rejected. The batch lands in the
`rescue-tagger-runs` volume under synth/ and is printed between markers, so the GitHub workflow can
commit it. Launched by .github/workflows/synth.yml.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/repo"
GEN_MODEL = "Qwen/Qwen2.5-32B-Instruct"     # Apache-2.0
VERIFY_MODEL = "microsoft/phi-4"            # MIT, a different family from the generator

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.2", "openai>=1.40", "jsonschema>=4.23", "referencing>=0.35",
                 "pyyaml>=6.0")
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(REPO_ROOT, REMOTE_ROOT, ignore=[".venv", ".git", "outputs", "data/raw",
                                                     "data/processed", "**/__pycache__"])
)
runs = modal.Volume.from_name("rescue-tagger-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("rescue-tagger-hf-cache", create_if_missing=True)
app = modal.App("rescue-synth", image=image)
PORT = 8000


def serve(model: str) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                             "--model", model, "--port", str(PORT), "--max-model-len", "8192",
                             "--gpu-memory-utilization", "0.92"])
    import urllib.request

    for _ in range(180):  # up to 30 min for download + load
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode} while loading {model}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            return proc
        except OSError:
            time.sleep(10)
    proc.kill()
    raise RuntimeError(f"vLLM did not come up for {model}")


@app.function(gpu="H100", timeout=3 * 3600,
              volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def generate(name: str, n: int, seed: int, focus: bool, gen_model: str,
             verify_model: str) -> dict[str, str]:
    os.chdir(REMOTE_ROOT)
    os.environ.update({"OPENAI_BASE_URL": f"http://localhost:{PORT}/v1",
                       "OPENAI_SLM_API_KEY": "local"})
    out = Path(f"/runs/synth/{name}")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"

    def run(*args: str) -> None:
        print("+", " ".join(args), flush=True)
        subprocess.run([sys.executable, "-m", "rescue_tagger.data.synthetic", *args], check=True)

    proc = serve(gen_model)
    try:
        run("--n", str(n), "--seed", str(seed), "--out", str(raw), "--gen-model", gen_model,
            "--verify-model", "", "--workers", "32", *(["--focus"] if focus else []))
    finally:
        proc.kill()
        proc.wait()
    hf_cache.commit()
    proc = serve(verify_model)
    try:
        run("--verify-file", str(raw), "--verify-model", verify_model, "--out",
            str(out / f"{name}.jsonl"), "--workers", "32")
    finally:
        proc.kill()
        proc.wait()
    hf_cache.commit()
    rejected = "".join(p.read_text() for p in (out / "raw.rejected.jsonl",
                                                out / f"{name}.rejected.jsonl") if p.exists())
    (out / f"{name}.all-rejected.jsonl").write_text(rejected)
    runs.commit()
    return {"kept": (out / f"{name}.jsonl").read_text(), "rejected": rejected}


@app.local_entrypoint()
def main(name: str, n: int = 30, seed: int = 300000, focus: bool = True,
         gen_model: str = GEN_MODEL, verify_model: str = VERIFY_MODEL) -> None:
    res = generate.remote(name, n, seed, focus, gen_model, verify_model)
    Path(f"{name}.jsonl").write_text(res["kept"])
    Path(f"{name}.rejected.jsonl").write_text(res["rejected"])
    print(f"kept {len(res['kept'].splitlines())}, rejected {len(res['rejected'].splitlines())}")
