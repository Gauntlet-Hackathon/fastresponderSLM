"""Generate synthetic training data with open-weights models on a Modal GPU (no Claude, no OpenAI).

  modal run modal_jobs/synth.py --name b0005-focus-gemma --n 30 --seed 300000 --mode focus

A vLLM server (OpenAI-compatible) holds one model at a time. Phase 1: the generator writes messages
for label-first specs (rescue_tagger.data.synthetic). Phase 2: a model from another family labels
each message blind; rows where it disagrees with the spec are rejected. The batch lands in the
`rescue-tagger-runs` volume under synth/ and is returned to the local entrypoint, which writes it
for the GitHub workflow to commit. Launched by .github/workflows/synth.yml.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/repo"
GEN_MODEL = "google/gemma-4-31B-it"         # Apache-2.0 (Paawan chose Gemma over Qwen)
VERIFY_MODEL = "microsoft/phi-4"            # MIT, a different family from the generator

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Gemma 4 needs vLLM >= 0.27 (gemma4 model code) and transformers 5; vLLM pins the
    # transformers range it was built against.
    .pip_install("vllm==0.31.0", "openai>=2.25,<4", "jsonschema>=4.23", "referencing>=0.35",
                 "pyyaml>=6.0")
    # FlashInfer's sampler is compiled on first use and needs nvcc, which this slim image lacks;
    # vLLM's own top-k/top-p sampler needs no compiler.
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src", "HF_HUB_ENABLE_HF_TRANSFER": "0",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(REPO_ROOT, REMOTE_ROOT, ignore=[".venv", ".git", "outputs", "data/raw",
                                                     "data/processed", "**/__pycache__"])
)
runs = modal.Volume.from_name("rescue-tagger-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("rescue-tagger-hf-cache", create_if_missing=True)
app = modal.App("rescue-synth", image=image)
PORT = 8000
MODES = {"focus": ["--focus"], "vague": ["--vague"], "hoax": ["--hoax"], "mix": ["--mix"],
         "plain": []}


def stop(proc: subprocess.Popen) -> None:
    """Stop vLLM and its engine child process (a plain kill orphans the child, which keeps the
    GPU memory), then wait until the GPU is free for the next model."""
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(60):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if used.returncode == 0 and all(int(x) < 2000 for x in used.stdout.split()):
            return
        time.sleep(5)
    try:  # the server exited but its engine child hung: kill what is left of the group
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise RuntimeError("GPU memory was not released after stopping vLLM")


def serve(model: str) -> subprocess.Popen:
    args = ["--model", model, "--port", str(PORT), "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.92"]
    if "gemma-4" in model.lower():
        # Gemma 4 checkpoints carry a vision tower; text only frees its memory for
        # the KV cache. In bf16 the 31B weights took 58 GiB and left room for ~1.5 requests of
        # 8k tokens; FP8 weights (one scale per output channel, activations scaled per token;
        # H100-native) halve that so many requests run at once.
        # Thinking stays off (the chat template's default), so the constrained JSON answer
        # starts right away.
        args += ["--language-model-only", "--quantization", "fp8_per_channel"]
    proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args],
                            start_new_session=True)
    import urllib.request

    for _ in range(180):  # up to 30 min for download + load
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode} while loading {model}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            return proc
        except OSError:
            time.sleep(10)
    stop(proc)
    raise RuntimeError(f"vLLM did not come up for {model}")


@app.function(gpu="H100", timeout=2 * 3600,
              volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})])
def generate(name: str, n: int, seed: int, mode: str, gen_model: str,
             verify_model: str, reverify: str = "") -> dict[str, str]:
    os.chdir(REMOTE_ROOT)
    os.environ.update({"OPENAI_BASE_URL": f"http://localhost:{PORT}/v1",
                       "OPENAI_SLM_API_KEY": "local"})
    out = Path(f"/runs/synth/{name}")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"

    def run(*args: str) -> None:
        print("+", " ".join(args), flush=True)
        subprocess.run([sys.executable, "-m", "rescue_tagger.data.synthetic", *args], check=True)

    if reverify:
        # Re-check an earlier batch's unverified generations with the current verifier rules
        # (no generation, so only the verifier's GPU time is spent).
        if reverify == name:
            raise SystemExit("reverify needs a new batch name, not the source batch's")
        src = Path(f"/runs/synth/{reverify}")
        if not (src / "raw.jsonl").exists():
            raise SystemExit(f"no raw generations for {reverify}")
        shutil.copy(src / "raw.jsonl", raw)
        if (src / "raw.rejected.jsonl").exists():
            shutil.copy(src / "raw.rejected.jsonl", out / "raw.rejected.jsonl")
    else:
        proc = serve(gen_model)
        try:
            run("--n", str(n), "--seed", str(seed), "--out", str(raw), "--gen-model", gen_model,
                "--verify-model", "", "--workers", "32", *MODES[mode])
        finally:
            stop(proc)
        hf_cache.commit()
    proc = serve(verify_model)
    try:
        run("--verify-file", str(raw), "--verify-model", verify_model, "--out",
            str(out / f"{name}.jsonl"), "--workers", "32")
    finally:
        stop(proc)
    hf_cache.commit()
    rejected = "".join(p.read_text() for p in (out / "raw.rejected.jsonl",
                                                out / f"{name}.rejected.jsonl") if p.exists())
    (out / f"{name}.all-rejected.jsonl").write_text(rejected)
    runs.commit()
    return {"kept": (out / f"{name}.jsonl").read_text(), "rejected": rejected}


@app.local_entrypoint()
def main(name: str, n: int = 30, seed: int = 300000, mode: str = "focus",
         gen_model: str = GEN_MODEL, verify_model: str = VERIFY_MODEL, reverify: str = "") -> None:
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {sorted(MODES)}")
    res = generate.remote(name, n, seed, mode, gen_model, verify_model, reverify)
    Path(f"{name}.jsonl").write_text(res["kept"])
    Path(f"{name}.rejected.jsonl").write_text(res["rejected"])
    print(f"kept {len(res['kept'].splitlines())}, rejected {len(res['rejected'].splitlines())}")
