"""Overnight loop driver, run by .github/workflows/loop.yml every 30 minutes.

  python -m rescue_tagger.scheduler tick  [--dry-run]
  python -m rescue_tagger.scheduler final [--dry-run]

`tick` reads the experiment queue (configs/loop_queue.yaml), what has finished (decision.json files
on the Modal volume), what is running (GitHub Actions), and launches the next queued run if no
stop rule has hit. `final` scores the current best model on the test sets once (morning check).

Stop rules (the loop then waits for a person):
  - max_runs trained runs reached (re-scoring an existing model does not count)
  - `max_rejections` rejected runs in a row (seed repeats don't count: they're expected to tie)
  - the synthetic-vs-real gap widened on 2 runs in a row
  - the same queue item failed twice
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

QUEUE = Path("configs/loop_queue.yaml")
VOLUME = "rescue-tagger-runs"
WORKFLOW = "train.yml"


@dataclass
class State:
    decisions: dict[str, dict[str, Any]]  # run_name -> decision.json
    active: list[str]                      # run names queued or running on GitHub
    failures: dict[str, int]               # run name -> failed workflow attempts
    run_dirs: list[str] = field(default_factory=list)  # top-level dirs on the volume


def run_name(night: str, item: dict[str, Any]) -> str:
    return f"loop-{night}-{item['name']}"


def resolve_reuse(spec: str, run_dirs: list[str]) -> str | None:
    """`latest:<prefix>` -> newest run dir with that prefix (names end in a timestamp)."""
    if not spec.startswith("latest:"):
        return spec
    prefix = spec.split(":", 1)[1] + "-"
    matches = sorted(d for d in run_dirs if d.startswith(prefix))
    return matches[-1] if matches else None


def next_action(queue: dict[str, Any], st: State) -> dict[str, Any]:
    """Pure decision: {"action": "launch", ...} | {"action": "wait"|"stop", "why": ...}."""
    night, items = queue["night"], queue["items"]
    max_parallel = queue.get("max_parallel", 1)
    trained = rejections = gap_streak = 0
    pending = []
    for item in items:
        name = run_name(night, item)
        d = st.decisions.get(name)
        if st.failures.get(name, 0) >= 2:
            return {"action": "stop", "why": f"{name} failed twice"}
        if d is None:
            if name not in st.active:
                pending.append(item)
            continue
        if not item.get("reuse"):
            trained += 1
        if d["promote"]:
            rejections = gap_streak = 0
            continue
        if item.get("purpose") != "seed":
            rejections += 1
        gap_streak = gap_streak + 1 if any("gap widened" in r for r in d["reasons"]) else 0
        if gap_streak >= 2:
            return {"action": "stop", "why": "synthetic-vs-real gap widened on 2 runs in a row"}
        if rejections >= queue.get("max_rejections", 3):
            return {"action": "stop", "why": f"{rejections} rejected runs in a row"}
    active_trains = sum(1 for a in st.active if a.startswith(f"loop-{night}-"))
    if trained + active_trains >= queue.get("max_runs", 8) and pending:
        return {"action": "stop", "why": f"run cap of {queue.get('max_runs', 8)} reached"}
    if not pending:
        return {"action": "stop" if not st.active else "wait",
                "why": "queue finished" if not st.active else "last runs still going"}
    if len(st.active) >= max_parallel:
        return {"action": "wait", "why": f"{len(st.active)} run(s) in progress"}
    item = pending[0]
    inputs = {"config": item["config"], "publish": "false", "final": "false",
              "overrides": " ".join(f"{k}={v}" for k, v in item.get("overrides", {}).items()),
              "run_name": run_name(night, item), "reuse": ""}
    if item.get("reuse"):
        src = resolve_reuse(item["reuse"], st.run_dirs)
        if src is None:
            return {"action": "skip", "item": item["name"],
                    "why": f"no run matches {item['reuse']}"}
        inputs["reuse"] = src
    return {"action": "launch", "inputs": inputs}


# ---- IO: GitHub Actions and the Modal volume -------------------------------------------------

def _gh(method: str, path: str, body: dict | None = None) -> Any:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}{path}",
        method=method, data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def _github_runs() -> tuple[list[str], dict[str, int]]:
    runs = _gh("GET", f"/actions/workflows/{WORKFLOW}/runs?per_page=100")["workflow_runs"]
    active, failures = [], {}
    for r in runs:
        title = r.get("display_title", "")
        if not title.startswith("train "):
            continue
        name = title.split(" ", 1)[1]
        if r["status"] in ("queued", "in_progress", "waiting", "requested", "pending"):
            active.append(name)
        elif r.get("conclusion") in ("failure", "timed_out", "cancelled"):
            failures[name] = failures.get(name, 0) + 1
    return active, failures


def _volume_state() -> tuple[dict[str, dict[str, Any]], list[str]]:
    import modal

    vol = modal.Volume.from_name(VOLUME)
    dirs = [e.path.strip("/") for e in vol.listdir("/")]
    decisions = {}
    for d in dirs:
        try:
            decisions[d] = json.loads(b"".join(vol.read_file(f"{d}/decision.json")))
        except Exception:  # noqa: BLE001 - no decision yet (running, failed, or old pipeline)
            continue
    return decisions, dirs


def _dispatch(inputs: dict[str, str], ref: str) -> None:
    _gh("POST", f"/actions/workflows/{WORKFLOW}/dispatches", {"ref": ref, "inputs": inputs})


def tick(dry_run: bool = False) -> dict[str, Any]:
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    decisions, dirs = _volume_state()
    active, failures = _github_runs()
    st = State(decisions, active, failures, dirs)
    skipped: list[str] = []
    while True:
        act = next_action(queue, st)
        if act["action"] != "skip":
            break
        skipped.append(f"{act['item']}: {act['why']}")
        queue["items"] = [i for i in queue["items"] if i["name"] != act["item"]]
    act["skipped"] = skipped
    if act["action"] == "launch" and not dry_run:
        _dispatch(act["inputs"], os.environ.get("GITHUB_REF_NAME", "main"))
    return act


def final(dry_run: bool = False) -> dict[str, Any]:
    import modal

    vol = modal.Volume.from_name(VOLUME)
    try:
        champ = json.loads(b"".join(vol.read_file("champion.json")))
    except Exception:  # noqa: BLE001
        return {"action": "stop", "why": "no best model yet"}
    day = time.strftime("%Y-%m-%d", time.gmtime())
    name = f"final-{day}"
    active, _ = _github_runs()
    dirs = [e.path.strip("/") for e in vol.listdir("/")]
    if name in dirs or name in active:
        return {"action": "wait", "why": f"{name} already scored"}
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    cfg = next((i["config"] for i in queue["items"]
                if run_name(queue["night"], i) == champ["run"]), "configs/train_laya.yaml")
    inputs = {"config": cfg, "publish": "false", "final": "true", "overrides": "",
              "run_name": name, "reuse": champ["run"]}
    if not dry_run:
        _dispatch(inputs, os.environ.get("GITHUB_REF_NAME", "main"))
    return {"action": "launch", "inputs": inputs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["tick", "final"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = tick(args.dry_run) if args.cmd == "tick" else final(args.dry_run)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
