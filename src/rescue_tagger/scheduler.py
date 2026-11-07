"""Overnight loop driver, run by .github/workflows/loop.yml when a train run finishes and every
30 minutes.

  python -m rescue_tagger.scheduler tick  [--dry-run]
  python -m rescue_tagger.scheduler final [--dry-run]

`tick` reads the experiment queue (configs/loop_queue.yaml), what has finished (decision.json files
on the Modal volume), what is running (GitHub Actions), and launches the next queued run if no
stop rule has hit. `final` scores the current best model on the test sets once (morning check).

Stop rules (the loop then waits for a person):
  - max_runs trained runs reached (re-scoring an existing model does not count)
  - `max_rejections` rejected runs in a row (seed repeats and baselines don't count)
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
    run_dirs: list[str] = field(default_factory=list)  # finished run dirs on the volume
    succeeded: set[str] = field(default_factory=set)   # run names whose workflow succeeded


def run_name(night: str, item: dict[str, Any]) -> str:
    return f"loop-{night}-{item['name']}"


def resolve_reuse(spec: str, run_dirs: list[str]) -> str | None:
    """`latest:<prefix>` -> newest run dir with that prefix (names end in a timestamp)."""
    if not spec.startswith("latest:"):
        return spec
    prefix = spec.split(":", 1)[1] + "-"
    matches = sorted(d for d in run_dirs if d.startswith(prefix))
    return matches[-1] if matches else None


def in_quiet_window(queue: dict[str, Any], hhmm: str) -> bool:
    """No new launches from `launch_cutoff_utc` until `resume_utc`, so the morning test-set check
    finds nothing running and scores a best model that can no longer change under it."""
    start, end = queue.get("launch_cutoff_utc"), queue.get("resume_utc")
    return bool(start and end and start <= hhmm < end)


def next_action(queue: dict[str, Any], st: State, hhmm: str | None = None) -> dict[str, Any]:
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
            if name in st.succeeded:
                return {"action": "stop", "why": f"{name} finished without a decision; check it"}
            if name not in st.active:
                pending.append(item)
            continue
        if not item.get("reuse"):
            trained += 1
        if d["promote"]:
            rejections = gap_streak = 0
            continue
        if item.get("purpose") in ("seed", "baseline"):
            # Seed repeats are expected to tie, and a baseline (another backbone or data version) is
            # a comparison, not a tuning step; neither counts toward the rejection or gap streaks.
            continue
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
    if hhmm and in_quiet_window(queue, hhmm):
        return {"action": "wait", "why": f"morning test-set window ({hhmm} UTC)"}
    item = pending[0]
    # Seed repeats are never promoted and only measure noise at a fixed setting, so up to
    # `seed_parallel` of them may run beside the main run. Items still launch in queue order; a
    # seed may use an idle main slot too, and runs not in the queue count as main.
    seed_parallel = queue.get("seed_parallel", 0)
    seed_names = ({run_name(night, i) for i in items if i.get("purpose") == "seed"}
                  if seed_parallel else set())
    active_seeds = sum(1 for a in st.active if a in seed_names)
    active_main = len(st.active) - min(active_seeds, seed_parallel)  # extra seeds hold main slots
    if item.get("purpose") == "seed" and seed_parallel:
        # A seed may take its own slot or an idle main slot.
        if active_seeds >= seed_parallel and active_main >= max_parallel:
            return {"action": "wait", "why": f"{len(st.active)} run(s) in progress"}
    elif active_main >= max_parallel:
        return {"action": "wait", "why": f"{active_main} run(s) in progress"}
    overrides = dict(item.get("overrides", {}))
    if item.get("purpose"):
        overrides["purpose"] = item["purpose"]
    inputs = {"config": item["config"], "publish": "false", "final": "false",
              "overrides": _overrides(overrides),
              "run_name": run_name(night, item), "reuse": ""}
    if item.get("reuse"):
        src = resolve_reuse(item["reuse"], st.run_dirs)
        if src is None:
            return {"action": "skip", "item": item["name"],
                    "why": f"no run matches {item['reuse']}"}
        inputs["reuse"] = src
    return {"action": "launch", "inputs": inputs}


def _overrides(d: dict[str, Any]) -> str:
    return " ".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in d.items())


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


def _github_runs() -> tuple[list[str], dict[str, int], set[str]]:
    runs = _gh("GET", f"/actions/workflows/{WORKFLOW}/runs?per_page=100")["workflow_runs"]
    active, failures, succeeded = [], {}, set()
    for r in runs:
        title = r.get("display_title", "")
        if title == "train":  # launched before runs carried names; still counts while running
            name = f"unnamed-{r['id']}"
        elif title.startswith("train "):
            name = title.split(" ", 1)[1]
        else:
            continue
        if r["status"] in ("queued", "in_progress", "waiting", "requested", "pending"):
            active.append(name)
        elif r.get("conclusion") in ("failure", "timed_out", "cancelled"):
            failures[name] = failures.get(name, 0) + 1
        elif r.get("conclusion") == "success":
            succeeded.add(name)
    return active, failures, succeeded


def _dirs(vol: Any) -> list[str]:
    """Run directories only (champion.json and eval_reads.jsonl live at the root too)."""
    from modal.volume import FileEntryType

    return [e.path.strip("/") for e in vol.listdir("/") if e.type == FileEntryType.DIRECTORY]


def _volume_state() -> tuple[dict[str, dict[str, Any]], list[str]]:
    import modal

    vol = modal.Volume.from_name(VOLUME)
    decisions = {}
    finished = []
    for d in _dirs(vol):
        names = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(d)}
        # Only finished runs (result.json written at the very end) can be reused; a directory
        # without it is still training, and its best/ is a mid-run checkpoint.
        if "result.json" in names:
            finished.append(d)
        if "decision.json" in names:
            decisions[d] = json.loads(b"".join(vol.read_file(f"{d}/decision.json")))
    return decisions, finished


def _dispatch(inputs: dict[str, str], ref: str) -> None:
    _gh("POST", f"/actions/workflows/{WORKFLOW}/dispatches", {"ref": ref, "inputs": inputs})


def wait_for_run(run_id: str, timeout_s: int = 300, poll_s: int = 10) -> bool:
    """A train run hands off to the loop from its own last step, so it is still in progress when
    this tick starts; wait until GitHub reports it completed so it isn't counted as active."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _gh("GET", f"/actions/runs/{run_id}")["status"] == "completed":
            return True
        time.sleep(poll_s)
    return False


def tick(dry_run: bool = False) -> dict[str, Any]:
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    after = os.environ.get("AFTER_RUN", "").strip()
    if after and not wait_for_run(after):
        return {"action": "wait", "why": f"run {after} still not completed", "skipped": []}
    # GitHub first: a run seen as succeeded has already committed its decision before the
    # volume is read, so it can't look finished-without-a-decision by a timing accident.
    active, failures, succeeded = _github_runs()
    decisions, dirs = _volume_state()
    # A run that succeeded before decisions existed (old pipeline) has none; only loop runs count.
    st = State(decisions, active, failures, dirs, {n for n in succeeded if n.startswith("loop-")})
    skipped: list[str] = []
    while True:
        act = next_action(queue, st, time.strftime("%H:%M", time.gmtime()))
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
    except FileNotFoundError:
        return {"action": "stop", "why": "no best model yet"}
    day = time.strftime("%Y-%m-%d", time.gmtime())
    name = f"final-{day}"
    active, _, _ = _github_runs()
    if name in active:
        return {"action": "wait", "why": f"{name} is running"}
    if (name in _dirs(vol)
            and "result.json" in {e.path.rsplit("/", 1)[-1] for e in vol.listdir(name)}):
        return {"action": "wait", "why": f"{name} already scored"}
    if active:
        # Score the best model only once nothing can replace it; retried until 14:37 UTC.
        return {"action": "wait", "why": f"{len(active)} run(s) still going"}
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    item = next((i for i in queue["items"] if run_name(queue["night"], i) == champ["run"]),
                {"config": "configs/train_laya.yaml"})
    inputs = {"config": item["config"], "publish": "false", "final": "true",
              "overrides": _overrides(item.get("overrides", {})),
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
