"""Gates for the overnight training loop: is a finished run better than the current best model?

  python -m rescue_tagger.loop compare --candidate /runs/<run>/result.json \
      --champion /runs/champion.json [--runs-dir /runs] [--promote]

Every run is scored on the "select" half of dev (the other half fits calibration), split into real
human text (Munro, HumAID) and synthetic. A run is promoted only if it passes every gate below;
otherwise the reasons are printed. Test sets and Harvey are never used here: picking between runs
by test score would make the test score meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

# Fields the rules engine branches on; a candidate may not lose more than MAX_DROP on any of them.
CRITICAL = ("is_request_accuracy", "needs_f1", "vulnerable_f1", "location_type_accuracy")
MAX_DROP = 0.01
MIN_GAIN = 0.005
MAX_ECE = 0.10        # absolute ceiling on any field's calibration error
MAX_ECE_RISE = 0.02   # and no field may get worse than the current best by more than this
MAX_LATENCY_MS = 100.0
# Metrics that make up the real-text score (only those present in the report count).
REAL_METRICS = ("is_request_accuracy", "needs_f1", "request_kind_accuracy")
SCORED_SUFFIXES = ("_accuracy", "_f1", "location_exact")


def _score(report: dict[str, Any], keys: tuple[str, ...] | None = None) -> float | None:
    vals = [v for k, v in report.items()
            if (k in keys if keys else k.endswith(SCORED_SUFFIXES)) and isinstance(v, float)]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Selection numbers for one run, from its dev reports."""
    ev = result["eval"]
    real, synth, both = ev["dev_real"], ev["dev_synthetic"], ev["dev_rubric"]
    real_score = _score(real, REAL_METRICS)
    shared = tuple(k for k in REAL_METRICS if k in real and k in synth)
    synth_same = _score(synth, shared) if shared else None
    return {
        "run": result["run"],
        "real_score": real_score,
        "synthetic_score": _score(synth),
        # Same metrics on both: if synthetic climbs while real text does not, the model is learning
        # the generator's style.
        "gap": None if synth_same is None or real_score is None
        else round(synth_same - real_score, 4),
        "critical": {k: both.get(k) for k in CRITICAL},
        "ece": {k: v for k, v in both.items() if k.endswith("_ece")},
        "latency_ms": both.get("latency_median_ms"),
        "consistency_changed_pct": both.get("consistency_changed_pct"),
        "signature": signature(result["config"]),
        "reused": bool(result.get("reused_weights_from")),
    }


def signature(cfg: dict[str, Any]) -> str:
    """Config identity ignoring seed and output paths, to group seed repeats."""
    skip = {"seed", "out_dir", "data_dir"}
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in skip}, sort_keys=True)


def seed_spread(candidate: dict[str, Any], runs_dir: Path | None) -> float | None:
    """Standard deviation of real-text score across finished runs that differ only by seed."""
    if runs_dir is None:
        return None
    scores = []
    for p in sorted(runs_dir.glob("*/result.json")):
        try:
            s = summarize(json.loads(p.read_text()))
        except (KeyError, json.JSONDecodeError):
            continue  # runs from before dev reports existed
        if (s["signature"] == candidate["signature"] and s["real_score"] is not None
                and not s["reused"]):
            scores.append(s["real_score"])
    return round(statistics.stdev(scores), 4) if len(scores) >= 2 else None


def gates(cand: dict[str, Any], champ: dict[str, Any] | None,
          spread: float | None = None) -> tuple[bool, list[str]]:
    """(promote?, reasons). Sanity gates always apply; comparison gates need a champion."""
    fails: list[str] = []
    if cand["consistency_changed_pct"] not in (0, 0.0):
        fails.append(f"output changed across repeats ({cand['consistency_changed_pct']}%)")
    if cand["latency_ms"] is None or cand["latency_ms"] >= MAX_LATENCY_MS:
        fails.append(f"latency {cand['latency_ms']} ms is not under {MAX_LATENCY_MS}")
    for k, v in cand["ece"].items():
        if v > MAX_ECE:
            fails.append(f"{k} {v} above {MAX_ECE}")
    if cand["real_score"] is None:
        fails.append("no real-text dev score")
    if champ is None:
        return not fails, fails or ["first run: becomes the baseline"]

    need = max(MIN_GAIN, 2 * spread) if spread else MIN_GAIN
    gain = (cand["real_score"] or 0) - (champ["real_score"] or 0)
    if gain < need:
        fails.append(f"real-text score {cand['real_score']} vs {champ['real_score']}: "
                     f"gain {gain:+.4f} is below {need:.4f}")
    for k, v in cand["ece"].items():
        b = champ.get("ece", {}).get(k)
        if b is not None and v - b > MAX_ECE_RISE:
            fails.append(f"{k} worsened {b} -> {v}")
    for k in CRITICAL:
        c, b = cand["critical"].get(k), champ["critical"].get(k)
        if c is not None and b is not None and b - c > MAX_DROP:
            fails.append(f"{k} dropped {b:.4f} -> {c:.4f}")
    if cand["gap"] is not None and champ.get("gap") is not None and cand["gap"] > champ["gap"]:
        fails.append(f"synthetic-vs-real gap widened {champ['gap']} -> {cand['gap']}")
    return not fails, fails or [f"real-text score up {gain:+.4f}"]


def compare(candidate: Path, champion: Path, runs_dir: Path | None = None,
            promote: bool = False) -> dict[str, Any]:
    cand = summarize(json.loads(candidate.read_text()))
    champ = json.loads(champion.read_text()) if champion.exists() else None
    spread = seed_spread(cand, runs_dir)
    ok, reasons = gates(cand, champ, spread)
    decision = {"candidate": cand, "champion": champ and champ["run"], "seed_spread": spread,
                "promote": ok, "reasons": reasons,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ok and promote:
        champion.write_text(json.dumps({**cand, "promoted_at": decision["time"]}, indent=2)
                            + "\n")
    return decision


def log_eval_read(log: Path, run: str, eval_set: str) -> None:
    """Every scoring of a test-only set is recorded, so we can show how rarely it was opened."""
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "run": run, "set": eval_set}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--champion", type=Path, required=True)
    c.add_argument("--runs-dir", type=Path, default=None)
    c.add_argument("--promote", action="store_true", help="write champion.json if it passes")
    args = ap.parse_args(argv)
    print(json.dumps(compare(args.candidate, args.champion, args.runs_dir, args.promote),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
