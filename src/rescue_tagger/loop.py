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
import math
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
# Small dev slices are noisy: one flipped row out of 72 moves accuracy 1.4 points. A drop or rise
# counts only past two standard errors of the champion's own estimate on that many rows.
# Only small slices get the allowance: real-text fields (thousands of rows) keep the fixed limits.
SMALL_N = 300
ECE_NOISE = 0.5       # ECE standard error is roughly ECE_NOISE / sqrt(n) for 10 bins
MAX_ECE_TOL = 0.05    # the ECE-rise allowance never exceeds this
MAX_DROP_TOL = 0.08   # nor the critical-drop allowance
MAX_LATENCY_MS = 100.0
# Data versions whose real-text dev rows and labels are identical (v3 only added a synthetic batch),
# so a run on the newer version must not lose real-text score to the older champion.
REAL_DEV_SAME = {(2, 3), (3, 4), (2, 4), (4, 5), (3, 5), (2, 5)}  # v4: product dev rows are kept
# out of dev_real; v5 adds synthetic rows only
GAP_TOLERANCE = 0.01
# One bad seed can inflate the measured spread; cap the bar so the loop does not reject everything.
MAX_NEED = 0.02
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
        # Real-text dev only: identical across data versions in REAL_DEV_SAME, so these can be
        # checked against the previous version's champion at a hand-off.
        "critical_real": {k: real.get(k) for k in CRITICAL},
        "n_real": {k: field_n(real, k) for k in CRITICAL},
        "n": {k: field_n(both, k) for k in (*CRITICAL, *(k for k in both if k.endswith("_ece")))},
        "ece": {k: v for k, v in both.items() if k.endswith("_ece")},
        "latency_ms": both.get("latency_median_ms"),
        "cpu": both.get("cpu"),
        "consistency_changed_pct": both.get("consistency_changed_pct"),
        "signature": signature(result["config"]),
        "reused": bool(result.get("reused_weights_from")),
        "backbone": result["config"].get("backbone"),
        "data_version": result["config"].get("data_version"),
    }


# Which row count a metric was measured on. ECE keys name the field, not the head.
_N_KEY = {"location": "location_type", "location_type": "location_type"}


def field_n(report: dict[str, Any], metric: str) -> int | None:
    """Rows behind `metric` in an evaluate() report: its head's *_n, else the location rows (the
    synthetic slice every span/vulnerable/hazard head is labeled on, for reports from before
    multi-label heads reported n), else all rows."""
    base = metric.rsplit("_", 1)[0]
    for key in (f"{_N_KEY.get(base, base)}_n", "location_type_n", "n_rows"):
        if isinstance(report.get(key), int) and report[key] > 0:
            return report[key]
    return None


def noise(p: float, n: int | None) -> float:
    """Two standard errors of a proportion-like metric p measured on n rows. p is smoothed
    (Laplace) so a perfect 1.0 or 0.0 on a small slice still allows for one flipped row."""
    if not n:
        return 0.0
    p = (min(max(p, 0.0), 1.0) * n + 1) / (n + 2)
    return 2 * math.sqrt(p * (1 - p) / n)


def same_real_dev(a: Any, b: Any) -> bool:
    """Same real-text dev rows and labels, so real-text scores (and their seed noise) compare."""
    return a == b or (a, b) in REAL_DEV_SAME or (b, a) in REAL_DEV_SAME


def signature(cfg: dict[str, Any]) -> str:
    """Config identity ignoring seed and output paths, to group seed repeats."""
    skip = {"seed", "out_dir", "data_dir", "purpose"}
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in skip}, sort_keys=True)


def seed_spread(candidate: dict[str, Any], runs_dir: Path | None) -> float | None:
    """Run-to-run noise for the candidate's backbone: the largest standard deviation of real-text
    score within any group of finished runs that differ only by seed (the seed repeats of the
    baseline). It applies to every experiment on that backbone, not only to seed repeats."""
    if runs_dir is None:
        return None
    groups: dict[str, list[float]] = {}
    for p in sorted(runs_dir.glob("*/result.json")):
        try:
            s = summarize(json.loads(p.read_text()))
        except (KeyError, json.JSONDecodeError):
            continue  # runs from before dev reports existed
        if (s["backbone"] == candidate.get("backbone") and s["real_score"] is not None
                and same_real_dev(s["data_version"], candidate.get("data_version"))
                and not s["reused"]):
            groups.setdefault(s["signature"], []).append(s["real_score"])
    spreads = [statistics.stdev(v) for v in groups.values() if len(v) >= 2]
    return round(max(spreads), 4) if spreads else None


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

    need = min(max(MIN_GAIN, 2 * spread), MAX_NEED) if spread else MIN_GAIN
    gain = (cand["real_score"] or 0) - (champ["real_score"] or 0)
    if gain < need:
        fails.append(f"real-text score {cand['real_score']} vs {champ['real_score']}: "
                     f"gain {gain:+.4f} is below {need:.4f}")
    def rows(k: str) -> int | None:
        """The larger of the two runs' row counts (the stricter allowance), and only for small
        slices; None means "use the fixed limit"."""
        vals = [x for x in ((champ.get("n") or {}).get(k), (cand.get("n") or {}).get(k)) if x]
        n = max(vals) if vals else None
        return n if n and n < SMALL_N else None

    for k, v in cand["ece"].items():
        b = champ.get("ece", {}).get(k)
        n = rows(k)
        tol = min(max(MAX_ECE_RISE, 2 * ECE_NOISE / math.sqrt(n)), MAX_ECE_TOL) if n \
            else MAX_ECE_RISE
        if b is not None and v - b > tol:
            fails.append(f"{k} worsened {b} -> {v} (past noise {tol:.3f})")
    for k in CRITICAL:
        c, b = cand["critical"].get(k), champ["critical"].get(k)
        if c is None or b is None:
            continue
        # F1 is treated like a proportion over rows, which errs strict. Both runs score the same
        # items (a paired comparison), so the champion's sampling error is not the right
        # allowance on large slices; those keep the fixed limit.
        tol = min(max(MAX_DROP, noise(b, rows(k))), MAX_DROP_TOL)
        if b - c > tol:
            fails.append(f"{k} dropped {b:.4f} -> {c:.4f} (past noise {tol:.4f})")
    # The gap counts as widened only past noise, and only when real text did not improve.
    gap_tol = max(GAP_TOLERANCE, 2 * spread) if spread else GAP_TOLERANCE
    if (cand["gap"] is not None and champ.get("gap") is not None
            and cand["gap"] - champ["gap"] > gap_tol and gain <= 0):
        fails.append(f"synthetic-vs-real gap widened {champ['gap']} -> {cand['gap']}")
    return not fails, fails or [f"real-text score up {gain:+.4f}"]


def compare(candidate: Path, champion: Path, runs_dir: Path | None = None,
            promote: bool = False) -> dict[str, Any]:
    result = json.loads(candidate.read_text())
    cand = summarize(result)
    champ = json.loads(champion.read_text()) if champion.exists() else None
    floor_fails: list[str] = []
    held: list[str] = []
    previous = None
    if champ is not None and champ.get("data_version") != cand["data_version"]:
        # Scores on different dev labels aren't comparable; the first run on the new labels
        # becomes the baseline (the old champion stays on record in its decision). When only the
        # synthetic rows changed, real text still is: the new baseline must hold the old score
        # and the old real-text critical fields.
        previous = champ["run"]
        if (champ.get("data_version"), cand["data_version"]) in REAL_DEV_SAME:
            old_spread = seed_spread(champ, runs_dir)
            tol = min(max(MAX_DROP, 2 * old_spread), MAX_NEED) if old_spread else MAX_DROP
            c, b = cand["real_score"], champ.get("real_score")
            if c is not None and b is not None:
                msg = f"real-text score {c} vs {b} on the previous data version"
                if b - c > tol:
                    floor_fails.append(f"{msg}: drop {b - c:.4f} is past {tol:.4f}")
                else:
                    held.append(f"held {msg} (change {c - b:+.4f}, allowed drop {tol:.4f})")
            if not champ.get("critical_real") and runs_dir is not None:
                # Champions promoted before critical_real existed: rebuild from their result.
                src = runs_dir / champ["run"] / "result.json"
                try:
                    rebuilt = summarize(json.loads(src.read_text()))
                    champ = {**champ, "critical_real": rebuilt["critical_real"],
                             "n_real": rebuilt["n_real"]}
                except (OSError, KeyError, json.JSONDecodeError):
                    pass  # falls through to "not checked"
            old_crit = champ.get("critical_real") or {}
            if not old_crit:
                held.append("previous champion has no real-text critical fields recorded; "
                            "not checked")
            for k in CRITICAL:
                c, b = cand.get("critical_real", {}).get(k), old_crit.get(k)
                if c is None or b is None:
                    continue
                n = max(x for x in ((champ.get("n_real") or {}).get(k),
                                    (cand.get("n_real") or {}).get(k), 0) if x is not None)
                ktol = min(max(MAX_DROP, noise(b, n if 0 < n < SMALL_N else None)),
                           MAX_DROP_TOL)
                if b - c > ktol:
                    floor_fails.append(f"real-text {k} dropped {b:.4f} -> {c:.4f} on the "
                                       f"previous data version (past noise {ktol:.4f})")
        champ = None
    spread = seed_spread(cand, runs_dir)
    ok, reasons = gates(cand, champ, spread)
    if floor_fails:
        ok, reasons = False, [r for r in reasons if "baseline" not in r] + floor_fails
    elif ok:
        reasons = reasons + held
    if result["config"].get("purpose") == "seed" and ok:
        # A seed repeat measures run-to-run noise; promoting the luckiest seed would bake that
        # noise into the bar every later run has to beat.
        ok, reasons = False, ["seed repeat: measures noise, never promoted"]
    decision = {"candidate": cand, "champion": champ and champ["run"],
                "previous_version_champion": previous, "seed_spread": spread,
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
