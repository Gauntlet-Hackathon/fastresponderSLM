import json

from rescue_tagger.loop import compare, gates, log_eval_read, summarize
from rescue_tagger.records import dev_half


def _result(run, real=0.80, synth=0.95, crit=None, ece=0.03, latency=20.0, changed=0.0, seed=13):
    crit = crit or {}
    rubric = {"is_request_accuracy": crit.get("is_request_accuracy", 0.85),
              "needs_f1": crit.get("needs_f1", 0.7),
              "vulnerable_f1": crit.get("vulnerable_f1", 0.8),
              "location_type_accuracy": crit.get("location_type_accuracy", 0.9),
              "is_request_ece": ece, "latency_median_ms": latency,
              "consistency_changed_pct": changed}
    return {"run": run, "config": {"backbone": "b", "seed": seed, "out_dir": f"/runs/{run}"},
            "eval": {"dev_real": {"is_request_accuracy": real, "needs_f1": real},
                     "dev_synthetic": {"is_request_accuracy": synth, "needs_f1": synth},
                     "dev_rubric": rubric}}


def test_dev_half_is_fixed_and_balanced():
    ids = [f"munro:{i}" for i in range(4000)]
    halves = [dev_half(i) for i in ids]
    assert halves == [dev_half(i) for i in ids]
    assert 0.45 < halves.count("calib") / 4000 < 0.55


def test_first_run_becomes_baseline_if_sane():
    ok, why = gates(summarize(_result("r1")), None)
    assert ok and "baseline" in why[0]


def test_sanity_gates_block_even_without_champion():
    ok, why = gates(summarize(_result("r1", ece=0.15, latency=150.0, changed=2.0)), None)
    assert not ok
    assert any("ece" in w for w in why) and any("latency" in w for w in why)
    assert any("repeats" in w for w in why)


def test_promotes_only_real_text_gains():
    champ = summarize(_result("c", real=0.80, synth=0.95))
    assert gates(summarize(_result("a", real=0.82, synth=0.96)), champ)[0]
    # synthetic up, real flat: no promotion, and the widening gap is named
    ok, why = gates(summarize(_result("b", real=0.80, synth=0.99)), champ)
    assert not ok and any("gap" in w for w in why)


def test_critical_field_drop_blocks_promotion():
    champ = summarize(_result("c", real=0.80))
    cand = summarize(_result("a", real=0.85, synth=0.99,
                             crit={"vulnerable_f1": 0.75}))
    ok, why = gates(cand, champ)
    assert not ok and any("vulnerable_f1" in w for w in why)


def test_seed_spread_raises_the_bar():
    champ = summarize(_result("c", real=0.80))
    cand = summarize(_result("a", real=0.81, synth=0.955))
    assert gates(cand, champ, spread=None)[0]
    assert not gates(cand, champ, spread=0.01)[0]  # needs +0.02 when seeds differ by 0.01


def test_compare_writes_champion_and_measures_seed_spread(tmp_path):
    for i, (real, seed) in enumerate([(0.80, 1), (0.84, 2)]):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_result(f"r{i}", real=real, seed=seed)))
    champ = tmp_path / "champion.json"
    first = compare(tmp_path / "r0" / "result.json", champ, tmp_path, promote=True)
    assert first["promote"] and json.loads(champ.read_text())["run"] == "r0"
    second = compare(tmp_path / "r1" / "result.json", champ, tmp_path, promote=True)
    assert second["seed_spread"] is not None
    assert not second["promote"]  # +0.04 does not beat 2x the 0.028 spread between seeds


def test_eval_reads_are_logged(tmp_path):
    log = tmp_path / "reads.jsonl"
    log_eval_read(log, "r1", "harvey")
    log_eval_read(log, "r1", "test_real")
    assert [json.loads(x)["set"] for x in log.read_text().splitlines()] == ["harvey", "test_real"]


def test_calibration_may_not_get_worse_than_champion():
    champ = summarize(_result("c", real=0.80, ece=0.03))
    ok, why = gates(summarize(_result("a", real=0.85, synth=0.99, ece=0.07)), champ)
    assert not ok and any("worsened" in w for w in why)
