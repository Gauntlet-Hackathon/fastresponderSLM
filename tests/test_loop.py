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
    for i, (real, seed) in enumerate([(0.80, 1), (0.812, 2)]):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_result(f"r{i}", real=real, seed=seed)))
    champ = tmp_path / "champion.json"
    first = compare(tmp_path / "r0" / "result.json", champ, tmp_path, promote=True)
    assert first["promote"] and json.loads(champ.read_text())["run"] == "r0"
    second = compare(tmp_path / "r1" / "result.json", champ, tmp_path, promote=True)
    assert second["seed_spread"] is not None
    assert not second["promote"]  # +0.012 does not beat 2x the 0.0085 spread between seeds


def test_eval_reads_are_logged(tmp_path):
    log = tmp_path / "reads.jsonl"
    log_eval_read(log, "r1", "harvey")
    log_eval_read(log, "r1", "test_real")
    assert [json.loads(x)["set"] for x in log.read_text().splitlines()] == ["harvey", "test_real"]


def test_calibration_may_not_get_worse_than_champion():
    champ = summarize(_result("c", real=0.80, ece=0.03))
    ok, why = gates(summarize(_result("a", real=0.85, synth=0.99, ece=0.07)), champ)
    assert not ok and any("worsened" in w for w in why)


def test_small_gap_noise_does_not_count_as_widening():
    champ = summarize(_result("c", real=0.80, synth=0.95))
    ok, why = gates(summarize(_result("a", real=0.80, synth=0.955)), champ)
    assert not ok and not any("gap" in w for w in why)


def test_seed_repeats_are_never_promoted(tmp_path):
    r = _result("s", real=0.95, synth=0.96)
    r["config"]["purpose"] = "seed"
    (tmp_path / "r.json").write_text(json.dumps(r))
    champ = tmp_path / "champion.json"
    champ.write_text(json.dumps(summarize(_result("c", real=0.80))))
    d = compare(tmp_path / "r.json", champ, None, promote=True)
    assert not d["promote"] and json.loads(champ.read_text())["run"] == "c"


def test_baseline_seed_noise_applies_to_other_experiments(tmp_path):
    for i, (real, seed) in enumerate([(0.80, 1), (0.812, 2)]):
        d = tmp_path / f"s{i}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_result(f"s{i}", real=real, seed=seed)))
    exp = _result("lr", real=0.815, synth=0.96)
    exp["config"]["lr_encoder"] = 2e-5  # different config, same backbone
    (tmp_path / "lr.json").write_text(json.dumps(exp))
    champ = tmp_path / "champion.json"
    champ.write_text(json.dumps(summarize(_result("s0", real=0.80))))
    d = compare(tmp_path / "lr.json", champ, tmp_path)
    assert d["seed_spread"] is not None
    assert not d["promote"]  # +0.015 clears MIN_GAIN but not 2x the 0.0085 spread


def test_gain_bar_is_capped_when_seed_noise_is_large():
    champ = summarize(_result("c", real=0.80))
    cand = summarize(_result("a", real=0.825, synth=0.97))
    assert gates(cand, champ, spread=0.05)[0]  # bar capped at 0.02, not 0.10


def test_runs_on_other_label_versions_are_not_compared(tmp_path):
    old = _result("old", real=0.95)
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "result.json").write_text(json.dumps(old))
    champ = tmp_path / "champion.json"
    champ.write_text(json.dumps(summarize(old)))
    new = _result("new", real=0.80, seed=7)
    new["config"]["data_version"] = 2
    (tmp_path / "new.json").write_text(json.dumps(new))
    d = compare(tmp_path / "new.json", champ, tmp_path, promote=True)
    assert d["promote"] and d["champion"] is None and d["seed_spread"] is None
    assert json.loads(champ.read_text())["run"] == "new"


def _with_n(res, **ns):
    res["eval"]["dev_rubric"].update({f"{k}_n": v for k, v in ns.items()})
    return res


def test_one_flipped_row_out_of_72_does_not_reject():
    """dev_rubric's vulnerable/location heads come from ~72 synthetic rows: 1.4 points is noise."""
    champ = summarize(_with_n(_result("c", real=0.80), location_type=72, vulnerable=72,
                              is_request=3000))
    cand = summarize(_with_n(_result("a", real=0.83, crit={"location_type_accuracy": 0.886}),
                             location_type=72, vulnerable=72, is_request=3000))
    ok, why = gates(cand, champ)
    assert ok, why


def test_a_real_drop_on_72_rows_still_rejects():
    champ = summarize(_with_n(_result("c", real=0.80), location_type=72))
    cand = summarize(_with_n(_result("a", real=0.83, crit={"location_type_accuracy": 0.75}),
                             location_type=72))
    ok, why = gates(cand, champ)
    assert not ok and any("location_type_accuracy" in w for w in why)


def test_large_slices_keep_the_strict_drop_limit():
    champ = summarize(_with_n(_result("c", real=0.80), is_request=20000))
    cand = summarize(_with_n(_result("a", real=0.83, crit={"is_request_accuracy": 0.835}),
                             is_request=20000))
    ok, why = gates(cand, champ)
    assert not ok and any("is_request_accuracy" in w for w in why)


def test_ece_rise_on_72_rows_is_noise_but_large_rise_is_not():
    champ = summarize(_with_n(_result("c", real=0.80, ece=0.03), is_request=72))
    assert gates(summarize(_with_n(_result("a", real=0.83, ece=0.07), is_request=72)), champ)[0]
    # the allowance is capped at 0.05: a rise of 0.069 on 72 rows is real
    assert not gates(summarize(_with_n(_result("b", real=0.83, ece=0.099), is_request=72)),
                     champ)[0]
    big = summarize(_with_n(_result("c2", real=0.80, ece=0.03), is_request=5000))
    assert not gates(summarize(_with_n(_result("d", real=0.83, ece=0.06), is_request=5000)),
                     big)[0]


def test_a_perfect_score_on_72_rows_still_allows_one_flipped_row():
    champ = summarize(_with_n(_result("c", real=0.80, crit={"location_type_accuracy": 1.0}),
                              location_type=72))
    cand = summarize(_with_n(_result("a", real=0.83, crit={"location_type_accuracy": 71 / 72}),
                             location_type=72))
    ok, why = gates(cand, champ)
    assert ok, why


def test_real_text_fields_keep_the_fixed_limit():
    champ = summarize(_with_n(_result("c", real=0.80, crit={"needs_f1": 0.65}), needs=1233,
                              is_request=1233))
    cand = summarize(_with_n(_result("a", real=0.83, crit={"needs_f1": 0.635}), needs=1233,
                             is_request=1233))
    ok, why = gates(cand, champ)
    assert not ok and any("needs_f1" in w for w in why)


def _with(res, **rubric):
    res["eval"]["dev_rubric"].update(rubric)
    return res


def test_f1_noise_uses_gold_positive_items_not_rows():
    """needs_f1 on 1,202 rows but 100 gold needs: a 2-point dip is noise, a 10-point one is not."""
    champ = summarize(_with(_result("c", real=0.80, crit={"needs_f1": 0.85}),
                            needs_n=1202, needs_pos=100))
    dip = summarize(_with(_result("d", real=0.82, crit={"needs_f1": 0.83}),
                          needs_n=1202, needs_pos=100))
    assert gates(dip, champ)[0]
    big = summarize(_with(_result("e", real=0.82, crit={"needs_f1": 0.75}),
                          needs_n=1202, needs_pos=100))
    ok, why = gates(big, champ)
    assert not ok and any("needs_f1 dropped" in w for w in why)
    # A champion from before *_pos existed still gets the positive count from the candidate.
    old = summarize(_with(_result("o", real=0.80, crit={"needs_f1": 0.85}), needs_n=1202))
    assert gates(dip, old)[0]
    # Without positives on either side, large slices keep the fixed limit.
    no_pos = summarize(_with(_result("f", real=0.82, crit={"needs_f1": 0.83}), needs_n=1202))
    assert not gates(no_pos, old)[0]


def test_same_architecture_latency_noise_does_not_reject():
    champ = summarize(_result("c", real=0.80, latency=57.4))
    slow = summarize(_result("s", real=0.82, latency=105.0))
    assert gates(slow, champ)[0]
    other = _result("x", real=0.82, latency=105.0)
    other["config"]["backbone"] = "other"
    ok, why = gates(summarize(other), champ)
    assert not ok and any("latency" in w for w in why)
    longer = _result("l", real=0.82, latency=105.0)
    longer["config"]["max_length"] = 512
    assert not gates(summarize(longer), champ)[0]
    assert not gates(summarize(_result("r", latency=105.0)), None)[0]
