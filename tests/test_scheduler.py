from rescue_tagger.scheduler import State, next_action, resolve_reuse, run_name

Q = {"night": "n1", "max_runs": 3, "max_rejections": 2, "max_parallel": 1, "items": [
    {"name": "base", "config": "c.yaml", "reuse": "latest:train_laya"},
    {"name": "seed7", "config": "c.yaml", "purpose": "seed", "overrides": {"seed": 7}},
    {"name": "lr", "config": "c.yaml", "overrides": {"lr_encoder": 2e-5}},
    {"name": "drop", "config": "c.yaml", "overrides": {"dropout": 0.2}},
    {"name": "ep", "config": "c.yaml", "overrides": {"epochs": 6}},
]}


def dec(promote, *reasons):
    return {"promote": promote, "reasons": list(reasons) or ["x"]}


def st(decisions=None, active=None, failures=None, dirs=("train_laya-20261107-0432",)):
    names = {k: run_name("n1", {"name": k}) for k in ("base", "seed7", "lr", "drop", "ep")}
    return State({names[k]: v for k, v in (decisions or {}).items()},
                 [names[a] for a in (active or [])],
                 {names[k]: v for k, v in (failures or {}).items()}, list(dirs))


def test_launches_first_pending_with_resolved_reuse():
    act = next_action(Q, st())
    assert act["action"] == "launch"
    assert act["inputs"]["reuse"] == "train_laya-20261107-0432"
    assert act["inputs"]["run_name"] == "loop-n1-base"


def test_waits_while_a_run_is_going_and_passes_overrides():
    assert next_action(Q, st(active=["base"]))["action"] == "wait"
    act = next_action(Q, st(decisions={"base": dec(True)}))
    assert act["inputs"]["overrides"] == "seed=7"


def test_seed_repeats_do_not_count_as_rejections():
    act = next_action(Q, st(decisions={"base": dec(True), "seed7": dec(False), "lr": dec(False)}))
    assert act["action"] == "launch" and act["inputs"]["run_name"] == "loop-n1-drop"


def test_stops_after_rejections_in_a_row():
    act = next_action(Q, st(decisions={"base": dec(True), "lr": dec(False), "drop": dec(False)}))
    assert act["action"] == "stop" and "rejected" in act["why"]


def test_stops_when_gap_widens_twice():
    q = {**Q, "max_rejections": 9}
    g = "synthetic-vs-real gap widened 0.1 -> 0.2"
    act = next_action(q, st(decisions={"base": dec(True), "seed7": dec(False, g),
                                       "lr": dec(False, g)}))
    assert act["action"] == "stop" and "gap" in act["why"]


def test_stops_at_run_cap_and_on_repeated_failure():
    done = {"base": dec(True), "seed7": dec(True), "lr": dec(True), "drop": dec(True)}
    act = next_action(Q, st(decisions=done))
    assert act["action"] == "stop" and "cap" in act["why"]
    assert next_action(Q, st(failures={"base": 2}))["action"] == "stop"


def test_unresolvable_reuse_is_skipped():
    assert next_action(Q, st(dirs=()))["action"] == "skip"
    assert resolve_reuse("latest:x", ["x-1", "x-2", "y-3"]) == "x-2"
