from rescue_tagger import scheduler


def test_wait_for_run_returns_when_completed(monkeypatch):
    states = iter(["in_progress", "in_progress", "completed"])
    monkeypatch.setattr(scheduler, "_gh", lambda m, p, b=None: {"status": next(states)})
    monkeypatch.setattr(scheduler.time, "sleep", lambda s: None)
    assert scheduler.wait_for_run("1", timeout_s=60, poll_s=1)


def test_wait_for_run_gives_up(monkeypatch):
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(scheduler, "_gh", lambda m, p, b=None: {"status": "in_progress"})
    monkeypatch.setattr(scheduler.time, "sleep", lambda s: None)
    monkeypatch.setattr(scheduler.time, "time", lambda: next(clock))
    assert not scheduler.wait_for_run("1", timeout_s=30, poll_s=10)
