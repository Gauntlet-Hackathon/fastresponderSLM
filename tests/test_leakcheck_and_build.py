import json
from pathlib import Path

from rescue_tagger.data.build import relabel_requests, synthetic_split
from rescue_tagger.data.leakcheck import LeakIndex, normalize

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize():
    assert normalize("Mi mamá usa OXÍGENO!!  ") == "mi mama usa oxigeno"


def test_exact_and_near_verbatim_leaks_are_caught():
    protected = ["HELP! Trapped on the roof with 3 kids, water rising fast, behind the Kroger on "
                 "Westheimer"]
    idx = LeakIndex(protected)
    assert idx.leaks("help trapped on the roof with 3 kids water rising fast behind the kroger on "
                     "westheimer")
    assert idx.leaks("PLEASE HELP! Trapped on the roof with 3 kids, water rising fast, behind the "
                     "Kroger on Westheimer Rd")
    assert not idx.leaks("We are on the roof at 15 Kirby Drive, two adults, water rising")


def test_contract_fixtures_are_protected():
    texts = [json.loads(x)["raw_text"]
             for x in (FIXTURES / "contract_cases.jsonl").read_text("utf-8").splitlines()]
    idx = LeakIndex(texts)
    assert all(idx.leaks(t) for t in texts)


def test_synthetic_split_is_fixed_and_balanced():
    ids = [f"synthetic:m:{i}" for i in range(5000)]
    splits = [synthetic_split(i, 10, 5) for i in ids]
    assert splits == [synthetic_split(i, 10, 5) for i in ids]
    assert 0.03 < splits.count("test") / 5000 < 0.07
    assert 0.08 < splits.count("dev") / 5000 < 0.12


def test_unset_eval_repo_fails_loudly(monkeypatch):
    import pytest

    from rescue_tagger.data.build import protected_texts

    monkeypatch.delenv("EVAL_REPO", raising=False)
    with pytest.raises(SystemExit, match="EVAL_REPO"):
        protected_texts(["$EVAL_REPO/evals/sets/*.jsonl"])


def _rec(split, meta, labels=None):
    return {"split": split, "meta": meta, "labels": labels or {
        "is_request": True, "request_kind": "supply",
        "needs": {"food": True, "water": False, "meds": False}}}


def test_request_relabel_rules_apply_to_train_and_dev_only():
    munro = {"not_request": {"genre": ["news"]}}
    news = _rec("train", {"genre": "news"})
    relabel_requests(news, munro)
    assert news["labels"] == {"is_request": False, "request_kind": "other",
                              "needs": {"food": False, "water": False, "meds": False}}
    direct = _rec("dev", {"genre": "direct"})
    relabel_requests(direct, munro)
    assert direct["labels"]["is_request"] is True
    test_news = _rec("test", {"genre": "news"})
    relabel_requests(test_news, munro)
    assert test_news["labels"]["is_request"] is True  # test sets keep upstream labels

    humaid = {"unreliable_is_request": {"class_label": ["sympathy_and_support"]}}
    sym = _rec("train", {"class_label": "sympathy_and_support"},
               {"is_request": False, "request_kind": "other"})
    relabel_requests(sym, humaid)
    assert sym["labels"] == {}
    harvey = _rec("test", {"class_label": "sympathy_and_support", "harvey_holdout": True},
                  {"is_request": False, "request_kind": "other"})
    relabel_requests(harvey, humaid)
    assert harvey["labels"]["is_request"] is False  # Harvey is never touched
