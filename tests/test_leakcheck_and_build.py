import json
from pathlib import Path

from rescue_tagger.data.build import synthetic_split
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
