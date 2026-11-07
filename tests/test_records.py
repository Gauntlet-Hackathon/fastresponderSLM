import copy
import json
from pathlib import Path

import pytest

from rescue_tagger import labels as L
from rescue_tagger.records import validate_extracted, validate_record

FIXTURES = Path(__file__).parent / "fixtures"


def _base() -> dict:
    return {
        "id": "synthetic:test:1",
        "source": "synthetic",
        "lang": "en",
        "text": "3 of us on the roof at 15 Kirby Drive",
        "split": "train",
        "labels": {"is_request": True, "hazards": {k: k == "on_roof" for k in L.HAZARD_KINDS}},
        "spans": [{"start": 0, "end": 7, "label": "COUNT"},
                  {"start": 23, "end": 37, "label": "LOCATION"}],
        "people_count": 3,
    }


def test_valid_record_passes():
    validate_record(_base())


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda r: r.update(source="opus"), "unknown source"),
        (lambda r: r["labels"].update(is_request="yes"), "must be bool"),
        (lambda r: r["labels"].update(hazards={"fire": True}), "every kind"),
        (lambda r: r["labels"].update(request_kind="boat"), "not in"),
        (lambda r: r["labels"].update(priority="P1"), "unknown head"),
        (lambda r: r["spans"].append({"start": 30, "end": 99, "label": "TIME"}), "out of bounds"),
        (lambda r: r["spans"].append({"start": 2, "end": 9, "label": "TIME"}), "overlapping"),
        (lambda r: r["spans"].__setitem__(0, {"start": 0, "end": 8, "label": "COUNT"}),
         "whitespace"),
        (lambda r: r.update(people_count=-1), "people_count"),
        (lambda r: r.update(weight=0), "weight"),
    ],
)
def test_invalid_records_fail(mutate, msg):
    rec = copy.deepcopy(_base())
    mutate(rec)
    with pytest.raises(ValueError, match=msg):
        validate_record(rec)


def test_contract_fixture_payloads_validate():
    """Every hand-written `extracted` payload in the product fixtures passes our validator."""
    for line in (FIXTURES / "contract_cases.jsonl").read_text("utf-8").splitlines():
        validate_extracted(json.loads(line)["extracted"])


def test_invalid_payload_is_rejected():
    payload = json.loads((FIXTURES / "contract_cases.jsonl").read_text("utf-8").splitlines()[0])
    bad = payload["extracted"] | {"request_kind": "boat"}
    with pytest.raises(ValueError):
        validate_extracted(bad)
