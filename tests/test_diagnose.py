import pytest

from rescue_tagger.diagnose import diagnose, main


class FakeEx:
    def __init__(self, p):
        self.p = p

    def raw(self, text):
        return {"probs": {"is_request": self.p[text]}}


def _row(i, text, y, cat):
    return {"id": f"h:{i}", "source": "humaid", "lang": "en", "text": text,
            "labels": {"is_request": y}, "meta": {"class_label": cat}}


def test_rare_class_report():
    rows = [_row(i, f"t{i}", i < 2, "req" if i < 2 else "other") for i in range(20)]
    p = {f"t{i}": (0.9 if i == 0 else 0.1) for i in range(20)}
    r = diagnose(FakeEx(p), rows, show_errors=0)
    assert r["majority_baseline_accuracy"] == 0.9
    assert r["at_0.5"]["recall"] == 0.5 and r["at_0.5"]["precision"] == 1.0
    assert r["by_category"]["req"]["predicted_request_pct"] == 50.0
    assert "false_negatives" not in r


def test_refuses_to_print_eval_errors(tmp_path):
    f = tmp_path / "harvey.jsonl"
    f.write_text("")
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--data", str(f), "--show-errors", "5"])


def test_refuses_eval_rows_even_in_a_renamed_file(tmp_path):
    import json

    f = tmp_path / "copy.jsonl"
    f.write_text(json.dumps({**_row(0, "x", True, "c"), "split": "test"}) + "\n")
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--data", str(f), "--show-errors", "5"])
