import json

from rescue_tagger.publish import model_card


def test_model_card_states_license_gating_sources_and_citation(tmp_path):
    (tmp_path / "tagger_config.json").write_text(json.dumps({"backbone": "org/encoder"}))
    card = model_card(tmp_path, {"dev": {"is_request_accuracy": 0.9}})
    assert "license: cc-by-nc-sa-4.0" in card
    assert "extra_gated_prompt" in card and "non-commercial research" in card
    for source in ("HumAID", "Munro", "Synthetic", "org/encoder"):
        assert source in card
    assert "@inproceedings{alam2021humaid," in card
    assert '"is_request_accuracy": 0.9' in card
