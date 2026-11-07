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


def test_model_card_lists_soup_ingredients_and_discloses_opus_labels(tmp_path):
    import json

    (tmp_path / "tagger_config.json").write_text(json.dumps({"backbone": "b"}))
    (tmp_path / "soup.json").write_text(json.dumps({"ingredients": [
        "/root/.cache/huggingface/hub/models--org--tagger/snapshots/" + "a" * 40, "/tmp/x/run9"]}))
    card = model_card(tmp_path, None, opus_labels=True)
    assert f"hf:org/tagger@{'a' * 40}" in card and "`run9`" in card and "/root" not in card
    assert "Model soup" in card and "Opus rubric labels" in card
    assert "Opus rubric labels" not in model_card(tmp_path, None)
