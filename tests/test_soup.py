import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from rescue_tagger.soup import average


def _model(d, enc, head, cfg=None):
    (d / "encoder").mkdir(parents=True)
    (d / "tokenizer").mkdir()
    save_file({"w": torch.tensor(enc)}, d / "encoder" / "model.safetensors")
    save_file({"heads.h.weight": torch.tensor(head)}, d / "heads.safetensors")
    (d / "tagger_config.json").write_text(json.dumps(cfg or {"backbone": "b"}))
    (d / "calibration.json").write_text("{}")
    return d


def test_average_is_uniform_and_drops_stale_calibration(tmp_path):
    a = _model(tmp_path / "a", [1.0, 3.0], [2.0])
    b = _model(tmp_path / "b", [3.0, 5.0], [4.0])
    out = average([a, b], tmp_path / "soup")
    assert load_file(out / "encoder" / "model.safetensors")["w"].tolist() == [2.0, 4.0]
    assert load_file(out / "heads.safetensors")["heads.h.weight"].tolist() == [3.0]
    assert not (out / "calibration.json").exists()   # must be re-fitted for the soup


def test_average_refuses_different_layouts(tmp_path):
    a = _model(tmp_path / "a", [1.0], [2.0])
    b = _model(tmp_path / "b", [1.0], [2.0], cfg={"backbone": "other"})
    with pytest.raises(ValueError):
        average([a, b], tmp_path / "soup")
