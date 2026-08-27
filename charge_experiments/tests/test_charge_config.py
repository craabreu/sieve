from __future__ import annotations

import pytest
import yaml


def _write_yaml(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base_raw():
    return {
        "run": {"experiment": "charge-smoke", "seed": 0},
        "data": {"store": "dash-molecules", "split_column": "split"},
        "predictor": {"name": "global_mean", "params": {}},
    }


def test_load_config_round_trips_a_minimal_yaml(tmp_path):
    from charge_experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    assert cfg.run.experiment == "charge-smoke"
    assert cfg.run.seed == 0
    assert cfg.data.store == "dash-molecules"
    assert cfg.data.split_column == "split"
    assert cfg.data.train_split == "train"
    assert cfg.data.val_split == "val"
    assert cfg.data.eval_split == "test"
    assert cfg.predictor.name == "global_mean"


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    from charge_experiments.config import load_config

    raw = _base_raw()
    raw["bogus"] = 1
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_load_config_rejects_invalid_split_column(tmp_path):
    from charge_experiments.config import load_config

    raw = _base_raw()
    raw["data"]["split_column"] = "not_a_real_column"
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="split_column"):
        load_config(path)


def test_load_config_applies_set_overrides(tmp_path):
    from charge_experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path, overrides=["predictor.params.max_wl_depth=3"])
    assert cfg.predictor.params["max_wl_depth"] == 3


def test_to_dict_and_to_flat_params_round_trip(tmp_path):
    from charge_experiments.config import load_config, to_dict, to_flat_params

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    d = to_dict(cfg)
    assert d["run"]["experiment"] == "charge-smoke"
    flat = to_flat_params(cfg)
    assert flat["data.store"] == "dash-molecules"
