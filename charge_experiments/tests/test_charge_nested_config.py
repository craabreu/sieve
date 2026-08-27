from __future__ import annotations

import pytest
import yaml


def _write_yaml(tmp_path, data):
    path = tmp_path / "nested_config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base_raw():
    return {
        "run": {"experiment": "charge-nested-smoke", "seed": 0},
        "data": {"store": "dash-molecules", "split_column": "split"},
        "predictor": {"name": "dash", "params": {}},
        "tree_stats": {"save_path": "artifacts/stats.npz"},
        "children": ["std_weighted", "equal_weighted"],
    }


def test_load_nested_config_round_trips(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_nested_config(path)

    assert cfg.run.experiment == "charge-nested-smoke"
    assert cfg.predictor.name == "dash"
    assert cfg.tree_stats.save_path == "artifacts/stats.npz"
    assert cfg.tree_stats.load_path is None
    assert cfg.children == ("std_weighted", "equal_weighted")


def test_load_nested_config_rejects_unknown_normalization_scheme(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["children"] = ["not_a_real_scheme"]
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="unknown normalization scheme"):
        load_nested_config(path)


def test_load_nested_config_requires_at_least_one_child(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["children"] = []
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="at least one"):
        load_nested_config(path)


def test_load_nested_config_allows_a_predictor_with_no_tree_stats(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["predictor"] = {"name": "dash_pretrained", "params": {}}
    del raw["tree_stats"]
    path = _write_yaml(tmp_path, raw)

    cfg = load_nested_config(path)
    assert cfg.predictor.name == "dash_pretrained"
    assert cfg.tree_stats.save_path is None
    assert cfg.tree_stats.load_path is None


def test_load_nested_config_rejects_unknown_top_level_key(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["bogus"] = 1
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="unknown key"):
        load_nested_config(path)
