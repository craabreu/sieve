from __future__ import annotations

import pytest
import yaml
from experiments.config import _build, to_dict, to_flat_params


def _write_yaml(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base_raw():
    return {
        "run": {"experiment": "charge-smoke", "seed": 0},
        "data": {"store": "dash-molecules", "split_column": "split"},
        "target": {"atom_property": "MBIScharge", "molecule_property": "net_charge"},
        "predictor": {"name": "global_mean", "params": {}},
    }


def test_load_config_round_trips_a_minimal_yaml(tmp_path):
    from experiments.config import load_config

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
    assert cfg.run.batch_id is None


def test_load_config_reads_batch_id(tmp_path):
    from experiments.config import load_config

    raw = _base_raw()
    raw["run"]["batch_id"] = "dash-10fold-2026-09-03"
    path = _write_yaml(tmp_path, raw)
    cfg = load_config(path)
    assert cfg.run.batch_id == "dash-10fold-2026-09-03"


def test_load_config_sets_batch_id_via_override(tmp_path):
    from experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path, overrides=["run.batch_id=my-batch"])
    assert cfg.run.batch_id == "my-batch"


def test_to_dict_and_to_flat_params_include_batch_id(tmp_path):
    from experiments.config import load_config, to_dict, to_flat_params

    raw = _base_raw()
    raw["run"]["batch_id"] = "my-batch"
    path = _write_yaml(tmp_path, raw)
    cfg = load_config(path)
    assert to_dict(cfg)["run"]["batch_id"] == "my-batch"
    assert to_flat_params(cfg)["run.batch_id"] == "my-batch"


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    from experiments.config import load_config

    raw = _base_raw()
    raw["bogus"] = 1
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_load_config_rejects_invalid_split_column(tmp_path):
    from experiments.config import load_config

    raw = _base_raw()
    raw["data"]["split_column"] = "not_a_real_column"
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="split_column"):
        load_config(path)


def test_load_config_applies_set_overrides(tmp_path):
    from experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path, overrides=["predictor.params.max_wl_depth=3"])
    assert cfg.predictor.params["max_wl_depth"] == 3


def test_to_dict_and_to_flat_params_round_trip(tmp_path):
    from experiments.config import load_config, to_dict, to_flat_params

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    d = to_dict(cfg)
    assert d["run"]["experiment"] == "charge-smoke"
    flat = to_flat_params(cfg)
    assert flat["data.store"] == "dash-molecules"


def test_load_config_defaults_normalization_to_none(tmp_path):
    from experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    assert cfg.normalization is None


def test_load_config_accepts_a_known_normalization_scheme(tmp_path):
    from experiments.config import load_config

    raw = _base_raw()
    raw["normalization"] = "std_weighted"
    path = _write_yaml(tmp_path, raw)
    cfg = load_config(path)
    assert cfg.normalization == "std_weighted"


def test_load_config_rejects_an_unknown_normalization_scheme(tmp_path):
    from experiments.config import load_config

    raw = _base_raw()
    raw["normalization"] = "bogus_scheme"
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="normalization"):
        load_config(path)


def test_to_dict_and_to_flat_params_include_normalization(tmp_path):
    from experiments.config import load_config, to_dict, to_flat_params

    raw = _base_raw()
    raw["normalization"] = "equal_weighted"
    path = _write_yaml(tmp_path, raw)
    cfg = load_config(path)
    assert to_dict(cfg)["normalization"] == "equal_weighted"
    assert to_flat_params(cfg)["normalization"] == "equal_weighted"


def test_to_dict_and_to_flat_params_spell_out_none_normalization(tmp_path):
    from experiments.config import load_config, to_dict, to_flat_params

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    assert to_dict(cfg)["normalization"] is None
    assert to_flat_params(cfg)["normalization"] == "None"


def _minimal_raw():
    return {
        "run": {"experiment": "e", "seed": 0},
        "data": {"store": "s", "split_column": "split"},
        "target": {"atom_property": "MBIScharge"},
        "predictor": {"name": "global_mean"},
    }


def test_target_section_parsed():
    cfg = _build(_minimal_raw())
    assert cfg.target.atom_property == "MBIScharge"
    assert cfg.target.molecule_property is None
    assert cfg.target.axis_label == "MBIScharge"


def test_target_label_overrides_axis_label():
    raw = _minimal_raw()
    raw["target"]["label"] = "charge (e)"
    assert _build(raw).target.axis_label == "charge (e)"


def test_missing_target_section_raises():
    raw = _minimal_raw()
    del raw["target"]
    with pytest.raises(ValueError, match="target"):
        _build(raw)


def test_missing_atom_property_raises():
    raw = _minimal_raw()
    raw["target"] = {}
    with pytest.raises(ValueError, match="atom_property"):
        _build(raw)


def test_unknown_target_key_raises():
    raw = _minimal_raw()
    raw["target"]["units"] = "e"
    with pytest.raises(ValueError, match="units"):
        _build(raw)


def test_normalization_without_molecule_property_raises():
    raw = _minimal_raw()
    raw["normalization"] = "equal_weighted"
    with pytest.raises(ValueError, match="molecule_property"):
        _build(raw)


def test_normalization_with_molecule_property_ok():
    raw = _minimal_raw()
    raw["target"]["molecule_property"] = "net_charge"
    raw["normalization"] = "equal_weighted"
    assert _build(raw).normalization == "equal_weighted"


def test_target_round_trips_through_to_dict_and_flat_params():
    raw = _minimal_raw()
    raw["target"]["molecule_property"] = "net_charge"
    cfg = _build(raw)
    assert to_dict(cfg)["target"] == {
        "atom_property": "MBIScharge",
        "molecule_property": "net_charge",
        "label": None,
    }
    flat = to_flat_params(cfg)
    assert flat["target.atom_property"] == "MBIScharge"
    assert flat["target.molecule_property"] == "net_charge"
