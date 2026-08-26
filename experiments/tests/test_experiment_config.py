"""Tests for experiments/sieve_experiments/config.py."""

from __future__ import annotations

import pytest
import yaml
from sieve_experiments.config import (
    ExperimentCfg,
    apply_overrides,
    load_config,
    to_flat_params,
)
from sieve_experiments.data import REPO_ROOT

CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_every_tracked_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentCfg)
    assert cfg.data.split_column in {"split", "biased_split"}
    assert cfg.predictor.name


def test_split_columns_cover_both_options():
    """The tracked configs exercise both split columns, not just one."""
    columns = {load_config(p).data.split_column for p in ALL_CONFIGS}
    assert columns == {"split", "biased_split"}


def test_biased_and_random_siblings_differ_only_in_split_column(tmp_path):
    dash_random = load_config(CONFIGS_DIR / "dash-random.yaml")
    dash_biased = load_config(CONFIGS_DIR / "dash-biased.yaml")
    assert dash_random.data.split_column == "split"
    assert dash_biased.data.split_column == "biased_split"
    assert dash_random.predictor == dash_biased.predictor
    assert dash_random.data.scheme == dash_biased.data.scheme
    assert dash_random.data.store == dash_biased.data.store


def test_unknown_top_level_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run: {experiment: x, seed: 0}\n"
        "data: {store: s, scheme: sc, split_column: split}\n"
        "predictor: {name: n}\n"
        "extra_section: {}\n"
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_config(bad)


def test_unknown_nested_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run: {experiment: x, seed: 0}\n"
        "data: {store: s, scheme: sc, split_column: split, typo_field: 1}\n"
        "predictor: {name: n}\n"
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_config(bad)


def test_invalid_split_column_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run: {experiment: x, seed: 0}\n"
        "data: {store: s, scheme: sc, split_column: not_a_real_column}\n"
        "predictor: {name: n}\n"
    )
    with pytest.raises(ValueError, match="split_column"):
        load_config(bad)


def test_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run: {experiment: x, seed: 0}\n"
        "data: {store: s, scheme: sc, split_column: split}\n"
    )
    with pytest.raises(ValueError, match="predictor"):
        load_config(bad)


# --- overrides ---------------------------------------------------------


def test_set_override_changes_nested_predictor_param():
    cfg = load_config(
        CONFIGS_DIR / "dash-random.yaml", overrides=["predictor.params.max_depth=8"]
    )
    assert cfg.predictor.params["max_depth"] == 8
    # unrelated params untouched
    assert cfg.predictor.params["attention_threshold"] == 5.2


def test_set_override_parses_scalar_types():
    raw = yaml.safe_load((CONFIGS_DIR / "dash-random.yaml").read_text())
    out = apply_overrides(
        raw,
        [
            "predictor.params.max_depth=8",
            "predictor.params.charge_reconciliation=none",
            "run.seed=3",
        ],
    )
    assert out["predictor"]["params"]["max_depth"] == 8
    assert isinstance(out["predictor"]["params"]["max_depth"], int)
    assert out["predictor"]["params"]["charge_reconciliation"] == "none"
    assert out["run"]["seed"] == 3


def test_override_without_equals_raises():
    raw = yaml.safe_load((CONFIGS_DIR / "dash-random.yaml").read_text())
    with pytest.raises(ValueError, match=r"key\.path=value"):
        apply_overrides(raw, ["not-an-override"])


# --- to_flat_params ------------------------------------------------------


def test_to_flat_params_round_trips_predictor_params():
    cfg = load_config(CONFIGS_DIR / "dash-random.yaml")
    flat = to_flat_params(cfg)
    assert flat["predictor.name"] == "dash"
    assert flat["predictor.params.max_depth"] == "16"
    assert flat["data.split_column"] == "split"
    assert flat["run.seed"] == "0"
    assert all(isinstance(v, str) for v in flat.values())
