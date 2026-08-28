"""Smoke tests for nested_runner.py's orchestration -- a fake, in-process
NormalizableChargePredictor stands in for dash/dash_pretrained (both need
the real DASH-tree clone; this file tests execute_nested's own logic, not
DASH-tree matching), mirroring test_charge_smoke.py's synthetic-data
pattern. tracking=None throughout: no real MLflow server/tracking dir
needed here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction, RawPrediction

from charge_experiments.tests.helpers import synthetic_molecule_set


class _FakeNormalizablePredictor:
    """fit() sets one scalar (train's own mean atom charge); predict_raw()
    returns that scalar for every atom, plus a constant std. Good enough to
    exercise execute_nested's own orchestration (parent/children run-dirs,
    save/load-skips-fit, no re-matching per child) without rdkit or a real
    DASH-tree clone."""

    name: ClassVar[str] = "fake_normalizable"

    def __init__(self) -> None:
        self.fit_calls = 0
        self.predict_raw_calls = 0
        self._value: float | None = None

    def fit(self, train, val, *, rng) -> None:
        del val, rng
        self.fit_calls += 1
        self._value = float(train.atom_charge.mean())

    def predict(self, test) -> Prediction:
        return Prediction(atom_charge=self.predict_raw(test).atom_charge)

    def predict_raw(self, test) -> RawPrediction:
        self.predict_raw_calls += 1
        assert self._value is not None
        n = test.n_atoms
        return RawPrediction(
            atom_charge=np.full(n, self._value), atom_std=np.full(n, 0.1)
        )

    def save_model_state(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"value": self._value}))

    def load_model_state(self, path) -> None:
        self._value = json.loads(Path(path).read_text())["value"]


register("fake_normalizable", lambda params: _FakeNormalizablePredictor())


def _nested_cfg(tmp_path, *, load_path=None):
    from charge_experiments.config import DataCfg, PredictorCfg, RunCfg
    from charge_experiments.nested_config import NestedExperimentCfg, TreeStatsCfg

    return NestedExperimentCfg(
        run=RunCfg(experiment="charge-nested-smoke", seed=0),
        data=DataCfg(store="synthetic", split_column="split"),
        predictor=PredictorCfg(name="fake_normalizable", params={}),
        tree_stats=TreeStatsCfg(load_path=load_path),
        children=("std_weighted", "equal_weighted"),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=n_mol, p=[0.6, 0.2, 0.2])
    return {name: labels == name for name in ("train", "val", "test")}


def test_execute_nested_writes_a_parent_and_two_child_runs(tmp_path):
    from charge_experiments.nested_runner import execute_nested

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _nested_cfg(tmp_path)

    result = execute_nested(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert result.parent.run_dir.is_dir()
    assert (result.parent.run_dir / "metrics.json").exists()
    assert (result.parent.run_dir / "manifest.json").exists()
    assert set(result.children) == {"std_weighted", "equal_weighted"}
    for name, child in result.children.items():
        assert child.run_dir.is_dir()
        assert (child.run_dir / "metrics.json").exists()
        manifest = json.loads((child.run_dir / "manifest.json").read_text())
        assert manifest["normalization"] == name


def test_execute_nested_children_reuse_the_same_raw_predictions(tmp_path, monkeypatch):
    """predict_raw must be called exactly once per split, not once per
    child -- children only re-normalize."""
    import charge_experiments.nested_runner as nested_runner_mod
    from charge_experiments.nested_runner import execute_nested
    from charge_experiments.predictors import build

    mset = synthetic_molecule_set(n_mol=12, seed=0)
    masks = _synthetic_masks(12, seed=1)
    cfg = _nested_cfg(tmp_path)

    captured: list = []

    def _spying_build(name, params):
        predictor = build(name, params)
        captured.append(predictor)
        return predictor

    monkeypatch.setattr(nested_runner_mod, "build", _spying_build)
    execute_nested(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert len(captured) == 1
    # one predict_raw call per non-empty split (train/val/test), not per child
    assert captured[0].predict_raw_calls <= 3


def test_execute_nested_auto_saves_tree_stats_inside_the_parent_run_dir(tmp_path):
    """No configurable save path: a fit()-based parent run always writes
    its own tree_stats.npz alongside its own metrics.json/manifest.json --
    the run directory itself is the provenance record."""
    from charge_experiments.nested_runner import execute_nested

    mset = synthetic_molecule_set(n_mol=12, seed=0)
    masks = _synthetic_masks(12, seed=1)
    cfg = _nested_cfg(tmp_path)

    result = execute_nested(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert (result.parent.run_dir / "tree_stats.npz").exists()


def test_execute_nested_load_tree_stats_skips_fit(tmp_path, monkeypatch):
    from charge_experiments.nested_runner import execute_nested

    mset = synthetic_molecule_set(n_mol=12, seed=0)
    masks = _synthetic_masks(12, seed=1)

    fit_cfg = _nested_cfg(tmp_path)
    fit_result = execute_nested(
        fit_cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    stats_path = fit_result.parent.run_dir / "tree_stats.npz"
    assert stats_path.exists()

    load_cfg = _nested_cfg(tmp_path, load_path=str(stats_path))
    import charge_experiments.nested_runner as nested_runner_mod
    from charge_experiments.predictors import build

    captured: list = []

    def _spying_build(name, params):
        predictor = build(name, params)
        captured.append(predictor)
        return predictor

    monkeypatch.setattr(nested_runner_mod, "build", _spying_build)
    result = execute_nested(
        load_cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert captured[0].fit_calls == 0
    manifest = json.loads((result.parent.run_dir / "manifest.json").read_text())
    assert manifest["tree_stats_source"] == "loaded"
    # a loaded-from run never re-saves its own tree_stats.npz
    assert not (result.parent.run_dir / "tree_stats.npz").exists()


def test_execute_nested_rejects_a_dirty_tree_by_default(tmp_path, monkeypatch):
    import charge_experiments.nested_runner as nested_runner_mod
    import pytest
    from charge_experiments.nested_runner import execute_nested

    monkeypatch.setattr(
        nested_runner_mod._runner,
        "_git_info",
        lambda repo_root: {
            "commit": "deadbeef", "branch": "main", "dirty": True,
            "describe": "deadbeef-dirty",
        },
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _nested_cfg(tmp_path)

    with pytest.raises(RuntimeError, match="dirty"):
        execute_nested(
            cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None
        )
