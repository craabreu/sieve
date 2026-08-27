# charge_experiments/tests/test_charge_smoke.py
"""End-to-end smoke test on a synthetic store -- no download, no network,
no mlflow required (tracking=None)."""

from __future__ import annotations

import json

import numpy as np
from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
from charge_experiments.runner import execute

from charge_experiments.tests.helpers import synthetic_molecule_set


def _tiny_cfg() -> ExperimentCfg:
    return ExperimentCfg(
        run=RunCfg(experiment="charge-smoke-test", seed=0, tags={"stage": "smoke"}),
        data=DataCfg(
            store="synthetic", split_column="split",
            train_split="train", val_split="val", eval_split="test",
        ),
        predictor=PredictorCfg(name="global_mean", params={}),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=n_mol, p=[0.6, 0.2, 0.2])
    return {name: labels == name for name in ("train", "val", "test")}


def test_smoke_pipeline_writes_every_artifact(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    run_dir = result.run_dir
    assert run_dir.is_dir()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "stdout.log").exists()


def test_smoke_metrics_include_r2_and_charge_conservation(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert np.isfinite(result.metrics["mae"])
    assert np.isfinite(result.metrics["rmse"])
    assert "r2" in result.metrics
    assert "charge_conservation/mae" in result.metrics
    assert "max_abs_residual" not in result.metrics


def test_smoke_rejects_dirty_tree_by_default(tmp_path, monkeypatch):
    import charge_experiments.runner as runner_mod
    import pytest

    monkeypatch.setattr(
        runner_mod, "_git_info",
        lambda repo_root: {
            "commit": "deadbeef",
            "branch": "main",
            "dirty": True,
            "describe": "deadbeef-dirty",
        },
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()

    with pytest.raises(RuntimeError, match="dirty"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None)


def test_smoke_handles_an_empty_test_split(tmp_path):
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    all_train = np.ones(10, dtype=bool)
    none = np.zeros(10, dtype=bool)
    masks = {"train": all_train, "val": none, "test": none}
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.metrics["n_test_conformers"] == 0
    assert np.isnan(result.metrics["mae"])


def test_smoke_metrics_json_matches_returned_metrics(tmp_path):
    mset = synthetic_molecule_set(n_mol=15, seed=2)
    masks = _synthetic_masks(15, seed=3)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    on_disk = json.loads((result.run_dir / "metrics.json").read_text())
    assert on_disk.keys() == result.metrics.keys()
    for key in on_disk:
        if isinstance(on_disk[key], float) and np.isnan(on_disk[key]):
            assert np.isnan(result.metrics[key])
        else:
            assert on_disk[key] == result.metrics[key]
