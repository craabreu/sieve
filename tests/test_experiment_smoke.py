"""End-to-end smoke test on a synthetic store -- no cosmolayer, no rdkit,
no network, no mlflow required (tracking=None). Should run in well under a
second."""

from __future__ import annotations

import json

import numpy as np
from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
from sieve_experiments.runner import execute

from tests.helpers import synthetic_molecule_set


def _tiny_cfg(**run_overrides) -> ExperimentCfg:
    return ExperimentCfg(
        run=RunCfg(
            experiment="smoke-test", seed=0, tags={"stage": "smoke"}, **run_overrides
        ),
        data=DataCfg(
            store="synthetic",
            scheme="cosmo-sac-2010",
            split_column="split",
            train_split="train",
            val_split="val",
            eval_split="test",
        ),
        predictor=PredictorCfg(name="global_mean", params={}),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    """train/val/test masks over a synthetic_molecule_set, deterministic."""
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


def test_smoke_metrics_are_finite_and_charge_has_no_r2(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert "charge/r2" not in result.metrics
    assert "area/r2" in result.metrics  # global_mean supplies area


def test_smoke_manifest_records_git_and_seed(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["seed"] == 0
    assert "commit" in manifest["git"]
    assert manifest["data"]["n_train_molecules"] > 0
    assert manifest["data"]["n_test_molecules"] > 0


def test_smoke_rejects_dirty_tree_by_default(tmp_path, monkeypatch):
    import sieve_experiments.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_git_info",
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

    import pytest

    with pytest.raises(RuntimeError, match="dirty"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None)


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
