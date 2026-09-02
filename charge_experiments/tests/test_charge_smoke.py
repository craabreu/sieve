# charge_experiments/tests/test_charge_smoke.py
"""End-to-end smoke test on a synthetic store -- no download, no network,
no mlflow required (tracking=None)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
from charge_experiments.runner import execute

from charge_experiments.tests.helpers import synthetic_molecule_set


def _tiny_cfg() -> ExperimentCfg:
    return ExperimentCfg(
        run=RunCfg(experiment="charge-smoke-test", seed=0, tags={"stage": "smoke"}),
        data=DataCfg(
            store="synthetic",
            split_column="split",
            train_split="train",
            val_split="val",
            eval_split="test",
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


def test_smoke_reports_featurize_time_for_sieve_predictor(tmp_path):
    """time/featurize_s surfaces SievePredictor's own build_codes/from_rdkit
    time, separately from time/fit_s -- both of which already include it
    without breaking it out. Absent (not zero, absent) for predictors that
    don't expose last_featurize_s, since runner.py reads it via getattr and
    only sets the key when present."""
    import pytest

    pytest.importorskip("rdkit")

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()
    cfg = cfg.__class__(
        run=cfg.run,
        data=cfg.data,
        predictor=PredictorCfg(name="sieve", params={"max_wl_depth": 2}),
    )

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    # Not compared against time/fit_s: featurize_s is measured as a sub-span
    # inside fit's own timed span, so it is structurally <= fit_s, but at
    # this corpus's sub-millisecond scale perf_counter's own granularity
    # makes a strict numeric comparison flaky -- the real point of this test
    # is that the key surfaces at all, correctly, only for a predictor that
    # actually tracks it (see the "omits" test below for the negative case).
    assert "time/featurize_s" in result.metrics
    assert result.metrics["time/featurize_s"] >= 0.0


def test_smoke_omits_featurize_time_for_predictors_without_it(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()  # global_mean predictor -- no last_featurize_s attribute

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert "time/featurize_s" not in result.metrics


def test_smoke_rejects_dirty_tree_by_default(tmp_path, monkeypatch):
    import charge_experiments.runner as runner_mod
    import pytest

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


def _sieve_cfg(**params):
    cfg = _tiny_cfg()
    return cfg.__class__(
        run=cfg.run,
        data=cfg.data,
        predictor=PredictorCfg(name="sieve", params={"max_wl_depth": 2, **params}),
    )


def test_report_loo_adds_train_loo_metrics(tmp_path):
    """train_loo/* appears only when the predictor opts in. Off is the
    default precisely so an existing run's key set does not shift under
    anyone, so the negative case is asserted too."""
    import pytest

    pytest.importorskip("rdkit")

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)

    on = execute(
        _sieve_cfg(report_loo=True),
        mset,
        masks,
        runs_root=tmp_path / "on",
        allow_dirty=True,
        tracking=None,
    )
    assert "train_loo/mae" in on.metrics
    assert "train_loo/r2" in on.metrics
    assert "time/train_loo_predict_s" in on.metrics

    off = execute(
        _sieve_cfg(),
        mset,
        masks,
        runs_root=tmp_path / "off",
        allow_dirty=True,
        tracking=None,
    )
    assert not any(k.startswith("train_loo/") for k in off.metrics)
    assert "time/train_loo_predict_s" not in off.metrics


def test_report_loo_is_ignored_by_predictors_without_it(tmp_path):
    """runner reads report_loo/predict_loo_raw via getattr/hasattr, so a
    predictor that has neither is unaffected rather than erroring."""
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    result = execute(
        _tiny_cfg(), mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert not any(k.startswith("train_loo/") for k in result.metrics)


def test_normalization_conserves_charge_on_every_split(tmp_path):
    """With normalization set, predict_raw's output is renormalized before
    scoring -- test/train/val/LOO should all come out charge-conserving
    (charge_conservation/mae ~ 0), which plain predict() does not
    guarantee (see sieve_predictor.py's own docstring)."""
    import pytest

    pytest.importorskip("rdkit")

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = replace(_sieve_cfg(report_loo=True), normalization="equal_weighted")

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.metrics["charge_conservation/mae"] < 1e-8
    assert result.metrics["train/charge_conservation/mae"] < 1e-8
    assert result.metrics["val/charge_conservation/mae"] < 1e-8
    assert result.metrics["train_loo/charge_conservation/mae"] < 1e-8


def test_normalization_none_matches_plain_predict(tmp_path):
    """The default (no normalization key) is byte-for-byte today's
    behavior: predict()'s own output, unnormalized."""
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)

    plain = execute(
        _tiny_cfg(),
        mset,
        masks,
        runs_root=tmp_path / "a",
        allow_dirty=True,
        tracking=None,
    )
    explicit_none = execute(
        replace(_tiny_cfg(), normalization=None),
        mset,
        masks,
        runs_root=tmp_path / "b",
        allow_dirty=True,
        tracking=None,
    )
    assert plain.metrics["mae"] == explicit_none.metrics["mae"]


def test_normalization_rejects_a_predictor_without_predict_raw(tmp_path):
    """global_mean has no predict_raw -- there is nothing to normalize, so
    this must raise rather than silently ignore the setting."""
    import pytest

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = replace(_tiny_cfg(), normalization="std_weighted")

    with pytest.raises(AttributeError, match="predict_raw"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)
