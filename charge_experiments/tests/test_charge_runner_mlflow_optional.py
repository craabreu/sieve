"""MLflow-dependent tests for runner._ensure_experiment's retry-on-race
behavior and runner.promote_run. Skipped when mlflow is absent -- same
convention as test_charge_aggregate_optional.py."""

from __future__ import annotations

import json
import time

import pytest
import yaml

pytest.importorskip("mlflow")


def _write_completed_run(run_dir, *, experiment="exp", store="synthetic"):
    """A minimal run directory shaped like a real, finished run: only
    config.resolved.yaml + metrics.json -- exactly what promote_run reads,
    and all it requires (no predictions.npz/manifest.json/plots needed)."""
    run_dir.mkdir(parents=True)
    raw = {
        "run": {"experiment": experiment, "seed": 0, "tags": {}},
        "data": {
            "store": store,
            "split_column": "split",
            "train_split": "train",
            "val_split": "val",
            "eval_split": "test",
        },
        "predictor": {"name": "global_mean", "params": {}},
        "normalization": None,
        "tree_stats_load_path": None,
        "save_tree_stats": False,
    }
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(raw))
    (run_dir / "metrics.json").write_text(json.dumps({"mae": 0.1, "r2": 0.9}))


def _operational_error() -> Exception:
    from sqlalchemy.exc import OperationalError

    return OperationalError("CREATE TABLE ...", {}, Exception("already exists"))


def test_ensure_experiment_retries_past_a_transient_operational_error(
    tmp_path, monkeypatch
):
    """The real failure mode: several run processes racing to run
    Alembic's one-time schema migration against a freshly created sqlite
    db all hit a genuine OperationalError except the winner. A transient
    one is retried, not fatal -- once it stops recurring, the call
    succeeds."""
    import mlflow

    from charge_experiments import runner

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(time, "sleep", lambda _: None)

    real_get = mlflow.get_experiment_by_name
    calls = {"n": 0}

    def _flaky_get(name):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _operational_error()
        return real_get(name)

    monkeypatch.setattr(mlflow, "get_experiment_by_name", _flaky_get)

    runner._ensure_experiment("exp", artifact_root=tmp_path / "art")

    assert calls["n"] == 3
    assert mlflow.get_experiment_by_name("exp") is not None


def test_ensure_experiment_raises_after_exhausting_retries(tmp_path, monkeypatch):
    """A persistent OperationalError (not just a one-time migration race)
    still surfaces, with the original error chained as its cause."""
    import mlflow

    from charge_experiments import runner

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        mlflow,
        "get_experiment_by_name",
        lambda name: (_ for _ in ()).throw(_operational_error()),
    )

    with pytest.raises(RuntimeError, match="could not initialize") as excinfo:
        runner._ensure_experiment("exp", artifact_root=tmp_path / "art")
    assert excinfo.value.__cause__ is not None


def test_promote_run_backfills_a_missing_mlflow_record(tmp_path):
    """The back-fill case: a run that finished writing local artifacts
    but was never logged (--no-tracking, or a crash after local write)
    gets an MLflow record now, under its own config.run.experiment."""
    import mlflow
    from charge_experiments.runner import promote_run

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    run_dir = tmp_path / "runs" / "exp" / "r1"
    _write_completed_run(run_dir, experiment="exp")

    result = promote_run(
        run_dir,
        runs_root=tmp_path / "runs",
        tracking=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "art",
    )

    assert result.logged is True
    assert result.moved is False
    assert result.experiment == "exp"
    assert result.run_dir == run_dir

    exp = mlflow.get_experiment_by_name("exp")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.tags["run_dir"] == str(run_dir)


def test_promote_run_is_idempotent(tmp_path):
    """A second promote_run on the same run_dir does not create a
    duplicate MLflow record."""
    import mlflow
    from charge_experiments.runner import promote_run

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    run_dir = tmp_path / "runs" / "exp" / "r1"
    _write_completed_run(run_dir, experiment="exp")

    first = promote_run(
        run_dir,
        runs_root=tmp_path / "runs",
        tracking=tracking,
        artifact_root=tmp_path / "art",
    )
    second = promote_run(
        run_dir,
        runs_root=tmp_path / "runs",
        tracking=tracking,
        artifact_root=tmp_path / "art",
    )

    assert first.logged is True
    assert second.logged is False
    exp = mlflow.get_experiment_by_name("exp")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    assert len(runs) == 1


def test_promote_run_moves_to_a_different_experiment(tmp_path):
    """The exploration -> baseline case: target_experiment moves the
    run_dir under runs_root/<target>/<name> and logs it there."""
    import mlflow
    from charge_experiments.runner import promote_run

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "exp-exploration" / "r1"
    _write_completed_run(run_dir, experiment="exp-exploration")

    result = promote_run(
        run_dir,
        target_experiment="exp-real",
        runs_root=runs_root,
        tracking=tracking,
        artifact_root=tmp_path / "art",
    )

    assert result.moved is True
    assert result.logged is True
    assert result.experiment == "exp-real"
    assert result.run_dir == runs_root / "exp-real" / "r1"
    assert not run_dir.exists()
    assert (runs_root / "exp-real" / "r1" / "metrics.json").exists()

    exp = mlflow.get_experiment_by_name("exp-real")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.tags["run_dir"] == str(runs_root / "exp-real" / "r1")


def test_promote_run_with_target_equal_to_own_experiment_does_not_move(tmp_path):
    import mlflow
    from charge_experiments.runner import promote_run

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "exp" / "r1"
    _write_completed_run(run_dir, experiment="exp")

    result = promote_run(
        run_dir,
        target_experiment="exp",
        runs_root=runs_root,
        tracking=tracking,
        artifact_root=tmp_path / "art",
    )

    assert result.moved is False
    assert result.run_dir == run_dir
    assert run_dir.exists()


def test_promote_run_raises_for_an_incomplete_run(tmp_path):
    from charge_experiments.runner import promote_run

    run_dir = tmp_path / "runs" / "exp" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text("run: {}\n")
    # no metrics.json -- run never finished

    with pytest.raises(FileNotFoundError, match=r"metrics\.json|config\.resolved"):
        promote_run(run_dir, runs_root=tmp_path / "runs")


def test_execute_logs_batch_id_as_an_mlflow_tag(tmp_path):
    """run.batch_id shows up as its own MLflow tag (independent of the
    run_name string it also prefixes), for querying a whole sweep at
    once."""
    pytest.importorskip("rdkit")

    import mlflow
    import numpy as np
    from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from charge_experiments.runner import execute

    from charge_experiments.tests.helpers import synthetic_molecule_set

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = ExperimentCfg(
        run=RunCfg(experiment="exp", seed=0, batch_id="10fold-2026"),
        data=DataCfg(store="synthetic", split_column="split"),
        predictor=PredictorCfg(name="global_mean", params={}),
    )
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    rng = np.random.default_rng(1)
    labels = rng.choice(["train", "val", "test"], size=20, p=[0.6, 0.2, 0.2])
    masks = {name: labels == name for name in ("train", "val", "test")}

    execute(
        cfg,
        mset,
        masks,
        runs_root=tmp_path / "runs",
        allow_dirty=True,
        tracking=tracking,
    )

    mlflow.set_tracking_uri(tracking)
    exp = mlflow.get_experiment_by_name("exp")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.tags["batch_id"] == "10fold-2026"
    assert runs[0].info.run_name.startswith("10fold-2026__")
