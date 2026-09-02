"""MLflow-dependent aggregate tests. Skipped when mlflow is absent --
matches this series' own *_optional.py convention for optional deps."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mlflow")


def test_both_sources_yield_identical_run_rows(tmp_path):
    """The test that keeps the two readers honest. A run logged to MLflow
    the way _log_mlflow_run logs it, and the same run read off disk, must
    produce equal params and equal metrics -- otherwise `--source` silently
    changes what a curve means."""
    import mlflow
    from charge_experiments.aggregate import (
        read_runs_from_dirs,
        read_runs_from_mlflow,
    )

    config = {"predictor": {"name": "sieve", "params": {"max_wl_depth": 3}}}
    metrics = {"mae": 0.017, "train/mae": 0.009, "r2": 0.98}

    run_dir = tmp_path / "runs" / "exp" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    mlflow.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("exp")
    with mlflow.start_run(run_name="r1"):
        mlflow.log_params(
            {"predictor.name": "sieve", "predictor.params.max_wl_depth": "3"}
        )
        # exactly what _log_mlflow_run does: every key prefixed with "test/"
        mlflow.log_metrics({f"test/{k}": v for k, v in metrics.items()})

    from_dirs = read_runs_from_dirs(tmp_path / "runs")[0]
    from_mlflow = read_runs_from_mlflow(tracking, "exp")[0]

    assert from_mlflow.params == from_dirs.params
    assert from_mlflow.metrics == from_dirs.metrics
