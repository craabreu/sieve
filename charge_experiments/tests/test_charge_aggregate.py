"""Reader/aggregation tests for the sweep command. Never touches the real
store or the real runs tree -- every fixture is a synthetic tmp_path run."""

from __future__ import annotations

import json
from pathlib import Path


def write_run(root: Path, experiment: str, name: str, *, params, metrics) -> Path:
    """A minimal run directory: manifest.json (carrying the resolved config
    that params are read from) plus metrics.json."""
    run_dir = root / experiment / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": params}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def test_read_runs_from_dirs_flattens_config_to_dotted_params(tmp_path):
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(
        tmp_path,
        "exp",
        "r1",
        params={"predictor": {"name": "sieve", "params": {"max_wl_depth": 3}}},
        metrics={"mae": 0.017, "train/mae": 0.009},
    )
    rows = read_runs_from_dirs(tmp_path)
    assert len(rows) == 1
    assert rows[0].params["predictor.params.max_wl_depth"] == "3"
    assert rows[0].params["predictor.name"] == "sieve"
    assert rows[0].metrics["mae"] == 0.017
    assert rows[0].metrics["train/mae"] == 0.009


def test_read_runs_from_dirs_filters_by_experiment(tmp_path):
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(tmp_path, "a", "r1", params={}, metrics={"mae": 1.0})
    write_run(tmp_path, "b", "r1", params={}, metrics={"mae": 2.0})
    assert len(read_runs_from_dirs(tmp_path)) == 2
    assert [r.metrics["mae"] for r in read_runs_from_dirs(tmp_path, "a")] == [1.0]


def test_read_runs_from_dirs_keeps_a_run_with_no_metrics(tmp_path):
    """A run with a manifest but no metrics.json (crashed early, or a
    `describe` run) is returned with empty metrics, not dropped --
    _cmd_summarize still lists it, and build_curve contributes no point for
    a run missing the requested metric anyway."""
    from charge_experiments.aggregate import read_runs_from_dirs

    run_dir = tmp_path / "exp" / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {}}))
    rows = read_runs_from_dirs(tmp_path)
    assert len(rows) == 1
    assert rows[0].metrics == {}


def test_nan_metrics_are_dropped_not_read_as_values(tmp_path):
    """metrics.json stores NaN for an undefined r2 (json.dumps writes bare
    NaN, which json.loads accepts). A NaN must not survive into a plotted
    point."""
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(
        tmp_path, "exp", "r1", params={}, metrics={"mae": 0.1, "r2": float("nan")}
    )
    row = read_runs_from_dirs(tmp_path)[0]
    assert row.metrics["mae"] == 0.1
    assert "r2" not in row.metrics


def test_mlflow_metric_names_normalize_to_the_run_dir_spelling():
    """_log_mlflow_run prefixes EVERY metric with "test/", so run-dir `mae`
    is MLflow `test/mae` and run-dir `train/mae` is MLflow `test/train/mae`.
    Exactly one leading `test/` is stripped -- stripping greedily would turn
    a hypothetical `test/test/x` into `x`, and stripping none would make the
    two sources disagree about what a curve means."""
    from charge_experiments.aggregate import normalize_mlflow_metric_name as norm

    assert norm("test/mae") == "mae"
    assert norm("test/train/mae") == "train/mae"
    assert norm("test/train_loo/r2") == "train_loo/r2"
    assert norm("test/charge_conservation/mae") == "charge_conservation/mae"
    assert norm("mae") == "mae"  # already normalized: left alone
