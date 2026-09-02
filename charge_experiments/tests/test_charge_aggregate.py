"""Reader/aggregation tests for the sweep command. Never touches the real
store or the real runs tree -- every fixture is a synthetic tmp_path run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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


def _rows(triples):
    """(max_wl_depth, mae, train_mae) -> RunRows."""
    from charge_experiments.aggregate import RunRow

    return [
        RunRow(
            run_dir=f"r{i}",
            params={"predictor.params.max_wl_depth": str(d)},
            metrics={"mae": m, "train/mae": t},
            meta={},
        )
        for i, (d, m, t) in enumerate(triples)
    ]


def test_split_names_map_to_metric_key_prefixes():
    """Test metrics are UNPREFIXED in metrics.json (`mae`); every other
    split is prefixed (`train/mae`). Series selection has to know that."""
    from charge_experiments.aggregate import SPLIT_PREFIX

    assert SPLIT_PREFIX["test"] == ""
    assert SPLIT_PREFIX["train"] == "train/"
    assert SPLIT_PREFIX["train_loo"] == "train_loo/"


def test_build_curve_aggregates_repeated_x_to_mean_and_min_max():
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(1, 0.10, 0.05), (1, 0.20, 0.05), (2, 0.05, 0.01)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    points = table.series[("test", "mae")]
    assert [p.x_label for p in points] == ["1", "2"]
    assert points[0].mean == pytest.approx(0.15)
    assert (points[0].lo, points[0].hi) == pytest.approx((0.10, 0.20))
    assert points[0].std == pytest.approx(0.05)  # population std of [0.10, 0.20]
    assert points[0].n_runs == 2
    assert points[1].n_runs == 1
    assert points[1].std == 0.0  # a single value has no dispersion


def test_build_curve_sorts_numeric_x_numerically_not_lexically():
    """Depths 2 and 10 must not sort as "10" < "2"."""
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(10, 0.01, 0.0), (2, 0.05, 0.0)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    assert [p.x_label for p in table.series[("test", "mae")]] == ["2", "10"]


def test_build_curve_drops_a_run_missing_the_metric():
    """A run without the requested key contributes no point -- it is never
    coerced to 0, which would plot as a real and excellent value."""
    from charge_experiments.aggregate import RunRow, build_curve

    rows = [
        RunRow("r0", {"d": "1"}, {"mae": 0.1}, {}),
        RunRow("r1", {"d": "1"}, {}, {}),  # crashed before scoring
    ]
    table = build_curve(rows, x="d", metrics=["mae"], splits=["test"])
    assert table.series[("test", "mae")][0].n_runs == 1


def test_build_curve_groups_by_a_second_parameter():
    from charge_experiments.aggregate import RunRow, build_curve

    rows = [
        RunRow("r0", {"d": "1", "n": "a"}, {"mae": 0.1}, {}),
        RunRow("r1", {"d": "1", "n": "b"}, {"mae": 0.2}, {}),
    ]
    table = build_curve(rows, x="d", metrics=["mae"], splits=["test"], group_by="n")
    assert set(table.series) == {("test n=a", "mae"), ("test n=b", "mae")}


def test_raw_rows_hold_every_run_not_the_aggregate():
    """The CSV must record what was measured; the band is never the only
    record."""
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(1, 0.10, 0.05), (1, 0.20, 0.05)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    assert len(table.raw_rows) == 2
    assert {r["mae"] for r in table.raw_rows} == {"0.1", "0.2"}


def test_aggregate_rows_persists_the_mean_min_max_n_runs_band():
    """The pooled stat build_curve computes but raw_rows never carries --
    persisted separately, not instead of raw_rows (see the test above)."""
    from charge_experiments.aggregate import aggregate_rows, build_curve

    table = build_curve(
        _rows([(1, 0.10, 0.05), (1, 0.20, 0.05), (2, 0.05, 0.01)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    rows = aggregate_rows(table)
    assert len(rows) == 2

    depth_1 = next(r for r in rows if r["x_label"] == "1")
    assert depth_1["series"] == "test"
    assert depth_1["metric"] == "mae"
    assert depth_1["x"] == "predictor.params.max_wl_depth"
    assert depth_1["mean"] == "0.15"
    assert depth_1["lo"] == "0.1"
    assert depth_1["hi"] == "0.2"
    assert depth_1["std"] == "0.05"
    assert depth_1["n_runs"] == "2"

    depth_2 = next(r for r in rows if r["x_label"] == "2")
    assert depth_2["n_runs"] == "1"
    assert depth_2["mean"] == depth_2["lo"] == depth_2["hi"] == "0.05"


def test_aggregate_rows_pools_every_run_sharing_one_x_value():
    """Five runs of the same predictor across five different stores (no
    per-store distinction on the x axis) pool into a single point with
    n_runs=5 -- the exact "pool the 5 store runs" use case."""
    from charge_experiments.aggregate import RunRow, aggregate_rows, build_curve

    rows = [
        RunRow(f"r{i}", {"predictor.name": "dash"}, {"mae": m}, {})
        for i, m in enumerate([0.0188, 0.0191, 0.0192, 0.0189, 0.0188])
    ]
    table = build_curve(rows, x="predictor.name", metrics=["mae"], splits=["test"])
    agg = aggregate_rows(table)
    assert len(agg) == 1
    assert agg[0]["n_runs"] == "5"
    assert agg[0]["lo"] == "0.0188"
    assert agg[0]["hi"] == "0.0192"
