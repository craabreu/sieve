"""Tests for ``sieve_experiments summarize`` (cli.py's ``_cmd_summarize``):
collects every ``runs/<experiment>/<run_name>/manifest.json`` +
``metrics.json`` pair into one ``results/summary.csv``. Pure filesystem/CSV
work -- no real run needed, so a fake ``runs_root`` with hand-built
manifest/metrics files is enough.
"""

from __future__ import annotations

import argparse
import csv
import json

from sieve_experiments import cli


def _argparse_namespace() -> argparse.Namespace:
    """``_cmd_summarize`` ignores its args entirely (``del args``); any
    object works."""
    return argparse.Namespace()


def _write_run(
    runs_root,
    *,
    experiment: str,
    run_name: str,
    predictor: str,
    split_column: str,
    scheme: str = "cosmo-sac-2010",
    seed: int = 0,
    commit: str = "deadbeef",
    metrics: dict | None = None,
) -> None:
    run_dir = runs_root / experiment / run_name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_name": run_name,
        "seed": seed,
        "git": {"commit": commit},
        "data": {"split_column": split_column, "scheme": scheme},
        "config": {"predictor": {"name": predictor}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics))


def _read_summary_rows(runs_root):
    out_path = runs_root.parent / "results" / "summary.csv"
    with out_path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_summarize_collects_every_run_into_one_csv(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", runs_root)

    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="dash-biased_split__a",
        predictor="dash",
        split_column="biased_split",
        metrics={
            "n_test": 5333,
            "profile/w1_norm_mean": 0.407123,
            "area/r2": 0.952,
            "charge/mae": 0.0922,
            "atom/profile/w1_norm_mean": 1.012,
            "time/fit_s": 93.5,
        },
    )
    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="global_mean-random__b",
        predictor="global_mean",
        split_column="split",
        metrics={"n_test": 5333, "profile/w1_norm_mean": 1.024},
    )

    exit_code = cli._cmd_summarize(_argparse_namespace())

    assert exit_code == 0
    rows = _read_summary_rows(runs_root)
    assert len(rows) == 2
    assert {r["predictor"] for r in rows} == {"dash", "global_mean"}


def test_summarize_writes_every_summary_column_including_atom_and_timing_metrics(
    tmp_path, monkeypatch
):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", runs_root)

    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="dash-biased_split__a",
        predictor="dash",
        split_column="biased_split",
        metrics={
            "n_test": 5333,
            "profile/w1_norm_mean": 0.407123,
            "profile/w1_norm_area_weighted": 0.40,
            "area/mae": 8.0,
            "area/r2": 0.952,
            "charge/mae": 0.0922,
            "charge/max_abs_residual": 0.39,
            "atom/profile/w1_norm_mean": 1.012,
            "atom/area/r2": 0.944,
            "atom/charge/mae": 0.00757,
            "time/fit_s": 93.5,
            "time/predict_s": 16.2,
            "time/data_s": 0.4,
        },
    )

    cli._cmd_summarize(_argparse_namespace())

    rows = _read_summary_rows(runs_root)
    assert set(rows[0]) == set(cli.SUMMARY_COLUMNS)
    row = rows[0]
    assert row["run_dir"] == str(runs_root / "milestone1" / "dash-biased_split__a")
    assert row["git_commit"] == "deadbeef"
    # floats are formatted to 6 significant figures, not repr'd raw
    assert row["profile/w1_norm_mean"] == "0.407123"
    assert row["atom/profile/w1_norm_mean"] == "1.012"
    assert row["time/fit_s"] == "93.5"


def test_summarize_leaves_missing_metrics_blank_rather_than_raising(
    tmp_path, monkeypatch
):
    """A run without a metrics.json at all (e.g. a crashed run) still gets a
    row -- every SUMMARY_COLUMNS metric column just comes out empty, not a
    KeyError."""
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", runs_root)

    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="crashed__a",
        predictor="dash",
        split_column="biased_split",
        metrics=None,
    )

    exit_code = cli._cmd_summarize(_argparse_namespace())

    assert exit_code == 0
    rows = _read_summary_rows(runs_root)
    assert len(rows) == 1
    assert rows[0]["profile/w1_norm_mean"] == ""
    assert rows[0]["time/fit_s"] == ""


def test_summarize_sorts_by_split_column_predictor_scheme_seed(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", runs_root)

    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="z-run",
        predictor="global_mean",
        split_column="split",
        metrics={},
    )
    _write_run(
        runs_root,
        experiment="milestone1",
        run_name="a-run",
        predictor="dash",
        split_column="biased_split",
        metrics={},
    )

    cli._cmd_summarize(_argparse_namespace())

    rows = _read_summary_rows(runs_root)
    assert [r["split_column"] for r in rows] == ["biased_split", "split"]
