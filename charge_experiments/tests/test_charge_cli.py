"""Argparse-level tests for cli.py -- verifies flags parse and route to the
right handler function, not end-to-end execution (that's
test_charge_nested_runner.py's/test_charge_smoke.py's job)."""

from __future__ import annotations


def test_build_parser_accepts_run_nested_with_all_flags():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run-nested",
            "--config",
            "charge_experiments/configs/dash-nested-charge-example.yaml",
            "--set",
            "predictor.params.max_depth=8",
            "--limit",
            "100",
            "--allow-dirty",
            "--no-tracking",
        ]
    )
    assert args.command == "run-nested"
    assert str(args.config) == (
        "charge_experiments/configs/dash-nested-charge-example.yaml"
    )
    assert args.set == ["predictor.params.max_depth=8"]
    assert args.limit == 100
    assert args.allow_dirty is True
    assert args.no_tracking is True


def test_build_parser_run_nested_defaults():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run-nested", "--config", "some-config.yaml"])
    assert args.set == []
    assert args.limit is None
    assert args.allow_dirty is False
    assert args.no_tracking is False


def test_build_parser_subsample_store_defaults():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["subsample-store", "my-small-store"])
    assert args.dest == "my-small-store"
    assert args.source == "dash-molecules"
    assert args.n_molecules == 50_000
    assert args.conformers_per_molecule == 1
    assert args.seed == 0
    assert args.n_stores == 1


def test_build_parser_subsample_store_accepts_all_flags():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "subsample-store",
            "my-small-store",
            "--source",
            "some-other-store",
            "--n-molecules",
            "1000",
            "--conformers-per-molecule",
            "3",
            "--seed",
            "42",
            "--n-stores",
            "5",
        ]
    )
    assert args.dest == "my-small-store"
    assert args.source == "some-other-store"
    assert args.n_molecules == 1000
    assert args.conformers_per_molecule == 3
    assert args.seed == 42
    assert args.n_stores == 5


def test_build_parser_to_united_atom_defaults():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["to-united-atom", "my-ua-store"])
    assert args.dest == "my-ua-store"
    assert args.source == "dash-molecules"


def test_build_parser_to_united_atom_accepts_source_flag():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["to-united-atom", "my-ua-store", "--source", "some-other-store"]
    )
    assert args.dest == "my-ua-store"
    assert args.source == "some-other-store"


def test_summarize_renders_a_non_finite_metric_as_empty(tmp_path, monkeypatch):
    """metrics.json stores NaN for an undefined r2. The shared reader drops
    non-finite values, so the cell is empty rather than the literal "nan"
    the pre-2026-09 inline loop wrote."""
    import csv
    import json

    from charge_experiments import cli

    run_dir = tmp_path / "runs" / "exp" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {}}))
    (run_dir / "metrics.json").write_text(
        json.dumps({"mae": 0.1, "charge_conservation/r2": float("nan")})
    )
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    assert cli.main(["summarize"]) == 0

    row = next(iter(csv.DictReader((tmp_path / "results" / "summary.csv").open())))
    assert row["mae"] == "0.1"
    assert row["charge_conservation/r2"] == ""


def _sweep_tree(root):
    """Three synthetic runs at depths 1/2/3 -- no store, no MLflow, no real
    experiment."""
    import json

    for i, depth in enumerate([1, 2, 3]):
        run_dir = root / "runs" / "exp" / f"r{i}"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"config": {"predictor": {"params": {"max_wl_depth": depth}}}})
        )
        (run_dir / "metrics.json").write_text(
            json.dumps({"mae": 0.3 / depth, "train/mae": 0.2 / depth})
        )
    return root / "runs"


def test_sweep_writes_a_curve_csv_from_run_dirs(tmp_path, monkeypatch):
    """The CSV holds one raw row per run, and is written whether or not
    matplotlib is installed."""
    import csv

    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    rc = cli.main(
        [
            "sweep",
            "--x",
            "predictor.params.max_wl_depth",
            "--metric",
            "mae",
            "--split",
            "test",
            "--split",
            "train",
        ]
    )
    assert rc == 0

    out = tmp_path / "results" / "max_wl_depth" / "curve.csv"
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    assert {r["predictor.params.max_wl_depth"] for r in rows} == {"1", "2", "3"}
    assert rows[0]["mae"] and rows[0]["train/mae"]


def test_sweep_out_name_defaults_to_the_x_paths_last_segment(tmp_path, monkeypatch):
    """Repeated sweeps over one parameter overwrite one directory rather
    than accumulating timestamped ones."""
    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    assert cli.main(["sweep", "--x", "predictor.params.max_wl_depth"]) == 0
    assert (tmp_path / "results" / "max_wl_depth" / "curve.csv").exists()

    assert cli.main(["sweep", "--x", "predictor.params.max_wl_depth"]) == 0
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["max_wl_depth"]


def test_sweep_reports_no_match_instead_of_writing_an_empty_curve(
    tmp_path, monkeypatch, capsys
):
    """An x path no run carries is a user error, not an empty plot."""
    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    assert cli.main(["sweep", "--x", "predictor.params.nonexistent"]) == 1
    assert "no runs matched" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
