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
    args = parser.parse_args(
        ["run-nested", "--config", "some-config.yaml"]
    )
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


def test_build_parser_subsample_store_accepts_all_flags():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "subsample-store", "my-small-store",
            "--source", "some-other-store",
            "--n-molecules", "1000",
            "--conformers-per-molecule", "3",
            "--seed", "42",
        ]
    )
    assert args.dest == "my-small-store"
    assert args.source == "some-other-store"
    assert args.n_molecules == 1000
    assert args.conformers_per_molecule == 3
    assert args.seed == 42


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
