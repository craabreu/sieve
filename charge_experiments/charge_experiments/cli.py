"""``python -m charge_experiments <command> ...``"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from charge_experiments.config import load_config
from charge_experiments.data import DEFAULT_STORES_ROOT
from charge_experiments.nested_config import load_nested_config
from charge_experiments.runner import DEFAULT_RUNS_ROOT, DEFAULT_TRACKING_URI, run

SUMMARY_COLUMNS = [
    "run_name",
    "predictor",
    "split_column",
    "seed",
    "n_test_atoms",
    "mae",
    "rmse",
    "r2",
    "charge_conservation/mae",
    "charge_conservation/rmse",
    "charge_conservation/r2",
    "train/mae",
    "train/r2",
    "val/mae",
    "val/r2",
    "time/fit_s",
    "time/predict_s",
    "time/data_s",
    "git_commit",
    "run_dir",
]


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.set)
    tracking = None if args.no_tracking else DEFAULT_TRACKING_URI
    result = run(
        cfg,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
        tracking=tracking,
        limit=args.limit,
    )
    print(f"run written to {result.run_dir}")
    for key in sorted(result.metrics):
        print(f"  {key}: {result.metrics[key]}")
    return 0


def _cmd_run_nested(args: argparse.Namespace) -> int:
    from charge_experiments.nested_runner import (
        DEFAULT_RUNS_ROOT as NESTED_RUNS_ROOT,
    )
    from charge_experiments.nested_runner import (
        DEFAULT_TRACKING_URI as NESTED_TRACKING_URI,
    )
    from charge_experiments.nested_runner import run_nested

    cfg = load_nested_config(args.config, overrides=args.set)
    tracking = None if args.no_tracking else NESTED_TRACKING_URI
    result = run_nested(
        cfg,
        runs_root=NESTED_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
        tracking=tracking,
        limit=args.limit,
    )
    print(f"parent run written to {result.parent.run_dir}")
    for name, child in result.children.items():
        print(f"child run ({name}) written to {child.run_dir}")
    return 0


def _cmd_prepare_store(args: argparse.Namespace) -> int:
    from charge_experiments.prepare_store import prepare_store

    prepare_store(args.store, stores_root=DEFAULT_STORES_ROOT, sdf_path=args.sdf_path)
    return 0


def _cmd_subsample_store(args: argparse.Namespace) -> int:
    from charge_experiments.prepare_store import subsample_store

    summary_text = subsample_store(
        args.source,
        args.dest,
        stores_root=DEFAULT_STORES_ROOT,
        n_molecules=args.n_molecules,
        conformers_per_molecule=args.conformers_per_molecule,
        seed=args.seed,
    )
    print(f"wrote {args.dest!r} (subsampled from {args.source!r}):\n{summary_text}")
    return 0


def _cmd_to_united_atom(args: argparse.Namespace) -> int:
    from charge_experiments.prepare_store import to_united_atom_store

    to_united_atom_store(args.source, args.dest, stores_root=DEFAULT_STORES_ROOT)
    print(f"wrote {args.dest!r} (united-atom version of {args.source!r})")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    del args
    runs_root = DEFAULT_RUNS_ROOT
    rows = []
    for manifest_path in sorted(runs_root.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        metrics_path = run_dir / "metrics.json"
        run_metrics = (
            json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        )
        row = {
            "run_name": manifest.get("run_name", ""),
            "predictor": manifest.get("config", {})
            .get("predictor", {})
            .get("name", ""),
            "split_column": manifest.get("data", {}).get("split_column", ""),
            "seed": manifest.get("seed", ""),
            "git_commit": manifest.get("git", {}).get("commit", ""),
            "run_dir": str(run_dir),
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                value = run_metrics.get(key, "")
                if value == "":
                    row[key] = ""
                elif isinstance(value, float):
                    row[key] = f"{value:.6g}"
                else:
                    row[key] = value
        rows.append(row)

    rows.sort(key=lambda r: (r["split_column"], r["predictor"], r["seed"]))

    out_path = runs_root.parent / "results" / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="charge_experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one experiment from a YAML config")
    p_run.add_argument("--config", required=True, type=Path)
    p_run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key.path=value",
        help="override a config value; may be passed multiple times",
    )
    p_run.add_argument(
        "--limit", type=int, default=None, help="use only the first N conformers"
    )
    p_run.add_argument(
        "--allow-dirty", action="store_true", help="run with an uncommitted git tree"
    )
    p_run.add_argument(
        "--no-tracking", action="store_true", help="skip MLflow logging for this run"
    )
    p_run.set_defaults(func=_cmd_run)

    p_run_nested = sub.add_parser(
        "run-nested",
        help="run one predictor's fit()+save+raw-predict, plus a nested "
        "child run per normalization scheme",
    )
    p_run_nested.add_argument("--config", required=True, type=Path)
    p_run_nested.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key.path=value",
        help="override a config value; may be passed multiple times",
    )
    p_run_nested.add_argument(
        "--limit", type=int, default=None, help="use only the first N conformers"
    )
    p_run_nested.add_argument(
        "--allow-dirty", action="store_true", help="run with an uncommitted git tree"
    )
    p_run_nested.add_argument(
        "--no-tracking", action="store_true", help="skip MLflow logging for this run"
    )
    p_run_nested.set_defaults(func=_cmd_run_nested)

    p_prepare = sub.add_parser(
        "prepare-store", help="download, parse, and split the DASH molecules SDF"
    )
    p_prepare.add_argument("store", nargs="?", default="dash-molecules")
    p_prepare.add_argument(
        "--sdf-path",
        type=Path,
        default=None,
        help="use an already-downloaded SDF instead of downloading a fresh copy",
    )
    p_prepare.set_defaults(func=_cmd_prepare_store)

    p_subsample = sub.add_parser(
        "subsample-store",
        help="build a smaller store by subsampling molecules from an "
        "already-split store, preserving its own split fractions",
    )
    p_subsample.add_argument("dest", help="name of the new, subsampled store")
    p_subsample.add_argument(
        "--source",
        default="dash-molecules",
        help="name of the already-split source store (default: dash-molecules)",
    )
    p_subsample.add_argument(
        "--n-molecules",
        type=int,
        default=50_000,
        help="target total molecule count across all splits (default: 50000)",
    )
    p_subsample.add_argument(
        "--conformers-per-molecule",
        type=int,
        default=1,
        help="max conformers kept per selected molecule (default: 1)",
    )
    p_subsample.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducible sampling"
    )
    p_subsample.set_defaults(func=_cmd_subsample_store)

    p_ua = sub.add_parser(
        "to-united-atom",
        help="build a united-atom (hydrogens removed, folded into their "
        "heavy-atom neighbor's charge) version of an already-prepared store",
    )
    p_ua.add_argument("dest", help="name of the new, united-atom store")
    p_ua.add_argument(
        "--source",
        default="dash-molecules",
        help="name of the already-prepared source store (default: dash-molecules)",
    )
    p_ua.set_defaults(func=_cmd_to_united_atom)

    p_summary = sub.add_parser(
        "summarize", help="collect runs/**/metrics.json into a CSV"
    )
    p_summary.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
