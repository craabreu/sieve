"""``python -m sieve_experiments <command> ...``"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from sieve_experiments.config import load_config
from sieve_experiments.data import DEFAULT_STORES_ROOT
from sieve_experiments.runner import DEFAULT_RUNS_ROOT, DEFAULT_TRACKING_URI, run

SUMMARY_COLUMNS = [
    "run_name",
    "predictor",
    "split_column",
    "scheme",
    "seed",
    "n_test",
    "profile/w1_norm_mean",
    "profile/w1_norm_area_weighted",
    "area/mae",
    "area/r2",
    "charge/mae",
    "charge/max_abs_residual",
    "atom/profile/w1_norm_mean",
    "atom/area/r2",
    "atom/charge/mae",
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


def _cmd_prepare_store(args: argparse.Namespace) -> int:
    from sieve_experiments.prepare_store import prepare_store

    prepare_store(args.store, stores_root=DEFAULT_STORES_ROOT)
    return 0


def _cmd_coarse_grain_store(args: argparse.Namespace) -> int:
    from sieve_experiments.prepare_store import prepare_ua_store

    dest = args.dest or f"{args.source}-ua"
    prepare_ua_store(args.source, dest, stores_root=DEFAULT_STORES_ROOT)
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
            "scheme": manifest.get("data", {}).get("scheme", ""),
            "seed": manifest.get("seed", ""),
            "git_commit": manifest.get("git", {}).get("commit", ""),
            "run_dir": str(run_dir),
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                value = run_metrics.get(key, "")
                row[key] = (
                    ""
                    if value == ""
                    else f"{value:.6g}"
                    if isinstance(value, float)
                    else value
                )
        rows.append(row)

    rows.sort(key=lambda r: (r["split_column"], r["predictor"], r["scheme"], r["seed"]))

    out_path = runs_root.parents[0] / "results" / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sieve_experiments")
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
        "--limit",
        type=int,
        default=None,
        help="use only the first N molecules per split",
    )
    p_run.add_argument(
        "--allow-dirty", action="store_true", help="run with an uncommitted git tree"
    )
    p_run.add_argument(
        "--no-tracking", action="store_true", help="skip MLflow logging for this run"
    )
    p_run.set_defaults(func=_cmd_run)

    p_prepare = sub.add_parser("prepare-store", help="download and split a COSMO store")
    p_prepare.add_argument("store", nargs="?", default="chaos-store")
    p_prepare.set_defaults(func=_cmd_prepare_store)

    p_coarse = sub.add_parser(
        "coarse-grain-store",
        help="build a united-atom store (H merged into their heavy-atom "
        "neighbor) from an already-prepared source store",
    )
    p_coarse.add_argument("source", nargs="?", default="chaos-store")
    p_coarse.add_argument(
        "--dest",
        default=None,
        help="destination store name; default is '<source>-ua'",
    )
    p_coarse.set_defaults(func=_cmd_coarse_grain_store)

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
