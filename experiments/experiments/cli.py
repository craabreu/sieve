"""``python -m experiments <command> ...``"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from experiments.config import load_config
from experiments.data import DEFAULT_STORES_ROOT
from experiments.runner import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RUNS_ROOT,
    DEFAULT_TRACKING_URI,
    run,
)

SUMMARY_COLUMNS = [
    "run_name",
    "predictor",
    "split_column",
    "seed",
    "n_test_atoms",
    "mae",
    "rmse",
    "r2",
    "sum_constraint/mae",
    "sum_constraint/rmse",
    "sum_constraint/r2",
    "train/mae",
    "train/r2",
    "val/mae",
    "val/r2",
    "time/fit_s",
    "time/predict_s",
    "time/featurize_s",
    "time/data_s",
    "git_commit",
    "run_dir",
]


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.set)
    tracking = DEFAULT_TRACKING_URI if args.track else None
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


def _cmd_dash_depth_sweep(args: argparse.Namespace) -> int:
    depths = [int(d) for d in args.depths.split(",")]
    if args.store is not None:
        if args.fold is not None:
            raise SystemExit("--store and --fold are mutually exclusive")
        from experiments.dash_depth_sweep import run_store

        results = run_store(
            config_path=args.config,
            store=args.store,
            depths=depths,
            experiment=args.experiment,
            label=args.label,
            runs_root=DEFAULT_RUNS_ROOT,
            allow_dirty=args.allow_dirty,
            limit=args.limit,
        )
    elif args.fold is not None:
        from experiments.dash_depth_sweep import run_fold

        results = run_fold(
            config_path=args.config,
            store=f"{args.store_prefix}-{args.fold}",
            depths=depths,
            experiment=args.experiment,
            fold=args.fold,
            runs_root=DEFAULT_RUNS_ROOT,
            allow_dirty=args.allow_dirty,
            limit=args.limit,
        )
    else:
        from experiments.dash_depth_sweep import run_sweep

        results = run_sweep(
            config_path=args.config,
            store_prefix=args.store_prefix,
            n_folds=args.n_folds,
            depths=depths,
            experiment=args.experiment,
            runs_root=DEFAULT_RUNS_ROOT,
            allow_dirty=args.allow_dirty,
            limit=args.limit,
        )
    if not results:
        print("nothing to do -- every fold already complete for every depth")
        return 0
    for result in results:
        print(f"{result.run_dir}: mae={result.metrics.get('mae')}")
    return 0


def _cmd_merge_shards(args: argparse.Namespace) -> int:
    from experiments.dash_depth_sweep import merge_fold_shards

    out = merge_fold_shards(
        from_experiment=args.from_experiment,
        depth=args.depth,
        n_folds=args.n_folds,
        out_path=args.out,
        runs_root=DEFAULT_RUNS_ROOT,
    )
    print(f"merged shard: {out}")
    return 0


def _cmd_promote_run(args: argparse.Namespace) -> int:
    from experiments.runner import promote_run

    rc = 0
    for run_dir in args.run_dir:
        try:
            result = promote_run(
                run_dir,
                target_experiment=args.to,
                runs_root=DEFAULT_RUNS_ROOT,
                tracking=DEFAULT_TRACKING_URI,
                artifact_root=DEFAULT_ARTIFACT_ROOT,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"skipped {run_dir!r}: {exc}")
            rc = 1
            continue
        if not result.logged:
            print(f"{run_dir!r} already tracked as {result.run_dir} -- skipped")
            continue
        moved_note = f" (moved to {result.run_dir})" if result.moved else ""
        print(f"promoted {run_dir!r} -> experiment {result.experiment!r}{moved_note}")
    return rc


def _cmd_prepare_store(args: argparse.Namespace) -> int:
    from experiments.prepare_dash import prepare_store

    prepare_store(
        args.store,
        stores_root=DEFAULT_STORES_ROOT,
        sdf_path=args.sdf_path,
        n_shards=args.n_shards,
    )
    return 0


def _cmd_cluster_report(args: argparse.Namespace) -> int:
    from experiments.prepare_dash import cluster_size_report

    candidates = tuple(int(n) for n in args.candidates.split(","))
    report = cluster_size_report(
        DEFAULT_STORES_ROOT / args.store,
        train=args.train,
        test=args.test,
        candidate_n_shards=candidates,
    )
    print(report)
    return 0


def _cmd_build_sieve_codes(args: argparse.Namespace) -> int:
    from experiments.config import TargetCfg
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
        save_codes,
    )
    from experiments.runner import load_molecule_set

    attributes = (
        tuple(args.attributes.split(","))
        if args.attributes is not None
        else DEFAULT_ATTRIBUTES
    )
    edge_attributes = (
        tuple(a for a in args.edge_attributes.split(",") if a)
        if args.edge_attributes is not None
        else ("bond_type",)
    )

    mset, masks = load_molecule_set(
        args.store,
        target=TargetCfg(atom_property=args.atom_property),
        split_column=args.split_column,
        splits=(args.train_split,),
    )
    train = mset.select(masks[args.train_split])

    config = _build_config(
        train.mols,
        attributes=attributes,
        edge_attributes=edge_attributes,
        target_dim=1,
        max_wl_depth=0,  # irrelevant to code discovery; SieveConfig needs one
        minimum_support=1,
        shrinkage_strength=None,
    )
    save_codes(config.attribute_codes, config.edge_codes, args.out)
    print(f"wrote codes: {args.out}")
    return 0


def _cmd_merge_states(args: argparse.Namespace) -> int:
    from experiments.predictors import build as build_predictor

    predictor = build_predictor(args.predictor, {})
    merge_states = getattr(predictor, "merge_states", None)
    if merge_states is None:
        raise SystemExit(f"predictor {args.predictor!r} has no merge_states method")
    merge_states(args.shard, args.out)
    print(f"merged {len(args.shard)} shard(s) -> {args.out}")
    return 0


def _cmd_subsample_store(args: argparse.Namespace) -> int:
    from experiments.store_ops import subsample_store

    result = subsample_store(
        args.source,
        args.dest,
        stores_root=DEFAULT_STORES_ROOT,
        n_molecules=args.n_molecules,
        conformers_per_molecule=args.conformers_per_molecule,
        seed=args.seed,
        n_stores=args.n_stores,
    )
    if args.n_stores == 1:
        names, summaries = [args.dest], [result]
    else:
        names = [f"{args.dest}-{i + 1}" for i in range(args.n_stores)]
        summaries = result
    for name, summary_text in zip(names, summaries, strict=True):
        print(f"wrote {name!r} (subsampled from {args.source!r}):\n{summary_text}")
    return 0


def _cmd_partition_store(args: argparse.Namespace) -> int:
    from experiments.store_ops import partition_store

    result = partition_store(
        args.source,
        args.dest,
        stores_root=DEFAULT_STORES_ROOT,
        n_stores=args.n_stores,
        conformers_per_molecule=args.conformers_per_molecule,
        seed=args.seed,
    )
    if args.n_stores == 1:
        names, summaries = [args.dest], [result]
    else:
        names = [f"{args.dest}-{i + 1}" for i in range(args.n_stores)]
        summaries = result
    for name, summary_text in zip(names, summaries, strict=True):
        print(f"wrote {name!r} (partitioned from {args.source!r}):\n{summary_text}")
    return 0


def _cmd_to_united_atom(args: argparse.Namespace) -> int:
    from experiments.store_ops import to_united_atom_store

    to_united_atom_store(
        args.source,
        args.dest,
        stores_root=DEFAULT_STORES_ROOT,
        atom_property=args.atom_property,
    )
    print(f"wrote {args.dest!r} (united-atom version of {args.source!r})")
    return 0


def _cmd_cv_fit_dash_shards(args: argparse.Namespace) -> int:
    from experiments.cv import run_dash_shard_fits

    paths = run_dash_shard_fits(
        store=args.store,
        n_shards=args.n_shards,
        max_depth=args.max_depth,
        seed=args.seed,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
    )
    for p in paths:
        print(p)
    return 0


def _cmd_cv_fit_sieve_shards(args: argparse.Namespace) -> int:
    import json as _json

    from experiments.cv import run_sieve_shard_fits

    depths = [int(d) for d in args.depths.split(",")]
    predictor_params = (
        _json.loads(args.predictor_params) if args.predictor_params else {}
    )
    result = run_sieve_shard_fits(
        store=args.store,
        n_shards=args.n_shards,
        depths=depths,
        codes_path=args.codes_path,
        config_label=args.config_label,
        predictor_params=predictor_params,
        seed=args.seed,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
    )
    for depth, paths in result.items():
        for p in paths:
            print(f"w{depth}: {p}")
    return 0


def _cmd_cv_run_dash(args: argparse.Namespace) -> int:
    from experiments.cv import run_dash_cv

    depths = [int(d) for d in args.depths.split(",")]
    repeats = [int(r) for r in args.repeats.split(",")]
    results = run_dash_cv(
        store=args.store,
        n_shards=args.n_shards,
        depths=depths,
        repeats=repeats,
        k=args.k,
        max_depth=args.max_depth,
        normalization=args.normalization,
        method=args.method,
        experiment=args.experiment,
        seed=args.seed,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
    )
    if not results:
        print("nothing to do -- every (repeat, fold, depth) already done")
        return 0
    for r in results:
        print(f"{r.run_dir}: mae={r.metrics.get('mae')}")
    return 0


def _cmd_cv_run_sieve(args: argparse.Namespace) -> int:
    import json as _json

    from experiments.cv import run_sieve_cv

    depths = [int(d) for d in args.depths.split(",")]
    repeats = [int(r) for r in args.repeats.split(",")]
    predictor_params = (
        _json.loads(args.predictor_params) if args.predictor_params else {}
    )
    results = run_sieve_cv(
        store=args.store,
        n_shards=args.n_shards,
        depths=depths,
        repeats=repeats,
        codes_path=args.codes_path,
        config_label=args.config_label,
        predictor_params=predictor_params,
        k=args.k,
        normalization=args.normalization,
        method=args.method,
        experiment=args.experiment,
        seed=args.seed,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
    )
    if not results:
        print("nothing to do -- every (repeat, fold, depth) already done")
        return 0
    for r in results:
        print(f"{r.run_dir}: mae={r.metrics.get('mae')}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import json as _json

    from experiments.compare import read_cv_table, repeated_measures_anova, tukey_hsd

    experiments_ = args.experiment
    depth_by_method = (
        _json.loads(args.depth_by_method) if args.depth_by_method else None
    )

    methods, table = read_cv_table(
        DEFAULT_RUNS_ROOT,
        experiments_,
        depth_by_method=depth_by_method,
        metric=args.metric,
    )
    anova = repeated_measures_anova(methods, table)
    print(
        f"ANOVA: F({anova.df_method},{anova.df_error})={anova.f_stat:.4g} "
        f"p={anova.p_value:.4g}"
    )
    for m in methods:
        print(f"  mean {args.metric}[{m}] = {anova.method_means[m]:.6g}")

    comparisons = tukey_hsd(methods, table, alpha=args.alpha, anova=anova)
    for c in comparisons:
        print(
            f"  {c.a} - {c.b}: diff={c.diff:.6g} "
            f"CI=[{c.ci_lo:.6g}, {c.ci_hi:.6g}] p={c.p_value:.4g}"
        )

    if args.out is not None:
        from experiments.compare import write_tukey_plot

        provenance = f"store(s): {', '.join(experiments_)}; n={table.shape[0]}"
        write_tukey_plot(
            comparisons,
            args.out,
            title=f"Tukey HSD ({args.metric})",
            metric_label=f"{args.metric} difference",
            provenance=provenance,
        )
        print(f"wrote {args.out}")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    del args
    from experiments.aggregate import read_runs_from_dirs

    runs_root = DEFAULT_RUNS_ROOT
    rows = []
    for run_row in read_runs_from_dirs(runs_root):
        row = {
            "run_name": run_row.meta["run_name"],
            "predictor": run_row.params.get("predictor.name", ""),
            "split_column": run_row.meta["split_column"],
            "seed": run_row.meta["seed"],
            "git_commit": run_row.meta["git_commit"],
            "run_dir": run_row.run_dir,
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                # read_runs_from_dirs drops non-finite metrics, so a NaN r2
                # comes through as absent -> "" (matching a genuinely missing
                # metric). The pre-2026-09 inline loop wrote the literal "nan"
                # here instead.
                value = run_row.metrics.get(key, "")
                row[key] = f"{value:.6g}" if isinstance(value, float) else value
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


def _cmd_sweep(args: argparse.Namespace) -> int:
    from experiments.aggregate import (
        AGGREGATE_FIELDNAMES,
        aggregate_rows,
        build_curve,
        markdown_report,
        read_runs_from_dirs,
        read_runs_from_mlflow,
    )

    args.metric = args.metric or ["mae", "rmse", "r2"]
    args.split = args.split or ["test", "train"]

    if args.source == "mlflow":
        if not args.experiment:
            raise SystemExit("--source mlflow requires --experiment")
        rows = read_runs_from_mlflow(args.tracking, args.experiment)
    else:
        rows = read_runs_from_dirs(DEFAULT_RUNS_ROOT, args.experiment)

    table = build_curve(
        rows,
        x=args.x,
        metrics=args.metric,
        splits=args.split,
        group_by=args.group_by,
    )
    if not table.raw_rows:
        print(f"no runs matched {args.x!r}; nothing written")
        return 1

    name = args.out or args.x.rsplit(".", 1)[-1]
    out_dir = DEFAULT_RUNS_ROOT.parent / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in table.raw_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = out_dir / "curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(table.raw_rows)
    print(f"wrote {len(table.raw_rows)} row(s) to {csv_path}")

    agg_rows = aggregate_rows(table)
    agg_path = out_dir / "aggregate.csv"
    with agg_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"wrote {len(agg_rows)} row(s) to {agg_path}")

    report_path = out_dir / "report.md"
    report_path.write_text(markdown_report(table))
    print(f"wrote {report_path}")

    try:
        from experiments.plots import curve_panel

        curve_panel(table, out_dir / "curve.png", suptitle=args.x, band=args.band)
        print(f"wrote {out_dir / 'curve.png'}")
    except ImportError:
        logging.getLogger("experiments").warning(
            "matplotlib not installed; wrote curve.csv only"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments")
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
        "--track",
        action="store_true",
        help="log this run to MLflow (default: off -- see runner.execute's "
        "own docstring for why)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_dds = sub.add_parser(
        "dash-depth-sweep",
        help="sweep the dash predictor's max_depth across a range of "
        "already-partitioned fold stores, one fit + one tree-matching "
        "walk per fold (see experiments.dash_depth_sweep's own docstring "
        "for why this is far cheaper than one independent run per "
        "(depth, fold) pair)",
    )
    p_dds.add_argument(
        "--config", required=True, type=Path, help="a dash predictor config"
    )
    p_dds.add_argument(
        "--store-prefix",
        default="dash-molecules-10fold",
        help="fold stores are named <prefix>-1 .. <prefix>-n-folds "
        "(default: dash-molecules-10fold)",
    )
    p_dds.add_argument(
        "--n-folds", type=int, default=10, help="number of fold stores (default: 10)"
    )
    p_dds.add_argument(
        "--fold",
        type=int,
        default=None,
        help="sweep only this one fold (1-indexed) instead of 1..n-folds -- "
        "for dispatching folds to separate, parallel processes",
    )
    p_dds.add_argument(
        "--store",
        default=None,
        help="sweep this one store instead of a fold partition -- e.g. the "
        "whole corpus (dash-molecules). Mutually exclusive with --fold; "
        "--store-prefix/--n-folds are ignored when it is given",
    )
    p_dds.add_argument(
        "--label",
        default="full",
        help="run.batch_id suffix for a --store sweep, giving d<depth>-<label> "
        "(default: full); ignored for fold sweeps, which label by fold",
    )
    p_dds.add_argument(
        "--depths",
        default="1,2,4,6,8,10,12,14,16",
        help="comma-separated max_depth values to sweep "
        "(default: 1,2,4,6,8,10,12,14,16)",
    )
    p_dds.add_argument(
        "--experiment",
        default="dash-depth-sweep",
        help="run.experiment for every written run (default: dash-depth-sweep)",
    )
    p_dds.add_argument(
        "--limit", type=int, default=None, help="use only the first N conformers"
    )
    p_dds.add_argument(
        "--allow-dirty", action="store_true", help="run with an uncommitted git tree"
    )
    p_dds.set_defaults(func=_cmd_dash_depth_sweep)

    p_merge = sub.add_parser(
        "merge-shards",
        help="merge a depth sweep's per-fold tree_stats.npz shards (all "
        "saved at one depth) into a single node-stats artifact -- "
        "fold_node_stats, exact, no re-fit",
    )
    p_merge.add_argument(
        "--from-experiment",
        default="dash-depth-sweep",
        help="experiment whose per-fold shards to merge (default: dash-depth-sweep)",
    )
    p_merge.add_argument(
        "--depth",
        type=int,
        required=True,
        help="the max_depth the shards were saved at",
    )
    p_merge.add_argument(
        "--n-folds", type=int, default=10, help="number of fold shards (default: 10)"
    )
    p_merge.add_argument(
        "--out", type=Path, required=True, help="output .npz path for the merged shard"
    )
    p_merge.set_defaults(func=_cmd_merge_shards)

    p_promote = sub.add_parser(
        "promote-run",
        help="give an already-completed run directory an MLflow record, "
        "optionally moving it to a different experiment",
    )
    p_promote.add_argument(
        "run_dir", nargs="+", help="path(s) to a completed run directory"
    )
    p_promote.add_argument(
        "--to",
        dest="to",
        default=None,
        metavar="EXPERIMENT",
        help="re-log (and move runs/<EXPERIMENT>/<name>) under this "
        "experiment instead of the run's own config.run.experiment",
    )
    p_promote.set_defaults(func=_cmd_promote_run)

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
    p_prepare.add_argument(
        "--n-shards",
        type=int,
        default=25,
        help="cluster-clean train shards for CV (s00..s{n-1}); choose via "
        "cluster-report first (default: 25)",
    )
    p_prepare.set_defaults(func=_cmd_prepare_store)

    p_cluster_report = sub.add_parser(
        "cluster-report",
        help="report train-split cluster sizes and achieved shard balance "
        "for candidate n_shards values, to choose one before splitting",
    )
    p_cluster_report.add_argument("store", nargs="?", default="dash-molecules")
    p_cluster_report.add_argument("--train", type=float, default=0.9)
    p_cluster_report.add_argument("--test", type=float, default=0.1)
    p_cluster_report.add_argument(
        "--candidates",
        default="10,25,50,100",
        help="comma-separated candidate n_shards values (default: 10,25,50,100)",
    )
    p_cluster_report.set_defaults(func=_cmd_cluster_report)

    p_build_codes = sub.add_parser(
        "build-sieve-codes",
        help="freeze a Sieve attribute/edge vocabulary over a store's whole "
        "train split, for CV shard fits to share (sieve_predictor.SievePredictor's "
        "own codes_path) -- without this, shards fit on disjoint molecule "
        "sets can discover different vocabularies and refuse to merge",
    )
    p_build_codes.add_argument("store", nargs="?", default="dash-molecules")
    p_build_codes.add_argument("--split-column", default="split")
    p_build_codes.add_argument("--train-split", default="train")
    p_build_codes.add_argument("--atom-property", default="MBIScharge")
    p_build_codes.add_argument(
        "--attributes",
        default=None,
        help="comma-separated attribute names (default: sieve_predictor's "
        "own DEFAULT_ATTRIBUTES)",
    )
    p_build_codes.add_argument(
        "--edge-attributes",
        default=None,
        help="comma-separated edge attribute names, empty string for none "
        "(default: bond_type)",
    )
    p_build_codes.add_argument("--out", required=True, type=Path)
    p_build_codes.set_defaults(func=_cmd_build_sieve_codes)

    p_merge_states = sub.add_parser(
        "merge-states",
        help="merge N saved predictor model-state shards (tree_stats.npz) "
        "into one, exact, no re-fit -- generalizes merge-shards to either "
        "predictor via its own merge_states",
    )
    p_merge_states.add_argument(
        "--predictor", required=True, choices=("dash", "sieve"), help="which predictor"
    )
    p_merge_states.add_argument(
        "shard", nargs="+", type=Path, help="shard tree_stats.npz path(s) to merge"
    )
    p_merge_states.add_argument("--out", required=True, type=Path)
    p_merge_states.set_defaults(func=_cmd_merge_states)

    p_cv_fit_dash = sub.add_parser(
        "cv-fit-dash-shards",
        help="fit DASH on each of a store's s00..s{n-1} shards, predicting "
        "nothing (a shard is only ever used merged)",
    )
    p_cv_fit_dash.add_argument("store", nargs="?", default="dash-molecules")
    p_cv_fit_dash.add_argument("--n-shards", type=int, required=True)
    p_cv_fit_dash.add_argument(
        "--max-depth",
        type=int,
        required=True,
        help="the deepest depth the CV sweep will need -- one fit serves "
        "every shallower depth (node stats are depth-invariant)",
    )
    p_cv_fit_dash.add_argument("--seed", type=int, default=0)
    p_cv_fit_dash.add_argument("--allow-dirty", action="store_true")
    p_cv_fit_dash.set_defaults(func=_cmd_cv_fit_dash_shards)

    p_cv_fit_sieve = sub.add_parser(
        "cv-fit-sieve-shards",
        help="fit Sieve on each of a store's s00..s{n-1} shards, one shard "
        "set per depth (continuation's estimate depends on its own "
        "deepest level, so shallow depths are not derivable from a deep fit)",
    )
    p_cv_fit_sieve.add_argument("store", nargs="?", default="dash-molecules")
    p_cv_fit_sieve.add_argument("--n-shards", type=int, required=True)
    p_cv_fit_sieve.add_argument(
        "--depths", required=True, help="comma-separated max_wl_depth values"
    )
    p_cv_fit_sieve.add_argument(
        "--codes-path", required=True, type=Path, help="from build-sieve-codes"
    )
    p_cv_fit_sieve.add_argument(
        "--config-label",
        required=True,
        help="names this Sieve configuration's own shards (e.g. element-eb)",
    )
    p_cv_fit_sieve.add_argument(
        "--predictor-params",
        default=None,
        help="JSON object of SievePredictor kwargs other than max_wl_depth/"
        "codes_path (e.g. attributes, class_estimator, shrinkage_weight)",
    )
    p_cv_fit_sieve.add_argument("--seed", type=int, default=0)
    p_cv_fit_sieve.add_argument("--allow-dirty", action="store_true")
    p_cv_fit_sieve.set_defaults(func=_cmd_cv_fit_sieve_shards)

    p_cv_run_dash = sub.add_parser(
        "cv-run-dash",
        help="assemble each CV sample's training model by merging DASH's "
        "own shards, evaluate at every requested depth",
    )
    p_cv_run_dash.add_argument("store", nargs="?", default="dash-molecules")
    p_cv_run_dash.add_argument("--n-shards", type=int, required=True)
    p_cv_run_dash.add_argument("--depths", required=True)
    p_cv_run_dash.add_argument(
        "--repeats", default="0", help="comma-separated repeat seeds (default: 0)"
    )
    p_cv_run_dash.add_argument("--k", type=int, default=5)
    p_cv_run_dash.add_argument("--max-depth", type=int, required=True)
    p_cv_run_dash.add_argument("--normalization", default="std_weighted")
    p_cv_run_dash.add_argument("--method", default="dash")
    p_cv_run_dash.add_argument("--experiment", default="dash-cv")
    p_cv_run_dash.add_argument("--seed", type=int, default=0)
    p_cv_run_dash.add_argument("--allow-dirty", action="store_true")
    p_cv_run_dash.set_defaults(func=_cmd_cv_run_dash)

    p_cv_run_sieve = sub.add_parser(
        "cv-run-sieve",
        help="assemble each CV sample's training model by merging Sieve's "
        "own shards (per depth), evaluate at every requested depth",
    )
    p_cv_run_sieve.add_argument("store", nargs="?", default="dash-molecules")
    p_cv_run_sieve.add_argument("--n-shards", type=int, required=True)
    p_cv_run_sieve.add_argument("--depths", required=True)
    p_cv_run_sieve.add_argument("--repeats", default="0")
    p_cv_run_sieve.add_argument(
        "--codes-path", required=True, type=Path, help="from build-sieve-codes"
    )
    p_cv_run_sieve.add_argument("--config-label", required=True)
    p_cv_run_sieve.add_argument("--predictor-params", default=None)
    p_cv_run_sieve.add_argument("--k", type=int, default=5)
    p_cv_run_sieve.add_argument("--normalization", default="equal_weighted")
    p_cv_run_sieve.add_argument(
        "--method", default=None, help="defaults to sieve-<config-label>"
    )
    p_cv_run_sieve.add_argument("--experiment", default="sieve-cv")
    p_cv_run_sieve.add_argument("--seed", type=int, default=0)
    p_cv_run_sieve.add_argument("--allow-dirty", action="store_true")
    p_cv_run_sieve.set_defaults(func=_cmd_cv_run_sieve)

    p_compare = sub.add_parser(
        "compare",
        help="repeated-measures ANOVA + Tukey HSD across cv.py runs "
        "(Ash/Wognum/Rodriguez-Perez JCIM 2025 protocol)",
    )
    p_compare.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="a run.experiment to read CV runs from; repeatable (e.g. "
        "--experiment dash-cv --experiment sieve-cv)",
    )
    p_compare.add_argument("--metric", default="mae")
    p_compare.add_argument(
        "--depth-by-method",
        default=None,
        help='JSON object, e.g. \'{"dash": 16, "sieve-element-eb": 6}\' -- '
        "each method's own Study-A-selected depth",
    )
    p_compare.add_argument("--alpha", type=float, default=0.05)
    p_compare.add_argument(
        "--out", type=Path, default=None, help="write a Tukey CI plot here"
    )
    p_compare.set_defaults(func=_cmd_compare)

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
    p_subsample.add_argument(
        "--n-stores",
        type=int,
        default=1,
        help="number of mutually disjoint stores to draw, named DEST-1 ... "
        "DEST-N (default: 1, which keeps the bare DEST name)",
    )
    p_subsample.set_defaults(func=_cmd_subsample_store)

    p_partition = sub.add_parser(
        "partition-store",
        help="exhaustively split every molecule in a store into N disjoint "
        "stores, preserving each split's own fractions -- unlike "
        "subsample-store, nothing is left unused",
    )
    p_partition.add_argument("dest", help="name prefix for the partitioned stores")
    p_partition.add_argument(
        "--source",
        default="dash-molecules",
        help="name of the already-split source store (default: dash-molecules)",
    )
    p_partition.add_argument(
        "--n-stores",
        type=int,
        required=True,
        help="number of disjoint stores to divide every molecule into, "
        "named DEST-1 ... DEST-N (DEST alone if 1)",
    )
    p_partition.add_argument(
        "--conformers-per-molecule",
        type=int,
        default=None,
        help="max conformers kept per molecule (default: unlimited -- "
        "every conformer of every molecule is kept)",
    )
    p_partition.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducible partitioning"
    )
    p_partition.set_defaults(func=_cmd_partition_store)

    p_ua = sub.add_parser(
        "to-united-atom",
        help="build a united-atom (hydrogens removed, folded into their "
        "heavy-atom neighbor's own atom property) version of an "
        "already-prepared store",
    )
    p_ua.add_argument("dest", help="name of the new, united-atom store")
    p_ua.add_argument(
        "--source",
        default="dash-molecules",
        help="name of the already-prepared source store (default: dash-molecules)",
    )
    p_ua.add_argument(
        "--atom-property",
        default="MBIScharge",
        help="atom property to fold from a removed H onto its heavy-atom "
        "neighbor (default: MBIScharge)",
    )
    p_ua.set_defaults(func=_cmd_to_united_atom)

    p_summary = sub.add_parser(
        "summarize", help="collect runs/**/metrics.json into a CSV"
    )
    p_summary.set_defaults(func=_cmd_summarize)

    p_sweep = sub.add_parser(
        "sweep",
        help="plot metrics against a run parameter, gathered across many runs",
    )
    p_sweep.add_argument(
        "--x",
        required=True,
        help="dotted config path for the x axis, e.g. predictor.params.max_wl_depth",
    )
    p_sweep.add_argument("--experiment", default=None)
    p_sweep.add_argument("--source", choices=("runs", "mlflow"), default="runs")
    p_sweep.add_argument("--tracking", default=DEFAULT_TRACKING_URI)
    p_sweep.add_argument(
        "--metric",
        action="append",
        default=None,
        help="repeatable; default: mae, rmse, r2",
    )
    p_sweep.add_argument(
        "--split",
        action="append",
        default=None,
        help="repeatable; default: test, train",
    )
    p_sweep.add_argument("--group-by", dest="group_by", default=None)
    p_sweep.add_argument(
        "--band",
        choices=("none", "errorbar", "fill"),
        default="fill",
        help="how to render each point's std dev on the plot (default: fill)",
    )
    p_sweep.add_argument("--out", default=None, help="results/<NAME>/")
    p_sweep.set_defaults(func=_cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
