#!/usr/bin/env python
"""Write the manuscript's figure snapshots from this repository's own runs.

The figures in ``../sieve_paper/figures`` are drawn from JSON snapshots of
per-sample measurements rather than from the run directories, so the
manuscript builds without this repository present and its numbers cannot
drift with a later re-run. Those snapshots were previously produced by hand,
which is why they carried a ``source_repo_commit`` recorded by whoever made
them and no way to remake them.

This produces both, from the runs, with that provenance filled in
automatically:

  data/study-b.json      -- Study B, every arm at its selected depth
  data/depth-sweep.json  -- Study A, the depth curve per arm

Statistics are not reimplemented here. The analysis of variance and the Tukey
comparisons come from ``experiments.compare``, the same module the figure
scripts import, so a snapshot cannot disagree with the study it summarizes.
The per-sample values are written as measured and every derived quantity is
recomputed by the figure script, so nothing is carried across twice.

Usage, from the repository root:

    .venv/bin/python experiments/workflows/paper_snapshots.py [--out DIR]

``--out`` defaults to ``../sieve_paper/figures/data``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

from experiments.compare import (
    read_cv_table,
    repeated_measures_anova,
    tukey_hsd,
)

RUNS = REPO / "experiments" / "runs"
STORE = "dash-molecules"
METRICS = ("rmse", "mae", "r2", "sum_constraint/rmse")

# Study B: the seven arms and the depth each is fixed at, mirroring
# cv_charges.sh's SIEVE_VARIANTS, DASH_SELECTED_DEPTH and HOSE_SELECTED_RADIUS.
STUDY_B_EXPERIMENTS = ("dash-cv-study-b", "sieve-cv-study-b", "hose-cv-study-b")
DEPTH_BY_METHOD = {
    "dash": 16,
    "hose": 5,
    "sieve-element-pooled": 5,
    "sieve-element-pooled-eb": 5,
    "sieve-element-continuation": 5,
    "sieve-element-continuation-eb": 5,
    "sieve-element-continuation-cutoff": 5,
}
LABELS = {
    "dash": "DASH",
    "hose": "HOSE",
    "sieve-element-pooled": "pooled",
    "sieve-element-pooled-eb": "pooled + EB",
    "sieve-element-continuation": "continuation",
    "sieve-element-continuation-eb": "continuation + EB",
    "sieve-element-continuation-cutoff": "continuation + cutoff",
}

# Study A: one panel per arm, with the axis label each arm's setting carries.
DEPTH_SWEEP_ARMS = (
    {
        "experiment": "dash-cv-study-a",
        "method": "dash",
        "label": "DASH",
        "panel": "DASH",
        "x_label": "Maximum Path Depth",
        "min_depth": None,
        "normalization": "std_weighted",
    },
    {
        "experiment": "sieve-cv-study-a",
        "method": "sieve-element-pooled",
        "label": "Sieve, pooled",
        "panel": "Sieve",
        "x_label": "Refinement Radius",
        "min_depth": None,
        "normalization": "equal_weighted",
    },
    {
        "experiment": "sieve-cv-study-a",
        "method": "sieve-element-continuation",
        "label": "Sieve, continuation",
        "panel": "Sieve",
        "x_label": "Refinement Radius",
        "min_depth": None,
        "normalization": "equal_weighted",
    },
)
DEPTH_SWEEP_METRICS = ("rmse", "r2")


def head_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _stamp(study: str, note: str) -> dict:
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_repo_commit": head_commit(),
        "store": STORE,
        "study": study,
        "note": note,
    }


def study_b_snapshot() -> dict:
    out = _stamp(
        "Study B, five repeats x five folds = 25 paired samples per arm",
        "Per-sample values as measured, rows paired by (repeat, fold); the "
        "figure script recomputes every mean, interval and test from them.",
    )
    out["depth_by_method"] = dict(DEPTH_BY_METHOD)
    out["labels"] = dict(LABELS)
    out["metrics"] = {}
    for metric in METRICS:
        methods, table = read_cv_table(
            RUNS,
            STUDY_B_EXPERIMENTS,
            depth_by_method=DEPTH_BY_METHOD,
            metric=metric,
        )
        anova = repeated_measures_anova(methods, table)
        out["metrics"][metric] = {
            "methods": list(methods),
            "samples": [[float(v) for v in row] for row in table],
            "anova": {
                "f": float(anova.f_stat),
                "p": float(anova.p_value),
                "df_method": int(anova.df_method),
                "df_error": int(anova.df_error),
            },
            "tukey": [
                {
                    "a": c.a,
                    "b": c.b,
                    "diff": float(c.diff),
                    "ci_lo": float(c.ci_lo),
                    "ci_hi": float(c.ci_hi),
                    "p": float(c.p_value),
                }
                for c in tukey_hsd(methods, table, anova=anova)
            ],
        }
    return out


def _fold_series(experiment: str, method: str, metric: str) -> list[dict]:
    """Per-depth fold values for one arm, plus the training-side series and
    the fold sizes the figure's weighted means need."""
    from experiments.aggregate import read_runs_from_dirs

    by_depth: dict[int, dict[str, list]] = {}
    for row in read_runs_from_dirs(RUNS, experiment):
        params = row.params
        if params.get("cv.method") != method:
            continue
        depth = params.get("cv.depth")
        if depth is None:
            continue
        d = by_depth.setdefault(
            int(depth),
            {
                "fold_values": [],
                "fold_n_test": [],
                "fold_n_train": [],
                "train_fold_values": [],
            },
        )
        metrics = row.metrics
        if metrics.get(metric) is None:
            continue
        d["fold_values"].append(float(metrics[metric]))
        for key, name in (("n_test", "fold_n_test"), ("n_train", "fold_n_train")):
            v = metrics.get(key)
            d[name].append(float(v) if v is not None else float("nan"))
        tv = metrics.get(f"train/{metric}")
        d["train_fold_values"].append(float(tv) if tv is not None else float("nan"))

    series = []
    for depth in sorted(by_depth):
        entry = {"depth": depth}
        entry.update(by_depth[depth])
        series.append(entry)
    return series


def depth_sweep_snapshot() -> dict:
    out = _stamp(
        "Study A, one repeat (seed 0), k=5 folds over 50 cluster-clean shards",
        "Per-fold values as measured; the figure script recomputes means and "
        "Nadeau-Bengio corrected intervals from them, so nothing derived is "
        "carried across.",
    )
    arms = []
    for spec in DEPTH_SWEEP_ARMS:
        arm = dict(spec)
        arm["metrics"] = {
            m: _fold_series(spec["experiment"], spec["method"], m)
            for m in DEPTH_SWEEP_METRICS
        }
        arm["n_runs"] = sum(
            len(e["fold_values"]) for e in arm["metrics"][DEPTH_SWEEP_METRICS[0]]
        )
        arm["omitted_depths"] = []
        arms.append(arm)
    out["arms"] = arms
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO.parent / "sieve_paper" / "figures" / "data"),
        help="directory to write study-b.json and depth-sweep.json into",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, snapshot in (
        ("study-b.json", study_b_snapshot()),
        ("depth-sweep.json", depth_sweep_snapshot()),
    ):
        path = out_dir / name
        path.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
