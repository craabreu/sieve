"""Gather many runs into one table, for the ``sweep`` command.

Two sources -- run directories and MLflow -- normalized to the same
``RunRow`` shape, so ``--source`` never changes what a curve means. See
docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRow:
    """One run. ``params`` are flat dotted keys spelled exactly as
    ``config.to_flat_params`` spells them (and therefore as MLflow stores
    them); ``metrics`` are spelled as ``metrics.json`` spells them, which is
    what the MLflow reader normalizes back to."""

    run_dir: str
    params: dict[str, str]
    metrics: dict[str, float]
    meta: dict[str, str]
    """Manifest fields outside ``config`` -- run_name, split_column, seed,
    git_commit. ``_cmd_summarize`` needs them, and carrying them here is what
    lets it share this reader instead of re-opening every manifest."""


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested resolved config -> dotted string params, matching
    ``config.to_flat_params``' own spelling."""
    out: dict[str, str] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_params(value, path))
        else:
            out[path] = str(value)
    return out


def _finite_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    """Only real numbers survive. metrics.json stores NaN for an undefined
    r2 and json.loads accepts it; a NaN reaching a plot becomes a silent gap
    or an axis blow-up, so drop it here rather than downstream."""
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            out[key] = float(value)
    return out


_MLFLOW_METRIC_PREFIX = "test/"


def normalize_mlflow_metric_name(name: str) -> str:
    """MLflow's spelling -> metrics.json's spelling.

    ``runner._log_mlflow_run`` logs every metric as ``f"test/{key}"``,
    including the ones already prefixed with their own split -- so run-dir
    ``mae`` is MLflow ``test/mae`` and run-dir ``train/mae`` is MLflow
    ``test/train/mae``. Exactly one prefix is removed. That quirk is worked
    around here rather than fixed at the source, since changing it would
    break comparison with every run already logged.
    """
    if name.startswith(_MLFLOW_METRIC_PREFIX):
        return name[len(_MLFLOW_METRIC_PREFIX) :]
    return name


def read_runs_from_mlflow(tracking_uri: str, experiment: str) -> list[RunRow]:
    """Read runs from an MLflow tracking backend, normalized to the same
    ``RunRow`` shape ``read_runs_from_dirs`` returns.

    Uses ``MlflowClient.search_runs`` (a typed ``list[Run]``) rather than
    the top-level ``mlflow.search_runs`` (which returns a pandas DataFrame
    by default).
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return []

    rows: list[RunRow] = []
    for run in client.search_runs([found.experiment_id]):
        params = dict(run.data.params)
        raw_metrics = {
            normalize_mlflow_metric_name(key): value
            for key, value in run.data.metrics.items()
        }
        tags = run.data.tags
        rows.append(
            RunRow(
                run_dir=str(tags.get("run_dir", "")),
                params=params,
                metrics=_finite_metrics(raw_metrics),
                meta={
                    "run_name": str(tags.get("mlflow.runName", "")),
                    "split_column": str(params.get("data.split_column", "")),
                    "seed": str(params.get("run.seed", "")),
                    "git_commit": "",
                },
            )
        )
    return rows


def read_runs_from_dirs(runs_root: Path, experiment: str | None = None) -> list[RunRow]:
    """Walk ``runs_root/<experiment>/<run>/`` for manifest+metrics pairs.

    A run with no ``metrics.json`` (crashed before writing one, or a
    ``describe`` run that only writes a manifest) is returned with an empty
    ``metrics`` dict rather than dropped -- ``_cmd_summarize`` still lists
    it, and ``build_curve`` already contributes no point for a run missing
    the requested metric.
    """
    pattern = f"{experiment}/*/manifest.json" if experiment else "*/*/manifest.json"
    rows: list[RunRow] = []
    for manifest_path in sorted(Path(runs_root).glob(pattern)):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics.json"
        raw_metrics = (
            json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        )
        manifest = json.loads(manifest_path.read_text())
        rows.append(
            RunRow(
                run_dir=str(run_dir),
                params=flatten_params(manifest.get("config", {})),
                metrics=_finite_metrics(raw_metrics),
                meta={
                    "run_name": str(manifest.get("run_name", "")),
                    "split_column": str(
                        manifest.get("data", {}).get("split_column", "")
                    ),
                    "seed": str(manifest.get("seed", "")),
                    "git_commit": str(manifest.get("git", {}).get("commit", "")),
                },
            )
        )
    return rows
