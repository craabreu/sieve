"""Persisted, per-tree-node charge statistics -- what
``predictors/dash.py``'s ``DASHChargePredictor.fit()`` actually derives from
a train split (a mean and std of ``MBIScharge`` at every ``(branch_idx,
node_id)`` its own atoms matched), separated out so it can be saved once and
reloaded without re-running the expensive ``match_new_atom`` walk that
produces it -- see nested_runner.py, where a "raw predict" parent run saves
this and later children reload it. See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PathKey = tuple[int, int]
NodePath = list[PathKey]


@dataclass(frozen=True)
class LiteralTreeChargeProperties:
    """What ``apply_node_stats`` writes onto a ``DASHTree``'s own
    ``data_storage``, and what ``predictors.dash.predict_via_data_storage_walk``
    needs to read it back."""

    charge_column: str
    fallback_charge: float


@dataclass(frozen=True)
class TreeNodeStats:
    """One row per populated ``(branch_idx, node_id)``: ``mean``/``std``/
    ``count`` of the train atoms matched there, plus the two global
    fallbacks (mean/std over every train atom, regardless of node) a
    predictor uses when an atom's own path is entirely unpopulated. Parallel
    numpy arrays, not a dict -- this whole object is what gets saved to /
    loaded from disk."""

    branch_idx: NDArray[np.int64]
    node_id: NDArray[np.int64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]
    fallback_mean: float
    fallback_std: float


def compute_node_stats(
    paths: list[NodePath], atom_charge: NDArray[np.floating]
) -> TreeNodeStats:
    """Two-pass (sum, sum-of-squares, count) aggregation of ``atom_charge``
    over every node on every atom's own matched path -- pure numpy/python,
    no tree object needed, so this is testable without a real ``DASHTree``.
    A node with only one matching atom gets ``std=0.0`` (population std,
    ``ddof=0`` -- consistent with a single observation having no spread).
    """
    if len(paths) != len(atom_charge):
        raise ValueError("paths and atom_charge must have the same length")
    atom_charge = np.asarray(atom_charge, dtype=np.float64)

    charge_sum: dict[PathKey, float] = {}
    charge_sumsq: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}
    for path, charge in zip(paths, atom_charge, strict=True):
        for key in path:
            charge_sum[key] = charge_sum.get(key, 0.0) + float(charge)
            charge_sumsq[key] = charge_sumsq.get(key, 0.0) + float(charge) ** 2
            count[key] = count.get(key, 0) + 1

    keys = list(count)
    branch_idx = np.array([k[0] for k in keys], dtype=np.int64)
    node_id = np.array([k[1] for k in keys], dtype=np.int64)
    count_arr = np.array([count[k] for k in keys], dtype=np.int64)
    sum_arr = np.array([charge_sum[k] for k in keys], dtype=np.float64)
    sumsq_arr = np.array([charge_sumsq[k] for k in keys], dtype=np.float64)
    mean_arr = sum_arr / count_arr
    variance = np.clip(sumsq_arr / count_arr - mean_arr**2, 0.0, None)
    std_arr = np.sqrt(variance)

    return TreeNodeStats(
        branch_idx=branch_idx,
        node_id=node_id,
        mean=mean_arr,
        std=std_arr,
        count=count_arr,
        fallback_mean=float(atom_charge.mean()),
        fallback_std=float(atom_charge.std()),
    )


def save_node_stats(stats: TreeNodeStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        branch_idx=stats.branch_idx,
        node_id=stats.node_id,
        mean=stats.mean,
        std=stats.std,
        count=stats.count,
        fallback_mean=np.float64(stats.fallback_mean),
        fallback_std=np.float64(stats.fallback_std),
    )


def load_node_stats(path: str | Path) -> TreeNodeStats:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    with np.load(path) as data:
        return TreeNodeStats(
            branch_idx=data["branch_idx"],
            node_id=data["node_id"],
            mean=data["mean"],
            std=data["std"],
            count=data["count"],
            fallback_mean=float(data["fallback_mean"]),
            fallback_std=float(data["fallback_std"]),
        )


def apply_node_stats(
    tree: Any,
    stats: TreeNodeStats,
    *,
    mean_column: str = "dash_charge_mean",
    std_column: str = "dash_charge_std",
) -> tuple[LiteralTreeChargeProperties, LiteralTreeChargeProperties]:
    """Write ``stats``'s mean/std onto ``tree.data_storage`` (one column
    each, indexed by ``node_id`` -- a node with no entry stays ``NaN``,
    exactly ``DASHTree.get_property_noNAN``'s own missing-value semantics),
    grouped by branch. Returns the ``(mean_props, std_props)``
    ``predict_via_data_storage_walk`` needs."""
    by_branch: dict[int, list[int]] = {}
    for i, branch_idx in enumerate(stats.branch_idx):
        by_branch.setdefault(int(branch_idx), []).append(i)

    for branch_idx, indices in by_branch.items():
        df = tree.data_storage[branch_idx]
        n_rows = len(df)
        node_ids = stats.node_id[indices]
        means = np.full(n_rows, np.nan)
        stds = np.full(n_rows, np.nan)
        means[node_ids] = stats.mean[indices]
        stds[node_ids] = stats.std[indices]
        df[mean_column] = means
        df[std_column] = stds

    return (
        LiteralTreeChargeProperties(
            charge_column=mean_column, fallback_charge=stats.fallback_mean
        ),
        LiteralTreeChargeProperties(
            charge_column=std_column, fallback_charge=stats.fallback_std
        ),
    )
