"""Persisted, per-tree-node charge statistics -- what
``predictors/dash.py``'s ``DASHChargePredictor.fit()`` actually derives from
a train split (a mean and std of ``MBIScharge`` at every ``(branch_idx,
node_id)`` its own atoms matched), separated out so it can be saved once and
reloaded without re-running the expensive ``match_new_atom`` walk that
produces it -- see ``DASHChargePredictor.save_model_state``/
``load_model_state``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
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
    ``count`` of the train atoms matched there. Parallel numpy arrays, not a
    dict -- this whole object is what gets saved to / loaded from disk.

    Deliberately no global-mean/std fallback here: DASH's own
    ``get_property_noNAN`` returns ``np.nan`` (not a global mean) when a
    matched atom's whole hierarchy is unpopulated -- see
    ``apply_node_stats``'s own ``fallback_charge=float("nan")``. An earlier
    version of this predictor substituted the train split's own global mean
    for that case; that was never DASH's own behavior, just something
    invented here, and has been removed to match ``predictors/
    dash_pretrained.py``'s own "no invented fallback" principle."""

    branch_idx: NDArray[np.int64]
    node_id: NDArray[np.int64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]


def compute_node_stats(
    paths: list[NodePath], atom_value: NDArray[np.floating]
) -> TreeNodeStats:
    """Two-pass (sum, sum-of-squares, count) aggregation of ``atom_value``
    over every node on every atom's own matched path -- pure numpy/python,
    no tree object needed, so this is testable without a real ``DASHTree``.
    A node with only one matching atom gets ``std=0.0`` (population std,
    ``ddof=0`` -- consistent with a single observation having no spread).
    """
    if len(paths) != len(atom_value):
        raise ValueError("paths and atom_value must have the same length")
    atom_value = np.asarray(atom_value, dtype=np.float64)

    charge_sum: dict[PathKey, float] = {}
    charge_sumsq: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}
    for path, charge in zip(paths, atom_value, strict=True):
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
    )


def merge_node_stats(a: TreeNodeStats, b: TreeNodeStats) -> TreeNodeStats:
    """Combine two independently computed ``TreeNodeStats`` into exactly
    what ``compute_node_stats`` would have produced fitting the union of
    their two atom sets directly -- verified in
    ``test_merge_node_stats_matches_computing_on_the_union_directly``.

    Uses the same parallel mean/variance combination (Chan et al.) that
    ``sieve.merge`` already relies on for its own class statistics
    (``merge.py``'s ``merge_level``/``merge_models``, ``msd[i] = wA *
    msd[i] + wB * b.msd + wA * wB * delta * delta``) -- exact, not an
    approximation, in exact arithmetic; the usual floating-point summation-
    order noise applies same as any running-variance algorithm.

    Unlike ``sieve``'s own class ids -- dynamically discovered per shard,
    so its own merge needs to remap one shard's ids into the other's
    before combining -- DASH-tree's ``(branch_idx, node_id)`` keys come
    from the externally published tree topology, identical across every
    shard by construction. So this is a plain outer join on that key: a
    node populated only in ``a`` or only in ``b`` passes through
    unchanged; a node populated in both gets its ``(count, mean, std)``
    combined exactly, no remapping needed at all.

    Vectorized (``np.unique(..., axis=0)`` plus a segmented Chan
    combination), not the row-by-row Python ``dict`` this used to be: a CV
    scheme that merges shards approaching the whole corpus's ~7M populated
    nodes made the dict form (each key a boxed Python tuple, ~1.5GB of
    object overhead alone at that size) the dominant cost of assembling a
    training model. The arithmetic is unchanged -- same formula, applied
    once per key -- so results agree with the old implementation to
    floating-point noise, not just in shape.
    """
    keys_a = np.stack([a.branch_idx, a.node_id], axis=1)
    keys_b = np.stack([b.branch_idx, b.node_id], axis=1)
    keys = np.concatenate([keys_a, keys_b], axis=0)
    uniq_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_a = len(a.branch_idx)
    inv_a, inv_b = inverse[:n_a], inverse[n_a:]

    n_out = len(uniq_keys)
    count = np.zeros(n_out, dtype=np.float64)
    mean = np.zeros(n_out, dtype=np.float64)
    var = np.zeros(n_out, dtype=np.float64)
    present_a = np.zeros(n_out, dtype=bool)

    # a's own keys are unique among themselves (one row per populated node),
    # so this fancy assignment never has two source rows racing for the same
    # destination slot -- same for every indexed assignment below.
    count[inv_a] = a.count
    mean[inv_a] = a.mean
    var[inv_a] = a.std.astype(np.float64) ** 2
    present_a[inv_a] = True

    overlap = present_a[inv_b]
    idx_overlap = inv_b[overlap]
    n_b_ov = b.count[overlap].astype(np.float64)
    mean_b_ov = b.mean[overlap]
    var_b_ov = b.std[overlap].astype(np.float64) ** 2

    n_a_ov = count[idx_overlap]
    n_tot = n_a_ov + n_b_ov
    w_a = n_a_ov / n_tot
    w_b = n_b_ov / n_tot
    delta = mean_b_ov - mean[idx_overlap]
    mean[idx_overlap] = mean[idx_overlap] + w_b * delta
    var[idx_overlap] = (
        w_a * var[idx_overlap] + w_b * var_b_ov + w_a * w_b * delta * delta
    )
    count[idx_overlap] = n_tot

    idx_new = inv_b[~overlap]
    count[idx_new] = b.count[~overlap]
    mean[idx_new] = b.mean[~overlap]
    var[idx_new] = b.std[~overlap].astype(np.float64) ** 2

    return TreeNodeStats(
        branch_idx=uniq_keys[:, 0].astype(np.int64),
        node_id=uniq_keys[:, 1].astype(np.int64),
        mean=mean,
        std=np.sqrt(var),
        count=count.astype(np.int64),
    )


def fold_node_stats(shards: Iterable[TreeNodeStats]) -> TreeNodeStats:
    """``merge_node_stats`` over any number of shards -- the direct answer
    to "recover a K'-fold result from a K-fold partition (K' < K) by
    merging the shards back together," e.g. 5-fold from a 10-fold
    partition. A single shard is returned unchanged; at least one is
    required (there is no meaningful "empty" ``TreeNodeStats``, unlike
    ``sieve.merge.fold``'s ``SieveModel.empty()`` identity -- this module
    has no equivalent zero element to fall back to).

    Left-to-right reduction, not ``sieve.merge.fold``'s balanced-tree
    pairing: that optimization earns its complexity at large shard counts
    (its own docstring: O(N^2) sequential vs O(N log N) paired), but the
    partitions this module merges are counted in single/low-double digits
    (a 5- or 10-way split of one store), where the difference is noise.
    """
    shards = list(shards)
    if not shards:
        raise ValueError("fold_node_stats requires at least one shard")
    return reduce(merge_node_stats, shards)


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
        )


def apply_node_stats(
    tree: Any,
    stats: TreeNodeStats,
    *,
    mean_column: str = "dash_charge_mean",
    std_column: str = "dash_charge_std",
    reset_existing: bool = False,
) -> tuple[LiteralTreeChargeProperties, LiteralTreeChargeProperties]:
    """Write ``stats``'s mean/std onto ``tree.data_storage`` (one column
    each, indexed by ``node_id`` -- a node with no entry stays ``NaN``,
    exactly ``DASHTree.get_property_noNAN``'s own missing-value semantics),
    grouped by branch. Returns the ``(mean_props, std_props)``
    ``predict_via_data_storage_walk`` needs.

    ``reset_existing=True`` first NaNs out both columns across *every*
    branch in ``tree.data_storage`` that already carries them, before
    writing this call's own ``stats``. Needed whenever one long-lived
    ``DASHTree`` is reused across several ``apply_node_stats`` calls (a CV
    scheme merging shards into a fresh model per sample, without re-reading
    the published tree from disk each time): the loop below only iterates
    branches populated in *this* call's own ``stats``, so a branch a
    previous call populated and this one does not would otherwise silently
    keep the previous call's values -- stale-state contamination across
    samples, not a genuine "unpopulated" NaN. A one-time ``fit()`` on a
    freshly loaded tree has nothing to reset, so this is a no-op there.
    """
    if reset_existing:
        for df in tree.data_storage.values():
            if mean_column in df.columns:
                df[mean_column] = np.nan
            if std_column in df.columns:
                df[std_column] = np.nan

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
            charge_column=mean_column, fallback_charge=float("nan")
        ),
        LiteralTreeChargeProperties(
            charge_column=std_column, fallback_charge=float("nan")
        ),
    )
