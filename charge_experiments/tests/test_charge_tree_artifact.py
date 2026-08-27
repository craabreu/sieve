"""Pure-numpy/pandas tests for tree_artifact.py -- a fake tree-like object
stands in for a real DASHTree (same pattern as
test_charge_predictor_dash.py's own _FakeTree), so these need no rdkit and
no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

import pandas as pd


class _FakeTree:
    def __init__(self, branch_sizes: dict[int, int]):
        self.data_storage = {
            branch: pd.DataFrame(index=range(size))
            for branch, size in branch_sizes.items()
        }


def test_compute_node_stats_computes_mean_std_count():
    from charge_experiments.tree_artifact import compute_node_stats

    # Node (0, 1): charges 0.2, 0.4, 0.6 -> mean 0.4.
    paths = [[(0, 1)], [(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4, 0.6])

    stats = compute_node_stats(paths, atom_charge)

    assert stats.branch_idx.tolist() == [0]
    assert stats.node_id.tolist() == [1]
    assert stats.count.tolist() == [3]
    np.testing.assert_allclose(stats.mean, [0.4])
    np.testing.assert_allclose(stats.std, [np.std([0.2, 0.4, 0.6])])


def test_compute_node_stats_std_zero_for_singleton_node():
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)]]
    atom_charge = np.array([0.7])

    stats = compute_node_stats(paths, atom_charge)

    np.testing.assert_allclose(stats.std, [0.0])


def test_compute_node_stats_fallback_matches_global_mean_and_std():
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)], [(0, 2)], [(1, 1)]]
    atom_charge = np.array([0.1, 0.5, -0.3])

    stats = compute_node_stats(paths, atom_charge)

    assert stats.fallback_mean == pytest.approx(atom_charge.mean())
    assert stats.fallback_std == pytest.approx(atom_charge.std())


def test_compute_node_stats_aggregates_a_node_visited_via_multiple_paths():
    """A node reached along more than one atom's path pools all of them,
    same as populate_tree_with_charge_property's own prior behavior."""
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_charge = np.array([0.2, 0.4])

    stats = compute_node_stats(paths, atom_charge)

    idx = stats.node_id.tolist().index(1)
    assert stats.count[idx] == 2
    assert stats.mean[idx] == pytest.approx(0.3)


def test_apply_node_stats_writes_mean_and_std_columns():
    from charge_experiments.tree_artifact import apply_node_stats, compute_node_stats

    tree = _FakeTree({0: 3})
    paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4])
    stats = compute_node_stats(paths, atom_charge)

    mean_props, std_props = apply_node_stats(tree, stats)

    df = tree.data_storage[0]
    assert df.loc[1, mean_props.charge_column] == pytest.approx(0.3)
    assert pd.isna(df.loc[0, mean_props.charge_column])
    assert df.loc[1, std_props.charge_column] == pytest.approx(0.1)
    assert mean_props.fallback_charge == pytest.approx(0.3)
    assert std_props.fallback_charge == pytest.approx(np.std([0.2, 0.4]))


def test_save_and_load_node_stats_round_trips(tmp_path):
    from charge_experiments.tree_artifact import (
        compute_node_stats,
        load_node_stats,
        save_node_stats,
    )

    paths = [[(0, 1)], [(0, 1), (1, 3)]]
    atom_charge = np.array([0.2, 0.4])
    stats = compute_node_stats(paths, atom_charge)

    path = tmp_path / "stats.npz"
    save_node_stats(stats, path)
    loaded = load_node_stats(path)

    np.testing.assert_array_equal(loaded.branch_idx, stats.branch_idx)
    np.testing.assert_array_equal(loaded.node_id, stats.node_id)
    np.testing.assert_allclose(loaded.mean, stats.mean)
    np.testing.assert_allclose(loaded.std, stats.std)
    np.testing.assert_array_equal(loaded.count, stats.count)
    assert loaded.fallback_mean == pytest.approx(stats.fallback_mean)
    assert loaded.fallback_std == pytest.approx(stats.fallback_std)


def test_apply_node_stats_from_loaded_stats_matches_direct_apply(tmp_path):
    from charge_experiments.tree_artifact import (
        apply_node_stats,
        compute_node_stats,
        load_node_stats,
        save_node_stats,
    )

    paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_charge = np.array([0.2, 0.5])
    stats = compute_node_stats(paths, atom_charge)

    tree_direct = _FakeTree({0: 3})
    direct_mean_props, _ = apply_node_stats(tree_direct, stats)

    path = tmp_path / "stats.npz"
    save_node_stats(stats, path)
    loaded = load_node_stats(path)
    tree_loaded = _FakeTree({0: 3})
    loaded_mean_props, _ = apply_node_stats(tree_loaded, loaded)

    pd.testing.assert_frame_equal(
        tree_direct.data_storage[0], tree_loaded.data_storage[0]
    )
    assert direct_mean_props == loaded_mean_props
