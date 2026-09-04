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


def test_merge_node_stats_matches_computing_on_the_union_directly():
    """merge_node_stats(compute_node_stats(A), compute_node_stats(B)) must
    equal compute_node_stats(A + B) exactly -- the same property
    sieve.merge's own test_fold_matches_sequential_merge asserts for sieve
    classes, here for DASH-tree's own (branch_idx, node_id) stats. Both
    shards touch node (0, 1); only A touches (0, 2); only B touches (1, 0)
    -- exercising overlap and each shard's own exclusive nodes together."""
    from charge_experiments.tree_artifact import compute_node_stats, merge_node_stats

    paths_a = [[(0, 1)], [(0, 1), (0, 2)], [(0, 2)]]
    charge_a = np.array([0.10, 0.30, 0.50])
    paths_b = [[(0, 1)], [(0, 1)], [(1, 0)]]
    charge_b = np.array([0.20, 0.90, -0.40])

    a = compute_node_stats(paths_a, charge_a)
    b = compute_node_stats(paths_b, charge_b)
    merged = merge_node_stats(a, b)

    whole = compute_node_stats(paths_a + paths_b, np.concatenate([charge_a, charge_b]))

    def as_dict(stats):
        return {
            (int(br), int(nd)): (float(m), float(s), int(c))
            for br, nd, m, s, c in zip(
                stats.branch_idx,
                stats.node_id,
                stats.mean,
                stats.std,
                stats.count,
                strict=True,
            )
        }

    merged_d, whole_d = as_dict(merged), as_dict(whole)
    assert merged_d.keys() == whole_d.keys()
    for key in whole_d:
        m_mean, m_std, m_count = merged_d[key]
        w_mean, w_std, w_count = whole_d[key]
        assert m_count == w_count
        assert m_mean == pytest.approx(w_mean)
        assert m_std == pytest.approx(w_std)


def test_merge_node_stats_a_node_exclusive_to_one_side_passes_through():
    from charge_experiments.tree_artifact import compute_node_stats, merge_node_stats

    a = compute_node_stats([[(0, 1)]], np.array([0.5]))
    b = compute_node_stats([[(0, 2)]], np.array([0.7]))

    merged = merge_node_stats(a, b)

    assert set(
        zip(merged.branch_idx.tolist(), merged.node_id.tolist(), strict=True)
    ) == {
        (0, 1),
        (0, 2),
    }
    assert merged.count.tolist() == [1, 1]
    np.testing.assert_allclose(sorted(merged.mean), [0.5, 0.7])


def test_merge_node_stats_is_commutative():
    from charge_experiments.tree_artifact import compute_node_stats, merge_node_stats

    a = compute_node_stats([[(0, 1)], [(0, 1)]], np.array([0.1, 0.3]))
    b = compute_node_stats([[(0, 1)], [(0, 1)], [(0, 1)]], np.array([0.2, 0.4, 0.9]))

    ab = merge_node_stats(a, b)
    ba = merge_node_stats(b, a)

    idx_ab = ab.node_id.tolist().index(1)
    idx_ba = ba.node_id.tolist().index(1)
    assert ab.count[idx_ab] == ba.count[idx_ba]
    assert ab.mean[idx_ab] == pytest.approx(ba.mean[idx_ba])
    assert ab.std[idx_ab] == pytest.approx(ba.std[idx_ba])


def test_fold_node_stats_reduces_many_shards_left_to_right():
    """The N-way convenience wrapper the 60k-1..5-style fold workflow
    actually needs: fold K independently-fit shards into one, exactly
    equal to fitting their union directly."""
    from charge_experiments.tree_artifact import (
        compute_node_stats,
        fold_node_stats,
    )

    shards_paths = [[[(0, 1)]], [[(0, 1)], [(0, 2)]], [[(0, 2)]]]
    shards_charge = [np.array([0.1]), np.array([0.2, 0.4]), np.array([0.6])]
    shards = [
        compute_node_stats(p, c)
        for p, c in zip(shards_paths, shards_charge, strict=True)
    ]

    folded = fold_node_stats(shards)

    whole = compute_node_stats(
        [p for paths in shards_paths for p in paths],
        np.concatenate(shards_charge),
    )

    def as_dict(stats):
        return {
            (int(br), int(nd)): (float(m), float(s), int(c))
            for br, nd, m, s, c in zip(
                stats.branch_idx,
                stats.node_id,
                stats.mean,
                stats.std,
                stats.count,
                strict=True,
            )
        }

    folded_d, whole_d = as_dict(folded), as_dict(whole)
    assert folded_d.keys() == whole_d.keys()
    for key in whole_d:
        f_mean, f_std, f_count = folded_d[key]
        w_mean, w_std, w_count = whole_d[key]
        assert f_count == w_count
        assert f_mean == pytest.approx(w_mean)
        assert f_std == pytest.approx(w_std)


def test_fold_node_stats_of_a_single_shard_is_that_shard():
    from charge_experiments.tree_artifact import compute_node_stats, fold_node_stats

    stats = compute_node_stats([[(0, 1)]], np.array([0.3]))
    folded = fold_node_stats([stats])

    assert folded.count.tolist() == stats.count.tolist()
    np.testing.assert_allclose(folded.mean, stats.mean)
    np.testing.assert_allclose(folded.std, stats.std)


def test_fold_node_stats_rejects_an_empty_sequence():
    from charge_experiments.tree_artifact import fold_node_stats

    with pytest.raises(ValueError, match="at least one shard"):
        fold_node_stats([])


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
    # No invented global-mean/std fallback -- DASH's own get_property_noNAN
    # returns NaN, not a mean, when a hierarchy is entirely unpopulated.
    assert np.isnan(mean_props.fallback_charge)
    assert np.isnan(std_props.fallback_charge)


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
    assert direct_mean_props.charge_column == loaded_mean_props.charge_column
    # fallback_charge is NaN on both sides -- nan != nan, so compare via
    # isnan rather than dataclass equality.
    assert np.isnan(direct_mean_props.fallback_charge)
    assert np.isnan(loaded_mean_props.fallback_charge)
