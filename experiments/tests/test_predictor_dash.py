"""Pure-numpy/pandas tests for dash.py's tree-populate/predict-walk logic --
a fake tree-like object stands in for a real DASHTree, so these need no
rdkit and no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd


class _FakeTree:
    """A minimal stand-in exposing only ``data_storage`` (a dict of
    branch_idx -> pandas DataFrame), the one attribute
    populate_tree_with_charge_property/predict_via_data_storage_walk touch."""

    def __init__(self, branch_sizes: dict[int, int]):
        self.data_storage = {
            branch: pd.DataFrame(index=range(size))
            for branch, size in branch_sizes.items()
        }


def test_populate_tree_with_charge_property_writes_node_means():
    from experiments.predictors.dash import populate_tree_with_charge_property

    tree = _FakeTree({0: 3})
    # Two atoms both matched at path [(0, 1)] (root only); charges 0.2 and 0.4.
    paths = [[(0, 1)], [(0, 1)]]
    atom_value = np.array([0.2, 0.4])

    props = populate_tree_with_charge_property(tree, paths, atom_value)

    df = tree.data_storage[0]
    assert df.loc[1, props.charge_column] == pytest.approx(0.3)
    assert pd.isna(df.loc[0, props.charge_column])
    assert pd.isna(df.loc[2, props.charge_column])
    # No invented global-mean fallback -- DASH's own get_property_noNAN
    # returns NaN, not a mean, when a hierarchy is entirely unpopulated.
    assert np.isnan(props.fallback_charge)


def test_predict_via_data_storage_walk_prefers_deepest_populated_node():
    from experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    # atom0: shallow only; atom1: shallow+deep
    train_paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_value = np.array([0.1, 0.5])
    props = populate_tree_with_charge_property(tree, train_paths, atom_value)

    # Predict for an atom matched at both node 1 (populated) and node 2
    # (also populated, deepest) -> should use node 2's own mean (0.5), not
    # node 1's blended mean.
    test_paths = [[(0, 1), (0, 2)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.5)


def test_predict_via_data_storage_walk_backs_off_to_shallower_node():
    from experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    train_paths = [[(0, 1)]]
    atom_value = np.array([0.7])
    props = populate_tree_with_charge_property(tree, train_paths, atom_value)

    # Deepest node (3) was never populated at train time -> back off to node 1.
    test_paths = [[(0, 1), (0, 3)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.7)


def test_predict_via_data_storage_walk_is_nan_for_unmatched_atom():
    """No invented fallback: an atom whose own path-matching failed
    entirely (an empty path) gets NaN, not a substituted global mean --
    matching DASH's own get_property_noNAN, which returns NaN (not a
    mean of anything) when a hierarchy is unpopulated."""
    from experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 2})
    train_paths = [[(0, 1)], [(0, 1)]]
    atom_value = np.array([0.2, 0.6])
    props = populate_tree_with_charge_property(tree, train_paths, atom_value)

    predicted = predict_via_data_storage_walk(tree, [[]], props)
    assert np.isnan(predicted[0])


def _fit_props(tree, paths, atom_value):
    """Both mean_props and std_props from one fit, exactly like
    DASHChargePredictor.fit() does it -- unlike
    populate_tree_with_charge_property (which only returns mean_props and
    would silently overwrite the other's columns if called twice on the
    same tree), apply_node_stats writes both from one TreeNodeStats."""
    from experiments.tree_artifact import apply_node_stats, compute_node_stats

    stats = compute_node_stats(paths, np.asarray(atom_value))
    return apply_node_stats(tree, stats)


def test_predict_raw_from_paths_truncates_before_backoff():
    """The whole point: deriving a shallow-depth prediction from paths
    already walked deep must match what walking at that shallower depth
    directly would have produced -- i.e. it must NOT see the deeper,
    populated node that a true max_depth=1 walk would never have reached."""
    from experiments.predictors.dash import predict_raw_from_paths

    tree = _FakeTree({0: 4})
    # Fit populates every level along a full depth-2 walk.
    mean_props, std_props = _fit_props(tree, [[(0, 1), (0, 2)]], [0.5])

    # One atom walked to depth 2 at predict time too.
    full_paths = [[(0, 1), (0, 2)]]

    at_depth_2 = predict_raw_from_paths(
        tree, full_paths, mean_props, std_props, max_depth=2
    )
    at_depth_1 = predict_raw_from_paths(
        tree, full_paths, mean_props, std_props, max_depth=1
    )

    # Both node 1 and node 2 are populated with the same value here, so
    # this alone wouldn't distinguish truncation from no truncation --
    # the next test does that with genuinely different populated values.
    assert at_depth_2.atom_value[0] == pytest.approx(0.5)
    assert at_depth_1.atom_value[0] == pytest.approx(0.5)


def test_predict_raw_from_paths_at_shallow_depth_ignores_deeper_populated_node():
    from experiments.predictors.dash import predict_raw_from_paths

    tree = _FakeTree({0: 4})
    # One fit (a single compute_node_stats call): a shallow-only training
    # atom gives node 1 mean=0.2 unblended, a deep-only training atom
    # gives node 2 mean=0.9 unblended (fake-tree test, so paths need not
    # respect real parent/child topology -- only the aggregation-by-
    # node-id logic is under test here).
    mean_props, std_props = _fit_props(tree, [[(0, 1)], [(0, 2)]], [0.2, 0.9])

    full_paths = [[(0, 1), (0, 2)]]

    at_depth_2 = predict_raw_from_paths(
        tree, full_paths, mean_props, std_props, max_depth=2
    )
    at_depth_1 = predict_raw_from_paths(
        tree, full_paths, mean_props, std_props, max_depth=1
    )

    # Untruncated: backs off from the deepest entry first -> node 2 (0.9).
    assert at_depth_2.atom_value[0] == pytest.approx(0.9)
    # Truncated to depth 1: node 2 is not even in the path -> node 1 (0.2),
    # exactly what a real max_depth=1 walk would have matched.
    assert at_depth_1.atom_value[0] == pytest.approx(0.2)


def test_predict_raw_from_paths_shape_matches_number_of_atoms():
    from experiments.predictors.dash import predict_raw_from_paths

    tree = _FakeTree({0: 3})
    mean_props, std_props = _fit_props(tree, [[(0, 1)]], [0.1])

    raw = predict_raw_from_paths(
        tree, [[(0, 1)], [], [(0, 1)]], mean_props, std_props, max_depth=1
    )
    assert raw.atom_value.shape == (3,)
    assert raw.atom_std.shape == (3,)


def test_merge_states_matches_fitting_the_union_directly(tmp_path):
    """DASHChargePredictor.merge_states(fit(A), fit(B)) must equal fitting
    the union of A and B directly -- the predictor-level seam over
    tree_artifact's own fold_node_stats/merge_node_stats, whose exactness
    is already pinned by test_tree_artifact.py."""
    from experiments.predictors.dash import DASHChargePredictor
    from experiments.tree_artifact import (
        compute_node_stats,
        save_node_stats,
    )

    paths_a = [[(0, 1)], [(0, 1), (0, 2)]]
    charge_a = np.array([0.10, 0.30])
    paths_b = [[(0, 1)], [(1, 0)]]
    charge_b = np.array([0.20, -0.40])

    stats_a = compute_node_stats(paths_a, charge_a)
    stats_b = compute_node_stats(paths_b, charge_b)

    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    save_node_stats(stats_a, path_a)
    save_node_stats(stats_b, path_b)

    merged_path = tmp_path / "merged.npz"
    DASHChargePredictor.merge_states([path_a, path_b], merged_path)

    from experiments.tree_artifact import load_node_stats

    merged = load_node_stats(merged_path)
    whole = compute_node_stats(paths_a + paths_b, np.concatenate([charge_a, charge_b]))

    def as_dict(stats):
        return {
            (int(br), int(nd)): (float(m), int(c))
            for br, nd, m, c in zip(
                stats.branch_idx, stats.node_id, stats.mean, stats.count, strict=True
            )
        }

    merged_d, whole_d = as_dict(merged), as_dict(whole)
    assert merged_d.keys() == whole_d.keys()
    for key in whole_d:
        assert merged_d[key][1] == whole_d[key][1]
        assert merged_d[key][0] == pytest.approx(whole_d[key][0])


def test_merge_states_rejects_an_empty_path_list(tmp_path):
    from experiments.predictors.dash import DASHChargePredictor

    with pytest.raises(ValueError, match="at least one"):
        DASHChargePredictor.merge_states([], tmp_path / "out.npz")
