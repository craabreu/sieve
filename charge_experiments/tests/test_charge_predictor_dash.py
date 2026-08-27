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
    from charge_experiments.predictors.dash import populate_tree_with_charge_property

    tree = _FakeTree({0: 3})
    # Two atoms both matched at path [(0, 1)] (root only); charges 0.2 and 0.4.
    paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4])

    props = populate_tree_with_charge_property(tree, paths, atom_charge)

    df = tree.data_storage[0]
    assert df.loc[1, props.charge_column] == pytest.approx(0.3)
    assert pd.isna(df.loc[0, props.charge_column])
    assert pd.isna(df.loc[2, props.charge_column])
    assert props.fallback_charge == pytest.approx(0.3)


def test_predict_via_data_storage_walk_prefers_deepest_populated_node():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    # atom0: shallow only; atom1: shallow+deep
    train_paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_charge = np.array([0.1, 0.5])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    # Predict for an atom matched at both node 1 (populated) and node 2
    # (also populated, deepest) -> should use node 2's own mean (0.5), not
    # node 1's blended mean.
    test_paths = [[(0, 1), (0, 2)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.5)


def test_predict_via_data_storage_walk_backs_off_to_shallower_node():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    train_paths = [[(0, 1)]]
    atom_charge = np.array([0.7])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    # Deepest node (3) was never populated at train time -> back off to node 1.
    test_paths = [[(0, 1), (0, 3)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.7)


def test_predict_via_data_storage_walk_falls_back_to_global_mean_for_unmatched_atom():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 2})
    train_paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.6])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    predicted = predict_via_data_storage_walk(tree, [[]], props)
    assert predicted[0] == pytest.approx(0.4)  # global mean, atom has empty path
