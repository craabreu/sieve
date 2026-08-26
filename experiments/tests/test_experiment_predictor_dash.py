"""Tests for DASH Stage A's supporting machinery (predictors/dash.py).

Two things are covered here, both pure numpy/pandas -- no rdkit, no DASH-tree
clone -- so they run in the fast suite:

- ``_atom_paths``: neighbor-dict reuse and unmatched-atom accounting, using a
  fake tree that records how it was called.
- ``populate_tree_with_sigma_properties``/``predict_via_data_storage_walk``:
  the fit/predict pair behind ``DASHPredictor``, using a fake tree that
  exposes only ``data_storage`` (the one attribute either function touches).

The real atom-matching path (RDKit + DASHTree) and ``DASHPredictor`` itself
are exercised only in the optional-data suite (needs the DASH-tree clone
under experiments/external/ and the real store), same pattern as
test_experiment_store.py (this directory).
"""

from __future__ import annotations

import numpy as np
import pytest

# --- _atom_paths: neighbor_dict reuse and unmatched-atom accounting --------
#
# These need rdkit (to parse SMILES) but NOT the DASH-tree clone: the tree is
# a fake that records how it was called. That keeps the two real defects
# these cover -- an O(n^2) rebuild of the per-molecule neighbor dict, and
# silently swallowing unmatched atoms -- under fast-suite coverage.

rdkit_only = pytest.importorskip("rdkit")

# Two atom-mapped molecules, mirroring the store's own SMILES convention.
_SMI_A = "[C:1]([H:2])([H:3])([H:4])[H:5]"  # 5 atoms
_SMI_B = "[O:1]([H:2])[H:3]"  # 3 atoms


def _mapped_set():
    from sieve_experiments.data import DEFAULT_GRID, MoleculeSet

    return MoleculeSet(
        smiles=[_SMI_A, _SMI_B],
        num_atoms=np.array([5, 3]),
        net_charge=np.array([0.0, 0.0]),
        grid=DEFAULT_GRID,
    )


class _FakeTree:
    """Records every match_new_atom call's neighbor_dict identity."""

    atom_feature_type = object()

    def __init__(self, fail_on=()):
        self.seen_neighbor_dicts = []
        self.fail_on = set(fail_on)
        self.calls = 0

    def match_new_atom(
        self, atom, mol, *, max_depth, attention_threshold, neighbor_dict=None
    ):
        self.calls += 1
        self.seen_neighbor_dicts.append(id(neighbor_dict))
        if atom in self.fail_on:
            raise KeyError((5, 3, 0, False, 3))  # what DASH really raises
        return [7, 0, 100 + atom]


def test_atom_paths_builds_one_neighbor_dict_per_molecule():
    """DASHTree.match_new_atom rebuilds the whole molecule's neighbor dict on
    every call unless one is passed in -- O(n_atoms^2) work per molecule.
    DASH's own _get_allAtoms_nodePaths hoists it; so must we."""
    from sieve_experiments.predictors.dash import _atom_paths

    tree = _FakeTree()
    built = []

    def fake_factory(mol, af):
        d = {"mol": id(mol)}
        built.append(d)
        return d

    _atom_paths(
        _mapped_set(),
        tree,
        max_depth=8,
        attention_threshold=10,
        neighbor_dict_factory=fake_factory,
    )

    assert len(built) == 2, "one neighbor dict per molecule, not per atom"
    assert tree.calls == 8  # 5 + 3 atoms
    # every atom of a molecule must receive that molecule's one dict
    assert tree.seen_neighbor_dicts[:5] == [id(built[0])] * 5
    assert tree.seen_neighbor_dicts[5:] == [id(built[1])] * 3


def test_atom_paths_counts_unmatched_atoms_instead_of_silently_dropping_them():
    """An atom DASH cannot match falls back to the global mean. That is a
    legitimate degradation, but it must be *counted* -- a baseline that
    silently predicts the global mean for part of the test set would look
    better or worse than it is with no way to tell."""
    from sieve_experiments.predictors.dash import _atom_paths

    tree = _FakeTree(fail_on={0})  # rdkit index 0 of each molecule
    paths, stats = _atom_paths(
        _mapped_set(),
        tree,
        max_depth=8,
        attention_threshold=10,
        neighbor_dict_factory=lambda mol, af: {},
    )

    assert stats["n_atoms"] == 8
    assert stats["n_unmatched_atoms"] == 2  # one per molecule
    assert stats["n_unmatched_molecules"] == 0  # the molecules themselves parsed
    assert sum(1 for p in paths if not p) == 2


def test_atom_paths_marks_a_whole_molecule_unmatched_when_its_neighbor_dict_fails():
    """The dominant real failure mode: init_neighbor_dict runs over the whole
    molecule, so one atom whose feature tuple is outside DASH's vocabulary
    (boron, Si, Ge, Sb, Te in chaos-store) takes every atom in that molecule
    down with it. That must be counted per molecule, not silently."""
    from sieve_experiments.predictors.dash import _atom_paths

    tree = _FakeTree()

    def factory(mol, af):
        if mol.GetNumAtoms() == 3:  # the O molecule
            raise KeyError((5, 3, 0, False, 3))
        return {}

    paths, stats = _atom_paths(
        _mapped_set(),
        tree,
        max_depth=8,
        attention_threshold=10,
        neighbor_dict_factory=factory,
    )

    assert len(paths) == 8, "still one path per atom, so the rollup stays aligned"
    assert stats["n_unmatched_molecules"] == 1
    assert stats["n_unmatched_atoms"] == 3  # all of the failed molecule's atoms
    assert paths[5:] == [[], [], []]


# --- populate_tree_with_sigma_properties / predict_via_data_storage_walk ---
#
# No rdkit/DASH-tree needed here: populate_tree_with_sigma_properties only
# ever touches tree.data_storage (an arbitrary object exposing that one
# dict-of-DataFrames attribute), never DASHTree itself.

_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def _spike(height: float, index: int, n: int = 5) -> list[float]:
    row = [0.0] * n
    row[index] = height
    return row


class _FakePropertyTree:
    """Stands in for a real DASHTree: only ``data_storage`` is touched by
    populate_tree_with_sigma_properties."""

    def __init__(self, data_storage):
        self.data_storage = data_storage


def test_populate_tree_with_sigma_properties_writes_mean_std_and_leaves_unmatched_nan():
    """Two atoms both hit node (0,0); only the first also hits (0,1); node
    (0,2) is never matched at all. Spikes at index 3 (sigma=+1.0, _GRID's
    4th point) so charge = height*1.0 differs per atom, giving a
    non-degenerate std to check too."""
    import pandas as pd
    from sieve_experiments.predictors.dash import populate_tree_with_sigma_properties

    df = pd.DataFrame({"level": [0, 1, 2]})  # 3 nodes: ids 0, 1, 2
    tree = _FakePropertyTree({0: df})

    paths = [[(0, 0), (0, 1)], [(0, 0)]]
    atom_profile = np.array([_spike(4.0, 3), _spike(2.0, 3)])  # charges 4, 2

    props = populate_tree_with_sigma_properties(
        tree, paths, atom_profile, sigma_values=_GRID
    )
    out = tree.data_storage[0]

    # node (0,0): both atoms -> mean profile = mean(4,2) = 3 at bin 3;
    # mean charge = 3, variance = mean((4-3)^2,(2-3)^2) = 1 -> std = 1.0
    assert out.loc[0, "sigma_bin_3"] == 3.0
    assert out.loc[0, props.charge_std_column] == 1.0
    # node (0,1): only the first atom -> mean = 4, std = 0 (single point)
    assert out.loc[1, "sigma_bin_3"] == 4.0
    assert out.loc[1, props.charge_std_column] == 0.0
    # node (0,2): never matched -> NaN, exactly get_property_noNAN's own
    # missing-value semantics, no minimum_support concept at all
    assert np.isnan(out.loc[2, "sigma_bin_3"])
    assert np.isnan(out.loc[2, props.charge_std_column])

    assert props.profile_columns == [f"sigma_bin_{i}" for i in range(5)]
    np.testing.assert_allclose(props.fallback_profile, atom_profile.mean(axis=0))


def test_populate_tree_with_sigma_properties_requires_matching_lengths():
    from sieve_experiments.predictors.dash import populate_tree_with_sigma_properties

    with pytest.raises(ValueError):
        populate_tree_with_sigma_properties(
            _FakePropertyTree({}),
            [[(0, 0)]],
            np.array([[1.0], [2.0]]),
            sigma_values=np.array([0.0]),
        )


def test_populate_tree_with_sigma_properties_requires_profile_width_matches():
    from sieve_experiments.predictors.dash import populate_tree_with_sigma_properties

    with pytest.raises(ValueError, match="sigma_values"):
        populate_tree_with_sigma_properties(
            _FakePropertyTree({}),
            [[(0, 0)]],
            np.array([[1.0, 2.0]]),  # 2 bins
            sigma_values=_GRID,  # 5 points
        )


def _literal_props():
    from sieve_experiments.predictors.dash import LiteralTreeProperties

    return LiteralTreeProperties(
        profile_columns=["sigma_bin_0"],
        charge_std_column="sigma_charge_std",
        fallback_profile=np.array([7.0]),
        fallback_charge_std=0.5,
        sigma_values=np.array([1.0]),
    )


def test_predict_via_data_storage_walk_falls_back_when_path_is_empty():
    """An atom _atom_paths never matched at all (path == []) must use the
    global fallback -- there is no node to walk at all."""
    from sieve_experiments.predictors.dash import predict_via_data_storage_walk

    pred = predict_via_data_storage_walk(_FakePropertyTree({}), [[]], _literal_props())
    assert pred.atom_charge_std is not None
    np.testing.assert_allclose(pred.atom_profile, [[7.0]])
    np.testing.assert_allclose(pred.atom_charge_std, [0.5])


def test_predict_via_data_storage_walk_falls_back_for_a_branch_with_no_new_columns():
    """A branch with zero training atoms never gets the new sigma_bin_*/
    sigma_charge_std columns written at all (populate_tree_with_sigma_
    properties only ever touches branches it actually accumulated into) --
    a real full-store failure this test regresses: a test atom matching
    such a branch must fall back gracefully, not raise KeyError."""
    import pandas as pd
    from sieve_experiments.predictors.dash import predict_via_data_storage_walk

    # branch 3 exists in data_storage but was never populated -- only its
    # own original DASH columns are present, none of ours
    df = pd.DataFrame({"level": [0, 1]})
    tree = _FakePropertyTree({3: df})

    pred = predict_via_data_storage_walk(tree, [[(3, 0)]], _literal_props())
    assert pred.atom_charge_std is not None
    np.testing.assert_allclose(pred.atom_profile, [[7.0]])
    np.testing.assert_allclose(pred.atom_charge_std, [0.5])


def test_predict_via_data_storage_walk_uses_the_deepest_populated_node():
    import pandas as pd
    from sieve_experiments.predictors.dash import predict_via_data_storage_walk

    df = pd.DataFrame({"sigma_bin_0": [1.0, 9.0], "sigma_charge_std": [0.1, 0.2]})
    tree = _FakePropertyTree({0: df})

    pred = predict_via_data_storage_walk(tree, [[(0, 0), (0, 1)]], _literal_props())
    assert pred.atom_charge_std is not None
    np.testing.assert_allclose(pred.atom_profile, [[9.0]])  # deepest node: id 1
    np.testing.assert_allclose(pred.atom_charge_std, [0.2])


def test_predict_via_data_storage_walk_backs_off_when_the_deepest_node_is_unpopulated():
    """The deepest node's row is entirely NaN (never matched by a training
    atom) -- must back off to the next-shallowest populated node, exactly
    get_property_noNAN's own semantics."""
    import pandas as pd
    from sieve_experiments.predictors.dash import predict_via_data_storage_walk

    df = pd.DataFrame({"sigma_bin_0": [3.0, np.nan], "sigma_charge_std": [0.4, np.nan]})
    tree = _FakePropertyTree({0: df})

    pred = predict_via_data_storage_walk(tree, [[(0, 0), (0, 1)]], _literal_props())
    assert pred.atom_charge_std is not None
    np.testing.assert_allclose(pred.atom_profile, [[3.0]])  # backed off to id 0
    np.testing.assert_allclose(pred.atom_charge_std, [0.4])


def test_predict_via_data_storage_walk_falls_back_when_path_is_all_unpopulated():
    import pandas as pd
    from sieve_experiments.predictors.dash import predict_via_data_storage_walk

    df = pd.DataFrame({"sigma_bin_0": [np.nan], "sigma_charge_std": [np.nan]})
    tree = _FakePropertyTree({0: df})

    pred = predict_via_data_storage_walk(tree, [[(0, 0)]], _literal_props())
    assert pred.atom_charge_std is not None
    np.testing.assert_allclose(pred.atom_profile, [[7.0]])
    np.testing.assert_allclose(pred.atom_charge_std, [0.5])
