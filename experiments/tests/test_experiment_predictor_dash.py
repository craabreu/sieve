"""Tests for the DASH Stage A back-off algorithm (predictors/dash.py).

The fitting/prediction algorithm itself (``fit_backoff``/``predict_backoff``)
is pure numpy over pre-computed tree paths -- no rdkit, no DASH-tree clone --
so it runs in the fast suite. The atom-matching path (RDKit + DASHTree) is
exercised only in the optional-data suite (needs the DASH-tree clone under
experiments/external/ and the real store), same pattern as
test_experiment_store.py (this directory).
"""

from __future__ import annotations

import numpy as np
import pytest
from sieve_experiments.predictors.dash import fit_backoff, predict_backoff

from experiments.tests.helpers import synthetic_molecule_set

# --- fit_backoff / predict_backoff -----------------------------------------


def test_predict_uses_deepest_node_with_enough_support():
    # Two atoms share path [(0, 0), (0, 1)]; both retained at minimum_support=2.
    paths = [[(0, 0), (0, 1)], [(0, 0), (0, 1)]]
    atom_profile = np.array([[1.0, 0.0], [3.0, 0.0]])
    atom_area = np.array([1.0, 3.0])
    atom_charge = np.array([0.1, 0.3])

    stats = fit_backoff(paths, atom_profile, atom_area, atom_charge, minimum_support=2)
    pred = predict_backoff([[(0, 0), (0, 1)]], stats)
    assert pred.atom_area is not None
    assert pred.atom_charge is not None

    # deepest node (0, 1) mean: profile (2, 0), area 2, charge 0.2
    np.testing.assert_allclose(pred.atom_profile, [[2.0, 0.0]])
    np.testing.assert_allclose(pred.atom_area, [2.0])
    np.testing.assert_allclose(pred.atom_charge, [0.2])


def test_predict_backs_off_when_deepest_node_lacks_support():
    # (0, 0) is an ancestor of every path (support 3); (0, 1) only ever seen
    # once, alone -- below minimum_support and so never retained.
    paths = [
        [(0, 0)],
        [(0, 0)],
        [(0, 0), (0, 1)],
    ]
    atom_profile = np.array([[1.0], [3.0], [10.0]])
    atom_area = np.array([1.0, 3.0, 10.0])
    atom_charge = np.array([0.1, 0.3, 1.0])

    stats = fit_backoff(paths, atom_profile, atom_area, atom_charge, minimum_support=2)
    assert (0, 1) not in stats.nodes  # pruned: only 1 supporting atom
    # a new atom whose deepest match is (0, 1) -- unsupported -- must back
    # off to (0, 0), not use (0, 1)'s single-sample stats.
    pred = predict_backoff([[(0, 0), (0, 1)]], stats)
    assert pred.atom_area is not None
    assert pred.atom_charge is not None

    # (0, 0) mean over all 3 atoms (it's on every path): profile/area/charge
    np.testing.assert_allclose(pred.atom_profile, [[14.0 / 3]])
    np.testing.assert_allclose(pred.atom_area, [14.0 / 3])
    np.testing.assert_allclose(pred.atom_charge, [1.4 / 3])


def test_predict_falls_back_to_global_mean_when_no_node_retained():
    paths = [[(0, 0)], [(0, 0)]]
    atom_profile = np.array([[1.0], [3.0]])
    atom_area = np.array([1.0, 3.0])
    atom_charge = np.array([0.1, 0.3])

    # minimum_support higher than any node's count -> nothing retained
    stats = fit_backoff(
        paths, atom_profile, atom_area, atom_charge, minimum_support=100
    )
    pred = predict_backoff([[(0, 0)], [(7, 9)]], stats)
    assert pred.atom_area is not None
    assert pred.atom_charge is not None

    # both fall back to the unconditional global mean: profile 2.0, area 2.0
    np.testing.assert_allclose(pred.atom_profile, [[2.0], [2.0]])
    np.testing.assert_allclose(pred.atom_area, [2.0, 2.0])
    np.testing.assert_allclose(pred.atom_charge, [0.2, 0.2])


def test_predict_falls_back_to_global_mean_for_unseen_branch():
    paths = [[(0, 0)], [(0, 0)]]
    atom_profile = np.array([[1.0], [3.0]])
    atom_area = np.array([1.0, 3.0])
    atom_charge = np.array([0.1, 0.3])

    stats = fit_backoff(paths, atom_profile, atom_area, atom_charge, minimum_support=1)
    # a path in a branch never seen during fit
    pred = predict_backoff([[(99, 0)]], stats)

    np.testing.assert_allclose(pred.atom_profile, [[2.0]])


def test_charge_std_is_positive_and_finite():
    paths = [[(0, 0)], [(0, 0)], [(0, 0)]]
    atom_profile = np.array([[1.0], [2.0], [3.0]])
    atom_area = np.array([1.0, 2.0, 3.0])
    atom_charge = np.array([-0.5, 0.0, 0.5])

    stats = fit_backoff(paths, atom_profile, atom_area, atom_charge, minimum_support=1)
    pred = predict_backoff([[(0, 0)]], stats)

    assert pred.atom_charge_std is not None
    assert np.all(np.isfinite(pred.atom_charge_std))
    assert np.all(pred.atom_charge_std > 0)


# --- DASHBackoffPredictor wiring (no rdkit/DASH-tree needed) ---------------


def test_fit_atoms_coerces_string_stores_root_to_a_path(monkeypatch):
    """predictor.params comes straight from YAML, so stores_root arrives as a
    plain str -- load_atom_truth's ``stores_root / store_name`` would raise
    TypeError on a str, not a Path. Regression test for that."""
    from pathlib import Path

    from sieve_experiments.predictors.dash import DASHBackoffPredictor

    captured = {}

    def fake_load_atom_truth(store, *, scheme, smiles, num_atoms, stores_root):
        captured["stores_root"] = stores_root
        return (
            np.zeros((sum(num_atoms), 1)),
            np.zeros(sum(num_atoms)),
            np.zeros(sum(num_atoms)),
        )

    monkeypatch.setattr(
        "sieve_experiments.predictors.dash.load_atom_truth", fake_load_atom_truth
    )
    predictor = DASHBackoffPredictor(
        store="chaos-store", scheme="cosmo-sac-2010", stores_root="stores"
    )
    monkeypatch.setattr(
        predictor,
        "_paths_for",
        lambda mset, *, split: [[] for _ in range(mset.n_atoms)],
    )

    ms = synthetic_molecule_set(n_mol=2, seed=0)
    predictor.fit_atoms(ms, ms, rng=np.random.default_rng(0))

    assert isinstance(captured["stores_root"], Path)


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


def test_fit_backoff_requires_matching_lengths():
    with pytest.raises(ValueError):
        fit_backoff(
            [[(0, 0)]],
            np.array([[1.0], [2.0]]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            minimum_support=1,
        )
