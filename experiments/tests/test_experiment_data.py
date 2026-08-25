"""Tests for experiments/sieve_experiments/data.py (MoleculeSet, molecule_sum)."""

from __future__ import annotations

import numpy as np
import pytest
from sieve_experiments.data import DEFAULT_GRID, MoleculeSet, molecule_sum

from experiments.tests.helpers import synthetic_molecule_set


def hand_built_set():
    """4 molecules, small enough to hand-verify every field."""
    num_atoms = np.array([2, 1, 3, 2])
    net_charge = np.array([0.0, 1.0, -1.0, 0.0])
    atom_area = np.array([1.0, 2.0, 5.0, 0.5, 1.5, 2.0, 3.0, 4.0])
    atom_charge = np.array([0.0, 0.0, 1.0, -0.5, -0.5, 0.0, 0.1, -0.1])
    grid = DEFAULT_GRID
    atom_profile = np.zeros((8, grid.num_points))
    # give each atom's profile a distinct total equal to its area, all mass
    # in bin 0, so molecule_sum is trivial to check by hand.
    atom_profile[:, 0] = atom_area
    return MoleculeSet(
        smiles=["A", "B", "C", "D"],
        num_atoms=num_atoms,
        net_charge=net_charge,
        grid=grid,
        atom_profile=atom_profile,
        atom_area=atom_area,
        atom_charge=atom_charge,
    )


# --- molecule_sum ------------------------------------------------------


def test_molecule_sum_matches_hand_computation():
    ms = hand_built_set()
    mol_id = ms.atom_mol_id
    out = molecule_sum(ms.atom_area, mol_id, ms.n_molecules)
    np.testing.assert_allclose(out, [3.0, 5.0, 4.0, 7.0])


def test_molecule_sum_is_sum_not_average():
    ms = hand_built_set()
    mol_id = ms.atom_mol_id
    out = molecule_sum(ms.atom_charge, mol_id, ms.n_molecules)
    np.testing.assert_allclose(out, [0.0, 1.0, -1.0, 0.0])


def test_molecule_sum_2d_rows():
    ms = hand_built_set()
    mol_id = ms.atom_mol_id
    out = molecule_sum(ms.atom_profile, mol_id, ms.n_molecules)
    assert out.shape == (4, ms.grid.num_points)
    np.testing.assert_allclose(out[:, 0], [3.0, 5.0, 4.0, 7.0])
    np.testing.assert_allclose(out[:, 1:], 0.0)


# --- atom_mol_id ---------------------------------------------------------


def test_atom_mol_id_layout():
    ms = hand_built_set()
    np.testing.assert_array_equal(ms.atom_mol_id, [0, 0, 1, 2, 2, 2, 3, 3])
    assert ms.n_atoms == 8


def test_screening_charge_is_negated_net_charge():
    ms = MoleculeSet(
        smiles=["a", "b", "c"],
        num_atoms=np.array([1, 1, 1]),
        net_charge=np.array([1.0, -1.0, 0.0]),
    )
    np.testing.assert_array_equal(ms.screening_charge, [-1.0, 1.0, 0.0])


# --- select --------------------------------------------------------------


def test_select_slices_atoms_to_the_right_molecules():
    ms = hand_built_set()
    sub = ms.select(np.array([True, False, True, False]))
    assert sub.smiles == ["A", "C"]
    np.testing.assert_array_equal(sub.num_atoms, [2, 3])
    np.testing.assert_allclose(sub.net_charge, [0.0, -1.0])
    # molecule A's 2 atoms, then molecule C's 3 atoms
    np.testing.assert_allclose(sub.atom_area, [1.0, 2.0, 0.5, 1.5, 2.0])
    np.testing.assert_array_equal(sub.atom_mol_id, [0, 0, 1, 1, 1])


def test_select_preserves_grid_and_handles_none_fields():
    ms = hand_built_set()
    sub = ms.select(np.array([True, True, False, False]))
    assert sub.grid is ms.grid
    assert sub.mol_profile is None  # never set on hand_built_set
    assert sub.atom_area is not None


def test_select_full_mask_round_trips():
    ms = hand_built_set()
    sub = ms.select(np.ones(4, dtype=bool))
    assert sub.smiles == ms.smiles
    np.testing.assert_array_equal(sub.num_atoms, ms.num_atoms)
    np.testing.assert_allclose(sub.atom_area, ms.atom_area)


# --- synthetic_molecule_set fixture (used by the smoke test) -------------


def test_synthetic_molecule_set_is_internally_consistent():
    ms = synthetic_molecule_set(n_mol=6, seed=0)
    mol_id = ms.atom_mol_id
    np.testing.assert_allclose(
        molecule_sum(ms.atom_profile, mol_id, ms.n_molecules), ms.mol_profile
    )
    np.testing.assert_allclose(
        molecule_sum(ms.atom_area, mol_id, ms.n_molecules), ms.mol_area
    )
    np.testing.assert_allclose(
        molecule_sum(ms.atom_charge, mol_id, ms.n_molecules),
        ms.mol_charge,
        atol=1e-10,
    )
    # net_charge is the known input, and the fixture's atom charges already
    # sum to it -- exactly what a "none" charge-reconciliation predictor
    # would produce.
    np.testing.assert_allclose(ms.mol_charge, ms.net_charge, atol=1e-8)


def test_synthetic_molecule_set_reproducible_with_seed():
    a = synthetic_molecule_set(n_mol=6, seed=0)
    b = synthetic_molecule_set(n_mol=6, seed=0)
    np.testing.assert_array_equal(a.num_atoms, b.num_atoms)
    np.testing.assert_allclose(a.atom_profile, b.atom_profile)


# --- select_atoms_by_smiles -----------------------------------------------
#
# The join DASH (and later Sieve) uses to recover atom-level truth for a
# split: load_molecule_set never populates atom_* (design.md), so a predictor
# that needs it loads the full store's atom truth itself and joins back onto
# its train/test MoleculeSet by SMILES -- never by position, same idiom the
# design doc calls for on COSMO-NET's output join (risk #4).


def _full_store():
    full_smiles = ["A", "B", "C", "D"]
    full_num_atoms = np.array([2, 1, 3, 2])
    # atom arrays laid out in full-store order, values chosen so each atom's
    # value is easy to trace back to its (molecule, position).
    atom_profile = np.arange(8, dtype=np.float64)[:, None]
    atom_area = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    arrays = {"atom_profile": atom_profile, "atom_area": atom_area}
    return full_smiles, full_num_atoms, arrays


def test_select_atoms_by_smiles_reorders_and_subsets():
    from sieve_experiments.data import select_atoms_by_smiles

    full_smiles, full_num_atoms, arrays = _full_store()
    # wanted order: C then A (subset, reversed relative to store order)
    out = select_atoms_by_smiles(
        full_smiles,
        full_num_atoms,
        arrays,
        wanted_smiles=["C", "A"],
        wanted_num_atoms=np.array([3, 2]),
    )
    # C's atoms are 40,50,60 (offset 3:6); A's are 10,20 (offset 0:2)
    np.testing.assert_allclose(out["atom_area"], [40.0, 50.0, 60.0, 10.0, 20.0])


def test_select_atoms_by_smiles_rejects_atom_count_mismatch():
    from sieve_experiments.data import select_atoms_by_smiles

    full_smiles, full_num_atoms, arrays = _full_store()
    with pytest.raises(ValueError, match="atom count"):
        select_atoms_by_smiles(
            full_smiles,
            full_num_atoms,
            arrays,
            wanted_smiles=["A"],
            wanted_num_atoms=np.array([99]),
        )


def test_select_atoms_by_smiles_rejects_unknown_smiles():
    from sieve_experiments.data import select_atoms_by_smiles

    full_smiles, full_num_atoms, arrays = _full_store()
    with pytest.raises(KeyError, match="not found"):
        select_atoms_by_smiles(
            full_smiles,
            full_num_atoms,
            arrays,
            wanted_smiles=["Z"],
            wanted_num_atoms=np.array([1]),
        )


def test_select_atoms_by_smiles_rejects_duplicate_smiles_in_store():
    from sieve_experiments.data import select_atoms_by_smiles

    full_smiles = ["A", "A"]
    full_num_atoms = np.array([1, 1])
    arrays = {"atom_area": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="duplicate"):
        select_atoms_by_smiles(
            full_smiles,
            full_num_atoms,
            arrays,
            wanted_smiles=["A"],
            wanted_num_atoms=np.array([1]),
        )
