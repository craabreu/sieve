"""Pure-rdkit tests for data.py's Mol-blob serialize/deserialize round trip
and MoleculeSet -- no store, no download needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _mol_with_charges(smiles: str, charges: list[float]):
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    assert mol.GetNumAtoms() == len(charges)
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)
    return mol


def test_mol_to_blob_round_trip_preserves_atom_properties():
    from charge_experiments.data import blob_to_mol, mol_to_blob

    mol = _mol_with_charges("CO", [-0.1, 0.1])
    blob = mol_to_blob(mol)
    assert isinstance(blob, bytes)

    restored = blob_to_mol(blob)
    assert restored.GetNumAtoms() == 2
    restored_charges = [a.GetDoubleProp("MBIScharge") for a in restored.GetAtoms()]
    assert restored_charges == pytest.approx([-0.1, 0.1])


def test_mol_to_blob_round_trip_preserves_chiral_tags():
    from charge_experiments.data import blob_to_mol, mol_to_blob
    from rdkit import Chem

    mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
    assert mol is not None
    original_tags = [a.GetChiralTag() for a in mol.GetAtoms()]
    assert any(t != Chem.ChiralType.CHI_UNSPECIFIED for t in original_tags)

    restored = blob_to_mol(mol_to_blob(mol))
    restored_tags = [a.GetChiralTag() for a in restored.GetAtoms()]
    assert restored_tags == original_tags


def test_synthetic_molecule_set_select_preserves_alignment():
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=0)
    assert mset.n_conformers == 8
    assert mset.n_atoms == int(mset.num_atoms.sum())

    mask = np.array([True, False, True, False, True, False, True, False])
    sub = mset.select(mask)
    assert sub.n_conformers == 4
    assert sub.chembl_id == [mset.chembl_id[i] for i in range(8) if mask[i]]
    assert sub.dash_id == [mset.dash_id[i] for i in range(8) if mask[i]]
    np.testing.assert_array_equal(sub.net_charge, mset.net_charge[mask])
    # atom_charge stays consistent with net_charge after selection
    from charge_experiments.data import molecule_sum

    resummed = molecule_sum(sub.atom_charge, sub.atom_mol_id, sub.n_conformers)
    np.testing.assert_allclose(resummed, sub.net_charge, atol=1e-8)
