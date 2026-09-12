"""Pure-rdkit tests for data.py's Mol-blob serialize/deserialize round trip
and MoleculeSet -- no store, no download needed."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.data import MoleculeSet

from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.data import blob_to_mol, mol_to_blob

    mol = _mol_with_charges("CO", [-0.1, 0.1])
    blob = mol_to_blob(mol)
    assert isinstance(blob, bytes)

    restored = blob_to_mol(blob)
    assert restored.GetNumAtoms() == 2
    restored_charges = [a.GetDoubleProp("MBIScharge") for a in restored.GetAtoms()]
    assert restored_charges == pytest.approx([-0.1, 0.1])


def test_mol_to_blob_round_trip_preserves_chiral_tags():
    from experiments.data import blob_to_mol, mol_to_blob
    from rdkit import Chem

    mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
    assert mol is not None
    original_tags = [a.GetChiralTag() for a in mol.GetAtoms()]
    assert any(t != Chem.ChiralType.CHI_UNSPECIFIED for t in original_tags)

    restored = blob_to_mol(mol_to_blob(mol))
    restored_tags = [a.GetChiralTag() for a in restored.GetAtoms()]
    assert restored_tags == original_tags


def test_mol_to_blob_round_trip_preserves_bond_cip_codes():
    """rdCIPLabeler writes a double bond's canonical E/Z to that *bond's*
    own _CIPCode, which sieve's `bond_stereo` attribute reads. Serializing
    without BondProps dropped it silently: prepare_store ran the labeler and
    the mol-level CIP_LABELED_PROP marker survived, so _ensure_cip_labels
    short-circuited on load and every stored molecule read "none"."""
    from experiments.data import blob_to_mol, mol_to_blob
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    mol = Chem.MolFromSmiles("C/C=C/C")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    original = [b.GetPropsAsDict().get("_CIPCode") for b in mol.GetBonds()]
    assert "E" in original

    restored = blob_to_mol(mol_to_blob(mol))
    assert [b.GetPropsAsDict().get("_CIPCode") for b in restored.GetBonds()] == original


def test_synthetic_molecule_set_select_preserves_alignment():
    mset = synthetic_molecule_set(n_mol=8, seed=0)
    assert mset.n_conformers == 8
    assert mset.n_atoms == int(mset.num_atoms.sum())

    mask = np.array([True, False, True, False, True, False, True, False])
    sub = mset.select(mask)
    assert sub.n_conformers == 4
    assert sub.ids["chembl_id"] == [
        mset.ids["chembl_id"][i] for i in range(8) if mask[i]
    ]
    assert sub.ids["dash_id"] == [mset.ids["dash_id"][i] for i in range(8) if mask[i]]
    np.testing.assert_array_equal(sub.molecule_value, mset.molecule_value[mask])
    # atom_target stays consistent with molecule_value after selection
    from experiments.data import molecule_sum

    resummed = molecule_sum(sub.atom_target, sub.atom_mol_id, sub.n_conformers)
    np.testing.assert_allclose(resummed, sub.molecule_value, atol=1e-8)


def test_atom_target_reads_the_configured_property():
    from experiments.data import MoleculeSet
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CO")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp("alpha", float(i))
    mset = MoleculeSet(mols=[mol], atom_property="alpha")
    assert mset.atom_target.tolist() == [0.0, 1.0]


def test_missing_atom_property_raises_naming_it():
    from experiments.data import MoleculeSet
    from rdkit import Chem

    mset = MoleculeSet(mols=[Chem.MolFromSmiles("CO")], atom_property="alpha")
    with pytest.raises(KeyError, match="alpha"):
        _ = mset.atom_target


def test_molecule_value_is_none_when_unconfigured():
    mset = synthetic_molecule_set(n_mol=4)
    bare = MoleculeSet(mols=mset.mols, atom_property=mset.atom_property)
    assert bare.molecule_value is None
    assert bare.molecule_property is None


def test_ids_are_carried_and_subset_by_select():
    mset = synthetic_molecule_set(n_mol=4)
    assert set(mset.ids) == {"chembl_id", "conf_id", "dash_id"}
    sub = mset.select(np.array([True, False, True, False]))
    assert sub.n_conformers == 2
    assert sub.ids["conf_id"] == [mset.ids["conf_id"][0], mset.ids["conf_id"][2]]
    assert sub.atom_property == mset.atom_property
    assert sub.molecule_property == mset.molecule_property


def test_select_keeps_molecule_value_none_when_unset():
    mset = synthetic_molecule_set(n_mol=4)
    bare = MoleculeSet(mols=mset.mols, atom_property=mset.atom_property)
    assert bare.select(np.array([True, False, True, False])).molecule_value is None


def test_ids_length_mismatch_raises():
    mset = synthetic_molecule_set(n_mol=4)
    with pytest.raises(ValueError, match="conf_id"):
        MoleculeSet(
            mols=mset.mols,
            atom_property=mset.atom_property,
            ids={"conf_id": ["a", "b"]},
        )


def test_concat_molecule_sets_preserves_order_and_ids():
    from experiments.data import concat_molecule_sets

    a = synthetic_molecule_set(n_mol=3, seed=0)
    b = synthetic_molecule_set(n_mol=2, seed=1)

    combined = concat_molecule_sets([a, b])

    assert combined.n_conformers == a.n_conformers + b.n_conformers
    np.testing.assert_array_equal(
        combined.atom_target, np.concatenate([a.atom_target, b.atom_target])
    )
    assert list(combined.ids) == list(a.ids)
    for key in a.ids:
        assert combined.ids[key] == a.ids[key] + b.ids[key]
    np.testing.assert_array_equal(
        combined.molecule_value,
        np.concatenate([a.molecule_value, b.molecule_value]),
    )


def test_concat_molecule_sets_rejects_mismatched_atom_property():
    from experiments.data import concat_molecule_sets

    a = synthetic_molecule_set(n_mol=2, seed=0, atom_property="MBIScharge")
    b = synthetic_molecule_set(n_mol=2, seed=1, atom_property="alpha")

    with pytest.raises(ValueError, match="atom_property"):
        concat_molecule_sets([a, b])


def test_concat_molecule_sets_rejects_empty_list():
    from experiments.data import concat_molecule_sets

    with pytest.raises(ValueError, match="at least one"):
        concat_molecule_sets([])
