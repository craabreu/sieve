"""Fixtures shared across charge_experiments' test suite."""

from __future__ import annotations

import numpy as np


def synthetic_molecule_set(n_mol: int = 8, seed: int = 0):
    """A small, fully-populated ``MoleculeSet`` for fast harness tests --
    real RDKit ``Mol`` objects (small alkanes/alcohols), each atom carrying a
    fabricated but deterministic ``MBIScharge``, net_charge computed to be
    exactly consistent with it (sum of atom charges), so both the input and
    any rollup can be checked exactly."""
    from charge_experiments.data import MoleculeSet, molecule_sum
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    base_smiles = ["CO", "CCO", "CCC", "CC(C)O", "CCCC", "CC(=O)O", "CCN", "CCCl"]
    smiles = [base_smiles[i % len(base_smiles)] for i in range(n_mol)]

    mols = []
    num_atoms = []
    for smi in smiles:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        assert mol is not None, smi
        n_atoms = mol.GetNumAtoms()
        charges = rng.normal(scale=0.2, size=n_atoms)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("MBIScharge", float(charge))
        mols.append(mol)
        num_atoms.append(n_atoms)

    atom_charge = np.concatenate(
        [np.array([a.GetDoubleProp("MBIScharge") for a in m.GetAtoms()]) for m in mols]
    )
    mol_id = np.repeat(np.arange(n_mol), num_atoms)
    net_charge = molecule_sum(atom_charge, mol_id, n_mol)

    chembl_id: list[str | None] = [
        f"CHEMBL{1000 + i // 2}" for i in range(n_mol)
    ]  # 2 conformers/id
    conf_id = [f"conf_{i % 2:02d}" for i in range(n_mol)]

    return MoleculeSet(
        chembl_id=chembl_id,
        conf_id=conf_id,
        mols=mols,
        net_charge=net_charge,
        dash_id=[None] * n_mol,  # every row is chembl_id-schema in this fixture
    )
