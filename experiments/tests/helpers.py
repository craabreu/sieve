"""Fixtures shared across experiments' test suite."""

from __future__ import annotations

import numpy as np


def synthetic_molecule_set(
    n_mol: int = 8, seed: int = 0, atom_property: str = "MBIScharge"
):
    """A small, fully-populated ``MoleculeSet`` for fast harness tests --
    real RDKit ``Mol`` objects (small alkanes/alcohols), each atom carrying a
    fabricated but deterministic value of ``atom_property``, with
    ``molecule_value`` computed to be exactly its per-molecule sum, so both
    the input and any rollup can be checked exactly."""
    from experiments.data import MoleculeSet, molecule_sum
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
            atom.SetDoubleProp(atom_property, float(charge))
        mols.append(mol)
        num_atoms.append(n_atoms)

    atom_value = np.concatenate(
        [np.array([a.GetDoubleProp(atom_property) for a in m.GetAtoms()]) for m in mols]
    )
    mol_id = np.repeat(np.arange(n_mol), num_atoms)
    net_charge = molecule_sum(atom_value, mol_id, n_mol)

    chembl_id: list[str | None] = [
        f"CHEMBL{1000 + i // 2}" for i in range(n_mol)
    ]  # 2 conformers/id
    conf_id: list[str | None] = [f"conf_{i % 2:02d}" for i in range(n_mol)]

    return MoleculeSet(
        mols=mols,
        atom_property=atom_property,
        molecule_property="net_charge",
        molecule_value=net_charge,
        ids={
            "chembl_id": chembl_id,
            "conf_id": conf_id,
            "dash_id": [None] * n_mol,
        },
    )


def real_store_has_columns(store: str, *columns: str) -> bool:
    """True when the real (locally prepared) ``store``'s parquet carries
    every one of ``columns``.

    Optional tests that read the real store gate on this rather than on the
    parquet merely *existing*: a parsed-but-not-yet-split store is now a
    supported, reachable state (``prepare-store --stop-before-split``, the
    state ``cluster-report`` reads), and a gate that only checks existence
    lets such a store through to fail on a bare ``KeyError: 'split'``
    instead of skipping. Same principle the CV workflow's own guards
    follow: check the artifact's shape, not its presence.
    """
    from experiments.data import DEFAULT_STORES_ROOT

    path = DEFAULT_STORES_ROOT / store / "molecules.parquet"
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return False
    return set(columns) <= set(pq.ParquetFile(path).schema.names)
