"""Fixtures shared across the experiment-harness test suite (experiments/tests/).

Split out of tests/helpers.py so everything experiment-related stays
contained under experiments/ -- the core Sieve fixtures there
(simple_config/chain_batch/star_batch/split_batch) are unrelated to this
package and stay put.
"""

from __future__ import annotations

import numpy as np


def synthetic_molecule_set(n_mol=6, seed=0):
    """A small, fully-populated `MoleculeSet` for fast experiment-harness tests.

    No cosmolayer, no rdkit: atoms per molecule and every field are
    fabricated directly. Molecule-level truth is set exactly consistent with
    the atom-level truth (mol_profile = molecule_sum(atom_profile), etc.), so
    tests can check either the input or the rollup output.
    """
    from sieve_experiments.data import DEFAULT_GRID, MoleculeSet, molecule_sum

    rng = np.random.default_rng(seed)
    num_atoms = rng.integers(2, 6, size=n_mol)
    n_atoms = int(num_atoms.sum())
    grid = DEFAULT_GRID

    atom_profile = rng.random((n_atoms, grid.num_points)).astype(np.float32) + 0.1
    atom_area = atom_profile.sum(axis=1).astype(np.float64)
    atom_charge = rng.normal(scale=0.3, size=n_atoms)

    mol_id = np.repeat(np.arange(n_mol), num_atoms)
    net_charge = np.round(molecule_sum(atom_charge, mol_id, n_mol)).astype(np.float64)
    # Shift each molecule's atom charges so they already sum to its net_charge,
    # matching what a "none" reconciliation predictor would output exactly.
    residual = net_charge - molecule_sum(atom_charge, mol_id, n_mol)
    atom_charge = atom_charge + (residual / num_atoms)[mol_id]

    mol_profile = molecule_sum(atom_profile, mol_id, n_mol)
    mol_area = molecule_sum(atom_area, mol_id, n_mol)
    mol_charge = molecule_sum(atom_charge, mol_id, n_mol)

    smiles = [f"C{i}" for i in range(n_mol)]  # placeholders, never parsed
    return MoleculeSet(
        smiles=smiles,
        num_atoms=num_atoms,
        net_charge=net_charge,
        grid=grid,
        mol_profile=mol_profile,
        mol_area=mol_area,
        mol_charge=mol_charge,
        atom_profile=atom_profile,
        atom_area=atom_area,
        atom_charge=atom_charge,
    )
