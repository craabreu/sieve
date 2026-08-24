"""Fixtures shared across the suite."""

from dataclasses import replace

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import SieveConfig

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"SINGLE": 1, "DOUBLE": 2},
    max_wl_depth=2,
)


def simple_config(**kw):
    return replace(_BASE_CONFIG, **kw)


def chain_batch(n, d=1, seed=0, graphs=1):
    """`graphs` disjoint paths of n nodes each, alternating attributes."""
    rng = np.random.default_rng(seed)
    per = n
    total = per * graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for i in range(per - 1):
            src += [off + i, off + i + 1]
            dst += [off + i + 1, off + i]
        gid += [g] * per
    return NodeBatch(
        node_attrs=(np.arange(total) % 2).reshape(-1, 1).astype(np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)),
    )


def star_batch(n_leaves, d=1, seed=0, graphs=1):
    """`graphs` disjoint stars of one center and `n_leaves` leaves each.

    Unlike ``chain_batch`` (max degree 2 for every graph regardless of size),
    a star's max degree is ``n_leaves`` -- used to exercise batches whose max
    degree differs from another batch's, which chains alone never do.
    """
    rng = np.random.default_rng(seed)
    per = n_leaves + 1
    total = per * graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for leaf in range(1, per):
            src += [off, off + leaf]
            dst += [off + leaf, off]
        gid += [g] * per
    return NodeBatch(
        node_attrs=np.zeros((total, 1), np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)),
    )


def split_batch(batch, mask):
    """Take the sub-batch of nodes where `mask` is True, reindexing edges."""
    return batch[mask]


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
