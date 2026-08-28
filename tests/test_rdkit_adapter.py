import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import sieve
from sieve.batch import check_alignment
from sieve.config import SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_smiles
from sieve.refine import refine

SMILES = ["CCO", "c1ccccc1", "CC(=O)N", "CCl"]


def cfg_for(smiles, attrs=(("element",), ("aromatic", "hybridization"))):
    flat = [a for g in attrs for a in g]
    codes, edges = build_codes([Chem.MolFromSmiles(s) for s in smiles], flat)
    return SieveConfig(
        target_dim=1,
        attribute_levels=attrs,
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=2,
    )


def test_batch_shapes_match_the_molecules():
    cfg = cfg_for(SMILES)
    b = from_smiles(SMILES, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    assert b.n_nodes == sum(m.GetNumAtoms() for m in mols)
    assert b.n_edges == 2 * sum(m.GetNumBonds() for m in mols)


def test_edges_are_symmetric():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    fwd = {(int(a), int(c)) for a, c in zip(b.edge_src, b.edge_dst, strict=True)}
    assert all((c, a) in fwd for a, c in fwd)


def test_graph_id_separates_molecules():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    assert len(np.unique(b.graph_id)) == len(SMILES)
    # no edge may cross a molecule boundary
    assert np.all(b.graph_id[b.edge_src] == b.graph_id[b.edge_dst])


def test_unseen_category_gets_the_reserved_unknown_code():
    cfg = cfg_for(["CCO"])  # codes learned from C, O, H only
    b = from_smiles(["CCl"], config=cfg)
    unknown = max(cfg.attribute_codes["element"].values()) + 1
    assert unknown in b.node_attrs[:, 0].tolist()


def test_unknown_atoms_fall_back_rather_than_colliding():
    cfg = cfg_for(["CCO"])
    train = from_smiles(["CCO"], y=np.zeros((3, 1)), config=cfg)
    m = sieve.fit(train, cfg)
    p = sieve.predict_detailed(m, from_smiles(["CCl"], config=cfg))
    assert (p.matched_level == -1).any()


def test_wl_labels_agree_with_morgan_atom_environments():
    """literature.md 4.6: WL identifiers on molecules ARE ECFP identifiers, so
    RDKit gives an independent check on the encoder."""
    smis = ["CCO", "CCC", "c1ccccc1C", "CC(=O)O"]
    cfg = cfg_for(
        smis, attrs=(("element", "degree", "formal_charge", "num_h", "aromatic"),)
    )
    b = from_smiles(smis, config=cfg)
    ours = refine(b, cfg)[len(cfg.attribute_levels)].labels  # WL round 1
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=1, includeChirality=False)
    theirs, off = [], 0
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateAtomToBits()
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        env = ao.GetAtomToBits()
        theirs += [env[i][-1] for i in range(mol.GetNumAtoms())]
        off += mol.GetNumAtoms()
    assert _same_partition(np.array(ours), np.array(theirs))


def test_alignment_guard_catches_a_shuffled_target_array():
    cfg = cfg_for(["CCO"])
    y = np.arange(3, dtype=float).reshape(-1, 1)
    b = from_smiles(["CCO"], y=y, config=cfg)
    assert b.elements is not None
    with pytest.raises(ValueError, match="element"):
        check_alignment(b, np.array([3]), b.elements[::-1].copy())


def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_y_from_atom_prop_reads_scalar_per_atom():
    from sieve.io.rdkit_adapter import from_rdkit

    mols = []
    for smi, charges in [("CO", [-0.2, 0.2]), ("CCO", [-0.1, 0.05, 0.05])]:
        mol = Chem.MolFromSmiles(smi)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("q", charge)
        mols.append(mol)

    cfg = cfg_for(["CO", "CCO"])
    b = from_rdkit(mols, config=cfg, y_from_atom_prop="q")

    assert b.y is not None
    assert b.y.shape == (5, 1)
    expected = [-0.2, 0.2, -0.1, 0.05, 0.05]
    assert b.y[:, 0].tolist() == pytest.approx(expected)


def test_y_from_atom_prop_respects_node_order():
    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    charges = [-0.2, 0.2]
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("q", charge)

    cfg = cfg_for(["CO"])
    reversed_order = [np.array([1, 0])]
    b = from_rdkit([mol], config=cfg, node_order=reversed_order, y_from_atom_prop="q")

    assert b.y is not None
    assert b.y[:, 0].tolist() == pytest.approx([0.2, -0.2])


def test_y_and_y_from_atom_prop_are_mutually_exclusive():
    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("q", 0.0)
    cfg = cfg_for(["CO"])

    with pytest.raises(ValueError, match="y_from_atom_prop"):
        from_rdkit([mol], y=np.zeros((2, 1)), config=cfg, y_from_atom_prop="q")
