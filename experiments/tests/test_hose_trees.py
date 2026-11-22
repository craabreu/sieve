"""Tree keys for the HOSE lookup: the tree a HOSE code is written from, keyed
without the string. Skipped without the optional hosegen dependency, whose
charge code the atom labels share."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hosegen", reason="optional dependency: see the hose extra")

SMILES = (
    "OCC(O)CO",
    "c1ccc2ccccc2c1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[N+](C)(C)CC(=O)[O-]",
    "C1CC2CCC1C2",
    "C#CC1=CC=CC=C1",
    "C1CC1C1CCC1",
)


def _mol(smiles):
    from rdkit import Chem

    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _keys(mol, radius, closure):
    from experiments.predictors.hose_trees import tree_keys

    return tree_keys(mol, radius, closure=closure)


def _classes(keys_by_atom, k):
    """The partition of the atoms by their key at radius ``k``, as a set of
    frozensets of (molecule, atom) pairs."""
    groups: dict[str, set] = {}
    for where, keys in keys_by_atom:
        groups.setdefault(keys[k - 1], set()).add(where)
    return {frozenset(g) for g in groups.values()}


def _wl_classes(mols, k):
    """The Sieve classes with element and formal charge on the atoms and bond
    types on the bonds: k rounds of 1-WL refinement, as a partition."""
    labels = [
        [(a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms()] for mol in mols
    ]
    for _ in range(k):
        labels = [
            [
                (
                    lab[a.GetIdx()],
                    tuple(
                        sorted(
                            (str(b.GetBondType()), lab[b.GetOtherAtomIdx(a.GetIdx())])
                            for b in a.GetBonds()
                        )
                    ),
                )
                for a in mol.GetAtoms()
            ]
            for mol, lab in zip(mols, labels, strict=True)
        ]
    groups: dict = {}
    for m, lab in enumerate(labels):
        for i, x in enumerate(lab):
            groups.setdefault(x, set()).add((m, i))
    return {frozenset(g) for g in groups.values()}


@pytest.mark.parametrize("closure", [False, True])
def test_the_keys_do_not_depend_on_the_order_of_the_atoms(closure):
    from rdkit import Chem

    rng = np.random.default_rng(0)
    for smiles in SMILES:
        mol = _mol(smiles)
        order = [int(i) for i in rng.permutation(mol.GetNumAtoms())]
        moved = Chem.RenumberAtoms(mol, order)
        before = _keys(mol, 5, closure)
        after = _keys(moved, 5, closure)
        for j, i in enumerate(order):
            assert after[j] == before[i], (smiles, i)


def test_there_is_one_key_per_atom_and_radius():
    mol = _mol("OCC(O)CO")
    keys = _keys(mol, 4, True)
    assert len(keys) == mol.GetNumAtoms()
    assert all(len(k) == 4 for k in keys)


@pytest.mark.parametrize("closure", [False, True])
def test_a_key_is_cut_to_fewer_spheres_as_a_code_is(closure):
    """Every shorter key is a prefix of the longer one, cut by the same
    ``sphere_prefix`` as a code, so the backoff chain and the analytic
    training curve read tree keys as they read codes."""
    from experiments.predictors.hose_keys import sphere_prefix

    for smiles in SMILES:
        for keys in _keys(_mol(smiles), 6, closure):
            for k in range(1, 7):
                for j in range(1, k + 1):
                    assert sphere_prefix(keys[k - 1], j) == keys[j - 1]


def test_a_key_begins_with_the_element_of_its_atom():
    """The error breakdown reads element means off the one-sphere table by
    the text before the first ``-``, as it does for a HOSE code."""
    mol = _mol("C[N+](C)(C)CC(=O)[O-]")
    for closure in (False, True):
        for atom, keys in zip(mol.GetAtoms(), _keys(mol, 3, closure), strict=True):
            for key in keys:
                assert key.split("-", 1)[0] == atom.GetSymbol()


def test_without_closures_the_keys_are_the_sieve_classes():
    """The tree without ring closures is the unrolled tree of a Sieve label
    with element, formal charge and bond types."""
    mols = [_mol(s) for s in SMILES]
    keys = [
        ((m, i), k)
        for m, mol in enumerate(mols)
        for i, k in enumerate(_keys(mol, 5, False))
    ]
    for k in range(1, 6):
        assert _classes(keys, k) == _wl_classes(mols, k)


def test_the_closure_keys_refine_the_tree_keys():
    mols = [_mol(s) for s in SMILES]
    for mol in mols:
        plain, closed = _keys(mol, 5, False), _keys(mol, 5, True)
        for k in range(5):
            coarser: dict[str, str] = {}
            for p, c in zip(plain, closed, strict=True):
                assert coarser.setdefault(c[k], p[k]) == p[k]


@pytest.mark.parametrize("ring", ["C1CC1", "C1CCC1"])
def test_the_closure_tells_a_small_ring_from_cyclohexane(ring):
    """Two sphere-1 neighbors bonded to each other (cyclopropane), or reaching
    the same atom in sphere 2 (cyclobutane), are seen at two spheres with the
    closure and not without it."""
    small, large = _mol(ring), _mol("C1CCCCC1")
    for closure, differ in ((False, False), (True, True)):
        a = _keys(small, 2, closure)[0][1]
        b = _keys(large, 2, closure)[0][1]
        assert (a != b) == differ


def test_an_atom_reached_twice_in_one_sphere_is_written_under_both_walks():
    """The meeting rule is symmetric: in cyclobutane the opposite carbon is
    reached by both ring neighbors in sphere 2, and its hydrogens are written
    under both, so no walk is preferred and the key needs no tie-break. Those
    hydrogens are then reached by two walks in sphere 3, and are meetings
    too."""
    from experiments.predictors.hose_trees import closure_tree

    tree = closure_tree(_mol("C1CCC1"), 0, 3)
    meetings = [n for n in _walk(tree) if n.kind == "meet" and n.symbol == "C"]
    assert len(meetings) == 2
    assert {n.atom for n in meetings} == {2}
    for n in meetings:
        assert [c.kind for c in n.children if c.symbol == "H"] == ["meet", "meet"]


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def test_a_ring_atom_written_in_an_earlier_sphere_is_a_closure_with_nothing_under_it():
    from experiments.predictors.hose_trees import closure_tree

    tree = closure_tree(_mol("C1CC1"), 0, 3)
    closures = [n for n in _walk(tree) if n.kind == "closure"]
    assert len(closures) == 2
    assert all(not n.children for n in closures)


def test_a_radius_below_one_is_refused():
    with pytest.raises(ValueError, match="radius"):
        _keys(_mol("CC"), 0, True)
