"""The equivalence key: what it merges, and what it must not."""

from __future__ import annotations

from rdkit import Chem


def _mol(smiles: str):
    """A molecule with explicit hydrogens, as the store holds them."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_same_molecule_written_differently_shares_a_key():
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("CCO")) == collapse_key(_mol("OCC"))


def test_enantiomers_share_a_key():
    """Measured interchangeable (0.00780 per-atom RMS, the noise floor), and
    no featurization in this series reads chirality."""
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("N[C@@H](C)C(=O)O")) == collapse_key(
        _mol("N[C@H](C)C(=O)O")
    )


def test_diastereomers_do_not_share_a_key():
    """Not mirror images, and measured to differ by 0.00698 above the
    enantiomer control -- merging them would hide a real error."""
    from experiments.collapse import collapse_key

    a = _mol("C[C@H](O)[C@H](N)C")
    b = _mol("C[C@H](O)[C@@H](N)C")
    assert collapse_key(a) != collapse_key(b)


def test_ez_isomers_do_not_share_a_key():
    """mirror_mol leaves bond stereo untouched, so E and Z keys differ."""
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("C/C=C/C")) != collapse_key(_mol("C/C=C\\\\C"))


def test_achiral_molecule_key_is_its_own_canonical_smiles():
    from experiments.collapse import collapse_key

    m = _mol("c1ccccc1")
    assert collapse_key(m) == Chem.MolToSmiles(m)


def test_mirror_of_a_mirror_is_the_original():
    from experiments.collapse import mirror_mol

    m = _mol("N[C@@H](C)C(=O)O")
    twice = mirror_mol(mirror_mol(m))
    assert Chem.MolToSmiles(twice) == Chem.MolToSmiles(m)
