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


def _write_store(tmp_path, rows):
    """A minimal store: one row per (smiles, dash_id, split, cluster, shard)."""
    import pandas as pd
    from experiments.data import mol_to_blob

    recs = []
    for smiles, did, split, cluster, shard in rows:
        m = _mol(smiles)
        for atom in m.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 0.1 * atom.GetAtomicNum())
        recs.append(
            {
                "mol": mol_to_blob(m),
                "dash_id": did,
                "conf_id": "c0",
                "net_charge": 0.0,
                "split": split,
                "cluster": cluster,
                "shard": shard,
            }
        )
    store = tmp_path / "s"
    store.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(store / "molecules.parquet")
    return "s", tmp_path


def test_annotate_collapse_adds_key_and_counts(tmp_path):
    """Two conformers of one molecule, plus its enantiomer, plus an unrelated
    molecule: the first three share a key, the fourth does not."""
    import pandas as pd
    from experiments.store_ops import annotate_collapse

    store, root = _write_store(
        tmp_path,
        [
            ("N[C@@H](C)C(=O)O", "d1", "train", 0, "s00"),
            ("N[C@@H](C)C(=O)O", "d1", "train", 0, "s00"),
            ("N[C@H](C)C(=O)O", "d2", "train", 0, "s00"),
            ("CCO", "d3", "train", 1, "s01"),
        ],
    )
    out = annotate_collapse(store, stores_root=root)
    assert out == {"rows": 4, "keys": 2}

    df = pd.read_parquet(root / store / "molecules.parquet")
    assert set(df["collapse_key"]).__len__() == 2
    ala = df[df["dash_id"].isin(["d1", "d2"])]
    assert set(ala["n_collapsed"]) == {3}
    assert set(ala["n_molecules"]) == {2}
    assert set(ala["n_enantiomer_forms"]) == {2}
    other = df[df["dash_id"] == "d3"]
    assert set(other["n_collapsed"]) == {1}
    assert set(other["n_enantiomer_forms"]) == {1}


def test_annotate_collapse_is_idempotent(tmp_path):
    from experiments.store_ops import annotate_collapse

    store, root = _write_store(tmp_path, [("CCO", "d1", "train", 0, "s00")])
    first = annotate_collapse(store, stores_root=root)
    second = annotate_collapse(store, stores_root=root)
    assert first == second


def test_annotate_collapse_refuses_a_group_that_straddles_a_shard(tmp_path):
    """Verified not to happen on the real corpus; asserted so a future one
    cannot break it silently (spec section 4)."""
    import pytest
    from experiments.store_ops import annotate_collapse

    store, root = _write_store(
        tmp_path,
        [
            ("CCO", "d1", "train", 0, "s00"),
            ("CCO", "d2", "train", 0, "s01"),
        ],
    )
    with pytest.raises(ValueError, match="straddles"):
        annotate_collapse(store, stores_root=root)
