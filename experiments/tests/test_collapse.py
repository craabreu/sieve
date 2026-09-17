"""The equivalence key: what it merges, and what it must not."""

from __future__ import annotations

import pytest
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


def _charged(smiles: str, charges: dict[int, float]):
    """A molecule whose MBIScharge is set per atom index."""
    m = _mol(smiles)
    for atom in m.GetAtoms():
        atom.SetDoubleProp("MBIScharge", charges.get(atom.GetIdx(), 0.0))
    return m


def test_collapse_averages_conformers_of_one_molecule():
    import numpy as np
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0})
    b = _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0})
    key = collapse_key(a)
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={
            "collapse_key": [key, key],
            "dash_id": ["d1", "d1"],
            "conf_id": ["c0", "c1"],
        },
    )
    out = collapse_molecule_set(mset)
    assert out.n_conformers == 1
    heavy = out.atom_target[:3]
    np.testing.assert_allclose(sorted(heavy), sorted([2.0, 3.0, 4.0]))


def test_collapse_leaves_distinct_keys_alone():
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0})
    b = _charged("CCC", {0: 1.0})
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={
            "collapse_key": [collapse_key(a), collapse_key(b)],
            "dash_id": ["d1", "d2"],
            "conf_id": ["c0", "c0"],
        },
    )
    assert collapse_molecule_set(mset).n_conformers == 2


def test_collapse_matches_enantiomer_atoms_through_the_mirror():
    """The correspondence is the point: averaging positionally would pair
    atoms that are not the same atom."""
    import numpy as np
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("N[C@@H](C)C(=O)O", {})
    b = _charged("N[C@H](C)C(=O)O", {})
    for atom in a.GetAtoms():
        atom.SetDoubleProp("MBIScharge", float(atom.GetAtomicNum()))
    for atom in b.GetAtoms():
        atom.SetDoubleProp("MBIScharge", float(atom.GetAtomicNum()))
    key = collapse_key(a)
    assert key == collapse_key(b)
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={
            "collapse_key": [key, key],
            "dash_id": ["d1", "d2"],
            "conf_id": ["c0", "c0"],
        },
    )
    out = collapse_molecule_set(mset)
    assert out.n_conformers == 1
    # charges were set to the atomic number, identical under any correct
    # correspondence, so every averaged value must still be an integer
    np.testing.assert_allclose(out.atom_target, np.round(out.atom_target))


def test_collapse_equals_fractional_weighting_for_the_mean():
    """The equivalence that justifies the design (spec section 5): the
    collapsed class mean equals the 1/n_collapsed-weighted mean of all rows."""
    import numpy as np
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    rows = [
        _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
        _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
        _charged("CCO", {0: 5.0, 1: 6.0, 2: 7.0}),
    ]
    key = collapse_key(rows[0])
    mset = MoleculeSet(
        mols=rows,
        atom_property="MBIScharge",
        ids={
            "collapse_key": [key] * 3,
            "dash_id": ["d1"] * 3,
            "conf_id": ["c0", "c1", "c2"],
        },
    )
    collapsed = sorted(collapse_molecule_set(mset).atom_target[:3])
    # the 1/n-weighted mean of all rows, computed independently per atom index
    fractional = sorted(
        sum(m.GetAtomWithIdx(i).GetDoubleProp("MBIScharge") for m in rows) / len(rows)
        for i in range(3)
    )
    np.testing.assert_allclose(collapsed, fractional)
    np.testing.assert_allclose(collapsed, [3.0, 4.0, 5.0])


def test_weight_by_collapse_repeats_each_representative():
    """The migration setting: weighting by n_collapsed must reproduce the
    uncollapsed row count and leave the mean unchanged (spec section 9)."""
    import numpy as np
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    rows = [
        _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
        _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
        _charged("CCC", {0: 9.0}),
    ]
    # list[str | None], not the inferred list[str]: MoleculeSet.ids expects
    # Mapping[str, list[str | None]], and a variable assigned before the dict
    # literal can't be widened by the dict's own expected-type context the way
    # an inline list literal can.
    keys: list[str | None] = [collapse_key(m) for m in rows]
    mset = MoleculeSet(
        mols=rows,
        atom_property="MBIScharge",
        ids={
            "collapse_key": keys,
            "dash_id": ["d1", "d1", "d2"],
            "conf_id": ["c0", "c1", "c0"],
        },
    )
    plain = collapse_molecule_set(mset)
    weighted = collapse_molecule_set(mset, weight_by_collapse=True)

    assert plain.n_conformers == 2  # two keys
    assert weighted.n_conformers == 3  # CCO twice, CCC once
    # the CCO representative carries the same averaged target either way
    np.testing.assert_allclose(
        sorted(plain.atom_target[: rows[0].GetNumAtoms()]),
        sorted(weighted.atom_target[: rows[0].GetNumAtoms()]),
    )


def test_fit_sieve_shard_collapses_only_the_training_side(tmp_path):
    """The fit sees one row per key; the store and every held-out path still
    hold one row per conformer."""
    import json

    import pandas as pd
    from experiments.cv import fit_sieve_shard
    from experiments.predictors.sieve_predictor import _build_config, save_codes
    from experiments.store_ops import annotate_collapse

    from experiments.tests.helpers import synthetic_molecule_set

    store, root = _write_store(
        tmp_path,
        [
            ("CCO", "d1", "train", 0, "s00"),
            ("CCO", "d1", "train", 0, "s00"),
            ("CCC", "d2", "train", 0, "s00"),
        ],
    )
    annotate_collapse(store, stores_root=root)

    whole = synthetic_molecule_set(n_mol=4, seed=0)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=1,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    out = fit_sieve_shard(
        store=store,
        shard="s00",
        depth=1,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=tmp_path / "runs",
        stores_root=root,
        allow_dirty=True,
        collapse=True,
    )
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert manifest["n_train_conformers"] == 2  # 3 rows -> 2 keys
    assert manifest["collapse"] is True

    # the store itself is untouched
    assert len(pd.read_parquet(root / store / "molecules.parquet")) == 3


def test_held_out_floor_is_the_within_key_scatter():
    import numpy as np
    from experiments.collapse import collapse_key, held_out_floor
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0, 1: 1.0, 2: 1.0})
    b = _charged("CCO", {0: 3.0, 1: 3.0, 2: 3.0})
    c = _charged("CCC", {0: 5.0})
    key_ab, key_c = collapse_key(a), collapse_key(c)
    mset = MoleculeSet(
        mols=[a, b, c],
        atom_property="MBIScharge",
        ids={
            "collapse_key": [key_ab, key_ab, key_c],
            "dash_id": ["d1", "d2", "d3"],
            "conf_id": ["c0", "c0", "c0"],
        },
    )
    # _charged only sets the 3 heavy-atom indices (0, 1, 2); CCO's 6 hydrogens
    # (indices 3-8) default to 0.0 in BOTH a and b, so only the 3 heavy atoms
    # actually differ -- by 2.0 each, so each deviates by 1.0 from their mean.
    # The denominator is every atom in the whole held-out set (matching how
    # RMSE is computed for the fold elsewhere), not just the differing ones;
    # CCC is alone in its group and contributes nothing to the numerator, but
    # its atoms still count in the denominator.
    floor = held_out_floor(mset)
    n_heavy_differing = 3 * 2  # 3 heavy atoms, x2 molecules (a and b)
    total_atoms = a.GetNumAtoms() + b.GetNumAtoms() + c.GetNumAtoms()
    expected = np.sqrt(n_heavy_differing * 1.0**2 / total_atoms)
    assert floor == pytest.approx(expected)


def test_held_out_floor_is_zero_without_duplicates():
    from experiments.collapse import collapse_key, held_out_floor
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0})
    b = _charged("CCC", {0: 2.0})
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={
            "collapse_key": [collapse_key(a), collapse_key(b)],
            "dash_id": ["d1", "d2"],
            "conf_id": ["c0", "c0"],
        },
    )
    assert held_out_floor(mset) == pytest.approx(0.0)
