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
    """A minimal store: one row per (smiles, dash_id, split, cluster, shard).

    pandas and pyarrow live in the `charges` extra, which CI's [dev,chem]
    install does not carry -- guarded the way test_cv_optional.py guards its
    own store writes.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

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
    pd = pytest.importorskip("pandas")

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

    pd = pytest.importorskip("pandas")

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
    c = _charged("CCC", {1: 5.0})  # the middle carbon: no symmetric partner
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

    # Charges respect each molecule's symmetry -- propane's middle carbon has
    # no partner -- so no class holds unequal values either.
    a = _charged("CCO", {0: 1.0})
    b = _charged("CCC", {1: 2.0})
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


# --------------------------------------------------------------------------
# The stereo-blind floor
# --------------------------------------------------------------------------


def _floor_set(pairs):
    """A MoleculeSet from (smiles, {atom_idx: charge}) pairs, keyed properly."""
    from experiments.collapse import collapse_key
    from experiments.data import MoleculeSet

    mols = [_charged(smi, ch) for smi, ch in pairs]
    return MoleculeSet(
        mols=mols,
        atom_property="MBIScharge",
        ids={
            "collapse_key": [collapse_key(m) for m in mols],
            "dash_id": [f"d{i}" for i in range(len(mols))],
            "conf_id": ["c0"] * len(mols),
        },
    )


def test_stereo_blind_floor_sees_an_ez_pair_the_keyed_floor_cannot():
    """E/Z isomers are different keys, so held_out_floor counts none of their
    scatter -- but no arm in this series reads bond stereo, so they collide
    in one class and that scatter is irreducible for them."""
    from experiments.collapse import held_out_floor, held_out_floor_stereo_blind

    mset = _floor_set([("C/C=C/CO", {0: 1.0, 1: 2.0}), ("C/C=C\\CO", {0: 3.0, 1: 4.0})])
    assert len(set(mset.ids["collapse_key"])) == 2  # genuinely separate keys
    assert held_out_floor(mset) == 0.0
    assert held_out_floor_stereo_blind(mset) > 0.0


def test_stereo_blind_floor_sees_a_diastereomer_pair():
    from experiments.collapse import held_out_floor, held_out_floor_stereo_blind

    mset = _floor_set(
        [
            ("N[C@@H](C)[C@H](O)C", {0: 1.0, 1: 2.0}),
            ("N[C@H](C)[C@H](O)C", {0: 3.0, 1: 4.0}),
        ]
    )
    assert len(set(mset.ids["collapse_key"])) == 2
    assert held_out_floor(mset) == 0.0
    assert held_out_floor_stereo_blind(mset) > 0.0


def test_stereo_blind_floor_is_never_below_the_keyed_floor():
    """Its groups are supersets of the keyed floor's -- stripping stereo can
    only merge -- so it can only be larger. The invariant that makes the
    pair trustworthy."""
    from experiments.collapse import held_out_floor, held_out_floor_stereo_blind

    mset = _floor_set(
        [
            ("CCO", {0: 1.0, 1: 2.0}),
            ("CCO", {0: 1.5, 1: 2.5}),  # same key: conformer-like scatter
            ("C/C=C/CO", {0: 1.0}),
            ("C/C=C\\CO", {0: 2.0}),  # different keys, same stripped graph
            ("CCC", {0: 0.5}),  # a singleton
        ]
    )
    assert held_out_floor_stereo_blind(mset) >= held_out_floor(mset) > 0.0


def test_stereo_blind_floor_needs_no_collapse_key():
    """It derives its own grouping, so it works on an un-annotated set."""
    from experiments.collapse import held_out_floor, held_out_floor_stereo_blind
    from experiments.data import MoleculeSet

    mols = [_charged("C/C=C/CO", {0: 1.0}), _charged("C/C=C\\CO", {0: 2.0})]
    mset = MoleculeSet(mols=mols, atom_property="MBIScharge")
    assert held_out_floor(mset) == 0.0  # no key at all -> 0.0, as documented
    assert held_out_floor_stereo_blind(mset) > 0.0


def test_held_out_floors_reports_both():
    from experiments.collapse import held_out_floors

    out = held_out_floors(_floor_set([("CCO", {0: 1.0}), ("CCO", {0: 2.0})]))
    assert set(out) == {"floor/rmse", "floor/rmse_stereo_blind"}
    assert out["floor/rmse"] > 0.0


def test_floor_components_sum_to_the_directly_computed_floors():
    """The decomposition the per-shard cache rests on: a union's floors are
    recoverable from its parts' components, because no group spans a part."""
    from experiments.collapse import (
        floor_components,
        floors_from_components,
        held_out_floors,
    )
    from experiments.data import MoleculeSet

    # two disjoint "shards", each self-contained: no key spans them
    a = _floor_set([("CCO", {0: 1.0}), ("CCO", {0: 2.0}), ("CCC", {0: 0.5})])
    b = _floor_set(
        [("C/C=C/CO", {0: 1.0}), ("C/C=C\\CO", {0: 2.0}), ("CCN", {0: 0.25})]
    )
    union = MoleculeSet(
        mols=[*a.mols, *b.mols],
        atom_property="MBIScharge",
        ids={k: [*a.ids[k], *b.ids[k]] for k in a.ids},
    )

    combined = floors_from_components([floor_components(a), floor_components(b)])
    direct = held_out_floors(union)
    for key in ("floor/rmse", "floor/rmse_stereo_blind"):
        assert combined[key] == pytest.approx(direct[key], rel=1e-12), key


def test_floors_from_components_of_an_empty_set_is_zero():
    from experiments.collapse import floors_from_components

    out = floors_from_components([])
    assert out == {"floor/rmse": 0.0, "floor/rmse_stereo_blind": 0.0}


def test_alignment_never_pairs_atoms_of_different_elements():
    """The property the averaging rests on.

    _canonical_order may break ties differently for two rows whose atoms
    arrive in different orders, but a tie can only fall between atoms the
    refinement could not separate, and those agree on every local invariant.
    So a mis-ordering can swap a methyl's hydrogens -- for which there is no
    correct pairing anyway, the group being rotated -- and can never pair a
    carbon with a hydrogen. A refinement class is not always one orbit: in
    O=P(N1CC1)(N1CC1)N1CCN(P(=O)(N2CC2)N2CC2)CC1, the store's one such
    molecule, it merges aziridine and piperazine nitrogens -- still N with N.

    Verified exhaustively on the real corpus over all 16,125 collapse groups
    spanning more than one dash_id (94,624 rows): zero disagreements.
    """
    import numpy as np
    from experiments.collapse import _canonical_order, collapse_key
    from rdkit import Chem

    base = _charged("CC(=O)Nc1ccccc1O", {0: 0.1, 1: 0.2, 2: -0.3, 3: 0.4})
    rng = np.random.default_rng(0)
    key = collapse_key(base)
    ref = None
    for _ in range(12):
        perm = [int(i) for i in rng.permutation(base.GetNumAtoms())]
        other = Chem.RenumberAtoms(base, perm)
        assert collapse_key(other) == key
        order = _canonical_order(other, key)
        seq = [
            (
                other.GetAtomWithIdx(int(a)).GetSymbol(),
                other.GetAtomWithIdx(int(a)).GetDegree(),
                other.GetAtomWithIdx(int(a)).GetTotalNumHs(),
            )
            for a in order
        ]
        if ref is None:
            ref = seq
        assert seq == ref, "alignment paired atoms of differing invariants"


# --------------------------------------------------------------------------
# Symmetry orbits, and what is built on them
#
# Atoms related by a symmetry of the molecule are the same atom to every arm,
# so a collapsed target is their pooled mean and a floor counts their scatter,
# within one conformer too. The orbits come from automorphisms, not canonical
# ranks: CanonicalRankAtoms(breakTies=False) gives refinement classes, which
# with includeChirality=True split C2-related halves and without it merge the
# aziridine and piperazine nitrogens below.
# --------------------------------------------------------------------------

# Two methyl carbons, 0 and 2, related by the molecule's mirror plane.
_ISOPROPANOL = "CC(C)O"

# Two butenyl arms, one E and one Z: their methyl carbons, 7 and 11, are
# distinct atoms with stereo and the same atom without it.
_EZ_ARMS = "C/C=C/C(C/C=C/C)C/C=C\\C"

# Four aziridine nitrogens (2, 5, 14, 17) and two piperazine ones (8, 11)
# whose neighbourhoods a refinement cannot tell apart at any radius.
_AZIRIDINES = "O=P(N1CC1)(N1CC1)N1CCN(P(=O)(N2CC2)N2CC2)CC1"


def _same_orbit(labels, i, j):
    return bool(labels[i] == labels[j])


@pytest.mark.parametrize(
    ("smiles", "stereo", "same", "different"),
    [
        # C2: RDKit's chirality-aware ranking splits these halves.
        ("O[C@@H](C)[C@@H](C)O", True, [(0, 5), (1, 3), (2, 4)], []),
        # meso: the halves are related only through the mirror.
        ("O[C@@H](C)[C@H](C)O", True, [(0, 5), (1, 3), (2, 4)], []),
        (_EZ_ARMS, True, [], [(7, 11)]),
        (_EZ_ARMS, False, [(7, 11)], []),
        # the [nH] and the n: a Mol query ignores H counts unless told not to
        ("c1nc2ccccc2[nH]1", True, [], [(1, 8)]),
        # the methyls on N+ and on N: a neutral query atom matches any charge
        ("C[N+](C)(C)CCN(C)C", True, [(0, 2), (7, 8)], [(0, 7)]),
        (_AZIRIDINES, False, [(2, 5), (2, 14), (8, 11)], [(2, 8)]),
    ],
)
def test_symmetry_orbits_are_the_molecule_s_symmetry(smiles, stereo, same, different):
    from experiments.collapse import symmetry_orbits

    labels = symmetry_orbits(_mol(smiles), stereo=stereo)
    for i, j in same:
        assert _same_orbit(labels, i, j), (i, j)
    for i, j in different:
        assert not _same_orbit(labels, i, j), (i, j)


def test_hydrogens_share_their_host_s_orbit():
    from experiments.collapse import symmetry_orbits

    mol = _mol(_ISOPROPANOL)
    labels = symmetry_orbits(mol, stereo=True)
    methyl_hs = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetIdx() in (0, 2)
    ]
    assert len(methyl_hs) == 6
    assert len({int(labels[h]) for h in methyl_hs}) == 1


def test_a_collapsed_target_is_the_orbit_mean():
    import numpy as np
    from experiments.collapse import collapse_molecule_set

    a = _charged(_ISOPROPANOL, {0: 0.1, 2: -0.1, 1: 0.2})
    b = _charged(_ISOPROPANOL, {0: 0.3, 2: 0.1, 1: 0.4})
    out = collapse_molecule_set(_floor_set_of([a, b]))
    rep = out.mols[0]
    values = [rep.GetAtomWithIdx(i).GetDoubleProp("MBIScharge") for i in (0, 2, 1)]
    np.testing.assert_allclose(values, [0.1, 0.1, 0.3])


def test_collapse_keeps_the_molecule_level_value():
    """A set carrying a molecule-level property (net_charge, in the store)
    keeps one value per representative."""
    import numpy as np
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged(_ISOPROPANOL, {0: 0.1})
    b = _charged(_ISOPROPANOL, {0: 0.3})
    c = _charged("CCC", {1: 0.2})
    mset = MoleculeSet(
        mols=[a, b, c],
        atom_property="MBIScharge",
        molecule_property="net_charge",
        molecule_value=np.array([0.0, 0.0, 1.0]),
        ids={
            "collapse_key": [collapse_key(m) for m in (a, b, c)],
            "dash_id": ["d0", "d1", "d2"],
            "conf_id": ["c0"] * 3,
        },
    )
    out = collapse_molecule_set(mset)
    assert out.n_conformers == 2
    assert out.molecule_value is not None
    assert sorted(out.molecule_value.tolist()) == [0.0, 1.0]


def test_a_collapsed_target_does_not_depend_on_atom_order():
    """Renumbering a member must not move any target -- including in a
    molecule whose canonical ranks cannot tell its nitrogens apart, where a
    rank-based alignment could pair an aziridine N with a piperazine N."""
    import numpy as np
    from experiments.collapse import collapse_molecule_set

    def charged(mol):
        for atom in mol.GetAtoms():
            ring = atom.GetOwningMol().GetRingInfo()
            size = ring.MinAtomRingSize(atom.GetIdx()) if atom.IsInRing() else 0
            atom.SetDoubleProp("MBIScharge", atom.GetAtomicNum() * 0.01 + size * 0.1)
        return mol

    base = charged(_mol(_AZIRIDINES))
    expected = [
        base.GetAtomWithIdx(i).GetDoubleProp("MBIScharge")
        for i in range(base.GetNumAtoms())
    ]
    rng = np.random.default_rng(0)
    for _ in range(8):
        perm = [int(i) for i in rng.permutation(base.GetNumAtoms())]
        other = charged(Chem.RenumberAtoms(base, perm))
        rep = collapse_molecule_set(_floor_set_of([base, other])).mols[0]
        got = [
            rep.GetAtomWithIdx(i).GetDoubleProp("MBIScharge")
            for i in range(rep.GetNumAtoms())
        ]
        # every atom's value is fixed by its element and ring size, which
        # symmetry preserves, so a correct alignment changes nothing
        np.testing.assert_allclose(got, expected)


def test_the_floor_counts_symmetric_scatter_within_one_molecule():
    """A lone conformer still has irreducible error: its two methyl carbons
    differ, and every arm predicts them alike. A group of one used to count
    nothing."""
    import numpy as np
    from experiments.collapse import held_out_floors

    mol = _charged(_ISOPROPANOL, {0: 0.1, 2: -0.1})
    expected = np.sqrt(2 * 0.1**2 / mol.GetNumAtoms())
    floors = held_out_floors(_floor_set_of([mol]))
    assert floors["floor/rmse"] == pytest.approx(expected)
    assert floors["floor/rmse_stereo_blind"] == pytest.approx(expected)


def test_the_floor_does_not_depend_on_atom_order():
    """Renumbering a member must not move either floor. Paired by the
    tie-broken order, one member's methyl meets whichever methyl of the other
    the input order put first."""
    import numpy as np
    from experiments.collapse import held_out_floors

    first = _charged(_ISOPROPANOL, {0: 0.1, 2: -0.1, 1: 0.2})
    second = _charged(_ISOPROPANOL, {0: 0.3, 2: 0.0, 1: 0.25})
    reference = held_out_floors(_floor_set_of([first, second]))
    rng = np.random.default_rng(0)
    for _ in range(12):
        perm = [int(i) for i in rng.permutation(second.GetNumAtoms())]
        renumbered = Chem.RenumberAtoms(second, perm)
        got = held_out_floors(_floor_set_of([first, renumbered]))
        for name, value in reference.items():
            assert got[name] == pytest.approx(value, rel=1e-12), name


def test_only_the_stereo_blind_floor_merges_stereo_distinct_atoms():
    """The E and Z arms' methyls are different atoms to a model that reads
    stereo, and the same atom to one that does not."""
    import numpy as np
    from experiments.collapse import held_out_floors

    mol = _charged(_EZ_ARMS, {7: 0.2, 11: -0.2})
    floors = held_out_floors(_floor_set_of([mol]))
    assert floors["floor/rmse"] == pytest.approx(0.0)
    assert floors["floor/rmse_stereo_blind"] == pytest.approx(
        np.sqrt(2 * 0.2**2 / mol.GetNumAtoms())
    )


def _floor_set_of(mols):
    from experiments.collapse import collapse_key
    from experiments.data import MoleculeSet

    return MoleculeSet(
        mols=list(mols),
        atom_property="MBIScharge",
        ids={
            "collapse_key": [collapse_key(m) for m in mols],
            "dash_id": [f"d{i}" for i in range(len(mols))],
            "conf_id": ["c0"] * len(mols),
        },
    )


# --------------------------------------------------------------------------
# Within-structure statistics carried by the collapsed molecules
# --------------------------------------------------------------------------


def _within_sums(mset):
    import numpy as np

    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    sse = n = 0.0
    for m in mset.mols:
        for a in m.GetAtoms():
            sse += a.GetDoubleProp(mset.atom_property + WITHIN_SSE_SUFFIX)
            n += a.GetDoubleProp(mset.atom_property + WITHIN_N_SUFFIX)
    return np.float64(sse), np.float64(n)


def _multi_conformer_set():
    """Two conformers of ethanol that disagree, a lone propane whose symmetric
    methyl carbons disagree (orbit scatter in a group of one), and a lone
    methanol with no scatter at all."""
    return _floor_set_of(
        [
            _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
            _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
            _charged("CCC", {0: 1.0, 1: 0.5, 2: 2.0}),
            _charged("CO", {0: 0.3, 1: -0.3}),
        ]
    )


def _keyed_like(mset):
    """_floor_set_of numbers every row as its own dash_id; the two ethanols
    must share a collapse key for the test to mean anything."""
    keys = mset.ids["collapse_key"]
    assert keys[0] == keys[1]
    return mset


def test_collapse_attaches_sums_equal_to_the_floor_components():
    from experiments.collapse import collapse_molecule_set, floor_components

    mset = _keyed_like(_multi_conformer_set())
    sse, n = _within_sums(collapse_molecule_set(mset))
    parts = floor_components(mset)
    assert parts["sse"] > 0.0  # or the test is vacuous
    assert sse == pytest.approx(parts["sse"], rel=1e-12)
    assert n == pytest.approx(parts["n_atoms"], rel=1e-12)


def test_weight_by_collapse_leaves_the_sums_unchanged():
    from experiments.collapse import collapse_molecule_set

    mset = _keyed_like(_multi_conformer_set())
    plain = _within_sums(collapse_molecule_set(mset))
    weighted = _within_sums(collapse_molecule_set(mset, weight_by_collapse=True))
    assert weighted[0] == pytest.approx(plain[0], rel=1e-12)
    assert weighted[1] == pytest.approx(plain[1], rel=1e-12)


def test_a_collapsed_atom_carries_its_own_deviations():
    """Ethanol's C0 is 1 and 3 in the two conformers: mean 2, SSE 2, two members."""
    from experiments.collapse import collapse_molecule_set

    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    mset = _keyed_like(_multi_conformer_set())
    out = collapse_molecule_set(mset)
    ethanol = next(m for m in out.mols if m.GetNumAtoms() == 9)
    heavy = [ethanol.GetAtomWithIdx(i) for i in range(3)]
    sse = sorted(a.GetDoubleProp("MBIScharge" + WITHIN_SSE_SUFFIX) for a in heavy)
    assert sse == pytest.approx([2.0, 2.0, 2.0])
    assert all(a.GetDoubleProp("MBIScharge" + WITHIN_N_SUFFIX) == 2.0 for a in heavy)
