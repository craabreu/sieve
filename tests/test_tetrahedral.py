"""Tetrahedral handedness (spec 2026-09-24-tetrahedral-handedness-design.md, §7)."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import sieve
from sieve.config import KIND_AWARE, SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit
from sieve.level import blind_targets, class_kinds, mirror_targets
from sieve.refine import refine

BOTH = ("cis_trans", "tetrahedral")


def _mols(smiles, *, hs=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.AddHs(m) for m in mols] if hs else mols


def _config(mols, *, stereo=BOTH, depth=4, **kw):
    codes, edges = build_codes(mols, ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=depth,
        stereo=stereo,
        **kw,
    )


def _batch(mols, cfg, *, seed=0, node_order=None):
    b = from_rdkit(mols, y=None, config=cfg, node_order=node_order)
    y = np.random.default_rng(seed).normal(size=(b.n_nodes, 1))
    return dataclasses.replace(b, y=y)


def _stripped(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out


def _inverted(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.InvertChirality()
    return out


def test_track_order_is_normalised_and_duplicates_refused():
    mols = _mols(["CC"])
    a = _config(mols, stereo=("tetrahedral", "cis_trans"))
    b = _config(mols, stereo=("cis_trans", "tetrahedral"))
    assert a.stereo == BOTH and a.schema_version == b.schema_version
    with pytest.raises(ValueError, match="twice"):
        _config(mols, stereo=("cis_trans", "cis_trans"))


def test_three_digests_and_cis_trans_unchanged():
    mols = _mols(["CC"])
    digests = {
        _config(mols, stereo=s).schema_version for s in ((), ("cis_trans",), BOTH)
    }
    assert len(digests) == 3
    assert _config(mols, stereo=BOTH).stereo_radices == (4, 4)


from sieve.batch import NodeBatch, concat_batches


def _star_batch(centres):
    """Atom 0 bonded to atoms 1..4; two copies side by side."""
    src = [0, 1, 0, 2, 0, 3, 0, 4]
    dst = [1, 0, 2, 0, 3, 0, 4, 0]
    src += [s + 5 for s in src]
    dst += [d + 5 for d in dst]
    return NodeBatch(
        node_attrs=np.zeros((10, 1), np.int64),
        edge_src=np.array(src),
        edge_dst=np.array(dst),
        edge_attrs=np.zeros((16, 1), np.int64),
        graph_id=np.array([0] * 5 + [1] * 5),
        stereo_centres=np.array(centres, np.int64).reshape(-1, 6),
    )


@pytest.mark.parametrize(
    "row, message",
    [
        ([0, 1, 2, 3, 6, 1], "adjacent"),
        ([0, 1, -1, 3, 4, 1], "only in n3"),
        ([0, 1, 2, 3, -1, 1], "degree"),
        ([0, 1, 2, 3, 4, 0], "parity"),
        ([0, 1, 2, 3, 99, 1], "range"),
    ],
)
def test_stereo_centres_are_validated(row, message):
    with pytest.raises(ValueError, match=message):
        _star_batch([row])


def test_a_centre_listed_twice_is_refused():
    with pytest.raises(ValueError, match="once"):
        _star_batch([[0, 1, 2, 3, 4, 1], [0, 4, 3, 2, 1, 1]])


def test_an_edgeless_batch_with_no_centres_is_accepted():
    """A single heavy atom has no edges and no centres; the adjacency check
    must not fire on an empty lookup (found in review of PR #43)."""
    mols = [Chem.MolFromSmiles("[Na+]"), Chem.MolFromSmiles("[Cl-]")]
    b = from_rdkit(mols, config=_config(mols))
    assert b.stereo_centres is not None and b.stereo_centres.shape == (0, 6)


def test_stereo_centres_follow_slicing_and_concat():
    b = _star_batch([[0, 1, 2, 3, 4, 1], [5, 6, 7, 8, 9, -1]])
    second = b[b.graph_id == 1]
    assert second.stereo_centres is not None
    np.testing.assert_array_equal(second.stereo_centres, [[0, 1, 2, 3, 4, -1]])
    both = concat_batches([second, second])
    assert both.stereo_centres is not None
    np.testing.assert_array_equal(
        both.stereo_centres, [[0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, -1]]
    )
    with pytest.raises(ValueError, match="stereo_centres"):
        concat_batches([second, dataclasses.replace(second, stereo_centres=None)])


from sieve.io.rdkit_adapter import _stereo_centre_rows


def test_enantiomers_give_opposite_parity_on_identical_rows():
    (r,) = _stereo_centre_rows(_mols(["F[C@H](Cl)Br"])[0])
    (s,) = _stereo_centre_rows(_mols(["F[C@@H](Cl)Br"])[0])
    assert r[:5] == s[:5] and r[5] == -s[5]


def test_a_sulfoxide_has_a_virtual_fourth_neighbour():
    (row,) = _stereo_centre_rows(_mols(["C[S@](=O)CC"])[0])
    assert row[4] == -1


def test_untagged_and_two_neighbour_centres_give_no_row():
    assert _stereo_centre_rows(_mols(["CC(C)CC"])[0]) == []
    assert _stereo_centre_rows(Chem.MolFromSmiles("C[P@H]CC")) == []


def test_the_batch_carries_centres_only_under_the_track():
    mols = _mols(["F[C@H](Cl)Br", "C/C=C/C"])
    both = from_rdkit(mols, config=_config(mols))
    ct = from_rdkit(mols, config=_config(mols, stereo=("cis_trans",)))
    tet = from_rdkit(mols, config=_config(mols, stereo=("tetrahedral",)))
    assert both.stereo_centres is not None and both.stereo_bonds is not None
    assert both.stereo_centres.shape == (1, 6) and both.stereo_bonds.shape == (1, 7)
    assert ct.stereo_centres is None
    assert tet.stereo_bonds is None and tet.stereo_centres is not None
    assert tet.stereo_centres.shape == (1, 6)


def test_parallel_featurisation_carries_centres():
    mols = _mols(["F[C@H](Cl)Br", "N[C@@H](C)C(=O)O"] * 4)
    cfg = _config(mols)
    seq = from_rdkit(mols, config=cfg)
    par = from_rdkit(mols, config=cfg, n_jobs=2)
    assert seq.stereo_centres is not None and par.stereo_centres is not None
    np.testing.assert_array_equal(seq.stereo_centres, par.stereo_centres)


from sieve.stereo import (
    CODE_MINUS,
    CODE_NONE,
    CODE_PLUS,
    FingerprintWindow,
    mirror_codes,
    tetrahedral_codes,
)


def _row(n3, parity):
    return np.array([[0, 1, 2, 3, n3, parity]], np.int64)


@pytest.mark.parametrize(
    "fp, n3, parity, expected",
    [
        ([0, 10, 20, 30, 40], 4, 1, CODE_PLUS),  # already sorted: even
        ([0, 20, 10, 30, 40], 4, 1, CODE_MINUS),  # one inversion
        ([0, 20, 10, 30, 40], 4, -1, CODE_PLUS),  # CW flips it back
        ([0, 40, 30, 20, 10], 4, 1, CODE_PLUS),  # six inversions: even
        ([0, 10, 20, 30, 0], -1, 1, CODE_PLUS),  # virtual n3 adds none
        ([0, 20, 10, 30, 0], -1, 1, CODE_MINUS),
        ([0, 10, 10, 30, 40], 4, 1, CODE_NONE),  # a tie defers
        ([0, 10, 20, 30, 10], -1, 1, CODE_PLUS),  # a virtual n3 never ties
    ],
)
def test_the_code_is_parity_times_the_sorting_sign(fp, n3, parity, expected):
    got = tetrahedral_codes(_row(n3, parity), np.array(fp, np.uint64))
    assert got.tolist() == [expected]


def test_mirror_codes_swap_plus_and_minus_only():
    codes = np.array([CODE_NONE, CODE_PLUS, CODE_MINUS])
    assert mirror_codes(codes).tolist() == [CODE_NONE, CODE_MINUS, CODE_PLUS]


def test_the_window_serves_two_consecutive_radii():
    w = FingerprintWindow(iter([np.array([r]) for r in range(5)]))
    assert w.at(1).tolist() == [1] and w.at(0).tolist() == [0]
    assert w.at(3).tolist() == [3] and w.at(2).tolist() == [2]
    with pytest.raises(ValueError, match="radius"):
        w.at(0)


def _centre_labels(levels, atom):
    return [int(lv.labels[atom]) for lv in levels]


def test_the_radius_rule_under_element_only_attributes():
    """fp_0 hashes the whole attribute row, so the rule is only visible with
    element-only attributes: there CHFClBr's four neighbours differ at
    radius 0 and alanine's two carbons do not."""
    for smiles, first in (("F[C@H](Cl)Br", 1), ("N[C@@H](C)C(=O)O", 2)):
        mols = _mols([smiles])
        mirror = _inverted(mols)
        cfg = _config(mols + mirror, depth=3)
        lv = refine(from_rdkit(mols + mirror, config=cfg), cfg)
        (row,) = _stereo_centre_rows(mols[0])
        v, n = row[0], mols[0].GetNumAtoms()
        for k in range(1, 4):
            same = lv[k].labels[v] == lv[k].labels[v + n]
            assert same == (k < first), (smiles, k)


def test_ties_defer_at_every_radius():
    m = Chem.MolFromSmiles("CC(C)Cl")
    m.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    mols = [Chem.AddHs(m)]
    cfg = _config(mols, depth=4)
    for lv in refine(from_rdkit(mols, config=cfg), cfg):
        np.testing.assert_array_equal(lv.labels, lv.blind)


def test_five_orderings_of_alanine_agree():
    """Random SMILES of one molecule, so every copy is the same enantiomer."""
    base = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    smiles = list(Chem.MolToRandomSmilesVect(base, 5, randomSeed=0))
    for hs in (True, False):
        mols = _mols(smiles, hs=hs)
        cfg = _config(mols, depth=3)
        lv = refine(from_rdkit(mols, config=cfg), cfg)
        offs = np.cumsum([0] + [m.GetNumAtoms() for m in mols])
        centres = [
            o + _stereo_centre_rows(m)[0][0] for o, m in zip(offs, mols, strict=False)
        ]
        for level in lv:
            assert len({int(level.labels[c]) for c in centres}) == 1


def test_renumbering_leaves_the_classes_unchanged():
    mols = _mols(["N[C@@H](C)C(=O)O", "F[C@H](Cl)Br"])
    rng = np.random.default_rng(0)
    orders = [rng.permutation(m.GetNumAtoms()) for m in mols]
    cfg = _config(mols + mols, depth=3)
    natural = [np.arange(m.GetNumAtoms()) for m in mols]
    b = from_rdkit(mols + mols, config=cfg, node_order=natural + orders)
    sizes = [m.GetNumAtoms() for m in mols]
    n = sum(sizes)
    # Raw atom a of molecule j sits at first[j] + a in the natural copy and at
    # n + first[j] + position-of-a-in-order_j in the permuted copy.
    first = np.cumsum([0, *sizes[:-1]])
    natural_pos = np.concatenate(
        [f + np.arange(k) for f, k in zip(first, sizes, strict=True)]
    )
    permuted_pos = np.concatenate(
        [n + f + np.argsort(o) for f, o in zip(first, orders, strict=True)]
    )
    for lv in refine(b, cfg):
        np.testing.assert_array_equal(lv.labels[natural_pos], lv.labels[permuted_pos])


def test_mirror_rows_and_maps_are_consistent():
    mols = _mols(["N[C@@H](C)C(=O)O", r"C/C=C\[C@H](F)Cl", "CCCC"])
    cfg = _config(mols)
    for lv in refine(from_rdkit(mols, config=cfg), cfg):
        mt, bt = mirror_targets(lv), blind_targets(lv)
        np.testing.assert_array_equal(mt[mt], np.arange(lv.n_classes))
        np.testing.assert_array_equal(mt[lv.labels], lv.mirror)
        np.testing.assert_array_equal(bt[lv.mirror], lv.blind)
        assert np.all(class_kinds(lv)[lv.mirror] & KIND_AWARE)


CORPUS = [
    "N[C@@H](C)C(=O)O",
    "C[C@H](Br)[C@H](C)Br",
    "C[C@H](Br)[C@@H](C)Br",
    "F[C@H](Cl)Br",
    "C[S@](=O)CC",
    r"C/C=C\[C@H](F)Cl",
    "C/C=C/C",
    "CCCC",
]
EB = {"class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}


def test_a_class_and_its_mirror_hold_identical_statistics():
    mols = _mols(CORPUS)
    cfg = _config(mols)
    model = sieve.fit(_batch(mols, cfg), cfg)
    seen = 0
    for lv in model.levels:
        mt = mirror_targets(lv)
        chiral = np.flatnonzero(mt != np.arange(lv.n_classes))
        seen += chiral.size
        np.testing.assert_array_equal(lv.count[chiral], lv.count[mt[chiral]])
        # count (integer) is bit-exact; mean/msd sum the identical multiset
        # of atoms via two different scatter-reduce orderings (the
        # "differs" pass for one class, the "moved" pass for its mirror),
        # so IEEE 754 non-associativity can move the last bit.
        np.testing.assert_allclose(lv.mean[chiral], lv.mean[mt[chiral]], rtol=1e-12)
        np.testing.assert_allclose(lv.msd[chiral], lv.msd[mt[chiral]], rtol=1e-12)
    assert seen


def test_self_mirror_classes_count_each_atom_once():
    """meso-2,3-dibromobutane: rows naming a and M(a) are unchanged by
    reflection, and must not be counted twice."""
    mols = _mols(["C[C@H](Br)[C@@H](C)Br"])
    cfg = _config(mols)
    batch = _batch(mols, cfg)
    model, labels = sieve.fit(batch, cfg), refine(batch, cfg)
    for lv, fl in zip(labels, model.levels, strict=True):
        mt = mirror_targets(fl)
        for c in np.flatnonzero(
            (class_kinds(fl) == KIND_AWARE) & (mt == np.arange(fl.n_classes))
        ):
            assert fl.count[c] == int((lv.labels == c).sum())


def test_blind_classes_stay_the_stereo_blind_fit():
    mols = _mols(CORPUS)
    s_cfg, b_cfg = _config(mols, **EB), _config(mols, stereo=(), **EB)
    s_batch, b_batch = _batch(mols, s_cfg), _batch(mols, b_cfg)
    s, b = sieve.fit(s_batch, s_cfg), sieve.fit(b_batch, b_cfg)
    for ls, lb, fs, fb in zip(
        refine(s_batch, s_cfg), refine(b_batch, b_cfg), s.levels, b.levels, strict=True
    ):
        pairs = np.unique(np.stack([ls.blind, lb.labels], 1), axis=0)
        ids = pairs[:, 0]
        np.testing.assert_array_equal(fs.count[ids], fb.count[pairs[:, 1]])
        np.testing.assert_array_equal(fs.mean[ids], fb.mean[pairs[:, 1]])


def test_aware_variance_counts_each_mirror_orbit_once():
    from sieve.continuation import _aware_members

    mols = _mols(["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O", "N[C@@H](CC)C(=O)O"])
    cfg = _config(mols, **EB)
    model = sieve.fit(_batch(mols, cfg), cfg)
    chiral_seen = 0
    for lv in model.levels:
        aware = np.flatnonzero(class_kinds(lv) & KIND_AWARE)
        mt = mirror_targets(lv)
        members = _aware_members(lv)
        # Exactly one member per orbit: every aware class or its mirror is
        # present, never both unless it is its own mirror.
        orbits = {min(int(c), int(mt[c])) for c in aware}
        assert sorted(orbits) == sorted(int(c) for c in members)
        chiral_seen += int((mt[aware] != aware).sum())
    assert chiral_seen


def _predict(model, mols, cfg):
    return sieve.predict_detailed(model, from_rdkit(mols, config=cfg))


@pytest.mark.parametrize("rule", [{}, EB], ids=["none", "eb"])
def test_embedding_without_chiral_tags_is_the_cis_trans_model(rule):
    mols = _stripped(_mols(CORPUS))
    both, ct = _config(mols, **rule), _config(mols, stereo=("cis_trans",), **rule)
    pb = _predict(sieve.fit(_batch(mols, both), both), mols, both)
    pc = _predict(sieve.fit(_batch(mols, ct), ct), mols, ct)
    np.testing.assert_allclose(pb.value, pc.value, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(pb.stereo_refined, pc.stereo_refined)


@pytest.mark.parametrize("rule", [{}, EB], ids=["none", "eb"])
def test_predictions_are_mirror_invariant(rule):
    mols = _mols(CORPUS)
    cfg = _config(mols, **rule)
    model = sieve.fit(_batch(mols, cfg), cfg)
    np.testing.assert_allclose(
        _predict(model, _inverted(mols), cfg).value,
        _predict(model, mols, cfg).value,
        rtol=1e-12,
        atol=1e-15,
    )


def test_an_unseen_enantiomer_is_answered_like_the_trained_one():
    r, s = _mols(["N[C@@H](C)C(=O)O"]), _mols(["N[C@H](C)C(=O)O"])
    cfg = _config(r + s)
    model = sieve.fit(_batch(r, cfg), cfg)
    pr, ps = _predict(model, r, cfg), _predict(model, s, cfg)
    np.testing.assert_allclose(pr.value, ps.value, rtol=1e-12, atol=1e-15)
    assert ps.stereo_refined.any()


def test_diastereomers_separate():
    a, meso = _mols(["C[C@H](Br)[C@H](C)Br"]), _mols(["C[C@H](Br)[C@@H](C)Br"])
    cfg = _config(a + meso, depth=4)
    lv = refine(from_rdkit(a + meso, config=cfg), cfg)[-1]
    c = _stereo_centre_rows(a[0])[0][0]
    assert lv.labels[c] != lv.labels[c + a[0].GetNumAtoms()]


def test_a_fit_on_the_mirrored_corpus_predicts_the_same():
    mols = _mols(CORPUS)
    cfg = _config(mols, **EB)
    orig = sieve.fit(_batch(mols, cfg), cfg)
    mirr = sieve.fit(_batch(_inverted(mols), cfg), cfg)
    np.testing.assert_allclose(
        _predict(orig, mols, cfg).value,
        _predict(mirr, mols, cfg).value,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("cuts", [(3,), (2, 5)], ids=["two", "three"])
def test_merge_monoid_with_enantiomers_split(cuts):
    import itertools

    # Alanine's two hands land in different shards.
    mols = _mols([*CORPUS, "N[C@H](C)C(=O)O"])
    cfg = _config(mols, **EB)
    batch = _batch(mols, cfg)
    edges = [0, *cuts, len(mols)]
    merged = None
    for lo, hi in itertools.pairwise(edges):
        part = sieve.fit(batch[(batch.graph_id >= lo) & (batch.graph_id < hi)], cfg)
        merged = part if merged is None else merged.merge(part)
    assert merged is not None  # edges always yields >= 1 pair
    whole = sieve.fit(batch, cfg)
    assert [lv.n_classes for lv in merged.levels] == [
        lv.n_classes for lv in whole.levels
    ]
    np.testing.assert_allclose(
        sieve.predict(merged, batch),
        sieve.predict(whole, batch),
        rtol=1e-12,
        atol=1e-15,
    )
    for lm in merged.levels:
        mt = mirror_targets(lm)
        np.testing.assert_array_equal(mt[mt], np.arange(lm.n_classes))


def test_save_and_load_keep_mirror_of(tmp_path):
    mols = _mols(CORPUS)
    cfg = _config(mols)
    model = sieve.fit(_batch(mols, cfg), cfg)
    model.save(tmp_path / "m.npz")
    loaded = sieve.SieveModel.load(tmp_path / "m.npz")
    for a, b in zip(model.levels, loaded.levels, strict=True):
        np.testing.assert_array_equal(mirror_targets(a), mirror_targets(b))


def test_predictions_do_not_depend_on_a_pentavalent_neighbour():
    mols = _mols(CORPUS)
    p = _mols(["COP12(OC)NC(=O)O[C@]1(C(F)(F)F)c1ccccc1O2"])
    cfg = _config(mols + p, **EB)
    model = sieve.fit(_batch(mols + p, cfg), cfg)
    alone = sieve.predict(model, from_rdkit(mols, config=cfg))
    beside = sieve.predict(model, from_rdkit(mols + p, config=cfg))
    np.testing.assert_array_equal(alone, beside[: alone.shape[0]])
