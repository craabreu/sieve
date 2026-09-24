"""Stereo refines the stereo-blind class (spec 2026-09-23, section 8)."""

import dataclasses
import itertools

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import sieve
from sieve.config import KIND_AWARE, KIND_BLIND, KIND_BOTH, SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit
from sieve.level import blind_targets, class_kinds
from sieve.refine import refine

CORPUS = [
    "C/C=C/C",
    r"C/C=C\C",
    "C/C=C/CC",
    r"C/C=C\CC",
    "C/C(F)=C(Cl)/C",
    r"C/C(F)=C(Cl)\C",
    "CCCC",
    "CC(C)C",
    "C/C=C/Br",
    r"OC/C=C\CN",
    "CCN1/C(=C2/OC(=S)N(C)C2=O)Sc2ccccc21",
]
PHOSPHORUS = "COP12(OC)NC(=O)O[C@]1(C(F)(F)F)c1ccccc1O2"


def _config(smiles, *, stereo, depth=4, **kw):
    codes, edges = build_codes([Chem.MolFromSmiles(s) for s in smiles], ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=depth,
        stereo=("cis_trans",) if stereo else (),
        **kw,
    )


def _batch(smiles, cfg, seed=0):
    b = from_rdkit([Chem.MolFromSmiles(s) for s in smiles], y=None, config=cfg)
    y = np.random.default_rng(seed).normal(size=(b.n_nodes, 1))
    return dataclasses.replace(b, y=y)


def _class_map(src, dst):
    """src id -> dst id for two labelings of the same atoms; asserts a bijection."""
    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    assert len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1]))
    out = np.full(int(src.max()) + 1, -1, np.int64)
    out[pairs[:, 0]] = pairs[:, 1]
    return out


# --- refinement (section 3, 4) -----------------------------------------------


def test_blind_labels_partition_atoms_as_a_stereo_blind_refine_does():
    """Recursive blinding (section 8.4): the blind label is the incumbent's class."""
    s_cfg, b_cfg = _config(CORPUS, stereo=True), _config(CORPUS, stereo=False)
    s = refine(_batch(CORPUS, s_cfg), s_cfg)
    b = refine(_batch(CORPUS, b_cfg), b_cfg)
    for ls, lb in zip(s, b, strict=True):
        _class_map(ls.blind, lb.labels)


def test_methyls_of_the_two_butenes_share_a_blind_class_at_radius_three():
    """Fails if only the current round's trit is dropped: the methyl's
    neighbour already differs by geometry at radius 2."""
    cfg = _config(CORPUS, stereo=True, depth=3)
    lv = refine(_batch(["C/C=C/C", r"C/C=C\C"], cfg), cfg)[-1]
    assert lv.blind[0] == lv.blind[4]
    assert lv.labels[0] != lv.labels[4]


def test_kinds_and_blind_counterparts_are_consistent():
    cfg = _config(CORPUS, stereo=True)
    for lv in refine(_batch(CORPUS, cfg), cfg):
        kind, target = class_kinds(lv), blind_targets(lv)
        assert np.all(kind[lv.blind] & KIND_BLIND)
        assert np.all(kind[lv.labels] & KIND_AWARE)
        np.testing.assert_array_equal(target[lv.labels], lv.blind)
        np.testing.assert_array_equal(target[lv.blind], lv.blind)
        differs = lv.labels != lv.blind
        assert np.all(kind[lv.labels[differs]] == KIND_AWARE)


def test_no_stereo_bond_means_no_aware_only_class():
    cfg = _config(CORPUS, stereo=True)
    for lv in refine(_batch(["CCCC", "CC(C)C"], cfg), cfg):
        assert np.all(class_kinds(lv) == KIND_BOTH)
        np.testing.assert_array_equal(lv.labels, lv.blind)


# --- statistics (section 4) --------------------------------------------------


def _pair(**kw):
    s_cfg = _config(CORPUS, stereo=True, **kw)
    b_cfg = _config(CORPUS, stereo=False, **kw)
    s_batch, b_batch = _batch(CORPUS, s_cfg), _batch(CORPUS, b_cfg)
    maps = [
        _class_map(ls.blind, lb.labels)
        for ls, lb in zip(refine(s_batch, s_cfg), refine(b_batch, b_cfg), strict=True)
    ]
    return sieve.fit(s_batch, s_cfg), sieve.fit(b_batch, b_cfg), maps, s_batch, b_batch


def test_blind_classes_carry_exactly_the_stereo_blind_statistics():
    s, b, maps, *_ = _pair()
    for k, (ls, lb, m) in enumerate(zip(s.levels, b.levels, maps, strict=True)):
        ids = np.flatnonzero(class_kinds(ls) & KIND_BLIND)
        assert ids.size == lb.n_classes
        np.testing.assert_array_equal(ls.count[ids], lb.count[m[ids]])
        np.testing.assert_array_equal(ls.mean[ids], lb.mean[m[ids]])
        np.testing.assert_array_equal(ls.msd[ids], lb.msd[m[ids]])
        if k:
            np.testing.assert_array_equal(
                maps[k - 1][ls.parent[ids]], lb.parent[m[ids]]
            )


def test_an_aware_only_class_holds_its_own_atoms():
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    assert batch.y is not None
    model, labels = sieve.fit(batch, cfg), refine(batch, cfg)
    seen = 0
    for lv, fl in zip(labels, model.levels, strict=True):
        for c in np.flatnonzero(class_kinds(fl) == KIND_AWARE):
            members = lv.labels == c
            assert fl.count[c] == members.sum()
            np.testing.assert_allclose(fl.mean[c], batch.y[members].mean(axis=0))
            seen += 1
    assert seen


# --- estimation (section 5) --------------------------------------------------

EB = {"class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}


def test_continuation_and_tau_squared_match_the_stereo_blind_fit():
    from sieve.continuation import (
        atom_variance,
        child_counts,
        class_means,
        class_sibling_variance,
        sibling_variance,
    )

    s, b, maps, *_ = _pair(**EB)
    np.testing.assert_allclose(sibling_variance(s), sibling_variance(b), rtol=1e-12)
    np.testing.assert_allclose(atom_variance(s), atom_variance(b), rtol=1e-12)
    for k, m in enumerate(maps):
        ids = np.flatnonzero(class_kinds(s.levels[k]) & KIND_BLIND)
        for f in (class_means, child_counts, class_sibling_variance):
            np.testing.assert_allclose(f(s)[k][ids], f(b)[k][m[ids]], rtol=1e-12)


def test_an_aware_only_class_has_no_children_and_keeps_its_pooled_mean():
    from sieve.continuation import child_counts, class_means

    cfg = _config(CORPUS, stereo=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    for k, lv in enumerate(model.levels):
        only = class_kinds(lv) == KIND_AWARE
        assert np.all(child_counts(model)[k][only] == 0)
        np.testing.assert_array_equal(class_means(model)[k][only], lv.mean[only])


def test_aware_variance_is_the_debiased_spread_about_the_blind_counterpart():
    from sieve.continuation import aware_variance

    cfg = _config(CORPUS, stereo=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    tau = aware_variance(model)
    assert len(tau) == model.config.n_levels
    assert all(t != t or t >= 0.0 for t in tau)
    assert tau[0] != tau[0]  # attribute level: no aware-only class
    assert any(t == t for t in tau)  # the corpus has refined blind classes


COUNT = {"shrinkage_weight": "count", "shrinkage_strength": 2.0}


@pytest.mark.parametrize("rule", [COUNT, EB], ids=["count", "eb"])
def test_blind_shrunk_means_match_and_aware_only_shrink_toward_blind(rule):
    from sieve.shrinkage import empirical_bayes_weights, shrunk_means

    s, b, maps, *_ = _pair(**rule)
    ss, sb = shrunk_means(s), shrunk_means(b)
    seen = 0
    for k, m in enumerate(maps):
        lv = s.levels[k]
        ids = np.flatnonzero(class_kinds(lv) & KIND_BLIND)
        np.testing.assert_allclose(ss[k][ids], sb[k][m[ids]], rtol=1e-12)
        only = np.flatnonzero(class_kinds(lv) == KIND_AWARE)
        if only.size:
            n = lv.count[only].astype(float)[:, None]
            if rule is EB:
                w = empirical_bayes_weights(s)[k][only][:, None]
            else:
                w = n / (n + 2.0)
            target = ss[k][blind_targets(lv)[only]]
            np.testing.assert_allclose(
                ss[k][only], w * lv.mean[only] + (1 - w) * target, rtol=1e-12
            )
            seen += only.size
    assert seen


def test_predictive_variance_is_total_on_aware_only_classes():
    from sieve.uncertainty import predictive_variance

    cfg = _config(CORPUS, stereo=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    for table in predictive_variance(model):
        assert np.all(np.isfinite(table)) and np.all(table > 0)


# --- prediction (section 6) --------------------------------------------------

TRANS = ["C/C=C/C", "C/C=C/CC", "CC/C=C/CC", "C/C=C/CO"]
CIS = [r"C/C=C\C", r"C/C=C\CC", r"CC/C=C\CC", r"C/C=C\CO"]


def _mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


@pytest.mark.parametrize("rule", [{}, COUNT, EB], ids=["none", "count", "eb"])
def test_unrefined_atoms_get_exactly_the_stereo_blind_prediction(rule):
    s, b, _, s_batch, b_batch = _pair(**rule)
    ps = sieve.predict_detailed(s, s_batch)
    pb = sieve.predict_detailed(b, b_batch)
    assert ps.stereo_refined is not None and ps.stereo_refined.any()
    assert pb.stereo_refined is None
    keep = ~ps.stereo_refined
    np.testing.assert_allclose(ps.value[keep], pb.value[keep], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(ps.matched_level, pb.matched_level)


def test_an_isomer_absent_from_training_falls_back_to_the_blind_answer():
    smiles = TRANS + CIS
    s_cfg = _config(smiles, stereo=True, **EB)
    b_cfg = _config(smiles, stereo=False, **EB)
    s = sieve.fit(_batch(TRANS, s_cfg), s_cfg)
    b = sieve.fit(_batch(TRANS, b_cfg), b_cfg)
    ps = sieve.predict_detailed(s, from_rdkit(_mols(CIS), config=s_cfg))
    pb = sieve.predict(b, from_rdkit(_mols(CIS), config=b_cfg))
    assert ps.stereo_refined is not None and not ps.stereo_refined.any()
    np.testing.assert_allclose(ps.value, pb, rtol=1e-12, atol=1e-15)


def test_both_butenes_are_answered_by_their_own_aware_classes():
    """Section 8.3: both isomers separate, and each is answered by its own
    aware class, where a stereo-blind model would answer both with the mean."""
    cfg = _config(["C/C=C/C"], stereo=True, depth=3)
    batch = from_rdkit(_mols(["C/C=C/C", r"C/C=C\C"]), config=cfg)
    y = np.array([[1.0], [2.0], [2.0], [1.0], [3.0], [4.0], [4.0], [3.0]])
    model = sieve.fit(dataclasses.replace(batch, y=y), cfg)
    p = sieve.predict_detailed(model, batch)
    assert p.stereo_refined is not None and p.stereo_refined.all()
    np.testing.assert_array_equal(p.value, y)


def test_predictions_do_not_depend_on_a_pentavalent_neighbour():
    cfg = _config([*CORPUS, PHOSPHORUS], stereo=True, **EB)
    model = sieve.fit(_batch([*CORPUS, PHOSPHORUS], cfg), cfg)
    alone = sieve.predict(model, from_rdkit(_mols(CORPUS), config=cfg))
    beside = sieve.predict(model, from_rdkit(_mols([*CORPUS, PHOSPHORUS]), config=cfg))
    np.testing.assert_array_equal(alone, beside[: alone.shape[0]])


def test_predict_loo_refuses_a_stereo_model():
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    with pytest.raises(NotImplementedError, match="stereo"):
        sieve.predict_loo(sieve.fit(batch, cfg), batch)


# --- merge (section 7) -------------------------------------------------------


def _canonical(levels):
    """Each level's rows, kinds and blind counterparts, in an order and an id
    space fixed by content alone, so that two models fitted in different
    orders compare equal. Ids are rewritten level by level into sorted-row
    ranks, since a signature names the previous level's ids."""
    out, prev = [], None
    for lv in levels:
        sig = lv.signatures.copy()
        if prev is not None:
            sig[:, 0] = prev[sig[:, 0]]
            if sig.shape[1] > 1:
                filled = sig[:, 1:] >= 0
                lab, code = np.divmod(np.where(filled, sig[:, 1:], 0), N_EDGE_TYPES)
                sig[:, 1:] = np.sort(
                    np.where(filled, prev[lab] * N_EDGE_TYPES + code, -1), axis=1
                )
        order = np.lexsort(sig.T[::-1])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        out.append((sig[order], class_kinds(lv)[order], rank[blind_targets(lv)][order]))
        prev = rank
    return out


N_EDGE_TYPES = _config(CORPUS, stereo=True).n_edge_types


@pytest.mark.parametrize("cuts", [(6,), (3, 7)], ids=["two", "three"])
def test_merge_reproduces_kinds_and_blind_counterparts(cuts):
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    edges = [0, *cuts, len(CORPUS)]
    shards = [
        batch[(batch.graph_id >= lo) & (batch.graph_id < hi)]
        for lo, hi in itertools.pairwise(edges)
    ]
    merged = sieve.fit(shards[0], cfg)
    for shard in shards[1:]:
        merged = merged.merge(sieve.fit(shard, cfg))
    whole = sieve.fit(batch, cfg)
    for cm, cw in zip(_canonical(merged.levels), _canonical(whole.levels), strict=True):
        for x, y in zip(cm, cw, strict=True):
            np.testing.assert_array_equal(x, y)


def test_a_shard_holding_a_pentavalent_atom_mints_no_class_for_a_shared_molecule():
    """Section 8.5, shard form: a molecule gets the same classes whether or not
    its shard also holds a higher-degree atom."""
    cfg = _config([*CORPUS, PHOSPHORUS], stereo=True)
    with_p = sieve.fit(_batch([*CORPUS, PHOSPHORUS], cfg), cfg)
    without = sieve.fit(_batch(CORPUS, cfg), cfg)
    merged = with_p.merge(without)
    assert [lv.n_classes for lv in merged.levels] == [
        lv.n_classes for lv in with_p.levels
    ]


def test_merging_into_the_empty_model_keeps_the_kinds():
    cfg = _config(CORPUS, stereo=True)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    merged = sieve.SieveModel.empty(cfg).merge(model)
    for lm, lw in zip(merged.levels, model.levels, strict=True):
        np.testing.assert_array_equal(class_kinds(lm), class_kinds(lw))
        np.testing.assert_array_equal(blind_targets(lm), blind_targets(lw))


# --- schema and persistence (section 7) --------------------------------------


def test_the_schema_marks_the_construction_only_when_stereo_is_on():
    import hashlib
    import json

    on, off = _config(CORPUS, stereo=True), _config(CORPUS, stereo=False)
    assert on._schema_payload()["stereo_construction"] == "refines_blind"
    assert "stereo_construction" not in off._schema_payload()
    fused = {
        k: v for k, v in on._schema_payload().items() if k != "stereo_construction"
    }
    blob = json.dumps(fused, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() != on.schema_version


def test_save_and_load_keep_kinds_and_blind_counterparts(tmp_path):
    cfg = _config(CORPUS, stereo=True)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    model.save(tmp_path / "m.npz")
    loaded = sieve.SieveModel.load(tmp_path / "m.npz")
    assert any(lv.kind is not None for lv in loaded.levels)
    for a, b in zip(model.levels, loaded.levels, strict=True):
        assert (a.kind is None) == (b.kind is None)
        np.testing.assert_array_equal(class_kinds(a), class_kinds(b))
        np.testing.assert_array_equal(blind_targets(a), blind_targets(b))
