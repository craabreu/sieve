"""Within-structure variance (within-structure-variance spec, phase 1).

Collapse removes each structure's conformer-to-conformer scatter from the fit;
these two additive statistics carry it back, so the predictive variance can
add it to every class.
"""

from __future__ import annotations

import numpy as np
import pytest

import sieve
from sieve.model import SieveModel
from tests.helpers import chain_batch, simple_config


def _fitted(**kw):
    return sieve.fit(chain_batch(10, graphs=3), simple_config(**kw))


# ----------------------------------------------------------------- model --


def test_a_model_without_statistics_has_zero_within_variance():
    m = _fitted()
    assert m.within_sse is None
    assert m.within_n == 0.0
    np.testing.assert_array_equal(m.within_variance, np.zeros(1))


def test_with_within_structure_sets_the_sums():
    m = _fitted().with_within_structure(3.0e-3, 30)
    np.testing.assert_array_equal(m.within_sse, [3.0e-3])
    assert m.within_n == 30.0
    np.testing.assert_allclose(m.within_variance, [1.0e-4])


@pytest.mark.parametrize(
    ("sse", "n"), [(-1.0, 3), (np.nan, 3), (1.0, -2), (1.0, np.inf), (1.0, 0)]
)
def test_with_within_structure_refuses_impossible_sums(sse, n):
    with pytest.raises(ValueError):
        _fitted().with_within_structure(sse, n)


def test_with_within_structure_changes_neither_predictions_nor_the_digest():
    cfg = simple_config(max_wl_depth=2)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(1.0, 10)
    np.testing.assert_array_equal(sieve.predict(w, batch), sieve.predict(m, batch))
    assert w.config.schema_version == m.config.schema_version


def test_merge_adds_the_sums():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted().with_within_structure(1.0, 5)
    ab = a.merge(b)
    np.testing.assert_allclose(ab.within_sse, [3.0])
    assert ab.within_n == 15.0


def test_merge_with_a_model_without_statistics_keeps_the_other_side():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted()
    for merged in (a.merge(b), b.merge(a)):
        assert merged.within_sse is not None
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


def test_merge_of_two_models_without_statistics_has_none():
    ab = _fitted().merge(_fitted())
    assert ab.within_sse is None
    assert ab.within_n == 0.0


def test_the_empty_model_is_still_the_merge_identity():
    a = _fitted().with_within_structure(2.0, 10)
    e = SieveModel.empty(a.config)
    for merged in (a.merge(e), e.merge(a)):
        assert merged.within_sse is not None
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


# ----------------------------------------------------------- persistence --


def test_save_load_round_trips_the_sums(tmp_path):
    m = _fitted().with_within_structure(2.5e-3, 25)
    path = tmp_path / "m.npz"
    m.save(path)
    back = SieveModel.load(path)
    assert back.within_sse is not None and m.within_sse is not None
    np.testing.assert_array_equal(back.within_sse, m.within_sse)
    assert back.within_n == m.within_n


def test_a_file_without_statistics_loads_and_saves_back_without_them(tmp_path):
    m = _fitted()
    path = tmp_path / "m.npz"
    m.save(path)
    assert "within" not in np.load(path).files
    back = SieveModel.load(path)
    assert back.within_sse is None
    assert back.within_n == 0.0
    again = tmp_path / "again.npz"
    back.save(again)
    assert sorted(np.load(again).files) == sorted(np.load(path).files)


# ------------------------------------------------------ predictive variance --


def test_alpha_v_is_ten():
    from sieve.uncertainty import ALPHA_T, ALPHA_V, SELECTION_WEIGHT

    assert (ALPHA_V, ALPHA_T, SELECTION_WEIGHT) == (10.0, 1.0, 0.5)


def test_predictive_variance_adds_the_within_variance_exactly():
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    w = m.with_within_structure(4.0e-2, 100)
    for base, form_b in zip(
        predictive_variance(m), predictive_variance(w), strict=True
    ):
        np.testing.assert_array_equal(form_b, base + 4.0e-4)


def test_without_statistics_the_variance_is_the_three_terms():
    """σ²_w = 0 adds nothing: bit-identical to an explicit zero."""
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    z = m.with_within_structure(0.0, 0)
    for a, b in zip(predictive_variance(m), predictive_variance(z), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unmatched_nodes_fall_back_to_global_msd_plus_the_within_variance():
    from sieve.batch import NodeBatch

    cfg = simple_config(max_wl_depth=1, predictive_variance=True)
    m = sieve.fit(chain_batch(10, graphs=2), cfg).with_within_structure(1.0, 10)
    oov = NodeBatch(
        node_attrs=np.array([[7]], np.int64),  # an element code never fitted
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attrs=np.zeros((0, 1), np.int64),
        graph_id=np.zeros(1, np.int64),
    )
    out = sieve.predict_detailed(m, oov)
    assert out.matched_level[0] == -1
    assert out.predictive_variance is not None
    np.testing.assert_allclose(out.predictive_variance[0], m.global_msd + 0.1)


def test_matched_nodes_read_form_b():
    cfg = simple_config(max_wl_depth=2, predictive_variance=True)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(2.0, 10)
    base = sieve.predict_detailed(m, batch).predictive_variance
    form_b = sieve.predict_detailed(w, batch).predictive_variance
    assert base is not None and form_b is not None
    np.testing.assert_allclose(form_b, base + 0.2, rtol=1e-12)


# ----------------------------------------------------------------- batch --


def _with_within(batch, sse, n):
    from sieve.batch import NodeBatch

    return NodeBatch(**{**batch.__dict__, "within_sse": sse, "within_n": n})


def _within_batch(n=10, graphs=3, seed=0):
    b = chain_batch(n, graphs=graphs, seed=seed)
    rng = np.random.default_rng(seed + 100)
    counts = rng.integers(1, 5, size=b.n_nodes).astype(np.float64)
    sse = rng.uniform(0.0, 1e-3, size=(b.n_nodes, 1)) * counts[:, None]
    return _with_within(b, sse, counts)


def test_a_batch_carries_the_within_arrays_through_slicing():
    b = _within_batch()
    mask = b.graph_id != 1
    sub = b[mask]
    assert sub.within_sse is not None and sub.within_n is not None
    assert b.within_sse is not None and b.within_n is not None
    np.testing.assert_array_equal(sub.within_sse, b.within_sse[mask])
    np.testing.assert_array_equal(sub.within_n, b.within_n[mask])


def test_concat_joins_the_within_arrays():
    from sieve.batch import concat_batches

    a, b = _within_batch(seed=0), _within_batch(seed=1)
    ab = concat_batches([a, b])
    assert ab.within_sse is not None and ab.within_n is not None
    np.testing.assert_array_equal(
        ab.within_sse, np.concatenate([a.within_sse, b.within_sse])
    )
    np.testing.assert_array_equal(ab.within_n, np.concatenate([a.within_n, b.within_n]))


def test_concat_refuses_within_arrays_on_some_parts_only():
    from sieve.batch import concat_batches

    with pytest.raises(ValueError, match="within"):
        concat_batches([_within_batch(), chain_batch(10, graphs=3)])


@pytest.mark.parametrize(
    "bad",
    [
        {"sse": -1e-4},  # negative SSE
        {"sse": np.nan},  # not finite
        {"n": -1.0},  # negative count
        {"n": 0.0},  # positive SSE with no members
    ],
)
def test_impossible_within_arrays_are_refused(bad):
    b = chain_batch(4)
    sse = np.full((b.n_nodes, 1), bad.get("sse", 1e-4))
    n = np.full(b.n_nodes, bad.get("n", 2.0))
    with pytest.raises(ValueError, match="within"):
        _with_within(b, sse, n)


def test_within_arrays_must_come_together_and_match_the_node_count():
    b = chain_batch(4)
    with pytest.raises(ValueError, match="within"):
        _with_within(b, np.zeros((b.n_nodes, 1)), None)
    with pytest.raises(ValueError, match="within"):
        _with_within(b, np.zeros((b.n_nodes - 1, 1)), np.ones(b.n_nodes - 1))


def test_a_zero_count_is_allowed_where_the_sse_is_zero():
    b = chain_batch(4)
    _with_within(b, np.zeros((b.n_nodes, 1)), np.zeros(b.n_nodes))


# ------------------------------------------------------------------- fit --


def test_fit_sums_the_batch_within_arrays():
    b = _within_batch()
    m = sieve.fit(b, simple_config(max_wl_depth=2))
    assert m.within_sse is not None and b.within_sse is not None
    assert b.within_n is not None
    np.testing.assert_allclose(m.within_sse, b.within_sse.sum(axis=0), rtol=1e-12)
    assert m.within_n == pytest.approx(float(b.within_n.sum()), rel=1e-12)
    # statistics, not configuration: the digest and every class are unchanged
    plain = sieve.fit(chain_batch(10, graphs=3), simple_config(max_wl_depth=2))
    assert m.config.schema_version == plain.config.schema_version
    np.testing.assert_array_equal(sieve.predict(m, b), sieve.predict(plain, b))


def test_merge_of_two_fits_equals_the_fit_of_their_union():
    """The merge monoid (design.md 5.4), for the new statistics."""
    from tests.helpers import split_batch

    b = _within_batch(graphs=4)
    cfg = simple_config(max_wl_depth=2)
    mask = b.graph_id < 2
    merged = sieve.fit(split_batch(b, mask), cfg).merge(
        sieve.fit(split_batch(b, ~mask), cfg)
    )
    whole = sieve.fit(b, cfg)
    assert merged.within_sse is not None and whole.within_sse is not None
    np.testing.assert_allclose(merged.within_sse, whole.within_sse, rtol=1e-12)
    assert merged.within_n == pytest.approx(whole.within_n, rel=1e-12)


def test_chunked_fit_sums_the_within_statistics():
    b = _within_batch(graphs=6)
    whole = sieve.fit(b, simple_config(max_wl_depth=2))
    chunked = sieve.fit(b, simple_config(max_wl_depth=2, chunk_size=12))
    assert chunked.within_sse is not None and whole.within_sse is not None
    np.testing.assert_allclose(chunked.within_sse, whole.within_sse, rtol=1e-12)
    assert chunked.within_n == pytest.approx(whole.within_n, rel=1e-12)


def test_a_batch_without_within_arrays_fits_as_before():
    m = sieve.fit(chain_batch(10, graphs=3), simple_config())
    assert m.within_sse is None and m.within_n == 0.0


# --------------------------------------------------------------- adapter --


def _labelled_mols(smiles_list, *, within=True):
    from rdkit import Chem

    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    mols = []
    for j, smi in enumerate(smiles_list):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        for a in m.GetAtoms():
            a.SetDoubleProp("q", 0.01 * a.GetIdx())
            if within:
                a.SetDoubleProp("q" + WITHIN_SSE_SUFFIX, 1e-4 * (a.GetIdx() + j))
                a.SetDoubleProp("q" + WITHIN_N_SUFFIX, float(1 + j))
        mols.append(m)
    return mols


def _adapter_config(mols):
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes

    codes, edges = build_codes(mols, ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=1,
    )


def test_the_adapter_reads_the_companion_properties():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO", "CCN"])
    cfg = _adapter_config(mols)
    b = from_rdkit(mols, config=cfg, y_from_atom_prop="q", within_from_atom_prop="q")
    assert b.within_sse is not None and b.within_n is not None
    n0 = mols[0].GetNumAtoms()
    np.testing.assert_allclose(b.within_sse[:n0, 0], 1e-4 * np.arange(n0))
    np.testing.assert_array_equal(b.within_n[:n0], np.ones(n0))
    np.testing.assert_array_equal(b.within_n[n0:], np.full(b.n_nodes - n0, 2.0))


def test_without_within_from_atom_prop_the_batch_has_none():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO"])
    b = from_rdkit(mols, config=_adapter_config(mols), y_from_atom_prop="q")
    assert b.within_sse is None and b.within_n is None


def test_a_missing_within_property_is_refused():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO"]) + _labelled_mols(["CCN"], within=False)
    with pytest.raises(KeyError):
        from_rdkit(
            mols,
            config=_adapter_config(mols),
            y_from_atom_prop="q",
            within_from_atom_prop="q",
        )


def test_within_properties_survive_parallel_featurisation():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO", "CCN", "CCC", "OCO", "NCN", "CC=O"])
    cfg = _adapter_config(mols)
    seq = from_rdkit(mols, config=cfg, y_from_atom_prop="q", within_from_atom_prop="q")
    par = from_rdkit(
        mols, config=cfg, y_from_atom_prop="q", within_from_atom_prop="q", n_jobs=2
    )
    assert par.within_sse is not None and seq.within_sse is not None
    assert par.within_n is not None and seq.within_n is not None
    np.testing.assert_array_equal(par.within_sse, seq.within_sse)
    np.testing.assert_array_equal(par.within_n, seq.within_n)
