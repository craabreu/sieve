import numpy as np
import pytest

import sieve
from sieve.merge import fold
from tests.helpers import chain_batch, simple_config, split_batch, star_batch


def preds(model, batch):
    from sieve.predict import predict

    return predict(model, batch)


def test_merge_of_disjoint_shards_equals_fitting_the_union():
    """design.md 5.2: the headline property. Everything else is machinery."""
    cfg = simple_config()
    b = chain_batch(12, graphs=6)
    mask = b.graph_id < 3
    a = sieve.fit(split_batch(b, mask), cfg)
    c = sieve.fit(split_batch(b, ~mask), cfg)
    merged = a.merge(c)
    whole = sieve.fit(b, cfg)
    _assert_same_statistics(merged, whole)


def test_merge_is_commutative():
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    a = sieve.fit(split_batch(b, b.graph_id < 2), cfg)
    c = sieve.fit(split_batch(b, b.graph_id >= 2), cfg)
    _assert_same_statistics(a.merge(c), c.merge(a))


def test_merge_is_associative():
    cfg = simple_config()
    b = chain_batch(9, graphs=6)
    parts = [sieve.fit(split_batch(b, b.graph_id == g), cfg) for g in range(6)]
    left = parts[0].merge(parts[1]).merge(parts[2])
    right = parts[0].merge(parts[1].merge(parts[2]))
    _assert_same_statistics(left, right)


def test_empty_model_is_the_identity():
    cfg = simple_config()
    m = sieve.fit(chain_batch(8), cfg)
    e = sieve.SieveModel.empty(cfg)
    _assert_same_statistics(m.merge(e), m)
    _assert_same_statistics(e.merge(m), m)


def test_merge_reproduces_variance_not_just_the_mean():
    """Trap 2: swapping the delta and mean updates leaves means correct and
    corrupts every variance for classes present in both models."""
    cfg = simple_config()
    b = chain_batch(14, graphs=8, seed=3)
    a = sieve.fit(split_batch(b, b.graph_id < 4), cfg)
    c = sieve.fit(split_batch(b, b.graph_id >= 4), cfg)
    whole = sieve.fit(b, cfg)
    merged = a.merge(c)
    for k in range(len(whole.levels)):
        np.testing.assert_allclose(
            np.sort(merged.levels[k].msd, axis=0),
            np.sort(whole.levels[k].msd, axis=0),
            rtol=1e-10,
            atol=1e-12,
        )


def test_class_present_in_only_one_model_is_carried_through_exactly():
    cfg = simple_config()
    b = chain_batch(10, graphs=2)
    a = sieve.fit(split_batch(b, b.graph_id == 0), cfg)
    e = sieve.SieveModel.empty(cfg)
    m = e.merge(a)
    np.testing.assert_array_equal(m.levels[-1].count, a.levels[-1].count)
    np.testing.assert_allclose(m.levels[-1].msd, a.levels[-1].msd)


def test_incompatible_configs_are_rejected_loudly():
    b = chain_batch(8)
    a = sieve.fit(b, simple_config(max_wl_depth=2))
    c = sieve.fit(b, simple_config(max_wl_depth=3))
    with pytest.raises(ValueError, match="schema"):
        a.merge(c)


def test_parent_relations_survive_the_merge():
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(11, graphs=5)
    a = sieve.fit(split_batch(b, b.graph_id < 2), cfg)
    c = sieve.fit(split_batch(b, b.graph_id >= 2), cfg)
    m = a.merge(c)
    for k in range(1, len(m.levels)):
        assert m.levels[k].parent.min() >= 0
        assert m.levels[k].parent.max() < m.levels[k - 1].n_classes
        # support monotonicity must still hold after merging
        assert np.all(m.levels[k].count <= m.levels[k - 1].count[m.levels[k].parent])


def test_fold_matches_sequential_merge():
    cfg = simple_config()
    b = chain_batch(7, graphs=8)
    parts = [sieve.fit(split_batch(b, b.graph_id == g), cfg) for g in range(8)]
    seq = parts[0]
    for p in parts[1:]:
        seq = seq.merge(p)
    _assert_same_statistics(fold(parts, cfg), seq)


def test_merge_across_differing_max_degree_shards():
    """A shard's WL signature width tracks *its own* max degree (design.md
    7.2); merging shards with different max degrees must not misalign the
    padding columns and silently split classes that fitting the union keeps
    together."""
    from sieve.batch import AtomBatch

    cfg = simple_config()
    star = star_batch(4, graphs=2)  # max degree 4
    chain = chain_batch(3, graphs=2, seed=1)  # max degree 2
    chain = AtomBatch(
        node_attrs=chain.node_attrs,
        edge_src=chain.edge_src + star.n_atoms,
        edge_dst=chain.edge_dst + star.n_atoms,
        edge_attr=chain.edge_attr,
        graph_id=chain.graph_id + 2,
        y=chain.y,
    )
    whole = AtomBatch(
        node_attrs=np.concatenate([star.node_attrs, chain.node_attrs]),
        edge_src=np.concatenate([star.edge_src, chain.edge_src]),
        edge_dst=np.concatenate([star.edge_dst, chain.edge_dst]),
        edge_attr=np.concatenate([star.edge_attr, chain.edge_attr]),
        graph_id=np.concatenate([star.graph_id, chain.graph_id]),
        y=np.concatenate([star.y, chain.y]),
    )

    a = sieve.fit(split_batch(whole, whole.graph_id < 2), cfg)  # max degree 4
    c = sieve.fit(split_batch(whole, whole.graph_id >= 2), cfg)  # max degree 2
    merged = a.merge(c)
    fitted_whole = sieve.fit(whole, cfg)
    _assert_same_statistics(merged, fitted_whole)
    # Aggregate stats can coincidentally match under a wrong partition; the
    # actual per-node predictions are what a silent misalignment corrupts.
    np.testing.assert_allclose(preds(merged, whole), preds(fitted_whole, whole))


def test_chunked_fit_equals_single_chunk_fit():
    """design.md 4.1: chunk size is a memory decision, not a statistical one."""
    b = chain_batch(9, graphs=10)
    whole = sieve.fit(b, simple_config())
    chunked = sieve.fit(b, simple_config(chunk_size=20))
    _assert_same_statistics(chunked, whole)


def _assert_same_statistics(a, b):
    """Compare two models up to class-id permutation, level by level."""
    assert len(a.levels) == len(b.levels)
    assert a.global_count == b.global_count
    np.testing.assert_allclose(a.global_mean, b.global_mean, rtol=1e-12)
    for x, y in zip(a.levels, b.levels, strict=True):
        assert x.n_classes == y.n_classes
        ox = np.lexsort((x.msd[:, 0], x.mean[:, 0], x.count))
        oy = np.lexsort((y.msd[:, 0], y.mean[:, 0], y.count))
        np.testing.assert_array_equal(x.count[ox], y.count[oy])
        np.testing.assert_allclose(x.mean[ox], y.mean[oy], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(x.msd[ox], y.msd[oy], rtol=1e-10, atol=1e-12)
