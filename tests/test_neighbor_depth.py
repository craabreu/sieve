"""Coarse neighbor attributes: `neighbor_depth` (design.md 3.6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sieve
from sieve.batch import NodeBatch
from sieve.config import SieveConfig
from sieve.dedupe import dense_rows
from sieve.predict import predict, predict_detailed
from sieve.refine import refine
from tests.helpers import split_batch

_COARSE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",), ("degree",)),
    attribute_codes={"element": {"C": 0, "N": 1, "O": 2}, "degree": {"1": 0, "2": 1}},
    edge_codes={"SINGLE": 1, "DOUBLE": 2},
    max_wl_depth=3,
)


def coarse_config(**kw):
    return replace(_COARSE_CONFIG, **kw)


def random_batch(n_graphs=8, min_n=3, max_n=9, seed=0, d=1):
    """Small connected-ish graphs with irregular degree and two attribute
    columns -- variable degree is what actually exercises the multiset
    machinery both WL chains share; chain_batch/star_batch alone (uniform
    degree per graph) do not."""
    rng = np.random.default_rng(seed)
    node_attrs, graph_id = [], []
    edge_src, edge_dst, edge_attr = [], [], []
    off = 0
    for gi in range(n_graphs):
        n = int(rng.integers(min_n, max_n + 1))
        attrs = rng.integers(0, [3, 2], size=(n, 2))
        node_attrs.append(attrs)
        graph_id.append(np.full(n, gi))
        nodes = list(range(n))
        rng.shuffle(nodes)
        edges = set()
        for i in range(1, n):
            j = int(rng.integers(0, i))
            edges.add((nodes[i], nodes[j]))
        for _ in range(int(rng.integers(0, max(1, n // 2)))):
            a, b = (int(x) for x in rng.integers(0, n, size=2))
            if a != b:
                edges.add((min(a, b), max(a, b)))
        for a, b in edges:
            bond = int(rng.integers(1, 3))
            edge_src += [off + a, off + b]
            edge_dst += [off + b, off + a]
            edge_attr += [bond, bond]
        off += n
    return NodeBatch(
        node_attrs=np.concatenate(node_attrs, axis=0).astype(np.int64),
        edge_src=np.array(edge_src, np.int64),
        edge_dst=np.array(edge_dst, np.int64),
        edge_attr=np.array(edge_attr, np.int64),
        graph_id=np.concatenate(graph_id).astype(np.int64),
        y=rng.normal(size=(off, d)),
    )


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two label arrays induce exactly the same equivalence classes."""
    fwd: dict[int, set[int]] = {}
    bwd: dict[int, set[int]] = {}
    for x, y in zip(a.tolist(), b.tolist(), strict=True):
        fwd.setdefault(x, set()).add(y)
        bwd.setdefault(y, set()).add(x)
    return all(len(v) == 1 for v in fwd.values()) and all(
        len(v) == 1 for v in bwd.values()
    )


def _naive_coarse_wl(batch: NodeBatch, config: SieveConfig) -> list[np.ndarray]:
    """The reference implementation this feature's collapse is checked
    against: literal per-round sorted multisets of the coarse chain's own
    neighbor labels, computed with none of refine()'s own machinery beyond
    csr()/dense_rows (so this cannot share a bug with refine()'s WL_PAIR
    branch). Returns h_0 .. h_{L-1}, the main chain's WL rounds only.
    """
    m = config.neighbor_depth
    assert m is not None
    n_edge_types = config.n_edge_types
    csr = batch.csr()
    n = batch.n_nodes

    # Coarse base: attribute levels 0 .. m-1 only, chained the same way
    # refine() chains attribute levels (parent id, then this level's cols).
    used = 0
    coarse_base = None
    for group in config.attribute_levels[:m]:
        cols = batch.node_attrs[:, used : used + len(group)]
        used += len(group)
        sig = (
            cols
            if coarse_base is None
            else np.concatenate([coarse_base[:, None], cols], axis=1)
        )
        coarse_base, _ = dense_rows(sig)

    # Fine base: every attribute level, same chaining.
    used = 0
    fine_base = None
    for group in config.attribute_levels:
        cols = batch.node_attrs[:, used : used + len(group)]
        used += len(group)
        sig = (
            cols
            if fine_base is None
            else np.concatenate([fine_base[:, None], cols], axis=1)
        )
        fine_base, _ = dense_rows(sig)

    assert coarse_base is not None and fine_base is not None  # both groups nonempty
    g_prev = coarse_base
    h_prev = fine_base
    naive_h = []
    for _ in range(config.max_wl_depth):
        pair_g = g_prev[csr.dst] * n_edge_types + csr.attr
        pad_g = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
        pad_g[csr.src, csr.slot] = pair_g
        pad_g.sort(axis=1)
        g_new, _ = dense_rows(np.concatenate([g_prev[:, None], pad_g], axis=1))

        # h's neighbor multiset uses the COARSE chain's *previous-round*
        # labels (g_prev, i.e. g at r-1) -- the literal §3.6 recurrence.
        pair_h = g_prev[csr.dst] * n_edge_types + csr.attr
        pad_h = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
        pad_h[csr.src, csr.slot] = pair_h
        pad_h.sort(axis=1)
        h_new, _ = dense_rows(np.concatenate([h_prev[:, None], pad_h], axis=1))

        naive_h.append(h_new)
        g_prev, h_prev = g_new, h_new
    return naive_h


@pytest.mark.parametrize("seed", range(5))
def test_wl_pair_collapse_matches_the_naive_full_multiset_reference(seed):
    """The algebraic shortcut refine() relies on: h_r's signature can be
    collapsed to the pair (h_{r-1}, g_r) instead of a literal sorted
    multiset of neighbors' g_{r-1} labels, because h_{r-1} already
    determines g_{r-1} for every node (h refines g at every round). This
    must be verified against an independent reference, not assumed."""
    batch = random_batch(n_graphs=10, seed=seed)
    cfg = coarse_config(neighbor_depth=1)
    levels = refine(batch, cfg)
    a = len(cfg.attribute_levels)
    depth = cfg.max_wl_depth
    first_h = a + depth
    ours = [levels[first_h + r].labels for r in range(depth)]
    naive = _naive_coarse_wl(batch, cfg)
    for r, (o, nv) in enumerate(zip(ours, naive, strict=True)):
        assert _same_partition(o, nv), f"round {r} diverges from the reference"


def test_neighbor_depth_produces_more_or_equal_reach():
    """Coarsening never adds information (design.md 3.6): the coarsened
    chain's partition at every WL round must be a coarsening of the
    uncoarsened one -- never finer. (neighbor_depth == len(attribute_levels)
    normalizes to None -- see test_neighbor_depth_none_reduces_to_todays_
    chain -- so 1 is the only genuinely-coarsening depth this 2-attribute
    fixture can exercise.)"""
    batch = random_batch(n_graphs=12, seed=1)
    full = refine(batch, coarse_config(neighbor_depth=None))
    coarse = refine(batch, coarse_config(neighbor_depth=1))
    a = len(_COARSE_CONFIG.attribute_levels)
    depth = _COARSE_CONFIG.max_wl_depth
    for r in range(depth):
        full_labels = full[a + r].labels
        coarse_labels = coarse[a + depth + r].labels
        # coarse must be a coarsening: every full-class maps into exactly
        # one coarse-class, but not necessarily the reverse.
        fwd: dict[int, set[int]] = {}
        for x, y in zip(full_labels.tolist(), coarse_labels.tolist(), strict=True):
            fwd.setdefault(x, set()).add(y)
        assert all(len(v) == 1 for v in fwd.values())


def test_neighbor_depth_none_reduces_to_todays_chain():
    """neighbor_depth left unset (or set equal to len(attribute_levels), a
    validated-equivalent spelling) must produce byte-identical labels to
    the single-chain behavior this feature must not disturb."""
    batch = random_batch(n_graphs=6, seed=2)
    base = refine(batch, coarse_config())
    explicit = refine(batch, coarse_config(neighbor_depth=2))  # == n_attr
    assert len(base) == len(explicit)
    for x, y in zip(base, explicit, strict=True):
        np.testing.assert_array_equal(x.labels, y.labels)
        np.testing.assert_array_equal(x.signatures, y.signatures)
        np.testing.assert_array_equal(x.parent, y.parent)


def test_neighbor_depth_none_reduces_predictions_too():
    batch = random_batch(n_graphs=10, seed=3)
    train, test = (
        split_batch(batch, batch.graph_id < 7),
        split_batch(batch, batch.graph_id >= 7),
    )
    m_none = sieve.fit(train, coarse_config(neighbor_depth=None))
    m_full = sieve.fit(train, coarse_config(neighbor_depth=2))
    np.testing.assert_allclose(predict(m_none, test), predict(m_full, test))
    p_none = predict_detailed(m_none, test)
    p_full = predict_detailed(m_full, test)
    np.testing.assert_array_equal(p_none.matched_level, p_full.matched_level)


def test_fit_produces_n_levels_matching_config():
    batch = random_batch(n_graphs=6, seed=4)
    cfg = coarse_config(neighbor_depth=1)
    m = sieve.fit(batch, cfg)
    a, depth = len(cfg.attribute_levels), cfg.max_wl_depth
    assert cfg.n_levels == a + 2 * depth  # attr levels, coarse chain, main chain
    assert len(m.levels) == cfg.n_levels


def test_neighbor_depth_is_dropped_when_there_are_no_wl_rounds():
    """With max_wl_depth == 0 no level ever reads a neighbor, so coarsening
    is a no-op and must normalize away -- the coarse chain would otherwise be
    zero levels long while level_parents still tried to write its branch
    root, indexing off the end of its own list."""
    cfg = coarse_config(max_wl_depth=0, neighbor_depth=1)
    assert cfg.neighbor_depth is None
    assert cfg.n_levels == len(cfg.attribute_levels)
    assert cfg.level_parents == (-1, 0)
    assert cfg.backoff_path == (0, 1)
    assert cfg.schema_version == coarse_config(max_wl_depth=0).schema_version


def test_chunked_fit_matches_whole_corpus_fit_when_coarsened():
    """chunk_size fits shards and folds them (model.py's own recursion), so
    it goes through merge_models' two-remap threading on a path the direct
    merge tests do not exercise."""
    batch = random_batch(n_graphs=14, seed=11)
    cfg = coarse_config(neighbor_depth=1)
    whole = sieve.fit(batch, cfg)
    chunked = sieve.fit(batch, replace(cfg, chunk_size=batch.n_nodes // 4))
    np.testing.assert_allclose(predict(whole, batch), predict(chunked, batch))
    assert [lv.n_classes for lv in whole.levels] == [
        lv.n_classes for lv in chunked.levels
    ]


def test_predict_loo_works_when_coarsened():
    from sieve.predict import predict_loo

    batch = random_batch(n_graphs=12, seed=12)
    cfg = coarse_config(neighbor_depth=1)
    m = sieve.fit(batch, cfg)
    p = predict_loo(m, batch)
    assert np.isfinite(p.value).all()
    assert p.matched_level.max() < len(cfg.backoff_path)


def test_zero_shrinkage_reproduces_the_unshrunk_prediction_when_coarsened():
    """Exercises the shrinkage path -- and so shrunk_means' walk over the
    branching parent chain -- with an assertion that pins the result rather
    than merely checking it runs: at strength 0 the convex combination
    collapses to the raw estimate, so it must match the no-shrinkage model
    exactly."""
    batch = random_batch(n_graphs=12, seed=13)
    cfg = coarse_config(neighbor_depth=1)
    plain = sieve.fit(batch, cfg)
    shrunk = sieve.fit(batch, replace(cfg, shrinkage_strength=0.0))
    np.testing.assert_allclose(predict(plain, batch), predict(shrunk, batch))


def test_merge_of_disjoint_shards_equals_fitting_the_union_when_coarsened():
    """design.md 5.2's headline property, exercised with neighbor_depth set:
    the two-remap threading through merge_models must still make shard-fit
    + merge equal a whole-corpus fit."""
    cfg = coarse_config(neighbor_depth=1)
    batch = random_batch(n_graphs=12, seed=5)
    mask = batch.graph_id < 6
    a = sieve.fit(split_batch(batch, mask), cfg)
    b = sieve.fit(split_batch(batch, ~mask), cfg)
    whole = sieve.fit(batch, cfg)
    merged = a.merge(b)
    np.testing.assert_allclose(predict(merged, batch), predict(whole, batch))
    for x, y in zip(merged.levels, whole.levels, strict=True):
        assert x.n_classes == y.n_classes


def test_parent_relations_survive_the_merge_when_coarsened():
    cfg = coarse_config(neighbor_depth=1)
    batch = random_batch(n_graphs=10, seed=6)
    mask = batch.graph_id < 5
    a = sieve.fit(split_batch(batch, mask), cfg)
    b = sieve.fit(split_batch(batch, ~mask), cfg)
    m = a.merge(b)
    parents = cfg.level_parents
    for k in range(1, len(m.levels)):
        p = parents[k]
        if p < 0:
            continue
        assert m.levels[k].parent.min() >= 0
        assert m.levels[k].parent.max() < m.levels[p].n_classes
        assert np.all(m.levels[k].count <= m.levels[p].count[m.levels[k].parent])


def test_save_load_round_trip_preserves_neighbor_depth(tmp_path):
    cfg = coarse_config(neighbor_depth=1)
    batch = random_batch(n_graphs=8, seed=7)
    m = sieve.fit(batch, cfg)
    path = tmp_path / "model.npz"
    m.save(path)
    loaded = sieve.SieveModel.load(path)
    assert loaded.config.neighbor_depth == 1
    assert loaded.config.schema_version == cfg.schema_version
    np.testing.assert_allclose(predict(loaded, batch), predict(m, batch))


def test_matched_level_is_position_along_backoff_path_not_raw_index():
    """With coarsening on, matched_level must skip the coarse chain's own
    levels -- otherwise it isn't comparable to an uncoarsened config's
    matched_level, which is exactly the comparison design.md 3.6's own
    "reach" measurement needs."""
    batch = random_batch(n_graphs=10, seed=8)
    train, test = (
        split_batch(batch, batch.graph_id < 7),
        split_batch(batch, batch.graph_id >= 7),
    )
    cfg = coarse_config(neighbor_depth=1, minimum_support=1)
    m = sieve.fit(train, cfg)
    p = predict_detailed(m, test)
    n_backoff = len(cfg.backoff_path)
    assert p.matched_level.max() < n_backoff  # never a raw (larger) index
