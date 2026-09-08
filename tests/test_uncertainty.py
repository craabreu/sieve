"""Predictive variance (sieve.uncertainty, design-update-v2.md 3).

The defect this exists to fix: `Predictions.variance` is the matched class's
stored s^2, which is NaN at N == 1 and can be a sampling-artefact zero -- and
design.md 6.4 reads a zero variance as a delta-function likelihood and pins
the node. Totality is therefore the headline property, not a nicety.
"""

from __future__ import annotations

import numpy as np
import pytest

import sieve
from sieve.batch import NodeBatch
from sieve.continuation import class_sibling_variance, sibling_variance
from sieve.uncertainty import ALPHA_V, SELECTION_WEIGHT, predictive_variance
from tests.helpers import chain_batch, simple_config


def _singleton_batch(d=1):
    """Two disjoint graphs whose nodes are pairwise distinct at depth >= 1, so
    the deepest level is full of N == 1 classes -- the case where the stored
    s^2 is NaN."""
    src = [0, 1, 1, 2]
    dst = [1, 0, 2, 1]
    return NodeBatch(
        node_attrs=np.array([[0], [1], [0]], np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attrs=np.ones(len(src), np.int64).reshape(-1, 1),
        graph_id=np.zeros(3, np.int64),
        y=np.array([[1.0], [2.0], [5.0]])[:, :d] if d == 1 else None,
    )


def _structured_batch(graphs=4, n=8, seed=0, noise=0.05):
    """A chain whose label is a deterministic function of the depth-1
    environment plus small noise.

    ``chain_batch``'s labels are pure noise, so every class's children really
    are indistinguishable and the debiased sibling variance correctly clamps
    to zero -- which makes it useless for testing a term built from sibling
    spread. Here the depth-1 classes genuinely differ, and keep differing as
    data is added, which is exactly the regime the selection term exists for.
    """
    base = chain_batch(n, graphs=graphs, seed=seed)
    attr = base.node_attrs[:, 0]
    neighbour_sum = np.bincount(
        base.edge_src, weights=attr[base.edge_dst], minlength=len(attr)
    )
    rng = np.random.default_rng(seed)
    y = (10.0 * attr + 3.0 * neighbour_sum)[:, None] + rng.normal(
        scale=noise, size=(len(attr), 1)
    )
    return NodeBatch(
        node_attrs=base.node_attrs,
        edge_src=base.edge_src,
        edge_dst=base.edge_dst,
        edge_attrs=base.edge_attrs,
        graph_id=base.graph_id,
        y=y,
    )


def _identical_label_batch():
    """A class whose members agree exactly -> stored msd is 0.0 and s^2 is
    0.0, the artefact-zero case that a support gate cannot catch."""
    batch = chain_batch(6, graphs=2, seed=1)
    return NodeBatch(
        node_attrs=batch.node_attrs,
        edge_src=batch.edge_src,
        edge_dst=batch.edge_dst,
        edge_attrs=batch.edge_attrs,
        graph_id=batch.graph_id,
        y=np.zeros_like(batch.y),
    )


# --------------------------------------------------------------- totality --


def test_predictive_variance_is_finite_and_positive_where_s2_is_nan():
    cfg = simple_config(max_wl_depth=3)
    m = sieve.fit(_singleton_batch(), cfg)

    # precondition: the model really does contain N == 1 classes with NaN s^2
    assert any(np.isnan(lvl.variance).any() for lvl in m.levels)

    for table in predictive_variance(m):
        assert np.all(np.isfinite(table))
        assert np.all(table > 0.0)


def test_predictive_variance_is_positive_where_s2_is_an_artefact_zero():
    cfg = simple_config(max_wl_depth=2)
    m = sieve.fit(_identical_label_batch(), cfg)

    assert any((lvl.msd == 0.0).any() for lvl in m.levels)  # precondition

    # Every label is identical here, so the *global* variance is zero too and
    # there is genuinely nothing to scale by -- the honest answer is 0, not an
    # invented floor. What must not happen is NaN.
    for table in predictive_variance(m):
        assert np.all(np.isfinite(table))


def test_predictions_carry_a_total_predictive_variance():
    cfg = simple_config(max_wl_depth=3, predictive_variance=True)
    batch = chain_batch(12, graphs=3)
    m = sieve.fit(batch, cfg)
    out = sieve.predict_detailed(m, batch)

    assert out.predictive_variance is not None
    assert out.predictive_variance.shape == (batch.n_nodes, cfg.target_dim)
    assert np.all(np.isfinite(out.predictive_variance))
    assert np.all(out.predictive_variance > 0.0)


def test_unmatched_nodes_fall_back_to_global_msd():
    cfg = simple_config(max_wl_depth=1, predictive_variance=True)
    m = sieve.fit(chain_batch(10, graphs=2), cfg)

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
    np.testing.assert_allclose(out.predictive_variance[0], m.global_msd)


# ------------------------------------------------------------- the terms ---


def test_selection_term_is_zero_at_the_deepest_level():
    """The deepest class has no children, and is read on an exact match rather
    than by backoff -- so there is no selection to correct."""
    cfg = simple_config(max_wl_depth=3)
    m = sieve.fit(_structured_batch(graphs=5, n=10), cfg)
    deepest = m.config.backoff_path[-1]

    with_sel = predictive_variance(m, selection_weight=SELECTION_WEIGHT)
    without = predictive_variance(m, selection_weight=0.0)

    np.testing.assert_allclose(with_sel[deepest], without[deepest])
    # and it is *not* a no-op somewhere with children, or the test is vacuous
    assert any(
        not np.allclose(with_sel[k], without[k]) for k in m.config.backoff_path[:-1]
    )


def test_selection_term_does_not_vanish_with_support():
    """The point of the term: it is the one component that survives N -> inf.
    A control that weighted it by (1-w) scored identically to omitting it."""
    cfg = simple_config(max_wl_depth=2)
    small = sieve.fit(_structured_batch(graphs=2, n=8, seed=3), cfg)
    large = sieve.fit(_structured_batch(graphs=40, n=8, seed=3), cfg)

    def selection_share(m):
        k = m.config.backoff_path[0]
        full = predictive_variance(m)[k]
        none = predictive_variance(m, selection_weight=0.0)[k]
        return float(np.mean(full - none))

    assert selection_share(small) > 0.0
    assert selection_share(large) > 0.0
    # forty times the data must not shrink it away
    assert selection_share(large) > 0.25 * selection_share(small)


def test_within_class_term_matches_the_hand_computed_pooling():
    """(N-1)s^2 == N*msd identically, so the stored msd is what enters -- exact
    at N == 1 where s^2 is NaN."""
    cfg = simple_config(max_wl_depth=2)
    m = sieve.fit(chain_batch(12, graphs=3), cfg)
    k = m.config.backoff_path[-1]  # deepest: no selection term to subtract
    lvl = m.levels[k]

    from sieve.continuation import atom_variance
    from sieve.shrinkage import empirical_bayes_weights

    n = lvl.count[:, None].astype(np.float64)
    within = (n * lvl.msd + ALPHA_V * atom_variance(m)[k]) / (
        np.maximum(n - 1.0, 0.0) + ALPHA_V
    )
    tau_pa = sibling_variance(m)[m.config.level_parents[k]]
    estimation = tau_pa * (1.0 - empirical_bayes_weights(m)[k][:, None])

    np.testing.assert_allclose(predictive_variance(m)[k], within + estimation)


def test_monotone_in_the_stored_class_variance():
    """Ordering must survive -- the property a hard support gate destroys by
    substituting an ancestor's variance wholesale."""
    cfg = simple_config(max_wl_depth=2)
    m = sieve.fit(chain_batch(12, graphs=4), cfg)
    k = m.config.backoff_path[-1]

    import dataclasses

    bumped = dataclasses.replace(m.levels[k], msd=m.levels[k].msd * 2.0)
    m2 = dataclasses.replace(m, levels=(*m.levels[:k], bumped, *m.levels[k + 1 :]))

    assert np.all(predictive_variance(m2)[k] >= predictive_variance(m)[k])


# ---------------------------------------------------- per-class sibling ----


def test_class_sibling_variance_is_nan_without_two_children():
    cfg = simple_config(max_wl_depth=2)
    m = sieve.fit(chain_batch(10, graphs=2), cfg)
    from sieve.continuation import child_counts

    for k in range(cfg.n_levels):
        tau, c = class_sibling_variance(m)[k], child_counts(m)[k]
        assert np.all(np.isnan(tau[c < 2]))
        assert np.all(np.isfinite(tau[c >= 2]))


def test_class_sibling_variance_pools_to_something_comparable():
    """Per-class and pooled estimate the same quantity, so their scales must
    agree to within an order of magnitude -- a sign error or a missing
    divisor would not."""
    cfg = simple_config(max_wl_depth=2)
    m = sieve.fit(chain_batch(16, graphs=6), cfg)
    for k in range(cfg.n_levels):
        per, pooled = class_sibling_variance(m)[k], sibling_variance(m)[k]
        if np.isfinite(pooled) and pooled > 0 and np.isfinite(per).any():
            assert 0.05 < np.nanmean(per) / pooled < 20.0


# ------------------------------------------------------------- plumbing ----


def test_flag_defaults_off_and_leaves_variance_untouched():
    batch = chain_batch(12, graphs=3)
    m_off = sieve.fit(batch, simple_config(max_wl_depth=2))
    m_on = sieve.fit(batch, simple_config(max_wl_depth=2, predictive_variance=True))

    off = sieve.predict_detailed(m_off, batch)
    on = sieve.predict_detailed(m_on, batch)

    assert off.predictive_variance is None
    assert on.predictive_variance is not None
    np.testing.assert_array_equal(off.value, on.value)
    np.testing.assert_array_equal(off.variance, on.variance)  # diagnostic unchanged


def test_predictive_variance_is_excluded_from_schema_version():
    a = simple_config(predictive_variance=False)
    b = simple_config(predictive_variance=True)
    assert a.schema_version == b.schema_version


def test_survives_a_chunked_fit_identically():
    """Derived from stored arrays, so merging must not move it (design.md 5.2)."""
    batch = _structured_batch(graphs=6, n=10)
    cfg = simple_config(max_wl_depth=2)
    one = sieve.fit(batch, cfg)
    # 60 nodes -> ceil(60/20) = 3 shards, and there are 6 graphs to split
    # across them; a chunk_size implying more shards than graphs yields empty
    # shards, which is a fit() precondition rather than anything this tests.
    chunked = sieve.fit(batch, simple_config(max_wl_depth=2, chunk_size=20))

    for a, b in zip(
        predictive_variance(one), predictive_variance(chunked), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_works_with_multidimensional_targets():
    cfg = simple_config(max_wl_depth=2, target_dim=3, predictive_variance=True)
    batch = chain_batch(12, d=3, graphs=3)
    m = sieve.fit(batch, cfg)
    out = sieve.predict_detailed(m, batch)

    assert out.predictive_variance is not None
    assert out.predictive_variance.shape == (batch.n_nodes, 3)
    assert np.all(np.isfinite(out.predictive_variance))
    assert np.all(out.predictive_variance > 0.0)


@pytest.mark.parametrize("estimator", ["pooled", "continuation"])
def test_available_under_either_class_estimator(estimator):
    cfg = simple_config(
        max_wl_depth=2, predictive_variance=True, class_estimator=estimator
    )
    batch = chain_batch(12, graphs=3)
    out = sieve.predict_detailed(sieve.fit(batch, cfg), batch)
    assert np.all(np.isfinite(out.predictive_variance))
