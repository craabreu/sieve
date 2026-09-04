"""Tests for the continuation class estimator (design.md 4.4)."""

from __future__ import annotations

import numpy as np

import sieve
from sieve.continuation import class_means
from sieve.shrinkage import shrunk_means
from tests.helpers import chain_batch, simple_config


def test_pooled_estimator_is_the_stored_means_unchanged():
    """No copy, no arithmetic -- the same arrays predict.py already read
    before this module existed, so pooled predictions must stay bit-identical
    to today's."""
    cfg = simple_config(max_wl_depth=2, class_estimator="pooled")
    m = sieve.fit(chain_batch(15, graphs=3), cfg)
    means = class_means(m)
    for k, lvl in enumerate(m.levels):
        assert np.shares_memory(means[k], lvl.mean)


def test_continuation_matches_a_hand_computed_child_average():
    """The unweighted mean of a class's children's own stored means -- one
    level deep, over the *stored* (level-1) means, not recursively over
    another continuation estimate."""
    cfg = simple_config(max_wl_depth=2, class_estimator="continuation")
    m = sieve.fit(chain_batch(15, graphs=3), cfg)
    means = class_means(m)

    parent_level, child_level = m.levels[0], m.levels[1]
    expected = np.array(
        [
            child_level.mean[child_level.parent == c].mean(axis=0)
            if (child_level.parent == c).any()
            else parent_level.mean[c]
            for c in range(len(parent_level.mean))
        ]
    )
    np.testing.assert_allclose(means[0], expected)
    # every level-0 class in a fitted model has >=1 child, so the "no
    # children" branch above is dead in practice here -- assert that too,
    # so a change that breaks class minting would be caught alongside it.
    assert all((child_level.parent == c).any() for c in range(len(parent_level.mean)))


def test_continuation_and_pooled_agree_on_the_deepest_level():
    """The deepest backoff level has no children by construction (nothing
    refines it further), so it keeps its pooled mean under either
    estimator -- this is the level every exactly-matched query reads."""
    cfg_pooled = simple_config(max_wl_depth=2, class_estimator="pooled")
    cfg_cont = simple_config(max_wl_depth=2, class_estimator="continuation")
    b = chain_batch(15, graphs=3)
    mp = sieve.fit(b, cfg_pooled)
    mc = sieve.fit(b, cfg_cont)
    np.testing.assert_array_equal(class_means(mp)[-1], class_means(mc)[-1])
    np.testing.assert_array_equal(class_means(mc)[-1], mc.levels[-1].mean)


def test_continuation_only_changes_predictions_for_backed_off_nodes():
    """A node whose own finest class is present matches at the deepest level,
    which keeps its pooled mean either way -- so continuation can only move
    predictions for nodes that back off above it."""
    cfg_pooled = simple_config(max_wl_depth=2, class_estimator="pooled")
    cfg_cont = simple_config(max_wl_depth=2, class_estimator="continuation")
    b = chain_batch(15, graphs=3)
    mp, mc = sieve.fit(b, cfg_pooled), sieve.fit(b, cfg_cont)
    pp, pc = sieve.predict_detailed(mp, b), sieve.predict_detailed(mc, b)
    deepest = cfg_pooled.n_levels - 1
    at_deepest = pp.matched_level == deepest
    assert at_deepest.any(), "test needs at least one exactly-matched node"
    np.testing.assert_allclose(pp.value[at_deepest], pc.value[at_deepest])


def test_search_agrees_with_shrunk_means_under_continuation_and_shrinkage():
    """The change-consistency test: predict.py's in-loop shrinkage
    recomputation and shrinkage.py's own shrunk_means must read the same
    per-level table, or the two silently diverge. Non-LOO only -- this is
    exactly the case where the loop's recomputed value is defined to equal
    shrunk_means (see predict.py's own comment on that block)."""
    cfg = simple_config(
        max_wl_depth=2, class_estimator="continuation", shrinkage_strength=3.0
    )
    b = chain_batch(15, graphs=3)
    m = sieve.fit(b, cfg)
    detailed = sieve.predict_detailed(m, b)
    shrunk = shrunk_means(m)

    backoff_path = cfg.backoff_path
    for pos, k in enumerate(backoff_path):
        sel = detailed.matched_level == pos
        if not sel.any():
            continue
        expected = shrunk[k][detailed.class_id[sel]]
        np.testing.assert_allclose(detailed.value[sel], expected)


def test_predict_loo_rejects_continuation():
    cfg = simple_config(max_wl_depth=2, class_estimator="continuation")
    b = chain_batch(15, graphs=3)
    m = sieve.fit(b, cfg)
    try:
        sieve.predict_loo(m, b)
    except NotImplementedError as e:
        assert "continuation" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


def test_with_params_switches_class_estimator_without_refitting():
    cfg = simple_config(max_wl_depth=2, class_estimator="pooled")
    b = chain_batch(15, graphs=3)
    m = sieve.fit(b, cfg)
    switched = m.with_params(class_estimator="continuation")
    assert switched.config.class_estimator == "continuation"
    # same fitted arrays, only the config differs
    for lvl_a, lvl_b in zip(m.levels, switched.levels, strict=True):
        np.testing.assert_array_equal(lvl_a.mean, lvl_b.mean)
