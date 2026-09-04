"""Tests for the continuation class estimator (design.md 4.4)."""

from __future__ import annotations

import numpy as np

import sieve
from sieve.continuation import child_counts, class_means
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


def test_recursive_agrees_with_flat_at_the_two_deepest_levels():
    """The level above the deepest averages children that are themselves
    pooled, so recursion has nothing to bite on there; only levels at least
    two steps above the deepest can differ."""
    b = chain_batch(20, graphs=4)
    mf = sieve.fit(b, simple_config(max_wl_depth=3, class_estimator="continuation"))
    mr = sieve.fit(
        b, simple_config(max_wl_depth=3, class_estimator="continuation_recursive")
    )
    cf, cr = class_means(mf), class_means(mr)
    np.testing.assert_array_equal(cf[-1], cr[-1])
    np.testing.assert_allclose(cf[-2], cr[-2])
    assert not np.allclose(cf[0], cr[0]), "levels above that must actually differ"


def test_recursive_matches_a_hand_computed_two_level_composition():
    """Level k averages level k+1's *continuation* estimates, which are
    themselves averages of level k+2's stored pooled means."""
    b = chain_batch(20, graphs=4)
    m = sieve.fit(
        b, simple_config(max_wl_depth=3, class_estimator="continuation_recursive")
    )
    rec = class_means(m)
    k = len(m.levels) - 3  # deepest level that recursion actually changes
    child = m.levels[k + 1]
    expected = np.array(
        [
            rec[k + 1][child.parent == c].mean(axis=0)
            for c in range(len(m.levels[k].mean))
        ]
    )
    np.testing.assert_allclose(rec[k], expected)


def test_child_counts_are_kneser_ney_n1plus():
    b = chain_batch(20, graphs=4)
    m = sieve.fit(b, simple_config(max_wl_depth=3))
    counts = child_counts(m)
    for k in range(len(m.levels) - 1):
        child = m.levels[k + 1]
        expected = np.array(
            [(child.parent == c).sum() for c in range(len(m.levels[k].mean))]
        )
        np.testing.assert_array_equal(counts[k], expected)
    np.testing.assert_array_equal(counts[-1], np.zeros(len(m.levels[-1].mean)))


def test_diversity_weight_matches_the_kneser_ney_lambda():
    """lambda = min(D*C/N, 1) on the parent, applied against the *shrunk*
    parent exactly as the count rule is."""
    cfg = simple_config(
        max_wl_depth=3, shrinkage_strength=0.5, shrinkage_weight="diversity"
    )
    b = chain_batch(20, graphs=4)
    m = sieve.fit(b, cfg)
    sh = shrunk_means(m)
    means, counts = class_means(m), child_counts(m)
    for k in range(1, len(m.levels)):
        lvl = m.levels[k]
        n = lvl.count[:, None].astype(float)
        lam = np.minimum(0.5 * counts[k][:, None] / np.maximum(n, 1.0), 1.0)
        expected = (1 - lam) * means[k] + lam * sh[k - 1][lvl.parent]
        np.testing.assert_allclose(sh[k], expected, rtol=1e-12)


def test_diversity_weight_leaves_the_deepest_level_unshrunk():
    """C == 0 there, so lambda == 0 -- the deepest class is the only one read
    on an exact match rather than by backoff."""
    cfg = simple_config(
        max_wl_depth=3, shrinkage_strength=2.0, shrinkage_weight="diversity"
    )
    m = sieve.fit(chain_batch(20, graphs=4), cfg)
    np.testing.assert_allclose(shrunk_means(m)[-1], class_means(m)[-1])


def test_search_agrees_with_shrunk_means_under_diversity_weighting():
    """Same change-consistency guard as the count rule: predict.py's own
    per-node path and shrunk_means must not drift apart."""
    cfg = simple_config(
        max_wl_depth=3,
        class_estimator="continuation",
        shrinkage_strength=0.5,
        shrinkage_weight="diversity",
    )
    b = chain_batch(20, graphs=4)
    m = sieve.fit(b, cfg)
    det = sieve.predict_detailed(m, b)
    shrunk = shrunk_means(m)
    for pos, k in enumerate(cfg.backoff_path):
        sel = det.matched_level == pos
        if sel.any():
            np.testing.assert_allclose(det.value[sel], shrunk[k][det.class_id[sel]])


def test_predict_loo_rejects_recursive_and_diversity():
    b = chain_batch(20, graphs=4)
    for kw in (
        {"class_estimator": "continuation_recursive"},
        {"shrinkage_weight": "diversity", "shrinkage_strength": 0.5},
    ):
        m = sieve.fit(b, simple_config(max_wl_depth=3, **kw))
        try:
            sieve.predict_loo(m, b)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"expected NotImplementedError for {kw}")


def test_empirical_bayes_weight_matches_the_hand_derived_posterior():
    """Non-deepest: C/(C+tau2_k/tau2_parent). Deepest: N/(N+sigma2/tau2_parent)
    -- two estimands, two units of replication (design.md 4.4)."""
    from sieve.continuation import atom_variance, root_variance, sibling_variance
    from sieve.shrinkage import empirical_bayes_weights

    cfg = simple_config(
        max_wl_depth=3,
        class_estimator="continuation",
        shrinkage_weight="empirical_bayes",
    )
    m = sieve.fit(chain_batch(20, graphs=4), cfg)
    w = empirical_bayes_weights(m)
    tau, sigma, counts = sibling_variance(m), atom_variance(m), child_counts(m)
    root, parents = root_variance(m), cfg.level_parents

    for k, lvl in enumerate(m.levels):
        tp = root if parents[k] < 0 else tau[parents[k]]
        if tp != tp or tp <= 0:
            continue  # nan/zero branches asserted separately below
        n, c = lvl.count.astype(float), counts[k]
        num = np.where(c > 0, tau[k], sigma[k])
        size = np.where(c > 0, c, n)
        expected = size / (size + np.where(num == num, num / tp, 0.0))
        np.testing.assert_allclose(w[k], expected, rtol=1e-12)


def test_empirical_bayes_ignores_shrinkage_strength():
    """alpha is estimated -- sweeping the knob must do nothing, which is the
    whole point of the mode."""
    b = chain_batch(20, graphs=4)
    vals = []
    for strength in (None, 0.5, 100.0):
        cfg = simple_config(
            max_wl_depth=3,
            class_estimator="continuation",
            shrinkage_weight="empirical_bayes",
            shrinkage_strength=strength,
        )
        vals.append(sieve.predict(sieve.fit(b, cfg), b))
    np.testing.assert_array_equal(vals[0], vals[1])
    np.testing.assert_array_equal(vals[0], vals[2])


def testempirical_bayes_weights_are_valid_probabilities():
    b = chain_batch(20, graphs=4)
    cfg = simple_config(
        max_wl_depth=3,
        class_estimator="continuation",
        shrinkage_weight="empirical_bayes",
    )
    from sieve.shrinkage import empirical_bayes_weights

    for w in empirical_bayes_weights(sieve.fit(b, cfg)):
        assert np.all(np.isfinite(w)) and np.all((w >= 0) & (w <= 1))


def test_sibling_variance_is_nan_at_the_deepest_level():
    from sieve.continuation import sibling_variance

    m = sieve.fit(chain_batch(20, graphs=4), simple_config(max_wl_depth=3))
    assert np.isnan(sibling_variance(m)[-1])


def test_diversity_without_a_strength_does_not_crash_and_does_not_shrink():
    """Regression: predict._search and shrunk_means must agree on *whether*
    shrinkage applies. Widening one guard without the other made this config
    -- a non-count weight with the default shrinkage_strength=None -- raise a
    TypeError in predict while shrunk_means quietly skipped shrinkage."""
    b = chain_batch(20, graphs=4)
    shrunk_cfg = simple_config(max_wl_depth=3, shrinkage_weight="diversity")
    plain = simple_config(max_wl_depth=3)
    assert shrunk_cfg.applies_shrinkage is False
    np.testing.assert_array_equal(
        sieve.predict(sieve.fit(b, shrunk_cfg), b),
        sieve.predict(sieve.fit(b, plain), b),
    )


def test_empirical_bayes_is_active_without_a_strength():
    """The mirror of the above: alpha is estimated, so EB must apply even
    though shrinkage_strength is None."""
    cfg = simple_config(
        max_wl_depth=3,
        class_estimator="continuation",
        shrinkage_weight="empirical_bayes",
    )
    assert cfg.applies_shrinkage is True
    b = chain_batch(20, graphs=4)
    plain = simple_config(max_wl_depth=3, class_estimator="continuation")
    assert not np.allclose(
        sieve.predict(sieve.fit(b, cfg), b),
        sieve.predict(sieve.fit(b, plain), b),
    )


def test_empirical_bayes_uses_every_target_dimension():
    """Regression: the tau^2 estimators once read column 0 only, so a vector
    target's shrinkage ignored every dimension but the first -- silently, and
    exactly where it matters most (design.md 9.2's own example is
    target_dim=50)."""
    from sieve.batch import NodeBatch
    from sieve.continuation import sibling_variance

    cfg = simple_config(max_wl_depth=3, target_dim=4)
    b = chain_batch(20, graphs=4, d=4)
    base = sibling_variance(sieve.fit(b, cfg))
    y = b.y.copy()
    y[:, 3] *= 1000.0
    scaled = sibling_variance(sieve.fit(NodeBatch(**{**b.__dict__, "y": y}), cfg))
    assert any(
        (x == x) and (z == z) and not np.isclose(x, z)
        for x, z in zip(base, scaled, strict=True)
    ), "scaling a non-zero target dimension must move tau^2"


def test_empirical_bayes_reports_its_estimated_weight():
    """Under EB the weight varies for a reason other than support, which
    makes it the one mode where the diagnostic is worth carrying."""
    cfg = simple_config(
        max_wl_depth=3,
        class_estimator="continuation",
        shrinkage_weight="empirical_bayes",
    )
    b = chain_batch(20, graphs=4)
    det = sieve.predict_detailed(sieve.fit(b, cfg), b)
    w = det.shrinkage_weight
    assert w is not None and np.all(np.isfinite(w)) and np.all((w >= 0) & (w <= 1))
