import numpy as np

import sieve
from tests.helpers import chain_batch, simple_config


def test_class_of_two_returns_the_other_member():
    """design.md 10.4: the sharpest statement of what LOO means."""
    cfg = simple_config(max_wl_depth=1)
    b = chain_batch(10, graphs=2, seed=5)
    m = sieve.fit(b, cfg)
    p = sieve.predict_loo(m, b)
    from sieve.refine import refine

    labels = refine(b, cfg)[-1].labels
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if members.size == 2 and np.all(p.matched_level[members] == cfg.n_levels - 1):
            i, j = members
            np.testing.assert_allclose(p.value[i], b.y[j])
            np.testing.assert_allclose(p.value[j], b.y[i])


def test_singleton_classes_back_off_instead_of_dividing_by_zero():
    """design.md 10.3: a class of exactly one member must never be used
    directly for a LOO estimate (that divides by zero); a class of *two*
    is fine post-LOO (support 1 there is a legitimate other-member estimate,
    which this fixture's symmetric path graph genuinely produces at its
    deepest level -- so the assertion checks the class actually *used* for
    each matched atom, not the raw support count)."""
    cfg = simple_config(max_wl_depth=4)
    b = chain_batch(25, graphs=1)
    m = sieve.fit(b, cfg)
    p = sieve.predict_loo(m, b)
    assert np.all(np.isfinite(p.value))
    for k in range(cfg.n_levels):
        at_k = p.matched_level == k
        if at_k.any():
            assert np.all(m.levels[k].count[p.class_id[at_k]] != 1)


def test_loo_is_strictly_worse_than_in_sample():
    """The point of the method: in-sample scores are meaningless at n_min=1."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(30, graphs=4, seed=2)
    m = sieve.fit(b, cfg)
    ins = np.mean((sieve.predict(m, b) - b.y) ** 2)
    loo = np.mean((sieve.predict_loo(m, b).value - b.y) ** 2)
    assert loo > ins


def test_loo_is_strictly_worse_than_in_sample_under_shrinkage():
    """design.md 10.3: LOO exists to catch leakage. With alpha set, shrinkage
    must not reintroduce the held-out node's own label by shrinking the
    model's raw (non-LOO) class mean instead of the LOO-adjusted one."""
    cfg = simple_config(max_wl_depth=3, alpha=1.0)
    b = chain_batch(30, graphs=4, seed=2)
    m = sieve.fit(b, cfg)
    ins = np.mean((sieve.predict(m, b) - b.y) ** 2)
    loo = sieve.predict_loo(m, b)
    assert np.mean((loo.value - b.y) ** 2) > ins
    # raw_value is always the LOO estimate; under LOO, value (shrunk) and
    # raw_value must agree on which N was actually used to weight it.
    assert loo.shrinkage_weight is not None
    matched = loo.matched_level >= 0
    n = loo.support[matched].astype(float)
    np.testing.assert_allclose(loo.shrinkage_weight[matched], n / (n + cfg.alpha))


def test_loo_requires_targets():
    import pytest

    b = chain_batch(6)
    m = sieve.fit(b, simple_config())
    with pytest.raises(ValueError, match="targets"):
        sieve.predict_loo(m, type(b)(**{**b.__dict__, "y": None}))
