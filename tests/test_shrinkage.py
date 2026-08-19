import numpy as np

import sieve
from sieve.shrinkage import shrunk_means
from tests.helpers import chain_batch, simple_config


def test_shrinkage_strength_zero_reproduces_raw_means():
    """design.md 10.4."""
    cfg = simple_config(shrinkage_strength=0.0)
    b = chain_batch(15, graphs=3)
    m = sieve.fit(b, cfg)
    for lvl, sh in zip(m.levels, shrunk_means(m), strict=True):
        np.testing.assert_allclose(sh[lvl.count > 0], lvl.mean[lvl.count > 0])


def test_large_shrinkage_strength_approaches_the_global_mean():
    cfg = simple_config(shrinkage_strength=1e12)
    b = chain_batch(15, graphs=3)
    m = sieve.fit(b, cfg)
    for sh in shrunk_means(m):
        np.testing.assert_allclose(
            sh, np.broadcast_to(m.global_mean, sh.shape), rtol=1e-4
        )


def test_shrinkage_uses_the_shrunk_parent_not_the_raw_parent():
    cfg = simple_config(shrinkage_strength=3.0, max_wl_depth=2)
    b = chain_batch(20, graphs=3)
    m = sieve.fit(b, cfg)
    sh = shrunk_means(m)
    a = cfg.shrinkage_strength
    for k in range(1, len(m.levels)):
        lvl = m.levels[k]
        n = lvl.count[:, None].astype(float)
        expect = (n * lvl.mean + a * sh[k - 1][lvl.parent]) / (n + a)
        np.testing.assert_allclose(sh[k], expect, rtol=1e-12)


def test_level_zero_shrinks_toward_the_global_mean():
    cfg = simple_config(shrinkage_strength=2.0)
    m = sieve.fit(chain_batch(12), cfg)
    lvl, sh = m.levels[0], shrunk_means(m)[0]
    n = lvl.count[:, None].astype(float)
    expect = (n * lvl.mean + 2.0 * m.global_mean) / (n + 2.0)
    np.testing.assert_allclose(sh, expect)


def test_predict_exposes_raw_value_and_weight_when_shrinkage_strength_is_set():
    cfg = simple_config(shrinkage_strength=2.0)
    b = chain_batch(12)
    p = sieve.predict_detailed(sieve.fit(b, cfg), b)
    assert p.raw_value is not None and p.shrinkage_weight is not None
    matched = p.matched_level >= 0
    n = p.support[matched].astype(float)
    np.testing.assert_allclose(p.shrinkage_weight[matched], n / (n + 2.0))


def test_predict_omits_shrinkage_fields_when_shrinkage_strength_is_none():
    b = chain_batch(12)
    p = sieve.predict_detailed(sieve.fit(b, simple_config()), b)
    assert p.raw_value is None and p.shrinkage_weight is None
