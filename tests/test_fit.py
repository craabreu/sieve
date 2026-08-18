import numpy as np
import pytest

import sieve
from tests.helpers import chain_batch, simple_config


def test_fit_produces_one_level_per_config_level():
    b = chain_batch(6)
    m = sieve.fit(b, simple_config())
    assert len(m.levels) == simple_config().n_levels


def test_support_is_monotone_non_increasing_along_the_chain():
    """design.md 2.3: refinement can only split a class."""
    m = sieve.fit(chain_batch(30), simple_config(max_wl_depth=3))
    for k in range(1, len(m.levels)):
        child, par = m.levels[k], m.levels[k - 1]
        assert np.all(child.count <= par.count[child.parent])


def test_class_means_match_a_direct_groupby():
    b = chain_batch(20)
    m = sieve.fit(b, simple_config())
    from sieve.refine import refine

    labels = refine(b, simple_config())[-1].labels
    for c in np.unique(labels):
        np.testing.assert_allclose(m.levels[-1].mean[c], b.y[labels == c].mean(axis=0))


def test_changing_targets_leaves_class_ids_unchanged():
    """design.md 10.4: the partition depends on structure only."""
    from sieve.refine import refine

    b = chain_batch(15)
    other = type(b)(**{**b.__dict__, "y": b.y * 3.0 + 7.0})
    a = refine(b, simple_config())[-1].labels
    c = refine(other, simple_config())[-1].labels
    assert np.array_equal(a, c)


def test_empty_model_has_no_classes():
    m = sieve.SieveModel.empty(simple_config())
    assert all(lvl.n_classes == 0 for lvl in m.levels)
    assert m.global_count == 0


def test_with_params_shares_arrays():
    m = sieve.fit(chain_batch(10), simple_config())
    m2 = m.with_params(n_min=5)
    assert m2.config.n_min == 5
    assert m2.levels[0].mean is m.levels[0].mean  # no copy


def test_fit_rejects_a_batch_without_targets():
    b = chain_batch(5)
    with pytest.raises(ValueError, match="targets"):
        sieve.fit(type(b)(**{**b.__dict__, "y": None}), simple_config())


def test_target_dim_must_match_config():
    b = chain_batch(5)
    with pytest.raises(ValueError, match="target_dim"):
        sieve.fit(b, simple_config(target_dim=7))
