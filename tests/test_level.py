import numpy as np
import pytest
from wllr.level import FrozenLevel, fit_level, global_stats
from wllr.refine import LevelLabels

def lv(labels, nc=None):
    labels = np.asarray(labels, np.int64)
    nc = nc or int(labels.max()) + 1
    return LevelLabels(labels, np.zeros((nc, 1), np.int64), np.full(nc, -1, np.int32))

def test_mean_and_population_variance():
    y = np.array([[1.0], [3.0], [10.0]])
    f = fit_level(lv([0, 0, 1]), y)
    assert f.count.tolist() == [2, 1]
    np.testing.assert_allclose(f.mean[:, 0], [2.0, 10.0])
    np.testing.assert_allclose(f.msd[:, 0], [1.0, 0.0])   # divisor N, not N-1

def test_variance_accessor_applies_bessel_and_is_nan_at_one():
    f = fit_level(lv([0, 0, 1]), np.array([[1.0], [3.0], [10.0]]))
    v = f.variance
    np.testing.assert_allclose(v[0, 0], 2.0)              # 2/1 * 1.0
    assert np.isnan(v[1, 0]), "N=1 must be NaN, never 0.0"

def test_vector_targets_are_not_a_special_case():
    y = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 0.0]])
    f = fit_level(lv([0, 0, 0]), y)
    np.testing.assert_allclose(f.mean[0], [3.0, 10.0])
    np.testing.assert_allclose(f.msd[0], [np.var([1, 3, 5]), np.var([10, 20, 0])])

def test_centring_survives_a_large_offset():
    """design.md 7.4: power sums give negative variances here; centring does not."""
    rng = np.random.default_rng(0)
    y = (1e6 + rng.normal(0, 3, size=(2000, 1)))
    f = fit_level(lv(np.zeros(2000, np.int64)), y)
    exact = y.var()
    assert abs(f.msd[0, 0] - exact) / exact < 1e-6
    assert f.msd[0, 0] > 0

def test_empty_classes_are_zero_not_carried_forward():
    """design.md 7.4: reduceat returns y[start] for empty groups; P must not."""
    y = np.array([[10.0], [20.0], [30.0], [40.0]])
    f = fit_level(lv([0, 0, 3, 3], nc=4), y)
    assert f.count.tolist() == [2, 0, 0, 2]
    np.testing.assert_allclose(f.mean[:, 0], [15.0, 0.0, 0.0, 35.0])

def test_global_stats_matches_numpy():
    y = np.array([[1.0, 2.0], [3.0, 6.0]])
    n, mean, msd = global_stats(y)
    assert n == 2
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(msd, [1.0, 4.0])
