"""Analytic tests for experiments/sieve_experiments/metrics.py.

Pure numpy, no cosmolayer/rdkit -- every case here is a hand-computed number,
not a regression snapshot, so a change in behavior shows up as a wrong number,
not just a diff.
"""

from __future__ import annotations

import numpy as np
import pytest
from sieve_experiments.metrics import (
    charge_metrics,
    molecule_metrics,
    normalize_rows,
    regression_metrics,
    wasserstein1,
    weighted_mean,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BIN_WIDTH = 0.001
N_BINS = 51


def one_hot(index: int, n: int = N_BINS) -> np.ndarray:
    row = np.zeros(n)
    row[index] = 1.0
    return row


# --- wasserstein1 -----------------------------------------------------------


def test_wasserstein1_two_point_masses_exact():
    """W1 between two one-hot rows is exactly the bin distance times width.

    This is the test that pins the metric's units: a missing bin_width
    factor, a transposed axis, or an accidental normalization all break it.
    """
    y_true = np.stack([one_hot(5), one_hot(0)])
    y_pred = np.stack([one_hot(20), one_hot(10)])
    w1 = wasserstein1(y_true, y_pred, bin_width=BIN_WIDTH)
    np.testing.assert_allclose(w1, [15 * BIN_WIDTH, 10 * BIN_WIDTH])


def test_wasserstein1_self_is_zero():
    rng = np.random.default_rng(0)
    y = rng.random((4, N_BINS))
    w1 = wasserstein1(y, y, bin_width=BIN_WIDTH)
    np.testing.assert_allclose(w1, 0.0, atol=1e-12)


def test_wasserstein1_symmetric():
    rng = np.random.default_rng(1)
    a = rng.random((6, N_BINS))
    b = rng.random((6, N_BINS))
    np.testing.assert_allclose(
        wasserstein1(a, b, bin_width=BIN_WIDTH),
        wasserstein1(b, a, bin_width=BIN_WIDTH),
    )


def test_wasserstein1_homogeneous():
    """W1(c*p, c*q) == c * W1(p, q) for the unnormalized form."""
    y_true = np.stack([one_hot(5)])
    y_pred = np.stack([one_hot(20)])
    base = wasserstein1(y_true, y_pred, bin_width=BIN_WIDTH)
    scaled = wasserstein1(3.0 * y_true, 3.0 * y_pred, bin_width=BIN_WIDTH)
    np.testing.assert_allclose(scaled, 3.0 * base)


def test_wasserstein1_shift_on_uniform_ladder():
    """Shifting a 3-point-mass distribution by 3 bins moves each unit mass
    3 bins, so the total cost is 3 masses * 3 bins * bin_width."""
    base = one_hot(10) + one_hot(11) + one_hot(12)
    shifted = one_hot(13) + one_hot(14) + one_hot(15)
    w1 = wasserstein1(base[None, :], shifted[None, :], bin_width=BIN_WIDTH)
    np.testing.assert_allclose(w1, [3 * 3 * BIN_WIDTH])


# --- normalize_rows -----------------------------------------------------


def test_normalize_rows_sums_to_one():
    rng = np.random.default_rng(2)
    p = rng.random((5, N_BINS))
    out = normalize_rows(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_normalize_rows_scale_invariance_of_w1():
    """The defining property of the primary metric: any positive per-row
    scale factor on either side must not move w1(normalize(true), normalize(pred))."""
    rng = np.random.default_rng(3)
    true = rng.random((4, N_BINS)) + 0.1
    pred = rng.random((4, N_BINS)) + 0.1
    base = wasserstein1(normalize_rows(true), normalize_rows(pred), bin_width=BIN_WIDTH)

    c = np.array([1.0, 2.5, 0.1, 10.0])[:, None]
    d = np.array([3.0, 0.5, 7.0, 1.0])[:, None]
    scaled = wasserstein1(
        normalize_rows(c * true), normalize_rows(d * pred), bin_width=BIN_WIDTH
    )
    np.testing.assert_allclose(scaled, base, rtol=1e-10)


def test_normalize_rows_degenerate_row_is_nan_no_warning(recwarn):
    """A degenerate (all-zero, or negative-sum) row must come back as NaN,
    and must NOT raise -- pyproject.toml promotes RuntimeWarning to an error,
    so a naive `p / p.sum(1)` here would fail the whole test suite."""
    p = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = normalize_rows(p)
    np.testing.assert_allclose(out[0], [1 / 6, 2 / 6, 3 / 6])
    assert np.all(np.isnan(out[1]))
    assert len(recwarn) == 0


# --- weighted_mean --------------------------------------------------------


def test_weighted_mean_equal_weights_is_plain_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    assert weighted_mean(values, weights) == pytest.approx(values.mean())


def test_weighted_mean_dominant_weight():
    values = np.array([1.0, 100.0])
    weights = np.array([1.0, 1e6])
    assert weighted_mean(values, weights) == pytest.approx(100.0, rel=1e-4)


# --- regression_metrics ----------------------------------------------------


def test_regression_metrics_matches_sklearn():
    rng = np.random.default_rng(4)
    y_true = rng.random((20, N_BINS))
    y_pred = y_true + rng.normal(scale=0.05, size=(20, N_BINS))
    out = regression_metrics(y_true, y_pred)
    flat_true, flat_pred = y_true.ravel(), y_pred.ravel()
    assert out["mae"] == pytest.approx(mean_absolute_error(flat_true, flat_pred))
    assert out["rmse"] == pytest.approx(mean_squared_error(flat_true, flat_pred) ** 0.5)
    assert out["r2"] == pytest.approx(r2_score(flat_true, flat_pred))


def test_regression_metrics_with_r2_false_omits_key():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    out = regression_metrics(y_true, y_pred, with_r2=False)
    assert "r2" not in out
    assert set(out) == {"mae", "rmse"}


def test_regression_metrics_key_is_ascii_r2_not_unicode():
    """MLflow rejects non-ASCII metric keys; today's scripts use 'R²'."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    out = regression_metrics(y_true, y_pred)
    assert "r2" in out
    assert "R²" not in out


# --- molecule_metrics (the combiner) ---------------------------------------


def test_molecule_metrics_profile_only():
    """A molecule-level predictor (no area/charge) yields no area/charge keys."""
    rng = np.random.default_rng(5)
    true = rng.random((10, N_BINS)) + 0.1
    pred = true + rng.normal(scale=0.01, size=true.shape)
    out = molecule_metrics(profile_true=true, profile_pred=pred, bin_width=BIN_WIDTH)
    assert "profile/w1_norm_mean" in out
    assert "profile/w1_norm_area_weighted" in out
    assert "profile/w1_abs_mean" in out
    assert not any(k.startswith("area/") for k in out)
    assert not any(k.startswith("charge/") for k in out)
    assert out["n_test"] == 10


def test_molecule_metrics_charge_has_no_r2():
    rng = np.random.default_rng(6)
    true = rng.random((5, N_BINS)) + 0.1
    pred = true.copy()
    out = molecule_metrics(
        profile_true=true,
        profile_pred=pred,
        bin_width=BIN_WIDTH,
        charge_true=np.array([0.0, 0.0, 1.0, -1.0, 0.0]),
        charge_pred=np.array([0.01, -0.02, 0.9, -1.1, 0.0]),
    )
    assert "charge/mae" in out
    assert "charge/rmse" in out
    assert "charge/max_abs_residual" in out
    assert "charge/r2" not in out


def test_molecule_metrics_area_has_r2():
    rng = np.random.default_rng(7)
    true = rng.random((5, N_BINS)) + 0.1
    pred = true.copy()
    area_true = true.sum(axis=1)
    area_pred = area_true * 1.01
    out = molecule_metrics(
        profile_true=true,
        profile_pred=pred,
        bin_width=BIN_WIDTH,
        area_true=area_true,
        area_pred=area_pred,
    )
    assert "area/mae" in out
    assert "area/r2" in out


def test_molecule_metrics_excludes_degenerate_rows_from_normalized_w1():
    true = np.stack([one_hot(5), np.zeros(N_BINS)])
    pred = np.stack([one_hot(5), one_hot(10)])
    out = molecule_metrics(profile_true=true, profile_pred=pred, bin_width=BIN_WIDTH)
    assert out["n_degenerate"] == 1
    # only the non-degenerate row (perfect match) contributes -> exactly 0
    assert out["profile/w1_norm_mean"] == pytest.approx(0.0, abs=1e-12)


def test_molecule_metrics_handles_an_empty_test_split_without_warning(recwarn):
    """A tiny --limit run can land zero molecules in the eval split (e.g. the
    real chaos-store's biased_split with --limit 50: every one of the first
    50 rows falls in train). ``profile_true``/``profile_pred`` are then
    0-row arrays -- np.mean of an empty slice must not raise (pyproject.toml
    promotes RuntimeWarning to an error), and n_test must read 0."""
    empty = np.zeros((0, N_BINS))
    out = molecule_metrics(profile_true=empty, profile_pred=empty, bin_width=BIN_WIDTH)
    assert not recwarn.list
    assert out["n_test"] == 0
    assert np.isnan(out["profile/w1_abs_mean"])


def test_charge_metrics_standalone_has_no_r2():
    out = charge_metrics(np.array([0.0, 1.0]), np.array([0.1, 0.9]))
    assert "r2" not in out
    assert "max_abs_residual" in out


def test_charge_metrics_empty_input_is_nan_not_an_error():
    out = charge_metrics(np.zeros(0), np.zeros(0))
    assert np.isnan(out["max_abs_residual"])
    assert np.isnan(out["mae"])
