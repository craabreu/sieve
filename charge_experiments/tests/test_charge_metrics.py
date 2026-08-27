"""Pure-numpy metrics tests -- hand-computed numbers, no rdkit needed."""

from __future__ import annotations

import numpy as np
import pytest


def test_regression_metrics_mae_rmse_r2_hand_computed():
    from charge_experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 5.0])

    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(np.sqrt((0 + 0 + 0 + 4) / 4))
    ss_res = 4.0
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    assert out["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_regression_metrics_perfect_prediction_is_r2_one():
    from charge_experiments.metrics import regression_metrics

    y = np.array([-0.3, 0.1, 0.5, -0.1])
    out = regression_metrics(y, y.copy())
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)


def test_regression_metrics_empty_input_is_nan_not_a_crash():
    """pyproject.toml promotes RuntimeWarning to an error, so a bare
    np.mean(empty) must never be reached."""
    from charge_experiments.metrics import regression_metrics

    out = regression_metrics(np.array([]), np.array([]))
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])


def test_regression_metrics_excludes_nan_pairs_and_reports_the_count():
    """A predictor faithful to its own source's real missing-value behavior
    (e.g. predictors/dash_pretrained.py, which returns NaN rather than
    inventing a fallback for an atom it can't match) must not have that one
    NaN poison the whole run's aggregate metrics."""
    from charge_experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, np.nan, 3.0])

    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["n_nan"] == pytest.approx(1.0)


def test_regression_metrics_all_nan_is_nan_not_a_crash():
    from charge_experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0])
    y_pred = np.array([np.nan, np.nan])

    out = regression_metrics(y_true, y_pred)
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])
    assert out["n_nan"] == pytest.approx(2.0)


def test_regression_metrics_no_nan_reports_zero():
    from charge_experiments.metrics import regression_metrics

    out = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert out["n_nan"] == pytest.approx(0.0)


def test_charge_conservation_metrics_sums_atoms_per_conformer():
    from charge_experiments.metrics import charge_conservation_metrics

    # Two conformers: atoms [0,0,1,1,1] -> conformer 0 has 2 atoms, conformer 1 has 3.
    mol_id = np.array([0, 0, 1, 1, 1])
    atom_charge_pred = np.array([0.2, -0.1, 0.05, 0.05, -0.2])
    net_charge_true = np.array([0.0, 0.0])

    out = charge_conservation_metrics(atom_charge_pred, mol_id, net_charge_true, 2)
    pred_sums = np.array([0.1, -0.1])
    expected_mae = float(np.mean(np.abs(pred_sums - net_charge_true)))
    assert out["mae"] == pytest.approx(expected_mae)


def test_charge_conservation_metrics_perfect_conservation_is_zero_error():
    from charge_experiments.metrics import charge_conservation_metrics

    mol_id = np.array([0, 0, 1])
    atom_charge_pred = np.array([0.5, -0.5, 1.0])
    net_charge_true = np.array([0.0, 1.0])

    out = charge_conservation_metrics(atom_charge_pred, mol_id, net_charge_true, 2)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
