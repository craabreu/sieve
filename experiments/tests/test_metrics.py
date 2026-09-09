"""Pure-numpy metrics tests -- hand-computed numbers, no rdkit needed."""

from __future__ import annotations

import numpy as np
import pytest


def test_regression_metrics_mae_rmse_r2_hand_computed():
    from experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 5.0])

    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(np.sqrt((0 + 0 + 0 + 4) / 4))
    ss_res = 4.0
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    assert out["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_regression_metrics_perfect_prediction_is_r2_one():
    from experiments.metrics import regression_metrics

    y = np.array([-0.3, 0.1, 0.5, -0.1])
    out = regression_metrics(y, y.copy())
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)


def test_regression_metrics_empty_input_is_nan_not_a_crash():
    """pyproject.toml promotes RuntimeWarning to an error, so a bare
    np.mean(empty) must never be reached."""
    from experiments.metrics import regression_metrics

    out = regression_metrics(np.array([]), np.array([]))
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])


def test_regression_metrics_excludes_nan_pairs_and_reports_the_count():
    """A predictor faithful to its own source's real missing-value behavior
    (e.g. predictors/dash_pretrained.py, which returns NaN rather than
    inventing a fallback for an atom it can't match) must not have that one
    NaN poison the whole run's aggregate metrics."""
    from experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, np.nan, 3.0])

    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)
    assert out["n_nan"] == pytest.approx(1.0)


def test_regression_metrics_all_nan_is_nan_not_a_crash():
    from experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0])
    y_pred = np.array([np.nan, np.nan])

    out = regression_metrics(y_true, y_pred)
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])
    assert out["n_nan"] == pytest.approx(2.0)


def test_regression_metrics_no_nan_reports_zero():
    from experiments.metrics import regression_metrics

    out = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert out["n_nan"] == pytest.approx(0.0)


def test_sum_constraint_metrics_sums_atoms_per_conformer():
    from experiments.metrics import sum_constraint_metrics

    # Two conformers: atoms [0,0,1,1,1] -> conformer 0 has 2 atoms, conformer 1 has 3.
    mol_id = np.array([0, 0, 1, 1, 1])
    atom_value_pred = np.array([0.2, -0.1, 0.05, 0.05, -0.2])
    molecule_value_true = np.array([0.0, 0.0])

    out = sum_constraint_metrics(atom_value_pred, mol_id, molecule_value_true, 2)
    pred_sums = np.array([0.1, -0.1])
    expected_mae = float(np.mean(np.abs(pred_sums - molecule_value_true)))
    assert out["mae"] == pytest.approx(expected_mae)


def test_sum_constraint_metrics_perfect_conservation_is_zero_error():
    from experiments.metrics import sum_constraint_metrics

    mol_id = np.array([0, 0, 1])
    atom_value_pred = np.array([0.5, -0.5, 1.0])
    molecule_value_true = np.array([0.0, 1.0])

    out = sum_constraint_metrics(atom_value_pred, mol_id, molecule_value_true, 2)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)


def test_sum_constraint_metrics_against_hand_computed_numbers():
    from experiments.metrics import sum_constraint_metrics

    # two molecules, 2 atoms each; predicted sums 1.0 and 4.0 vs true 1.5, 3.0
    pred = np.array([0.4, 0.6, 2.0, 2.0])
    mol_id = np.array([0, 0, 1, 1])
    true = np.array([1.5, 3.0])
    out = sum_constraint_metrics(pred, mol_id, true, 2)
    assert out["mae"] == pytest.approx(0.75)
    assert out["rmse"] == pytest.approx(np.sqrt((0.5**2 + 1.0**2) / 2))
