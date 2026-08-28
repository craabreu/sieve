"""Shared metrics for the charges experiment harness.

Pure numpy: no rdkit, no pandas, no mlflow -- unit-testable with hand
computed numbers (see charge_experiments/tests/test_charge_metrics.py).

Unlike cosmo_experiments' ``charge_metrics`` (MAE/RMSE/max_abs_residual, no
R2 -- net *molecular* charge clusters near zero, destabilizing ss_tot), this
series' primary target is per-atom ``MBIScharge``, which has real spread, so
R2 is informative and reported; ``max_abs_residual`` is dropped. See the
design spec's Metrics decision.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import molecule_sum


def regression_metrics(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> dict[str, float]:
    """MAE/RMSE/R2 flattened over all elements.

    NaN-aware: a predictor faithful to its own source's real missing-value
    behavior (e.g. a pretrained-tree baseline that returns NaN rather than
    inventing a fallback value for an atom it can't match -- see
    predictors/dash_pretrained.py) must not have one such atom silently
    poison the whole run's aggregate metrics into NaN. Pairs where either
    side is NaN are excluded from the mae/rmse/r2 computation; ``n_nan``
    reports how many were excluded, so that exclusion is never silent
    either. NaN for every key on empty (or all-NaN) input rather than a
    RuntimeWarning-turned-error from averaging zero elements --
    pyproject.toml promotes RuntimeWarning to an error.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = ~(np.isnan(y_true) | np.isnan(y_pred))
    n_nan = float(finite.size - int(finite.sum()))
    if not finite.any():
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "n_nan": n_nan,
        }
    y_true_f = y_true[finite]
    y_pred_f = y_pred[finite]
    err = y_pred_f - y_true_f
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true_f - y_true_f.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_nan": n_nan,
    }


def charge_conservation_metrics(
    atom_charge_pred: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    net_charge_true: NDArray[np.floating],
    n_molecules: int,
) -> dict[str, float]:
    """Secondary diagnostic: how well each conformer's summed predicted atom
    charges reproduce its own molblock ``M CHG`` total (``net_charge`` --
    unlike cosmo_experiments' sigma-derived "charge", there is no sign flip
    here: ``MBIScharge`` is a real atomic partial charge, and a conformer's
    atoms should sum to its own formal charge directly).
    """
    pred_sum = molecule_sum(atom_charge_pred, mol_id, n_molecules)
    return regression_metrics(net_charge_true, pred_sum)
