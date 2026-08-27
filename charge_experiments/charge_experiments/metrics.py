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

    NaN for every key on empty input rather than a RuntimeWarning-turned-
    error from averaging zero elements -- pyproject.toml promotes
    RuntimeWarning to an error.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
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
