"""Shared metrics for the experiment harness.

Pure numpy: no cosmolayer, no rdkit, no mlflow. That is what makes this module
unit-testable with hand-computed numbers
(see experiments/tests/test_experiment_metrics.py) independent of any store
or predictor.

The Wasserstein-1 and regression-metric formulas are promoted, unchanged, from
scripts/train_chaos_sigma_profile.py (wasserstein1_per_row, regression_metrics),
which were themselves adapted from the user's own ProfileWasserstein /
AreaWeightedProfileWasserstein (~/data/cosmo/sigma-prediction/sigma_nn.py,
a ``ChempropMetric`` subclass defined there -- not part of Chemprop itself,
despite the name). Three things are fixed here, deliberately, versus that
script:

- Metric keys are ASCII ("r2", not "R²") because MLflow rejects non-ASCII keys.
- ``normalize_rows`` masks a zero/negative row sum explicitly rather than
  dividing and suppressing the warning: pyproject.toml promotes
  ``RuntimeWarning`` to an error, so a bare ``p / p.sum(1, keepdims=True)``
  would fail the test suite on any degenerate row.
- ``molecule_metrics`` never reports W1/MAE/RMSE/R² on *unnormalized*
  profiles (the old script's "w1_abs_mean"/"profile/mae" etc.) -- that
  conflates shape error with total-area error into one number, and was
  only ever kept for continuity with the old script's numbers, never as
  the metric to actually reason about a predictor by. The normalized W1
  (``profile/w1_norm_*``) is the only profile-shape metric reported;
  ``area/*`` already covers total-area error on its own.
- ``wasserstein1`` no longer scales by a grid ``bin_width``. The scaled
  version turns the CDF-gap sum into a distance in sigma's own physical
  unit (e/A^2); dropped since this project's sigma grid is fixed everywhere
  it matters (chaos-store, DASH, and -- per design.md -- COSMO-NET all
  share it), so that scaling was only ever a constant rescale of every
  number, never something that changed a relative comparison between
  predictors or runs. What's reported now is W1 in units of bin-width
  ("how many bins apart"), not an absolute sigma-space distance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wasserstein1(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Per-row Wasserstein-1 distance between two 1D-histogram rows, in
    units of bin-width (see module docstring for why there's no bin_width
    scaling here).

    Both inputs are read directly off their CDFs: this is the standard
    1D-histogram W1 formula, valid whether the rows are normalized (sum to 1)
    or unnormalized (sum to some other total, e.g. an area).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    cdf_gap = np.cumsum(y_true, axis=1) - np.cumsum(y_pred, axis=1)
    return np.abs(cdf_gap).sum(1)


def normalize_rows(p: NDArray[np.floating]) -> NDArray[np.float64]:
    """Divide each row by its own sum; rows summing to <= 0 become NaN.

    Never divides by zero without masking first -- see module docstring.
    """
    p = np.asarray(p, dtype=np.float64)
    totals = p.sum(axis=1)
    out = np.full_like(p, np.nan)
    good = totals > 0
    out[good] = p[good] / totals[good, None]
    return out


def weighted_mean(values: NDArray[np.floating], weights: NDArray[np.floating]) -> float:
    """True weighted average (sum(w*x)/sum(w)), NaN if all weights are <= 0."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def regression_metrics(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating], *, with_r2: bool = True
) -> dict[str, float]:
    """MAE/RMSE (and, unless with_r2=False, R²) flattened over all elements.

    NaN for every key on empty input (e.g. a --limit run whose eval split
    happens to be empty) rather than a RuntimeWarning-turned-error from
    averaging zero elements -- pyproject.toml promotes RuntimeWarning to an
    error, so a bare np.mean(empty) would fail the run, not just look odd.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        out = {"mae": float("nan"), "rmse": float("nan")}
        if with_r2:
            out["r2"] = float("nan")
        return out
    err = y_pred - y_true
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }
    if with_r2:
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def charge_metrics(
    charge_true: NDArray[np.floating], charge_pred: NDArray[np.floating]
) -> dict[str, float]:
    """MAE/RMSE/max-abs-residual on total molecule charge. Deliberately no R²:
    net charge is ~0 for nearly every molecule in the store, so ss_tot is
    near-zero and R² is either NaN or amplified noise, not a useful number.
    """
    out = regression_metrics(charge_true, charge_pred, with_r2=False)
    residual = np.abs(np.asarray(charge_pred) - np.asarray(charge_true))
    out["max_abs_residual"] = float(np.max(residual)) if residual.size else float("nan")
    return out


def _prefixed(prefix: str, d: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def molecule_metrics(
    *,
    profile_true: NDArray[np.floating],
    profile_pred: NDArray[np.floating],
    area_true: NDArray[np.floating] | None = None,
    area_pred: NDArray[np.floating] | None = None,
    charge_true: NDArray[np.floating] | None = None,
    charge_pred: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """The one metrics dict a run reports, at molecule level.

    ``profile_true``/``profile_pred`` are required and unnormalized (bins sum
    to the molecule's own area, per design.md 11.4). ``area_*``/``charge_*``
    are optional -- pass them only when the predictor actually supplies that
    quantity, and the corresponding metric block is simply absent otherwise
    (e.g. a molecule-level predictor with no per-atom charge output).
    """
    profile_true = np.asarray(profile_true, dtype=np.float64)
    profile_pred = np.asarray(profile_pred, dtype=np.float64)

    norm_true = normalize_rows(profile_true)
    norm_pred = normalize_rows(profile_pred)
    degenerate = np.isnan(norm_true).any(axis=1) | np.isnan(norm_pred).any(axis=1)
    keep = ~degenerate

    w1_norm = wasserstein1(norm_true[keep], norm_pred[keep])
    true_area = profile_true.sum(axis=1)

    out: dict[str, float] = {
        "n_test": float(len(profile_true)),
        "n_degenerate": float(degenerate.sum()),
        "profile/w1_norm_mean": (
            float(np.mean(w1_norm)) if len(w1_norm) else float("nan")
        ),
        "profile/w1_norm_median": (
            float(np.median(w1_norm)) if len(w1_norm) else float("nan")
        ),
        "profile/w1_norm_p90": (
            float(np.percentile(w1_norm, 90)) if len(w1_norm) else float("nan")
        ),
        "profile/w1_norm_area_weighted": weighted_mean(w1_norm, true_area[keep]),
    }

    if area_true is not None and area_pred is not None:
        out.update(_prefixed("area", regression_metrics(area_true, area_pred)))

    if charge_true is not None and charge_pred is not None:
        out.update(_prefixed("charge", charge_metrics(charge_true, charge_pred)))

    return out
