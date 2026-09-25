"""Study F: calibration of Sieve's predictive variance.

Spec: docs/superpowers/specs/2026-09-25-study-f-calibration-design.md.

The shipped predictive variance (form B, within-structure-variance spec) and
four ablations, each dropping one ingredient, are scored on the Study B
incumbent's runs: calibration (NLL, E[z^2], coverage, E[z^2] by matched
radius, within-conformer ranking) and the variance-weighted normalisation they
drive. Every arm is a per-atom variance from one prediction, so all five are
scored on the same atoms. Scores go to a sidecar beside each run's own
metrics, like Study D's subset scores (``stereo_subsets``), so a run's record
is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

METRICS_FILE = "calibration_metrics.json"
PREFIX = "calibration"
COVERAGE = (0.5, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class Arm:
    """One predictive variance: which terms, at which constants."""

    name: str
    alpha_v: float
    selection_weight: float
    estimation: bool
    within_structure: bool


ARMS = (
    Arm("form_b", 10.0, 0.5, True, True),
    Arm("no_sigma2_w", 10.0, 0.5, True, False),
    Arm("alpha_v_30", 30.0, 0.5, True, True),
    Arm("no_selection", 10.0, 0.0, True, True),
    Arm("no_estimation", 10.0, 0.5, False, True),
)


def arm_variances(model: Any, prediction: Any) -> dict[str, np.ndarray]:
    """Per-atom sigma^2 of every arm, ``(n,)`` each, for a ``d == 1`` model.

    ``prediction.matched_level`` is a backoff-path position, while the
    variance tables are indexed by raw level, so positions are mapped through
    ``config.backoff_path`` first. Unmatched atoms take ``global_msd``, plus
    sigma2_w in the arms that carry it -- ``predict``'s own fallback.
    """
    from sieve.uncertainty import variance_terms

    backoff = np.asarray(model.config.backoff_path, dtype=np.int64)
    pos = np.asarray(prediction.matched_level)
    level = np.where(pos >= 0, backoff[np.maximum(pos, 0)], -1)
    cid = np.asarray(prediction.class_id)
    s2w = float(model.within_variance[0])
    gmsd = float(model.global_msd[0])
    terms: dict[float, list[dict[str, np.ndarray]]] = {}
    out: dict[str, np.ndarray] = {}
    for arm in ARMS:
        if arm.alpha_v not in terms:
            terms[arm.alpha_v] = variance_terms(model, alpha_v=arm.alpha_v)
        s2 = np.full(pos.shape[0], gmsd + (s2w if arm.within_structure else 0.0))
        for k in np.unique(level[level >= 0]):
            t = terms[arm.alpha_v][int(k)]
            # the same order as predictive_variance's sum, so form_b is
            # bit-identical to it
            table = t["within"] + arm.selection_weight * t["selection"]
            if arm.estimation:
                table = table + t["estimation"]
            if arm.within_structure:
                table = table + t["within_structure"]
            sel = level == k
            s2[sel] = table[cid[sel], 0]
        out[arm.name] = s2
    return out


def group_percentile_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each value's average rank within its group, divided by the group's size
    (pandas' ``groupby().rank(pct=True)``, without pandas)."""
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    n = values.shape[0]
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    size = np.diff(np.r_[start, n])
    pos = np.arange(n) - np.repeat(start, size)
    new_run = np.r_[True, (np.diff(g) != 0) | (np.diff(v) != 0)]
    run_id = np.cumsum(new_run) - 1
    first = np.flatnonzero(new_run)
    run_len = np.diff(np.r_[first, n])
    avg_rank = pos[first] + (run_len - 1) / 2.0 + 1.0
    out = np.empty(n)
    out[order] = avg_rank[run_id] / np.repeat(size, size)
    return out


def normalised_errors(
    normalised: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    d = np.asarray(normalised) - np.asarray(target)
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def calibration_metrics(
    *,
    raw: np.ndarray,
    target: np.ndarray,
    sigma2: np.ndarray,
    k_star: np.ndarray,
    conf: np.ndarray,
    molecule_value: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """One arm's calibration and normalisation scores (spec section 2.3).

    ``conf`` is each atom's conformer index, ``0..n_conformers-1``, and
    ``molecule_value`` each conformer's constrained total.
    """
    from scipy.stats import norm

    from experiments.normalize import variance_weighted_normalize

    e = np.asarray(raw) - np.asarray(target)
    z2 = e * e / sigma2
    out = {
        "nll": float(np.mean(0.5 * (np.log(2 * np.pi * sigma2) + z2))),
        "ez2": float(np.mean(z2)),
    }
    for c in COVERAGE:
        out[f"cov{round(100 * c)}"] = float(np.mean(z2 < norm.ppf(0.5 + c / 2) ** 2))
    for k in range(depth + 1):
        sel = k_star == k
        if sel.any():
            out[f"ez2_k{k}"] = float(np.mean(z2[sel]))
    ra = group_percentile_rank(np.abs(e), conf)
    rs = group_percentile_rank(sigma2, conf)
    out["rho"] = float(np.corrcoef(ra, rs)[0, 1])
    n_conf = int(np.asarray(molecule_value).shape[0])
    x = variance_weighted_normalize(raw, np.sqrt(sigma2), molecule_value, conf, n_conf)
    out["norm_rmse"], out["norm_mae"] = normalised_errors(x, target)
    return out
