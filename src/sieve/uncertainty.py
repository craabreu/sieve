r"""Predictive variance for a new node, derived on demand (design-update-v2.md 3).

What ``predict`` reports as ``variance`` is the matched class's own stored
$s^2$ -- a diagnostic, and not usable as an uncertainty: it is ``nan`` at
$N = 1$ and can be a sampling-artefact zero when a class's members happen to
agree, which design.md 6.4 would then read as a delta-function likelihood and
*pin* the node. This module supplies the quantity 6.4 actually needs.

Four terms, each with a distinct reason to exist:

.. math::

    \sigma^2_{\mathrm{pred}} =
      \underbrace{\frac{(N_c-1)s^2_c + \alpha^v \bar\sigma^2_k}
                       {(N_c-1) + \alpha^v}}_{\text{within-class, shrunk}}
    + a \underbrace{\frac{(C_c-1)\tau^2_c + \alpha^t \hat\tau^2_k}
                         {(C_c-1) + \alpha^t}}_{\text{selection}}
    + \underbrace{\hat\tau^2_{\mathrm{pa}(k)}(1 - w_c)}_{\text{mean estimation}}
    + \underbrace{\sigma^2_w}_{\text{within structure}}

**within-class** is the class's own spread, pooled toward its level's average
with degrees of freedom $N-1$. Raw $s^2$ is unusable, not merely noisy:
measured $E[z^2] = 3.5\times10^{25}$, driven by near-zero classes. Pooling
also makes the whole expression total -- no ``nan``, no zero -- with no floor
constant, which is what fixes the defect above.

**selection** is the correction for *which* population reads this class. By
the prefix property (design.md 2.2) a non-deepest class is consulted only by
a query whose own child class was absent, so it is read by exactly the
sub-population it under-represents. The spread among its children measures
that. This term is deliberately **not** weighted by support: it is the one
error component that does not vanish as $N \to \infty$, and a control that
multiplied it by $(1-w)$ so that it did vanish scored identically to having
no term at all. Zero at the deepest level, which is read on an exact match
rather than by backoff and so has no selection to correct.

**mean estimation** is the posterior variance of the class mean under the
same normal-normal model ``shrinkage.empirical_bayes_weights`` already fits:
$\tau^2(1-w) = \sigma^2/(N+\alpha)$. Note this is *smaller* than the naive
$\sigma^2/N$ -- shrinkage makes the mean more certain, not less -- so it is
not, and never was, the fix for low-support miscalibration.

**within structure** is the scatter collapse removed from the fit: the
expected squared deviation of one conformer's charge from its structure's
mean over conformers and symmetry-equivalent atoms
(``SieveModel.within_variance``). Training classes hold one row per structure,
so none of the three terms above can see it, while every held-out atom is an
individual conformer and carries it. Without it the variance was five times
overconfident at the deepest radius (within-structure-variance spec 1). Zero
for a model that carries no within-structure statistics. Per class, when the
fit carried per-class sums, it is
$\sigma^2_{w,c} = (\mathrm{sse}_c + \beta \sigma^2_w)/(n_c + \beta)$, shrunk
toward the pooled value (``WITHIN_SHRINKAGE``); $\beta = \infty$ is the pooled
value in every class.

Derived on demand and never stored, like ``shrinkage.shrunk_means`` and
``continuation.class_means``, and for the same reason: every value depends on
its full ancestor chain, so one added node would invalidate essentially all of
them.
"""

from __future__ import annotations

import numpy as np

from sieve.config import KIND_AWARE
from sieve.continuation import (
    atom_variance,
    aware_variance,
    child_counts,
    class_sibling_variance,
    root_variance,
    sibling_variance,
)
from sieve.level import class_kinds
from sieve.shrinkage import empirical_bayes_weights

# ALPHA_T and SELECTION_WEIGHT were selected on the val split of
# dash-molecules-10fold-1 by Gaussian NLL over a 200-point grid. ALPHA_V was
# re-selected on 2026-09-25, with the within-structure term in place, by
# rotating over the five folds of the Study B incumbent's repeat 0 (chosen on
# four, scored on the fifth): 10 in every fold, by both NLL and normalised
# RMSE, against the earlier 30 (within-structure-variance spec 1).
#
# Deliberately module constants rather than SieveConfig fields. The objective
# is flat -- the gain of 10 over 30 is 0.006 in NLL -- so exposing them as
# knobs would invite tuning that the measurement says cannot pay. They are
# named, not magic, and a caller who genuinely needs to sweep can pass them to
# `predictive_variance`.
ALPHA_V = 10.0  # within-class shrinkage toward the level-pooled variance
ALPHA_T = 1.0  # per-class sibling variance shrinkage toward the pooled one
SELECTION_WEIGHT = 0.5  # coefficient `a` on the selection term
# beta: per-class within-structure variance shrinkage toward the pooled one
# (within-structure-variance spec 4). Infinite -- the pooled sigma2_w in every
# class, phase 1 exactly -- until the out-of-fold experiment decides.
WITHIN_SHRINKAGE = np.inf


def _within_structure(lvl, pooled: np.ndarray, beta: float) -> np.ndarray:
    """sigma2_w per class: (sse_c + beta*pooled) / (n_c + beta), and the pooled
    value itself where the class carries no sums or beta is infinite -- the
    limit, taken explicitly because inf/inf is nan."""
    if lvl.within_sse is None or lvl.within_n is None or np.isinf(beta):
        return pooled
    n = lvl.within_n[:, None]
    den = n + beta
    with np.errstate(invalid="ignore", divide="ignore"):
        blend = (lvl.within_sse + beta * pooled) / den
    return np.where(den > 0, blend, pooled)


def predictive_variance(
    model,
    *,
    alpha_v: float = ALPHA_V,
    alpha_t: float = ALPHA_T,
    selection_weight: float = SELECTION_WEIGHT,
    within_shrinkage: float = WITHIN_SHRINKAGE,
) -> list[np.ndarray]:
    """Per-level ``(n_classes, d)`` predictive variance for a *new* node drawn
    from each class.

    Total by construction: every entry is finite and strictly positive
    whenever the level-pooled ``atom_variance`` is, which holds for any level
    with at least one labeled node. Callers still need their own fallback for
    a query that matched *nothing* -- ``predict`` uses ``global_msd`` there,
    since no class means no row to index.

    Broadcast across the ``d`` target dimensions from scalars: the level-pooled
    and sibling variances are summed over dimensions (they exist to feed a
    scalar weight), while ``msd`` is per-dimension. At ``d == 1`` this is
    exact. At ``d > 1`` the sum makes the two scale terms whole-target
    quantities rather than per-component ones, which is a stated approximation
    -- design-update-v2.md 3 records that a multi-dimensional target wants the
    per-dimension variants of both, which is the same computation without the
    sum over ``j``.
    """
    if not within_shrinkage >= 0:  # also refuses nan
        raise ValueError(
            f"within_shrinkage must be >= 0 (inf for the pooled value), got "
            f"{within_shrinkage}"
        )
    cfg = model.config
    parents = cfg.level_parents
    av = np.asarray(atom_variance(model), dtype=np.float64)
    tau_pooled = np.asarray(sibling_variance(model), dtype=np.float64)
    tau_class = class_sibling_variance(model)
    counts = child_counts(model)
    weights = empirical_bayes_weights(model)
    root = root_variance(model)
    tau_aware = aware_variance(model)

    sigma2_w = model.within_variance

    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        n = lvl.count[:, None].astype(np.float64)
        dof = np.maximum(n - 1.0, 0.0)
        level_var = av[k] if np.isfinite(av[k]) else 0.0

        # within-class: (N-1)s^2 = N*msd identically, so the stored msd is
        # what goes in -- exact at N == 1 (contributing nothing) where s^2 is
        # nan. design.md 4.3 makes the same substitution for the same reason.
        within = (n * lvl.msd + alpha_v * level_var) / (dof + alpha_v)

        # selection: zero where the class has no children, both because there
        # is nothing to measure and because such a class is read on an exact
        # match rather than by backoff.
        c = counts[k][:, None]
        dC = np.maximum(c - 1.0, 0.0)
        pooled_k = tau_pooled[k] if np.isfinite(tau_pooled[k]) else 0.0
        own = np.where(np.isfinite(tau_class[k]), tau_class[k], 0.0)[:, None]
        selection = np.where(
            c > 0, (dC * own + alpha_t * pooled_k) / (dC + alpha_t), 0.0
        )

        # mean estimation: the EB posterior variance of the class mean.
        p = parents[k]
        tau_parent = root if p < 0 else tau_pooled[p]
        if not np.isfinite(tau_parent):
            tau_parent = 0.0
        estimation = tau_parent * (1.0 - weights[k][:, None])
        # An aware-only class is shrunk toward its blind counterpart, so the
        # prior spread of its mean is the aware one (stereo-refines-blind
        # spec, section 5), not its parent's.
        only = class_kinds(lvl) == KIND_AWARE
        if only.any():
            t = tau_aware[k] if np.isfinite(tau_aware[k]) else 0.0
            estimation[only] = t * (1.0 - weights[k][only][:, None])

        out.append(
            within
            + selection_weight * selection
            + estimation
            + _within_structure(lvl, sigma2_w, within_shrinkage)
        )
    return out
