r"""Predictive variance for a new node, derived on demand (design-update-v2.md 3).

What ``predict`` reports as ``variance`` is the matched class's own stored
$s^2$ -- a diagnostic, and not usable as an uncertainty: it is ``nan`` at
$N = 1$ and can be a sampling-artefact zero when a class's members happen to
agree, which design.md 6.4 would then read as a delta-function likelihood and
*pin* the node. This module supplies the quantity 6.4 actually needs.

Three terms, each with a distinct reason to exist:

.. math::

    \sigma^2_{\mathrm{pred}} =
      \underbrace{\frac{(N_c-1)s^2_c + \alpha^v \bar\sigma^2_k}
                       {(N_c-1) + \alpha^v}}_{\text{within-class, shrunk}}
    + a \underbrace{\frac{(C_c-1)\tau^2_c + \alpha^t \hat\tau^2_k}
                         {(C_c-1) + \alpha^t}}_{\text{selection}}
    + \underbrace{\hat\tau^2_{\mathrm{pa}(k)}(1 - w_c)}_{\text{mean estimation}}

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

Derived on demand and never stored, like ``shrinkage.shrunk_means`` and
``continuation.class_means``, and for the same reason: every value depends on
its full ancestor chain, so one added node would invalidate essentially all of
them.
"""

from __future__ import annotations

import numpy as np

from sieve.continuation import (
    atom_variance,
    child_counts,
    class_sibling_variance,
    root_variance,
    sibling_variance,
)
from sieve.shrinkage import empirical_bayes_weights

# Selected on the val split of dash-molecules-10fold-1 by Gaussian NLL over a
# 200-point grid, then reported on test: the val pick reaches the best
# achievable *test* NLL to four decimals (gap +0.0000), so honest selection
# costs nothing here.
#
# Deliberately module constants rather than SieveConfig fields. The objective
# is flat -- the top twelve grid points span 0.004 in NLL and 0.006 in
# within-molecule ranking -- so exposing them as knobs would invite tuning
# that the measurement says cannot pay. They are named, not magic, and a
# caller who genuinely needs to sweep can pass them to `predictive_variance`.
ALPHA_V = 30.0  # within-class shrinkage toward the level-pooled variance
ALPHA_T = 1.0  # per-class sibling variance shrinkage toward the pooled one
SELECTION_WEIGHT = 0.5  # coefficient `a` on the selection term


def predictive_variance(
    model,
    *,
    alpha_v: float = ALPHA_V,
    alpha_t: float = ALPHA_T,
    selection_weight: float = SELECTION_WEIGHT,
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
    cfg = model.config
    parents = cfg.level_parents
    av = np.asarray(atom_variance(model), dtype=np.float64)
    tau_pooled = np.asarray(sibling_variance(model), dtype=np.float64)
    tau_class = class_sibling_variance(model)
    counts = child_counts(model)
    weights = empirical_bayes_weights(model)
    root = root_variance(model)

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

        out.append(within + selection_weight * selection + estimation)
    return out
