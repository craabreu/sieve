"""Hierarchically shrunk means, derived on demand (design.md 4.2, 4.4)."""

from __future__ import annotations

import numpy as np

from sieve.config import (
    SHRINKAGE_WEIGHT_DIVERSITY,
    SHRINKAGE_WEIGHT_EMPIRICAL_BAYES,
)
from sieve.continuation import (
    atom_variance,
    child_counts,
    class_means,
    root_variance,
    sibling_variance,
)


def shrunk_means(model) -> list[np.ndarray]:
    """Compute shrunk class means top-down, level 0 first.

    Each level consumes the *already-shrunk* parent, not the raw parent mean --
    which is why the pass must run downward and cannot be vectorized across
    levels. "Raw" here means ``class_means(model)[k]`` -- the class's own
    estimate under ``model.config.class_estimator`` (design.md 4.4), pooled or
    continuation, computed one level deep and never itself recursive. Only the
    top-down *shrinkage* pass recurses; the per-level raw estimate it starts
    from at each step does not.

    This pairing is not incidental. By the prefix property (design.md 2.2), a
    class at any non-deepest level is read only by a query that backed off to
    it -- so shrinking such a class toward its own shrunk parent interpolates
    between two backoff-target reads, which is exactly interpolated
    Kneser-Ney's shape under ``class_estimator="continuation"``, not merely a
    composition that happens not to break.

    ``shrinkage_weight`` selects how that blend is weighted:

    - ``"count"`` (default, the original rule): the class's own estimate gets
      $N/(N+\\alpha)$, a function of its atom count alone.
    - ``"diversity"``: the *parent* gets $\\lambda=\\min(\\alpha C/N, 1)$ with
      $C$ the number of children, which is interpolated Kneser-Ney's own
      $\\lambda = D\\,N_{1+}(\\text{context})/c(\\text{context})$
      [KneserNey1995Improved] -- so ``shrinkage_strength`` reads as KN's
      discount $D$, not as a pseudo-count. The motivating argument was that a
      class whose atoms are spread over many distinct children is a worse
      summary of any one of them, a distinction the count rule cannot express.

    **Measured: the count rule wins.** On DASH charges (element, depth 6, 10
    folds, against flat continuation without shrinkage) the count rule gains
    +0.000081 at $\\alpha=0.5$ (t=19.7, 10/10 folds), while diversity gains
    only +0.000028 at $D=0.3$ (t=5.5) and *loses* 0.000336 at $D=0.75$
    (t=-23.4, 0/10 folds) -- i.e. it is actively harmful at the discount the
    language-modeling literature actually recommends. ``"count"`` stays the
    default; ``"diversity"`` is kept so the comparison stays reproducible.

    At the deepest level $C=0$, so the diversity rule applies no shrinkage
    there. That is the literal translation rather than a special case: the
    deepest class is the only one read on an exact match rather than by
    backoff, and the measured optimum under the count rule was already
    $\\alpha\\approx0.5$, i.e. nearly none.

    A caution for anyone revisiting this: the failure of $\\lambda=D C/N$ does
    *not* show that $C$ is the wrong statistic, only that this functional
    form is. Under continuation the estimand is a class's typical *child*
    mean, for which the unit of replication is the child, so an
    empirical-Bayes treatment would put $C$ in the *effective sample size*
    ($C/(C+\\alpha)$), not as a multiplier on $\\lambda$. Untested; see
    design.md 4.4.

    Never stored as model state: any added data changes the global mean and
    every estimate depends on its full ancestor chain, so one new node
    invalidates essentially every value. There is no incremental patch.

    Walks ``config.level_parents`` rather than assuming a level's parent is
    always ``k - 1``: with ``neighbor_depth`` set (design.md 3.6) the main
    chain's first WL round branches off the last attribute level, not the
    coarse chain's own last level, so the two branch roots need the actual
    parent level looked up. ``level_parents[k] < k`` always, so a single
    forward pass already sees each parent before it's needed.
    """
    cfg = model.config
    shrinkage_strength = cfg.shrinkage_strength
    parents = cfg.level_parents
    means = class_means(model)
    diversity = cfg.effective_shrinkage_weight == SHRINKAGE_WEIGHT_DIVERSITY
    eb = cfg.effective_shrinkage_weight == SHRINKAGE_WEIGHT_EMPIRICAL_BAYES
    applies = cfg.applies_shrinkage
    counts = child_counts(model) if diversity else None
    eb_w = empirical_bayes_weights(model) if eb else None
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        n = lvl.count[:, None].astype(np.float64)
        raw = means[k]
        p = parents[k]
        parent_est = (
            np.broadcast_to(model.global_mean, raw.shape)
            if p < 0
            else out[p][lvl.parent]
        )
        if eb:
            # alpha is estimated, so shrinkage_strength plays no part -- there
            # is no "off" setting to check for here.
            assert eb_w is not None  # set iff `eb`; the checker cannot see that
            w = eb_w[k][:, None]
            out.append(np.where(n > 0, w * raw + (1.0 - w) * parent_est, parent_est))
        elif not applies:
            out.append(np.where(n > 0, raw, parent_est))
        elif diversity:
            # KN's lambda: weight on the parent grows with the number of
            # distinct children per atom. Clipped at 1 -- C > N/alpha is
            # possible for a small, highly fragmented class, and beyond that
            # the class contributes nothing of its own.
            assert counts is not None  # set iff `diversity`
            lam = np.minimum(
                shrinkage_strength * counts[k][:, None] / np.maximum(n, 1.0), 1.0
            )
            out.append(
                np.where(n > 0, (1.0 - lam) * raw + lam * parent_est, parent_est)
            )
        else:
            out.append(
                (n * raw + shrinkage_strength * parent_est) / (n + shrinkage_strength)
            )
    return out


def empirical_bayes_weights(model) -> list[np.ndarray]:
    r"""Per-class weight on a class's *own* estimate under
    ``shrinkage_weight="empirical_bayes"``.

    Two cases, because the estimand differs (design.md 4.4):

    - **Non-deepest class** under continuation: it estimates its typical
      child's mean, whose unit of replication is the child, so the
      normal-normal posterior weight is $C/(C+\alpha_k)$ with
      $\alpha_k=\hat\tau^2_k/\hat\tau^2_{k-1}$.
    - **Deepest class**: no children, so it estimates its own pooled mean and
      ordinary atom-level EB applies -- $N/(N+\bar\sigma^2/\hat\tau^2_{k-1})$,
      exactly Morris's original with $\alpha$ estimated rather than swept.

    $\hat\tau^2_{k-1}=0$ means the parent's children are indistinguishable,
    so $\alpha=\infty$ and the class is shrunk fully -- the reading
    design.md 13 item 9 gives that clamp. A non-estimable $\hat\tau^2$
    (``nan``, no level has two siblings anywhere) falls back to no shrinkage,
    which is what the level would do with no evidence to shrink on.
    """
    parents = model.config.level_parents
    tau = sibling_variance(model)
    sigma = atom_variance(model)
    counts = child_counts(model)
    root = root_variance(model)

    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        p = parents[k]
        tau_parent = root if p < 0 else tau[p]
        n = lvl.count.astype(np.float64)
        c = counts[k]
        # numerator: the class's own scale -- sibling spread where it has
        # children, atom-level noise where it does not.
        num = np.where(c > 0, tau[k] if tau[k] == tau[k] else np.nan, sigma[k])
        size = np.where(c > 0, c, n)
        if tau_parent != tau_parent:  # nan: nothing to shrink toward
            out.append(np.ones(len(n)))
            continue
        if tau_parent <= 0.0:
            out.append(np.zeros(len(n)))  # alpha -> inf: shrink fully
            continue
        alpha = np.where(num == num, num / tau_parent, 0.0)
        out.append(size / (size + alpha))
    return out
