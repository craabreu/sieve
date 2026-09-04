"""Hierarchically shrunk means, derived on demand (design.md 4.2, 4.4)."""

from __future__ import annotations

import numpy as np

from sieve.config import SHRINKAGE_WEIGHT_DIVERSITY
from sieve.continuation import child_counts, class_means


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
      discount $D$, not as a pseudo-count. A class whose atoms are spread
      over many distinct children is a worse summary of any one of them and
      defers harder to its parent than an equally-sized class concentrated in
      a few; the count rule cannot express that distinction at all.

    At the deepest level $C=0$, so the diversity rule applies no shrinkage
    there. That is the literal translation rather than a special case: the
    deepest class is the only one read on an exact match rather than by
    backoff, and the measured optimum under the count rule was already
    $\\alpha\\approx0.5$, i.e. nearly none.

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
    diversity = cfg.shrinkage_weight == SHRINKAGE_WEIGHT_DIVERSITY
    counts = child_counts(model) if diversity else None
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
        if shrinkage_strength is None or shrinkage_strength == 0.0:
            out.append(np.where(n > 0, raw, parent_est))
        elif diversity:
            # KN's lambda: weight on the parent grows with the number of
            # distinct children per atom. Clipped at 1 -- C > N/alpha is
            # possible for a small, highly fragmented class, and beyond that
            # the class contributes nothing of its own.
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
