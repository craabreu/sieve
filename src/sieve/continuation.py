"""Continuation-count class estimates (design.md 4.4).

The Kneser-Ney correction to hierarchical backoff, adapted from language
modeling [KneserNey1995Improved; ChenGoodman1999Smoothing]: a class's stored
mean is the atom-weighted pool of everything beneath it, dominated by its most
abundant children, yet that value is read *only* by a query whose own child
class was absent -- i.e. only by backoff (design.md 2.2's prefix property). The
pooled estimate is calibrated for the wrong population.

The continuation estimate aggregates a class's children as *units* instead:
the unweighted mean of the children's own stored means, so a rare-but-real
child stops being swamped by an abundant one. See literature.md 4.2.

"Continuation" names Kneser-Ney's type-counting correction only. It is *not*
the recursive base-measure construction of the hierarchical Bayesian form of
the same idea [Teh2006HierarchicalPitmanYor], where each level's prior mean
is the coarser level's own modeled distribution. Sieve does recurse that way
-- in ``shrinkage.shrunk_means`` (design.md 4.2), not here. Keeping exactly
one recursing mechanism is deliberate; see design.md 4.4.
"""

from __future__ import annotations

import numpy as np

from sieve.config import CLASS_ESTIMATOR_POOLED


def _child_of_level(cfg) -> list[int]:
    """Absolute index of the level supplying level ``k``'s children, or ``-1``
    if ``k`` has none on the backoff path.

    Deliberately *not* "every level whose ``parent`` points at k": under
    ``neighbor_depth``, the attribute level at ``neighbor_depth - 1`` parents
    both the main chain's first WL round (on ``backoff_path``) and the coarse
    chain's root (off it, scaffolding only -- design.md 3.6). Including the
    coarse chain would average the same atoms twice under two different
    labels. Walking ``backoff_path`` in order already excludes it.
    """
    path = cfg.backoff_path
    child = [-1] * cfg.n_levels
    for i in range(len(path) - 1):
        child[path[i]] = path[i + 1]
    return child


def class_means(model) -> list[np.ndarray]:
    """Per-level class-mean table, matching ``model.config.class_estimator``.

    Always **one level deep**, over the *stored* means of the level below --
    never recursive, never averaging other levels' own continuation
    estimates. A level with no children on the backoff path (the deepest
    backoff level, or any level off it entirely) keeps its stored, pooled
    mean.

    Derived on demand, like ``shrinkage.shrunk_means``: never stored as model
    state, so it stays consistent with ``model.config.class_estimator``
    (excluded from ``schema_version``) without any invalidation bookkeeping.
    """
    cfg = model.config
    if cfg.class_estimator == CLASS_ESTIMATOR_POOLED:
        return [lvl.mean for lvl in model.levels]

    child_of = _child_of_level(cfg)
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        c = child_of[k]
        if c < 0:
            out.append(lvl.mean)
            continue
        child = model.levels[c]
        d = lvl.mean.shape[1]
        tot = np.empty((len(lvl.mean), d))
        for j in range(d):
            tot[:, j] = np.bincount(
                child.parent, weights=child.mean[:, j], minlength=len(lvl.mean)
            )
        cnt = np.bincount(child.parent, minlength=len(lvl.mean)).astype(np.float64)
        # cnt == 0 only for a level-k class with no observed children -- not
        # reachable in a fitted model (classes are only minted for observed
        # rows, so every non-deepest class has >=1 child) but kept as a
        # defensive fallback to the pooled mean rather than a division by
        # zero, consistent with how the rest of the codebase treats this
        # kind of structural invariant.
        out.append(
            np.where(cnt[:, None] > 0, tot / np.maximum(cnt, 1)[:, None], lvl.mean)
        )
    return out
