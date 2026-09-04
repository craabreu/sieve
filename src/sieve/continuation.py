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

from sieve.config import (
    CLASS_ESTIMATOR_CONTINUATION_RECURSIVE,
    CLASS_ESTIMATOR_POOLED,
)


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


def child_counts(model) -> list[np.ndarray]:
    """Per-level ``C``: how many children each class has on the backoff path.

    Kneser-Ney's $N_{1+}(\\text{context},\\bullet)$ -- the number of distinct
    continuations of a context. Zero at the deepest backoff level and at any
    level off the path, both of which have no children by construction.

    Derived on demand, like ``class_means``. Used by
    ``shrinkage.shrunk_means`` under ``shrinkage_weight="diversity"``.
    """
    child_of = _child_of_level(model.config)
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        c = child_of[k]
        if c < 0:
            out.append(np.zeros(len(lvl.mean), np.float64))
        else:
            out.append(
                np.bincount(model.levels[c].parent, minlength=len(lvl.mean)).astype(
                    np.float64
                )
            )
    return out


def class_means(model) -> list[np.ndarray]:
    """Per-level class-mean table, matching ``model.config.class_estimator``.

    ``"continuation"`` is **one level deep**, over the *stored* means of the
    level below -- never averaging other levels' own continuation estimates.
    ``"continuation_recursive"`` is the same aggregation applied to the
    children's own continuation estimates instead, which forces a
    deepest-level-first pass (a level now depends on the level *below* it,
    the opposite direction from ``shrunk_means``). A level with no children
    on the backoff path (the deepest backoff level, or any level off it
    entirely) keeps its stored, pooled mean under both.

    The two differ only at levels at least two steps above the deepest: the
    level immediately above it averages children that are themselves pooled,
    so flat and recursive agree there by construction.

    Derived on demand, like ``shrinkage.shrunk_means``: never stored as model
    state, so it stays consistent with ``model.config.class_estimator``
    (excluded from ``schema_version``) without any invalidation bookkeeping.
    """
    cfg = model.config
    if cfg.class_estimator == CLASS_ESTIMATOR_POOLED:
        return [lvl.mean for lvl in model.levels]

    recursive = cfg.class_estimator == CLASS_ESTIMATOR_CONTINUATION_RECURSIVE
    child_of = _child_of_level(cfg)
    out: list[np.ndarray] = [lvl.mean for lvl in model.levels]
    # Deepest first: under `recursive` a level reads the level below's
    # already-computed estimate, so that one must be finished first. The flat
    # form is order-independent, and shares the loop.
    for k in range(len(model.levels) - 1, -1, -1):
        lvl = model.levels[k]
        c = child_of[k]
        if c < 0:
            continue  # no children: keep the stored pooled mean
        source = out[c] if recursive else model.levels[c].mean
        parent_of_child = model.levels[c].parent
        d = lvl.mean.shape[1]
        tot = np.empty((len(lvl.mean), d))
        for j in range(d):
            tot[:, j] = np.bincount(
                parent_of_child, weights=source[:, j], minlength=len(lvl.mean)
            )
        cnt = np.bincount(parent_of_child, minlength=len(lvl.mean)).astype(np.float64)
        # cnt == 0 only for a level-k class with no observed children -- not
        # reachable in a fitted model (classes are only minted for observed
        # rows, so every non-deepest class has >=1 child) but kept as a
        # defensive fallback to the pooled mean rather than a division by
        # zero, consistent with how the rest of the codebase treats this
        # kind of structural invariant.
        out[k] = np.where(cnt[:, None] > 0, tot / np.maximum(cnt, 1)[:, None], lvl.mean)
    return out
