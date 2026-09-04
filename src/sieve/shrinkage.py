"""Hierarchically shrunk means, derived on demand (design.md 4.2, 4.4)."""

from __future__ import annotations

import numpy as np

from sieve.continuation import class_means


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

    One choice here is deliberate and unmeasured: the blend weight stays
    ``lvl.count`` (atom count) in both estimator modes, not "number of
    children" -- closer to Kneser-Ney's own lambda. Untested; not a knob.

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
    shrinkage_strength = model.config.shrinkage_strength
    parents = model.config.level_parents
    means = class_means(model)
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
        else:
            out.append(
                (n * raw + shrinkage_strength * parent_est) / (n + shrinkage_strength)
            )
    return out
