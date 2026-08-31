"""Hierarchically shrunk means, derived on demand (design.md 4.2)."""

from __future__ import annotations

import numpy as np


def shrunk_means(model) -> list[np.ndarray]:
    """Compute shrunk class means top-down, level 0 first.

    Each level consumes the *already-shrunk* parent, not the raw parent mean --
    which is why the pass must run downward and cannot be vectorized across
    levels.

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
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        n = lvl.count[:, None].astype(np.float64)
        p = parents[k]
        parent_est = (
            np.broadcast_to(model.global_mean, lvl.mean.shape)
            if p < 0
            else out[p][lvl.parent]
        )
        if shrinkage_strength is None or shrinkage_strength == 0.0:
            out.append(np.where(n > 0, lvl.mean, parent_est))
        else:
            out.append(
                (n * lvl.mean + shrinkage_strength * parent_est)
                / (n + shrinkage_strength)
            )
    return out
