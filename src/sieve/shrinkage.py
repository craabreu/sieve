"""Hierarchically shrunk means, derived on demand (design.md 4.2)."""

from __future__ import annotations

import numpy as np


def shrunk_means(model) -> list[np.ndarray]:
    """Compute shrunk class means top-down, level 0 first.

    Each level consumes the *already-shrunk* parent, not the raw parent mean --
    which is why the pass must run downward and cannot be vectorised across
    levels.

    Never stored as model state: any added data changes the global mean and
    every estimate depends on its full ancestor chain, so one new node
    invalidates essentially every value. There is no incremental patch.
    """
    alpha = model.config.alpha
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        n = lvl.count[:, None].astype(np.float64)
        parent_est = (
            np.broadcast_to(model.global_mean, lvl.mean.shape)
            if k == 0
            else out[k - 1][lvl.parent]
        )
        if alpha is None or alpha == 0.0:
            out.append(np.where(n > 0, lvl.mean, parent_est))
        else:
            out.append((n * lvl.mean + alpha * parent_est) / (n + alpha))
    return out
