"""Bottom-up backoff through the refinement chain (design.md 6, 12)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wllr.batch import AtomBatch
from wllr.merge import _lookup_rows as _lookup
from wllr.merge import _translate
from wllr.refine import refine


@dataclass(frozen=True)
class Predictions:
    """Columnar prediction detail (design.md 12).

    These are diagnostics, not calibrated uncertainties: the triple
    (matched_level, support, variance) says how specific the environment was,
    how much support it had, and how heterogeneous its labels were.
    """

    value: np.ndarray             # (n, d)
    matched_level: np.ndarray     # (n,) k*, -1 for global fallback
    class_id: np.ndarray          # (n,) id at the matched level, -1 if none
    support: np.ndarray           # (n,) N at the matched class
    variance: np.ndarray          # (n, d) s^2, NaN where support == 1
    threshold_bound: np.ndarray   # (n,) stopped by n_min rather than by OOV
    raw_value: np.ndarray | None = None
    shrinkage_weight: np.ndarray | None = None


def predict_detailed(model, batch: AtomBatch) -> Predictions:
    """Predict with full metadata.

    The search runs upward through the chain. Both stopping conditions are
    valid only because matched levels form a prefix and support is monotone
    non-increasing (design.md 2.2, 2.3).
    """
    cfg = model.config
    n, d = batch.n_atoms, cfg.target_dim
    n_attr = len(cfg.attribute_levels)
    query = refine(batch, cfg)

    value = np.broadcast_to(model.global_mean, (n, d)).copy()
    matched = np.full(n, -1, np.int64)
    class_id = np.full(n, -1, np.int64)
    support = np.zeros(n, np.int64)
    variance = np.full((n, d), np.nan)
    threshold = np.zeros(n, bool)

    remap = None            # query class ids -> model class ids, previous level
    alive = np.ones(n, bool)
    for k in range(cfg.n_levels):
        lvl = model.levels[k]
        q = query[k]
        sig = _translate(q.signatures, remap, cfg.n_bond, is_wl=k >= n_attr)
        found = _lookup(sig, lvl.signatures)          # per query class
        remap = found
        if not alive.any():
            break                                     # graph-level stop (6.2)
        cid = found[q.labels]
        ok = alive & (cid >= 0)
        enough = np.zeros(n, bool)
        enough[ok] = lvl.count[cid[ok]] >= cfg.n_min
        # Stopped by support rather than by OOV: a different situation with a
        # different remedy, and indistinguishable without this flag.
        threshold |= alive & (cid >= 0) & ~enough
        hit = ok & enough
        value[hit] = lvl.mean[cid[hit]]
        matched[hit] = k
        class_id[hit] = cid[hit]
        support[hit] = lvl.count[cid[hit]]
        variance[hit] = lvl.variance[cid[hit]]
        alive = hit                                   # prefix property (2.2)

    if cfg.alpha is not None:
        from wllr.shrinkage import shrunk_means
        raw = value.copy()
        weight = np.zeros(n)
        shrunk = shrunk_means(model)
        for k in range(cfg.n_levels):
            sel = matched == k
            if sel.any():
                value[sel] = shrunk[k][class_id[sel]]
                nn = support[sel].astype(np.float64)
                weight[sel] = nn / (nn + cfg.alpha)
        return Predictions(value, matched, class_id, support, variance, threshold,
                           raw_value=raw, shrinkage_weight=weight)

    return Predictions(value, matched, class_id, support, variance, threshold)


def predict(model, batch: AtomBatch) -> np.ndarray:
    return predict_detailed(model, batch).value
