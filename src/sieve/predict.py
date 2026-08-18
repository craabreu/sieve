"""Bottom-up backoff through the refinement chain (design.md 6, 10.3, 12)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sieve.batch import AtomBatch
from sieve.merge import _lookup_rows as _lookup
from sieve.merge import _translate
from sieve.refine import refine


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
    support: np.ndarray           # (n,) N at the matched class (eff_n if LOO)
    variance: np.ndarray          # (n, d) s^2, NaN where support == 1
    threshold_bound: np.ndarray   # (n,) stopped by n_min rather than by OOV
    raw_value: np.ndarray | None = None
    shrinkage_weight: np.ndarray | None = None


def _search(model, batch: AtomBatch, loo_y: np.ndarray | None = None) -> Predictions:
    """Bottom-up backoff, shared by ``predict_detailed`` and ``predict_loo``.

    With ``loo_y`` set, a training node's own contribution is subtracted from
    its class mean before the support check, and a class with only one member
    is treated as unsupported (design.md 10.3) rather than dividing by zero.
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
        est = np.zeros((n, d))
        eff_n_full = np.zeros(n)
        if ok.any():
            cnt = lvl.count[cid[ok]].astype(np.float64)
            if loo_y is None:
                est_ok = lvl.mean[cid[ok]]
                eff_n = cnt
            else:
                eff_n = cnt - 1.0
                # N == 1 leaves nothing behind: treat as unsupported and let
                # the search back off to the parent rather than dividing by 0.
                est_ok = np.where(eff_n[:, None] > 0,
                                  (cnt[:, None] * lvl.mean[cid[ok]] - loo_y[ok]) /
                                  np.maximum(eff_n, 1.0)[:, None],
                                  np.nan)
            enough[ok] = eff_n >= cfg.n_min
            est[ok] = est_ok
            eff_n_full[ok] = eff_n
        # Stopped by support rather than by OOV: a different situation with a
        # different remedy, and indistinguishable without this flag.
        threshold |= alive & (cid >= 0) & ~enough
        hit = ok & enough
        value[hit] = est[hit]
        matched[hit] = k
        class_id[hit] = cid[hit]
        support[hit] = eff_n_full[hit].astype(np.int64)
        variance[hit] = lvl.variance[cid[hit]]
        alive = hit                                   # prefix property (2.2)

    if cfg.alpha is not None:
        from sieve.shrinkage import shrunk_means
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


def predict_detailed(model, batch: AtomBatch) -> Predictions:
    """Predict with full metadata."""
    return _search(model, batch)


def predict(model, batch: AtomBatch) -> np.ndarray:
    return predict_detailed(model, batch).value


def predict_loo(model, batch: AtomBatch) -> Predictions:
    """Predict training nodes with their own contribution removed (design.md 10.3).

    A training node contributes its own label to its class mean, so any
    in-sample score is meaningless -- at n_min=1 and large L it approaches
    perfect recall. This is the standard remedy from the target-encoding
    literature, and the cheapest test that the implementation is not leaking.
    """
    if batch.y is None:
        raise ValueError("predict_loo requires targets; batch.y is None")
    return _search(model, batch, loo_y=batch.y)
