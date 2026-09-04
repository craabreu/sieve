"""Bottom-up backoff through the refinement chain (design.md 6, 10.3, 12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import (
    CLASS_ESTIMATOR_POOLED,
    LEVEL_WL,
    SHRINKAGE_WEIGHT_COUNT,
    SHRINKAGE_WEIGHT_DIVERSITY,
    SHRINKAGE_WEIGHT_EMPIRICAL_BAYES,
)
from sieve.continuation import child_counts, class_means
from sieve.merge import _lookup_rows as _lookup
from sieve.merge import _translate
from sieve.refine import refine


@dataclass(frozen=True)
class Predictions:
    """Columnar prediction detail (design.md 12).

    These are diagnostics, not calibrated uncertainties: the triple
    (matched_level, support, variance) says how specific the environment was,
    how much support it had, and how heterogeneous its labels were.

    Under ``cfg.class_estimator == "continuation"`` (design.md 4.4), `value`
    at a non-deepest level is the unweighted mean of the matched class's
    *children* -- `support`/`variance` still describe the matched class's
    own pooled population, not the children averaged into `value`.
    """

    value: np.ndarray  # (n, d)
    matched_level: np.ndarray  # (n,) k*, -1 for global fallback
    class_id: np.ndarray  # (n,) id at the matched level, -1 if none
    support: np.ndarray  # (n,) N at the matched class (eff_n if LOO)
    # (n, d) s^2, NaN where support == 1. Always the model's stored class
    # variance (design.md 10.3's LOO formula covers only the mean); under
    # predict_loo this is *not* adjusted for the held-out node's own label,
    # unlike `value` and `support`.
    variance: np.ndarray
    threshold_bound: np.ndarray  # (n,) stopped by minimum_support rather than by OOV
    raw_value: np.ndarray | None = None
    shrinkage_weight: np.ndarray | None = None


def _search(model, batch: NodeBatch, loo_y: np.ndarray | None = None) -> Predictions:
    """Bottom-up backoff, shared by ``predict_detailed`` and ``predict_loo``.

    With ``loo_y`` set, a training node's own contribution is subtracted from
    its class mean before the support check, and a class with only one member
    is treated as unsupported (design.md 10.3) rather than dividing by zero.

    When ``cfg.neighbor_depth`` is set, ``query``/``model.levels`` also carry
    the coarse neighbor chain (design.md 3.6). Its levels are translated and
    looked up like any other -- the fine ``LEVEL_WL_PAIR`` rounds need their
    class ids -- but never scored or backed off over; only levels on
    ``cfg.backoff_path`` update ``matched``/``value``/etc. ``matched_level``
    on the returned ``Predictions`` reports *position along that path*, not
    the raw level index, so it stays comparable across different
    ``neighbor_depth`` settings (reduces to the raw index when unset).

    ``cfg.class_estimator == "continuation"`` is not yet supported under LOO
    (``loo_y`` set): the LOO correction needs the held-out node's own child
    class identity, which this loop does not currently retain across
    iterations. Deliberate scope cut, not a structural limitation -- the
    correction has a closed form and the information is one array away
    (each node's child ``cid`` is already computed the iteration before it
    drops out of ``alive``); left out because the continuation experiment
    this exists for does not exercise ``predict_loo``. A caller reaches this
    almost certainly by way of ``report_loo``, which should raise earlier
    still -- see ``charge_experiments``' ``SievePredictor.__init__``.
    """
    cfg = model.config
    if loo_y is not None and cfg.class_estimator != CLASS_ESTIMATOR_POOLED:
        raise NotImplementedError(
            f"predict_loo does not yet support class_estimator={cfg.class_estimator!r}"
        )
    if loo_y is not None and cfg.shrinkage_weight != SHRINKAGE_WEIGHT_COUNT:
        # lambda is built from the matched class's own C and N, both of which
        # the held-out node contributes to; correcting it needs the same child
        # identity predict_loo already lacks for continuation.
        raise NotImplementedError(
            f"predict_loo does not yet support "
            f"shrinkage_weight={cfg.shrinkage_weight!r}"
        )
    n, d = batch.n_nodes, cfg.target_dim
    query = refine(batch, cfg)

    means = class_means(model)
    kinds = cfg.level_kinds
    parents = cfg.level_parents
    neighbor_src = cfg.neighbor_source
    backoff_path = cfg.backoff_path
    on_backoff = np.zeros(cfg.n_levels, bool)
    on_backoff[list(backoff_path)] = True
    backoff_pos = np.full(cfg.n_levels, -1, np.int64)
    backoff_pos[list(backoff_path)] = np.arange(len(backoff_path))

    value = np.broadcast_to(model.global_mean, (n, d)).copy()
    matched = np.full(n, -1, np.int64)
    class_id = np.full(n, -1, np.int64)
    support = np.zeros(n, np.int64)
    variance = np.full((n, d), np.nan)
    threshold = np.zeros(n, bool)

    remaps: list[np.ndarray] = []  # query class ids -> model class ids, per level
    alive = np.ones(n, bool)
    for k in range(cfg.n_levels):
        if not alive.any():
            break  # graph-level stop (6.2)
        lvl = model.levels[k]
        q = query[k]
        kind = kinds[k]
        p = parents[k]
        remap_prev = None if p < 0 else remaps[p]
        ns = neighbor_src[k]
        remap_neighbor = None if ns is None else remaps[ns]
        sig = _translate(
            q.signatures, remap_prev, cfg.n_edge_types, kind, remap_neighbor
        )
        found = _lookup(sig, lvl.signatures, kind == LEVEL_WL)  # per query class
        remaps.append(found)
        if not on_backoff[k]:
            continue  # coarse-chain scaffolding: never a backoff target itself
        cid = found[q.labels]
        ok = alive & (cid >= 0)
        enough = np.zeros(n, bool)
        est = np.zeros((n, d))
        eff_n_full = np.zeros(n)
        if ok.any():
            cnt = lvl.count[cid[ok]].astype(np.float64)
            if loo_y is None:
                est_ok = means[k][cid[ok]]
                eff_n = cnt
            else:
                eff_n = cnt - 1.0
                # N == 1 leaves nothing behind: treat as unsupported and let
                # the search back off to the parent rather than dividing by 0.
                # lvl.mean, not means[k]: this branch only runs when loo_y is
                # set, and continuation is already rejected above in that
                # case, so cfg.class_estimator is guaranteed "pooled" here.
                est_ok = np.where(
                    eff_n[:, None] > 0,
                    (cnt[:, None] * lvl.mean[cid[ok]] - loo_y[ok])
                    / np.maximum(eff_n, 1.0)[:, None],
                    np.nan,
                )
            enough[ok] = eff_n >= cfg.minimum_support
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
        alive = hit  # prefix property (2.2)

    # `matched` stays the raw level index for the shrinkage loop below (it
    # indexes model.levels/shrunk, both raw-indexed); only the value handed
    # back on Predictions is translated to backoff-path position, right
    # before each return.
    def _matched_out(matched: np.ndarray) -> np.ndarray:
        return np.where(matched >= 0, backoff_pos[matched], -1)

    if cfg.applies_shrinkage:
        from sieve.shrinkage import empirical_bayes_weights, shrunk_means

        raw = value.copy()
        weight = np.zeros(n)
        # shrunk_means(model) is the model's own class-indexed shrunk means,
        # unaware of any query. Reusing shrunk[k][class_id] directly here
        # would be correct for `predict` but wrong for `predict_loo`: it
        # shrinks the class mean *before* the node's own label is removed,
        # silently reintroducing exactly the leakage §10.3 exists to avoid.
        # Recomputing shrinkage from `raw`/`support` -- the per-node value
        # this loop already derived from `est`/`eff_n` at the matched level,
        # LOO-adjusted when `loo_y` is set -- combined with the *parent's*
        # shrunk estimate reduces to the identical formula (and identical
        # values) for `predict`, while getting `predict_loo` right too.
        #
        # Under shrinkage_weight="diversity" that per-node recomputation has
        # nothing to recompute: lambda is a pure class property (C and N of
        # the matched class), not a function of the query's own support, so
        # the class-indexed shrunk value *is* the answer. LOO is refused for
        # that mode above, which is what makes reading it directly safe here.
        shrunk = shrunk_means(model)
        diversity = cfg.shrinkage_weight == SHRINKAGE_WEIGHT_DIVERSITY
        eb = cfg.shrinkage_weight == SHRINKAGE_WEIGHT_EMPIRICAL_BAYES
        counts = child_counts(model) if diversity else None
        eb_w = empirical_bayes_weights(model) if eb else None
        for k in backoff_path:
            sel = matched == k
            if sel.any():
                nn = support[sel].astype(np.float64)
                p = parents[k]
                if p < 0:
                    parent_est = np.broadcast_to(model.global_mean, (sel.sum(), d))
                else:
                    parent_est = shrunk[p][model.levels[k].parent[class_id[sel]]]
                if eb:
                    # The weight is estimated per class, so the class-indexed
                    # shrunk value is already the answer -- and the estimated
                    # weight is itself the interesting diagnostic here, since
                    # it is the only mode where it varies for a reason other
                    # than support.
                    value[sel] = shrunk[k][class_id[sel]]
                    weight[sel] = eb_w[k][class_id[sel]]
                elif diversity:
                    cls_n = model.levels[k].count[class_id[sel]].astype(np.float64)
                    lam = np.minimum(
                        cfg.shrinkage_strength
                        * counts[k][class_id[sel]]
                        / np.maximum(cls_n, 1.0),
                        1.0,
                    )
                    value[sel] = (1.0 - lam[:, None]) * raw[sel] + lam[
                        :, None
                    ] * parent_est
                    weight[sel] = 1.0 - lam
                else:
                    value[sel] = (
                        nn[:, None] * raw[sel] + cfg.shrinkage_strength * parent_est
                    ) / (nn[:, None] + cfg.shrinkage_strength)
                    weight[sel] = nn / (nn + cfg.shrinkage_strength)
        return Predictions(
            value,
            _matched_out(matched),
            class_id,
            support,
            variance,
            threshold,
            raw_value=raw,
            shrinkage_weight=weight,
        )

    return Predictions(
        value, _matched_out(matched), class_id, support, variance, threshold
    )


def predict_detailed(model, batch: NodeBatch) -> Predictions:
    """Predict with full metadata."""
    return _search(model, batch)


def predict(model, batch: NodeBatch) -> np.ndarray:
    return predict_detailed(model, batch).value


def predict_loo(model, batch: NodeBatch) -> Predictions:
    """Predict training nodes with their own contribution removed (design.md 10.3).

    A training node contributes its own label to its class mean, so any
    in-sample score is meaningless -- at minimum_support=1 and large L it
    approaches perfect recall. This is the standard remedy from the
    target-encoding literature, and the cheapest test that the implementation
    is not leaking.
    """
    if batch.y is None:
        raise ValueError("predict_loo requires targets; batch.y is None")
    return _search(model, batch, loo_y=batch.y)
