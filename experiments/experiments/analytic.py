"""Training diagnostics computed from a fit's own statistics, with no data.

A fitted model already stores, per class, the count, mean and mean-squared
deviation of the training atoms it holds. That is enough to reproduce the
training error exactly -- and the training R^2, eta^2, the support
distribution and the leave-one-out error with it -- without loading a single
molecule or running a single walk.

The argument, in one line: a training atom's own class exists at every level
it reaches, because the atom itself contributed to it, so the backoff search
is decided entirely by the stored counts. Whichever class answers, the
prediction is that class's stored estimate, and the error against the class's
own stored moments is algebra.

See ``experiments/docs/analytic-diagnostics-and-conformer-collapse.md``
sections 1-3, which measured the identity to 1.7e-18 against a real
``--score-train`` run before any of this existed.

This module is deliberately an *identity on the stored statistics*, not an
independent measurement. If ``predict`` ever disagreed with what the fit
stored, the analytic curve would agree with the bug -- so ``--score-train``
stays available as an occasional cross-check rather than being deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# The support thresholds whose atom share every curve reports. 12 is this
# series' own `minimum_support` ("at least four molecules" at three conformers
# each); the others bracket it.
SUPPORT_THRESHOLDS: tuple[int, ...] = (2, 4, 12, 50)


@dataclass(frozen=True)
class TrainStats:
    """One row of an analytic curve.

    ``sse`` is summed over target dimensions and ``n_atoms`` counts atoms, so
    ``rmse`` divides by their product -- the same convention the harness'
    own metrics use for a vector target.

    ``eta_squared`` is a property of the partition alone (the share of total
    variance lying between classes at the deepest level), while ``r_squared``
    is a property of the fitted estimator (1 - SSE/TSS). They coincide
    exactly when no shrinkage applies and every atom matches its deepest
    class, and diverge as soon as either is false -- which is the point of
    reporting both.
    """

    depth: int
    n_classes: int
    n_atoms: int
    sse: float
    rmse: float
    r_squared: float
    eta_squared: float
    matched_fraction: float
    support_fractions: dict[int, float]

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "depth": self.depth,
            "n_classes": self.n_classes,
            "n_atoms": self.n_atoms,
            "sse": self.sse,
            "rmse": self.rmse,
            "r2": self.r_squared,
            "eta2": self.eta_squared,
            "matched_fraction": self.matched_fraction,
        }
        for threshold, fraction in self.support_fractions.items():
            row[f"frac_support_lt_{threshold}"] = fraction
        return row


def _moments(count: NDArray, mean: NDArray, msd: NDArray) -> tuple[NDArray, ...]:
    """Per class: ``(n, sum y, sum y^2)``, the form that adds across classes.

    ``msd`` is the population variance, so ``sum y^2 = n * (msd + mean^2)``.
    Reconstructing the second moment this way rather than storing it costs
    one multiply and keeps the model format untouched.
    """
    n = count.astype(np.float64)
    s = n[:, None] * mean
    q = n[:, None] * (msd + mean * mean)
    return n, s, q


def _sse_against(
    n: NDArray, s: NDArray, q: NDArray, value: NDArray, scale: NDArray | float = 1.0
) -> float:
    """``sum_i scale_c * (y_i - v)^2`` for a group summarized by ``(n, sum y,
    sum y^2)``, one term per class *c* before collapsing to a single number.

    Expanded rather than centered because the group's own mean is not the
    prediction here: under shrinkage, backoff or LOO the value comes from
    somewhere else entirely, and the expansion is the only form that takes
    an arbitrary ``v``.

    ``scale`` must be applied *before* the sum, not after: under LOO it is a
    per-class array (``_loo_scale``), and multiplying it onto the
    already-summed total would broadcast one group's grand total across
    every class's own scale factor instead of scaling each class's own
    contribution -- silently wrong, and it also turns ``sse`` from a running
    float into an array whose length changes with the class count at each
    level, which is what a naive post-hoc multiply did here before this was
    caught by the shape mismatch it produces across levels of different size.
    """
    return float((scale * (q - 2.0 * value * s + n[:, None] * value * value)).sum())


def sieve_train_stats(
    model: Any,
    *,
    loo: bool = False,
    thresholds: tuple[int, ...] = SUPPORT_THRESHOLDS,
) -> TrainStats:
    """Exact training (or leave-one-out) statistics for one fitted model.

    Walks the backoff chain from the deepest level up, exactly as
    ``sieve.predict._search`` walks it down. At each level a class either has
    the support the config demands -- in which case every training atom still
    carrying it is answered there -- or it does not, and its atoms' moments
    are folded into its parent class and tried again one level up. Atoms that
    run out of levels are answered by the global mean.

    This is the general form. Section 1 of the diagnostics note derives only
    the ``minimum_support=1``, pooled, unshrunk case, where the walk stops
    immediately at the deepest level and the whole thing collapses to
    ``SSE = sum_c n_c var_c``; that case is reproduced here exactly, as one
    branch of the same code.

    ``loo=True`` removes each atom's own contribution from the class that
    answers it, which is closed-form: the residual of atom *i* against the
    leave-one-out mean of an *N*-atom class is ``N/(N-1)`` times its ordinary
    residual, so the group's SSE scales by ``N^2/(N-1)^2``. Refused for the
    estimators ``sieve.predict_loo`` itself refuses, for the same reason --
    the correction needs a child class identity this walk does not carry.
    """
    from sieve.shrinkage import shrunk_means

    cfg = model.config
    if loo:
        _refuse_loo_unsupported(cfg)

    path = cfg.backoff_path
    parents = cfg.level_parents
    minimum_support = cfg.minimum_support
    # Populated classes take their own estimate and empty ones their parent's,
    # which is exactly what `shrunk_means` returns with no shrinkage applied --
    # so this one call covers pooled, continuation and every shrinkage mode.
    values = shrunk_means(model)

    deepest = model.levels[path[-1]]
    n_in, s_in, q_in = _moments(deepest.count, deepest.mean, deepest.msd)
    n_total = float(n_in.sum())
    if n_total == 0:
        raise ValueError("model holds no training atoms; nothing to score")

    sse = 0.0
    n_matched = 0.0
    for k in reversed(path):
        level = model.levels[k]
        carried = n_in > 0
        supported = level.count >= (minimum_support + 1 if loo else minimum_support)
        hit = carried & supported
        if hit.any():
            scale = _loo_scale(level.count[hit]) if loo else 1.0
            centre = level.mean[hit] if loo else values[k][hit]
            sse += _sse_against(n_in[hit], s_in[hit], q_in[hit], centre, scale=scale)
            n_matched += float(n_in[hit].sum())

        rest = carried & ~supported
        parent = parents[k]
        if parent < 0:
            if rest.any():
                sse += _global_sse(model, n_in[rest], s_in[rest], q_in[rest])
            break
        if not rest.any():
            break  # every atom answered; the levels above hold nothing more
        n_in, s_in, q_in = _fold_into_parent(
            model.levels[parent], level.parent, rest, n_in, s_in, q_in
        )
    else:
        # `path` exhausted without reaching a level whose parent is -1, which
        # only a malformed config could produce.
        raise AssertionError("backoff path ended without a root level")

    return _finish(
        model, deepest, sse, n_total, n_matched, thresholds, depth=cfg.max_wl_depth
    )


def _loo_scale(count: NDArray) -> NDArray:
    """``(N/(N-1))^2``, the factor a leave-one-out residual picks up.

    Applied per class, against the class's own *full* count: the held-out
    atom is one of that class's own members whatever level answered it.

    A singleton never reaches this: under LOO a class must hold at least
    ``minimum_support + 1 >= 2`` atoms to answer, so a class of one fails the
    support test and is folded into its parent instead. That invariant lives
    in the caller's mask rather than here, so it is asserted -- without it
    this would return ``inf`` rather than fail, and a wrong floor would be
    silently absorbed into the SSE.
    """
    n = count.astype(np.float64)
    if n.size and n.min() < 2.0:
        raise AssertionError(
            "_loo_scale received a class of fewer than 2 atoms; a singleton "
            "must be folded into its parent, not scored in place"
        )
    return (n / (n - 1.0))[:, None] ** 2


def supports_loo(cfg: Any) -> bool:
    """Whether the analytic LOO is exact for this reading of the tables.

    A caller that wants LOO "where it is available" should ask this rather
    than catch the refusal below, so the two cannot drift apart.
    """
    from sieve.config import CLASS_ESTIMATOR_POOLED

    return cfg.class_estimator == CLASS_ESTIMATOR_POOLED and not cfg.applies_shrinkage


def _refuse_loo_unsupported(cfg: Any) -> None:
    from sieve.config import CLASS_ESTIMATOR_POOLED

    if cfg.class_estimator != CLASS_ESTIMATOR_POOLED:
        raise NotImplementedError(
            f"analytic LOO does not support class_estimator="
            f"{cfg.class_estimator!r}: a non-deepest class estimates its "
            f"children's mean, so removing one atom needs the child identity "
            f"this walk does not carry -- the same limit sieve.predict_loo has"
        )
    if cfg.applies_shrinkage:
        raise NotImplementedError(
            "analytic LOO does not support shrinkage: the shrunk value "
            "depends on the class count the held-out atom contributes to, so "
            "the correction does not factor out of the class"
        )


def _global_sse(model: Any, n: NDArray, s: NDArray, q: NDArray) -> float:
    """SSE for atoms no level answered, against the whole-corpus mean.

    No LOO correction is applied here even when ``loo=True``: ``sieve.predict.
    _search`` initializes ``value`` to the raw, un-adjusted ``global_mean``
    *before* its backoff loop runs, and an atom that fails every level never
    gets that initial value touched again -- so ``predict_loo`` itself does
    not leave-one-out-correct its own fallback, and reproducing it exactly
    means not correcting here either. Verified against ``predict_loo`` on a
    real fit: applying ``(N/(N-1))^2`` here, as every other group in this
    walk does, overstated this group's contribution and was the whole of an
    earlier mismatch against ``predict_loo``.
    """
    value = np.broadcast_to(model.global_mean, s.shape)
    return _sse_against(n, s, q, value)


def _fold_into_parent(
    parent_level: Any,
    parent_of: NDArray,
    rest: NDArray,
    n_in: NDArray,
    s_in: NDArray,
    q_in: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Aggregate unanswered classes' moments onto their parent classes.

    ``np.add.at`` rather than ``bincount``: several child classes share one
    parent, and the accumulation is over a (n_classes, d) array.
    """
    n_parent = parent_level.count.shape[0]
    d = parent_level.mean.shape[1]
    n_out = np.zeros(n_parent)
    s_out = np.zeros((n_parent, d))
    q_out = np.zeros((n_parent, d))
    target = parent_of[rest]
    np.add.at(n_out, target, n_in[rest])
    np.add.at(s_out, target, s_in[rest])
    np.add.at(q_out, target, q_in[rest])
    return n_out, s_out, q_out


def _finish(
    model: Any,
    deepest: Any,
    sse: float,
    n_total: float,
    n_matched: float,
    thresholds: tuple[int, ...],
    *,
    depth: int,
) -> TrainStats:
    d = model.global_mean.shape[0]
    tss = float(n_total * np.sum(model.global_msd))
    within = float((deepest.count[:, None] * deepest.msd).sum())
    counts = deepest.count.astype(np.float64)
    populated = counts > 0
    return TrainStats(
        depth=depth,
        n_classes=int(populated.sum()),
        n_atoms=int(n_total),
        sse=sse,
        rmse=float(np.sqrt(sse / (n_total * d))),
        r_squared=float(1.0 - sse / tss) if tss > 0 else float("nan"),
        eta_squared=float(1.0 - within / tss) if tss > 0 else float("nan"),
        matched_fraction=float(n_matched / n_total),
        support_fractions={
            t: float(counts[populated & (counts < t)].sum() / n_total)
            for t in thresholds
        },
    )


def sieve_curve(
    model: Any,
    depths: list[int],
    *,
    loo: bool = False,
    thresholds: tuple[int, ...] = SUPPORT_THRESHOLDS,
) -> list[TrainStats]:
    """``sieve_train_stats`` at each depth, by truncating the fitted model.

    Truncation is exact -- WL refinement never looks ahead, so a depth-*D*
    fit's levels ``0..d`` are bit-for-bit what a native depth-*d* fit stored
    (``cv.truncate_model``). One merged model therefore gives the whole
    curve at the cost of one pass per depth over the class tables.
    """
    from experiments.cv import truncate_model

    out = []
    for depth in sorted(depths):
        out.append(
            sieve_train_stats(
                truncate_model(model, depth), loo=loo, thresholds=thresholds
            )
        )
    return out


def hose_train_stats(
    state: Any,
    radius: int,
    *,
    n_min: int = 1,
    loo: bool = False,
    thresholds: tuple[int, ...] = SUPPORT_THRESHOLDS,
) -> TrainStats:
    """Training statistics for one HOSE radius, from its own key table.

    Restricted to the baseline ``n_min=1``, where the deepest key always
    answers a training atom -- it contributed to that key -- so there is no
    backoff and the walk collapses to a single level. Generalizing would mean
    implementing prefix backoff over the key tables, i.e. analysing a method
    nobody runs (spec section 2).

    ``loo=True`` is refused; see the error it raises.
    """
    if loo:
        raise NotImplementedError(
            "analytic LOO is not available for HOSE: removing an atom empties "
            "its own deepest key, and the predictor then backs off to a "
            "SHALLOWER RADIUS by sphere prefix, which this walk does not "
            "implement. An earlier version sent such atoms to the global mean "
            "instead -- mirroring Sieve's fallback, which only fires once its "
            "backoff chain is exhausted -- and that is not what predict does. "
            "The error is not small: at radius 8, 51.6% of training atoms sit "
            "in singleton keys, so the curve climbed to the global-mean error "
            "and its argmin was an artifact"
        )
    if n_min != 1:
        raise NotImplementedError(
            f"analytic HOSE statistics assume the baseline n_min=1, got "
            f"{n_min}: at a higher threshold a training atom backs off to "
            f"its own sphere prefix, which this does not implement"
        )
    if not state.has_second_moment:
        raise ValueError(
            "this HOSE state predates the sumsq column and carries no second "
            "moment, so its training error is not recoverable -- refit to "
            "use the analytic curve"
        )
    table = state.tables[radius]
    if not table:
        raise ValueError(f"state has no keys at radius {radius}")

    n = np.array([c for _, _, c in table.values()], dtype=np.float64)
    s = np.array([v for v, _, _ in table.values()], dtype=np.float64)[:, None]
    q = np.array([qq for _, qq, _ in table.values()], dtype=np.float64)[:, None]
    value = s / n[:, None]
    n_total = float(n.sum())

    if not loo:
        sse = _sse_against(n, s, q, value)
        n_matched = n_total  # n_min=1: every training atom answers here
    else:
        supported = n >= 2
        sse = 0.0
        n_matched = 0.0
        if supported.any():
            scale = _loo_scale(n[supported])
            sse += _sse_against(
                n[supported],
                s[supported],
                q[supported],
                value[supported],
                scale=scale,
            )
            n_matched += float(n[supported].sum())
        singleton = ~supported
        if singleton.any():
            gvalue = np.broadcast_to(state.global_mean, s[singleton].shape)
            sse += _sse_against(n[singleton], s[singleton], q[singleton], gvalue)
            n_matched += float(n[singleton].sum())

    # Always the plain (non-LOO) partition, matching sieve_train_stats' own
    # eta^2: it describes the partition, not whichever estimator was asked
    # for.
    within = _sse_against(n, s, q, value)
    tss = float(state.global_sumsq - state.global_sum**2 / state.global_count)
    populated = n > 0

    return TrainStats(
        depth=radius,
        n_classes=int(populated.sum()),
        n_atoms=int(n_total),
        sse=sse,
        rmse=float(np.sqrt(sse / n_total)),
        r_squared=float(1.0 - sse / tss) if tss > 0 else float("nan"),
        eta_squared=float(1.0 - within / tss) if tss > 0 else float("nan"),
        matched_fraction=float(n_matched / n_total),
        support_fractions={
            t: float(n[populated & (n < t)].sum() / n_total) for t in thresholds
        },
    )


def hose_curve(
    state: Any,
    radii: list[int],
    *,
    n_min: int = 1,
    loo: bool = False,
    thresholds: tuple[int, ...] = SUPPORT_THRESHOLDS,
) -> list[TrainStats]:
    """``hose_train_stats`` at each radius.

    Unlike ``sieve_curve`` there is no truncation: a HOSE state's tables at
    different radii are independently generated linearizations
    (``hose_artifact``'s own module docstring), not a prefix relationship, so
    each radius is read from the state's own table for that radius directly.
    """
    return [
        hose_train_stats(state, r, n_min=n_min, loo=loo, thresholds=thresholds)
        for r in sorted(radii)
    ]
