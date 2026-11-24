"""Sieve's class estimators read over the classes of a HOSE tree state.

The HOSE lookup answers an atom with the raw mean of the deepest class it
matches. Sieve answers it from the same deepest class, but with the
continuation estimate of eq. (continuation) of the paper and, optionally,
empirical-Bayes shrinkage toward the parent (sieve.continuation,
sieve.shrinkage). Both read only per-class sufficient statistics and the
parent of every class, so they apply to any nested chain of classes, and the
classes of a tree state form one: a key at ``k`` spheres is cut to its parent
at ``k - 1`` by ``sphere_prefix``, and a one-sphere key has the class of its
central atom (``HoseState.roots``) as parent.

This module lays a tree state out as a ``SieveModel`` with one attribute
level (element and formal charge) and ``max_radius`` refinement levels, and
lets Sieve's own functions compute the estimates, so that no formula is
written twice. On plain trees the chain is exactly that of the Sieve label
with element and formal charge on the atoms and bond types on the bonds, and
the estimates are that model's; on closure trees the chain differs from it by
the ring closure alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.predictors.hose_keys import sphere_prefix

RAW = "raw"
CONTINUATION = "continuation"
CONTINUATION_EB = "continuation-eb"
ESTIMATORS: tuple[str, ...] = (RAW, CONTINUATION, CONTINUATION_EB)


def check_estimator(estimator: str) -> str:
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    return estimator


def _level(keys: list[str], stats: list[tuple[float, float, int]], parent: np.ndarray):
    from sieve.level import FrozenLevel

    s = np.array([t[0] for t in stats], np.float64)
    qq = np.array([t[1] for t in stats], np.float64)
    c = np.array([t[2] for t in stats], np.int64)
    mean = s / c
    msd = np.maximum(qq / c - mean**2, 0.0)
    return FrozenLevel(
        signatures=np.zeros((len(keys), 1), np.int64),
        count=c,
        mean=mean[:, None],
        msd=msd[:, None],
        parent=parent.astype(np.int32),
    )


def class_estimates(state: Any, estimator: str) -> list[dict[str, float]]:
    """``out[k][key]``: the estimate of class ``key`` at ``k`` spheres, ``k =
    0 .. state.max_radius``, level 0 keyed by the roots. ``k = max_radius``
    is the deepest level, which keeps its pooled mean under continuation."""
    import sieve
    from sieve.config import SieveConfig
    from sieve.shrinkage import shrunk_means

    check_estimator(estimator)
    if estimator == RAW:
        raise ValueError("the raw estimator reads the tables directly")
    if state.roots is None:
        raise ValueError(
            "this state carries no roots, the class of each one-sphere key's "
            "central atom, which Sieve's estimators need; refit its shards"
        )
    if not state.has_second_moment:
        raise ValueError("Sieve's estimators need the second moment of the classes")
    radius = state.max_radius

    # level 0: sums of the one-sphere statistics by root
    level0: dict[str, tuple[float, float, int]] = {}
    for key, (s, qq, c) in state.tables[1].items():
        ps, pqq, pc = level0.get(state.roots[key], (0.0, 0.0, 0))
        level0[state.roots[key]] = (ps + s, pqq + qq, pc + c)
    keys = [sorted(level0)] + [sorted(state.tables[k]) for k in range(1, radius + 1)]
    index = [{key: i for i, key in enumerate(ks)} for ks in keys]
    levels = [
        _level(
            keys[0],
            [level0[key] for key in keys[0]],
            np.full(len(keys[0]), -1),
        )
    ]
    for k in range(1, radius + 1):
        if k == 1:
            parent = [index[0][state.roots[key]] for key in keys[1]]
        else:
            parent = [index[k - 1][sphere_prefix(key, k - 1)] for key in keys[k]]
        levels.append(
            _level(keys[k], [state.tables[k][key] for key in keys[k]], np.array(parent))
        )

    config = SieveConfig(
        target_dim=1,
        attribute_levels=(("element", "formal_charge"),),
        attribute_codes={"element": {}, "formal_charge": {}},
        edge_codes={"bond_type": {}},
        max_wl_depth=radius,
        class_estimator="continuation",
        shrinkage_weight="empirical_bayes" if estimator == CONTINUATION_EB else None,
    )
    mean = state.global_sum / state.global_count
    model = sieve.SieveModel(
        config=config,
        levels=tuple(levels),
        global_count=state.global_count,
        global_mean=np.array([mean]),
        global_msd=np.array(
            [max(state.global_sumsq / state.global_count - mean**2, 0.0)]
        ),
    )
    est = shrunk_means(model)
    return [
        {key: float(v) for key, v in zip(keys[k], est[k][:, 0], strict=True)}
        for k in range(radius + 1)
    ]
