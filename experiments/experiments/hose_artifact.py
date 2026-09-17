"""Sufficient statistics for the HOSE-code lookup arm, and their exact merge.

The counterpart of ``tree_artifact.py`` for DASH and of ``sieve.merge`` for
Sieve: what one shard's fit writes to disk, and how several shards' writes
combine into the model a CV fold trains on.

A HOSE fit is a table per sphere count *k*, mapping a *k*-sphere key to the
mean and count of the training atoms carrying it. Stored here as **sum and
count** instead, because those are what add: two shards' tables merge by
summing both columns key-wise, and the mean is recovered at the end. Storing
means would force a weighted average at every merge step and lose exactness
to rounding across a 40-shard fold.

One state belongs to exactly one ``max_radius``. Unlike DASH's node stats and
Sieve's levels, a deeper state does NOT contain the shallower ones: HOSE codes
are a linearization whose sphere ordering consults what lies beyond it, so
generating deeper re-renders shallower spheres (8.5% of prefixes differ --
docs/superpowers/specs/2026-09-16-hose-baseline-design.md section 7). Merging
two states of different radii is therefore refused rather than reconciled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HoseState:
    """``tables[k]`` maps a *k*-sphere key to ``(sum, count)``, for
    ``k = 1 .. max_radius``. ``global_sum``/``global_count`` are the whole
    training split's, used when no key at any radius answers."""

    max_radius: int
    tables: tuple[dict[str, tuple[float, int]], ...]  # index k, [0] unused
    global_sum: float
    global_count: int

    def __post_init__(self) -> None:
        if len(self.tables) != self.max_radius + 1:
            raise ValueError(
                f"expected {self.max_radius + 1} tables (index 0 unused), "
                f"got {len(self.tables)}"
            )

    @property
    def global_mean(self) -> float:
        if self.global_count == 0:
            raise ValueError("no training atoms, so no global mean")
        return self.global_sum / self.global_count


def merge_hose_states(a: HoseState, b: HoseState) -> HoseState:
    """Key-wise sum of two states fitted at the SAME radius, exact.

    Refuses differing radii: a radius-6 state's 3-sphere table is not a
    radius-3 state's, so combining them would silently pool keys cut from
    two different linearizations (spec section 7).
    """
    if a.max_radius != b.max_radius:
        raise ValueError(
            f"cannot merge HOSE states fitted at different radii "
            f"({a.max_radius} and {b.max_radius}): a deeper fit's shallower "
            f"spheres are re-rendered, not shared (spec section 7)"
        )
    tables: list[dict[str, tuple[float, int]]] = [{}]
    for k in range(1, a.max_radius + 1):
        merged = dict(a.tables[k])
        for key, (s, c) in b.tables[k].items():
            if (prev := merged.get(key)) is None:
                merged[key] = (s, c)
            else:
                merged[key] = (prev[0] + s, prev[1] + c)
        tables.append(merged)
    return HoseState(
        max_radius=a.max_radius,
        tables=tuple(tables),
        global_sum=a.global_sum + b.global_sum,
        global_count=a.global_count + b.global_count,
    )


def fold_hose_states(shards: Iterable[HoseState]) -> HoseState:
    """Left fold of ``merge_hose_states`` over several shards' states."""
    it = iter(shards)
    try:
        out = next(it)
    except StopIteration:
        raise ValueError("cannot fold an empty sequence of HOSE states") from None
    for s in it:
        out = merge_hose_states(out, s)
    return out


def save_hose_state(state: HoseState, path: str | Path) -> None:
    """One ``.npz``: per radius, a key array and its two statistic columns."""
    arrays: dict[str, NDArray] = {
        "max_radius": np.asarray(state.max_radius, dtype=np.int64),
        "global_sum": np.asarray(state.global_sum, dtype=np.float64),
        "global_count": np.asarray(state.global_count, dtype=np.int64),
    }
    for k in range(1, state.max_radius + 1):
        table = state.tables[k]
        keys = list(table)
        arrays[f"r{k}_keys"] = np.asarray(keys, dtype=np.str_)
        arrays[f"r{k}_sum"] = np.asarray(
            [table[key][0] for key in keys], dtype=np.float64
        )
        arrays[f"r{k}_count"] = np.asarray(
            [table[key][1] for key in keys], dtype=np.int64
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_hose_state(path: str | Path) -> HoseState:
    with np.load(path, allow_pickle=False) as z:
        max_radius = int(z["max_radius"])
        tables: list[dict[str, tuple[float, int]]] = [{}]
        for k in range(1, max_radius + 1):
            keys = z[f"r{k}_keys"]
            sums = z[f"r{k}_sum"]
            counts = z[f"r{k}_count"]
            tables.append(
                {
                    str(key): (float(s), int(c))
                    for key, s, c in zip(keys, sums, counts, strict=True)
                }
            )
        return HoseState(
            max_radius=max_radius,
            tables=tuple(tables),
            global_sum=float(z["global_sum"]),
            global_count=int(z["global_count"]),
        )
