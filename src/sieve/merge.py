"""The merge monoid (design.md 5).

Counts add; both moments are weighted averages. This is the law of total
variance in intensive variables -- Chan, Golub and LeVeque's parallel form --
not an ad hoc correction, which is why the full-covariance upgrade is a
one-term change and why merging is order-independent.
"""

from __future__ import annotations

import numpy as np

from sieve.config import SieveConfig, check_mergeable
from sieve.dedupe import dense_rows
from sieve.level import FrozenLevel

_OOV_NEIGHBOR = -2  # distinct from both a real code (>=0) and the pad sentinel (-1)


def _translate(
    sig: np.ndarray, remap_prev: np.ndarray | None, n_bond: int, is_wl: bool
) -> np.ndarray:
    """Rewrite B's signature rows in the merged id space.

    A signature row is written in terms of the *previous level's local ids*, so
    B's row [3, 7, 12] and A's row [3, 7, 12] generally denote different
    classes. Without this step the merge silently unions unrelated classes.

    ``remap_prev`` can contain -1 here (predict.py's lookup leaves an unmatched
    query class as -1; merge.py's own remaps never do). A neighbor with an
    OOV previous-level class must make the row unmatchable, not accidentally
    plausible: ``-1 * n_bond + bond`` lands on the real pad sentinel -1 itself
    whenever ``bond == n_bond - 1``, so an OOV neighbor reached by the
    top bond code was indistinguishable from a node with one fewer neighbor
    -- a false match at exactly the classes it should confidently miss.
    """
    if remap_prev is None:  # level 0: attribute codes are already global
        return sig
    out = sig.copy()
    out[:, 0] = remap_prev[sig[:, 0]]
    if is_wl and sig.shape[1] > 1:
        pad = sig[:, 1:]
        filled = pad >= 0
        lab, bond = np.divmod(np.where(filled, pad, 0), n_bond)
        remapped_lab = remap_prev[lab]
        oov = filled & (remapped_lab < 0)
        new = remapped_lab * n_bond + bond
        out[:, 1:] = np.where(oov, _OOV_NEIGHBOR, np.where(filled, new, -1))
        # Remapping changes the sort order, so the multiset must be
        # re-canonicalized or equal multisets stop comparing equal.
        out[:, 1:] = np.sort(out[:, 1:], axis=1)
    return out


def _widen(sig: np.ndarray, width: int, is_wl: bool) -> np.ndarray:
    """Pad a signature block out to `width` columns, keeping it comparable.

    Column 0 is a parent id for every level except attribute level 0. For WL
    levels the remaining columns are a *sorted* multiset, padded with -1 at
    the low end because a node's pad count is fixed and -1 sorts first
    (design.md 7.2) -- but that pad count is the *batch's* max degree, so two
    batches fitted separately generally have different widths for the same
    level. Right-padding a narrower row (the old behavior) shifts its real
    values off the columns the wider row's real values occupy, so equal
    classes stop comparing equal. Left-padding instead keeps every row's real
    values right-aligned, which is what -1-first sorting already assumes.
    """
    if sig.shape[1] == width:
        return np.ascontiguousarray(sig)
    out = np.full((sig.shape[0], width), -1, np.int64)
    if is_wl and sig.shape[1] > 1:
        out[:, 0] = sig[:, 0]
        pad = width - sig.shape[1]
        out[:, 1 + pad :] = sig[:, 1:]
    else:
        out[:, : sig.shape[1]] = sig
    return out


def _lookup_rows(sig: np.ndarray, table: np.ndarray, is_wl: bool) -> np.ndarray:
    """Row-wise membership: index of each `sig` row in `table`, or -1.

    Shared with ``predict.py``'s lookup -- same problem (find a signature row
    in a sorted vocabulary), so it must not grow a second implementation.
    """
    if table.shape[0] == 0 or sig.shape[0] == 0:
        return np.full(sig.shape[0], -1, np.int64)
    width = max(sig.shape[1], table.shape[1])
    s, t = _widen(sig, width, is_wl), _widen(table, width, is_wl)
    dt = np.dtype((np.void, s.dtype.itemsize * width))
    sv = s.view(dt).ravel()
    tv = t.view(dt).ravel()
    order = np.argsort(tv)
    pos = np.searchsorted(tv, sv, sorter=order)
    pos = np.clip(pos, 0, order.size - 1)
    cand = order[pos]
    hit = tv[cand] == sv
    return np.where(hit, cand, -1).astype(np.int64)


def merge_level(
    a: FrozenLevel,
    b: FrozenLevel,
    remap_prev: np.ndarray | None,
    n_bond: int,
    is_wl: bool,
) -> tuple[FrozenLevel, np.ndarray]:
    """Merge one level. A's ids are preserved; only B's are remapped.

    Pinning A means A's ``parent`` array needs no translation at all -- only
    B's, carried forward across levels by ``remap_prev``.

    A's rows are matched against B's by *lookup*, not by re-deduplicating the
    concatenation: ``dense_rows`` numbers by sorted order, so stacking A ahead
    of B and re-running it does **not** keep A's original ids when the two
    value ranges interleave -- despite reading that way from the "A comes
    first" framing. Looking B's rows up in A's table instead leaves A's ids
    untouched and only mints new ones for what A truly lacks.
    """
    d = a.mean.shape[1] if a.n_classes else b.mean.shape[1]
    width = max(a.signatures.shape[1], b.signatures.shape[1], 1)

    a_sig = _widen(a.signatures, width, is_wl)
    b_sig = _widen(_translate(b.signatures, remap_prev, n_bond, is_wl), width, is_wl)
    m = a.n_classes

    found = _lookup_rows(b_sig, a_sig, is_wl)  # -1 where B's row is new to A
    new_mask = found < 0
    if np.any(new_mask):
        new_local, new_uniq = dense_rows(b_sig[new_mask])
        remap_new = (new_local + m).astype(np.int32)
    else:
        new_uniq = np.zeros((0, width), np.int64)
        remap_new = np.zeros(0, np.int32)

    remap = np.empty(b.n_classes, np.int32)
    remap[~new_mask] = found[~new_mask].astype(np.int32)
    remap[new_mask] = remap_new

    n_new = m + new_uniq.shape[0]
    uniq = np.concatenate([a_sig, new_uniq], axis=0) if new_uniq.shape[0] else a_sig

    count = np.zeros(n_new, np.int64)
    mean = np.zeros((n_new, d), np.float64)
    msd = np.zeros((n_new, d), np.float64)
    parent = np.full(n_new, -1, np.int32)
    count[:m] = a.count
    mean[:m] = a.mean
    msd[:m] = a.msd
    parent[:m] = a.parent

    i = remap  # a bijection: no duplicate scatter writes
    nA = count[i].astype(np.float64)
    nB = b.count.astype(np.float64)
    n = nA + nB
    safe = np.maximum(n, 1.0)
    wA, wB = (nA / safe)[:, None], (nB / safe)[:, None]

    delta = b.mean - mean[i]  # MUST precede the mean update below
    msd[i] = wA * msd[i] + wB * b.msd + wA * wB * delta * delta
    mean[i] = wA * mean[i] + wB * b.mean
    count[i] = n.astype(np.int64)

    b_parent = (
        np.full(b.n_classes, -1, np.int32)
        if remap_prev is None
        else remap_prev[b.parent].astype(np.int32)
    )
    # For classes in both models the parent must already agree (design.md 2.1).
    # A real `raise`, not `assert`: this is a structural invariant the merge
    # relies on, and `assert` is compiled away under `python -O`.
    both = count[i] > nB
    if np.any(both) and not np.array_equal(parent[i][both], b_parent[both]):
        raise AssertionError("parent disagreement: the nesting invariant is broken")
    parent[i] = np.where(nA > 0, parent[i], b_parent)
    return FrozenLevel(uniq, count, mean, msd, parent), remap


def merge_models(a, b):
    """Merge two fitted models (design.md 5)."""
    from sieve.model import SieveModel

    check_mergeable(a.config, b.config)
    cfg: SieveConfig = a.config
    n_attr_levels = len(cfg.attribute_levels)

    levels, remap = [], None
    for k in range(cfg.n_levels):
        lvl, remap = merge_level(
            a.levels[k], b.levels[k], remap, cfg.n_bond, is_wl=k >= n_attr_levels
        )
        levels.append(lvl)

    nA, nB = float(a.global_count), float(b.global_count)
    n = nA + nB
    if n == 0:
        return SieveModel(cfg, tuple(levels), 0, a.global_mean, a.global_msd)
    wA, wB = nA / n, nB / n
    delta = b.global_mean - a.global_mean
    g_msd = wA * a.global_msd + wB * b.global_msd + wA * wB * delta * delta
    g_mean = wA * a.global_mean + wB * b.global_mean
    return SieveModel(cfg, tuple(levels), int(n), g_mean, g_msd)


def fold(models, config: SieveConfig):
    """Combine shards as a balanced tree (design.md 5.4).

    Each merge costs O(|A| + |B|), so a sequential reduce over N shards is
    O(N^2 s) -- the accumulator grows and is re-copied every step. Pairwise
    reduction is O(N s log N).
    """
    from sieve.model import SieveModel

    items = list(models)
    if not items:
        return SieveModel.empty(config)
    while len(items) > 1:
        items = [
            merge_models(items[i], items[i + 1]) if i + 1 < len(items) else items[i]
            for i in range(0, len(items), 2)
        ]
    return items[0]
