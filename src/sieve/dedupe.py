"""Row deduplication through a 64-bit row key (design.md 7.2)."""

from __future__ import annotations

import numpy as np

_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def _row_keys(m: np.ndarray) -> np.ndarray:
    """Mix each row down to one uint64: FNV-1a over the row's 8-byte lanes,
    then a splitmix64 finalizer.

    The accumulation loop alone mixes poorly at the bottom of the word, because
    multiplication only carries bits upward: bit 0 of the result comes out as
    exactly the XOR of bit 0 of every column, and the *last* column passes
    through a single multiply, so flipping its low bit moved only ~9 of 64 key
    bits. The finalizer's shift-xor-multiply rounds push every input bit across
    the whole word, which is what lets the 64-bit birthday bound in
    ``dense_rows`` be taken at face value. It is one pass over the keys, so it
    costs a few percent of a call that is otherwise dominated by the sort.
    """
    h = np.full(m.shape[0], _FNV_OFFSET, np.uint64)
    for j in range(m.shape[1]):
        h ^= m[:, j].view(np.uint64)
        h *= _FNV_PRIME
    h ^= h >> _S30
    h *= _MIX_A
    h ^= h >> _S27
    h *= _MIX_B
    h ^= h >> _S31
    return h


def _group(keys: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number the distinct values of ``keys``, carrying ``m``'s rows along.

    Representatives come back by scatter rather than by first-occurrence index:
    every row of a class writes identical bytes, so the last write wins with
    nothing lost -- and not asking ``np.unique`` for ``return_index`` keeps it
    off the stable sort that answer would require (~28% dearer here).
    """
    uniq, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    rows = np.empty((uniq.shape[0], m.shape[1]), m.dtype)
    rows[inv] = m
    return inv.astype(np.int64, copy=False), rows


def _dense_rows_exact(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group on the raw row bytes. Exact by construction, and slower.

    ``np.unique(..., axis=0)`` is *no faster than a Python dict loop* (0.70 s
    vs 0.71 s over 6 levels of 147k atoms); viewing each row as one ``np.void``
    scalar does the same work in 0.27 s.
    """
    v = m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).ravel()
    return _group(v, m)


def dense_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each distinct row of ``mat`` a dense id.

    Returns ``(labels, unique_rows)`` with ``unique_rows[labels] == mat``.

    Grouping is by a 64-bit key per row, not by the row bytes themselves.
    Sorting ``np.void`` rows makes numpy call a byte-comparison function for
    every comparison and shuffle a full row's worth of bytes per move; sorting
    one ``uint64`` per row hits its type-specialized path over an eighth of the
    data, which is ~4x faster across this corpus's levels -- and deduplication
    is the majority of both ``fit`` and ``predict``.

    Equal keys are only *evidence* of equal rows. A collision would fuse two
    distinct environments into one class and never raise, so the grouping is
    verified against the actual rows and falls back to the exact byte path if
    it does not hold. The check is one vectorized comparison and is included
    in the speedup above.

    Ids are numbered by key, so they are **not** ordered by row content. The
    partition and ``unique_rows[labels] == mat`` are the contract; the specific
    numbering is not, and no caller may rely on it (``merge._lookup_rows``
    builds its own ordering when it needs one).
    """
    m = np.ascontiguousarray(mat)
    if m.dtype.itemsize != 8:
        return _dense_rows_exact(m)  # the key mixer reads 8-byte lanes
    labels, rows = _group(_row_keys(m), m)
    if not np.array_equal(rows[labels], m):
        return _dense_rows_exact(m)
    return labels, rows
