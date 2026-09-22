"""Content-rank fingerprints and the cis/trans code built on them.

Kept out of ``refine.py`` because it has its own tests and ``refine`` is one
readable function; ``refine`` calls in and folds the result.
"""

from __future__ import annotations

import numpy as np

from sieve.batch import CSRLayout
from sieve.dedupe import _row_keys


def _mix(*columns: np.ndarray) -> np.ndarray:
    """Mix several int64/uint64 columns into one uint64 per row.

    Every column is reinterpreted (``.view``, not ``.astype``) to ``int64``
    before stacking, never converted: ``np.stack`` promotes a mix of int64
    and uint64 arrays to float64 under NumPy's own casting rules, which would
    corrupt the bits ``_row_keys`` needs to mix. ``.view`` changes none of
    them, so every column keeps its 64 raw bits.

    Reuses ``dedupe._row_keys`` rather than a second mixer so the birthday
    argument ``dense_rows`` already makes for itself covers these keys too.
    """
    stacked = np.stack([c.view(np.int64) for c in columns], axis=1)
    return _row_keys(stacked)


def content_ranks(
    node_attrs: np.ndarray,
    csr: CSRLayout,
    edge_code: np.ndarray,
    n_rounds: int,
) -> list[np.ndarray]:
    """Radius-resolved content fingerprints, ``fp[j]`` for radius ``j``.

    ``fp_0`` is the attribute row; ``fp_j`` mixes ``fp_{j-1}`` of the atom with
    the sorted multiset of its neighbors' ``fp_{j-1}`` paired with the bond.
    A fingerprint ``fp_j`` is a function of the atom's radius-*j* environment
    and nothing else, so two atoms with the same environment share it in any
    batch, and adding molecules to the batch cannot change it. Class ids do
    not have that property: ``dense_rows`` numbers them by hashing the row and
    its docstring states that no caller may rely on that numbering.

    ``edge_code`` is the **static** edge alphabet, never a stereo code. A
    stereo-aware fingerprint would let the code reorder the substituents it is
    ranking, so remapping or mirroring could change which substituent wins and
    a merged model would disagree with a whole one.

    Padding uses ``0``, not ``-1``: fingerprints are already ``uint64`` hashes
    with no sign bit to spend on a sentinel, and a padding slot colliding with
    a real hash of exactly zero is the same order of risk ``dense_rows``
    already accepts for its own 64-bit keys. ``0`` sorts first, so a lower
    degree still yields more leading zeros than a higher one and the two
    remain distinguishable, matching the reason ``refine.py`` pads with -1.
    """
    n = node_attrs.shape[0]
    fp = [_row_keys(node_attrs)]
    width = max(int(csr.max_deg), 1)
    for _ in range(n_rounds):
        prev = fp[-1]
        nb = _mix(prev[csr.dst], edge_code)
        pad = np.zeros((n, width), np.uint64)
        pad[csr.src, csr.slot] = nb
        pad.sort(axis=1)
        fp.append(_mix(prev, *(pad[:, j] for j in range(width))))
    return fp


CODE_NONE, CODE_CIS, CODE_TRANS = 0, 1, 2


def cis_trans_codes(stereo_bonds: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """One code in ``{none, cis, trans}`` per stereogenic double bond.

    At each end the substituent with the larger fingerprint wins; equal
    fingerprints mean the bond is not distinguishable at this radius and the
    feature defers rather than guessing. The stored relation holds between the
    *first* controlling atom of each end, so it flips exactly when one winner
    -- and not both -- is the second.
    """
    a1, a2, b1, b2, cis = (stereo_bonds[:, j] for j in (2, 3, 4, 5, 6))
    zero = np.uint64(0)

    def winner_is_first(first: np.ndarray, second: np.ndarray):
        has = second >= 0
        f1 = fp[first]
        f2 = np.where(has, fp[np.where(has, second, 0)], zero)
        return (~has) | (f1 > f2), has & (f1 == f2)

    a_first, a_tie = winner_is_first(a1, a2)
    b_first, b_tie = winner_is_first(b1, b2)
    flipped = a_first ^ b_first
    same_side = cis.astype(bool) ^ flipped
    return np.where(
        a_tie | b_tie, CODE_NONE, np.where(same_side, CODE_CIS, CODE_TRANS)
    ).astype(np.int64)


def directed_positions(
    csr: CSRLayout, n_nodes: int, stereo_bonds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Positions of each stereo bond's two directed halves in CSR order.

    Resolved once, before the round loop: the bond set never changes, only
    the code on it does. Positions index the CSR-ordered edge arrays, which
    is the order ``refine`` folds the codes into.
    """
    key = csr.src * n_nodes + csr.dst
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    def locate(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        want = u * n_nodes + v
        pos = np.searchsorted(sorted_key, want)
        if (pos >= sorted_key.shape[0]).any() or (sorted_key[pos] != want).any():
            raise ValueError("stereo_bonds names a pair that is not an edge")
        return order[pos]

    a, b = stereo_bonds[:, 0], stereo_bonds[:, 1]
    return locate(a, b), locate(b, a)
