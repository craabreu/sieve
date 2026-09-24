"""Content-rank fingerprints and the cis/trans and tetrahedral codes built
on them.

Kept out of ``refine.py`` because it has its own tests and ``refine`` is one
readable function; ``refine`` calls in and folds the result.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sieve.batch import CSRLayout
from sieve.dedupe import _row_keys

#: Neighbor slots hashed into every fingerprint, whatever the batch holds.
#: Twice the valence of any atom in the DASH store (its maximum is 5, a
#: handful of pentavalent phosphorus), and above every common coordination.
CONTENT_RANK_WIDTH = 8


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
) -> Iterator[np.ndarray]:
    """Radius-resolved content fingerprints, ``fp_0 .. fp_{n_rounds}``.

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

    **Every row pads to ``CONTENT_RANK_WIDTH``, not to the batch's maximum
    degree.** The padded row is hashed as a whole, so every slot, zeros
    included, enters the fingerprint. Padding to ``csr.max_deg`` once let one
    pentavalent phosphorus change the fingerprints, and hence the cis/trans
    codes, of every molecule batched beside it. ``refine.py`` can pad its own
    rows to the batch maximum because ``merge._widen`` left-pads them to a
    common width before comparing; nothing re-pads a hash. A batch holding an
    atom of higher degree is refused rather than truncated.

    **Yields rather than returning a list.** ``refine`` reads ``fp_j`` at WL
    round ``j + 2``, so ``j`` rises by exactly one per round: the
    fingerprints are consumed strictly in order, one at a time, and nothing
    ever indexes backwards. Retaining all of them costs ~1.6 GB on the
    38.9M-atom train split at depth 6, against ~310 MB for the single array
    actually in use. A caller that genuinely wants them all can still say
    ``list(content_ranks(...))``.
    """
    width = CONTENT_RANK_WIDTH
    if n_rounds > 0 and csr.max_deg > width:
        raise ValueError(
            f"an atom has degree {int(csr.max_deg)}, above the "
            f"{width} neighbor slots every content fingerprint hashes; "
            "raise sieve.stereo.CONTENT_RANK_WIDTH, which changes every "
            "fingerprint and so invalidates every stereo-aware model"
        )
    n = node_attrs.shape[0]
    fp = _row_keys(node_attrs)
    yield fp
    for _ in range(n_rounds):
        nb = _mix(fp[csr.dst], edge_code)
        pad = np.zeros((n, width), np.uint64)
        pad[csr.src, csr.slot] = nb
        pad.sort(axis=1)
        fp = _mix(fp, *(pad[:, j] for j in range(width)))
        yield fp


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


CODE_PLUS, CODE_MINUS = 1, 2  # tetrahedral: parity times the sorting sign


def tetrahedral_codes(stereo_centres: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """One code in ``{none, plus, minus}`` per tetrahedral centre.

    The sign is the tag's parity times the sign of the permutation that sorts
    the present neighbours by fingerprint, counted as inversions in reference
    order. An absent fourth position contributes no inversion: it sits last
    in RDKit's reference order, and any fixed place would flip every such
    sign alike. Two present neighbours with equal fingerprints make the
    centre unresolvable at this radius, and the code defers to ``none``.
    """
    nb = stereo_centres[:, 1:5]
    present = nb >= 0
    f = np.where(present, fp[np.where(present, nb, 0)], np.uint64(0))
    tie = np.zeros(nb.shape[0], bool)
    odd = np.zeros(nb.shape[0], bool)
    for i in range(4):
        for j in range(i + 1, 4):
            both = present[:, i] & present[:, j]
            tie |= both & (f[:, i] == f[:, j])
            odd ^= both & (f[:, i] > f[:, j])
    sign = np.where(odd, -1, 1) * stereo_centres[:, 5]
    return np.where(tie, CODE_NONE, np.where(sign > 0, CODE_PLUS, CODE_MINUS)).astype(
        np.int64
    )


def mirror_codes(codes: np.ndarray) -> np.ndarray:
    """The codes of the mirror image: plus and minus swap, none stays."""
    out = codes.copy()
    out[codes == CODE_PLUS] = CODE_MINUS
    out[codes == CODE_MINUS] = CODE_PLUS
    return out


def centre_positions(
    csr: CSRLayout, n_nodes: int, stereo_centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CSR positions of every out-edge of a centre, and that centre's row.

    The code rides on all of a centre's out-edges, so it enters the centre's
    own signature row and no other; resolved once, before the round loop.
    """
    row_of = np.full(n_nodes, -1, np.int64)
    row_of[stereo_centres[:, 0]] = np.arange(stereo_centres.shape[0])
    pos = np.flatnonzero(row_of[csr.src] >= 0)
    return pos, row_of[csr.src[pos]]


class FingerprintWindow:
    """Serves ``fp_r`` and ``fp_{r-1}`` from a generator consumed in order.

    With both tracks on, round *k* reads ``fp_{k-1}`` (tetrahedral) and
    ``fp_{k-2}`` (cis/trans). ``content_ranks`` yields strictly in order and
    retaining every radius costs ~1.6 GB on the train split, so this keeps
    one fingerprint of history and refuses anything older.
    """

    def __init__(self, fingerprints: Iterator[np.ndarray]) -> None:
        self._it = fingerprints
        self._radius = -1
        self._cur: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    def at(self, radius: int) -> np.ndarray:
        while self._radius < radius:
            self._prev, self._cur = self._cur, next(self._it)
            self._radius += 1
        if radius == self._radius and self._cur is not None:
            return self._cur
        if radius == self._radius - 1 and self._prev is not None:
            return self._prev
        raise ValueError(
            f"radius {radius} is no longer held (window at {self._radius})"
        )


_EMPTY = np.iinfo(np.int64).max


@dataclass(frozen=True)
class Reach:
    """Which stereo codes have reached each atom's aware row by some round.

    Tracks, per atom, the smallest and largest id of the tetrahedral centres
    whose codes have reached it, whether two or more distinct centres have,
    and whether any cis/trans code has. Two ids suffice for the one question
    asked -- one centre or several -- and make the union exact: a single
    centre reached along two paths (a ring) stays one centre.
    """

    first: np.ndarray  # (n,) int64, smallest centre id reached; _EMPTY if none
    last: np.ndarray  # (n,) int64, largest centre id reached; -1 if none
    many: np.ndarray  # (n,) bool, two or more distinct centres reached
    ez: np.ndarray  # (n,) bool, a cis/trans code reached

    @property
    def informative(self) -> np.ndarray:
        """Where the aware row carries something the mirror quotient keeps:
        the relative configuration of two centres, or a cis/trans code. A
        single centre's sign alone is erased by the quotient, since the class
        and its mirror are pooled."""
        return self.many | self.ez


def advance_reach(
    prev: Reach | None,
    csr: CSRLayout,
    n_nodes: int,
    tet_atoms: np.ndarray,
    ez_atoms: np.ndarray,
) -> Reach:
    """The reach one WL round later.

    An atom's aware row at round *k* holds its own row's codes of this round
    (``tet_atoms``: centres whose code fired; ``ez_atoms``: ends of bonds
    whose cis/trans code fired) plus its own and its neighbours' rows of
    round *k* - 1, which is exactly the union taken here.
    """
    if prev is None:
        first = np.full(n_nodes, _EMPTY, np.int64)
        last = np.full(n_nodes, -1, np.int64)
        many = np.zeros(n_nodes, bool)
        ez = np.zeros(n_nodes, bool)
    else:
        first, last = prev.first.copy(), prev.last.copy()
        many, ez = prev.many.copy(), prev.ez.copy()
        np.minimum.at(first, csr.src, prev.first[csr.dst])
        np.maximum.at(last, csr.src, prev.last[csr.dst])
        np.logical_or.at(many, csr.src, prev.many[csr.dst])
        np.logical_or.at(ez, csr.src, prev.ez[csr.dst])
    first[tet_atoms] = np.minimum(first[tet_atoms], tet_atoms)
    last[tet_atoms] = np.maximum(last[tet_atoms], tet_atoms)
    ez[ez_atoms] = True
    many |= (last >= 0) & (first != last)
    return Reach(first, last, many, ez)
