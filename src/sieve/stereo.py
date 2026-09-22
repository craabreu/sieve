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
