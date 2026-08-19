"""Row deduplication through a void view (design.md 7.2)."""

from __future__ import annotations

import numpy as np


def dense_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each distinct row of ``mat`` a dense id.

    Returns ``(labels, unique_rows)`` with ``unique_rows[labels] == mat``.

    Deduplication happens in 1-D by viewing each row as a single ``np.void``
    scalar. ``np.unique(..., axis=0)`` is *no faster than a Python dict loop*
    (0.70 s vs 0.71 s over 6 levels of 147k atoms); the void view does the same
    work in 0.27 s.

    ``return_index`` is deliberately not requested. Asking for it forces
    ``np.unique`` onto a *stable* sort (mergesort) so that ``index`` can mean
    "first occurrence", and that costs about 28% over the default quicksort on
    this workload -- for an answer this function does not need. Every row of a
    class is byte-identical by construction, so scattering all of them recovers
    the same representatives that picking the first one would.
    """
    m = np.ascontiguousarray(mat)
    v = m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).ravel()
    uniq, inv = np.unique(v, return_inverse=True)
    inv = inv.ravel()
    rows = np.empty((uniq.shape[0], m.shape[1]), m.dtype)
    rows[inv] = m  # every write within a class writes identical bytes
    return inv.astype(np.int64, copy=False), rows
