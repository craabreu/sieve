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
    """
    m = np.ascontiguousarray(mat)
    v = m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).ravel()
    # numpy returns (unique, index, inverse) in a FIXED order, whatever order
    # the keywords are passed in. Binding these wrongly is silent: the ids stay
    # plausible and `unique_rows[labels] == mat` quietly stops holding.
    uniq, idx, inv = np.unique(v, return_index=True, return_inverse=True)
    return inv.ravel().astype(np.int64), m[idx]
