import numpy as np
from wllr.dedupe import dense_rows

def test_identical_rows_share_an_id():
    m = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], np.int64)
    labels, uniq = dense_rows(m)
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert uniq.shape[0] == 3

def test_labels_are_dense_from_zero():
    m = np.array([[9, 9], [4, 4], [9, 9], [1, 1]], np.int64)
    labels, uniq = dense_rows(m)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert labels.dtype == np.int64

def test_unique_rows_are_indexed_by_label():
    """The representative row for class j must actually be a row of class j.

    This is the assertion that catches the np.unique return-order trap: if
    `index` and `inverse` are swapped, uniq[labels[i]] stops equalling m[i].
    """
    rng = np.random.default_rng(0)
    m = rng.integers(0, 5, size=(500, 4)).astype(np.int64)
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)

def test_matches_a_dict_based_reference():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 3, size=(200, 3)).astype(np.int64)
    labels, _ = dense_rows(m)
    seen, ref = {}, []
    for row in map(tuple, m):
        ref.append(seen.setdefault(row, len(seen)))
    # ids may be numbered differently, but the partition must be identical
    assert _same_partition(labels, np.array(ref))

def _same_partition(a, b):
    return len({(int(x), int(y)) for x, y in zip(a, b)}) == len(set(a.tolist())) == len(set(b.tolist()))

def test_handles_non_contiguous_input():
    m = np.arange(40, dtype=np.int64).reshape(10, 4)[:, ::2]
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)
