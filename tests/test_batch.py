import time

import numpy as np
import pytest

from sieve.batch import AtomBatch, check_alignment


def ring(n):
    """A single undirected cycle of `n` nodes, both directions stored."""
    fwd = np.arange(n, dtype=np.int64)
    nxt = (fwd + 1) % n
    return {
        "node_attrs": np.zeros((n, 1), np.int64),
        "edge_src": np.concatenate([fwd, nxt]),
        "edge_dst": np.concatenate([nxt, fwd]),
        "edge_attr": np.ones(2 * n, np.int64),
        "graph_id": np.zeros(n, np.int64),
        "y": None,
    }


def python_set_check(src, dst):
    """The reference both-direction check, written the obvious slow way.

    Used as a self-calibrating timing baseline so the performance assertions
    below mean the same thing on any machine.
    """
    fwd = {(int(a), int(b)) for a, b in zip(src, dst, strict=True)}
    return not any((b, a) not in fwd for a, b in fwd)


def tri():
    """Triangle 0-1-2 plus an isolated node 3, as two graphs."""
    return AtomBatch(
        node_attrs=np.array([[0], [1], [1], [0]], np.int64),
        edge_src=np.array([0, 1, 1, 2, 2, 0], np.int64),
        edge_dst=np.array([1, 0, 2, 1, 0, 2], np.int64),
        edge_attr=np.array([1, 1, 1, 1, 2, 2], np.int64),
        graph_id=np.array([0, 0, 0, 1], np.int64),
        y=np.array([[1.0], [2.0], [3.0], [4.0]]),
    )


def test_shapes_are_validated():
    with pytest.raises(ValueError, match="graph_id"):
        AtomBatch(
            node_attrs=np.zeros((3, 1), np.int64),
            edge_src=np.zeros(0, np.int64),
            edge_dst=np.zeros(0, np.int64),
            edge_attr=np.zeros(0, np.int64),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_edges_must_be_symmetric():
    with pytest.raises(ValueError, match="both directions"):
        AtomBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0], np.int64),
            edge_dst=np.array([1], np.int64),
            edge_attr=np.array([1], np.int64),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_edge_indices_must_be_within_the_node_range():
    """An index past the last atom currently surfaces as an IndexError deep
    inside csr(), or as silent garbage. It is also what makes the vectorized
    both-direction check's src*n+dst key a bijection, so it is checked here."""
    with pytest.raises(ValueError, match="edge index"):
        AtomBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, 5], np.int64),
            edge_dst=np.array([5, 0], np.int64),
            edge_attr=np.array([1, 1], np.int64),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_negative_edge_index_is_rejected():
    with pytest.raises(ValueError, match="edge index"):
        AtomBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, -1], np.int64),
            edge_dst=np.array([-1, 0], np.int64),
            edge_attr=np.array([1, 1], np.int64),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_validation_does_not_build_a_python_set():
    """The both-direction check used to cost more than an entire fit: it built
    a Python set of every edge tuple. It must be vectorized, so constructing a
    batch has to come in well under the cost of that set."""
    kw = ring(150_000)
    t0 = time.perf_counter()
    python_set_check(kw["edge_src"], kw["edge_dst"])
    reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    AtomBatch(**kw)
    construction = time.perf_counter() - t0

    assert construction < 0.5 * reference, (
        f"construction ({construction * 1000:.1f} ms) should be far under the "
        f"Python-set reference ({reference * 1000:.1f} ms)"
    )


def test_slicing_does_not_revalidate_edges():
    """``__getitem__`` keeps an edge only when both endpoints are selected, so
    (a,b) survives exactly when (b,a) does: the sub-batch is bidirectional by
    construction and re-checking it is pure overhead (design.md 11.1)."""
    batch = AtomBatch(**ring(150_000))
    mask = batch.graph_id == 0  # keep everything, so the edge work is maximal

    t0 = time.perf_counter()
    AtomBatch(**ring(150_000))
    full = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch[mask]
    sliced = time.perf_counter() - t0

    assert sliced < 0.5 * full, (
        f"slicing ({sliced * 1000:.1f} ms) still pays the construction-time "
        f"edge validation ({full * 1000:.1f} ms)"
    )


def test_a_sliced_batch_is_still_bidirectional():
    """The invariant that makes skipping revalidation on a slice safe."""
    batch = AtomBatch(**ring(50))
    keep = np.zeros(50, bool)
    keep[:30] = True  # cuts the ring, dropping the two edges across the cut
    sub = batch[keep]
    assert python_set_check(sub.edge_src, sub.edge_dst)


def test_parallel_edges_are_accepted():
    """Pins existing semantics: the check compares edge *sets*, so a repeated
    edge is not by itself a missing-reverse-direction error."""
    AtomBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([0, 0, 1], np.int64),
        edge_dst=np.array([1, 1, 0], np.int64),
        edge_attr=np.array([1, 1, 1], np.int64),
        graph_id=np.zeros(2, np.int64),
        y=None,
    )


def test_self_loop_is_its_own_reverse():
    AtomBatch(
        node_attrs=np.zeros((1, 1), np.int64),
        edge_src=np.array([0], np.int64),
        edge_dst=np.array([0], np.int64),
        edge_attr=np.array([1], np.int64),
        graph_id=np.zeros(1, np.int64),
        y=None,
    )


def test_a_batch_with_no_edges_is_valid():
    AtomBatch(
        node_attrs=np.zeros((3, 1), np.int64),
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attr=np.zeros(0, np.int64),
        graph_id=np.zeros(3, np.int64),
        y=None,
    )


def test_csr_slot_is_position_within_each_source_block():
    c = tri().csr()
    assert c.max_deg == 2
    # every node's slots are exactly 0..deg-1
    for node in range(4):
        got = sorted(c.slot[c.src == node].tolist())
        assert got == list(range(len(got)))
    assert c.indptr.tolist() == [0, 2, 4, 6, 6]
    assert np.all(np.diff(c.src) >= 0), "csr arrays must be sorted by source"


def test_isolated_node_has_degree_zero():
    c = tri().csr()
    assert c.indptr[4] - c.indptr[3] == 0


def test_alignment_guard_accepts_matching_corpus():
    b = tri()
    check_alignment(
        b, atom_counts=np.array([3, 1]), elements=np.array([6, 1, 1, 8], np.int64)
    )


def test_alignment_guard_catches_count_mismatch():
    b = tri()
    with pytest.raises(ValueError, match="atom count"):
        check_alignment(
            b, atom_counts=np.array([2, 1]), elements=np.array([6, 1, 1, 8], np.int64)
        )


def test_alignment_guard_catches_permutation():
    """The bug counts alone cannot catch: same atoms, wrong order."""
    b = AtomBatch(**{**tri().__dict__, "elements": np.array([6, 1, 1, 8], np.int64)})
    with pytest.raises(ValueError, match="element"):
        check_alignment(
            b, atom_counts=np.array([3, 1]), elements=np.array([1, 6, 1, 8], np.int64)
        )
