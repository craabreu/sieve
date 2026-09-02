import dataclasses
import pickle
import time

import numpy as np
import pytest

from sieve.batch import NodeBatch, check_alignment, concat_batches


def ring(n):
    """A single undirected cycle of `n` nodes, both directions stored."""
    fwd = np.arange(n, dtype=np.int64)
    nxt = (fwd + 1) % n
    return {
        "node_attrs": np.zeros((n, 1), np.int64),
        "edge_src": np.concatenate([fwd, nxt]),
        "edge_dst": np.concatenate([nxt, fwd]),
        "edge_attrs": np.ones(2 * n, np.int64).reshape(-1, 1),
        "graph_id": np.zeros(n, np.int64),
        "y": None,
    }


def best_of_paired(fn_a, fn_b, *, repeats=8):
    """The minimum CPU time of `fn_a()` and `fn_b()` each, over several
    repeats, interleaved (a, b, a, b, ...) rather than run as two separate
    back-to-back blocks.

    `perf_counter` measures wall-clock time, so it also counts every
    millisecond the process spends *descheduled* while a noisy neighbor on a
    shared CI runner gets the CPU instead. `process_time` counts only CPU
    time actually spent executing this process, so a scheduling gap costs it
    nothing -- but CPU time alone does not rule out running slower *while
    scheduled* under memory-bandwidth/frequency contention from a
    co-located tenant on shared CI hardware, which is a real risk whenever
    the two things being compared have different sensitivity to that (see
    `test_slicing_does_not_revalidate_edges`, the remaining user of this
    helper -- `test_validation_does_not_build_a_python_set` moved off
    timing entirely, to `tracemalloc` peak-memory tracking, once exactly
    this failure mode hit it twice in real CI). Interleaving the two
    functions' own timing loops at least correlates a short-lived
    contention episode across both measurements instead of letting it land
    unevenly on whichever one happened to be running during the dip -- run
    back-to-back as two separate blocks, the same episode can skew only one
    side and shift the ratio. The minimum-over-repeats is kept as a second
    line of defense against unrelated per-call noise (cache effects,
    allocator work).
    """
    best_a = best_b = float("inf")
    for _ in range(repeats):
        t0 = time.process_time()
        fn_a()
        best_a = min(best_a, time.process_time() - t0)

        t0 = time.process_time()
        fn_b()
        best_b = min(best_b, time.process_time() - t0)
    return best_a, best_b


def python_set_check(src, dst):
    """The reference both-direction check, written the obvious slow way.

    Used as a self-calibrating timing baseline so the performance assertions
    below mean the same thing on any machine.
    """
    fwd = {(int(a), int(b)) for a, b in zip(src, dst, strict=True)}
    return not any((b, a) not in fwd for a, b in fwd)


def tri():
    """Triangle 0-1-2 plus an isolated node 3, as two graphs."""
    return NodeBatch(
        node_attrs=np.array([[0], [1], [1], [0]], np.int64),
        edge_src=np.array([0, 1, 1, 2, 2, 0], np.int64),
        edge_dst=np.array([1, 0, 2, 1, 0, 2], np.int64),
        edge_attrs=np.array([1, 1, 1, 1, 2, 2], np.int64).reshape(-1, 1),
        graph_id=np.array([0, 0, 0, 1], np.int64),
        y=np.array([[1.0], [2.0], [3.0], [4.0]]),
    )


def test_shapes_are_validated():
    with pytest.raises(ValueError, match="graph_id"):
        NodeBatch(
            node_attrs=np.zeros((3, 1), np.int64),
            edge_src=np.zeros(0, np.int64),
            edge_dst=np.zeros(0, np.int64),
            edge_attrs=np.zeros(0, np.int64).reshape(-1, 1),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_edges_must_be_symmetric():
    with pytest.raises(ValueError, match="both directions"):
        NodeBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0], np.int64),
            edge_dst=np.array([1], np.int64),
            edge_attrs=np.array([1], np.int64).reshape(-1, 1),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_edge_indices_must_be_within_the_node_range():
    """An index past the last node currently surfaces as an IndexError deep
    inside csr(), or as silent garbage. It is also what makes the vectorized
    both-direction check's src*n+dst key a bijection, so it is checked here."""
    with pytest.raises(ValueError, match="edge index"):
        NodeBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, 5], np.int64),
            edge_dst=np.array([5, 0], np.int64),
            edge_attrs=np.array([1, 1], np.int64).reshape(-1, 1),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_negative_edge_index_is_rejected():
    with pytest.raises(ValueError, match="edge index"):
        NodeBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, -1], np.int64),
            edge_dst=np.array([-1, 0], np.int64),
            edge_attrs=np.array([1, 1], np.int64).reshape(-1, 1),
            graph_id=np.zeros(2, np.int64),
            y=None,
        )


def test_one_way_edges_are_caught_with_narrow_integer_dtypes():
    """The ``src * n + dst`` key is only a bijection when it cannot wrap.

    Computed in the array's own dtype it wraps mod 2**32 for int32 inputs, and
    then distinct edges can share a key. With ``n - 1 == 2**16`` the edge
    (65536, 0) collides with its own reverse, so a corpus missing that reverse
    passes a check whose entire job is to catch exactly that.
    """
    n = 65537
    with pytest.raises(ValueError, match="both directions"):
        NodeBatch(
            node_attrs=np.zeros((n, 1), np.int64),
            edge_src=np.array([65536], np.int32),
            edge_dst=np.array([0], np.int32),
            edge_attrs=np.ones(1, np.int64).reshape(-1, 1),
            graph_id=np.zeros(n, np.int64),
            y=None,
        )


def test_trusted_constructor_rejects_unknown_fields():
    """It bypasses __init__, so an unrecognized name would otherwise be set as
    a stray attribute rather than rejected."""
    with pytest.raises(TypeError, match="unexpected"):
        NodeBatch._with_trusted_edges(**{**ring(4), "elements": None, "bogus": 1})


def test_edge_attrs_must_be_two_dimensional():
    """edge_attrs is (n_edges, n_edge_attr) per design.md 11.1. A 1-D array is
    the pre-2026-09 shape and must raise rather than broadcast into a single
    column, which would silently accept a batch built against the old API."""
    with pytest.raises(ValueError, match="edge_attrs"):
        NodeBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, 1], np.int64),
            edge_dst=np.array([1, 0], np.int64),
            edge_attrs=np.ones(2, np.int64),  # 1-D: wrong
            graph_id=np.zeros(2, np.int64),
        )


def test_edge_attrs_supports_a_zero_width_schema():
    """An empty edge schema is legal (spec: edge_attributes == ()), so an
    (n_edges, 0) array must construct and slice cleanly rather than tripping
    a shape check written for the one-column case."""
    b = NodeBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([0, 1], np.int64),
        edge_dst=np.array([1, 0], np.int64),
        edge_attrs=np.zeros((2, 0), np.int64),
        graph_id=np.zeros(2, np.int64),
    )
    assert b.edge_attrs.shape == (2, 0)
    assert b[np.array([0, 1])].edge_attrs.shape == (2, 0)
    assert b.csr().attr.shape == (2, 0)


def test_validation_does_not_build_a_python_set():
    """The both-direction check used to cost more than an entire fit: it built
    a Python set of every edge tuple. It must be vectorized -- i.e. it must
    not materialize O(edges) Python-level objects (a set of int tuples) the
    way the reference implementation below does.

    That is a structural property of the algorithm, not a question of how
    fast today's hardware happens to run it, so it's tested directly via
    ``tracemalloc``'s peak-memory tracking rather than via wall/CPU time.
    An earlier version of this test compared elapsed time against the same
    reference and needed two rounds of fixes (wall-clock -> CPU time, then
    interleaved measurements with a loosened threshold) because CPU time
    only rules out being *descheduled*, not running slower *while
    scheduled* under memory-bandwidth/frequency contention from a
    co-located tenant on shared CI hardware -- and the two workloads here
    (vectorized numpy vs. a Python set of boxed int tuples) plausibly have
    different sensitivity to that, so the timing ratio itself could shift
    under contention, not just get noisier around a fixed mean (two real CI
    failures measured 0.58 and 0.66 against a quiet-machine baseline of
    ~0.35-0.37). Peak allocated memory has no such dependency: it counts
    what got allocated, not how long the CPU took to do it, so it can't
    flake this way at all -- confirmed empirically (~9.6 MB vectorized vs.
    ~44 MB for the reference on a 150k-node ring, a ratio of ~0.22).
    """
    import tracemalloc

    kw = ring(150_000)
    NodeBatch(**kw)  # warm up allocator pools before either measurement
    python_set_check(kw["edge_src"], kw["edge_dst"])

    tracemalloc.start()
    NodeBatch(**kw)
    _, peak_construction = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    python_set_check(kw["edge_src"], kw["edge_dst"])
    _, peak_reference = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_construction < 0.5 * peak_reference, (
        f"construction ({peak_construction / 1e6:.2f} MB peak) should use "
        f"far less memory than the Python-set reference "
        f"({peak_reference / 1e6:.2f} MB peak)"
    )


def test_slicing_does_not_revalidate_edges():
    """``__getitem__`` keeps an edge only when both endpoints are selected, so
    (a,b) survives exactly when (b,a) does: the sub-batch is bidirectional by
    construction and re-checking it is pure overhead (design.md 11.1)."""
    batch = NodeBatch(**ring(150_000))
    mask = batch.graph_id == 0  # keep everything, so the edge work is maximal

    full, sliced = best_of_paired(
        lambda: NodeBatch(**ring(150_000)),
        lambda: batch[mask],
    )

    assert sliced < 0.5 * full, (
        f"slicing ({sliced * 1000:.1f} ms) still pays the construction-time "
        f"edge validation ({full * 1000:.1f} ms)"
    )


def test_slicing_carries_every_field():
    """``_with_trusted_edges`` sets fields by name onto a bare instance, so a
    dropped one does not raise: every optional field has a dataclass default,
    which is a *class* attribute, and the instance silently reads through to it.
    ``elements`` quietly becoming None would disable the alignment guard
    (design.md 11.3) rather than fail.
    """
    kw = ring(40)
    kw["y"] = np.arange(40, dtype=np.float64).reshape(-1, 1)
    kw["elements"] = np.arange(40, dtype=np.int64)
    parent = NodeBatch(**kw)
    sub = parent[np.arange(25)]

    for f in dataclasses.fields(NodeBatch):
        assert getattr(sub, f.name) is not None, f"field {f.name} was dropped"
    assert parent.elements is not None and sub.elements is not None
    assert parent.y is not None and sub.y is not None
    assert np.array_equal(sub.elements, parent.elements[:25])
    assert np.array_equal(sub.y, parent.y[:25])


def test_a_sliced_batch_survives_a_pickle_round_trip():
    """Sub-batches are built through ``_with_trusted_edges``, which bypasses
    ``__init__`` entirely. Parallel fitting ships batches to workers by pickle,
    so a half-initialized object would surface as a worker-side crash rather
    than anything the constructor could catch."""
    kw = ring(40)
    kw["y"] = np.arange(40, dtype=np.float64).reshape(-1, 1)
    kw["elements"] = np.full(40, 6, np.int64)
    sub = NodeBatch(**kw)[np.arange(25)]
    back = pickle.loads(pickle.dumps(sub))

    # every field, generically: a bypass that drops one would otherwise only
    # surface wherever that field happens to be read
    for f in dataclasses.fields(NodeBatch):
        original, restored = getattr(sub, f.name), getattr(back, f.name)
        assert np.array_equal(original, restored), f"field {f.name} did not survive"
    assert back.csr().max_deg == sub.csr().max_deg


def test_a_sliced_batch_is_still_bidirectional():
    """The invariant that makes skipping revalidation on a slice safe."""
    batch = NodeBatch(**ring(50))
    keep = np.zeros(50, bool)
    keep[:30] = True  # cuts the ring, dropping the two edges across the cut
    sub = batch[keep]
    assert python_set_check(sub.edge_src, sub.edge_dst)


def test_parallel_edges_are_accepted():
    """Pins existing semantics: the check compares edge *sets*, so a repeated
    edge is not by itself a missing-reverse-direction error."""
    NodeBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([0, 0, 1], np.int64),
        edge_dst=np.array([1, 1, 0], np.int64),
        edge_attrs=np.array([1, 1, 1], np.int64).reshape(-1, 1),
        graph_id=np.zeros(2, np.int64),
        y=None,
    )


def test_self_loop_is_its_own_reverse():
    NodeBatch(
        node_attrs=np.zeros((1, 1), np.int64),
        edge_src=np.array([0], np.int64),
        edge_dst=np.array([0], np.int64),
        edge_attrs=np.array([1], np.int64).reshape(-1, 1),
        graph_id=np.zeros(1, np.int64),
        y=None,
    )


def test_a_batch_with_no_edges_is_valid():
    NodeBatch(
        node_attrs=np.zeros((3, 1), np.int64),
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attrs=np.zeros(0, np.int64).reshape(-1, 1),
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
        b, node_counts=np.array([3, 1]), elements=np.array([6, 1, 1, 8], np.int64)
    )


def test_alignment_guard_catches_count_mismatch():
    b = tri()
    with pytest.raises(ValueError, match="atom count"):
        check_alignment(
            b, node_counts=np.array([2, 1]), elements=np.array([6, 1, 1, 8], np.int64)
        )


def test_alignment_guard_catches_permutation():
    """The bug counts alone cannot catch: same atoms, wrong order."""
    b = NodeBatch(**{**tri().__dict__, "elements": np.array([6, 1, 1, 8], np.int64)})
    with pytest.raises(ValueError, match="element"):
        check_alignment(
            b, node_counts=np.array([3, 1]), elements=np.array([1, 6, 1, 8], np.int64)
        )


def _small_batch(offset_graph_id, n_graphs, seed):
    """A batch of `n_graphs` disjoint triangles, each node carrying a
    deterministic attribute/target, for concat_batches' own tests. graph_id
    deliberately starts at `offset_graph_id` and is non-contiguous with
    other calls' output -- concat_batches must not assume 0-based ids."""
    rng = np.random.default_rng(seed)
    per = 3
    n = per * n_graphs
    src, dst, gid = [], [], []
    for g in range(n_graphs):
        off = g * per
        for i in range(per):
            j = (i + 1) % per
            src += [off + i, off + j]
            dst += [off + j, off + i]
        gid += [offset_graph_id + g] * per
    return NodeBatch(
        node_attrs=rng.integers(0, 4, size=(n, 2)).astype(np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attrs=np.ones(len(src), np.int64).reshape(-1, 1),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(n, 1)),
        elements=rng.integers(1, 20, size=n).astype(np.int64),
    )


def test_concat_batches_is_the_inverse_of_a_graph_id_split():
    whole = _small_batch(offset_graph_id=0, n_graphs=8, seed=0)
    mask = whole.graph_id < 4
    parts = [whole[mask], whole[~mask]]
    r = concat_batches(parts)

    assert r.n_nodes == whole.n_nodes
    assert r.n_edges == whole.n_edges
    assert r.y is not None and whole.y is not None
    np.testing.assert_array_equal(r.node_attrs, whole.node_attrs)
    np.testing.assert_allclose(r.y, whole.y)
    np.testing.assert_array_equal(r.elements, whole.elements)
    assert len(np.unique(r.graph_id)) == len(np.unique(whole.graph_id))
    # Re-validate through the real constructor, not just _with_trusted_edges'
    # bypass -- proves the edges concat_batches built are genuinely
    # bidirectional and in range, not merely accepted because the check was
    # skipped.
    NodeBatch(
        node_attrs=r.node_attrs,
        edge_src=r.edge_src,
        edge_dst=r.edge_dst,
        edge_attrs=r.edge_attrs,
        graph_id=r.graph_id,
        y=r.y,
        elements=r.elements,
    )


def test_concat_batches_renumbers_colliding_graph_ids():
    """The bug this function exists to prevent: two parts whose own graph_id
    both start at 0 (e.g. two from_rdkit chunks) must not have their graph 0
    silently merged into one."""
    a = _small_batch(offset_graph_id=0, n_graphs=3, seed=1)
    b = _small_batch(offset_graph_id=0, n_graphs=2, seed=2)  # also starts at 0
    r = concat_batches([a, b])

    assert len(np.unique(r.graph_id)) == 5
    counts = np.bincount(r.graph_id)
    assert (counts == 3).all()  # every triangle stays 3 nodes, none merged


def test_concat_batches_handles_non_contiguous_input_graph_ids():
    """graph_id is densified per part, not assumed already 0-based/contiguous."""
    a = _small_batch(offset_graph_id=10, n_graphs=2, seed=3)
    b = _small_batch(offset_graph_id=100, n_graphs=2, seed=4)
    r = concat_batches([a, b])
    assert len(np.unique(r.graph_id)) == 4
    assert set(r.graph_id.tolist()) == {0, 1, 2, 3}


def test_concat_batches_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        concat_batches([])


def test_concat_batches_rejects_mixed_y():
    a = _small_batch(offset_graph_id=0, n_graphs=2, seed=5)
    b = NodeBatch(
        node_attrs=a.node_attrs,
        edge_src=a.edge_src,
        edge_dst=a.edge_dst,
        edge_attrs=a.edge_attrs,
        graph_id=a.graph_id,
        y=None,
        elements=a.elements,
    )
    with pytest.raises(ValueError, match="y is set on some but not all"):
        concat_batches([a, b])


def test_concat_batches_rejects_mixed_elements():
    a = _small_batch(offset_graph_id=0, n_graphs=2, seed=6)
    b = NodeBatch(
        node_attrs=a.node_attrs,
        edge_src=a.edge_src,
        edge_dst=a.edge_dst,
        edge_attrs=a.edge_attrs,
        graph_id=a.graph_id,
        y=a.y,
        elements=None,
    )
    with pytest.raises(ValueError, match="elements is set on some but not all"):
        concat_batches([a, b])


def test_concat_batches_single_part_is_identity():
    a = _small_batch(offset_graph_id=0, n_graphs=3, seed=7)
    r = concat_batches([a])
    np.testing.assert_array_equal(r.node_attrs, a.node_attrs)
    np.testing.assert_array_equal(r.edge_src, a.edge_src)
    np.testing.assert_array_equal(r.graph_id, a.graph_id)
