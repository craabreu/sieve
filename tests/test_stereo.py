import numpy as np

from sieve.batch import NodeBatch


def _chain(n_nodes, attrs):
    """A path graph 0-1-2-..., edges in both directions."""
    src = np.repeat(np.arange(n_nodes - 1), 2)
    dst = src.copy()
    src[0::2] = np.arange(n_nodes - 1)
    dst[0::2] = np.arange(1, n_nodes)
    src[1::2] = np.arange(1, n_nodes)
    dst[1::2] = np.arange(n_nodes - 1)
    return NodeBatch(
        node_attrs=np.array(attrs, np.int64).reshape(n_nodes, -1),
        edge_src=src,
        edge_dst=dst,
        edge_attrs=np.zeros((src.shape[0], 1), np.int64),
        graph_id=np.zeros(n_nodes, np.int64),
    )


def test_fingerprint_zero_separates_exactly_the_attribute_rows():
    from sieve.stereo import content_ranks

    batch = _chain(4, [[0], [1], [1], [0]])
    csr = batch.csr()
    fp = content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 0)
    assert len(fp) == 1
    assert fp[0][0] == fp[0][3] and fp[0][1] == fp[0][2]
    assert fp[0][0] != fp[0][1]


def test_fingerprints_grow_with_radius():
    from sieve.stereo import content_ranks

    # 0-1-2-3 with identical attributes: ends differ from the middle at r=1.
    batch = _chain(4, [[0], [0], [0], [0]])
    csr = batch.csr()
    fp = content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 2)
    assert len(fp) == 3
    assert fp[0][0] == fp[0][1]  # radius 0: all identical
    assert fp[1][0] != fp[1][1]  # radius 1: degree differs
    assert fp[1][0] == fp[1][3]  # radius 1: the two ends agree


def test_fingerprints_are_independent_of_batch_composition():
    """The whole point: an id is batch-local, a fingerprint is not."""
    from sieve.stereo import content_ranks

    alone = _chain(4, [[0], [1], [1], [0]])
    csr_a = alone.csr()
    fp_a = content_ranks(
        alone.node_attrs, csr_a, np.zeros(csr_a.dst.shape[0], np.int64), 2
    )

    # The same path graph, with a second disconnected copy appended.
    src = np.concatenate([alone.edge_src, alone.edge_src + 4])
    dst = np.concatenate([alone.edge_dst, alone.edge_dst + 4])
    together = NodeBatch(
        node_attrs=np.concatenate([alone.node_attrs, alone.node_attrs]),
        edge_src=src,
        edge_dst=dst,
        edge_attrs=np.zeros((src.shape[0], 1), np.int64),
        graph_id=np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64),
    )
    csr_t = together.csr()
    fp_t = content_ranks(
        together.node_attrs, csr_t, np.zeros(csr_t.dst.shape[0], np.int64), 2
    )
    for j in range(3):
        assert np.array_equal(fp_a[j], fp_t[j][:4])


def test_the_stored_relation_is_used_when_both_winners_are_the_first():
    from sieve.stereo import cis_trans_codes

    rows = np.array([[0, 1, 2, -1, 3, -1, 1]], np.int64)  # cis, no ties possible
    fp = np.arange(4, dtype=np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [1]  # 1 == cis


def test_the_relation_flips_when_exactly_one_winner_differs():
    from sieve.stereo import cis_trans_codes

    # End a has substituents 2 and 4; 4 outranks 2, so a's winner is not a1.
    rows = np.array([[0, 1, 2, 4, 3, -1, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 20], np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [2]  # 2 == trans


def test_the_relation_is_restored_when_both_winners_differ():
    from sieve.stereo import cis_trans_codes

    rows = np.array([[0, 1, 2, 4, 3, 5, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 20, 30], np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [1]


def test_a_tie_at_either_end_defers():
    from sieve.stereo import cis_trans_codes

    rows = np.array([[0, 1, 2, 4, 3, -1, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 10], np.uint64)  # 2 and 4 tie
    assert cis_trans_codes(rows, fp).tolist() == [0]  # 0 == none


def test_an_end_with_one_substituent_cannot_tie():
    from sieve.stereo import cis_trans_codes

    rows = np.array([[0, 1, 2, -1, 3, -1, 0]], np.int64)
    fp = np.zeros(4, np.uint64)  # everything ties
    assert cis_trans_codes(rows, fp).tolist() == [2]
