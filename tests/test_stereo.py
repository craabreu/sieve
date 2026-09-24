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
    fp = list(
        content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 0)
    )
    assert len(fp) == 1
    assert fp[0][0] == fp[0][3] and fp[0][1] == fp[0][2]
    assert fp[0][0] != fp[0][1]


def test_fingerprints_grow_with_radius():
    from sieve.stereo import content_ranks

    # 0-1-2-3 with identical attributes: ends differ from the middle at r=1.
    batch = _chain(4, [[0], [0], [0], [0]])
    csr = batch.csr()
    fp = list(
        content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 2)
    )
    assert len(fp) == 3
    assert fp[0][0] == fp[0][1]  # radius 0: all identical
    assert fp[1][0] != fp[1][1]  # radius 1: degree differs
    assert fp[1][0] == fp[1][3]  # radius 1: the two ends agree


def test_fingerprints_are_independent_of_batch_composition():
    """The whole point: an id is batch-local, a fingerprint is not."""
    from sieve.stereo import content_ranks

    alone = _chain(4, [[0], [1], [1], [0]])
    csr_a = alone.csr()
    fp_a = list(
        content_ranks(
            alone.node_attrs, csr_a, np.zeros(csr_a.dst.shape[0], np.int64), 2
        )
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
    fp_t = list(
        content_ranks(
            together.node_attrs, csr_t, np.zeros(csr_t.dst.shape[0], np.int64), 2
        )
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


def test_content_ranks_yields_incrementally_without_retaining_every_radius():
    """refine consumes fp[j] with j = wl_round - 2, which increases by exactly
    one per round, so the fingerprints are used strictly in order and one at
    a time. Materializing all of them costs ~1.6 GB on the 38.9M-atom train
    split at depth 6, against ~310 MB for the one actually in use."""
    import types

    from sieve.stereo import content_ranks

    batch = _chain(4, [[0], [1], [1], [0]])
    csr = batch.csr()
    got = content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 3)
    assert isinstance(got, types.GeneratorType), "must not build the whole list"
    assert sum(1 for _ in got) == 4  # fp_0 .. fp_3


def _star(degree):
    """A centre bonded to ``degree`` leaves, edges in both directions."""
    leaves = np.arange(1, degree + 1)
    return NodeBatch(
        node_attrs=np.array([[1]] + [[0]] * degree, np.int64),
        edge_src=np.concatenate([np.zeros(degree, np.int64), leaves]),
        edge_dst=np.concatenate([leaves, np.zeros(degree, np.int64)]),
        edge_attrs=np.zeros((2 * degree, 1), np.int64),
        graph_id=np.zeros(degree + 1, np.int64),
    )


def _concat(first, second):
    n = first.node_attrs.shape[0]
    src = np.concatenate([first.edge_src, second.edge_src + n])
    return NodeBatch(
        node_attrs=np.concatenate([first.node_attrs, second.node_attrs]),
        edge_src=src,
        edge_dst=np.concatenate([first.edge_dst, second.edge_dst + n]),
        edge_attrs=np.zeros((src.shape[0], 1), np.int64),
        graph_id=np.concatenate([first.graph_id, second.graph_id + 1]),
    )


def _fingerprints(batch, n_rounds):
    from sieve.stereo import content_ranks

    csr = batch.csr()
    edge_code = np.zeros(csr.dst.shape[0], np.int64)
    return list(content_ranks(batch.node_attrs, csr, edge_code, n_rounds))


def test_fingerprints_do_not_depend_on_the_batch_maximum_degree():
    """A high-degree atom elsewhere in the batch must not change a molecule's
    fingerprints: a pentavalent phosphorus once changed the cis/trans codes,
    and hence the predictions, of every molecule batched beside it."""
    path = _chain(4, [[0], [1], [1], [0]])
    alone = _fingerprints(path, 2)
    beside = _fingerprints(_concat(path, _star(5)), 2)
    for fp_alone, fp_beside in zip(alone, beside, strict=True):
        np.testing.assert_array_equal(fp_alone, fp_beside[:4])


def test_content_ranks_refuses_a_degree_above_its_width():
    import pytest

    from sieve.stereo import CONTENT_RANK_WIDTH

    with pytest.raises(ValueError, match="degree"):
        _fingerprints(_star(CONTENT_RANK_WIDTH + 1), 1)
