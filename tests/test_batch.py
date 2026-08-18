import numpy as np
import pytest
from wllr.batch import AtomBatch, check_alignment

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
        AtomBatch(node_attrs=np.zeros((3, 1), np.int64),
                  edge_src=np.zeros(0, np.int64), edge_dst=np.zeros(0, np.int64),
                  edge_attr=np.zeros(0, np.int64),
                  graph_id=np.zeros(2, np.int64), y=None)

def test_edges_must_be_symmetric():
    with pytest.raises(ValueError, match="both directions"):
        AtomBatch(node_attrs=np.zeros((2, 1), np.int64),
                  edge_src=np.array([0], np.int64), edge_dst=np.array([1], np.int64),
                  edge_attr=np.array([1], np.int64),
                  graph_id=np.zeros(2, np.int64), y=None)

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
    check_alignment(b, atom_counts=np.array([3, 1]),
                    elements=np.array([6, 1, 1, 8], np.int64))

def test_alignment_guard_catches_count_mismatch():
    b = tri()
    with pytest.raises(ValueError, match="atom count"):
        check_alignment(b, atom_counts=np.array([2, 1]),
                        elements=np.array([6, 1, 1, 8], np.int64))

def test_alignment_guard_catches_permutation():
    """The bug counts alone cannot catch: same atoms, wrong order."""
    b = AtomBatch(**{**tri().__dict__, "elements": np.array([6, 1, 1, 8], np.int64)})
    with pytest.raises(ValueError, match="element"):
        check_alignment(b, atom_counts=np.array([3, 1]),
                        elements=np.array([1, 6, 1, 8], np.int64))
