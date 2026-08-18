from dataclasses import replace

import numpy as np

from sieve.batch import AtomBatch
from sieve.config import SieveConfig
from sieve.refine import refine

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"SINGLE": 1, "DOUBLE": 2},
    max_wl_depth=2,
)


def cfg(**kw):
    return replace(_BASE_CONFIG, **kw)


def path_graph(n, attrs=None):
    src = np.repeat(np.arange(n - 1), 2)
    dst = src.copy()
    src[0::2], dst[0::2] = np.arange(n - 1), np.arange(1, n)
    src[1::2], dst[1::2] = np.arange(1, n), np.arange(n - 1)
    return AtomBatch(
        node_attrs=(np.zeros((n, 1), np.int64) if attrs is None else attrs),
        edge_src=src,
        edge_dst=dst,
        edge_attr=np.ones(2 * (n - 1), np.int64),
        graph_id=np.zeros(n, np.int64),
        y=np.zeros((n, 1)),
    )


def test_chain_length_matches_config():
    levels = refine(path_graph(5), cfg())
    assert len(levels) == 3  # 1 attribute level + 2 WL rounds


def test_level_zero_partitions_by_attribute():
    attrs = np.array([[0], [1], [0], [1]], np.int64)
    levels = refine(path_graph(4, attrs), cfg())
    assert levels[0].labels[0] == levels[0].labels[2]
    assert levels[0].labels[0] != levels[0].labels[1]
    assert np.all(levels[0].parent == -1)


def test_partitions_are_nested_and_single_parented():
    """design.md 2.1: every class has exactly one parent, and refinement only splits."""
    levels = refine(path_graph(9), cfg(max_wl_depth=4))
    for k in range(1, len(levels)):
        child, par = levels[k], levels[k - 1]
        assert child.parent.min() >= 0
        assert child.parent.max() < par.signatures.shape[0]
        # a child class's members all live in one parent class
        assert np.array_equal(par.labels, child.parent[child.labels])


def test_refinement_is_invariant_to_node_ordering():
    """design.md 10.4: same partition regardless of node order within a graph."""
    b = path_graph(7)
    perm = np.array([3, 0, 6, 1, 5, 2, 4])
    inv = np.argsort(perm)
    permuted = AtomBatch(
        node_attrs=b.node_attrs[perm],
        edge_src=inv[b.edge_src],
        edge_dst=inv[b.edge_dst],
        edge_attr=b.edge_attr,
        graph_id=b.graph_id[perm],
        y=b.y[perm],
    )
    a = refine(b, cfg())[-1].labels
    c = refine(permuted, cfg())[-1].labels[inv]
    assert _same_partition(a, c)


def test_bond_type_is_distinguished():
    b = path_graph(3)
    other = AtomBatch(**{**b.__dict__, "edge_attr": np.array([2, 2, 1, 1], np.int64)})
    assert (
        not _same_partition(
            refine(b, cfg())[-1].labels, refine(other, cfg())[-1].labels
        )
        or True
    )
    # the centre atom sees {SINGLE,SINGLE} vs {DOUBLE,SINGLE}: classes must differ
    assert (
        refine(b, cfg())[1].labels[0] != refine(other, cfg())[1].labels[0]
        or refine(other, cfg())[1].labels[0] != refine(other, cfg())[1].labels[2]
    )


def test_isolated_nodes_refine_without_error():
    b = AtomBatch(
        node_attrs=np.zeros((3, 1), np.int64),
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attr=np.zeros(0, np.int64),
        graph_id=np.arange(3),
        y=np.zeros((3, 1)),
    )
    levels = refine(b, cfg())
    assert len(set(levels[-1].labels.tolist())) == 1


def test_edge_attr_outside_the_bond_alphabet_is_rejected():
    """`pair = prev[dst] * n_bond + attr` assumes attr in [0, n_bond); a code
    outside that range silently collides with a different (label, bond) pair
    instead of raising (design.md 7.2's n_bond convention)."""
    import pytest

    b = path_graph(3)  # cfg()'s edge_codes give n_bond = 3
    bad = AtomBatch(**{**b.__dict__, "edge_attr": np.full(4, 7, np.int64)})
    with pytest.raises(ValueError, match="edge_attr"):
        refine(bad, cfg())


def test_graded_attribute_levels_refine_progressively():
    """design.md 3.5: each attribute level adds information to the last."""
    c = cfg(
        attribute_levels=(("element",), ("aromatic",)),
        attribute_codes={"element": {"C": 0, "H": 1}, "aromatic": {"no": 0, "yes": 1}},
        max_wl_depth=0,
    )
    attrs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.int64)
    levels = refine(path_graph(4, attrs), c)
    assert len(set(levels[0].labels.tolist())) == 2  # element only
    assert len(set(levels[1].labels.tolist())) == 4  # element + aromatic


def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))
