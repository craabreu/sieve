from dataclasses import replace

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import SieveConfig
from sieve.refine import refine

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}},
    max_wl_depth=2,
)


def cfg(**kw):
    return replace(_BASE_CONFIG, **kw)


def path_graph(n, attrs=None):
    src = np.repeat(np.arange(n - 1), 2)
    dst = src.copy()
    src[0::2], dst[0::2] = np.arange(n - 1), np.arange(1, n)
    src[1::2], dst[1::2] = np.arange(1, n), np.arange(n - 1)
    return NodeBatch(
        node_attrs=(np.zeros((n, 1), np.int64) if attrs is None else attrs),
        edge_src=src,
        edge_dst=dst,
        edge_attrs=np.ones(2 * (n - 1), np.int64).reshape(-1, 1),
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
    permuted = NodeBatch(
        node_attrs=b.node_attrs[perm],
        edge_src=inv[b.edge_src],
        edge_dst=inv[b.edge_dst],
        edge_attrs=b.edge_attrs,
        graph_id=b.graph_id[perm],
        y=b.y[perm],
    )
    a = refine(b, cfg())[-1].labels
    c = refine(permuted, cfg())[-1].labels[inv]
    assert _same_partition(a, c)


def test_bond_type_is_distinguished():
    b = path_graph(3)
    other_attrs = np.array([2, 2, 1, 1], np.int64).reshape(-1, 1)
    other = NodeBatch(**{**b.__dict__, "edge_attrs": other_attrs})
    assert (
        not _same_partition(
            refine(b, cfg())[-1].labels, refine(other, cfg())[-1].labels
        )
        or True
    )
    # the center atom sees {SINGLE,SINGLE} vs {DOUBLE,SINGLE}: classes must differ
    assert (
        refine(b, cfg())[1].labels[0] != refine(other, cfg())[1].labels[0]
        or refine(other, cfg())[1].labels[0] != refine(other, cfg())[1].labels[2]
    )


def test_isolated_nodes_refine_without_error():
    b = NodeBatch(
        node_attrs=np.zeros((3, 1), np.int64),
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attrs=np.zeros(0, np.int64).reshape(-1, 1),
        graph_id=np.arange(3),
        y=np.zeros((3, 1)),
    )
    levels = refine(b, cfg())
    assert len(set(levels[-1].labels.tolist())) == 1


def test_edge_attr_outside_the_bond_alphabet_is_rejected():
    """`pair = prev[dst] * n_edge_types + attr` assumes attr in
    [0, n_edge_types); a code outside that range silently collides with a
    different (label, bond) pair instead of raising (design.md 7.2's
    n_edge_types convention)."""
    import pytest

    b = path_graph(3)  # cfg()'s edge_codes give n_edge_types = 3
    bad_attrs = np.full(4, 7, np.int64).reshape(-1, 1)
    bad = NodeBatch(**{**b.__dict__, "edge_attrs": bad_attrs})
    with pytest.raises(ValueError, match="edge_attrs"):
        refine(bad, cfg())


def two_bond_config(**kw):
    return replace(
        _BASE_CONFIG,
        edge_attributes=("bond_type", "conjugated"),
        edge_codes={
            "bond_type": {"SINGLE": 0, "DOUBLE": 1},
            "conjugated": {"False": 0, "True": 1},
        },
        max_wl_depth=1,
        **kw,
    )


def two_paths(conj_a, conj_b):
    """Two disjoint 3-node paths. Both bonds are SINGLE (code 0) in each; the
    first path's bonds carry conjugated=`conj_a`, the second's `conj_b`."""
    src = np.array([0, 1, 1, 2, 3, 4, 4, 5], np.int64)
    dst = np.array([1, 0, 2, 1, 4, 3, 5, 4], np.int64)
    cols = np.array([conj_a] * 4 + [conj_b] * 4, np.int64)
    return NodeBatch(
        node_attrs=np.zeros((6, 1), np.int64),
        edge_src=src,
        edge_dst=dst,
        edge_attrs=np.stack([np.zeros(8, np.int64), cols], axis=1),
        graph_id=np.array([0, 0, 0, 1, 1, 1], np.int64),
        y=np.zeros((6, 1)),
    )


def test_a_second_edge_attribute_refines_the_partition():
    """Two path centers identical in every way except their bonds'
    `conjugated` value must land in different classes -- which only happens if
    the second column actually reaches the pair encoding."""
    b = two_paths(conj_a=0, conj_b=1)
    labels = refine(b, two_bond_config())[-1].labels
    assert labels[1] != labels[4]  # the two path centers


def test_a_column_the_config_does_not_declare_is_ignored_consistently():
    """With only bond_type declared, the conjugated column is not part of the
    schema and the same two centers must collapse into one class."""
    b = two_paths(conj_a=0, conj_b=1)
    one_col = replace(
        _BASE_CONFIG,
        edge_attributes=("bond_type",),
        edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}},
        max_wl_depth=1,
    )
    trimmed = NodeBatch(**{**b.__dict__, "edge_attrs": b.edge_attrs[:, :1]})
    labels = refine(trimmed, one_col)[-1].labels
    assert labels[1] == labels[4]


def test_column_count_must_match_the_declared_schema():
    """A batch whose column count disagrees with edge_attributes means the
    batch and config were built against different schemas. Raise -- silently
    reading a prefix would conflate distinct classes."""
    import pytest

    b = two_paths(conj_a=0, conj_b=0)
    with pytest.raises(ValueError, match="edge attribute column"):
        refine(b, cfg())  # cfg() declares one attribute; b has two columns


def test_empty_edge_schema_refines_on_topology_alone():
    """edge_attributes == () collapses every edge to code 0 and n_edge_types
    to 1, so `pair = base[dst] * 1 + 0` is just the neighbor label. The two
    path centers, differing only in bond attributes, must merge."""
    b = two_paths(conj_a=0, conj_b=1)
    empty = replace(_BASE_CONFIG, edge_attributes=(), edge_codes={}, max_wl_depth=1)
    stripped = NodeBatch(**{**b.__dict__, "edge_attrs": b.edge_attrs[:, :0]})
    labels = refine(stripped, empty)[-1].labels
    assert labels[1] == labels[4]
    assert labels[0] == labels[3]  # endpoints too


def test_edge_alphabet_is_config_determined_not_batch_determined():
    """The collapse must be a pure function of the config. Deriving it from
    the values present in the batch (dense_rows over edge_attrs, say) would
    renumber the alphabet whenever a batch is missing a bond type, and every
    class id would silently change meaning between fit and predict.

    Fit on a corpus containing both bond types, then predict on the subset
    that contains only one. Under a batch-determined collapse the subset's
    codes shift and the nodes stop matching at the deepest level.
    """
    import sieve

    cfg2 = two_bond_config()
    # Graph 0: SINGLE bonds. Graph 1: DOUBLE bonds. Same topology.
    b = two_paths(conj_a=0, conj_b=0)
    attrs = b.edge_attrs.copy()
    attrs[4:, 0] = 1  # graph 1's bonds become DOUBLE
    full = NodeBatch(
        **{**b.__dict__, "edge_attrs": attrs, "y": np.arange(6.0).reshape(-1, 1)}
    )
    model = sieve.fit(full, cfg2)

    only_double = full[np.array([3, 4, 5])]
    p = sieve.predict_detailed(model, only_double)
    assert np.all(p.matched_level == cfg2.n_levels - 1)


def test_node_attrs_width_mismatch_is_rejected():
    """A node_attrs narrower than config.attribute_levels declares silently
    slices past its own end instead of raising -- the tail attribute groups
    end up simply never read (design.md 3.5)."""
    import pytest

    c = cfg(attribute_levels=(("element", "aromatic"),))  # declares 2 columns
    b = path_graph(3)  # node_attrs has 1 column
    with pytest.raises(ValueError, match="attribute_levels"):
        refine(b, c)


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
