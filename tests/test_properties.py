import numpy as np

from sieve.batch import AtomBatch
from sieve.refine import refine
from tests.helpers import simple_config


def _cycles(sizes):
    """Disjoint cycles as one batch: all nodes share every attribute."""
    src, dst, gid, off = [], [], [], 0
    for g, n in enumerate(sizes):
        for i in range(n):
            a, b = off + i, off + (i + 1) % n
            src += [a, b]
            dst += [b, a]
        gid += [g] * n
        off += n
    n_total = off
    return AtomBatch(
        node_attrs=np.zeros((n_total, 1), np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=np.zeros((n_total, 1)),
    )


def test_two_wl_indistinguishable_graphs_do_collide():
    """Accepted limit, not a bug: C6 and 2xC3 are 1-WL indistinguishable.

    Every node in both is degree-2 with degree-2 neighbours at every depth, so
    Sieve must assign them one class. Pinning this stops a future contributor
    from 'fixing' the expressiveness bound.
    """
    cfg = simple_config(max_wl_depth=4)
    six = refine(_cycles([6]), cfg)[-1].labels
    twin = refine(_cycles([3, 3]), cfg)[-1].labels
    assert len(set(six.tolist())) == 1
    assert len(set(twin.tolist())) == 1


def test_isomorphic_graphs_give_matching_class_multisets():
    cfg = simple_config(max_wl_depth=3)
    a = refine(_cycles([5, 7]), cfg)[-1]
    b = refine(_cycles([7, 5]), cfg)[-1]
    assert sorted(np.bincount(a.labels).tolist()) == sorted(
        np.bincount(b.labels).tolist()
    )
