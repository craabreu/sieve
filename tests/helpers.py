"""Fixtures shared across the suite."""

from dataclasses import replace

import numpy as np

from sieve.batch import AtomBatch
from sieve.config import SieveConfig

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"SINGLE": 1, "DOUBLE": 2},
    max_wl_depth=2,
)


def simple_config(**kw):
    return replace(_BASE_CONFIG, **kw)


def chain_batch(n, d=1, seed=0, graphs=1):
    """`graphs` disjoint paths of n nodes each, alternating attributes."""
    rng = np.random.default_rng(seed)
    per = n
    total = per * graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for i in range(per - 1):
            src += [off + i, off + i + 1]
            dst += [off + i + 1, off + i]
        gid += [g] * per
    return AtomBatch(
        node_attrs=(np.arange(total) % 2).reshape(-1, 1).astype(np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)),
    )


def star_batch(n_leaves, d=1, seed=0, graphs=1):
    """`graphs` disjoint stars of one centre and `n_leaves` leaves each.

    Unlike ``chain_batch`` (max degree 2 for every graph regardless of size),
    a star's max degree is ``n_leaves`` -- used to exercise batches whose max
    degree differs from another batch's, which chains alone never do.
    """
    rng = np.random.default_rng(seed)
    per = n_leaves + 1
    total = per * graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for leaf in range(1, per):
            src += [off, off + leaf]
            dst += [off + leaf, off]
        gid += [g] * per
    return AtomBatch(
        node_attrs=np.zeros((total, 1), np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)),
    )


def split_batch(batch, mask):
    """Take the sub-batch of atoms where `mask` is True, reindexing edges."""
    return batch[mask]
