"""Fixtures shared across the suite."""
import numpy as np
from sieve.batch import AtomBatch
from sieve.config import SieveConfig

def simple_config(**kw):
    d = dict(target_dim=1, attribute_levels=(("element",),),
             attribute_codes={"element": {"C": 0, "H": 1}},
             edge_codes={"SINGLE": 1, "DOUBLE": 2}, max_wl_depth=2)
    d.update(kw)
    return SieveConfig(**d)

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
        edge_src=np.array(src, np.int64), edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)))

def split_batch(batch, mask):
    """Take the sub-batch of atoms where `mask` is True, reindexing edges."""
    idx = np.flatnonzero(mask)
    remap = np.full(batch.n_atoms, -1, np.int64)
    remap[idx] = np.arange(idx.size)
    keep = mask[batch.edge_src] & mask[batch.edge_dst]
    return AtomBatch(
        node_attrs=batch.node_attrs[idx],
        edge_src=remap[batch.edge_src[keep]], edge_dst=remap[batch.edge_dst[keep]],
        edge_attr=batch.edge_attr[keep], graph_id=batch.graph_id[idx],
        y=None if batch.y is None else batch.y[idx])
