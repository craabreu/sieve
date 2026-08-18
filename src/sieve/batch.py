"""The columnar input contract and its CSR layout (design.md 7.1, 11)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSRLayout:
    """Edges sorted by source, with each edge's position inside its source block.

    ``slot`` is computed once and reused at every refinement level (design.md 7.1).
    """

    order: np.ndarray
    indptr: np.ndarray
    slot: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    attr: np.ndarray
    max_deg: int


@dataclass(frozen=True)
class AtomBatch:
    """One block-diagonal graph over a whole corpus (design.md 11.1).

    Edges are stored in **both directions** for undirected graphs. ``graph_id``
    is not optional: it is what makes graph-level splitting possible, and a
    batch that loses it cannot be validated correctly.
    """

    node_attrs: np.ndarray  # (n_atoms, n_attr) int64, encoded categoricals
    edge_src: np.ndarray  # (n_edges,) int64
    edge_dst: np.ndarray  # (n_edges,) int64
    edge_attr: np.ndarray  # (n_edges,) int64
    graph_id: np.ndarray  # (n_atoms,) int64
    y: np.ndarray | None = None  # (n_atoms, d) float64
    elements: np.ndarray | None = None  # (n_atoms,) int64, for the alignment guard

    def __post_init__(self) -> None:
        n = self.node_attrs.shape[0]
        if self.graph_id.shape != (n,):
            raise ValueError(
                f"graph_id must have shape ({n},), got {self.graph_id.shape}"
            )
        e = self.edge_src.shape[0]
        for name in ("edge_dst", "edge_attr"):
            if getattr(self, name).shape != (e,):
                raise ValueError(f"{name} must have shape ({e},)")
        if self.y is not None and self.y.shape[0] != n:
            raise ValueError(f"y must have {n} rows, got {self.y.shape[0]}")
        if e:
            # Undirected graphs must carry both directions; the CSR construction
            # assumes it, and a one-way corpus silently halves every neighborhood.
            fwd = {
                (int(a), int(b))
                for a, b in zip(self.edge_src, self.edge_dst, strict=True)
            }
            if any((b, a) not in fwd for a, b in fwd):
                raise ValueError("edges must be stored in both directions")

    @property
    def n_atoms(self) -> int:
        return int(self.node_attrs.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])

    @property
    def shape(self) -> tuple[int]:
        """One "row" per atom (design.md 10.2).

        Exists so scikit-learn's ``_safe_indexing``/``_num_samples`` treat an
        ``AtomBatch`` as array-like instead of falling back to indexing it one
        element at a time -- without this, ``GraphKFold`` cannot actually be
        used with ``cross_val_score``/``GridSearchCV`` despite that being the
        whole point of this module.
        """
        return (self.n_atoms,)

    def __len__(self) -> int:
        return self.n_atoms

    def __getitem__(self, key) -> AtomBatch:
        """Select a sub-batch by boolean mask or integer index array.

        Edges are kept only when both endpoints are selected; a selection
        that cuts a graph in half silently drops its cut edges rather than
        reindexing across a boundary that no longer exists. Callers (e.g.
        ``GraphKFold``) are responsible for selecting whole graphs.

        scikit-learn's array indexing calls ``X[key, ...]`` (design.md 10.2):
        for a 1-D array that's the same as ``X[key]``, but the literal key
        received here is the tuple ``(key, Ellipsis)``, which needs
        unwrapping before it reaches `np.asarray`.
        """
        if isinstance(key, tuple):
            key = key[0]
        mask = np.zeros(self.n_atoms, bool)
        idx = np.asarray(key)
        if idx.dtype == bool:
            mask[:] = idx
        else:
            mask[idx] = True
        sel = np.flatnonzero(mask)
        remap = np.full(self.n_atoms, -1, np.int64)
        remap[sel] = np.arange(sel.size)
        keep = mask[self.edge_src] & mask[self.edge_dst]
        return AtomBatch(
            node_attrs=self.node_attrs[sel],
            edge_src=remap[self.edge_src[keep]],
            edge_dst=remap[self.edge_dst[keep]],
            edge_attr=self.edge_attr[keep],
            graph_id=self.graph_id[sel],
            y=None if self.y is None else self.y[sel],
            elements=None if self.elements is None else self.elements[sel],
        )

    def csr(self) -> CSRLayout:
        n = self.n_atoms
        order = np.argsort(self.edge_src, kind="stable")
        src = self.edge_src[order]
        deg = np.bincount(src, minlength=n)
        indptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
        # Position within the source's adjacency block. Valid only because
        # `src` is sorted, which is why `order` is applied first.
        slot = np.arange(src.shape[0], dtype=np.int64) - indptr[src]
        return CSRLayout(
            order=order,
            indptr=indptr,
            slot=slot,
            src=src,
            dst=self.edge_dst[order],
            attr=self.edge_attr[order],
            max_deg=int(np.max(deg)) if n else 0,
        )


def check_alignment(
    batch: AtomBatch, atom_counts: np.ndarray, elements: np.ndarray
) -> None:
    """Verify targets line up with parsed molecules (design.md 7.5, 11.3).

    This is the highest-severity failure mode in the system: a misalignment
    corrupts every label and raises nothing, surfacing only as unexplained
    inaccuracy. Counts alone cannot catch it -- a permutation preserves them --
    so the element check is the one that actually does the work.
    """
    _, sizes = np.unique(batch.graph_id, return_counts=True)
    if sizes.shape != atom_counts.shape or not np.array_equal(sizes, atom_counts):
        raise ValueError(
            f"atom count mismatch: batch has {sizes.tolist()[:5]}..., "
            f"corpus reports {np.asarray(atom_counts).tolist()[:5]}..."
        )
    if batch.elements is not None:
        bad = np.flatnonzero(batch.elements != elements)
        if bad.size:
            raise ValueError(
                f"element mismatch at {bad.size} atoms (first at index {bad[0]}): "
                "the parsed molecule and its target rows are in different orders"
            )
