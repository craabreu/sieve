"""The columnar input contract and its CSR layout (design.md 7.1, 11)."""

from __future__ import annotations

from dataclasses import dataclass, fields

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
class NodeBatch:
    """One block-diagonal graph over a whole corpus (design.md 11.1).

    Edges are stored in **both directions** for undirected graphs. ``graph_id``
    is not optional: it is what makes graph-level splitting possible, and a
    batch that loses it cannot be validated correctly.
    """

    node_attrs: np.ndarray  # (n_nodes, n_attr) int64, encoded categoricals
    edge_src: np.ndarray  # (n_edges,) int64
    edge_dst: np.ndarray  # (n_edges,) int64
    edge_attrs: np.ndarray  # (n_edges, n_edge_attr) int64, encoded categoricals
    graph_id: np.ndarray  # (n_nodes,) int64
    y: np.ndarray | None = None  # (n_nodes, d) float64
    elements: np.ndarray | None = None  # (n_nodes,) int64, for the alignment guard

    def __post_init__(self) -> None:
        self._check_shapes()
        self._check_edges()

    def _check_shapes(self) -> None:
        n = self.node_attrs.shape[0]
        if self.graph_id.shape != (n,):
            raise ValueError(
                f"graph_id must have shape ({n},), got {self.graph_id.shape}"
            )
        e = self.edge_src.shape[0]
        if self.edge_dst.shape != (e,):
            raise ValueError(f"edge_dst must have shape ({e},)")
        if self.edge_attrs.ndim != 2 or self.edge_attrs.shape[0] != e:
            raise ValueError(
                f"edge_attrs must have shape ({e}, n_edge_attr), got "
                f"{self.edge_attrs.shape}"
            )
        if self.y is not None and self.y.shape[0] != n:
            raise ValueError(f"y must have {n} rows, got {self.y.shape[0]}")

    def _check_edges(self) -> None:
        """Endpoints are in range, and every edge carries its reverse.

        Undirected graphs must carry both directions; the CSR construction
        assumes it, and a one-way corpus silently halves every neighborhood.

        Both halves are vectorized. The obvious spelling -- a Python set of
        ``(int(src), int(dst))`` tuples -- cost more than an entire ``fit()``
        on a 227k-atom corpus and was paid again on every sub-batch, so it
        dominated anything that slices (``GraphKFold``, ``chunk_size``,
        sharded fitting). Encoding an edge as ``src * n + dst`` is a bijection
        once the endpoints are known to be in ``[0, n)``, which is what makes
        comparing the edge set against its own reverse exact here; ``np.unique``
        rather than a plain sort because the check has always compared edge
        *sets*, so a repeated edge is not by itself a missing reverse.
        """
        if not self.edge_src.shape[0]:
            return
        n = self.node_attrs.shape[0]
        lo = min(int(self.edge_src.min()), int(self.edge_dst.min()))
        hi = max(int(self.edge_src.max()), int(self.edge_dst.max()))
        if lo < 0 or hi >= n:
            raise ValueError(
                f"edge index out of range: endpoints span [{lo}, {hi}], "
                f"but the batch has {n} nodes"
            )
        # In int64. Computed in a narrower input dtype the product wraps and
        # the key stops being a bijection: at n - 1 == 2**16 the edge (65536, 0)
        # collides with its own reverse under int32, so a one-way corpus would
        # pass the one check that exists to catch it.
        src = self.edge_src.astype(np.int64, copy=False)
        dst = self.edge_dst.astype(np.int64, copy=False)
        fwd = np.unique(src * n + dst)
        rev = np.unique(dst * n + src)
        if not np.array_equal(fwd, rev):
            raise ValueError("edges must be stored in both directions")

    @classmethod
    def _with_trusted_edges(cls, **kw) -> NodeBatch:
        """Build a batch whose edges are already known to satisfy `_check_edges`.

        Only for callers that *derive* a batch from an already-valid one in a
        way that provably preserves the invariant -- see ``__getitem__``. Shape
        checks still run; it is the O(edges) edge check that is skipped.
        """
        names = {f.name for f in fields(cls)}
        if missing := names - set(kw):
            raise TypeError(f"_with_trusted_edges is missing {sorted(missing)}")
        if unexpected := set(kw) - names:
            raise TypeError(f"_with_trusted_edges got unexpected {sorted(unexpected)}")
        obj = cls.__new__(cls)
        for name, value in kw.items():
            object.__setattr__(obj, name, value)
        obj._check_shapes()
        return obj

    @property
    def n_nodes(self) -> int:
        return int(self.node_attrs.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])

    @property
    def shape(self) -> tuple[int]:
        """One "row" per node (design.md 10.2).

        Exists so scikit-learn's ``_safe_indexing``/``_num_samples`` treat an
        ``NodeBatch`` as array-like instead of falling back to indexing it one
        element at a time -- without this, ``GraphKFold`` cannot actually be
        used with ``cross_val_score``/``GridSearchCV`` despite that being the
        whole point of this module.
        """
        return (self.n_nodes,)

    def __len__(self) -> int:
        return self.n_nodes

    def __getitem__(self, key) -> NodeBatch:
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
        mask = np.zeros(self.n_nodes, bool)
        idx = np.asarray(key)
        if idx.dtype == bool:
            mask[:] = idx
        else:
            mask[idx] = True
        sel = np.flatnonzero(mask)
        remap = np.full(self.n_nodes, -1, np.int64)
        remap[sel] = np.arange(sel.size)
        keep = mask[self.edge_src] & mask[self.edge_dst]
        # An edge is kept only when *both* endpoints are selected, so (a,b)
        # survives exactly when (b,a) does, and `remap` is injective on the
        # selection: the sub-batch inherits bidirectionality and in-range
        # endpoints from a parent that already had them. Re-deriving that is
        # the single most expensive thing this class does, so it is skipped.
        return NodeBatch._with_trusted_edges(
            node_attrs=self.node_attrs[sel],
            edge_src=remap[self.edge_src[keep]],
            edge_dst=remap[self.edge_dst[keep]],
            edge_attrs=self.edge_attrs[keep],
            graph_id=self.graph_id[sel],
            y=None if self.y is None else self.y[sel],
            elements=None if self.elements is None else self.elements[sel],
        )

    def csr(self) -> CSRLayout:
        n = self.n_nodes
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
            attr=self.edge_attrs[order],
            max_deg=int(np.max(deg)) if n else 0,
        )


def concat_batches(parts: list[NodeBatch]) -> NodeBatch:
    """Concatenate several batches into one, as disjoint graphs.

    The inverse of ``__getitem__``/boolean masking (``split_batch`` in the
    test suite): where those split one batch into several, this rejoins
    several into one. Written for parallel featurization -- each worker
    featurizes its own slice of molecules into its own small batch, and the
    parent process reassembles them -- but it is a general concat, not
    specific to that caller.

    Node-indexed arrays concatenate directly. Edge endpoints are offset by a
    running node count so they still index correctly into the combined
    ``node_attrs``. ``graph_id`` cannot simply concatenate: each part's own
    ids are local to that part (e.g. ``from_rdkit`` numbers every batch's
    graphs from 0), so without renumbering, graph 0 of part 2 would silently
    collide with graph 0 of part 1 -- ``check_alignment``'s
    ``np.unique(graph_id)`` would merge two unrelated molecules into one atom
    count, a wrong answer with no error. Each part's ids are first densified
    to ``0..n_graphs-1`` via ``np.unique(..., return_inverse=True)`` (not
    assumed to already be dense or 0-based -- only that no part's own graph
    ids alias across nodes that belong to different graphs, which
    ``NodeBatch``'s own invariants already require), then offset by a running
    graph count.

    ``y``/``elements`` must be all-``None`` or all-set across every part --
    concatenating a real array with a placeholder zero-column would silently
    fabricate targets/elements for the parts that had none.
    """
    if not parts:
        raise ValueError("concat_batches requires at least one batch")

    has_y = {p.y is not None for p in parts}
    if len(has_y) > 1:
        raise ValueError("cannot concat batches where y is set on some but not all")
    has_elements = {p.elements is not None for p in parts}
    if len(has_elements) > 1:
        raise ValueError(
            "cannot concat batches where elements is set on some but not all"
        )

    node_attrs = np.concatenate([p.node_attrs for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0) if has_y == {True} else None
    elements = (
        np.concatenate([p.elements for p in parts], axis=0)
        if has_elements == {True}
        else None
    )

    edge_src, edge_dst, edge_attrs, graph_id = [], [], [], []
    node_off = 0
    graph_off = 0
    for p in parts:
        edge_src.append(p.edge_src + node_off)
        edge_dst.append(p.edge_dst + node_off)
        edge_attrs.append(p.edge_attrs)
        _, inv = np.unique(p.graph_id, return_inverse=True)
        dense_gid = np.asarray(inv, dtype=np.int64).ravel()
        graph_id.append(dense_gid + graph_off)
        node_off += p.n_nodes
        graph_off += int(np.max(dense_gid)) + 1 if dense_gid.size else 0

    # _with_trusted_edges skips the O(edges) bidirectionality/range check
    # (_check_edges): legitimate here because each part already satisfies it
    # on its own, and both the node and graph offsets above are injective and
    # applied identically to src and dst, so an edge's forward and reverse
    # direction move together and stay in range.
    # parts is non-empty (checked above), so every list here has >= 1 entry.
    return NodeBatch._with_trusted_edges(
        node_attrs=node_attrs,
        edge_src=np.concatenate(edge_src),
        edge_dst=np.concatenate(edge_dst),
        edge_attrs=np.concatenate(edge_attrs, axis=0),
        graph_id=np.concatenate(graph_id),
        y=y,
        elements=elements,
    )


def check_alignment(
    batch: NodeBatch, node_counts: np.ndarray, elements: np.ndarray
) -> None:
    """Verify targets line up with parsed molecules (design.md 7.5, 11.3).

    This is the highest-severity failure mode in the system: a misalignment
    corrupts every label and raises nothing, surfacing only as unexplained
    inaccuracy. Counts alone cannot catch it -- a permutation preserves them --
    so the element check is the one that actually does the work.
    """
    _, sizes = np.unique(batch.graph_id, return_counts=True)
    if sizes.shape != node_counts.shape or not np.array_equal(sizes, node_counts):
        raise ValueError(
            f"atom count mismatch: batch has {sizes.tolist()[:5]}..., "
            f"corpus reports {np.asarray(node_counts).tolist()[:5]}..."
        )
    if batch.elements is not None:
        bad = np.flatnonzero(batch.elements != elements)
        if bad.size:
            raise ValueError(
                f"element mismatch at {bad.size} atoms (first at index {bad[0]}): "
                "the parsed molecule and its target rows are in different orders"
            )
