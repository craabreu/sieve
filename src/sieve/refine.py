"""Attribute levels then WL rounds, all vectorized (design.md 3.5, 7.1, 7.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import LEVEL_WL, SieveConfig
from sieve.dedupe import dense_rows


@dataclass(frozen=True)
class LevelLabels:
    """One level of the refinement chain.

    ``signatures[j]`` is the deduplicated signature row of class ``j`` -- this
    array *is* the vocabulary (design.md 9). Column 0 holds the parent's id at
    the previous level for every level above 0, which is why ``parent`` is free
    and single-parenthood is structural rather than asserted.
    """

    labels: np.ndarray  # (n_nodes,) int64
    signatures: np.ndarray  # (n_classes, width) int64
    parent: np.ndarray  # (n_classes,) int32; -1 at level 0

    @property
    def n_classes(self) -> int:
        return int(self.signatures.shape[0])


def refine(batch: NodeBatch, config: SieveConfig) -> list[LevelLabels]:
    """Build the full refinement chain for a corpus.

    One array operation per level over the whole block-diagonal corpus -- there
    is no per-molecule or per-atom loop anywhere in this function.

    When ``config.neighbor_depth`` is set, this also builds the coarse
    neighbor chain (design.md 3.6). Every level's shape and dependency on
    earlier levels comes from ``config.level_kinds``/``level_parents``/
    ``neighbor_source``, so this function does not special-case coarsening
    itself -- the loop below is generic either way, and reduces to today's
    single WL chain exactly when ``neighbor_depth`` is ``None``.
    """
    n = batch.n_nodes
    levels: list[LevelLabels] = []

    declared = sum(len(group) for group in config.attribute_levels)
    if batch.node_attrs.shape[1] != declared:
        # A narrower node_attrs silently slices past its own end instead of
        # raising, so the tail attribute groups get treated as declared but
        # never actually read -- a config/adapter mismatch that would
        # otherwise surface only as unexplained inaccuracy.
        raise ValueError(
            f"config.attribute_levels declares {declared} attribute columns "
            f"but batch.node_attrs has {batch.node_attrs.shape[1]}"
        )

    # --- attribute levels (design.md 3.5) --------------------------------
    # Level j introduces attribute group j on top of level j-1. Each level is
    # built from the previous plus strictly more information, which is the only
    # premise design.md 2 needs.
    used = 0
    for j, group in enumerate(config.attribute_levels):
        width = len(group)
        cols = batch.node_attrs[:, used : used + width]
        used += width
        if j == 0:
            sig = cols
        else:
            sig = np.concatenate([levels[-1].labels[:, None], cols], axis=1)
        labels, uniq = dense_rows(sig)
        parent = (
            np.full(uniq.shape[0], -1, np.int32)
            if j == 0
            else uniq[:, 0].astype(np.int32)
        )
        levels.append(LevelLabels(labels, uniq, parent))

    # --- WL rounds (design.md 7.2), plus the coarse neighbor chain when
    # configured (design.md 3.6) -------------------------------------------
    csr = batch.csr()
    n_edge_types = config.n_edge_types
    kinds = config.level_kinds[len(levels) :]
    parents = config.level_parents[len(levels) :]
    neighbor_src = config.neighbor_source[len(levels) :]
    if kinds and csr.attr.size:
        # `pair = base[dst] * n_edge_types + attr` assumes 0 <= attr <
        # n_edge_types. A code outside that range collides with a *different*
        # (label, bond) pair instead of raising, silently conflating two
        # distinct classes.
        bad = (csr.attr < 0) | (csr.attr >= n_edge_types)
        if bad.any():
            raise ValueError(
                f"edge_attr contains code {int(csr.attr[bad][0])}, outside "
                f"[0, {n_edge_types}) implied by config.edge_codes"
            )

    for offset, kind in enumerate(kinds):
        base = levels[parents[offset]].labels
        if kind == LEVEL_WL:
            # Encode (neighbor label, bond) as one integer so a row of
            # neighbors is a plain integer vector.
            pair = base[csr.dst] * n_edge_types + csr.attr
            pad = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
            pad[csr.src, csr.slot] = pair
            # Sorting canonicalizes the multiset; -1 pads sort first, and
            # because a node's pad count is fixed, degree stays encoded.
            pad.sort(axis=1)
            sig = np.concatenate([base[:, None], pad], axis=1)
        else:  # LEVEL_WL_PAIR: the coarse chain's own class at this round is
            # already aggregated over its neighbors, so no separate multiset
            # is needed here -- just the pair (self, coarse neighbor state).
            ns = neighbor_src[offset]
            if ns is None:
                # config guarantees this for LEVEL_WL_PAIR; a real raise, not
                # assert, since assert is compiled away under `python -O`.
                raise AssertionError(
                    "neighbor_source is None for a LEVEL_WL_PAIR level"
                )
            neighbor = levels[ns].labels
            sig = np.concatenate([base[:, None], neighbor[:, None]], axis=1)
        labels, uniq = dense_rows(sig)
        levels.append(LevelLabels(labels, uniq, uniq[:, 0].astype(np.int32)))

    return levels
