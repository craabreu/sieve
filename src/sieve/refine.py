"""Attribute levels then WL rounds, all vectorised (design.md 3.5, 7.1, 7.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sieve.batch import AtomBatch
from sieve.config import SieveConfig
from sieve.dedupe import dense_rows


@dataclass(frozen=True)
class LevelLabels:
    """One level of the refinement chain.

    ``signatures[j]`` is the deduplicated signature row of class ``j`` -- this
    array *is* the vocabulary (design.md 9). Column 0 holds the parent's id at
    the previous level for every level above 0, which is why ``parent`` is free
    and single-parenthood is structural rather than asserted.
    """

    labels: np.ndarray        # (n_atoms,) int64
    signatures: np.ndarray    # (n_classes, width) int64
    parent: np.ndarray        # (n_classes,) int32; -1 at level 0

    @property
    def n_classes(self) -> int:
        return int(self.signatures.shape[0])


def refine(batch: AtomBatch, config: SieveConfig) -> list[LevelLabels]:
    """Build the full refinement chain for a corpus.

    One array operation per level over the whole block-diagonal corpus -- there
    is no per-molecule or per-atom loop anywhere in this function.
    """
    n = batch.n_atoms
    levels: list[LevelLabels] = []

    # --- attribute levels (design.md 3.5) --------------------------------
    # Level j introduces attribute group j on top of level j-1. Each level is
    # built from the previous plus strictly more information, which is the only
    # premise design.md 2 needs.
    used = 0
    for j, group in enumerate(config.attribute_levels):
        width = len(group)
        cols = batch.node_attrs[:, used:used + width]
        used += width
        if j == 0:
            sig = cols
        else:
            sig = np.concatenate([levels[-1].labels[:, None], cols], axis=1)
        labels, uniq = dense_rows(sig)
        parent = (np.full(uniq.shape[0], -1, np.int32) if j == 0
                  else uniq[:, 0].astype(np.int32))
        levels.append(LevelLabels(labels, uniq, parent))

    # --- WL rounds (design.md 7.2) ---------------------------------------
    csr = batch.csr()
    n_bond = config.n_bond
    for _ in range(config.max_wl_depth):
        prev = levels[-1].labels
        # Encode (neighbour label, bond) as one integer so a row of neighbours
        # is a plain integer vector.
        pair = prev[csr.dst] * n_bond + csr.attr
        pad = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
        pad[csr.src, csr.slot] = pair
        # Sorting canonicalises the multiset; -1 pads sort first, and because a
        # node's pad count is fixed, degree stays encoded.
        pad.sort(axis=1)
        sig = np.concatenate([prev[:, None], pad], axis=1)
        labels, uniq = dense_rows(sig)
        levels.append(LevelLabels(labels, uniq, uniq[:, 0].astype(np.int32)))

    return levels
