"""Attribute levels then WL rounds, all vectorized (design.md 3.5, 7.1, 7.2)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import KIND_AWARE, KIND_BLIND, LEVEL_WL, SieveConfig
from sieve.dedupe import dense_rows
from sieve.stereo import cis_trans_codes, content_ranks, directed_positions


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
    # Set only on the WL levels of a stereo-aware chain (docs/superpowers/
    # specs/2026-09-23-stereo-refines-the-blind-class-design.md, section 3):
    # ``labels`` is then each atom's aware class and ``blind_labels`` its
    # stereo-blind class, both ids in the one vocabulary ``signatures``.
    blind_labels: np.ndarray | None = None  # (n_nodes,) int64
    kind: np.ndarray | None = None  # (n_classes,) uint8, KIND_* bits
    blind_of: np.ndarray | None = None  # (n_classes,) int64

    @property
    def n_classes(self) -> int:
        return int(self.signatures.shape[0])

    @property
    def blind(self) -> np.ndarray:
        """Each atom's stereo-blind class: its own class when no track is on."""
        return self.labels if self.blind_labels is None else self.blind_labels


def _wl_rows(
    base: np.ndarray, csr, full: np.ndarray, n: int, n_edge_types: int
) -> np.ndarray:
    """One WL signature row per atom: its own class, then the sorted multiset
    of (neighbor class, edge code) pairs."""
    # Encode (neighbor label, bond) as one integer so a row of neighbors is a
    # plain integer vector.
    pair = base[csr.dst] * n_edge_types + full
    pad = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
    pad[csr.src, csr.slot] = pair
    # Sorting canonicalizes the multiset; -1 pads sort first, and because a
    # node's pad count is fixed, degree stays encoded.
    pad.sort(axis=1)
    return np.concatenate([base[:, None], pad], axis=1)


def _union_level(sig_aware: np.ndarray, sig_blind: np.ndarray) -> LevelLabels:
    """One vocabulary holding every atom's aware and blind rows (spec section 3).

    Rows are deduplicated together, so an aware row equal to a blind row is
    the same class, flagged both -- the case for every atom with no stereo
    bond within reach. Only the rows that differ are stacked: the rest would
    deduplicate onto their blind twin anyway.
    """
    n = sig_blind.shape[0]
    differs = np.flatnonzero((sig_aware != sig_blind).any(axis=1))
    labels, uniq = dense_rows(np.concatenate([sig_blind, sig_aware[differs]]))
    blind = labels[:n]
    aware = blind.copy()
    aware[differs] = labels[n:]
    kind = np.zeros(uniq.shape[0], np.uint8)
    kind[blind] |= KIND_BLIND
    kind[aware] |= KIND_AWARE
    # Well defined: an atom's aware class determines its blind one, since the
    # aware row names the aware classes one level down, whose blind classes
    # are themselves determined, and the trit only adds to the edge code.
    blind_of = np.arange(uniq.shape[0], dtype=np.int64)
    blind_of[aware] = blind
    return LevelLabels(aware, uniq, uniq[:, 0].astype(np.int32), blind, kind, blind_of)


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
    radices = config.edge_radices
    if csr.attr.shape[1] != len(radices):
        raise ValueError(
            f"batch has {csr.attr.shape[1]} edge attribute columns, but config "
            f"declares {len(radices)}: {list(config.edge_attributes)}"
        )
    kinds = config.level_kinds[len(levels) :]
    parents = config.level_parents[len(levels) :]
    neighbor_src = config.neighbor_source[len(levels) :]
    if kinds and csr.attr.shape[0]:
        # Each column must stay inside its own radix. A code outside it makes
        # the mixed-radix fold below collide with a *different* combination
        # instead of raising, silently conflating two distinct classes.
        for j, (name, radix) in enumerate(
            zip(config.edge_attributes, radices, strict=True)
        ):
            col = csr.attr[:, j]
            bad = (col < 0) | (col >= radix)
            if bad.any():
                raise ValueError(
                    f"edge_attrs column {j} ({name!r}) contains code "
                    f"{int(col[bad][0])}, outside [0, {radix}) implied by "
                    f"config.edge_codes[{name!r}]"
                )
    # Collapse the per-attribute columns into one integer per edge, mixed
    # radix. Derived from the *config*, never from the values present in this
    # batch: dense_rows here would renumber the alphabet whenever a batch is
    # missing a value, so a model's class ids would mean one thing at fit time
    # and another at predict time (spec 2026-09-02, section 3).
    edge_code = np.zeros(csr.attr.shape[0], np.int64)
    for j, radix in enumerate(radices):
        edge_code = edge_code * radix + csr.attr[:, j]

    # --- stereo codes (docs/superpowers/specs/2026-09-22-cis-trans-
    # featurisation-design.md; since 2026-09-23-stereo-refines-the-blind-
    # class-design.md the code refines the blind class of each WL level
    # instead of replacing it, see _union_level) ----------------------------
    # Recomputed every WL round from the content-rank fingerprint, never
    # stored: what reaches a signature row is the pair encoding below, which
    # merge can remap freely. The fingerprint itself folds the *static*
    # edge_code only (never a stereo code), which is what keeps the ordering
    # -- and therefore which substituent wins -- independent of mirroring or
    # remapping.
    stereo_radix = math.prod(config.stereo_radices)
    fingerprints: Iterator[np.ndarray] = iter(())
    # Bound once here rather than read off the batch in the loop: non-None
    # exactly when a track is enabled, so the loop's own `is not None` both
    # narrows the type and says the same thing as `if config.stereo`.
    stereo_bonds: np.ndarray | None = None
    pos_ab = pos_ba = None
    if config.stereo:
        if batch.stereo_bonds is None:
            raise ValueError(
                f"config.stereo is {list(config.stereo)} but the batch carries "
                "no stereo_bonds; the adapter was run with a stereo-blind config"
            )
        stereo_bonds = batch.stereo_bonds
        n_wl = sum(1 for k in kinds if k == LEVEL_WL)
        # Consumed one at a time below, never indexed: config refuses
        # stereo with neighbor_depth, so the WL levels are a single chain
        # and the radius the code reads rises by exactly one per round.
        fingerprints = content_ranks(batch.node_attrs, csr, edge_code, max(n_wl - 2, 0))
        pos_ab, pos_ba = directed_positions(csr, n, stereo_bonds)

    wl_round = 0
    for offset, kind in enumerate(kinds):
        base = levels[parents[offset]].labels
        if kind == LEVEL_WL:
            wl_round += 1
            if stereo_bonds is not None:
                # The gather reaches distance 2, so an honest code needs
                # radius-(k-2) identities. At k = 1 there is no such radius:
                # the far substituent is two bonds away, outside a radius-1
                # neighborhood, so the feature stays silent rather than
                # asserting something the level cannot support.
                stereo_code = np.zeros(edge_code.shape[0], np.int64)
                if wl_round >= 2:
                    codes = cis_trans_codes(stereo_bonds, next(fingerprints))
                    stereo_code[pos_ab] = codes
                    stereo_code[pos_ba] = codes
                # The trit refines the stereo-blind class rather than
                # replacing it (spec 2026-09-23). The blind row is built
                # recursively -- blind parent and neighbours, trit 0 -- or
                # stereo inherited through the ids one level down would
                # survive in it. Both share the modulus, so a trit of 0 makes
                # the two rows coincide exactly when they should.
                sig_aware = _wl_rows(
                    base, csr, edge_code * stereo_radix + stereo_code, n, n_edge_types
                )
                sig_blind = _wl_rows(
                    levels[parents[offset]].blind,
                    csr,
                    edge_code * stereo_radix,
                    n,
                    n_edge_types,
                )
                levels.append(_union_level(sig_aware, sig_blind))
                continue
            sig = _wl_rows(base, csr, edge_code, n, n_edge_types)
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
