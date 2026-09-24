"""Attribute levels then WL rounds, all vectorized (design.md 3.5, 7.1, 7.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import KIND_AWARE, KIND_BLIND, LEVEL_WL, STEREO_RADIX, SieveConfig
from sieve.dedupe import dense_rows
from sieve.stereo import (
    CODE_NONE,
    FingerprintWindow,
    Reach,
    advance_reach,
    centre_positions,
    cis_trans_codes,
    content_ranks,
    directed_positions,
    mirror_codes,
    tetrahedral_codes,
)


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
    # Set only on the WL levels of a tetrahedral-aware chain (docs/
    # superpowers/specs/2026-09-24-tetrahedral-handedness-design.md, section
    # 3): ``labels`` is then each atom's aware class and ``mirror_labels`` its
    # aware class in the enantiomer of its molecule, both ids in the same
    # vocabulary ``signatures``.
    mirror_labels: np.ndarray | None = None  # (n_nodes,) int64
    mirror_of: np.ndarray | None = None  # (n_classes,) int64
    # Under the tetrahedral track only: per atom, whether its aware row at
    # this level carries something the mirror quotient keeps (two or more
    # distinct centres, or a cis/trans code). predict consults an atom's
    # aware class only where this holds.
    stereo_informative: np.ndarray | None = None  # (n_nodes,) bool

    @property
    def n_classes(self) -> int:
        return int(self.signatures.shape[0])

    @property
    def blind(self) -> np.ndarray:
        """Each atom's stereo-blind class: its own class when no track is on."""
        return self.labels if self.blind_labels is None else self.blind_labels

    @property
    def mirror(self) -> np.ndarray:
        """Each atom's class in its molecule's enantiomer: its own class when
        the tetrahedral track is off."""
        return self.labels if self.mirror_labels is None else self.mirror_labels


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


def _union_level(
    sig_aware: np.ndarray,
    sig_blind: np.ndarray,
    sig_mirror: np.ndarray | None = None,
) -> LevelLabels:
    """One vocabulary holding every atom's aware, blind and mirror rows
    (design D spec, section 3; tetrahedral-handedness spec, section 3).

    Rows are deduplicated together, so an aware row equal to a blind row is
    the same class, flagged both -- the case for every atom with no stereo
    code within reach. Only the rows that differ are stacked: the rest would
    deduplicate onto their blind (or aware) twin anyway.

    ``sig_mirror``, when given, is each atom's aware row in the enantiomer of
    its molecule (tetrahedral-handedness spec, section 3). An M class is
    itself an aware class -- it never coincides with a blind class, since
    that would force the blind class to equal its own mirror's blind class,
    which it already does by construction -- and ``mirror_of`` maps every
    aware class to its M counterpart (the identity on an achiral one, where
    the aware and mirror rows coincide).
    """
    n = sig_blind.shape[0]
    differs = np.flatnonzero((sig_aware != sig_blind).any(axis=1))
    stack = [sig_blind, sig_aware[differs]]
    mdiff = None
    if sig_mirror is not None:
        # A mirror row differs from its aware row only where a handedness
        # code is in reach; elsewhere it would deduplicate onto it anyway.
        mdiff = np.flatnonzero((sig_mirror != sig_aware).any(axis=1))
        stack.append(sig_mirror[mdiff])
    labels, uniq = dense_rows(np.concatenate(stack))
    blind = labels[:n]
    aware = blind.copy()
    aware[differs] = labels[n : n + differs.size]
    kind = np.zeros(uniq.shape[0], np.uint8)
    kind[blind] |= KIND_BLIND
    kind[aware] |= KIND_AWARE
    # Well defined: an atom's aware class determines its blind one, since the
    # aware row names the aware classes one level down, whose blind classes
    # are themselves determined, and the trits only add to the edge code.
    blind_of = np.arange(uniq.shape[0], dtype=np.int64)
    blind_of[aware] = blind
    mirror = mirror_of = None
    if mdiff is not None:
        mirror = aware.copy()
        mirror[mdiff] = labels[n + differs.size :]
        # An M class is the aware class of an enantiomer's atom.
        kind[mirror] |= KIND_AWARE
        blind_of[mirror] = blind
        mirror_of = np.arange(uniq.shape[0], dtype=np.int64)
        mirror_of[aware] = mirror
        mirror_of[mirror] = aware
    return LevelLabels(
        aware,
        uniq,
        uniq[:, 0].astype(np.int32),
        blind,
        kind,
        blind_of,
        mirror,
        mirror_of,
    )


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
    # featurisation-design.md, 2026-09-23-stereo-refines-the-blind-class-
    # design.md, 2026-09-24-tetrahedral-handedness-design.md) --------------
    # Recomputed every WL round from the content-rank fingerprint, never
    # stored: what reaches a signature row is the pair encoding below, which
    # merge can remap freely. The fingerprint itself folds the *static*
    # edge_code only (never a stereo code), which is what keeps the ordering
    # -- and therefore which substituent wins -- independent of mirroring or
    # remapping.
    stereo_radix = math.prod(config.stereo_radices)
    cis_trans = "cis_trans" in config.stereo
    tetrahedral = "tetrahedral" in config.stereo
    # Bound once here rather than read off the batch in the loop: non-None
    # exactly when its own track is enabled, so the loop's own `is not None`
    # checks both narrow the type and say the same thing as the track flags.
    stereo_bonds: np.ndarray | None = None
    stereo_centres: np.ndarray | None = None
    pos_ab = pos_ba = tet_pos = tet_row = None
    window: FingerprintWindow | None = None
    reach: Reach | None = None
    if config.stereo:
        if cis_trans:
            if batch.stereo_bonds is None:
                raise ValueError(
                    f"config.stereo is {list(config.stereo)} but the batch "
                    "carries no stereo_bonds; the adapter was run with a "
                    "stereo-blind config"
                )
            stereo_bonds = batch.stereo_bonds
            pos_ab, pos_ba = directed_positions(csr, n, stereo_bonds)
        if tetrahedral:
            if batch.stereo_centres is None:
                raise ValueError(
                    f"config.stereo is {list(config.stereo)} but the batch "
                    "carries no stereo_centres; the adapter was run without "
                    "the tetrahedral track"
                )
            stereo_centres = batch.stereo_centres
            tet_pos, tet_row = centre_positions(csr, n, stereo_centres)
        n_wl = sum(1 for k in kinds if k == LEVEL_WL)
        # Round k reads fp_{k-1} (tetrahedral, one bond away) and fp_{k-2}
        # (cis/trans, two bonds away). config refuses stereo with
        # neighbor_depth, so the WL levels are a single chain and both radii
        # rise by exactly one per round -- FingerprintWindow serves both from
        # one generator consumed strictly in order.
        rounds = max(n_wl - 1, 0) if tetrahedral else max(n_wl - 2, 0)
        window = FingerprintWindow(
            content_ranks(batch.node_attrs, csr, edge_code, rounds)
        )

    def stereo_full(ct: np.ndarray, tet: np.ndarray) -> np.ndarray:
        """Fold the enabled tracks' codes into the edge code, one digit per
        track in ``STEREO_TRACKS`` order."""
        code = np.zeros(edge_code.shape[0], np.int64)
        if cis_trans:
            code = code * STEREO_RADIX + ct
        if tetrahedral:
            code = code * STEREO_RADIX + tet
        return edge_code * stereo_radix + code

    wl_round = 0
    for offset, kind in enumerate(kinds):
        base = levels[parents[offset]].labels
        if kind == LEVEL_WL:
            wl_round += 1
            if window is not None:
                e = edge_code.shape[0]
                ct = np.zeros(e, np.int64)
                tet = np.zeros(e, np.int64)
                tet_mirror = np.zeros(e, np.int64)
                tet_atoms = ez_atoms = np.zeros(0, np.int64)
                if stereo_centres is not None:
                    assert tet_pos is not None and tet_row is not None
                    # The ranked neighbours sit one bond away, so radius
                    # k-1 is honest and the code can fire from round 1.
                    tcodes = tetrahedral_codes(stereo_centres, window.at(wl_round - 1))
                    tet[tet_pos] = tcodes[tet_row]
                    tet_mirror[tet_pos] = mirror_codes(tcodes)[tet_row]
                    tet_atoms = stereo_centres[tcodes != CODE_NONE, 0]
                if stereo_bonds is not None and wl_round >= 2:
                    assert pos_ab is not None and pos_ba is not None
                    # The gather reaches distance 2, so an honest code needs
                    # radius-(k-2) identities. At k = 1 there is no such
                    # radius: the far substituent is two bonds away, outside
                    # a radius-1 neighborhood, so the feature stays silent
                    # rather than asserting something the level cannot
                    # support.
                    ccodes = cis_trans_codes(stereo_bonds, window.at(wl_round - 2))
                    ct[pos_ab] = ccodes
                    ct[pos_ba] = ccodes
                    fired = stereo_bonds[ccodes != CODE_NONE]
                    ez_atoms = np.concatenate([fired[:, 0], fired[:, 1]])
                # The trits refine the stereo-blind class rather than
                # replacing it (spec 2026-09-23). The blind row is built
                # recursively -- blind parent and neighbours, both trits 0 --
                # or stereo inherited through the ids one level down would
                # survive in it. All three rows share the modulus, so a trit
                # of 0 makes the blind row coincide with the aware one
                # exactly when it should.
                parent = levels[parents[offset]]
                zero = np.zeros(e, np.int64)
                sig_aware = _wl_rows(base, csr, stereo_full(ct, tet), n, n_edge_types)
                sig_blind = _wl_rows(
                    parent.blind, csr, stereo_full(zero, zero), n, n_edge_types
                )
                # The mirror row set (tetrahedral-handedness spec, section 3)
                # is each atom's aware row in the enantiomer of its molecule:
                # same cis/trans code (E/Z survives reflection), handedness
                # negated, and parent/neighbours from their own mirror
                # classes -- which is what makes a class and its mirror hold
                # identical statistics without ever materialising a mirrored
                # molecule.
                sig_mirror = (
                    _wl_rows(
                        parent.mirror, csr, stereo_full(ct, tet_mirror), n, n_edge_types
                    )
                    if stereo_centres is not None
                    else None
                )
                level = _union_level(sig_aware, sig_blind, sig_mirror)
                if stereo_centres is not None:
                    # Which codes have reached each atom's aware row by now
                    # (tetrahedral-handedness spec, section 5): a lone
                    # centre's sign is erased by the mirror quotient, so its
                    # aware class is not consulted (predict).
                    reach = advance_reach(reach, csr, n, tet_atoms, ez_atoms)
                    level = replace(level, stereo_informative=reach.informative)
                levels.append(level)
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
