"""Per-class statistics: a count and two means (design.md 4.1, 7.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from sieve.config import KIND_AWARE, KIND_BOTH
from sieve.refine import LevelLabels


@dataclass(frozen=True)
class FrozenLevel:
    """Immutable statistics for one refinement level.

    Stores ``(N, ybar, sigma^2)`` where ``sigma^2`` is the *population*
    variance (divisor N). The reported ``s^2`` is derived on access. ``count``
    and ``parent`` stay one-dimensional even for vector targets, which is what
    keeps the merge weights scalar.
    """

    signatures: np.ndarray  # (nc, width) int64 -- the vocabulary
    count: np.ndarray  # (nc,) int64
    mean: np.ndarray  # (nc, d) float64
    msd: np.ndarray  # (nc, d) float64 -- population variance
    parent: np.ndarray  # (nc,) int32
    # Under a stereo track only (see class_kinds / blind_targets): which kind
    # each class is, and the blind class of the same atoms at this level.
    kind: np.ndarray | None = None  # (nc,) uint8
    blind_of: np.ndarray | None = None  # (nc,) int64
    # Under the tetrahedral track only (see mirror_targets): the class of the
    # same atoms in the molecule's enantiomer.
    mirror_of: np.ndarray | None = None  # (nc,) int64

    @property
    def n_classes(self) -> int:
        return int(self.count.shape[0])

    @property
    def variance(self) -> np.ndarray:
        """Bessel-corrected s^2, NaN where N == 1 (design.md 4.1).

        A stored zero would be indistinguishable from a genuinely homogeneous
        class and would read as confidence in every downstream diagnostic. The
        guard lives here, in one accessor, rather than in every merge.
        """
        n = self.count.astype(np.float64)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            s2 = np.where(n > 1, self.msd * n / np.maximum(n - 1, 1), np.nan)
        return s2


def class_kinds(level) -> np.ndarray:
    """Per-class kind bits; a stored ``None`` means every class is both."""
    if level.kind is None:
        return np.full(level.n_classes, KIND_BOTH, np.uint8)
    return level.kind


def blind_targets(level) -> np.ndarray:
    """Per-class blind counterpart; a stored ``None`` means the identity."""
    if level.blind_of is None:
        return np.arange(level.n_classes, dtype=np.int64)
    return level.blind_of


def mirror_targets(level) -> np.ndarray:
    """Per-class mirror image; a stored ``None`` means the identity."""
    if level.mirror_of is None:
        return np.arange(level.n_classes, dtype=np.int64)
    return level.mirror_of


def _reduce(
    labels: np.ndarray, y: np.ndarray, nc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class count, mean and population variance, by a sparse membership
    operator.

    Two passes, centering before reducing. Never ``sum(y**2)/N - mean**2``: on
    targets with mean 1e6 and spread 3 that form errs by 1.3e+02 relative and
    produces negative variances, against 5.4e-08 for this one.
    """
    n = y.shape[0]

    # Built once, reused across both passes and all d dimensions. bincount is
    # scalar-only and would need a loop over dimensions.
    P = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(nc, n))

    count = np.bincount(labels, minlength=nc).astype(np.int64)
    safe = np.maximum(count, 1)[:, None].astype(np.float64)
    mean = (P @ y) / safe
    resid = y - mean[labels]  # center first, then reduce
    msd = (P @ (resid * resid)) / safe
    # Classes with no members must be exactly zero, not whatever the reduction
    # happened to leave there.
    empty = count == 0
    mean[empty] = 0.0
    msd[empty] = 0.0
    return count, mean, msd


def fit_level(level: LevelLabels, y: np.ndarray) -> FrozenLevel:
    """Reduce one chunk to per-class statistics.

    Under a stereo track each atom accrues to its blind class and, where it
    differs, to its aware class (spec 2026-09-23, section 4). Blind classes
    reduce over every atom in atom order, exactly as a stereo-blind fit does,
    so their statistics are bit-identical to it. An atom whose aware class
    differs from its blind one sits in an aware-only class, which therefore
    holds exactly those atoms.

    Under the tetrahedral track, an atom also accrues to its class's mirror,
    where that differs from its own class (tetrahedral-handedness spec,
    section 4): a class and its mirror end up with identical statistics,
    which is the fit on the corpus plus every molecule's enantiomer without
    ever materialising one. A self-mirror class -- ``mirror_of[c] == c`` --
    has ``mirror_labels[i] == labels[i]`` for every one of its atoms (the
    mirror label is a function of the class alone), so it is excluded from
    this second pass and never double-counted.
    """
    nc = level.n_classes
    count, mean, msd = _reduce(level.blind, y, nc)
    if level.kind is not None:
        differs = level.labels != level.blind
        extra_labels, extra_y = [level.labels[differs]], [y[differs]]
        if level.mirror_labels is not None:
            moved = level.mirror_labels != level.labels
            extra_labels.append(level.mirror_labels[moved])
            extra_y.append(y[moved])
        lab = np.concatenate(extra_labels)
        if lab.size:
            yy = np.concatenate(extra_y)
            c2, m2, s2 = _reduce(lab, yy, nc)
            only = level.kind == KIND_AWARE
            count[only], mean[only], msd[only] = c2[only], m2[only], s2[only]
    return FrozenLevel(
        level.signatures,
        count,
        mean,
        msd,
        level.parent,
        level.kind,
        level.blind_of,
        level.mirror_of,
    )


def global_stats(y: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Whole-corpus fallback statistics, same convention as a class."""
    return int(y.shape[0]), y.mean(axis=0), y.var(axis=0)
