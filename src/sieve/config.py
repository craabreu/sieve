"""Fit-time and inference-time configuration. See design.md section 9.2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

FORMAT_VERSION = 5

# Level kinds returned by SieveConfig.level_kinds (design.md 3.6).
LEVEL_ATTR = "attr"
LEVEL_WL = "wl"
LEVEL_WL_PAIR = "wl_pair"

# How a class's estimate is formed from what is stored beneath it.
#
#   "pooled"       the class's own atom-weighted mean -- every atom under it
#                  counted once. The original rule.
#   "continuation" the unweighted mean of the class's *children's* means, so
#                  each distinct child environment counts once regardless of
#                  how many atoms it holds; a class with no children keeps its
#                  pooled mean. The Kneser-Ney continuation-count correction
#                  [KneserNey1995Improved; ChenGoodman1999Smoothing], see
#                  literature.md 4.2.
#   "continuation_recursive"
#                  as above, but each level averages its children's own
#                  *continuation* estimates rather than their stored pooled
#                  means -- the recursive base-measure construction of the
#                  hierarchical Bayesian form [Teh2006HierarchicalPitmanYor].
#                  Computed deepest-level-first. **Measured: no effect.** On
#                  DASH charges (element, depth 6, 10 folds) it lands within
#                  1e-6 of the flat form at every shrinkage setting tried
#                  (delta -0.000001, t = -0.7, 3/10 folds). Recursion only
#                  alters levels two or more steps above the deepest, which
#                  hold little of the error. Kept because it makes the
#                  comparison reproducible, not because it is preferable --
#                  "continuation" remains the default for being simpler.
CLASS_ESTIMATOR_POOLED = "pooled"
CLASS_ESTIMATOR_CONTINUATION = "continuation"
CLASS_ESTIMATOR_CONTINUATION_RECURSIVE = "continuation_recursive"
CLASS_ESTIMATORS = (
    CLASS_ESTIMATOR_POOLED,
    CLASS_ESTIMATOR_CONTINUATION,
    CLASS_ESTIMATOR_CONTINUATION_RECURSIVE,
)

# How shrinkage weights a class's own estimate against its shrunk parent
# (design.md 4.2, 4.4).
#
# ``None`` (the default) means no shrinkage at all; naming a rule is how
# shrinkage is requested. A bare ``shrinkage_strength`` with no weight still
# means "count", so configs predating this field are unchanged.
#
#   "count"     w = N / (N + alpha) -- the original rule, a function of the
#               class's atom count alone. Requires a shrinkage_strength.
#   "diversity" lambda = min(alpha * C / N, 1) is the weight on the *parent*,
#               with C the number of children. This is interpolated
#               Kneser-Ney's own lambda = D * N_1+(context) / c(context)
#               [KneserNey1995Improved], so `shrinkage_strength` reads as
#               KN's discount D rather than as a pseudo-count. A class whose
#               atoms are spread over many distinct children defers harder to
#               its parent than an equally-sized class concentrated in a few.
#               At the deepest level C == 0, so lambda == 0 and no shrinkage
#               is applied -- see design.md 4.4 on why that is defensible
#               rather than a degenerate case.
#   "empirical_bayes"
#               the normal-normal posterior weight with the estimand's own
#               unit of replication, and alpha estimated rather than swept
#               [Morris1983EmpiricalBayes]. Under continuation a non-deepest
#               class estimates its *typical child* mean, whose unit is the
#               child, giving w = C/(C+alpha_k) with
#               alpha_k = tau^2_k / tau^2_{k-1}; the deepest class has no
#               children and estimates its own pooled mean, so ordinary
#               atom-level EB applies there, w = N/(N+sigma^2/tau^2_{k-1}).
#               tau^2 is pooled per level (design.md 13 item 9's "it comes
#               out per level"): at C = 2-3 a per-class variance has 1-2
#               degrees of freedom and is mostly noise. `shrinkage_strength`
#               must be None in this mode -- supplying one is refused rather
#               than ignored, since alpha is estimated.
#               **Measured best**: +0.000105 vs flat continuation without
#               shrinkage (t=21.7, 10/10 folds), beating the swept count rule
#               by +0.000024 (t=16.6, 10/10) while removing its knob.
#               `"count"` stays the default only for backward compatibility.
SHRINKAGE_WEIGHT_COUNT = "count"
SHRINKAGE_WEIGHT_DIVERSITY = "diversity"
SHRINKAGE_WEIGHT_EMPIRICAL_BAYES = "empirical_bayes"
SHRINKAGE_WEIGHTS = (
    SHRINKAGE_WEIGHT_COUNT,
    SHRINKAGE_WEIGHT_DIVERSITY,
    SHRINKAGE_WEIGHT_EMPIRICAL_BAYES,
)


@dataclass(frozen=True)
class SieveConfig:
    """Immutable configuration for a Sieve model.

    ``attribute_levels`` declares the graded refinement order below WL depth 0
    (design.md section 3.5): each tuple is one level, introducing that group of
    attributes on top of the previous level.

    ``attribute_codes`` and ``edge_codes`` are part of what a class *means*, so
    they enter ``schema_version``, and so does ``edge_attributes`` -- it fixes
    which edge column is which: two models built with different encodings or a
    different column order cannot be merged even if every other field agrees.
    """

    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]
    attribute_codes: Mapping[str, Mapping[str, int]]
    edge_codes: Mapping[str, Mapping[str, int]]
    max_wl_depth: int
    edge_attributes: tuple[str, ...] = ("bond_type",)
    neighbor_depth: int | None = None
    minimum_support: int = 1
    shrinkage_strength: float | None = None
    class_estimator: str = CLASS_ESTIMATOR_POOLED
    shrinkage_weight: str | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if not self.attribute_levels:
            raise ValueError("at least one attribute level is required")
        if any(not group for group in self.attribute_levels):
            # A zero-width group makes refine()'s dense_rows() degrade
            # silently (an (n, 0) signature dedupes to zero classes instead
            # of one, breaking every array downstream) instead of raising
            # anywhere near the actual mistake.
            raise ValueError("each attribute level must declare >= 1 attribute")
        if self.target_dim < 1:
            raise ValueError("target_dim must be >= 1")
        if self.minimum_support < 1:
            raise ValueError("minimum_support must be >= 1")
        if self.class_estimator not in CLASS_ESTIMATORS:
            raise ValueError(
                f"class_estimator must be one of {CLASS_ESTIMATORS}, "
                f"got {self.class_estimator!r}"
            )
        if self.shrinkage_weight is not None:
            if self.shrinkage_weight not in SHRINKAGE_WEIGHTS:
                raise ValueError(
                    f"shrinkage_weight must be None or one of "
                    f"{SHRINKAGE_WEIGHTS}, got {self.shrinkage_weight!r}"
                )
            if self.shrinkage_weight == SHRINKAGE_WEIGHT_EMPIRICAL_BAYES:
                if self.shrinkage_strength is not None:
                    # Refuse rather than ignore: alpha is estimated here, so a
                    # supplied strength would silently do nothing, and a caller
                    # who set one plainly expected it to matter.
                    raise ValueError(
                        "shrinkage_weight='empirical_bayes' estimates its own "
                        "alpha; shrinkage_strength must be None, got "
                        f"{self.shrinkage_strength!r}"
                    )
            elif self.shrinkage_strength is None:
                raise ValueError(
                    f"shrinkage_weight={self.shrinkage_weight!r} needs a "
                    "shrinkage_strength; only 'empirical_bayes' estimates one"
                )
        if self.neighbor_depth is not None:
            a = len(self.attribute_levels)
            if not (1 <= self.neighbor_depth <= a):
                raise ValueError(
                    f"neighbor_depth must be between 1 and {a} "
                    f"(len(attribute_levels)), got {self.neighbor_depth}"
                )
            # "No coarsening" gets exactly one spelling, so configs that
            # behave identically also hash identically. Two ways to say it:
            # keeping every attribute level, or having no WL round for a
            # neighbor to be seen through in the first place -- with
            # max_wl_depth == 0 nothing ever reads a neighbor, so the coarse
            # chain would be zero levels long and pure dead weight (it also
            # left level_parents indexing off the end of its own list).
            if self.neighbor_depth == a or self.max_wl_depth == 0:
                object.__setattr__(self, "neighbor_depth", None)
        # An empty edge schema is legal -- it is the pure-topology control arm
        # -- so the zero-width rule above deliberately does not extend here.
        object.__setattr__(self, "edge_attributes", tuple(self.edge_attributes))
        if len(set(self.edge_attributes)) != len(self.edge_attributes):
            raise ValueError(f"edge_attributes has a duplicate: {self.edge_attributes}")
        if set(self.edge_attributes) != set(self.edge_codes):
            raise ValueError(
                "edge_attributes and edge_codes must name the same attributes: "
                f"{sorted(self.edge_attributes)} != {sorted(self.edge_codes)}"
            )
        self._freeze_mappings()

    def _freeze_mappings(self) -> None:
        """Wrap ``attribute_codes``/``edge_codes`` in ``MappingProxyType``.

        Shared by ``__post_init__`` and ``__setstate__`` so freezing is one
        piece of logic rather than two copies that can drift apart.
        """
        object.__setattr__(
            self,
            "attribute_codes",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.attribute_codes.items()}
            ),
        )
        object.__setattr__(
            self,
            "edge_codes",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.edge_codes.items()}
            ),
        )

    def __deepcopy__(self, memo: dict) -> SieveConfig:
        """Immutable, so a deep copy is never observably different from self.

        `copy.deepcopy` has no built-in support for `MappingProxyType`
        (`attribute_codes`/`edge_codes` are frozen into one in
        `__post_init__`) and raises on it -- which otherwise breaks anything
        that deep-copies a config, including scikit-learn's `clone()`
        (design.md 10.2), used internally by `cross_val_score`/`GridSearchCV`.
        """
        return self

    def __getstate__(self) -> dict:
        """Unwrap the ``MappingProxyType``s for pickling.

        The stdlib ``pickle`` protocol has no support for ``MappingProxyType``
        either (same root cause as ``__deepcopy__`` above), which otherwise
        breaks ``multiprocessing`` -- parallel fitting across shards (design.md
        5.1: "parallel and distributed fitting are a fold over independently
        fitted shards") needs a config that survives the trip to a worker
        process.
        """
        state = self.__dict__.copy()
        state["attribute_codes"] = {k: dict(v) for k, v in self.attribute_codes.items()}
        state["edge_codes"] = {k: dict(v) for k, v in self.edge_codes.items()}
        return state

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self._freeze_mappings()

    @property
    def n_levels(self) -> int:
        """Total refinement levels: attribute levels, then WL depths --
        doubled when ``neighbor_depth`` is set, since neighbors then get
        their own coarse WL chain (design.md 3.6) alongside the main one
        (see ``level_kinds``)."""
        extra = self.max_wl_depth if self.neighbor_depth is not None else 0
        return len(self.attribute_levels) + self.max_wl_depth + extra

    @property
    def applies_shrinkage(self) -> bool:
        """Whether the shrinkage pass does anything at all.

        One predicate, read by both ``shrinkage.shrunk_means`` and
        ``predict._search``, because they must agree: a config that shrinks in
        one and not the other silently produces two different models. That is
        not hypothetical -- widening ``_search``'s guard without widening this
        made ``shrinkage_weight="diversity"`` with the default
        ``shrinkage_strength=None`` crash in ``predict`` while
        ``shrunk_means`` quietly skipped shrinkage.

        ``"empirical_bayes"`` needs no strength: alpha is estimated, so the
        mode is always active. The other rules need a strength to scale.
        """
        if self.shrinkage_weight is not None:
            return True  # naming a rule is the request; validation ensures it is usable
        return self.shrinkage_strength is not None and self.shrinkage_strength != 0.0

    @property
    def effective_shrinkage_weight(self) -> str:
        """Which rule actually runs when ``applies_shrinkage``.

        ``shrinkage_weight=None`` with a bare ``shrinkage_strength`` keeps
        meaning the count rule, so configs written before the field existed
        behave exactly as they did.
        """
        if self.shrinkage_weight is None:
            return SHRINKAGE_WEIGHT_COUNT
        return self.shrinkage_weight

    @property
    def level_kinds(self) -> tuple[str, ...]:
        """One kind per level: ``LEVEL_ATTR``, ``LEVEL_WL`` (an ordinary
        sorted-multiset WL round), or ``LEVEL_WL_PAIR`` (a main-chain WL
        round whose neighbor identity comes from the coarse chain instead of
        its own previous round -- design.md 3.6).

        The single source of truth for level shape: ``refine``/``merge``/
        ``predict`` all read this instead of re-deriving it from level
        indices.
        """
        a = len(self.attribute_levels)
        kinds = [LEVEL_ATTR] * a + [LEVEL_WL] * self.max_wl_depth
        if self.neighbor_depth is not None:
            kinds += [LEVEL_WL_PAIR] * self.max_wl_depth
        return tuple(kinds)

    @property
    def level_parents(self) -> tuple[int, ...]:
        """Absolute index of the level each level's ``parent`` ids refer
        into (``-1`` at level 0, meaning "no parent"). ``k - 1`` everywhere
        except the two branch roots of a coarsened chain: the coarse chain's
        first WL round branches off attribute level ``neighbor_depth - 1``,
        and the main chain's first WL round still branches off the last
        attribute level -- both instead of the level immediately before
        them in the tuple.
        """
        a = len(self.attribute_levels)
        depth = self.max_wl_depth
        parents = [j - 1 for j in range(a)] + [a + r - 1 for r in range(depth)]
        if self.neighbor_depth is not None:
            parents[a] = self.neighbor_depth - 1  # coarse chain's own root
            first_h = a + depth
            parents += [(a - 1 if r == 0 else first_h + r - 1) for r in range(depth)]
        return tuple(parents)

    @property
    def neighbor_source(self) -> tuple[int | None, ...]:
        """For each ``LEVEL_WL_PAIR`` level, the absolute index of the
        coarse chain's level supplying its neighbor identity; ``None``
        everywhere else."""
        a = len(self.attribute_levels)
        depth = self.max_wl_depth
        src: list[int | None] = [None] * (a + depth)
        if self.neighbor_depth is not None:
            src += [a + r for r in range(depth)]
        return tuple(src)

    @property
    def backoff_path(self) -> tuple[int, ...]:
        """Absolute indices of the levels ``predict`` actually backs off
        over, in matching order -- the attribute levels then the main WL
        chain. Equal to ``range(n_levels)`` unless ``neighbor_depth`` is
        set, in which case it skips the coarse chain's own levels
        (scaffolding only, never a backoff target itself)."""
        a = len(self.attribute_levels)
        depth = self.max_wl_depth
        if self.neighbor_depth is None:
            return tuple(range(a + depth))
        first_h = a + depth
        return tuple(range(a)) + tuple(range(first_h, first_h + depth))

    @property
    def edge_radices(self) -> tuple[int, ...]:
        """Alphabet size per edge attribute, in ``edge_attributes`` order,
        including the code reserved for an unseen value."""
        return tuple(len(self.edge_codes[n]) + 1 for n in self.edge_attributes)

    @property
    def n_edge_types(self) -> int:
        """Size of the collapsed edge alphabet -- the modulus ``refine``,
        ``merge`` and ``predict`` encode a (neighbor label, edge) pair with.

        A product over ``edge_radices`` because the columns collapse mixed
        radix. The empty product is 1, which is exactly right for an empty
        edge schema: every edge collapses to code 0 and the pair encoding
        degenerates to the neighbor label alone.
        """
        return math.prod(self.edge_radices)

    @property
    def schema_version(self) -> str:
        """Digest over everything that changes what a class means (design.md 9.2).

        ``minimum_support``, ``shrinkage_strength``, ``class_estimator``,
        ``shrinkage_weight`` and ``chunk_size`` are deliberately excluded:
        they are read at prediction time and do not invalidate fitted
        statistics. ``class_estimator``/``shrinkage_weight`` belong here
        rather than in the digest because they only change which stored
        numbers an estimate is *read from*, and how they are combined -- the
        classes themselves, and every count and mean in them, are identical
        either way.
        """
        payload = {
            "target_dim": self.target_dim,
            "attribute_levels": [list(g) for g in self.attribute_levels],
            "attribute_codes": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.attribute_codes.items())
            },
            "edge_attributes": list(self.edge_attributes),
            "edge_codes": {
                k: dict(sorted(v.items())) for k, v in sorted(self.edge_codes.items())
            },
            "max_wl_depth": self.max_wl_depth,
            "neighbor_depth": self.neighbor_depth,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def check_mergeable(a: SieveConfig, b: SieveConfig) -> None:
    """Raise unless two models describe the same classes (design.md 5.4).

    Loud rejection is the point: silently truncating to ``min(K_a, K_b)`` would
    absorb config drift and produce a model whose classes mean two things.
    """
    if a.schema_version != b.schema_version:
        raise ValueError(
            f"cannot merge: schema_version differs ({a.schema_version[:12]} != "
            f"{b.schema_version[:12]}); attribute levels, codes, edge attributes, "
            "edge codes and max_wl_depth must all match"
        )
