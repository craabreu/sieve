"""Fit-time and inference-time configuration. See design.md section 9.2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

FORMAT_VERSION = 2


@dataclass(frozen=True)
class SieveConfig:
    """Immutable configuration for a Sieve model.

    ``attribute_levels`` declares the graded refinement order below WL depth 0
    (design.md section 3.5): each tuple is one level, introducing that group of
    attributes on top of the previous level.

    ``attribute_codes`` and ``edge_codes`` are part of what a class *means*, so
    they enter ``schema_version``: two models built with different encodings
    cannot be merged even if every other field agrees.
    """

    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]
    attribute_codes: Mapping[str, Mapping[str, int]]
    edge_codes: Mapping[str, int]
    max_wl_depth: int
    neighbor_schema: tuple[str, ...] | None = None
    minimum_support: int = 1
    shrinkage_strength: float | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.neighbor_schema is not None:
            raise NotImplementedError(
                "neighbor_schema is evaluated but not adopted; see design.md 3.6"
            )
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
        object.__setattr__(self, "edge_codes", MappingProxyType(dict(self.edge_codes)))

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
        state["edge_codes"] = dict(self.edge_codes)
        return state

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self._freeze_mappings()

    @property
    def n_levels(self) -> int:
        """Total refinement levels: attribute levels, then WL depths."""
        return len(self.attribute_levels) + self.max_wl_depth

    @property
    def n_edge_types(self) -> int:
        """Edge-code alphabet size, including the 0 slot reserved for padding."""
        return max(self.edge_codes.values()) + 1

    @property
    def schema_version(self) -> str:
        """Digest over everything that changes what a class means (design.md 9.2).

        ``minimum_support``, ``shrinkage_strength`` and ``chunk_size`` are
        deliberately excluded: they are read at prediction time and do not
        invalidate fitted statistics.
        """
        payload = {
            "target_dim": self.target_dim,
            "attribute_levels": [list(g) for g in self.attribute_levels],
            "attribute_codes": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.attribute_codes.items())
            },
            "edge_codes": dict(sorted(self.edge_codes.items())),
            "max_wl_depth": self.max_wl_depth,
            "neighbor_schema": self.neighbor_schema,
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
            f"{b.schema_version[:12]}); attribute levels, codes, edge codes and "
            "max_wl_depth must all match"
        )
