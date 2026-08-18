"""Fit-time and inference-time configuration. See design.md section 9.2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

FORMAT_VERSION = 1


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
    neighbour_schema: tuple[str, ...] | None = None
    n_min: int = 1
    alpha: float | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.neighbour_schema is not None:
            raise NotImplementedError(
                "neighbour_schema is evaluated but not adopted; see design.md 3.6"
            )
        if not self.attribute_levels:
            raise ValueError("at least one attribute level is required")
        if self.target_dim < 1:
            raise ValueError("target_dim must be >= 1")
        if self.n_min < 1:
            raise ValueError("n_min must be >= 1")
        # Freeze the mappings so the frozen dataclass is honest.
        object.__setattr__(
            self,
            "attribute_codes",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.attribute_codes.items()}
            ),
        )
        object.__setattr__(self, "edge_codes", MappingProxyType(dict(self.edge_codes)))

    @property
    def n_levels(self) -> int:
        """Total refinement levels: attribute levels, then WL depths."""
        return len(self.attribute_levels) + self.max_wl_depth

    @property
    def n_bond(self) -> int:
        """Edge-code alphabet size, including the 0 slot reserved for padding."""
        return max(self.edge_codes.values()) + 1

    @property
    def schema_version(self) -> str:
        """Digest over everything that changes what a class means (design.md 9.2).

        ``n_min``, ``alpha`` and ``chunk_size`` are deliberately excluded: they
        are read at prediction time and do not invalidate fitted statistics.
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
            "neighbour_schema": self.neighbour_schema,
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
