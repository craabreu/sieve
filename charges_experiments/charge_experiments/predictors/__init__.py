"""Predictor registry. Lazy imports: a predictor module needing an optional
dependency (rdkit, the DASH-tree clone) registers itself via ``register``
only when its own module is imported. Only ``global_mean`` (no optional
deps) is registered eagerly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from charge_experiments.predictors.base import Predictor
from charge_experiments.predictors.global_mean import GlobalMeanPredictor

_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Predictor]] = {
    "global_mean": lambda params: GlobalMeanPredictor(**params),
}

REGISTRY: Mapping[str, Callable[[Mapping[str, Any]], Predictor]] = _BUILDERS


def register(name: str, builder: Callable[[Mapping[str, Any]], Predictor]) -> None:
    _BUILDERS[name] = builder


def build(name: str, params: Mapping[str, Any]) -> Predictor:
    # One branch per optional-dependency predictor module, added as built:
    # predictors/dash.py -> "dash" (Task 10), predictors/sieve_predictor.py
    # -> "sieve" (Task 12).
    if name == "dash" and name not in REGISTRY:
        import charge_experiments.predictors.dash
    if name == "sieve" and name not in REGISTRY:
        import charge_experiments.predictors.sieve_predictor  # noqa: F401
    if name not in REGISTRY:
        raise ValueError(f"unknown predictor {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
