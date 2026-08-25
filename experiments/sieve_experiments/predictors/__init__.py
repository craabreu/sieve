"""Predictor registry.

Lazy imports: a predictor module that needs an optional dependency (rdkit,
cosmolayer, a subprocess venv, ...) registers itself via ``register`` only
when its own module is imported -- so importing ``sieve_experiments.predictors``
alone never pulls those dependencies in. Only ``global_mean`` (no optional
deps at all) is registered eagerly, here.

As predictors/dash.py and predictors/cosmonet.py are added, ``build`` grows
one more explicit lazy-import branch each -- see the comment below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sieve_experiments.predictors.base import Predictor
from sieve_experiments.predictors.global_mean import GlobalMeanPredictor

_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Predictor]] = {
    "global_mean": lambda params: GlobalMeanPredictor(**params),
}

REGISTRY: Mapping[str, Callable[[Mapping[str, Any]], Predictor]] = _BUILDERS


def register(name: str, builder: Callable[[Mapping[str, Any]], Predictor]) -> None:
    """Add a predictor to the registry. Called by a predictor module's own
    import (see ``build`` below), so it never runs until that name is asked for.
    """
    _BUILDERS[name] = builder


def build(name: str, params: Mapping[str, Any]) -> Predictor:
    # Add one branch per optional-dependency predictor module as it is built
    # (predictors/dash.py -> "dash_backoff", predictors/cosmonet.py ->
    # "cosmonet"); each module calls register() on import.
    if name == "dash_backoff" and name not in REGISTRY:
        import sieve_experiments.predictors.dash
    if name == "cosmonet" and name not in REGISTRY:
        import sieve_experiments.predictors.cosmonet
    if name == "chemprop_dmpnn" and name not in REGISTRY:
        import sieve_experiments.predictors.chemprop_dmpnn
    if name == "chemprop_atom" and name not in REGISTRY:
        import sieve_experiments.predictors.chemprop_atom  # noqa: F401
    if name not in REGISTRY:
        raise ValueError(f"unknown predictor {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
