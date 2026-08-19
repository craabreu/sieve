"""A mutable-estimator wrapper around the immutable core (design.md 10.2).

Contorting the core into fit-mutates-self would forfeit the merge monoid for
the sake of an interface. This adapter exists so GridSearchCV can sweep
shrinkage_strength, minimum_support and K without the core inheriting
mutable-estimator semantics.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from sieve.batch import NodeBatch
from sieve.config import SieveConfig
from sieve.model import fit as _fit
from sieve.predict import predict as _predict


class GraphKFold:
    """K-fold over *graphs*, never over nodes.

    Node-level random splitting puts WL-identical atoms from one molecule on
    both sides of the split and inflates scores badly.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 0):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, batch: NodeBatch, y=None, groups=None):
        graphs = np.unique(batch.graph_id)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(graphs)
        for part in np.array_split(graphs, self.n_splits):
            test_mask = np.isin(batch.graph_id, part)
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class SieveRegressor(BaseEstimator, RegressorMixin):
    """Inherits from sklearn's base classes purely for tag machinery.

    Modern scikit-learn (>=1.6) resolves estimator tags via
    ``__sklearn_tags__``, which only ``BaseEstimator`` provides -- without
    it, ``cross_val_score``/``GridSearchCV`` fail before ever calling `fit`.
    ``get_params``/``set_params`` stay explicit below rather than relying on
    ``BaseEstimator``'s signature introspection, since
    `minimum_support`/`shrinkage_strength` resolve from `config` at
    construction time.
    """

    def __init__(
        self,
        config: SieveConfig,
        minimum_support: int | None = None,
        shrinkage_strength: float | None = None,
    ):
        self.config = config
        self.minimum_support = (
            config.minimum_support if minimum_support is None else minimum_support
        )
        self.shrinkage_strength = (
            config.shrinkage_strength
            if shrinkage_strength is None
            else shrinkage_strength
        )
        self.model_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            "config": self.config,
            "minimum_support": self.minimum_support,
            "shrinkage_strength": self.shrinkage_strength,
        }

    def set_params(self, **params) -> SieveRegressor:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: NodeBatch, y=None) -> SieveRegressor:
        cfg = replace(
            self.config,
            minimum_support=self.minimum_support,
            shrinkage_strength=self.shrinkage_strength,
        )
        batch = X if y is None else NodeBatch(**{**X.__dict__, "y": y})
        self.model_ = _fit(batch, cfg)
        return self

    def predict(self, X: NodeBatch) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        return _predict(self.model_, X)

    # `score` is inherited from RegressorMixin: sklearn.metrics.r2_score
    # with the default multioutput="uniform_average", i.e. one R^2 per
    # target column, then averaged. A hand-rolled `1 - mse/np.var(y)` pools
    # every column's variance into one number instead, so a target column
    # with a much larger scale would dominate the score regardless of how
    # well the other columns were predicted.
