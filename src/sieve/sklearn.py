"""A mutable-estimator wrapper around the immutable core (design.md 10.2).

Contorting the core into fit-mutates-self would forfeit the merge monoid for
the sake of an interface. This adapter exists so GridSearchCV can sweep alpha,
n_min and K without the core inheriting mutable-estimator semantics.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sieve.batch import AtomBatch
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

    def split(self, batch: AtomBatch, y=None, groups=None):
        graphs = np.unique(batch.graph_id)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(graphs)
        for part in np.array_split(graphs, self.n_splits):
            test_mask = np.isin(batch.graph_id, part)
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class SieveRegressor:
    def __init__(
        self, config: SieveConfig, n_min: int | None = None, alpha: float | None = None
    ):
        self.config = config
        self.n_min = config.n_min if n_min is None else n_min
        self.alpha = config.alpha if alpha is None else alpha
        self.model_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {"config": self.config, "n_min": self.n_min, "alpha": self.alpha}

    def set_params(self, **params) -> "SieveRegressor":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: AtomBatch, y=None) -> "SieveRegressor":
        cfg = replace(self.config, n_min=self.n_min, alpha=self.alpha)
        batch = X if y is None else AtomBatch(**{**X.__dict__, "y": y})
        self.model_ = _fit(batch, cfg)
        return self

    def predict(self, X: AtomBatch) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        return _predict(self.model_, X)

    def score(self, X: AtomBatch, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(1 - np.mean((y - pred) ** 2) / np.var(y))
