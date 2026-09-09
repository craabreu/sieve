"""The simplest possible baseline: predict the training set's own mean
MBIScharge for every atom. No optional dependency, so this is the one
predictor registered eagerly (see predictors/__init__.py)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from experiments.data import MoleculeSet
from experiments.predictors.base import Prediction


class GlobalMeanPredictor:
    name: ClassVar[str] = "global_mean"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        if train.n_conformers == 0:
            raise ValueError("global_mean requires a non-empty train split")
        self._mean = float(np.mean(train.atom_target))

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._mean is None:
            raise RuntimeError("fit must be called before predict")
        atom_value: NDArray[np.float64] = np.full(test.n_atoms, self._mean)
        return Prediction(atom_value=atom_value)
