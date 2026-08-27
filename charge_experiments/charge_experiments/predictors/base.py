"""The predictor seam: one interface, one scalar-per-atom output.

Unlike cosmo_experiments' base.py, there is no AtomPredictor/
MoleculePredictor split and no profile/area/charge rollup machinery: every
predictor in this series predicts one scalar (MBIScharge) per atom
directly, and the molecule-level charge-conservation check
(metrics.charge_conservation_metrics) is computed by the caller
(runner.py), not by the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet


@dataclass(frozen=True)
class Prediction:
    """What every predictor returns: one predicted ``MBIScharge`` per atom,
    in ``test``'s own atom order (``test.atom_mol_id``-aligned)."""

    atom_charge: NDArray[np.float64]


class Predictor(Protocol):
    name: ClassVar[str]

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None: ...

    def predict(self, test: MoleculeSet) -> Prediction: ...


@dataclass(frozen=True)
class RawPrediction:
    """What a normalizable predictor's own ``predict_raw`` returns: the
    unnormalized per-atom walk output, plus its per-atom std (needed by
    ``normalize.std_weighted_normalize``) -- both required before any
    function in ``normalize.NORMALIZERS`` can be applied."""

    atom_charge: NDArray[np.float64]
    atom_std: NDArray[np.float64]


@runtime_checkable
class NormalizableChargePredictor(Predictor, Protocol):
    """A ``Predictor`` that can also return its raw, unnormalized walk
    output separately -- ``predictors/dash.py``'s ``DASHChargePredictor`` and
    ``predictors/dash_pretrained.py``'s ``DASHPretrainedChargePredictor``
    both implement this; ``nested_runner.py`` requires it."""

    def predict_raw(self, test: MoleculeSet) -> RawPrediction: ...
