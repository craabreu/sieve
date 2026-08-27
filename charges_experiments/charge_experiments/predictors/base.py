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
from typing import ClassVar, Protocol

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
