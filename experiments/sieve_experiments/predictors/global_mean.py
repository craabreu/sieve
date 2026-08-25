"""The floor predictor: every molecule's profile is the training mean atom
profile, scaled by that molecule's own atom count.

Two jobs: the fixture every real baseline must beat, and the fixture the
fast, no-optional-dependency smoke test
(experiments/tests/test_experiment_smoke.py) runs end to end. A
``MoleculePredictor`` rather than an ``AtomPredictor`` on purpose -- it only
needs molecule-level truth (always present), never atom-level truth (only
present when a predictor computed it itself).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sieve_experiments.data import MoleculeSet
from sieve_experiments.predictors.base import MoleculePredictor, Prediction


class GlobalMeanPredictor(MoleculePredictor):
    name = "global_mean"

    def __init__(self) -> None:
        self._mean_atom_profile: NDArray[np.float64] | None = None

    def fit_molecules(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng  # unused: this predictor has nothing to tune or seed
        if train.mol_profile is None:
            raise ValueError("global_mean requires train.mol_profile")
        total_atoms = int(train.num_atoms.sum())
        if total_atoms == 0:
            raise ValueError("cannot fit global_mean on an empty training set")
        # Area-weighted mean per-atom profile: total profile mass divided by
        # total atom count, so scaling by a molecule's own atom count
        # reproduces the (unnormalized) total profile scale on average.
        self._mean_atom_profile = train.mol_profile.sum(axis=0) / total_atoms

    def predict_molecules(self, test: MoleculeSet) -> Prediction:
        if self._mean_atom_profile is None:
            raise RuntimeError("fit_molecules must be called before predict_molecules")
        mol_profile = self._mean_atom_profile[None, :] * test.num_atoms[:, None]
        mol_area = mol_profile.sum(axis=1)
        # The naive floor: predict 0 screening charge, then reconcile to the
        # known target, as any real predictor's output would be. Note the
        # target is screening_charge (-net_charge), not net_charge itself --
        # see MoleculeSet.screening_charge's docstring.
        mol_charge_raw = np.zeros(test.n_molecules)
        return Prediction(
            mol_profile=mol_profile,
            mol_area=mol_area,
            mol_charge_raw=mol_charge_raw,
            mol_charge=test.screening_charge,
        )
