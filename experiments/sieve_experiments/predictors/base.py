"""The predictor seam: one interface, atom-level and molecule-level predictors.

``Predictor`` is a two-method protocol. ``Prediction`` requires only
``mol_profile``; every other field is optional, and ``metrics.molecule_metrics``
skips the corresponding metric block when a field is absent (see runner.py).
That is what lets a molecule-level model (COSMO-NET) and an atom-level model
(DASH, later Sieve) share one harness.

``AtomPredictor`` does the atom -> molecule rollup once, here, so no predictor
subclass reimplements it. Charge reconciliation is likewise defined once:
per-atom charges are adjusted so their molecule sum matches the known
``screening_charge`` target (``-net_charge`` -- see ``MoleculeSet.
screening_charge``'s docstring for why sigma-derived charge and the
molecule's own formal charge are opposite in sign), and the reconciled
total is re-summed into ``mol_charge``. ``mol_charge_raw`` -- the total
*before* reconciliation -- is what the charge metrics score, since that is
the honest answer to "how well would the raw output have reproduced the
given total."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from sieve_experiments.data import MoleculeSet, molecule_sum

VALID_CHARGE_RECONCILIATION = ("none", "shift", "std_weighted")


@dataclass(frozen=True)
class AtomPrediction:
    """What an ``AtomPredictor`` returns before rollup."""

    atom_profile: NDArray[np.float64]
    atom_area: NDArray[np.float64] | None = None
    atom_charge: NDArray[np.float64] | None = None
    atom_charge_std: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class Prediction:
    """What every predictor returns, at molecule level.

    Only ``mol_profile`` is required -- see module docstring.
    """

    mol_profile: NDArray[np.float64]
    mol_area: NDArray[np.float64] | None = None
    mol_charge_raw: NDArray[np.float64] | None = None
    mol_charge: NDArray[np.float64] | None = None
    atom_profile: NDArray[np.float64] | None = None
    atom_area: NDArray[np.float64] | None = None
    atom_charge: NDArray[np.float64] | None = None


class Predictor(Protocol):
    name: ClassVar[str]

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None: ...

    def predict(self, test: MoleculeSet) -> Prediction: ...


def reconcile_charge(
    atom_charge: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    target_charge: NDArray[np.floating],
    n_molecules: int,
    *,
    mode: str,
    atom_charge_std: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Adjust per-atom charges so each molecule's sum matches ``target_charge``.

    - "none": identity, no adjustment.
    - "shift": distribute the residual evenly across a molecule's atoms.
    - "std_weighted": distribute the residual proportionally to each atom's
      predicted charge std (requires ``atom_charge_std``) -- the same scheme
      DASHTree.get_molecules_partial_charges uses.

    After reconciliation each molecule's summed charge equals
    ``target_charge`` to floating-point precision, by construction. Callers
    reconciling a sigma-derived atom charge (the convention every predictor
    here uses) must pass ``MoleculeSet.screening_charge``, not
    ``net_charge`` directly -- see that property's docstring for why they
    differ by a sign.
    """
    if mode not in VALID_CHARGE_RECONCILIATION:
        raise ValueError(
            f"charge_reconciliation must be one of {VALID_CHARGE_RECONCILIATION}, "
            f"got {mode!r}"
        )
    atom_charge = np.asarray(atom_charge, dtype=np.float64)
    if mode == "none":
        return atom_charge

    current = molecule_sum(atom_charge, mol_id, n_molecules)
    residual = np.asarray(target_charge, dtype=np.float64) - current
    num_atoms_per_mol = np.bincount(mol_id, minlength=n_molecules)

    if mode == "shift":
        per_atom_share = (residual / num_atoms_per_mol)[mol_id]
        return atom_charge + per_atom_share

    # std_weighted
    if atom_charge_std is None:
        raise ValueError("std_weighted reconciliation requires atom_charge_std")
    std = np.clip(np.asarray(atom_charge_std, dtype=np.float64), 1e-12, None)
    mol_std_total = molecule_sum(std, mol_id, n_molecules)
    weight = std / mol_std_total[mol_id]
    return atom_charge + weight * residual[mol_id]


def roll_up(
    atom_pred: AtomPrediction,
    test: MoleculeSet,
    *,
    charge_reconciliation: str = "none",
) -> Prediction:
    """Sum an ``AtomPrediction`` up to molecule level (the only definition)."""
    mol_id = test.atom_mol_id
    n = test.n_molecules

    mol_profile = molecule_sum(atom_pred.atom_profile, mol_id, n)

    mol_area = None
    if atom_pred.atom_area is not None:
        mol_area = molecule_sum(atom_pred.atom_area, mol_id, n)

    mol_charge_raw = None
    mol_charge = None
    atom_charge_out = atom_pred.atom_charge
    if atom_pred.atom_charge is not None:
        mol_charge_raw = molecule_sum(atom_pred.atom_charge, mol_id, n)
        atom_charge_out = reconcile_charge(
            atom_pred.atom_charge,
            mol_id,
            test.screening_charge,
            n,
            mode=charge_reconciliation,
            atom_charge_std=atom_pred.atom_charge_std,
        )
        mol_charge = molecule_sum(atom_charge_out, mol_id, n)

    return Prediction(
        mol_profile=mol_profile,
        mol_area=mol_area,
        mol_charge_raw=mol_charge_raw,
        mol_charge=mol_charge,
        atom_profile=atom_pred.atom_profile,
        atom_area=atom_pred.atom_area,
        atom_charge=atom_charge_out,
    )


class AtomPredictor(ABC):
    """Base for predictors that predict per atom (DASH, later Sieve).

    Subclasses implement ``fit_atoms``/``predict_atoms``; ``predict`` (final)
    does the rollup via ``roll_up``, so no subclass reimplements it.
    """

    name: ClassVar[str]
    charge_reconciliation: str = "none"

    @abstractmethod
    def fit_atoms(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None: ...

    @abstractmethod
    def predict_atoms(self, test: MoleculeSet) -> AtomPrediction: ...

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        self.fit_atoms(train, val, rng=rng)

    def predict(self, test: MoleculeSet) -> Prediction:
        atom_pred = self.predict_atoms(test)
        return roll_up(
            atom_pred, test, charge_reconciliation=self.charge_reconciliation
        )


class MoleculePredictor(ABC):
    """Base for predictors that predict per molecule directly (COSMO-NET)."""

    name: ClassVar[str]

    @abstractmethod
    def fit_molecules(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None: ...

    @abstractmethod
    def predict_molecules(self, test: MoleculeSet) -> Prediction: ...

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        self.fit_molecules(train, val, rng=rng)

    def predict(self, test: MoleculeSet) -> Prediction:
        return self.predict_molecules(test)
