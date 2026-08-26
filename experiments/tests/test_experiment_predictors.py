"""Tests for the predictor seam: base.py's rollup/reconciliation, the
registry, and the global_mean floor predictor."""

from __future__ import annotations

import numpy as np
import pytest
from sieve_experiments.predictors import REGISTRY, build
from sieve_experiments.predictors.base import (
    AtomPrediction,
    reconcile_charge,
    roll_up,
)
from sieve_experiments.predictors.global_mean import GlobalMeanPredictor

from experiments.tests.helpers import synthetic_molecule_set

# --- reconcile_charge ------------------------------------------------------


def _toy_charges():
    mol_id = np.array([0, 0, 1, 1, 1])
    atom_charge = np.array([0.3, -0.1, 0.05, -0.2, 0.1])
    net_charge = np.array([1.0, -1.0])
    return atom_charge, mol_id, net_charge


def test_reconcile_charge_none_is_identity():
    atom_charge, mol_id, net_charge = _toy_charges()
    out = reconcile_charge(atom_charge, mol_id, net_charge, 2, mode="none")
    np.testing.assert_allclose(out, atom_charge)


def test_reconcile_charge_shift_hits_net_charge_exactly():
    from sieve_experiments.data import molecule_sum

    atom_charge, mol_id, net_charge = _toy_charges()
    out = reconcile_charge(atom_charge, mol_id, net_charge, 2, mode="shift")
    np.testing.assert_allclose(molecule_sum(out, mol_id, 2), net_charge, atol=1e-10)


def test_reconcile_charge_std_weighted_hits_net_charge_exactly():
    from sieve_experiments.data import molecule_sum

    atom_charge, mol_id, net_charge = _toy_charges()
    std = np.array([1.0, 2.0, 0.5, 0.5, 3.0])
    out = reconcile_charge(
        atom_charge, mol_id, net_charge, 2, mode="std_weighted", atom_charge_std=std
    )
    np.testing.assert_allclose(molecule_sum(out, mol_id, 2), net_charge, atol=1e-10)


def test_reconcile_charge_std_weighted_requires_std():
    atom_charge, mol_id, net_charge = _toy_charges()
    with pytest.raises(ValueError, match="atom_charge_std"):
        reconcile_charge(atom_charge, mol_id, net_charge, 2, mode="std_weighted")


def test_reconcile_charge_unknown_mode_raises():
    atom_charge, mol_id, net_charge = _toy_charges()
    with pytest.raises(ValueError, match="charge_reconciliation"):
        reconcile_charge(atom_charge, mol_id, net_charge, 2, mode="bogus")


def test_reconcile_charge_std_floor_changes_a_zero_std_atoms_share():
    """A near-zero-std atom is nearly excluded from the residual split under
    the default floor (1e-12), but gets a real share under a coarser floor
    (e.g. 0.1, DASH's own get_molecules_partial_charges default -- see
    reconcile_charge's docstring). One molecule, atom 0 has std=0, atom 1
    has std=1.0, residual to distribute is 1.0.

    Default floor: std used = [1e-12, 1.0], weight0 ~= 1e-12 -- atom 0 gets
    essentially nothing. floor=0.1: std used = [0.1, 1.0], weight0 =
    0.1/1.1 -- atom 0 gets ~9.1% of the residual, a real, non-negligible
    share driven entirely by the floor value.
    """
    atom_charge = np.array([0.0, 0.0])
    mol_id = np.array([0, 0])
    net_charge = np.array([1.0])
    std = np.array([0.0, 1.0])

    default_floor = reconcile_charge(
        atom_charge, mol_id, net_charge, 1, mode="std_weighted", atom_charge_std=std
    )
    coarse_floor = reconcile_charge(
        atom_charge,
        mol_id,
        net_charge,
        1,
        mode="std_weighted",
        atom_charge_std=std,
        std_floor=0.1,
    )

    assert default_floor[0] < 1e-8, "near-zero floor should give atom 0 ~nothing"
    np.testing.assert_allclose(coarse_floor[0], 0.1 / 1.1, atol=1e-10)
    np.testing.assert_allclose(coarse_floor[1], 1.0 / 1.1, atol=1e-10)
    # both floors must still hit the molecule's own target charge exactly
    np.testing.assert_allclose(default_floor.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(coarse_floor.sum(), 1.0, atol=1e-10)


# --- roll_up ---------------------------------------------------------------


def test_roll_up_profile_is_molecule_sum():
    ms = synthetic_molecule_set(n_mol=4, seed=1)
    atom_pred = AtomPrediction(
        atom_profile=ms.atom_profile, atom_area=ms.atom_area, atom_charge=ms.atom_charge
    )
    pred = roll_up(atom_pred, ms, charge_reconciliation="none")
    assert pred.mol_area is not None
    np.testing.assert_allclose(pred.mol_profile, ms.mol_profile)
    np.testing.assert_allclose(pred.mol_area, ms.mol_area)


def test_roll_up_charge_raw_is_pre_reconciliation_total():
    ms = synthetic_molecule_set(n_mol=4, seed=2)
    noisy_charge = ms.atom_charge + 0.5  # deliberately wrong, to see raw != reconciled
    atom_pred = AtomPrediction(atom_profile=ms.atom_profile, atom_charge=noisy_charge)
    pred = roll_up(atom_pred, ms, charge_reconciliation="shift")
    from sieve_experiments.data import molecule_sum

    expected_raw = molecule_sum(noisy_charge, ms.atom_mol_id, ms.n_molecules)
    assert pred.mol_charge_raw is not None
    assert pred.mol_charge is not None
    np.testing.assert_allclose(pred.mol_charge_raw, expected_raw)
    # but after reconciliation it hits the screening-charge target exactly
    # (-net_charge, not net_charge -- see MoleculeSet.screening_charge)
    np.testing.assert_allclose(pred.mol_charge, ms.screening_charge, atol=1e-10)


def test_roll_up_omits_area_and_charge_when_atom_predictor_omits_them():
    ms = synthetic_molecule_set(n_mol=3, seed=3)
    atom_pred = AtomPrediction(atom_profile=ms.atom_profile)
    pred = roll_up(atom_pred, ms)
    assert pred.mol_area is None
    assert pred.mol_charge is None
    assert pred.mol_charge_raw is None


# --- registry ----------------------------------------------------------


def test_global_mean_is_registered():
    assert "global_mean" in REGISTRY
    predictor = build("global_mean", {})
    assert isinstance(predictor, GlobalMeanPredictor)
    assert predictor.name == "global_mean"


def test_build_unknown_predictor_raises():
    with pytest.raises(ValueError, match="unknown predictor"):
        build("not_a_real_predictor", {})


def test_dash_is_lazily_registered_by_build():
    from sieve_experiments.predictors.dash import DASHPredictor

    predictor = build("dash", {"store": "chaos-store", "scheme": "cosmo-sac-2010"})
    assert isinstance(predictor, DASHPredictor)
    assert predictor.name == "dash"
    assert "dash" in REGISTRY


# --- GlobalMeanPredictor -------------------------------------------------


def test_global_mean_predicts_reasonable_shape_and_reconciles_charge():
    ms = synthetic_molecule_set(n_mol=10, seed=4)
    train = ms.select(np.array([True] * 6 + [False] * 4))
    test = ms.select(np.array([False] * 6 + [True] * 4))

    predictor = GlobalMeanPredictor()
    predictor.fit(train, train, rng=np.random.default_rng(0))
    pred = predictor.predict(test)

    assert pred.mol_area is not None
    assert pred.mol_charge is not None
    assert pred.mol_charge_raw is not None
    assert pred.mol_profile.shape == (test.n_molecules, ms.grid.num_points)
    np.testing.assert_allclose(pred.mol_area, pred.mol_profile.sum(axis=1))
    # screening_charge (-net_charge), not net_charge -- see
    # MoleculeSet.screening_charge's docstring.
    np.testing.assert_allclose(pred.mol_charge, test.screening_charge)
    np.testing.assert_allclose(pred.mol_charge_raw, np.zeros(test.n_molecules))


def test_global_mean_beats_nothing_but_is_deterministic():
    """Same train/test -> same prediction, twice."""
    ms = synthetic_molecule_set(n_mol=8, seed=5)
    train = ms.select(np.array([True] * 5 + [False] * 3))
    test = ms.select(np.array([False] * 5 + [True] * 3))

    a = GlobalMeanPredictor()
    a.fit(train, train, rng=np.random.default_rng(0))
    b = GlobalMeanPredictor()
    b.fit(train, train, rng=np.random.default_rng(1))
    np.testing.assert_allclose(a.predict(test).mol_profile, b.predict(test).mol_profile)
