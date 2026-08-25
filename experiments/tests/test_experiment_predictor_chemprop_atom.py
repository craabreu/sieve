"""Fast-suite tests for the per-atom Chemprop predictor (T11).

Mirrors test_experiment_predictor_chemprop.py's split: everything here runs
without chemprop installed, so it covers the pure-numpy helpers and the
constructor validation. The parts that need a real MolAtomBondMPNN (and the
atom-ordering guards, which are the whole correctness story) live in
test_experiment_predictor_chemprop_atom_optional.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sieve_experiments.predictors.chemprop_atom import (
    ATOM_FDIM,
    BOND_FDIM,
    ChempropAtomPredictor,
    atom_prediction_from_profile,
)

# --- constants -------------------------------------------------------------
#
# These are asserted against chemprop's own featurizers in the optional suite;
# here we only pin that they didn't get edited to something absurd.


def test_feature_dims_are_the_expected_widths():
    assert ATOM_FDIM == 72  # MultiHotAtomFeaturizer.v2()
    assert BOND_FDIM == 14  # MultiHotBondFeaturizer()


# --- atom_prediction_from_profile -----------------------------------------


def test_area_and_charge_are_derived_from_the_profile():
    """Area and charge are never predicted as their own quantities -- they
    must be exactly sum(p) and p @ sigma, the same convention every other
    profile predictor in this harness uses."""
    sigma = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    profile = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 7.0, 0.0, 0.0],
        ]
    )
    pred = atom_prediction_from_profile(profile, sigma)

    assert pred.atom_area is not None
    assert pred.atom_charge is not None
    np.testing.assert_allclose(pred.atom_area, [15.0, 7.0])
    # row 0: 1*-2 + 2*-1 + 3*0 + 4*1 + 5*2 = -2 -2 + 0 + 4 + 10 = 10
    # row 1: a spike at sigma=0 -> zero charge
    np.testing.assert_allclose(pred.atom_charge, [10.0, 0.0])
    np.testing.assert_allclose(pred.atom_profile, profile)


def test_atom_charge_std_is_not_supplied():
    """This model predicts no per-atom charge spread, which is exactly why
    charge_reconciliation="std_weighted" cannot be used with it."""
    pred = atom_prediction_from_profile(np.ones((3, 5)), np.zeros(5))
    assert pred.atom_charge_std is None


# --- constructor validation ------------------------------------------------


def test_rejects_an_unknown_charge_reconciliation():
    with pytest.raises(ValueError, match="charge_reconciliation"):
        ChempropAtomPredictor(
            store="chaos-store", scheme="cosmo-sac-2010", charge_reconciliation="bogus"
        )


def test_rejects_an_unknown_output_activation():
    with pytest.raises(ValueError, match="output_activation"):
        ChempropAtomPredictor(
            store="chaos-store", scheme="cosmo-sac-2010", output_activation="relu"
        )


def test_default_output_activation_is_squared():
    """softplus (T10's default) collapses to identically-zero output on
    atom-level targets -- see the module docstring's measured numbers."""
    predictor = ChempropAtomPredictor(store="chaos-store", scheme="cosmo-sac-2010")
    assert predictor.output_activation == "squared"


def test_default_charge_reconciliation_is_none():
    """Default "none" keeps the reported charge metrics the model's own raw
    output, directly comparable to T10's."""
    predictor = ChempropAtomPredictor(store="chaos-store", scheme="cosmo-sac-2010")
    assert predictor.charge_reconciliation == "none"


def test_predict_before_fit_raises():
    from experiments.tests.helpers import synthetic_molecule_set

    predictor = ChempropAtomPredictor(store="chaos-store", scheme="cosmo-sac-2010")
    with pytest.raises(RuntimeError, match="fit_atoms"):
        predictor.predict_atoms(synthetic_molecule_set(n_mol=1, seed=0))


# --- stores_root coercion --------------------------------------------------


def test_atom_truth_coerces_string_stores_root_to_a_path(monkeypatch):
    """predictor.params comes straight from YAML, so stores_root arrives as a
    plain str -- load_atom_truth's ``stores_root / store_name`` would raise
    TypeError on a str. Same regression DASH has its own test for."""
    captured = {}

    def fake_load_atom_truth(store, *, scheme, smiles, num_atoms, stores_root):
        captured["stores_root"] = stores_root
        n = int(sum(num_atoms))
        return np.zeros((n, 51)), np.zeros(n), np.zeros(n)

    monkeypatch.setattr(
        "sieve_experiments.predictors.chemprop_atom.load_atom_truth",
        fake_load_atom_truth,
    )
    predictor = ChempropAtomPredictor(
        store="chaos-store", scheme="cosmo-sac-2010", stores_root="stores"
    )

    from experiments.tests.helpers import synthetic_molecule_set

    predictor._atom_truth(synthetic_molecule_set(n_mol=2, seed=0))
    assert isinstance(captured["stores_root"], Path)
