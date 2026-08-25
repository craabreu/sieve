"""Tests for predictors/cosmonet.py -- the COSMO-NET prediction-lookup predictor."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import synthetic_molecule_set
from sieve_experiments.cosmonet_data import write_cosmonet_csv
from sieve_experiments.data import MoleculeSet
from sieve_experiments.predictors import build
from sieve_experiments.predictors.cosmonet import CosmonetPredictor


def _split_mset(mset, n_train, n_val):
    """Slice a MoleculeSet into train/val/test parts, in molecule order."""
    n = mset.n_molecules

    def slice_mask(start, stop):
        m = np.zeros(n, dtype=bool)
        m[start:stop] = True
        return m

    train = mset.select(slice_mask(0, n_train))
    val = mset.select(slice_mask(n_train, n_train + n_val))
    test = mset.select(slice_mask(n_train + n_val, n))
    return train, val, test


def _write_fixture_csv(tmp_path, train, val, test):
    category = (
        ["Training"] * train.n_molecules
        + ["Validation"] * val.n_molecules
        + ["Test"] * test.n_molecules
    )
    combined = MoleculeSet(
        smiles=train.smiles + val.smiles + test.smiles,
        num_atoms=np.concatenate([train.num_atoms, val.num_atoms, test.num_atoms]),
        net_charge=np.concatenate([train.net_charge, val.net_charge, test.net_charge]),
        grid=train.grid,
        mol_profile=np.concatenate(
            [train.mol_profile, val.mol_profile, test.mol_profile]
        ),
    )
    out_path = tmp_path / "predictions.csv"
    write_cosmonet_csv(combined, category, out_path)
    return out_path


def test_fit_then_predict_looks_up_by_smiles(tmp_path):
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    train, val, test = _split_mset(mset, n_train=3, n_val=2)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)

    predictor = CosmonetPredictor(
        predictions_csv=str(csv_path), store="chaos-store", scheme="cosmo-sac-2010"
    )
    predictor.fit_molecules(train, val, rng=np.random.default_rng(0))
    pred = predictor.predict_molecules(test)

    assert pred.mol_area is not None
    assert pred.mol_charge_raw is not None
    assert test.mol_profile is not None
    np.testing.assert_allclose(pred.mol_profile, test.mol_profile, rtol=1e-6)
    np.testing.assert_allclose(pred.mol_area, test.mol_profile.sum(axis=1), rtol=1e-6)
    expected_charge = test.mol_profile @ test.grid.values
    np.testing.assert_allclose(pred.mol_charge_raw, expected_charge, rtol=1e-6)


def test_predict_is_smiles_keyed_not_positional(tmp_path):
    mset = synthetic_molecule_set(n_mol=6, seed=1)
    train, val, test = _split_mset(mset, n_train=3, n_val=1)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)

    predictor = CosmonetPredictor(
        predictions_csv=str(csv_path), store="chaos-store", scheme="cosmo-sac-2010"
    )
    predictor.fit_molecules(train, val, rng=np.random.default_rng(0))

    reversed_test = MoleculeSet(
        smiles=list(reversed(test.smiles)),
        num_atoms=test.num_atoms[::-1],
        net_charge=test.net_charge[::-1],
        grid=test.grid,
        mol_profile=test.mol_profile[::-1],
    )
    pred = predictor.predict_molecules(reversed_test)
    assert reversed_test.mol_profile is not None
    np.testing.assert_allclose(pred.mol_profile, reversed_test.mol_profile, rtol=1e-6)


def test_fit_rejects_a_train_molecule_mislabeled_in_the_csv(tmp_path):
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    train, val, test = _split_mset(mset, n_train=3, n_val=2)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)

    # A predictions CSV built from a *different* split: this run's train
    # molecule 0 was actually "Test" when the checkpoint was trained.
    text = csv_path.read_text().replace("Training", "Test", 1)
    csv_path.write_text(text)

    predictor = CosmonetPredictor(
        predictions_csv=str(csv_path), store="chaos-store", scheme="cosmo-sac-2010"
    )
    with pytest.raises(ValueError, match="not labeled 'Training'"):
        predictor.fit_molecules(train, val, rng=np.random.default_rng(0))


def test_predict_rejects_a_test_molecule_mislabeled_in_the_csv(tmp_path):
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    train, val, test = _split_mset(mset, n_train=3, n_val=2)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)

    # This run's held-out test molecule was "Training" in the checkpoint's
    # own run -- exactly the leakage scenario the guard exists to catch.
    text = csv_path.read_text().replace("Test", "Training", 1)
    csv_path.write_text(text)

    predictor = CosmonetPredictor(
        predictions_csv=str(csv_path), store="chaos-store", scheme="cosmo-sac-2010"
    )
    predictor.fit_molecules(train, val, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="not labeled 'Test'"):
        predictor.predict_molecules(test)


def test_predict_before_fit_raises():
    predictor = CosmonetPredictor(
        predictions_csv="unused.csv", store="chaos-store", scheme="cosmo-sac-2010"
    )
    mset = synthetic_molecule_set(n_mol=2, seed=0)
    with pytest.raises(RuntimeError, match="fit_molecules must be called"):
        predictor.predict_molecules(mset)


def test_load_predictions_rejects_duplicate_smiles(tmp_path):
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    train, val, test = _split_mset(mset, n_train=3, n_val=2)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)
    lines = csv_path.read_text().splitlines()
    header, rows = lines[0], lines[1:]
    rows.append(rows[0])  # duplicate the first data row's SMILES
    csv_path.write_text("\n".join([header, *rows]) + "\n")

    predictor = CosmonetPredictor(
        predictions_csv=str(csv_path), store="chaos-store", scheme="cosmo-sac-2010"
    )
    with pytest.raises(ValueError, match="duplicate SMILES"):
        predictor.fit_molecules(train, val, rng=np.random.default_rng(0))


def test_registered_and_buildable_via_predictors_build(tmp_path):
    mset = synthetic_molecule_set(n_mol=3, seed=0)
    train, val, test = _split_mset(mset, n_train=1, n_val=1)
    csv_path = _write_fixture_csv(tmp_path, train, val, test)

    predictor = build(
        "cosmonet",
        {
            "predictions_csv": str(csv_path),
            "store": "chaos-store",
            "scheme": "cosmo-sac-2010",
        },
    )
    assert isinstance(predictor, CosmonetPredictor)
    assert predictor.name == "cosmonet"
    assert predictor.store == "chaos-store"
    assert predictor.scheme == "cosmo-sac-2010"
