"""Tests for cosmonet_data.py -- the chaos-store -> COSMO-NET CSV converter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import synthetic_molecule_set
from sieve_experiments.cosmonet_data import (
    _METADATA_COLUMNS,
    category_labels,
    write_cosmonet_csv,
)
from sieve_experiments.data import DEFAULT_GRID, MoleculeSet


def test_category_labels_maps_sieve_splits_to_cosmonet_vocabulary():
    masks = {
        "train": np.array([True, False, False, True]),
        "val": np.array([False, True, False, False]),
        "test": np.array([False, False, True, False]),
    }
    labels = category_labels(masks, n_molecules=4)
    assert list(labels) == ["Training", "Validation", "Test", "Training"]


def test_category_labels_rejects_a_molecule_in_zero_splits():
    masks = {
        "train": np.array([True, False]),
        "val": np.array([False, False]),
        "test": np.array([False, False]),
    }
    with pytest.raises(ValueError, match="zero or more than one split"):
        category_labels(masks, n_molecules=2)


def test_category_labels_rejects_a_molecule_in_two_splits():
    masks = {
        "train": np.array([True, False]),
        "val": np.array([True, False]),
        "test": np.array([False, False]),
    }
    with pytest.raises(ValueError, match="zero or more than one split"):
        category_labels(masks, n_molecules=2)


def test_write_cosmonet_csv_round_trips_smiles_category_and_profile(tmp_path):
    mset = synthetic_molecule_set(n_mol=5, seed=1)
    category = ["Training", "Training", "Validation", "Test", "Training"]
    out_path = tmp_path / "chaos-cosmonet.csv"

    write_cosmonet_csv(mset, category, out_path)
    df = pd.read_csv(out_path)

    # Exactly 10 metadata columns before the sigma columns -- the training
    # script slices data.columns[10:] positionally, so the count matters as
    # much as the names.
    assert list(df.columns[:10]) == _METADATA_COLUMNS
    assert len(df.columns) == 10 + DEFAULT_GRID.num_points

    assert list(df["CanonicalSMILES"]) == mset.smiles
    assert list(df["CATEGORY"]) == category

    sigma_cols = df.columns[10:]
    assert len(sigma_cols) == 51
    # Header text matches COSMO-NET's own convention ("-0.025" .. "0.025").
    assert sigma_cols[0] == "-0.025"
    assert sigma_cols[-1] == "0.025"
    assert "0.0" in sigma_cols

    got_profile = df[sigma_cols].to_numpy()
    np.testing.assert_allclose(got_profile, mset.mol_profile, rtol=1e-6)


def test_write_cosmonet_csv_rejects_a_mismatched_grid(tmp_path):
    from sieve_experiments.data import SigmaGridSpec

    mset = synthetic_molecule_set(n_mol=2, seed=0)
    wrong_grid = SigmaGridSpec(max_abs_sigma=0.03, num_points=51)
    mset = MoleculeSet(
        smiles=mset.smiles,
        num_atoms=mset.num_atoms,
        net_charge=mset.net_charge,
        grid=wrong_grid,
        mol_profile=mset.mol_profile,
    )
    with pytest.raises(ValueError, match="grid"):
        write_cosmonet_csv(mset, ["Training", "Test"], tmp_path / "out.csv")


def test_write_cosmonet_csv_rejects_missing_mol_profile(tmp_path):
    mset = synthetic_molecule_set(n_mol=2, seed=0)
    mset = MoleculeSet(
        smiles=mset.smiles, num_atoms=mset.num_atoms, net_charge=mset.net_charge
    )
    with pytest.raises(ValueError, match="mol_profile"):
        write_cosmonet_csv(mset, ["Training", "Test"], tmp_path / "out.csv")


def test_write_cosmonet_csv_rejects_category_length_mismatch(tmp_path):
    mset = synthetic_molecule_set(n_mol=3, seed=0)
    with pytest.raises(ValueError, match="category has"):
        write_cosmonet_csv(mset, ["Training", "Test"], tmp_path / "out.csv")
