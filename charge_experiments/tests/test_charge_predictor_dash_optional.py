"""End-to-end DASHChargePredictor test against the real pinned DASH-tree
clone. Skipped if that clone (charge_experiments/external/DASH-tree,
Task 9) is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists(),
    reason="charge_experiments/external/DASH-tree not cloned",
)


def test_dash_charge_predictor_fits_and_predicts_on_synthetic_molecules():
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHChargePredictor()
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))
    assert "train" in predictor.match_stats
    assert predictor.match_stats["train"]["n_conformers"] == mset.n_conformers


def test_dash_charge_predictor_predict_equals_predict_raw_atom_charge():
    """predict() must stay behavior-identical to today: unnormalized, i.e.
    exactly predict_raw(...).atom_charge."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHChargePredictor()
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    np.testing.assert_array_equal(pred.atom_charge, raw.atom_charge)
    assert raw.atom_std.shape == raw.atom_charge.shape


def test_dash_charge_predictor_save_and_load_model_state_round_trips(tmp_path):
    """A predictor that loads a saved artifact (no fit() call at all)
    predicts identically to a freshly-fit one on the same train data."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=6, seed=0)
    test = synthetic_molecule_set(n_mol=4, seed=1)
    rng = np.random.default_rng(0)

    fitted = DASHChargePredictor()
    fitted.fit(train, train, rng=rng)
    fitted_pred = fitted.predict(test)

    stats_path = tmp_path / "dash-tree-stats.npz"
    fitted.save_model_state(stats_path)

    loaded = DASHChargePredictor()
    loaded.load_model_state(stats_path)
    loaded_pred = loaded.predict(test)

    np.testing.assert_array_equal(loaded_pred.atom_charge, fitted_pred.atom_charge)


def test_dash_charge_predictor_load_model_state_skips_fit(tmp_path, monkeypatch):
    """Proves load_model_state never touches match_new_atom for train atoms
    -- the whole point of persisting stats."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=6, seed=0)
    test = synthetic_molecule_set(n_mol=4, seed=1)
    rng = np.random.default_rng(0)

    fitted = DASHChargePredictor()
    fitted.fit(train, train, rng=rng)
    stats_path = tmp_path / "dash-tree-stats.npz"
    fitted.save_model_state(stats_path)

    loaded = DASHChargePredictor()

    def _boom(*args, **kwargs):
        raise AssertionError("fit()/_atom_paths must not be called")

    monkeypatch.setattr(loaded, "fit", _boom)
    loaded.load_model_state(stats_path)  # must not raise
    pred = loaded.predict(test)
    assert pred.atom_charge.shape == (test.n_atoms,)
