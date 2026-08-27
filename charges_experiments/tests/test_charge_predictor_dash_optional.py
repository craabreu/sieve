"""End-to-end DASHChargePredictor test against the real pinned DASH-tree
clone. Skipped if that clone (charges_experiments/external/DASH-tree,
Task 9) is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists(),
    reason="charges_experiments/external/DASH-tree not cloned",
)


def test_dash_charge_predictor_fits_and_predicts_on_synthetic_molecules():
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charges_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHChargePredictor()
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))
    assert "train" in predictor.match_stats
    assert predictor.match_stats["train"]["n_conformers"] == mset.n_conformers
