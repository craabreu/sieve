"""End-to-end DASHPretrainedChargePredictor test against the real pinned
DASH-tree clone. Skipped if that clone (charge_experiments/external/
DASH-tree, Task 9) is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists(),
    reason="charge_experiments/external/DASH-tree not cloned",
)


def test_dash_pretrained_predictor_ignores_the_train_set_entirely():
    """The whole point of this predictor: fit() must not read train's own
    atom charges at all. Proven by fitting on two synthetic train sets with
    wildly different (fabricated) charges and checking predict() on the
    same test set gives bit-identical output either way."""
    from charge_experiments.predictors.dash_pretrained import (
        DASHPretrainedChargePredictor,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    test = synthetic_molecule_set(n_mol=4, seed=99)
    train_a = synthetic_molecule_set(n_mol=6, seed=0)
    train_b = synthetic_molecule_set(n_mol=6, seed=0)
    for mol in train_b.mols:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 999.0)  # deliberately absurd

    rng = np.random.default_rng(0)
    predictor_a = DASHPretrainedChargePredictor()
    predictor_a.fit(train_a, train_a, rng=rng)
    pred_a = predictor_a.predict(test)

    predictor_b = DASHPretrainedChargePredictor()
    predictor_b.fit(train_b, train_b, rng=rng)
    pred_b = predictor_b.predict(test)

    np.testing.assert_array_equal(pred_a.atom_charge, pred_b.atom_charge)


def test_dash_pretrained_predictor_runs_without_crashing_and_reports_coverage():
    """Does NOT assert every prediction is finite: the real, published
    tree's own "result" statistics are genuinely sparse for many ordinary
    small molecules at the pinned commit (measured directly: 18/20 of this
    same fixture's molecules hit an unmatched hierarchy at some atom -- see
    module docstring for the confirmed root cause). What matters here is
    that the predictor never crashes and honestly reports what it could
    and couldn't match, rather than silently guessing."""
    from charge_experiments.predictors.dash_pretrained import (
        DASHPretrainedChargePredictor,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHPretrainedChargePredictor()
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert predictor.match_stats["n_atoms"] == mset.n_atoms
    assert predictor.match_stats["n_conformers"] == mset.n_conformers

    # match_stats now reports coverage at all three levels a NaN can
    # originate from -- see the predictor module's own docstring. Each is
    # a superset of the one before it: n_unmatched_atoms (path-matching
    # itself failed) <= n_walk_nan_atoms (raw_charge came back NaN, also
    # covers a matched-but-nothing-populated path) <= n_final_nan_atoms
    # (atom_charge after std_weighted_normalize, also covers NaN
    # propagating to every other atom in a conformer with one bad atom).
    n_nan = int(np.isnan(pred.atom_charge).sum())
    assert predictor.match_stats["n_final_nan_atoms"] == n_nan
    assert (
        predictor.match_stats["n_unmatched_atoms"]
        <= predictor.match_stats["n_walk_nan_atoms"]
        <= predictor.match_stats["n_final_nan_atoms"]
    )
    assert 0 <= n_nan <= mset.n_atoms


def test_dash_pretrained_predictor_runs_end_to_end_via_run(tmp_path):
    """Through the real CLI-level run() pipeline, on synthetic data (no
    real store needed for this part -- run() itself is exercised by the
    real-store test below when that store is prepared)."""
    from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from charge_experiments.runner import execute

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=10, seed=1)
    rng = np.random.default_rng(2)
    labels = rng.choice(["train", "val", "test"], size=10, p=[0.6, 0.2, 0.2])
    masks = {name: labels == name for name in ("train", "val", "test")}
    cfg = ExperimentCfg(
        run=RunCfg(experiment="dash-pretrained-smoke", seed=0),
        data=DataCfg(store="synthetic", split_column="split"),
        predictor=PredictorCfg(name="dash_pretrained", params={}),
    )

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert "mae" in result.metrics
    assert "n_nan" in result.metrics


def test_dash_pretrained_predict_raw_matches_predict_before_normalization():
    from charge_experiments.normalize import std_weighted_normalize
    from charge_experiments.predictors.dash_pretrained import (
        DASHPretrainedChargePredictor,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHPretrainedChargePredictor()
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    expected = std_weighted_normalize(
        raw.atom_charge,
        raw.atom_std,
        mset.net_charge,
        mset.atom_mol_id,
        mset.n_conformers,
    )
    np.testing.assert_array_equal(pred.atom_charge, expected)
    assert predictor.match_stats["n_final_nan_atoms"] == int(
        np.isnan(pred.atom_charge).sum()
    )
