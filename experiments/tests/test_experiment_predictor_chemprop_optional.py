"""End-to-end ChempropDMPNNPredictor test against the real chaos-store.
Skipped unless rdkit, cosmolayer, chemprop, and the store are all present --
same pattern as test_experiment_predictor_dash_optional.py.

max_epochs=1 and a small --limit keep this fast: the point is to exercise
fit -> predict end to end (including a real, if under-trained, lightning
Trainer run) and check the headline structural claim -- zero negative
sigma-profile bins -- not to produce a meaningful trained model.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")
pytest.importorskip("chemprop")

from sieve_experiments.data import DEFAULT_STORES_ROOT

STORE_NAME = "chaos-store"
STORES_ROOT = DEFAULT_STORES_ROOT
STORE = STORES_ROOT / STORE_NAME
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="chaos-store absent")

TRAIN_LIMIT = 40
TEST_LIMIT = 10


def test_bond_fdim_matches_chemprops_own_default_featurizer():
    from chemprop.featurizers import MultiHotBondFeaturizer
    from sieve_experiments.predictors.chemprop_dmpnn import BOND_FDIM

    assert len(MultiHotBondFeaturizer()) == BOND_FDIM


def test_default_ffn_has_the_deepchem_equivalent_layer_count():
    """deepchem's real ffn_layers=3 means 3 TOTAL Linear layers
    (PositionwiseFeedForward.__init__ treats n_layers as the total count).
    chemprop's own MLP.build(n_layers=N) instead yields N+1 total layers (an
    *additional*-hidden-layers count). ChempropDMPNNPredictor's default
    ffn_n_layers=2 must therefore produce exactly 3 nn.Linear sublayers --
    getting this offset wrong in either direction silently changes the
    trained model's depth. See the module docstring for the full trace.
    """
    from chemprop.nn.predictors import RegressionFFN
    from sieve_experiments.predictors.chemprop_dmpnn import ChempropDMPNNPredictor
    from torch import nn

    predictor = ChempropDMPNNPredictor(store="chaos-store", scheme="cosmo-sac-2010")
    ffn = RegressionFFN(
        n_tasks=51,
        input_dim=predictor.hidden_size,
        hidden_dim=predictor.hidden_size,
        n_layers=predictor.ffn_n_layers,
        dropout=predictor.dropout,
    )
    n_linear = sum(1 for m in ffn.ffn.modules() if isinstance(m, nn.Linear))
    assert n_linear == 3, (
        f"expected 3 total Linear layers (matching deepchem's real "
        f"ffn_layers=3), got {n_linear} from ffn_n_layers={predictor.ffn_n_layers}"
    )


def test_fit_predict_produces_no_negative_bins(tmp_path):
    from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from sieve_experiments.data import load_molecule_set
    from sieve_experiments.runner import execute

    mset, masks = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=TRAIN_LIMIT,
        stores_root=STORES_ROOT,
    )
    train_mask = np.zeros(mset.n_molecules, dtype=bool)
    train_mask[: TRAIN_LIMIT - TEST_LIMIT] = True
    test_mask = ~train_mask
    masks = {"train": train_mask, "val": train_mask, "test": test_mask}

    cfg = ExperimentCfg(
        run=RunCfg(experiment="chemprop-smoke", seed=0),
        data=DataCfg(
            store=STORE_NAME, scheme="cosmo-sac-2010", split_column="biased_split"
        ),
        predictor=PredictorCfg(
            name="chemprop_dmpnn",
            params={
                "store": STORE_NAME,
                "scheme": "cosmo-sac-2010",
                "max_epochs": 1,
                "batch_size": 8,
            },
        ),
    )
    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert result.run_dir.is_dir()
    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert result.metrics["n_test"] > 0

    predictions = np.load(result.run_dir / "predictions.npz")
    profile_pred = predictions["mol_profile_pred"]
    n_negative = (profile_pred < 0).sum()
    assert n_negative == 0, (
        f"{n_negative}/{profile_pred.size} predicted bins are negative -- "
        "softplus should make this structurally impossible"
    )
