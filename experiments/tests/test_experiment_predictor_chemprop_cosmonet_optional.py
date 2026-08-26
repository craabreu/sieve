"""End-to-end ChempropCosmonetPredictor test against the real chaos-store.
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
    from sieve_experiments.predictors.chemprop_cosmonet import BOND_FDIM

    assert len(MultiHotBondFeaturizer()) == BOND_FDIM


def test_default_ffn_has_the_deepchem_equivalent_layer_count():
    """deepchem's real ffn_layers=3 means 3 TOTAL Linear layers
    (PositionwiseFeedForward.__init__ treats n_layers as the total count).
    chemprop's own MLP.build(n_layers=N) instead yields N+1 total layers (an
    *additional*-hidden-layers count). ChempropCosmonetPredictor's default
    ffn_n_layers=2 must therefore produce exactly 3 nn.Linear sublayers --
    getting this offset wrong in either direction silently changes the
    trained model's depth. See the module docstring for the full trace.
    """
    from chemprop.nn.predictors import RegressionFFN
    from sieve_experiments.predictors.chemprop_cosmonet import ChempropCosmonetPredictor
    from torch import nn

    predictor = ChempropCosmonetPredictor(store="chaos-store", scheme="cosmo-sac-2010")
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


# --- custom loss classes, hand-computed -----------------------------


def _tiny_model(loss_mode: str, y_min: np.ndarray, scale: np.ndarray):
    """Build a minimal real MPNN just to reach its constructed
    ``predictor.criterion`` -- the loss classes are nested inside
    ``_build_model`` (mirrors ``SoftplusRegressionFFN``'s own nesting), so
    this is the only way to get a real instance to test directly.
    """
    from chemprop.nn.transforms import UnscaleTransform
    from sieve_experiments.predictors.chemprop_cosmonet import _build_model

    output_transform = UnscaleTransform(mean=y_min, scale=scale)
    return _build_model(
        hidden_size=4,
        depth=1,
        dropout=0.0,
        ffn_n_layers=0,
        n_tasks=len(y_min),
        output_transform=output_transform,
        y_min=y_min,
        scale=scale,
        loss_mode=loss_mode,
    )


def test_w1_normalized_loss_matches_hand_computation():
    import torch

    # true=[1,2,3] (sum 6) -> normalized [1,2,3]/6; pred=[3,2,1] (sum 6) ->
    # normalized [3,2,1]/6. cumsum(true_norm)=[1,3,6]/6, cumsum(pred_norm)=
    # [3,5,6]/6 -> |diff|=[2,2,0]/6 -> sum = 4/6.
    model = _tiny_model("w1_normalized", np.zeros(3), np.ones(3))
    criterion = model.predictor.criterion

    preds = torch.tensor([[3.0, 2.0, 1.0]])
    targets = torch.tensor([[1.0, 2.0, 3.0]])
    criterion.update(preds, targets)
    assert abs(criterion.compute().item() - 4 / 6) < 1e-5


def test_mse_cumsum_loss_matches_hand_computation():
    import torch

    # cumsum(true)=[1,3,6], cumsum(pred)=[3,5,6] -> diffs=[-2,-2,0] ->
    # squared=[4,4,0] -> mean = 8/3.
    model = _tiny_model("mse_cumsum", np.zeros(3), np.ones(3))
    criterion = model.predictor.criterion

    preds = torch.tensor([[3.0, 2.0, 1.0]])
    targets = torch.tensor([[1.0, 2.0, 3.0]])
    criterion.update(preds, targets)
    assert abs(criterion.compute().item() - 8 / 3) < 1e-5


def test_loss_unscales_before_computing():
    """y_min/scale must actually be applied, not the raw (scaled) tensors --
    feeds inputs that only reproduce the hand-computed mse_cumsum example
    above once unscaled (real = scaled * scale + y_min)."""
    import torch

    y_min = np.array([10.0, 10.0, 10.0])
    scale = np.array([2.0, 2.0, 2.0])
    model = _tiny_model("mse_cumsum", y_min, scale)
    criterion = model.predictor.criterion

    # real target=[1,2,3] -> scaled=(real-10)/2=[-4.5,-4,-3.5]
    # real pred=[3,2,1]   -> scaled=(real-10)/2=[-3.5,-4,-4.5]
    preds = torch.tensor([[-3.5, -4.0, -4.5]])
    targets = torch.tensor([[-4.5, -4.0, -3.5]])
    criterion.update(preds, targets)
    assert abs(criterion.compute().item() - 8 / 3) < 1e-4


# --- end-to-end, all three loss modes ---------------------------------


@pytest.mark.parametrize("loss_mode", ["mse", "w1_normalized", "mse_cumsum"])
def test_fit_predict_produces_no_negative_bins(tmp_path, loss_mode):
    """Non-negativity comes from the softplus architecture, not the loss
    function -- must hold under every loss_mode."""
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
            name="chemprop_cosmonet",
            params={
                "store": STORE_NAME,
                "scheme": "cosmo-sac-2010",
                "max_epochs": 1,
                "batch_size": 8,
                "loss_mode": loss_mode,
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
        f"{n_negative}/{profile_pred.size} predicted bins are negative "
        f"(loss_mode={loss_mode!r}) -- softplus should make this "
        "structurally impossible"
    )
