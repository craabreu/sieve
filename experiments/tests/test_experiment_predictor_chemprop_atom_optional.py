"""End-to-end ChempropAtomPredictor tests against the real chaos-store (T11).

Skipped unless rdkit, cosmolayer, chemprop, and the store are all present --
same pattern as test_experiment_predictor_chemprop_optional.py.

The bulk of this file is the ATOM-ORDERING guard. A per-atom model that pairs
atom targets with the wrong graph nodes still produces perfectly finite,
plausible-looking metrics, so nothing else in the suite would catch it; these
tests are the only thing standing between a silent misalignment and a
published number. See predictors/chemprop_atom.py's module docstring for the
two independent ways it can go wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")
pytest.importorskip("chemprop")

from sieve_experiments.data import DEFAULT_STORES_ROOT, load_molecule_set

STORE_NAME = "chaos-store"
STORES_ROOT = DEFAULT_STORES_ROOT
STORE = STORES_ROOT / STORE_NAME
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="chaos-store absent")

TRAIN_LIMIT = 40
TEST_LIMIT = 10


def _mset(limit=TRAIN_LIMIT):
    mset, _ = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=limit,
        stores_root=STORES_ROOT,
    )
    return mset


# --- feature widths --------------------------------------------------------


def test_feature_dims_match_chemprops_own_featurizers():
    """ATOM_FDIM/BOND_FDIM size the encoder's input layer -- if chemprop ever
    changes either featurizer's width, this must fail loudly rather than
    silently building a mis-sized model."""
    from chemprop.featurizers import MultiHotAtomFeaturizer, MultiHotBondFeaturizer
    from sieve_experiments.predictors.chemprop_atom import ATOM_FDIM, BOND_FDIM

    assert len(MultiHotAtomFeaturizer.v2()) == ATOM_FDIM
    assert len(MultiHotBondFeaturizer()) == BOND_FDIM


def test_paper_atom_featurizer_could_not_represent_hydrogen():
    """Documents *why* T11 departs from T10's featurizer: the COSMO-NET
    paper's Table 1 vocabulary has no hydrogen, and a per-atom model runs on
    the explicit-H graph where most nodes are hydrogens."""
    from rdkit import Chem
    from sieve_experiments.predictors.chemprop_cosmonet import PaperAtomFeaturizer

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[O:1]([H:2])[H:3]", params)
    h_atom = next(a for a in mol.GetAtoms() if a.GetSymbol() == "H")

    features = PaperAtomFeaturizer()(h_atom)
    assert features[:8].sum() == 0.0, "H should one-hot to nothing in the paper's 8"


# --- atom ordering: within a molecule --------------------------------------


def test_make_mol_order_matches_the_stores_own_element_column():
    """The headline correctness guard, mirroring the DASH optional suite's
    own atom-map-order test. chemprop's make_mol(reorder_atoms=True) must
    reproduce the store's flat atom order exactly, element for element --
    otherwise every atom target is paired with the wrong graph node, which
    still yields perfectly finite metrics, so nothing else would catch it.

    This discriminates rather than passing vacuously: measured on these same
    40 molecules, dropping ``reorder_atoms=True`` changes the element
    sequence for 38 of them and mismatches 297/1855 atoms (16.0%).
    """
    from chemprop.utils import make_mol
    from cosmolayer.store import SegmentStore

    store = SegmentStore.load(STORE)
    df = store.molecules_df.iloc[:TRAIN_LIMIT].reset_index(drop=True)
    store_elements = store.atoms_df["element"].to_numpy()

    flat = 0
    checked = 0
    non_identity = 0
    for smi, n_atoms in zip(df.smiles, df.num_atoms, strict=True):
        mol = make_mol(smi, keep_h=True, add_h=False, reorder_atoms=True)
        assert mol is not None
        assert mol.GetNumAtoms() == n_atoms
        # did reorder_atoms actually have to move anything for this molecule?
        raw_maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
        if raw_maps != list(range(1, len(raw_maps) + 1)):
            non_identity += 1
        for j in range(n_atoms):
            assert mol.GetAtomWithIdx(j).GetSymbol() == store_elements[flat + j], (
                f"element mismatch at flat atom {flat + j} of {smi[:60]}"
            )
            checked += 1
        flat += n_atoms

    assert checked > 500, f"only certified {checked} atoms -- too few to trust"


def test_datapoints_reject_a_molecule_whose_atom_count_disagrees():
    """The per-molecule assert in _datapoints must actually fire."""
    from sieve_experiments.data import MoleculeSet
    from sieve_experiments.predictors.chemprop_atom import ChempropAtomPredictor

    mset = _mset(limit=3)
    lying = MoleculeSet(
        smiles=list(mset.smiles[:1]),
        num_atoms=np.array([int(mset.num_atoms[0]) + 5]),
        net_charge=np.array([0.0]),
        grid=mset.grid,
    )
    predictor = ChempropAtomPredictor(store=STORE_NAME, scheme="cosmo-sac-2010")
    with pytest.raises(ValueError, match="atom count mismatch"):
        predictor._datapoints(lying, None)


def test_datapoints_pair_each_molecules_own_target_block_with_its_graph():
    """np.split must hand each datapoint exactly its own molecule's rows, in
    order -- an off-by-one here would shift every molecule's targets."""
    from sieve_experiments.predictors.chemprop_atom import ChempropAtomPredictor

    mset = _mset(limit=5)
    # a marker per atom: molecule index in every bin, so a misassigned block
    # is immediately visible
    atom_y = np.repeat(np.arange(mset.n_molecules, dtype=np.float64), mset.num_atoms)[
        :, None
    ] * np.ones((1, 51))

    predictor = ChempropAtomPredictor(store=STORE_NAME, scheme="cosmo-sac-2010")
    datapoints = predictor._datapoints(mset, atom_y)

    assert len(datapoints) == mset.n_molecules
    for i, (dp, n_atoms) in enumerate(zip(datapoints, mset.num_atoms, strict=True)):
        assert dp.atom_y.shape == (int(n_atoms), 51)
        assert np.all(dp.atom_y == i), f"molecule {i} got another molecule's targets"
        assert dp.mol.GetNumAtoms() == int(n_atoms)


# --- model wiring ----------------------------------------------------------


def test_atom_head_input_dim_comes_from_the_message_passing_block():
    """input_dim is read off mp.output_dims[0] rather than hardcoded, as
    chemprop's own build_MAB_model does."""
    from sieve_experiments.predictors.chemprop_atom import _build_model

    model = _build_model(
        hidden_size=16,
        depth=2,
        dropout=0.0,
        ffn_n_layers=0,
        n_tasks=51,
        output_transform=None,
    )
    assert model.message_passing.output_dims[0] == 16
    assert model.mol_predictor is None
    assert model.bond_predictor is None
    assert model.agg is None
    first_linear = next(
        m for m in model.atom_predictor.ffn.modules() if hasattr(m, "in_features")
    )
    assert first_linear.in_features == 16


def test_nonneg_ffn_class_is_shared_with_the_molecule_level_predictor():
    """Both T10 and T11 must get their head from the same factory, so the
    activation-before-unscale invariant is defined exactly once."""
    from sieve_experiments.predictors.chemprop_atom import _build_model
    from sieve_experiments.predictors.chemprop_cosmonet import (
        nonneg_regression_ffn_class,
    )

    model = _build_model(
        hidden_size=8,
        depth=1,
        dropout=0.0,
        ffn_n_layers=0,
        n_tasks=51,
        output_transform=None,
    )
    assert type(model.atom_predictor) is nonneg_regression_ffn_class("squared")


def test_squared_activation_reaches_exact_zero_but_softplus_cannot():
    """The mechanism behind T11's one real departure from T10. A target of
    exactly zero is what atom profiles are 81.4% made of; softplus can only
    approach it as its pre-activation goes to -inf (where its own gradient
    vanishes and the head dies), while x**2 hits it at a finite x=0.
    """
    import torch
    from sieve_experiments.predictors.chemprop_cosmonet import (
        nonneg_regression_ffn_class,
    )

    squared = nonneg_regression_ffn_class("squared")
    softplus = nonneg_regression_ffn_class("softplus")
    assert squared is not softplus

    z = torch.zeros(1, requires_grad=True)
    # squared: exactly zero at a finite input
    assert (z**2).item() == 0.0
    # softplus: strictly positive everywhere, and its gradient vanishes as the
    # input goes negative -- the two facts that together kill the head
    assert torch.nn.functional.softplus(z).item() > 0.0
    very_negative = torch.tensor([-30.0], requires_grad=True)
    torch.nn.functional.softplus(very_negative).backward()
    assert very_negative.grad is not None
    assert very_negative.grad.abs().item() < 1e-10, "softplus gradient should vanish"


@pytest.mark.parametrize("activation", ["softplus", "squared", "abs"])
def test_every_activation_keeps_predictions_non_negative(activation):
    """Non-negativity must be structural for all three -- that is the whole
    point of applying the activation before unscaling."""
    from sieve_experiments.predictors.chemprop_atom import ChempropAtomPredictor

    mset = _mset(limit=8)
    predictor = ChempropAtomPredictor(
        store=STORE_NAME,
        scheme="cosmo-sac-2010",
        hidden_size=16,
        depth=1,
        max_epochs=1,
        batch_size=4,
        output_activation=activation,
    )
    predictor.fit_atoms(mset, mset, rng=np.random.default_rng(0))
    profile = predictor.predict_atoms(mset).atom_profile
    assert (profile < 0).sum() == 0


# --- atom ordering: between molecules --------------------------------------


def test_predictions_follow_molecule_order():
    """The between-molecule counterpart to the ordering guard. Predict on a
    molecule set, then on the SAME molecules reversed: each molecule's atom
    block must travel with its own molecule. A shuffled predict loader (or a
    drifted flat-array split) returns an identically-shaped array, so this is
    the only thing that would notice.
    """
    from sieve_experiments.predictors.chemprop_atom import ChempropAtomPredictor

    mset = _mset(limit=12)
    predictor = ChempropAtomPredictor(
        store=STORE_NAME,
        scheme="cosmo-sac-2010",
        hidden_size=16,
        depth=2,
        max_epochs=1,
        batch_size=4,
    )
    predictor.fit_atoms(mset, mset, rng=np.random.default_rng(0))

    forward = predictor.predict_atoms(mset).atom_profile

    from sieve_experiments.data import MoleculeSet

    order = np.arange(mset.n_molecules)[::-1]
    reversed_mset = MoleculeSet(
        smiles=[mset.smiles[i] for i in order],
        num_atoms=mset.num_atoms[order],
        net_charge=mset.net_charge[order],
        grid=mset.grid,
    )
    backward = predictor.predict_atoms(reversed_mset).atom_profile

    fwd_blocks = np.split(forward, np.cumsum(mset.num_atoms)[:-1])
    bwd_blocks = np.split(backward, np.cumsum(reversed_mset.num_atoms)[:-1])
    assert len(fwd_blocks) == len(bwd_blocks) == mset.n_molecules
    for i in range(mset.n_molecules):
        np.testing.assert_allclose(
            bwd_blocks[i],
            fwd_blocks[order[i]],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"molecule {order[i]}'s atom block did not follow it",
        )


def test_order_check_catches_a_shuffled_predict_loader(monkeypatch):
    """_check_molecule_order is the tripwire for a permuted loader. Force the
    permutation and confirm it actually fires -- an unexercised guard is no
    guard."""
    from chemprop.data import build_dataloader
    from sieve_experiments.predictors import chemprop_atom
    from sieve_experiments.predictors.chemprop_atom import ChempropAtomPredictor

    mset = _mset(limit=12)
    predictor = ChempropAtomPredictor(
        store=STORE_NAME,
        scheme="cosmo-sac-2010",
        hidden_size=16,
        depth=2,
        max_epochs=1,
        batch_size=4,
    )
    predictor.fit_atoms(mset, mset, rng=np.random.default_rng(0))

    real_build_dataloader = build_dataloader

    def shuffling_loader(dset, **kwargs):
        kwargs["shuffle"] = True
        return real_build_dataloader(dset, **kwargs)

    monkeypatch.setattr(
        chemprop_atom, "build_dataloader", shuffling_loader, raising=False
    )
    monkeypatch.setattr(
        "chemprop.data.build_dataloader", shuffling_loader, raising=False
    )

    with pytest.raises(RuntimeError, match=r"do not line up"):
        predictor.predict_atoms(mset)


# --- end to end ------------------------------------------------------------


def test_fit_predict_produces_no_negative_bins(tmp_path):
    """Non-negativity is architectural (softplus before unscaling, y_min>=0),
    exactly as in T10 -- and the atom/* metric block must actually appear."""
    from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from sieve_experiments.runner import execute

    mset = _mset()
    train_mask = np.zeros(mset.n_molecules, dtype=bool)
    train_mask[: TRAIN_LIMIT - TEST_LIMIT] = True
    masks = {"train": train_mask, "val": train_mask, "test": ~train_mask}

    cfg = ExperimentCfg(
        run=RunCfg(experiment="chemprop-atom-smoke", seed=0),
        data=DataCfg(
            store=STORE_NAME, scheme="cosmo-sac-2010", split_column="biased_split"
        ),
        predictor=PredictorCfg(
            name="chemprop_atom",
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
    # the atom/* block only appears when AtomPrediction carries area+charge
    assert np.isfinite(result.metrics["atom/profile/w1_norm_mean"])
    assert np.isfinite(result.metrics["atom/area/r2"])
    assert np.isfinite(result.metrics["atom/charge/mae"])
    assert result.metrics["atom/n_test"] > 0

    predictions = np.load(result.run_dir / "predictions.npz")
    profile_pred = predictions["mol_profile_pred"]
    n_negative = (profile_pred < 0).sum()
    assert n_negative == 0, (
        f"{n_negative}/{profile_pred.size} rolled-up bins are negative -- "
        "softplus should make this structurally impossible"
    )
