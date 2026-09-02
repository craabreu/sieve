"""Fast-suite tests for sieve_predictor.py's config-building and
batch-building helpers -- real rdkit, real sieve.fit/predict, but no store,
no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def test_build_config_learns_codes_from_training_mols():
    from charge_experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert config.target_dim == 1
    assert "element" in config.attribute_codes


def test_batch_for_reads_mbis_charge_directly_off_the_mols():
    from charge_experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _batch_for,
        _build_config,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=4, seed=1)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    batch = _batch_for(mset.mols, config, with_target=True)
    assert batch.n_nodes == mset.n_atoms
    np.testing.assert_allclose(batch.y[:, 0], mset.atom_charge)


def test_build_config_defaults_to_a_single_attribute_level():
    """Unchanged default shape: attribute_levels not passed at all still
    produces the original one-level config, so neighbor_depth stays
    unusable unless a caller opts into a graded attribute_levels."""
    from charge_experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert config.attribute_levels == (DEFAULT_ATTRIBUTES,)
    assert config.neighbor_depth is None


def test_build_config_accepts_graded_attribute_levels_and_neighbor_depth():
    """The actual plumbing this feature needed: an explicit attribute_levels
    grouping makes a real (non-normalized-away) neighbor_depth usable, and
    the flat name list build_codes needs is derived from the grouping, not
    from the (here, intentionally stale) `attributes` argument."""
    from charge_experiments.predictors.sieve_predictor import _build_config

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    levels = (("element",), ("degree", "aromatic"))
    config = _build_config(
        mset.mols,
        attributes=("this", "is", "unused", "when", "attribute_levels", "is", "set"),
        attribute_levels=levels,
        neighbor_depth=1,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert config.attribute_levels == levels
    assert config.neighbor_depth == 1
    assert set(config.attribute_codes) == {"element", "degree", "aromatic"}


def test_build_config_defaults_edge_attributes_to_bond_type():
    from charge_experiments.predictors.sieve_predictor import _build_config

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=("element",),
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert set(config.edge_codes) == {"bond_type"}


def test_build_config_accepts_no_edge_attributes():
    """edge_attributes=() -- a pure-topology refinement, no bond attribute
    at all (build_codes's own documented behavior for an empty tuple)."""
    from charge_experiments.predictors.sieve_predictor import _build_config

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=0,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert config.edge_codes == {}
    assert config.attribute_codes.keys() == {"element"}


def test_sieve_charge_predictor_accepts_edge_attributes_end_to_end():
    """SievePredictor.__init__ -> fit() actually threads edge_attributes
    through to _build_config, not just the private helper directly."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=0)
    predictor = SievePredictor(
        attributes=("element",), edge_attributes=(), max_wl_depth=0
    )
    predictor.fit(train, train, rng=np.random.default_rng(0))
    assert predictor._config.edge_codes == {}
    assert predictor._config.attribute_codes.keys() == {"element"}
    pred = predictor.predict(train)
    assert pred.atom_charge.shape == (train.n_atoms,)


def test_sieve_charge_predictor_n_jobs_matches_sequential():
    """n_jobs is an execution-strategy choice, never a change in meaning --
    fit+predict under n_jobs=4 must reproduce n_jobs=None exactly."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=12, seed=4)

    seq = SievePredictor(max_wl_depth=2, minimum_support=1)
    seq.fit(mset, mset, rng=np.random.default_rng(0))
    seq_pred = seq.predict(mset)

    par = SievePredictor(max_wl_depth=2, minimum_support=1, n_jobs=4)
    par.fit(mset, mset, rng=np.random.default_rng(0))
    par_pred = par.predict(mset)

    np.testing.assert_array_equal(par_pred.atom_charge, seq_pred.atom_charge)


def test_sieve_charge_predictor_fits_with_neighbor_depth_end_to_end():
    """The predictor-level path (not just _build_config directly): fit and
    predict must both run and produce finite output with coarsening on."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(
        attribute_levels=(("element",), ("degree", "aromatic", "num_h")),
        neighbor_depth=1,
        max_wl_depth=2,
        minimum_support=1,
    )
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))
    assert predictor._config.neighbor_depth == 1


def test_sieve_charge_predictor_save_load_round_trips_neighbor_depth(tmp_path):
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    fitted = SievePredictor(
        attribute_levels=(("element",), ("degree", "aromatic")),
        neighbor_depth=1,
        max_wl_depth=2,
        minimum_support=1,
    )
    fitted.fit(mset, mset, rng=rng)
    path = tmp_path / "model.npz"
    fitted.save_model_state(path)

    loaded = SievePredictor()
    loaded.load_model_state(path)
    assert loaded._config.neighbor_depth == 1
    np.testing.assert_allclose(
        fitted.predict(mset).atom_charge, loaded.predict(mset).atom_charge
    )


def test_sieve_charge_predictor_fits_and_predicts_end_to_end():
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))


def test_sieve_charge_predictor_predict_equals_predict_raw_atom_charge():
    """predict() must stay behavior-identical: exactly
    predict_raw(...).atom_charge, since sieve.predict is itself just
    sieve.predict_detailed(...).value."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    np.testing.assert_array_equal(pred.atom_charge, raw.atom_charge)
    assert raw.atom_std.shape == raw.atom_charge.shape


def test_sieve_charge_predictor_save_and_load_model_state_round_trips(tmp_path):
    """A predictor that loads a saved model (no fit() call at all) predicts
    identically to a freshly-fit one on the same train data."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=2)
    test = synthetic_molecule_set(n_mol=4, seed=3)
    rng = np.random.default_rng(0)

    fitted = SievePredictor(max_wl_depth=2, minimum_support=1)
    fitted.fit(train, train, rng=rng)
    fitted_pred = fitted.predict(test)

    model_path = tmp_path / "sieve-model.npz"
    fitted.save_model_state(model_path)

    loaded = SievePredictor(max_wl_depth=2, minimum_support=1)
    loaded.load_model_state(model_path)
    loaded_pred = loaded.predict(test)

    np.testing.assert_array_equal(loaded_pred.atom_charge, fitted_pred.atom_charge)


def test_sieve_charge_predictor_load_model_state_skips_fit(tmp_path, monkeypatch):
    """Proves load_model_state never calls sieve.fit() -- the whole point
    of persisting the model."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=2)
    test = synthetic_molecule_set(n_mol=4, seed=3)
    rng = np.random.default_rng(0)

    fitted = SievePredictor(max_wl_depth=2, minimum_support=1)
    fitted.fit(train, train, rng=rng)
    model_path = tmp_path / "sieve-model.npz"
    fitted.save_model_state(model_path)

    loaded = SievePredictor(max_wl_depth=2, minimum_support=1)

    def _boom(*args, **kwargs):
        raise AssertionError("fit() must not be called")

    monkeypatch.setattr(loaded, "fit", _boom)
    loaded.load_model_state(model_path)  # must not raise
    pred = loaded.predict(test)
    assert pred.atom_charge.shape == (test.n_atoms,)


def test_predict_loo_raw_backs_off_instead_of_recalling_the_node():
    """Leave-one-out removes a node's own contribution from its class mean
    before the support check, so at minimum_support=1 a singleton class has
    eff_n == 0 and fails it -- the node backs off to its parent instead of
    recalling itself. In-sample prediction has no such guard, so its error
    is optimistically low. The gap is the memorization signal."""
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=0)
    p = SievePredictor(max_wl_depth=3, minimum_support=1, report_loo=True)
    p.fit(mset, mset, rng=np.random.default_rng(0))

    in_sample = p.predict_raw(mset).atom_charge
    loo = p.predict_loo_raw(mset).atom_charge

    in_sample_mae = float(np.nanmean(np.abs(in_sample - mset.atom_charge)))
    loo_mae = float(np.nanmean(np.abs(loo - mset.atom_charge)))
    assert loo_mae > in_sample_mae


def test_predict_loo_raw_requires_a_fitted_model():
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    p = SievePredictor()
    with pytest.raises(RuntimeError, match="fit"):
        p.predict_loo_raw(synthetic_molecule_set(n_mol=2))


def test_report_loo_defaults_off_and_is_recorded_on_the_predictor():
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    assert SievePredictor().report_loo is False
    assert SievePredictor(report_loo=True).report_loo is True
