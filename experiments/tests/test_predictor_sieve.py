"""Fast-suite tests for sieve_predictor.py's config-building and
batch-building helpers -- real rdkit, real sieve.fit/predict, but no store,
no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def test_build_config_learns_codes_from_training_mols():
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )

    from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _batch_for,
        _build_config,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=4, seed=1)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    batch = _batch_for(
        mset.mols, config, atom_property=mset.atom_property, with_target=True
    )
    assert batch.n_nodes == mset.n_atoms
    np.testing.assert_allclose(batch.y[:, 0], mset.atom_target)


def test_build_config_defaults_to_a_single_attribute_level():
    """Unchanged default shape: attribute_levels not passed at all still
    produces the original one-level config, so neighbor_depth stays
    unusable unless a caller opts into a graded attribute_levels."""
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )

    from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.predictors.sieve_predictor import _build_config

    from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.predictors.sieve_predictor import _build_config

    from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.predictors.sieve_predictor import _build_config

    from experiments.tests.helpers import synthetic_molecule_set

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
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=0)
    predictor = SievePredictor(
        attributes=("element",), edge_attributes=(), max_wl_depth=0
    )
    predictor.fit(train, train, rng=np.random.default_rng(0))
    assert predictor._config.edge_codes == {}
    assert predictor._config.attribute_codes.keys() == {"element"}
    pred = predictor.predict(train)
    assert pred.atom_value.shape == (train.n_atoms,)


def test_sieve_charge_predictor_n_jobs_matches_sequential():
    """n_jobs is an execution-strategy choice, never a change in meaning --
    fit+predict under n_jobs=4 must reproduce n_jobs=None exactly."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=12, seed=4)

    seq = SievePredictor(max_wl_depth=2, minimum_support=1)
    seq.fit(mset, mset, rng=np.random.default_rng(0))
    seq_pred = seq.predict(mset)

    par = SievePredictor(max_wl_depth=2, minimum_support=1, n_jobs=4)
    par.fit(mset, mset, rng=np.random.default_rng(0))
    par_pred = par.predict(mset)

    np.testing.assert_array_equal(par_pred.atom_value, seq_pred.atom_value)


def test_sieve_charge_predictor_fits_with_neighbor_depth_end_to_end():
    """The predictor-level path (not just _build_config directly): fit and
    predict must both run and produce finite output with coarsening on."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

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

    assert pred.atom_value.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_value))
    assert predictor._config.neighbor_depth == 1


def test_sieve_charge_predictor_save_load_round_trips_neighbor_depth(tmp_path):
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

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
        fitted.predict(mset).atom_value, loaded.predict(mset).atom_value
    )


def test_sieve_charge_predictor_fits_and_predicts_end_to_end():
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_value.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_value))


def test_sieve_charge_predictor_predict_equals_predict_raw_atom_charge():
    """predict() must stay behavior-identical: exactly
    predict_raw(...).atom_value, since sieve.predict is itself just
    sieve.predict_detailed(...).value."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    np.testing.assert_array_equal(pred.atom_value, raw.atom_value)
    assert raw.atom_std.shape == raw.atom_value.shape


def test_sieve_charge_predictor_save_and_load_model_state_round_trips(tmp_path):
    """A predictor that loads a saved model (no fit() call at all) predicts
    identically to a freshly-fit one on the same train data."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

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

    np.testing.assert_array_equal(loaded_pred.atom_value, fitted_pred.atom_value)


def test_sieve_charge_predictor_load_model_state_skips_fit(tmp_path, monkeypatch):
    """Proves load_model_state never calls sieve.fit() -- the whole point
    of persisting the model."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

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
    assert pred.atom_value.shape == (test.n_atoms,)


def test_predict_loo_raw_backs_off_instead_of_recalling_the_node():
    """Leave-one-out removes a node's own contribution from its class mean
    before the support check, so at minimum_support=1 a singleton class has
    eff_n == 0 and fails it -- the node backs off to its parent instead of
    recalling itself. In-sample prediction has no such guard, so its error
    is optimistically low. The gap is the memorization signal."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=0)
    p = SievePredictor(max_wl_depth=3, minimum_support=1, report_loo=True)
    p.fit(mset, mset, rng=np.random.default_rng(0))

    in_sample = p.predict_raw(mset).atom_value
    loo = p.predict_loo_raw(mset).atom_value

    in_sample_mae = float(np.nanmean(np.abs(in_sample - mset.atom_target)))
    loo_mae = float(np.nanmean(np.abs(loo - mset.atom_target)))
    assert loo_mae > in_sample_mae


def test_predict_loo_raw_requires_a_fitted_model():
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    p = SievePredictor()
    with pytest.raises(RuntimeError, match="fit"):
        p.predict_loo_raw(synthetic_molecule_set(n_mol=2))


def test_report_loo_defaults_off_and_is_recorded_on_the_predictor():
    from experiments.predictors.sieve_predictor import SievePredictor

    assert SievePredictor().report_loo is False
    assert SievePredictor(report_loo=True).report_loo is True


def test_sieve_fits_a_non_charge_property():
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, atom_property="alpha")
    predictor = SievePredictor(max_wl_depth=1, minimum_support=1)
    predictor.fit(mset, mset, rng=np.random.default_rng(0))
    pred = predictor.predict(mset)
    assert pred.atom_value.shape == (mset.n_atoms,)
    assert np.isfinite(pred.atom_value).all()


def test_save_and_load_codes_round_trip(tmp_path):
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
        load_codes,
        save_codes,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )

    path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, path)
    codes, edge_codes = load_codes(path)

    assert codes == {k: dict(v) for k, v in config.attribute_codes.items()}
    assert edge_codes == {k: dict(v) for k, v in config.edge_codes.items()}


def test_build_config_with_frozen_codes_skips_build_codes(monkeypatch):
    """codes/edge_codes given -> build_codes must not be called at all,
    which is the whole point (a shard must not discover its own, shifted
    vocabulary)."""
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    frozen_codes = {"element": {"C": 0, "O": 1, "N": 2, "Cl": 3}}
    frozen_edge_codes = {"bond_type": {"SINGLE": 0, "DOUBLE": 1}}

    def _boom(*a, **k):
        raise AssertionError("build_codes must not be called with frozen codes")

    monkeypatch.setattr("sieve.io.rdkit_adapter.build_codes", _boom)

    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
        codes=frozen_codes,
        edge_codes=frozen_edge_codes,
    )
    assert dict(config.attribute_codes["element"]) == frozen_codes["element"]


def test_codes_path_freezes_the_vocabulary_across_disjoint_shards(tmp_path):
    """The blocking bug this exists to fix: two shards whose training
    molecules don't cover the same element set must still end up
    mergeable, because both used one frozen vocabulary rather than each
    discovering its own."""
    from experiments.predictors.sieve_predictor import (
        SievePredictor,
        _build_config,
        save_codes,
    )

    import sieve
    from experiments.tests.helpers import synthetic_molecule_set

    whole = synthetic_molecule_set(n_mol=16, seed=0)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=2,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    shard_a = whole.select(np.arange(whole.n_conformers) < whole.n_conformers // 2)
    shard_b = whole.select(np.arange(whole.n_conformers) >= whole.n_conformers // 2)

    pred_a = SievePredictor(
        attributes=("element",),
        edge_attributes=(),
        max_wl_depth=2,
        minimum_support=1,
        codes_path=codes_path,
    )
    pred_a.fit(shard_a, shard_a, rng=np.random.default_rng(0))
    pred_b = SievePredictor(
        attributes=("element",),
        edge_attributes=(),
        max_wl_depth=2,
        minimum_support=1,
        codes_path=codes_path,
    )
    pred_b.fit(shard_b, shard_b, rng=np.random.default_rng(0))

    assert pred_a._config.schema_version == pred_b._config.schema_version
    merged = sieve.merge.merge_models(pred_a._model, pred_b._model)
    assert merged is not None


def test_predict_raw_from_batch_matches_predict_raw(tmp_path):
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=2)
    test = synthetic_molecule_set(n_mol=4, seed=3)

    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(train, train, rng=np.random.default_rng(0))

    direct = predictor.predict_raw(test)

    batch = predictor.build_predict_batch(test.mols)
    from_batch = predictor.predict_raw_from_batch(batch)

    np.testing.assert_array_equal(direct.atom_value, from_batch.atom_value)
    np.testing.assert_array_equal(direct.atom_std, from_batch.atom_std)


def test_predict_raw_from_batch_reused_across_two_models_sharing_codes(tmp_path):
    """The actual point of the seam: one batch, two models at different
    depths but sharing one frozen vocabulary, predicting without a second
    featurization pass."""
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=2)
    test = synthetic_molecule_set(n_mol=4, seed=3)

    shallow = SievePredictor(max_wl_depth=1, minimum_support=1)
    shallow.fit(train, train, rng=np.random.default_rng(0))
    deep = SievePredictor(max_wl_depth=3, minimum_support=1)
    deep.fit(train, train, rng=np.random.default_rng(0))

    batch = shallow.build_predict_batch(test.mols)
    shallow_pred = shallow.predict_raw_from_batch(batch)
    deep_pred = deep.predict_raw_from_batch(batch)

    expected_shallow = shallow.predict_raw(test)
    expected_deep = deep.predict_raw(test)
    np.testing.assert_array_equal(shallow_pred.atom_value, expected_shallow.atom_value)
    np.testing.assert_array_equal(deep_pred.atom_value, expected_deep.atom_value)


def test_merge_states_matches_fitting_the_union_directly(tmp_path):
    """merge_states(fit(A), fit(B)) must equal fit(A + B) -- the CV
    assembly's own load-bearing invariant, checked at the predictor
    seam rather than only at sieve.merge's own unit-test level."""
    from experiments.predictors.sieve_predictor import (
        SievePredictor,
        _build_config,
        save_codes,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    whole = synthetic_molecule_set(n_mol=16, seed=0)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=2,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    half = whole.n_conformers // 2
    shard_a = whole.select(np.arange(whole.n_conformers) < half)
    shard_b = whole.select(np.arange(whole.n_conformers) >= half)

    def _fit_and_save(mset, path):
        predictor = SievePredictor(
            attributes=("element",),
            edge_attributes=(),
            max_wl_depth=2,
            minimum_support=1,
            codes_path=codes_path,
        )
        predictor.fit(mset, mset, rng=np.random.default_rng(0))
        predictor.save_model_state(path)
        return predictor

    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    _fit_and_save(shard_a, path_a)
    _fit_and_save(shard_b, path_b)

    merged_path = tmp_path / "merged.npz"
    SievePredictor.merge_states([path_a, path_b], merged_path)

    direct = SievePredictor(
        attributes=("element",),
        edge_attributes=(),
        max_wl_depth=2,
        minimum_support=1,
        codes_path=codes_path,
    )
    direct.fit(whole, whole, rng=np.random.default_rng(0))

    from_merge = SievePredictor(max_wl_depth=2, minimum_support=1)
    from_merge.load_model_state(merged_path)

    test = synthetic_molecule_set(n_mol=4, seed=99)
    np.testing.assert_allclose(
        from_merge.predict(test).atom_value, direct.predict(test).atom_value
    )
