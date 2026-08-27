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
        mset.mols, attributes=DEFAULT_ATTRIBUTES, target_dim=1,
        max_wl_depth=3, minimum_support=1, shrinkage_strength=None,
    )
    batch = _batch_for(mset.mols, config, with_target=True)
    assert batch.n_nodes == mset.n_atoms
    np.testing.assert_allclose(batch.y[:, 0], mset.atom_charge)


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
