"""Fast-suite tests for normalize.py's pure-numpy normalization schemes --
no real DASH-tree clone needed. Moved here from
test_charge_predictor_dash_pretrained.py (std_weighted_normalize's own
implementation moved from predictors/dash_pretrained.py to normalize.py --
see docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md)."""

from __future__ import annotations

import numpy as np
import pytest


def test_std_weighted_normalize_conserves_charge():
    """Verified against get_molecules_partial_charges' own real code (not
    the paper's printed eq 2/3, which has a sign error -- see
    predictors/dash_pretrained.py's own module docstring for both proofs):
    the renormalized atom charges of one conformer must sum exactly to that
    conformer's own net_charge."""
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2])  # sums to 0.0
    raw_std = np.array([0.2, 0.1, 0.3])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert out.sum() == pytest.approx(1.0)


def test_std_weighted_normalize_matches_hand_computed_example():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.5, -0.5])  # sums to 0.0
    raw_std = np.array([0.4, 0.1])  # sums to 0.5
    net_charge = np.array([0.5])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    residual = 0.5 - 0.0  # Q_formal - sum(Q)
    expected = np.array([0.5 + residual * 0.4 / 0.5, -0.5 + residual * 0.1 / 0.5])
    np.testing.assert_allclose(out, expected)


def test_std_weighted_normalize_floors_nonpositive_std_to_default():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([0.0, -1.0])  # both non-positive -> both floored to 0.1
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_floors_nan_std_to_default():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([np.nan, np.nan])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_propagates_nan_charge_to_whole_conformer():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, np.nan, -0.1])
    raw_std = np.array([0.2, 0.2, 0.2])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert np.all(np.isnan(out))


def test_std_weighted_normalize_multiple_conformers_are_independent():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2, np.nan, 0.0])
    raw_std = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    net_charge = np.array([1.0, 0.0])
    mol_id = np.array([0, 0, 0, 1, 1])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 2)

    assert out[:3].sum() == pytest.approx(1.0)
    assert np.all(np.isnan(out[3:]))


def test_equal_weighted_normalize_conserves_charge():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2])  # sums to 0.0
    raw_std = np.array([999.0, 0.0, -5.0])  # deliberately ignored
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert out.sum() == pytest.approx(1.0)


def test_equal_weighted_normalize_splits_residual_evenly():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.5, -0.5])  # sums to 0.0
    raw_std = np.array([0.0, 0.0])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    # residual 1.0 split evenly across 2 atoms -> +0.5 each
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_equal_weighted_normalize_ignores_raw_std():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.5, -0.5])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out_a = equal_weighted_normalize(
        raw_charge, np.array([0.1, 0.9]), net_charge, mol_id, 1
    )
    out_b = equal_weighted_normalize(
        raw_charge, np.array([50.0, 0.001]), net_charge, mol_id, 1
    )

    np.testing.assert_array_equal(out_a, out_b)


def test_equal_weighted_normalize_propagates_nan_charge_to_whole_conformer():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.3, np.nan, -0.1])
    raw_std = np.array([0.2, 0.2, 0.2])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert np.all(np.isnan(out))


def test_normalizers_registry_has_both_schemes():
    from charge_experiments.normalize import (
        NORMALIZERS,
        equal_weighted_normalize,
        std_weighted_normalize,
    )

    assert NORMALIZERS["std_weighted"] is std_weighted_normalize
    assert NORMALIZERS["equal_weighted"] is equal_weighted_normalize
