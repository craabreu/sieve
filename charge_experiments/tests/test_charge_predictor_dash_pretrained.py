"""Fast-suite tests for dash_pretrained.py's pure-logic normalization
(std_weighted_normalize) -- no real DASH-tree clone needed. The real-tree
layer is _optional-tested only, in
test_charge_predictor_dash_pretrained_optional.py."""

from __future__ import annotations

import numpy as np
import pytest


def test_std_weighted_normalize_conserves_charge():
    """Verified against get_molecules_partial_charges' own real code (not
    the paper's printed eq 2/3, which has a sign error -- see module
    docstring): the renormalized atom charges of one conformer must sum
    exactly to that conformer's own net_charge."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2])  # sums to 0.0
    raw_std = np.array([0.2, 0.1, 0.3])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert out.sum() == pytest.approx(1.0)


def test_std_weighted_normalize_matches_hand_computed_example():
    """Direct hand computation of eq 4 with the code's own (not the paper
    printed) sign: Q_i' = Q_i + (Q_formal - sum(Q)) * sigma_i / sum(sigma)."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.5, -0.5])  # sums to 0.0
    raw_std = np.array([0.4, 0.1])  # sums to 0.5
    net_charge = np.array([0.5])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    residual = 0.5 - 0.0  # Q_formal - sum(Q)
    expected = np.array(
        [0.5 + residual * 0.4 / 0.5, -0.5 + residual * 0.1 / 0.5]
    )
    np.testing.assert_allclose(out, expected)


def test_std_weighted_normalize_floors_nonpositive_std_to_default():
    """A zero or negative std is floored to 0.1 (default_std_value),
    matching get_molecules_partial_charges' own hardcoded guard."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([0.0, -1.0])  # both non-positive -> both floored to 0.1
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    # Equal effective std (0.1 each) -> residual split evenly.
    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_floors_nan_std_to_default():
    """nan > 0 is False, so a NaN std is floored the same way a
    non-positive one is -- not left as a missing value."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([np.nan, np.nan])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_propagates_nan_charge_to_whole_conformer():
    """A NaN raw_charge on ANY atom in a conformer -- unlike a NaN std --
    is never floored or substituted: it makes the whole conformer's sum
    NaN, so every atom in that conformer ends up NaN. This is
    get_molecules_partial_charges' own real behavior (its own
    tot_charge_tree = sum(tree_raw_charges) is NaN if any entry is NaN),
    not something invented here."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.3, np.nan, -0.1])
    raw_std = np.array([0.2, 0.2, 0.2])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert np.all(np.isnan(out))


def test_std_weighted_normalize_multiple_conformers_are_independent():
    """A NaN in one conformer must not leak into another conformer's own,
    otherwise-clean renormalization."""
    from charge_experiments.predictors.dash_pretrained import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2, np.nan, 0.0])
    raw_std = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    net_charge = np.array([1.0, 0.0])
    mol_id = np.array([0, 0, 0, 1, 1])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 2)

    assert out[:3].sum() == pytest.approx(1.0)
    assert np.all(np.isnan(out[3:]))
