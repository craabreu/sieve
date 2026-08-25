"""Fast tests for predictors/chemprop_dmpnn.py's pure, chemprop-free half:
the paper's exact 35/12 featurizers, min-max helpers, and the profile ->
Prediction conversion. No chemprop import at module scope (see the module
docstring) -- these must pass without chemprop installed.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from sieve_experiments.predictors.chemprop_dmpnn import (
    ATOM_FDIM,
    BOND_FDIM,
    PaperAtomFeaturizer,
    PaperBondFeaturizer,
    minmax_apply,
    minmax_fit,
    minmax_invert,
    prediction_from_profile,
)

# --- featurizer widths -------------------------------------------------


def test_atom_featurizer_width_matches_the_paper():
    assert len(PaperAtomFeaturizer()) == 35 == ATOM_FDIM


def test_bond_featurizer_width_matches_the_paper():
    assert len(PaperBondFeaturizer()) == 12 == BOND_FDIM


def test_atom_featurizer_output_has_the_declared_width():
    mol = Chem.MolFromSmiles("CCO")
    feat = PaperAtomFeaturizer()
    for atom in mol.GetAtoms():
        assert feat(atom).shape == (35,)


def test_bond_featurizer_output_has_the_declared_width():
    mol = Chem.MolFromSmiles("CCO")
    feat = PaperBondFeaturizer()
    for bond in mol.GetBonds():
        assert feat(bond).shape == (12,)


# --- atom featurizer, hand-checked one-hot blocks -----------------------


def _atom(smiles: str, idx: int):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetAtomWithIdx(idx)


def test_aromatic_carbon_in_benzene():
    # benzene: C, degree 3 (2 ring C + 1 implicit H... GetTotalDegree counts
    # heavy-atom bonds + implicit Hs = 2 ring neighbors + 1 H = 3), aromatic,
    # 1 implicit H, sp2, neutral, achiral.
    a = _atom("c1ccccc1", 0)
    x = PaperAtomFeaturizer()(a)
    element = x[0:8]
    degree = x[8:14]
    charge = x[14:19]
    hybridization = x[19:24]
    aromatic = x[24]
    total_h = x[25:30]
    chiral = x[30:34]
    mass = x[34]

    np.testing.assert_array_equal(element, [1, 0, 0, 0, 0, 0, 0, 0])  # C
    np.testing.assert_array_equal(degree, [0, 0, 0, 1, 0, 0])  # degree 3
    np.testing.assert_array_equal(charge, [0, 0, 1, 0, 0])  # 0
    np.testing.assert_array_equal(hybridization, [0, 1, 0, 0, 0])  # SP2
    assert aromatic == 1.0
    np.testing.assert_array_equal(total_h, [0, 1, 0, 0, 0])  # 1 H
    np.testing.assert_array_equal(chiral, [1, 0, 0, 0])  # unspecified
    assert 0.119 < mass < 0.121  # 0.01 * 12.011


def test_methyl_carbon():
    # ethanol's CH3: degree 4 (1 C neighbor + 3 H), sp3, 3 implicit H.
    a = _atom("CCO", 0)
    x = PaperAtomFeaturizer()(a)
    np.testing.assert_array_equal(x[8:14], [0, 0, 0, 0, 1, 0])  # degree 4
    np.testing.assert_array_equal(x[19:24], [0, 0, 1, 0, 0])  # SP3
    np.testing.assert_array_equal(x[25:30], [0, 0, 0, 1, 0])  # 3 H
    assert x[24] == 0.0  # not aromatic


def test_element_outside_the_papers_vocabulary_one_hots_to_all_zero():
    # chaos-store contains B, Si, Ge, Sb, Te -- outside COSMO-NET's own
    # 8-element vocabulary. No "unknown" slot in the paper's Table 1, so
    # this is a documented all-zero fallback, not a crash.
    a = _atom("[SiH4]", 0)
    x = PaperAtomFeaturizer()(a)
    np.testing.assert_array_equal(x[0:8], np.zeros(8))


def test_atom_featurizer_handles_none():
    np.testing.assert_array_equal(PaperAtomFeaturizer()(None), np.zeros(35))


# --- bond featurizer -----------------------------------------------------


def test_aromatic_conjugated_ring_bond():
    mol = Chem.MolFromSmiles("c1ccccc1")
    b = mol.GetBondWithIdx(0)
    x = PaperBondFeaturizer()(b)
    np.testing.assert_array_equal(x[0:4], [0, 0, 0, 1])  # AROMATIC
    assert x[4] == 1.0  # conjugated
    assert x[5] == 1.0  # in ring
    np.testing.assert_array_equal(x[6:12], [1, 0, 0, 0, 0, 0])  # stereo none


def test_non_ring_single_bond():
    mol = Chem.MolFromSmiles("CCO")
    b = mol.GetBondWithIdx(0)
    x = PaperBondFeaturizer()(b)
    np.testing.assert_array_equal(x[0:4], [1, 0, 0, 0])  # SINGLE
    assert x[5] == 0.0  # not in a ring


def test_bond_featurizer_handles_none():
    np.testing.assert_array_equal(PaperBondFeaturizer()(None), np.zeros(12))


# --- min-max helpers -------------------------------------------------


def test_minmax_round_trips():
    y = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    y_min, scale = minmax_fit(y)
    scaled = minmax_apply(y, y_min, scale)
    assert np.min(scaled[:, 0]) == 0.0
    assert np.max(scaled[:, 0]) == 1.0
    np.testing.assert_allclose(minmax_invert(scaled, y_min, scale), y)


def test_minmax_guards_a_constant_column():
    # A column with zero range must not divide by zero -- pyproject.toml
    # promotes RuntimeWarning to an error.
    y = np.array([[5.0], [5.0], [5.0]])
    y_min, scale = minmax_fit(y)
    assert scale[0] != 0.0
    scaled = minmax_apply(y, y_min, scale)
    np.testing.assert_allclose(scaled, 0.0)
    np.testing.assert_allclose(minmax_invert(scaled, y_min, scale), y)


def test_minmax_apply_uses_training_stats_not_its_own_data():
    train = np.array([[0.0], [10.0]])
    y_min, scale = minmax_fit(train)
    test = np.array([[20.0]])  # outside the training range on purpose
    scaled = minmax_apply(test, y_min, scale)
    assert scaled[0, 0] == 2.0  # extrapolates, doesn't reclip to [0,1]


# --- prediction_from_profile -------------------------------------------


def test_prediction_from_profile_area_and_charge():
    sigma_values = np.array([-1.0, 0.0, 1.0])
    profile = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    pred = prediction_from_profile(profile, sigma_values)
    assert pred.mol_area is not None
    assert pred.mol_charge_raw is not None
    np.testing.assert_allclose(pred.mol_profile, profile)
    np.testing.assert_allclose(pred.mol_area, np.array([6.0, 0.0]))
    np.testing.assert_allclose(pred.mol_charge_raw, np.array([2.0, 0.0]))  # -1+0+3
