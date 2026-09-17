"""HoseLookupPredictor against a synthetic MoleculeSet. Skipped without the
optional hosegen dependency."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tests.helpers import synthetic_molecule_set

pytest.importorskip("hosegen", reason="optional dependency: see the hose extra")


def _predictor(**kw):
    from experiments.predictors.hose import HoseLookupPredictor

    return HoseLookupPredictor(**kw)


def test_predicting_the_training_set_reproduces_its_class_means():
    """Every atom matches its own deepest class, so each prediction is the
    mean of the training atoms sharing that atom's full code."""
    train = synthetic_molecule_set(n_mol=8, seed=0)
    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(train).atom_value

    assert got.shape == (train.n_atoms,)
    assert np.isfinite(got).all()
    # the class means are an average of the targets, so they cannot leave the
    # range of the targets
    y = train.atom_target
    assert got.min() >= y.min() - 1e-12
    assert got.max() <= y.max() + 1e-12


def test_an_unseen_element_falls_back_to_the_global_mean():
    train = synthetic_molecule_set(n_mol=6, seed=1)
    # CCCl is in the synthetic set; CCBr is not, so bromine is unseen
    from experiments.data import MoleculeSet
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("CCBr", params)
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("MBIScharge", 0.0)
    test = MoleculeSet(mols=[mol], atom_property="MBIScharge")

    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(test).atom_value

    bromine = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == "Br"]
    assert len(bromine) == 1
    assert got[bromine[0]] == pytest.approx(float(np.mean(train.atom_target)))


def test_n_min_refuses_a_class_below_the_threshold():
    """With a threshold above any class's support, every atom falls all the
    way back to the global mean."""
    train = synthetic_molecule_set(n_mol=4, seed=2)
    p = _predictor(max_radius=5, n_min=10_000)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(train).atom_value
    assert got == pytest.approx(
        np.full(train.n_atoms, float(np.mean(train.atom_target)))
    )


def test_predict_before_fit_raises():
    train = synthetic_molecule_set(n_mol=2, seed=3)
    with pytest.raises(RuntimeError, match="fit must be called"):
        _predictor().predict(train)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="max_radius"):
        _predictor(max_radius=0)
    with pytest.raises(ValueError, match="n_min"):
        _predictor(n_min=0)


def test_a_radius_above_the_generator_ceiling_is_refused_up_front():
    """hosegen caps at 12 spheres and raises a bare IndexError past it. Catch
    that in the constructor rather than hours into a fit."""
    with pytest.raises(ValueError, match="12"):
        _predictor(max_radius=13)


def test_registered_under_its_name():
    from experiments.predictors import build

    p = build("hose", {"max_radius": 3})
    assert p.name == "hose"
    assert p.max_radius == 3


def test_counts_from_two_disjoint_halves_add_up_to_the_union():
    """Spec acceptance check 4. The accumulation is a plain sum over atoms, so
    fitting two halves and adding their counts must reproduce fitting the
    union. This checks the accumulation, not a property of the method."""
    from experiments.data import MoleculeSet

    whole = synthetic_molecule_set(n_mol=8, seed=5)
    half_a = MoleculeSet(mols=whole.mols[:4], atom_property=whole.atom_property)
    half_b = MoleculeSet(mols=whole.mols[4:], atom_property=whole.atom_property)

    fitted = []
    for part in (half_a, half_b, whole):
        p = _predictor(max_radius=5)
        p.fit(part, part, rng=np.random.default_rng(0))
        fitted.append(p)
    first, second, union = fitted

    for k in range(1, 6):
        keys = set(first._tables[k]) | set(second._tables[k])
        assert keys == set(union._tables[k])
        for key in keys:
            counted = (
                first._tables[k].get(key, (0.0, 0))[1]
                + second._tables[k].get(key, (0.0, 0))[1]
            )
            assert counted == union._tables[k][key][1]


def test_two_conformers_of_one_molecule_get_the_same_prediction():
    """Spec acceptance check 5. HOSE codes read the 2D graph, so conformers are
    indistinguishable to this arm and to the Sieve arms alike, which puts a
    floor under the error that neither can cross."""
    from experiments.data import MoleculeSet
    from rdkit import Chem

    train = synthetic_molecule_set(n_mol=8, seed=0)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mols = []
    for _ in range(2):
        mol = Chem.MolFromSmiles("CCO", params)
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 0.0)
        mols.append(mol)
    test = MoleculeSet(mols=mols, atom_property="MBIScharge")

    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(test).atom_value

    n = mols[0].GetNumAtoms()
    assert got[:n] == pytest.approx(got[n:])


def test_matched_radius_records_where_each_atom_was_answered():
    train = synthetic_molecule_set(n_mol=8, seed=0)
    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    p.predict(train)

    radius = p.matched_radius
    assert radius.shape == (train.n_atoms,)
    # every training atom's own full code is in the table, so all are answered
    # at the deepest radius
    assert (radius == 5).all()


def test_matched_radius_is_zero_for_the_global_mean_fallback():
    train = synthetic_molecule_set(n_mol=4, seed=2)
    p = _predictor(max_radius=5, n_min=10_000)
    p.fit(train, train, rng=np.random.default_rng(0))
    p.predict(train)
    assert (p.matched_radius == 0).all()


def test_matched_radius_before_predict_raises():
    train = synthetic_molecule_set(n_mol=2, seed=3)
    p = _predictor()
    p.fit(train, train, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="predict must be called"):
        _ = p.matched_radius
