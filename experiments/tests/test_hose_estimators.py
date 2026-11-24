"""Sieve's class estimators read over the classes of the HOSE trees.

Without ring closures, the tree of an atom is the unrolled tree of its Sieve
label with element and formal charge on the atoms and bond types on the bonds
(predictors/hose_trees.py), so the HOSE lookup with Sieve's estimator on
plain trees must reproduce that Sieve model exactly. That is what makes the
same estimator on closure trees differ from Sieve by the ring closure alone.
Skipped without the optional hosegen dependency, which the HOSE module
imports for its code keys.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hosegen", reason="optional dependency: see the hose extra")

TRAIN = [
    "c1ccccc1O",
    "C1CC1C(=O)[O-]",
    "C[N+](C)(C)C",
    "C1CCCCC1N",
    "c1ccncc1",
    "CC#N",
    "OCC1CCOC1",
    "CC(=O)OC",
    "C1CCC1",
    "Cc1ccccc1",
    "OC1CCCCC1",
    "CC(C)O",
]
TEST = [
    "c1ccccc1N",
    "C1CC1C(=O)O",
    "C[N+](C)(C)CC",
    "Oc1ccncc1",
    "C1CCCC1",
    "CCOC(C)=O",
]


def _mset(smiles, seed):
    from experiments.data import MoleculeSet
    from rdkit import Chem

    mols = []
    for smi in smiles:
        # per molecule, so a molecule carries the same charges in any set
        rng = np.random.default_rng([seed, *map(ord, smi)])
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        for atom in mol.GetAtoms():
            base = {"C": -0.1, "H": 0.1, "O": -0.5, "N": -0.4}[atom.GetSymbol()]
            atom.SetDoubleProp("MBIScharge", base + float(rng.normal(scale=0.05)))
        mols.append(mol)
    return MoleculeSet(mols=mols, atom_property="MBIScharge")


def _hose(**kw):
    from experiments.predictors.hose import HoseLookupPredictor

    return HoseLookupPredictor(**kw)


def _sieve(depth, shrinkage_weight):
    from experiments.predictors.sieve_predictor import SievePredictor

    return SievePredictor(
        attributes=("element", "formal_charge"),
        edge_attributes=("bond_type",),
        max_wl_depth=depth,
        class_estimator="continuation",
        shrinkage_weight=shrinkage_weight,
    )


@pytest.mark.parametrize(
    ("estimator", "shrinkage_weight"),
    [("continuation", None), ("continuation-eb", "empirical_bayes")],
)
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_plain_trees_with_sieves_estimator_reproduce_sieve(
    estimator, shrinkage_weight, depth
):
    train, test = _mset(TRAIN, 0), _mset(TEST + TRAIN[:3], 1)
    rng = np.random.default_rng(0)
    hose = _hose(max_radius=depth, keys="tree", estimator=estimator)
    hose.fit(train, train, rng=rng)
    sieve = _sieve(depth, shrinkage_weight)
    sieve.fit(train, train, rng=rng)
    got = hose.predict(test).atom_value
    expected = sieve.predict(test).atom_value
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    # the test set must exercise backoff, or the estimators are not tested
    assert (hose.matched_radius < depth).any()


@pytest.mark.parametrize("estimator", ["continuation", "continuation-eb"])
def test_the_estimator_survives_a_round_trip_through_merged_shard_states(
    tmp_path, estimator
):
    """A CV fold trains on merged shard states, never on molecules."""
    from experiments.hose_artifact import (
        load_hose_state,
        merge_hose_states,
        save_hose_state,
    )

    rng = np.random.default_rng(0)
    whole = _hose(max_radius=4, keys="closure-tree", estimator=estimator)
    whole.fit(_mset(TRAIN, 0), None, rng=rng)
    halves = []
    for i, part in enumerate((TRAIN[:6], TRAIN[6:])):
        p = _hose(max_radius=4, keys="closure-tree")
        p.fit(_mset(part, 0), None, rng=rng)
        path = tmp_path / f"s{i}.npz"
        save_hose_state(p.model_state(), path)
        halves.append(load_hose_state(path))
    merged = _hose(max_radius=4, keys="closure-tree", estimator=estimator)
    merged.set_model_state(merge_hose_states(*halves))
    test = _mset(TEST, 1)
    np.testing.assert_allclose(
        merged.predict(test).atom_value,
        whole.predict(test).atom_value,
        rtol=0,
        atol=1e-12,
    )


def test_closure_trees_with_the_estimator_differ_from_plain_trees():
    train, test = _mset(TRAIN, 0), _mset(TEST, 1)
    values = {}
    for keys in ("tree", "closure-tree"):
        p = _hose(max_radius=3, keys=keys, estimator="continuation")
        p.fit(train, train, rng=np.random.default_rng(0))
        values[keys] = p.predict(test).atom_value
    assert np.isfinite(values["closure-tree"]).all()
    assert not np.allclose(values["tree"], values["closure-tree"])


def test_an_unknown_estimator_is_refused():
    with pytest.raises(ValueError, match="estimator"):
        _hose(keys="tree", estimator="pitman-yor")


def test_the_estimator_needs_tree_keys():
    """A code's string does not say the charge of its central atom, which the
    attribute level of the Sieve chain needs."""
    with pytest.raises(ValueError, match="estimator"):
        _hose(keys="code", estimator="continuation")


def test_the_estimator_needs_the_global_fallback():
    with pytest.raises(ValueError, match="fallback"):
        _hose(keys="tree", estimator="continuation", fallback="none")


def test_a_state_without_roots_is_refused_by_the_estimator():
    """Shard fits saved before the roots existed lack the attribute level."""
    import dataclasses

    p = _hose(max_radius=2, keys="tree")
    p.fit(_mset(TRAIN, 0), None, rng=np.random.default_rng(0))
    legacy = dataclasses.replace(p.model_state(), roots=None)
    q = _hose(max_radius=2, keys="tree", estimator="continuation")
    q.set_model_state(legacy)
    with pytest.raises(ValueError, match="roots"):
        q.predict(_mset(TEST, 1))
