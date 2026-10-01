"""Tests for experiments/geometry.py: the geometry-against-graph columns, on hand-placed
coordinates, and their backfill into an existing store."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rdkit")


def _placed(smiles: str, coords):
    """``smiles`` (no hydrogens added) with one conformer at ``coords`` (angstrom)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*xyz))
    mol.AddConformer(conf, assignId=True)
    return mol


def _angle(theta_degrees: float, bond: float = 1.53):
    """Three atoms with two bonds of length ``bond`` at angle ``theta``."""
    t = math.radians(theta_degrees)
    return [
        (0.0, 0.0, 0.0),
        (bond, 0.0, 0.0),
        (bond - bond * math.cos(t), bond * math.sin(t), 0.0),
    ]


def test_a_healthy_geometry_is_not_flagged():
    from experiments.geometry import geometry_record

    g = geometry_record(_placed("CCC", _angle(112.0)))
    assert not g["stretched_bond"] and not g["close_contact"]
    assert 0.95 < g["max_bond_ratio"] < 1.05
    assert g["min_contact_13"] > 1.6
    assert math.isnan(g["min_contact_far"])


def test_a_stretched_bond_is_flagged():
    from experiments.geometry import STRETCHED, geometry_record

    coords = [(0.0, 0.0, 0.0), (1.53 * 1.3, 0.0, 0.0)]
    g = geometry_record(_placed("CC", coords))
    assert g["max_bond_ratio"] > STRETCHED
    assert g["stretched_bond"]


def test_a_13_pair_at_bonding_distance_is_flagged_but_a_tight_angle_is_not():
    from experiments.geometry import geometry_record

    # 60 degrees puts the two ends 1.53 A apart, a C-C bond the graph lacks.
    assert geometry_record(_placed("CCC", _angle(60.0)))["close_contact"]
    # 75 degrees: 1.86 A, ratio 1.22 -- tight, below the far bound, above the 1,3 one.
    g = geometry_record(_placed("CCC", _angle(75.0)))
    assert 1.15 < g["min_contact_13"] < 1.3
    assert not g["close_contact"]


def test_a_clash_three_bonds_apart_is_flagged():
    from experiments.geometry import geometry_record

    # Butane folded so that its terminal carbons are 1.65 A apart (ratio ~1.09).
    coords = [(0.0, 0.0, 0.0), (1.2, 1.0, 0.0), (2.6, 1.0, 0.0), (1.65, 0.0, 0.0)]
    g = geometry_record(_placed("CCCC", coords))
    assert g["min_contact_far"] < 1.15
    assert g["close_contact"]


def test_hydrogens_are_left_out_of_the_contacts():
    from experiments.geometry import geometry_record
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=1)
    g = geometry_record(mol)
    assert not g["close_contact"]
    assert g["min_contact_13"] > 1.2


def test_annotate_geometry_backfills_a_store(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from experiments.data import mol_to_blob
    from experiments.geometry import GEOMETRY_COLUMNS, annotate_geometry

    store = tmp_path / "s"
    store.mkdir()
    mols = [
        _placed("CCC", _angle(112.0)),
        _placed("CCC", _angle(60.0)),
        _placed("CC", [(0, 0, 0), (2.2, 0, 0)]),
    ]
    pd.DataFrame(
        {"id": ["a", "b", "c"], "mol": [mol_to_blob(m) for m in mols]}
    ).to_parquet(store / "molecules.parquet")

    out = annotate_geometry("s", stores_root=tmp_path, workers=2)
    assert out == {"rows": 3, "stretched_bond": 1, "close_contact": 1, "either": 2}
    df = pd.read_parquet(store / "molecules.parquet")
    assert set(GEOMETRY_COLUMNS) <= set(df.columns)
    assert df["close_contact"].tolist() == [False, True, False]
    assert df["stretched_bond"].tolist() == [False, False, True]
    assert not (store / "molecules.parquet.tmp").exists()
