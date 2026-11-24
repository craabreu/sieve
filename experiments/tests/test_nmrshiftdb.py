"""NMRShiftDB's lookup: its query stripping, its distinct-pair counting, and
its codes. The counting tests stand in for the Java generator with fixed codes;
the last test runs the generator and is skipped when setup.sh has not built
it."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.data import MoleculeSet
from experiments.predictors.hose_keys import sphere_prefix, stripped_code
from rdkit import Chem


def _mol(smiles: str, charges: list[float]):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, params))
    assert mol.GetNumAtoms() == len(charges)
    for atom, q in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", q)
    return mol


def _fake_codes(monkeypatch, codes_by_smiles: dict[str, list[tuple[str, ...]]]):
    """Replace the Java generator with fixed codes per molecule, looked up by
    the molecule's canonical SMILES."""
    import experiments.predictors.nmrshiftdb_codes as module

    def fake(mols, query_spheres=0, jobs=None):
        rows = []
        for mol in mols:
            for row in codes_by_smiles[Chem.MolToSmiles(Chem.RemoveHs(mol))]:
                rows.append(list(row[:2]) + ([row[2]] if query_spheres else []))
        return rows

    monkeypatch.setattr(module, "nmrshiftdb_codes", fake)


def _predictor(**kw):
    from experiments.predictors.hose import HoseLookupPredictor

    return HoseLookupPredictor(keys="nmrshiftdb", **kw)


@pytest.mark.parametrize(
    ("code", "k", "expected"),
    [
        ("C-4;C(O//)//", 1, "C-4;C("),
        ("C-4;C(O//)//", 3, "C-4;C(O//"),
        # the query of a hydrogen at three spheres ends in a delimiter the
        # generator adds; the pieces are what count
        ("H-1;C(HHC/HHO/)", 3, "H-1;C(HHC/HHO/"),
        ("H-1;C(HHC/HHO/H)//", 4, "H-1;C(HHC/HHO/H)"),
        ("H-1;C(HHC/HHO/H)//", 6, "H-1;C(HHC/HHO/H)//"),
    ],
)
def test_a_query_is_stripped_as_nmrshiftdb_strips_it(code, k, expected):
    assert stripped_code(code, k) == expected


@pytest.mark.parametrize("k", range(1, 7))
def test_a_stripped_code_is_the_prefix_a_stored_code_is_cut_to(k):
    """Matching by prefix (``like '<query>%'``) is matching the stored code
    cut at the same sphere, which is how the lookup is indexed."""
    code = "C-3;*C*CN(*C,*C,=O/*&,H,/H)//"
    assert stripped_code(code, k) == sphere_prefix(code, k)
    assert code.startswith(stripped_code(code, k))


def test_a_value_counts_once_per_molecule_and_class(monkeypatch):
    """The two hydrogens of formaldehyde share a class and a value; with both
    codes of each atom also in that class, the class still holds one pair, as
    ``SELECT DISTINCT VALUE, SPECTRUM_ID`` would."""
    form = _mol("C=O", [0.3, -0.4, 0.05, 0.05])
    acet = _mol("CC=O", [0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 0.01])
    h = ("H-1;C(=O/)", "H-1;C(=O/)", "H-1;C(=O/)")
    _fake_codes(
        monkeypatch,
        {
            "C=O": [("C-3;=O(/)",) * 3, ("O-1;=C(/)",) * 3, h, h],
            "CC=O": [("C-4;C(/)",) * 3, ("C-3;C=O(/)",) * 3, ("O-1;=C(/)",) * 3]
            + [("H-1;C(C/)",) * 3] * 3
            + [h],
        },
    )
    train = MoleculeSet(mols=[form, acet], atom_property="MBIScharge")
    p = _predictor(max_radius=2)
    p.fit(train, train, rng=np.random.default_rng(0))

    mean, count = p._tables[2]["H-1;C(=O/"]
    assert count == 2  # 0.05 from formaldehyde once, 0.01 from acetaldehyde
    assert mean == pytest.approx(0.03)
    mean, count = p._tables[2]["H-1;C(C/"]
    assert count == 1  # three methyl hydrogens, one value
    assert mean == pytest.approx(0.0)


def test_an_atom_matching_nothing_is_left_unpredicted(monkeypatch):
    form = _mol("C=O", [0.3, -0.4, 0.05, 0.05])
    hcn = _mol("C#N", [0.1, -0.2, 0.1])
    h = ("H-1;C(=O/)",) * 3
    _fake_codes(
        monkeypatch,
        {
            "C=O": [("C-3;=O(/)",) * 3, ("O-1;=C(/)",) * 3, h, h],
            "C#N": [("C-2;%N(/)",) * 3, ("N-1;%C(/)",) * 3, ("H-1;C(%N/)",) * 3],
        },
    )
    train = MoleculeSet(mols=[form], atom_property="MBIScharge")
    test = MoleculeSet(mols=[hcn], atom_property="MBIScharge")

    exact = _predictor(max_radius=2, fallback="none")
    exact.fit(train, train, rng=np.random.default_rng(0))
    got = exact.predict(test).atom_value
    # carbon and nitrogen of HCN miss at both spheres; the hydrogen misses at
    # two but matches the hydrogens of formaldehyde at one ("H-1;C(")
    assert np.isnan(got[:2]).all()
    assert got[2] == pytest.approx(0.05)
    assert list(exact.matched_radius) == [0, 0, 1]

    padded = _predictor(max_radius=2, fallback="global")
    padded.fit(train, train, rng=np.random.default_rng(0))
    mean = float(np.mean(train.atom_target))
    assert padded.predict(test).atom_value == pytest.approx([mean, mean, 0.05])


def test_the_generator_writes_nmrshiftdbs_codes():
    """Ethanol through the Java generator: a heavy atom is coded with its
    hydrogens implicit, a hydrogen with them explicit, and the code with ring
    sizes keeps the hydrogens of every atom."""
    from experiments.predictors.nmrshiftdb_codes import (
        nmrshiftdb_codes,
        nmrshiftdb_home,
    )

    try:
        nmrshiftdb_home()
    except FileNotFoundError:
        pytest.skip("NMRShiftDB generator not built (experiments/nmrshiftdb/setup.sh)")
    mol = _mol("CCO", [0.0] * 9)
    rows = nmrshiftdb_codes([mol], query_spheres=3)
    by_symbol = {}
    for atom, row in zip(mol.GetAtoms(), rows, strict=True):
        by_symbol.setdefault(atom.GetSymbol(), []).append(row)
    assert by_symbol["O"] == [["O-2;C(C//)//", "O-2;HC(HHC/HHH/)//", "O-2;C(C//)"]]
    hydroxyl = [r for r in by_symbol["H"] if r[0].startswith("H-1;O(")]
    assert hydroxyl == [["H-1;O(C/HHC/HHH)//", "H-1;O(C/HHC/HHH)//", "H-1;O(C/HHC/)"]]
