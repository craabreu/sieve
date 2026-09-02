import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import sieve
from sieve.batch import check_alignment
from sieve.config import SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit, from_smiles
from sieve.refine import refine

SMILES = ["CCO", "c1ccccc1", "CC(=O)N", "CCl"]


def cfg_for(smiles, attrs=(("element",), ("aromatic", "hybridization"))):
    flat = [a for g in attrs for a in g]
    codes, edges = build_codes([Chem.MolFromSmiles(s) for s in smiles], flat)
    return SieveConfig(
        target_dim=1,
        attribute_levels=attrs,
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=2,
    )


def test_batch_shapes_match_the_molecules():
    cfg = cfg_for(SMILES)
    b = from_smiles(SMILES, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    assert b.n_nodes == sum(m.GetNumAtoms() for m in mols)
    assert b.n_edges == 2 * sum(m.GetNumBonds() for m in mols)


def test_edges_are_symmetric():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    fwd = {(int(a), int(c)) for a, c in zip(b.edge_src, b.edge_dst, strict=True)}
    assert all((c, a) in fwd for a, c in fwd)


def test_graph_id_separates_molecules():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    assert len(np.unique(b.graph_id)) == len(SMILES)
    # no edge may cross a molecule boundary
    assert np.all(b.graph_id[b.edge_src] == b.graph_id[b.edge_dst])


def test_unseen_category_gets_the_reserved_unknown_code():
    cfg = cfg_for(["CCO"])  # codes learned from C, O, H only
    b = from_smiles(["CCl"], config=cfg)
    unknown = max(cfg.attribute_codes["element"].values()) + 1
    assert unknown in b.node_attrs[:, 0].tolist()


def test_period_and_group_attributes():
    """period/group are periodic-table position, not element identity: two
    different elements in the same column (O, S -- both group 16) must get
    the same group code, and lanthanides (no well-defined group) must fall
    back to the "none" bucket rather than a spurious column."""
    smis = ["CCO", "CS(=O)(=O)C", "[Nd]"]
    cfg = cfg_for(smis, attrs=(("element",), ("period", "group")))
    b = from_smiles(smis, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in smis]

    period_col = len(cfg.attribute_levels[0])  # element is level 0
    group_col = period_col + 1

    def global_idx(symbol):
        offset = 0
        for m in mols:
            for a in m.GetAtoms():
                if a.GetSymbol() == symbol:
                    return offset + a.GetIdx()
            offset += m.GetNumAtoms()
        raise AssertionError(f"no atom with symbol {symbol}")

    o_idx = global_idx("O")
    s_idx = global_idx("S")
    nd_idx = global_idx("Nd")

    assert b.node_attrs[o_idx, group_col] == b.node_attrs[s_idx, group_col]
    assert b.node_attrs[o_idx, period_col] != b.node_attrs[s_idx, period_col]
    none_code = cfg.attribute_codes["group"]["none"]
    assert b.node_attrs[nd_idx, group_col] == none_code


def test_electronegativity_attribute_buckets_by_rounded_pauling_scale():
    """electronegativity is the atom's Pauling electronegativity (from RDKit's
    bundled atomic_data table) rounded to the nearest _ELECTRONEGATIVITY_ROUNDING
    step: C (2.55) and S (2.58) land in the same 2.5 bucket, O (3.44) rounds up
    to 3.5, and atoms with no tabulated value (noble gases, the dummy atom) fall
    back to the shared "none" sentinel."""
    from sieve.io.rdkit_adapter import _ELECTRONEGATIVITY_ROUNDING

    assert _ELECTRONEGATIVITY_ROUNDING == 0.5

    smis = ["CCO", "CS(=O)(=O)C", "[He]", "[*]"]
    cfg = cfg_for(smis, attrs=(("element",), ("electronegativity",)))
    b = from_smiles(smis, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in smis]

    en_col = len(cfg.attribute_levels[0])
    value_of = {v: k for k, v in cfg.attribute_codes["electronegativity"].items()}

    def row(symbol):
        return _global_atom_idx(mols, lambda a: a.GetSymbol() == symbol)

    assert value_of[b.node_attrs[row("C"), en_col]] == "2.5"
    assert value_of[b.node_attrs[row("S"), en_col]] == "2.5"
    assert value_of[b.node_attrs[row("O"), en_col]] == "3.5"

    none_code = cfg.attribute_codes["electronegativity"]["none"]
    assert b.node_attrs[row("He"), en_col] == none_code
    assert b.node_attrs[row("*"), en_col] == none_code


def test_block_and_valence_electrons_attributes():
    """block is the periodic-table block (s/p/d/f) and valence_electrons is
    RDKit's outer-electron count; both are periodic-table position, so Fe sits
    in the d block with 8 outer electrons, O in the p block with 6, Na in the s
    block with 1, a lanthanide in the f block, and the dummy atom falls back to
    the shared "none" sentinel for both."""
    smis = ["[Na]O[Fe]", "[La]", "[*]"]
    cfg = cfg_for(smis, attrs=(("element",), ("block", "valence_electrons")))
    b = from_smiles(smis, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in smis]

    block_col = len(cfg.attribute_levels[0])
    ve_col = block_col + 1
    block_of = {v: k for k, v in cfg.attribute_codes["block"].items()}
    ve_of = {v: k for k, v in cfg.attribute_codes["valence_electrons"].items()}

    def row(symbol):
        return _global_atom_idx(mols, lambda a: a.GetSymbol() == symbol)

    assert block_of[b.node_attrs[row("Fe"), block_col]] == "d"
    assert ve_of[b.node_attrs[row("Fe"), ve_col]] == "8"
    assert block_of[b.node_attrs[row("O"), block_col]] == "p"
    assert ve_of[b.node_attrs[row("O"), ve_col]] == "6"
    assert block_of[b.node_attrs[row("Na"), block_col]] == "s"
    assert ve_of[b.node_attrs[row("Na"), ve_col]] == "1"
    assert block_of[b.node_attrs[row("La"), block_col]] == "f"

    assert b.node_attrs[row("*"), block_col] == cfg.attribute_codes["block"]["none"]
    assert (
        b.node_attrs[row("*"), ve_col]
        == cfg.attribute_codes["valence_electrons"]["none"]
    )


def _global_atom_idx(mols, predicate):
    """Row of the first atom (across the concatenated batch) satisfying
    ``predicate(atom)``."""
    offset = 0
    for m in mols:
        for a in m.GetAtoms():
            if predicate(a):
                return offset + a.GetIdx()
        offset += m.GetNumAtoms()
    raise AssertionError("no atom matched the predicate")


def test_min_ring_size_discovers_smallest_ring_per_atom():
    """min_ring_size is a molecule-level attribute (needs whole-molecule ring
    perception): each atom gets the size of the smallest SSSR ring it sits in,
    or "none" when acyclic."""
    smis = ["C1CC1", "c1ccccc1", "CCO"]
    cfg = cfg_for(smis, attrs=(("element",), ("min_ring_size",)))
    assert set(cfg.attribute_codes["min_ring_size"]) == {"3", "6", "none"}

    b = from_smiles(smis, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in smis]
    col = len(cfg.attribute_levels[0])
    codes = cfg.attribute_codes["min_ring_size"]

    cyclopropane_c = _global_atom_idx(mols, lambda a: a.IsInRingSize(3))
    benzene_c = _global_atom_idx(mols, lambda a: a.GetIsAromatic())
    ethanol_o = _global_atom_idx(mols, lambda a: a.GetSymbol() == "O")

    assert b.node_attrs[cyclopropane_c, col] == codes["3"]
    assert b.node_attrs[benzene_c, col] == codes["6"]
    assert b.node_attrs[ethanol_o, col] == codes["none"]


def test_min_ring_size_uses_smallest_ring_in_a_fused_system():
    """An atom shared between a 5- and a 6-membered ring (indane's ring
    fusion) reports 5, not 6 -- MinAtomRingSize, not just 'is in a ring'."""
    smi = "C1CCc2ccccc21"  # indane: fused cyclopentane + benzene
    cfg = cfg_for([smi], attrs=(("element",), ("min_ring_size",)))
    b = from_smiles([smi], config=cfg)
    mol = Chem.MolFromSmiles(smi)
    col = len(cfg.attribute_levels[0])
    codes = cfg.attribute_codes["min_ring_size"]

    fusion_atom = _global_atom_idx(
        [mol], lambda a: a.IsInRingSize(5) and a.IsInRingSize(6)
    )
    assert b.node_attrs[fusion_atom, col] == codes["5"]


def test_min_ring_size_respects_node_order():
    """The molecule-level provider returns values in RDKit atom-index order;
    a permuted node_order must still land each atom's value on its own row."""
    smi = "C1CCc2ccccc21"
    cfg = cfg_for([smi], attrs=(("element",), ("min_ring_size",)))
    mol = Chem.MolFromSmiles(smi)
    col = len(cfg.attribute_levels[0])
    codes = cfg.attribute_codes["min_ring_size"]
    rev = [list(range(mol.GetNumAtoms()))[::-1]]

    b = from_rdkit([mol], config=cfg, node_order=rev)
    # row r holds original atom index (n-1-r)
    n = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    for r in range(n):
        orig = n - 1 - r
        want = str(ri.MinAtomRingSize(orig)) if ri.NumAtomRings(orig) else "none"
        assert b.node_attrs[r, col] == codes[want]


def test_min_ring_size_unseen_size_falls_back_to_unknown_code():
    cfg = cfg_for(["c1ccccc1"], attrs=(("element",), ("min_ring_size",)))  # only "6"
    b = from_smiles(["C1CC1"], config=cfg)
    col = len(cfg.attribute_levels[0])
    unknown = max(cfg.attribute_codes["min_ring_size"].values()) + 1
    assert (b.node_attrs[:, col] == unknown).all()


@pytest.mark.parametrize("n_jobs", [None, 1, 2, 4])
def test_min_ring_size_n_jobs_is_deterministic(n_jobs):
    mols = _chiral_corpus()
    cfg = cfg_for(_CHIRAL_SMILES, attrs=(("element",), ("min_ring_size", "aromatic")))
    baseline = from_rdkit(_chiral_corpus(), config=cfg)
    got = from_rdkit(mols, config=cfg, n_jobs=n_jobs)
    np.testing.assert_array_equal(got.node_attrs, baseline.node_attrs)

    codes_baseline, _ = build_codes(_chiral_corpus(), ["min_ring_size"])
    codes_got, _ = build_codes(_chiral_corpus(), ["min_ring_size"], n_jobs=n_jobs)
    assert codes_got == codes_baseline


def test_unknown_atoms_fall_back_rather_than_colliding():
    cfg = cfg_for(["CCO"])
    train = from_smiles(["CCO"], y=np.zeros((3, 1)), config=cfg)
    m = sieve.fit(train, cfg)
    p = sieve.predict_detailed(m, from_smiles(["CCl"], config=cfg))
    assert (p.matched_level == -1).any()


def test_wl_labels_agree_with_morgan_atom_environments():
    """literature.md 4.6: WL identifiers on molecules ARE ECFP identifiers, so
    RDKit gives an independent check on the encoder."""
    smis = ["CCO", "CCC", "c1ccccc1C", "CC(=O)O"]
    cfg = cfg_for(
        smis, attrs=(("element", "degree", "formal_charge", "num_h", "aromatic"),)
    )
    b = from_smiles(smis, config=cfg)
    ours = refine(b, cfg)[len(cfg.attribute_levels)].labels  # WL round 1
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=1, includeChirality=False)
    theirs, off = [], 0
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateAtomToBits()
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        env = ao.GetAtomToBits()
        theirs += [env[i][-1] for i in range(mol.GetNumAtoms())]
        off += mol.GetNumAtoms()
    assert _same_partition(np.array(ours), np.array(theirs))


def test_alignment_guard_catches_a_shuffled_target_array():
    cfg = cfg_for(["CCO"])
    y = np.arange(3, dtype=float).reshape(-1, 1)
    b = from_smiles(["CCO"], y=y, config=cfg)
    assert b.elements is not None
    with pytest.raises(ValueError, match="element"):
        check_alignment(b, np.array([3]), b.elements[::-1].copy())


def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_y_from_atom_prop_reads_scalar_per_atom():
    from sieve.io.rdkit_adapter import from_rdkit

    mols = []
    for smi, charges in [("CO", [-0.2, 0.2]), ("CCO", [-0.1, 0.05, 0.05])]:
        mol = Chem.MolFromSmiles(smi)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("q", charge)
        mols.append(mol)

    cfg = cfg_for(["CO", "CCO"])
    b = from_rdkit(mols, config=cfg, y_from_atom_prop="q")

    assert b.y is not None
    assert b.y.shape == (5, 1)
    expected = [-0.2, 0.2, -0.1, 0.05, 0.05]
    assert b.y[:, 0].tolist() == pytest.approx(expected)


def test_y_from_atom_prop_respects_node_order():
    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    charges = [-0.2, 0.2]
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("q", charge)

    cfg = cfg_for(["CO"])
    reversed_order = [np.array([1, 0])]
    b = from_rdkit([mol], config=cfg, node_order=reversed_order, y_from_atom_prop="q")

    assert b.y is not None
    assert b.y[:, 0].tolist() == pytest.approx([0.2, -0.2])


def test_y_and_y_from_atom_prop_are_mutually_exclusive():
    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("q", 0.0)
    cfg = cfg_for(["CO"])

    with pytest.raises(ValueError, match="y_from_atom_prop"):
        from_rdkit([mol], y=np.zeros((2, 1)), config=cfg, y_from_atom_prop="q")


def test_chirality_is_a_molecule_level_attribute():
    """CIP labeling is whole-molecule precomputation (Chem.AssignStereochemistry
    / rdCIPLabeler), so chirality belongs in _MOL_ATTRS and runs once per
    molecule via the generic path -- not in _ATTRS with _ensure_cip_labels
    threaded separately through every call site."""
    from sieve.io.rdkit_adapter import _ATTRS, _MOL_ATTRS

    assert "chirality" in _MOL_ATTRS
    assert "chirality" not in _ATTRS


def test_chirality_and_min_ring_size_coexist_as_molecule_level_attrs():
    """Two molecule-level attributes in one config, each precomputed once per
    molecule, must not interfere -- and n_jobs stays byte-identical."""
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _chiral_corpus()
    cfg = cfg_for(_CHIRAL_SMILES, attrs=(("element",), ("chirality", "min_ring_size")))
    baseline = from_rdkit(_chiral_corpus(), config=cfg)
    par = from_rdkit(mols, config=cfg, n_jobs=4)
    np.testing.assert_array_equal(par.node_attrs, baseline.node_attrs)

    chir_col = len(cfg.attribute_levels[0])
    none_code = cfg.attribute_codes["chirality"]["none"]
    assert baseline.elements is not None
    assert (baseline.node_attrs[baseline.elements == 6, chir_col] != none_code).any()


def test_chirality_attribute_uses_canonical_cip_code_not_raw_tag():
    """Same physical stereoisomer, written with two different atom orders
    (so the raw ChiralTag differs between them -- verified directly with
    rdkit elsewhere), must get the SAME chirality attribute code, since
    _CIPCode (unlike the raw tag) is order-independent."""
    from sieve.io.rdkit_adapter import from_rdkit

    variant_a = Chem.MolFromSmiles("F[C@H](Cl)Br")
    variant_b = Chem.MolFromSmiles("Cl[C@@H](F)Br")  # same physical molecule

    cfg = cfg_for(["F[C@H](Cl)Br"], attrs=(("element", "chirality"),))
    ba = from_rdkit([variant_a], config=cfg)
    bb = from_rdkit([variant_b], config=cfg)

    assert ba.elements is not None
    assert bb.elements is not None
    center_a = next(i for i, e in enumerate(ba.elements) if e == 6)
    center_b = next(i for i, e in enumerate(bb.elements) if e == 6)
    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    assert (
        ba.node_attrs[center_a, chirality_col] == bb.node_attrs[center_b, chirality_col]
    )


def test_chirality_attribute_falls_back_when_not_rigorously_labeled():
    """A plain Chem.MolFromSmiles mol has a raw ChiralTag (from the @/@@
    marker) and even a legacy _CIPCode from default sanitization -- but not
    CIP_LABELED_PROP, so from_rdkit must still compute the rigorous label,
    not trust whatever legacy value happens to already be sitting there
    (see _ensure_cip_labels' own docstring for why presence of _CIPCode
    alone is not a safe gate)."""
    from sieve.io.rdkit_adapter import CIP_LABELED_PROP, from_rdkit

    mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
    assert not mol.HasProp(CIP_LABELED_PROP)  # sanity: not marked as prepped

    cfg = cfg_for(["F[C@H](Cl)Br"], attrs=(("element", "chirality"),))
    b = from_rdkit([mol], config=cfg)

    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    none_code = cfg.attribute_codes["chirality"]["none"]
    assert b.elements is not None
    center = next(i for i, e in enumerate(b.elements) if e == 6)
    assert b.node_attrs[center, chirality_col] != none_code
    assert mol.HasProp(CIP_LABELED_PROP)  # marked, so a second call is a no-op


def test_chirality_attribute_matches_between_prepped_and_unprepped_input():
    """The actual property that matters: a mol pre-labeled and marked the
    way charge_experiments' own prepare_store.py does, and the same mol
    left unprepped (triggering from_rdkit's fallback), must produce the
    same attribute code."""
    from rdkit.Chem import rdCIPLabeler

    from sieve.io.rdkit_adapter import CIP_LABELED_PROP, from_rdkit

    unprepped = Chem.MolFromSmiles("F[C@H](Cl)Br")
    prepped = Chem.MolFromSmiles("F[C@H](Cl)Br")
    Chem.AssignStereochemistry(prepped, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(prepped)
    prepped.SetBoolProp(CIP_LABELED_PROP, True)

    cfg = cfg_for(["F[C@H](Cl)Br"], attrs=(("element", "chirality"),))
    b_unprepped = from_rdkit([unprepped], config=cfg)
    b_prepped = from_rdkit([prepped], config=cfg)

    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    assert b_unprepped.elements is not None
    assert b_prepped.elements is not None
    c1 = next(i for i, e in enumerate(b_unprepped.elements) if e == 6)
    c2 = next(i for i, e in enumerate(b_prepped.elements) if e == 6)
    assert (
        b_unprepped.node_attrs[c1, chirality_col]
        == b_prepped.node_attrs[c2, chirality_col]
    )


def test_chirality_attribute_uses_rigorous_cip_not_legacy_sanitization_default():
    """Regression test for a real, confirmed discrepancy: Chem.MolFromSmiles'
    own default sanitization already sets a *legacy* _CIPCode, which can
    genuinely disagree with rdCIPLabeler's rigorous one for pseudo-asymmetric
    centers -- verified directly: this bridged-bicyclic SMILES gives legacy
    'S'/'S' for two specific atoms but rigorous 'r'/'s'. from_rdkit must
    reflect the rigorous value."""
    from rdkit.Chem import rdCIPLabeler

    from sieve.io.rdkit_adapter import from_rdkit

    smi = "[H]/N=C(/N)NC[C@@]12C[C@@H]3C[C@@H](C[C@H]1C3)C2"
    mol = Chem.MolFromSmiles(smi)

    # confirm legacy (from default sanitization) actually differs here,
    # so this test is exercising the real discrepancy, not a no-op
    legacy = {
        a.GetIdx(): a.GetPropsAsDict().get("_CIPCode")
        for a in mol.GetAtoms()
        if a.HasProp("_CIPCode")
    }
    rigorous_check = Chem.MolFromSmiles(smi)
    Chem.AssignStereochemistry(rigorous_check, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(rigorous_check)
    rigorous = {
        a.GetIdx(): a.GetPropsAsDict().get("_CIPCode")
        for a in rigorous_check.GetAtoms()
        if a.HasProp("_CIPCode")
    }
    assert legacy != rigorous  # sanity: this molecule exercises the bug

    cfg = cfg_for([smi], attrs=(("element", "chirality"),))
    b = from_rdkit([mol], config=cfg)
    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    code_to_value = {v: k for k, v in cfg.attribute_codes["chirality"].items()}
    got = {idx: code_to_value[b.node_attrs[idx, chirality_col]] for idx in rigorous}
    assert got == rigorous


def test_chirality_attribute_is_none_for_non_stereocenters():
    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CCO")  # no stereocenters at all
    cfg = cfg_for(["CCO"], attrs=(("element", "chirality"),))
    b = from_rdkit([mol], config=cfg)

    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    none_code = cfg.attribute_codes["chirality"]["none"]
    assert (b.node_attrs[:, chirality_col] == none_code).all()


def test_build_codes_discovers_cip_vocabulary_from_unprepped_mols():
    """build_codes runs before from_rdkit in the normal fit path
    (sieve_predictor.py's own _build_config) -- it must apply the same
    fallback, or an unprepped training corpus would silently discover only
    "none" as the whole chirality vocabulary."""
    mols = [Chem.MolFromSmiles("F[C@H](Cl)Br"), Chem.MolFromSmiles("CCO")]
    codes, _ = build_codes(mols, ["chirality"])
    assert set(codes["chirality"]) >= {"none", "R"} or set(codes["chirality"]) >= {
        "none",
        "S",
    }


# --- n_jobs: determinism and the specific silent-corruption hazards -------
#
# Everything below must produce output BYTE-IDENTICAL to n_jobs=None,
# because that is the whole safety argument for offering n_jobs at all: the
# chunking is an execution-strategy choice, never a change in meaning. Three
# things fail *silently* rather than raising if the chunking is wrong, which
# is exactly why each gets its own direct test rather than relying on the
# determinism tests to catch it as a side effect:
#   1. graph_id colliding across chunks (concat_batches' own job, but
#      exercised here through the real from_rdkit path)
#   2. naive pickling of live Mol objects instead of explicit
#      ToBinary(AllProps) -- chirality would quietly become "none"
#      everywhere rather than erroring
#   3. y/node_order sliced by the wrong index (molecule vs. cumulative atom
#      count)

_CHIRAL_SMILES = [
    "F[C@H](Cl)Br",
    "Cl[C@@H](F)Br",
    "CCO",
    "c1ccccc1C",
    "CC(=O)O",
    "CCN",
    "CCCl",
    "CO",
    "CCC",
    "F[C@@H](Cl)Br",
]


def _chiral_corpus(repeat=3):
    """A corpus mixing chiral and achiral molecules -- large enough, and
    varied enough in atom count, that a 4-chunk split lands mid-corpus."""
    return [Chem.MolFromSmiles(s) for s in _CHIRAL_SMILES * repeat]


@pytest.mark.parametrize("n_jobs", [None, 1, 2, 4])
def test_from_rdkit_n_jobs_is_deterministic(n_jobs):
    mols = _chiral_corpus()
    cfg = cfg_for(
        _CHIRAL_SMILES, attrs=(("element",), ("degree", "aromatic", "chirality"))
    )
    baseline = from_rdkit(_chiral_corpus(), config=cfg)
    got = from_rdkit(mols, config=cfg, n_jobs=n_jobs)

    np.testing.assert_array_equal(got.node_attrs, baseline.node_attrs)
    np.testing.assert_array_equal(got.edge_src, baseline.edge_src)
    np.testing.assert_array_equal(got.edge_dst, baseline.edge_dst)
    np.testing.assert_array_equal(got.edge_attr, baseline.edge_attr)
    np.testing.assert_array_equal(got.graph_id, baseline.graph_id)
    assert got.elements is not None and baseline.elements is not None
    np.testing.assert_array_equal(got.elements, baseline.elements)


@pytest.mark.parametrize("n_jobs", [None, 1, 2, 4])
def test_build_codes_n_jobs_is_deterministic(n_jobs):
    attrs = ["element", "degree", "aromatic", "chirality"]
    baseline, edge_baseline = build_codes(_chiral_corpus(), attrs)
    got, edge_got = build_codes(_chiral_corpus(), attrs, n_jobs=n_jobs)
    assert got == baseline
    assert edge_got == edge_baseline


def test_from_rdkit_n_jobs_does_not_silently_lose_chirality():
    """The specific hazard n_jobs introduces: naive pickling of a live Mol
    across a process boundary drops every atom property (RDKit's global
    default pickle setting is NoProps), and unlike MBIScharge-style targets
    (which fail loudly -- GetDoubleProp raises on a missing prop),
    chirality reads via a .get(..., "none") fallback and would silently
    become "none" for every atom instead of erroring. n_jobs > 1 must
    recover the same real R/S/r/s codes as sequential, not a corpus that
    quietly discovered only "none"."""
    mols = _chiral_corpus()
    cfg = cfg_for(_CHIRAL_SMILES, attrs=(("element", "chirality"),))
    chirality_col = list(cfg.attribute_levels[0]).index("chirality")
    none_code = cfg.attribute_codes["chirality"]["none"]

    b = from_rdkit(mols, config=cfg, n_jobs=4)
    assert b.elements is not None
    carbon_mask = b.elements == 6
    # F[C@H](Cl)Br's own chiral carbon must show a real code, not "none".
    assert (b.node_attrs[carbon_mask, chirality_col] != none_code).any()


def test_from_rdkit_n_jobs_respects_node_order_per_chunk():
    """node_order is indexed positionally per molecule -- a parallel chunk
    must slice it the same way it slices mols, not by a flat/global index."""
    from sieve.io.rdkit_adapter import from_rdkit

    smis = ["CCO", "CCC", "c1ccccc1C", "CC(=O)O", "CCN", "CCCl", "CCCC", "CO"] * 3
    mols = [Chem.MolFromSmiles(s) for s in smis]
    cfg = cfg_for(smis, attrs=(("element",),))
    rng = np.random.default_rng(0)
    node_order = [rng.permutation(m.GetNumAtoms()) for m in mols]

    seq = from_rdkit(mols, config=cfg, node_order=node_order, n_jobs=None)
    par = from_rdkit(mols, config=cfg, node_order=node_order, n_jobs=4)
    np.testing.assert_array_equal(seq.node_attrs, par.node_attrs)
    np.testing.assert_array_equal(seq.graph_id, par.graph_id)


def test_from_rdkit_n_jobs_slices_array_y_by_atom_count_not_molecule_count():
    """y, when passed as an array, is per-node -- a chunk boundary at
    molecule index k must slice y at the cumulative *atom* count up to k,
    not at k itself. Deliberately uses molecules of different sizes so a
    molecule-index slice would misalign."""
    from sieve.io.rdkit_adapter import from_rdkit

    smis = ["CO", "CCO", "CCCO", "CCCCO", "CO", "CCO", "CCCO", "CCCCO"] * 2
    mols = [Chem.MolFromSmiles(s) for s in smis]
    cfg = cfg_for(smis, attrs=(("element",),))
    n_atoms = sum(m.GetNumAtoms() for m in mols)
    y = np.arange(n_atoms, dtype=np.float64).reshape(-1, 1)  # unique per row

    seq = from_rdkit(mols, y=y, config=cfg, n_jobs=None)
    par = from_rdkit(mols, y=y, config=cfg, n_jobs=4)
    assert seq.y is not None and par.y is not None
    np.testing.assert_array_equal(seq.y, par.y)
    np.testing.assert_array_equal(seq.node_attrs, par.node_attrs)
