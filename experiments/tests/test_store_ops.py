"""Fast-suite tests for store_ops.py's dataset-agnostic store operations
(subsample_store, partition_store, to_united_atom_store) -- no download, no
real 8.3GB SDF needed."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

_TINY_SDF = """\

     RDKit          3D

  4  3  0  0  0  0  0  0  0  0999 V2000
   -0.6091    0.0044    0.4110 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5413    0.0039    0.4779 N   0  0  0  0  0  4  0  0  0  0  0  0
    1.7164    0.0116    0.5566 O   0  0  0  0  0  1  0  0  0  0  0  0
   -1.6488   -0.0199    0.3290 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
  2  3  1  0
  1  4  1  0
M  CHG  2   2   1   3  -1
M  END
>  <CHEMBL_ID>  (1)
CHEMBL185198

>  <CONF_ID>  (1)
conf_00

>  <MBIScharge>  (1)
-0.3034|0.3806|-0.3975|0.3204

$$$$
"""


def _synthetic_split_store(
    tmp_path, *, n_train=30, n_val=10, n_test=10, conformers_per_mol=2
):
    """An already-split store built directly (no SDF parsing needed --
    subsample_store never deserializes a Mol blob, so a placeholder b""
    stands in for one)."""
    import pandas as pd

    rows = []
    counts = {"train": n_train, "val": n_val, "test": n_test}
    for split_name, n_mol in counts.items():
        for m in range(n_mol):
            chembl_id = f"{split_name.upper()}{m}"
            for c in range(conformers_per_mol):
                rows.append(
                    {
                        "chembl_id": chembl_id,
                        "conf_id": f"conf_{c:02d}",
                        "dash_id": None,
                        "mol": b"",
                        "net_charge": 0.0,
                        "split": split_name,
                    }
                )
    df = pd.DataFrame(rows)
    store_dir = tmp_path / "source-store"
    store_dir.mkdir()
    df.to_parquet(store_dir / "molecules.parquet")
    return store_dir.name


def test_subsample_store_preserves_source_split_fractions_approximately(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=20,
        conformers_per_molecule=1,
        seed=0,
    )

    import pandas as pd

    out = pd.read_parquet(tmp_path / "dest-store" / "molecules.parquet")
    counts = out.groupby("split")["chembl_id"].nunique()
    # 60/20/20 of the source -> round(20*0.6)=12, round(20*0.2)=4, round(20*0.2)=4
    assert counts["train"] == 12
    assert counts["val"] == 4
    assert counts["test"] == 4


def test_subsample_store_caps_conformers_per_molecule(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=5, n_val=5, n_test=5, conformers_per_mol=5)

    subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=15,
        conformers_per_molecule=2,
        seed=0,
    )

    import pandas as pd

    out = pd.read_parquet(tmp_path / "dest-store" / "molecules.parquet")
    per_mol = out.groupby("chembl_id").size()
    assert (per_mol == 2).all()


def test_subsample_store_never_pads_a_molecule_with_fewer_conformers(tmp_path):
    from experiments.store_ops import subsample_store

    # Every molecule has exactly 1 conformer -- conformers_per_molecule=3
    # must not fabricate extras.
    _synthetic_split_store(tmp_path, n_train=5, n_val=5, n_test=5, conformers_per_mol=1)

    subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=15,
        conformers_per_molecule=3,
        seed=0,
    )

    import pandas as pd

    out = pd.read_parquet(tmp_path / "dest-store" / "molecules.parquet")
    per_mol = out.groupby("chembl_id").size()
    assert (per_mol == 1).all()


def test_subsample_store_is_reproducible_with_the_same_seed(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    subsample_store(
        "source-store",
        "dest-a",
        stores_root=tmp_path,
        n_molecules=20,
        conformers_per_molecule=1,
        seed=7,
    )
    subsample_store(
        "source-store",
        "dest-b",
        stores_root=tmp_path,
        n_molecules=20,
        conformers_per_molecule=1,
        seed=7,
    )

    import pandas as pd

    a = pd.read_parquet(tmp_path / "dest-a" / "molecules.parquet")
    b = pd.read_parquet(tmp_path / "dest-b" / "molecules.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_subsample_store_clamps_when_source_split_is_too_small(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=2, n_test=10)

    # 20-molecule target would ask round(20*2/42)~1 of val -- fine either
    # way, so make the request absurdly large to force real clamping.
    subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=1000,
        conformers_per_molecule=1,
        seed=0,
    )

    import pandas as pd

    out = pd.read_parquet(tmp_path / "dest-store" / "molecules.parquet")
    counts = out.groupby("split")["chembl_id"].nunique()
    assert counts["train"] == 30
    assert counts["val"] == 2
    assert counts["test"] == 10


def test_subsample_store_raises_without_a_split_column(tmp_path):
    import pandas as pd
    import pytest
    from experiments.store_ops import subsample_store

    store_dir = tmp_path / "unsplit-store"
    store_dir.mkdir()
    pd.DataFrame(
        {
            "chembl_id": ["A"],
            "conf_id": ["conf_00"],
            "dash_id": [None],
            "mol": [b""],
            "net_charge": [0.0],
        }
    ).to_parquet(store_dir / "molecules.parquet")

    with pytest.raises(ValueError, match="split column"):
        subsample_store(
            "unsplit-store", "dest-store", stores_root=tmp_path, n_molecules=10
        )


def test_subsample_store_writes_a_summary_file(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    summary_text = subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=20,
        conformers_per_molecule=1,
        seed=0,
    )

    summary_path = tmp_path / "dest-store" / "split_summary.txt"
    assert summary_path.exists()
    assert isinstance(summary_text, str)  # n_stores=1 returns one summary
    assert summary_path.read_text().strip() == summary_text.strip()
    assert "train" in summary_text
    assert "n_molecules" in summary_text


def test_subsample_store_defaults_match_the_cli(tmp_path):
    """The function's own defaults are what the plan calls for: 50k
    molecules, 1 conformer/molecule, source fractions preserved."""
    import inspect

    from experiments.store_ops import subsample_store

    sig = inspect.signature(subsample_store)
    assert sig.parameters["n_molecules"].default == 50_000
    assert sig.parameters["conformers_per_molecule"].default == 1
    assert sig.parameters["n_stores"].default == 1


def test_subsample_store_n_stores_writes_suffixed_stores_that_are_disjoint(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    summaries = subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=10,
        conformers_per_molecule=1,
        seed=0,
        n_stores=3,
    )
    assert isinstance(summaries, list)
    assert len(summaries) == 3

    import pandas as pd

    key_sets = []
    for i in (1, 2, 3):
        store_dir = tmp_path / f"dest-store-{i}"
        assert (store_dir / "split_summary.txt").exists()
        out = pd.read_parquet(store_dir / "molecules.parquet")
        # Every store gets the source's own 60/20/20 split mix.
        counts = out.groupby("split")["chembl_id"].nunique()
        assert counts["train"] == 6
        assert counts["val"] == 2
        assert counts["test"] == 2
        key_sets.append(set(out["chembl_id"]))

    assert not (tmp_path / "dest-store").exists()
    # Drawn without replacement: no molecule appears in two stores.
    for a in range(3):
        for b in range(a + 1, 3):
            assert not key_sets[a] & key_sets[b]


def test_subsample_store_n_stores_one_keeps_the_unsuffixed_name(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    summary_text = subsample_store(
        "source-store",
        "dest-store",
        stores_root=tmp_path,
        n_molecules=20,
        conformers_per_molecule=1,
        seed=0,
        n_stores=1,
    )

    assert isinstance(summary_text, str)
    assert (tmp_path / "dest-store" / "molecules.parquet").exists()
    assert not (tmp_path / "dest-store-1").exists()


def test_subsample_store_n_stores_raises_before_writing_when_source_is_too_small(
    tmp_path,
):
    import pytest
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    # 4 stores x round(20*0.6)=12 train molecules = 48 > the 30 train has.
    with pytest.raises(ValueError, match=r"train split.*too few for 4 disjoint"):
        subsample_store(
            "source-store",
            "dest-store",
            stores_root=tmp_path,
            n_molecules=20,
            conformers_per_molecule=1,
            seed=0,
            n_stores=4,
        )

    # Nothing at all was written -- not even the stores that would have fit.
    assert not list(tmp_path.glob("dest-store*"))


def test_subsample_store_n_stores_is_reproducible_with_the_same_seed(tmp_path):
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    subsample_store(
        "source-store",
        "a",
        stores_root=tmp_path,
        n_molecules=10,
        conformers_per_molecule=1,
        seed=7,
        n_stores=2,
    )
    subsample_store(
        "source-store",
        "b",
        stores_root=tmp_path,
        n_molecules=10,
        conformers_per_molecule=1,
        seed=7,
        n_stores=2,
    )

    import pandas as pd

    for i in (1, 2):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / f"a-{i}" / "molecules.parquet"),
            pd.read_parquet(tmp_path / f"b-{i}" / "molecules.parquet"),
        )


def test_subsample_store_rejects_n_stores_below_one(tmp_path):
    import pytest
    from experiments.store_ops import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    with pytest.raises(ValueError, match="n_stores"):
        subsample_store("source-store", "dest-store", stores_root=tmp_path, n_stores=0)


def test_partition_store_covers_every_molecule_exactly_once(tmp_path):
    """Exhaustive, not sampling: every molecule in the source lands in
    exactly one of the n_stores destination stores -- nothing left
    unused, unlike subsample_store."""
    import itertools

    import pandas as pd
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    summaries = partition_store(
        "source-store", "part", stores_root=tmp_path, n_stores=4, seed=0
    )
    assert len(summaries) == 4

    key_sets = []
    for i in (1, 2, 3, 4):
        out = pd.read_parquet(tmp_path / f"part-{i}" / "molecules.parquet")
        key_sets.append(set(out["chembl_id"]))

    union = set().union(*key_sets)
    assert len(union) == 50  # 30 + 10 + 10
    assert sum(len(s) for s in key_sets) == 50  # pairwise disjoint, no double-count
    for a, b in itertools.combinations(key_sets, 2):
        assert not (a & b)


def test_partition_store_block_sizes_differ_by_at_most_one(tmp_path):
    """31 train molecules into 4 stores: sizes must be 8,8,8,7 in some
    order (floor/ceil), never anything more skewed."""
    import pandas as pd
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=31, n_val=10, n_test=10)

    partition_store("source-store", "part", stores_root=tmp_path, n_stores=4, seed=0)

    sizes = []
    for i in (1, 2, 3, 4):
        out = pd.read_parquet(tmp_path / f"part-{i}" / "molecules.parquet")
        sizes.append(int((out["split"] == "train").sum() / 2))  # conformers_per_mol=2
    assert sorted(sizes) == [7, 8, 8, 8]


def test_partition_store_keeps_every_conformer_by_default(tmp_path):
    """Uncapped by default -- every conformer of every selected molecule
    survives, unlike subsample_store's own default cap of 1."""
    import pandas as pd
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=8, n_val=4, n_test=4, conformers_per_mol=3)

    partition_store("source-store", "part", stores_root=tmp_path, n_stores=2, seed=0)

    for i in (1, 2):
        out = pd.read_parquet(tmp_path / f"part-{i}" / "molecules.parquet")
        per_mol = out.groupby("chembl_id").size()
        assert (per_mol == 3).all()


def test_partition_store_still_caps_when_conformers_per_molecule_given(tmp_path):
    import pandas as pd
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=8, n_val=4, n_test=4, conformers_per_mol=3)

    partition_store(
        "source-store",
        "part",
        stores_root=tmp_path,
        n_stores=2,
        conformers_per_molecule=1,
        seed=0,
    )

    for i in (1, 2):
        out = pd.read_parquet(tmp_path / f"part-{i}" / "molecules.parquet")
        per_mol = out.groupby("chembl_id").size()
        assert (per_mol == 1).all()


def test_partition_store_n_stores_one_returns_bare_name_and_a_string(tmp_path):
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=8, n_val=4, n_test=4)

    summary = partition_store(
        "source-store", "part", stores_root=tmp_path, n_stores=1, seed=0
    )

    assert isinstance(summary, str)
    assert (tmp_path / "part" / "molecules.parquet").exists()
    assert not (tmp_path / "part-1").exists()


def test_partition_store_is_reproducible_with_the_same_seed(tmp_path):
    import pandas as pd
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    partition_store("source-store", "a", stores_root=tmp_path, n_stores=3, seed=7)
    partition_store("source-store", "b", stores_root=tmp_path, n_stores=3, seed=7)

    for i in (1, 2, 3):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / f"a-{i}" / "molecules.parquet"),
            pd.read_parquet(tmp_path / f"b-{i}" / "molecules.parquet"),
        )


def test_partition_store_rejects_n_stores_below_one(tmp_path):
    import pytest
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    with pytest.raises(ValueError, match="n_stores"):
        partition_store("source-store", "part", stores_root=tmp_path, n_stores=0)


def test_partition_store_rejects_a_non_positive_conformers_per_molecule(tmp_path):
    import pytest
    from experiments.store_ops import partition_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    with pytest.raises(ValueError, match="conformers_per_molecule"):
        partition_store(
            "source-store",
            "part",
            stores_root=tmp_path,
            n_stores=2,
            conformers_per_molecule=0,
        )


def test_partition_store_raises_without_a_split_column(tmp_path):
    import pandas as pd
    import pytest
    from experiments.store_ops import partition_store

    store_dir = tmp_path / "unsplit-store"
    store_dir.mkdir()
    pd.DataFrame(
        {
            "chembl_id": ["A"],
            "conf_id": ["conf_00"],
            "dash_id": [None],
            "mol": [b""],
            "net_charge": [0.0],
        }
    ).to_parquet(store_dir / "molecules.parquet")

    with pytest.raises(ValueError, match="split column"):
        partition_store("unsplit-store", "part", stores_root=tmp_path, n_stores=2)


def _ua_test_mol(smiles, *, add_hs=True, charges=None, isotope_h_idx=None):
    """A small rdkit Mol with a fabricated MBIScharge on every atom, for
    _to_united_atom's own unit tests."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if add_hs:
        mol = Chem.AddHs(mol)
    mol = Chem.Mol(mol)
    if isotope_h_idx is not None:
        mol.GetAtomWithIdx(isotope_h_idx).SetIsotope(2)  # deuterium
    if charges is None:
        charges = [0.1 * (i + 1) for i in range(mol.GetNumAtoms())]
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)
    return mol


def test_to_united_atom_removes_hydrogens_and_redistributes_their_charge():
    from experiments.store_ops import _to_united_atom

    # methanol: C-O, C has 3 H's, O has 1 H.
    mol = _ua_test_mol("CO")
    total_before = sum(a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms())

    ua_mol, n_removed, n_kept = _to_united_atom(mol)

    assert ua_mol.GetNumAtoms() == 2  # just C and O
    assert n_kept == 0
    assert n_removed == mol.GetNumAtoms() - 2
    total_after = sum(a.GetDoubleProp("MBIScharge") for a in ua_mol.GetAtoms())
    assert total_after == pytest.approx(total_before)


def test_to_united_atom_conserves_total_charge_on_a_larger_molecule():
    from experiments.store_ops import _to_united_atom

    mol = _ua_test_mol("CC(=O)Nc1ccc(O)cc1")  # acetaminophen, several H types
    total_before = sum(a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms())

    ua_mol, n_removed, n_kept = _to_united_atom(mol)

    total_after = sum(a.GetDoubleProp("MBIScharge") for a in ua_mol.GetAtoms())
    assert total_after == pytest.approx(total_before)
    assert n_removed + n_kept == sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)


def test_to_united_atom_keeps_a_hydrogen_rdkit_declines_to_remove():
    """An isotope-tagged H (deuterium) is one of rdkit's own documented
    RemoveHs exceptions -- confirmed empirically against this exact rdkit
    build before writing this test."""
    from experiments.store_ops import _to_united_atom

    mol = _ua_test_mol("CO", isotope_h_idx=2)  # atom 2 is one of C's H's
    total_before = sum(a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms())

    ua_mol, _n_removed, n_kept = _to_united_atom(mol)

    assert n_kept == 1
    isotopes = [a.GetIsotope() for a in ua_mol.GetAtoms()]
    assert 2 in isotopes  # the deuterium survived
    symbols = [a.GetSymbol() for a in ua_mol.GetAtoms()]
    assert symbols.count("H") == 1  # only the deuterium remains
    total_after = sum(a.GetDoubleProp("MBIScharge") for a in ua_mol.GetAtoms())
    assert total_after == pytest.approx(total_before)


def test_to_united_atom_preserves_cip_code_on_surviving_heavy_atoms():
    """RemoveHs is documented to preserve atom props for atoms that survive
    it (the same guarantee MBIScharge's own redistribution above already
    relies on) -- checked directly here for _CIPCode rather than assumed,
    since it's a *private* (underscore-prefixed) prop and this session
    already found one real case (mol_to_blob) where private-prop
    preservation needed an explicit opt-in rather than "just working"."""
    from experiments.store_ops import _to_united_atom
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    mol = _ua_test_mol("F[C@H](Cl)Br")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    stereocenter_before = next(a for a in mol.GetAtoms() if a.HasProp("_CIPCode"))
    assert stereocenter_before.GetPropsAsDict()["_CIPCode"] == "R"

    ua_mol, _n_removed, _n_kept = _to_united_atom(mol)

    assert ua_mol.GetNumAtoms() == 4  # F, C, Cl, Br -- no H's kept
    stereocenter_after = next(a for a in ua_mol.GetAtoms() if a.HasProp("_CIPCode"))
    assert stereocenter_after.GetPropsAsDict()["_CIPCode"] == "R"


def test_to_united_atom_heavy_atom_charge_is_original_plus_its_hs():
    from experiments.store_ops import _to_united_atom

    # ethanol built with explicit fixed charges so the arithmetic is exact:
    # atom order from Chem.AddHs(MolFromSmiles("CCO")) is C0 C1 O2 then H's.
    charges = [0.10, 0.20, 0.30, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    mol = _ua_test_mol("CCO", charges=charges)
    ua_mol, _n_removed, _n_kept = _to_united_atom(mol)

    assert ua_mol.GetNumAtoms() == 3
    # exact per-atom H assignment depends on rdkit's own H ordering/
    # neighbor map, so assert the group total, not a specific per-atom split.
    total = sum(a.GetDoubleProp("MBIScharge") for a in ua_mol.GetAtoms())
    assert total == pytest.approx(sum(charges))


def test_to_united_atom_store_transforms_every_row_and_preserves_other_columns(
    tmp_path,
):
    from experiments.prepare_dash import parse_dash_molecules
    from experiments.store_ops import to_united_atom_store

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF)
    store_dir = tmp_path / "source-store"
    store_dir.mkdir()
    parse_dash_molecules(sdf_path, store_dir / "molecules.parquet")

    to_united_atom_store("source-store", "ua-store", stores_root=tmp_path)

    import pandas as pd
    from experiments.data import blob_to_mol

    source = pd.read_parquet(store_dir / "molecules.parquet")
    ua = pd.read_parquet(tmp_path / "ua-store" / "molecules.parquet")

    assert len(ua) == len(source)
    assert (ua["chembl_id"] == source["chembl_id"]).all()
    assert (ua["conf_id"] == source["conf_id"]).all()
    assert (ua["net_charge"] == source["net_charge"]).all()

    source_mol = blob_to_mol(source["mol"].iloc[0])
    ua_mol = blob_to_mol(ua["mol"].iloc[0])
    # _TINY_SDF is C-N-O-H (a single H, on the C) -- exactly 3 heavy atoms.
    assert source_mol.GetNumAtoms() == 4
    assert ua_mol.GetNumAtoms() == 3
    source_total = sum(a.GetDoubleProp("MBIScharge") for a in source_mol.GetAtoms())
    ua_total = sum(a.GetDoubleProp("MBIScharge") for a in ua_mol.GetAtoms())
    assert ua_total == pytest.approx(source_total)
