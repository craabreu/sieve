"""Fast-suite tests for prepare_store.py's pure-logic pieces -- no download,
no real 8.3GB SDF needed. The real end-to-end parse/cluster/split path is
covered by test_charge_prepare_store_optional.py, gated on that file's
presence."""

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

# The real SDF turns out to hold a second record schema for its non-ChEMBL-
# sourced molecules (QMugs/prior-training-set/organic-liquids, per the DASH
# paper's own stated four-source composition) -- DASH_IDX instead of
# CHEMBL_ID/CONF_ID, MBIScharge still present. Property order/spacing mirrors
# a real record sampled directly from the file (record #518670).
_TINY_SDF_DASH_ID = """\

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
>  <DASH_IDX>  (1)
Rest_2

>  <MBIScharge>  (1)
-0.3034|0.3806|-0.3975|0.3204

$$$$
"""


def test_parse_dash_molecules_writes_one_row_per_record(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"
    assert df.loc[0, "conf_id"] == "conf_00"
    assert df.loc[0, "dash_id"] is None
    assert df.loc[0, "net_charge"] == pytest.approx(0.0)  # M CHG: +1 and -1 cancel

    from charge_experiments.data import blob_to_mol

    mol = blob_to_mol(df.loc[0, "mol"])
    assert mol.GetNumAtoms() == 4
    charges = [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()]
    assert charges == pytest.approx([-0.3034, 0.3806, -0.3975, 0.3204])


def test_parse_dash_molecules_handles_multiple_records(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    sdf_path = tmp_path / "tiny.sdf"
    two_records = _TINY_SDF + _TINY_SDF.replace("CHEMBL185198", "CHEMBL999999").replace(
        "conf_00", "conf_01"
    )
    sdf_path.write_text(two_records)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 2
    assert set(df["chembl_id"]) == {"CHEMBL185198", "CHEMBL999999"}


def test_parse_dash_molecules_handles_dash_id_only_record(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF_DASH_ID)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.loc[0, "chembl_id"] is None
    assert df.loc[0, "dash_id"] == "Rest_2"
    assert df.loc[0, "conf_id"] == "conf_0"  # synthesized, no CONF_ID in this schema
    assert df.loc[0, "net_charge"] == pytest.approx(0.0)


def test_parse_dash_molecules_synthesizes_sequential_conf_ids_per_dash_id(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    # Three records sharing one DASH_IDX -- mirrors the real file's own
    # pattern of ~3 conformer rows per unique DASH_IDX.
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF_DASH_ID * 3)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 3
    assert list(df["dash_id"]) == ["Rest_2", "Rest_2", "Rest_2"]
    assert list(df["conf_id"]) == ["conf_0", "conf_1", "conf_2"]


def test_parse_dash_molecules_skips_record_with_neither_identity(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    no_identity = _TINY_SDF_DASH_ID.replace(">  <DASH_IDX>  (1)\nRest_2\n\n", "")
    assert "DASH_IDX" not in no_identity  # sanity-check the replace actually fired

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(no_identity + _TINY_SDF)  # one bad record, one good one
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1  # only the good (CHEMBL_ID) record survives
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"


def test_assign_splits_never_splits_a_dash_id_across_splits(tmp_path):
    from charge_experiments.prepare_store import assign_splits, parse_dash_molecules

    # Three distinct DASH_IDX groups, each with 2 conformer rows -- the
    # dash_id-only counterpart of test_assign_splits_never_splits_a_
    # chembl_id_across_splits below, proving the coalesced mol_key groups
    # dash_id rows correctly too, with no chembl_id present anywhere.
    records = []
    for i in range(3):
        records.append(_TINY_SDF_DASH_ID.replace("Rest_2", f"Rest_{i}"))
        records.append(_TINY_SDF_DASH_ID.replace("Rest_2", f"Rest_{i}"))
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3)

    import pandas as pd

    df = pd.read_parquet(molecules_path)
    assert "split" in df.columns
    assert df["chembl_id"].isna().all()
    per_id = df.groupby("dash_id")["split"].nunique()
    assert (per_id == 1).all()


def test_assign_splits_handles_a_mixed_store_of_both_schemas(tmp_path):
    from charge_experiments.prepare_store import assign_splits, parse_dash_molecules

    records = []
    for i in range(3):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _TINY_SDF.replace("CHEMBL185198", f"CHEMBL{i}").replace("conf_00", conf)
            )
    for i in range(3):
        records.append(_TINY_SDF_DASH_ID.replace("Rest_2", f"Rest_{i}"))
        records.append(_TINY_SDF_DASH_ID.replace("Rest_2", f"Rest_{i}"))
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3)

    import pandas as pd

    df = pd.read_parquet(molecules_path)
    assert len(df) == 12  # 6 chembl_id rows + 6 dash_id rows
    assert df["split"].notna().all()
    mol_key = df["chembl_id"].fillna(df["dash_id"])
    per_key = df.groupby(mol_key)["split"].nunique()
    assert (per_key == 1).all()


def test_assign_splits_never_splits_a_chembl_id_across_splits(tmp_path):
    from charge_experiments.prepare_store import assign_splits, parse_dash_molecules

    # Three distinct chembl_ids, each with 2 conformers of the same tiny
    # molecule text (connectivity-identical, so Butina clustering behavior
    # doesn't matter here -- what's under test is that the split join
    # groups by chembl_id, not by row).
    records = []
    for i in range(3):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _TINY_SDF.replace("CHEMBL185198", f"CHEMBL{i}").replace("conf_00", conf)
            )
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3)

    import pandas as pd

    df = pd.read_parquet(molecules_path)
    assert "split" in df.columns
    per_id = df.groupby("chembl_id")["split"].nunique()
    assert (per_id == 1).all()


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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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

    from charge_experiments.prepare_store import subsample_store

    sig = inspect.signature(subsample_store)
    assert sig.parameters["n_molecules"].default == 50_000
    assert sig.parameters["conformers_per_molecule"].default == 1
    assert sig.parameters["n_stores"].default == 1


def test_subsample_store_n_stores_writes_suffixed_stores_that_are_disjoint(tmp_path):
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

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
    from charge_experiments.prepare_store import subsample_store

    _synthetic_split_store(tmp_path, n_train=30, n_val=10, n_test=10)

    with pytest.raises(ValueError, match="n_stores"):
        subsample_store("source-store", "dest-store", stores_root=tmp_path, n_stores=0)


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
    from charge_experiments.prepare_store import _to_united_atom

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
    from charge_experiments.prepare_store import _to_united_atom

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
    from charge_experiments.prepare_store import _to_united_atom

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
    from charge_experiments.prepare_store import _to_united_atom
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
    from charge_experiments.prepare_store import _to_united_atom

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
    from charge_experiments.prepare_store import (
        parse_dash_molecules,
        to_united_atom_store,
    )

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF)
    store_dir = tmp_path / "source-store"
    store_dir.mkdir()
    parse_dash_molecules(sdf_path, store_dir / "molecules.parquet")

    to_united_atom_store("source-store", "ua-store", stores_root=tmp_path)

    import pandas as pd
    from charge_experiments.data import blob_to_mol

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


def _chiral_sdf_text(chembl_id="CHEMBL_CHIRAL", conf_id="conf_00"):
    """A real, valid SDF record with a genuine tetrahedral stereocenter
    (CIP R, verified independently), built via rdkit itself rather than
    hand-crafted coordinates -- MolToMolBlock writes fully-specified parity
    bits (rdkit wedges the bond from the mol's own ChiralTag, generating 2D
    coords as needed), so re-parsing this record does NOT hit
    _assign_stereo_if_needed's own "unassigned center" branch, exactly the
    case the unconditional AssignStereochemistry/AssignCIPLabels call in
    _parse_one_record exists to still cover correctly."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
    charges = [0.1 * (i + 1) for i in range(mol.GetNumAtoms())]
    molblock = Chem.MolToMolBlock(mol, kekulize=True)
    charge_str = "|".join(str(c) for c in charges)
    return (
        f"{molblock}"
        f">  <CHEMBL_ID>  (1)\n{chembl_id}\n\n"
        f">  <CONF_ID>  (1)\n{conf_id}\n\n"
        f">  <MBIScharge>  (1)\n{charge_str}\n\n"
        f"$$$$\n"
    )


def test_parse_dash_molecules_sets_rigorous_cip_labels(tmp_path):
    """Regression test for the fix's own stated bug: a record whose stereo
    is already fully specified by the molblock's own parity bits (not the
    "unassigned, needs 3D perception" case) must still get a real _CIPCode
    -- confirmed here by checking FindMolChiralCenters reports no "?" (so
    _assign_stereo_if_needed's own conditional branch does not fire) while
    _CIPCode still ends up set. Reads the mol back from the written
    parquet, not the in-memory object, so this also exercises the
    mol_to_blob/blob_to_mol round trip."""
    from charge_experiments.data import blob_to_mol
    from charge_experiments.prepare_store import parse_dash_molecules
    from rdkit import Chem

    sdf_path = tmp_path / "chiral.sdf"
    sdf_path.write_text(_chiral_sdf_text())
    out_path = tmp_path / "molecules.parquet"
    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    mol = blob_to_mol(df.loc[0, "mol"])

    # Checked *before* any further rdkit call: FindMolChiralCenters (below)
    # would itself compute and set _CIPCode as a side effect if it were
    # missing, silently masking whether it actually survived mol_to_blob's
    # own round trip -- verified directly (a mol with a real chiral tag but
    # no _CIPCode gets one from FindMolChiralCenters alone). Checking here,
    # first, is what makes this a real regression test for that persistence.
    stereocenter = next(a for a in mol.GetAtoms() if a.HasProp("_CIPCode"))
    assert stereocenter.GetPropsAsDict()["_CIPCode"] == "R"

    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    assert not any(tag == "?" for _, tag in centers)  # sanity: parity bits alone


def test_parse_dash_molecules_sets_cip_labeled_marker(tmp_path):
    from charge_experiments.data import blob_to_mol
    from charge_experiments.prepare_store import parse_dash_molecules

    from sieve.io.rdkit_adapter import CIP_LABELED_PROP

    sdf_path = tmp_path / "chiral.sdf"
    sdf_path.write_text(_chiral_sdf_text())
    out_path = tmp_path / "molecules.parquet"
    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    mol = blob_to_mol(df.loc[0, "mol"])
    assert mol.HasProp(CIP_LABELED_PROP)
