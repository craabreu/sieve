"""Fast-suite tests for prepare_dash.py's pure-logic pieces -- no download,
no real 8.3GB SDF needed. The real end-to-end parse/cluster/split path is
covered by test_prepare_dash_optional.py, gated on that file's
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

# The real file's ChEMBL-schema records carry a DASH_IDX *as well as* their
# CHEMBL_ID/CONF_ID -- confirmed by a full-file tag scan: all 1,029,785
# records have a DASH_IDX (518,669 QMUGS500_*, 511,116 Rest_*), and the
# 348,935 distinct values match the corpus's unique-molecule count exactly.
# QMugs is itself derived from ChEMBL, which is why those molecules carry
# both identities. Property order/spacing mirrors a real record sampled
# from the head of the file (record #1).
_TINY_SDF_BOTH_IDS = _TINY_SDF.replace(
    ">  <MBIScharge>  (1)",
    ">  <DASH_IDX>  (1)\nQMUGS500_1\n\n>  <MBIScharge>  (1)",
)


def test_parse_dash_molecules_writes_one_row_per_record(tmp_path):
    from experiments.prepare_dash import parse_dash_molecules

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

    from experiments.data import blob_to_mol

    mol = blob_to_mol(df.loc[0, "mol"])
    assert mol.GetNumAtoms() == 4
    charges = [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()]
    assert charges == pytest.approx([-0.3034, 0.3806, -0.3975, 0.3204])


def test_parse_dash_molecules_handles_multiple_records(tmp_path):
    from experiments.prepare_dash import parse_dash_molecules

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
    from experiments.prepare_dash import parse_dash_molecules

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
    from experiments.prepare_dash import parse_dash_molecules

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
    from experiments.prepare_dash import parse_dash_molecules

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
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

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
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

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


def test_parse_dash_molecules_keeps_both_ids_when_a_record_carries_both(tmp_path):
    from experiments.prepare_dash import parse_dash_molecules

    # DASH_IDX is not the ChEMBL schema's *alternative* identity, it is a
    # universal one: a record carrying both must land with both columns
    # populated, not with dash_id dropped because chembl_id was found first.
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF_BOTH_IDS)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"
    assert df.loc[0, "dash_id"] == "QMUGS500_1"
    # The record's own CONF_ID wins; nothing is synthesized for it.
    assert df.loc[0, "conf_id"] == "conf_00"


def test_parse_dash_molecules_does_not_consume_a_conf_counter_for_a_real_conf_id(
    tmp_path,
):
    from experiments.prepare_dash import parse_dash_molecules

    # Two conformers of one both-ids molecule, then a Rest_* record sharing
    # no group with them: the synthesized counters must be keyed per group,
    # so the Rest_* row still starts at conf_0 rather than continuing some
    # counter the CONF_ID-carrying rows advanced.
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(
        _TINY_SDF_BOTH_IDS
        + _TINY_SDF_BOTH_IDS.replace("conf_00", "conf_01")
        + _TINY_SDF_DASH_ID
    )
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert list(df["conf_id"]) == ["conf_00", "conf_01", "conf_0"]
    assert list(df["dash_id"]) == ["QMUGS500_1", "QMUGS500_1", "Rest_2"]


def test_assign_splits_groups_both_ids_rows_by_their_shared_dash_id(tmp_path):
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

    # Every row carries both identities, and the two keys agree 1:1 (as they
    # do in the real corpus: 176,969 unique CHEMBL_IDs against 176,969
    # unique QMUGS500_* ids), so grouping by either must keep a molecule's
    # conformers together.
    records = []
    for i in range(3):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _TINY_SDF_BOTH_IDS.replace("CHEMBL185198", f"CHEMBL{i}")
                .replace("QMUGS500_1", f"QMUGS500_{i}")
                .replace("conf_00", conf)
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
    assert df["dash_id"].notna().all()
    assert df["chembl_id"].notna().all()
    assert (df.groupby("dash_id")["split"].nunique() == 1).all()
    assert (df.groupby("chembl_id")["split"].nunique() == 1).all()


def test_assign_splits_still_groups_a_legacy_store_without_dash_ids(tmp_path):
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

    # Stores built before this fix have dash_id NULL on every ChEMBL row.
    # The split key falls back to chembl_id, so such a store still splits --
    # no rebuild is required for an existing store to stay usable.
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
    assert df["dash_id"].isna().all()
    assert df["split"].notna().all()
    assert (df.groupby("chembl_id")["split"].nunique() == 1).all()


def test_assign_splits_never_splits_a_chembl_id_across_splits(tmp_path):
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

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
    from experiments.data import blob_to_mol
    from experiments.prepare_dash import parse_dash_molecules
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
    from experiments.data import blob_to_mol
    from experiments.prepare_dash import parse_dash_molecules

    from sieve.io.rdkit_adapter import CIP_LABELED_PROP

    sdf_path = tmp_path / "chiral.sdf"
    sdf_path.write_text(_chiral_sdf_text())
    out_path = tmp_path / "molecules.parquet"
    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    mol = blob_to_mol(df.loc[0, "mol"])
    assert mol.HasProp(CIP_LABELED_PROP)
