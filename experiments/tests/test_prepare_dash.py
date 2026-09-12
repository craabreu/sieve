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

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3, n_shards=1)

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

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3, n_shards=1)

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

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3, n_shards=1)

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

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3, n_shards=1)

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

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3, n_shards=1)

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


# --- conformer curation (DASH's own 0.4 e criterion, applied at parse time) ---
#
# The DASH paper filters conformers whose same-atom MBIS charge differs by
# more than 0.4 e across a molecule's conformers. Their published code
# implements only the per-element bound check (tree_constructor._check_charges,
# which drops nothing on the distributed corpus); this criterion exists only
# in the methods text, and 2,292 molecules of the released SDF violate it.


def _store_with_charges(tmp_path, per_conformer_charges, dash_id="Rest_1"):
    """A minimal molecules.parquet: one molecule, N conformers, 4 atoms each."""
    import pandas as pd
    from experiments.data import mol_to_blob
    from rdkit import Chem

    rows = []
    for i, charges in enumerate(per_conformer_charges):
        mol = Chem.AddHs(Chem.MolFromSmiles("C=O"))  # C, O, H, H
        assert mol.GetNumAtoms() == len(charges), (mol.GetNumAtoms(), len(charges))
        for atom, q in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("MBIScharge", float(q))
        rows.append(
            {
                "chembl_id": None,
                "conf_id": f"conf_{i}",
                "dash_id": dash_id,
                "mol": mol_to_blob(mol),
                "net_charge": 0.0,
            }
        )
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    pd.DataFrame(rows).to_parquet(store_dir / "molecules.parquet")
    return store_dir


def _kept_conf_ids(store_dir):
    import pandas as pd

    return sorted(pd.read_parquet(store_dir / "molecules.parquet")["conf_id"])


def test_curate_conformers_keeps_a_mutually_consistent_molecule(tmp_path):
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    store = _store_with_charges(
        tmp_path, [base, [q + 0.01 for q in base], [q - 0.01 for q in base]]
    )
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1", "conf_2"]


def test_curate_conformers_drops_only_the_conformer_that_disagrees_with_all(tmp_path):
    """A failed MBIS partition disagrees with every sibling, so it is
    identified without a tie-break; its siblings are kept."""
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    outlier = [3.00, 0.20, -0.30, 0.00]
    store = _store_with_charges(tmp_path, [base, [q + 0.01 for q in base], outlier])
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1"]


def test_curate_conformers_keeps_a_smooth_continuum(tmp_path):
    """A-B and A-C agree while B-C does not: the charge varies smoothly with
    geometry and no conformer is isolated, so all three are kept. Removing
    one would be arbitrary -- B and C are symmetric."""
    from experiments.prepare_dash import curate_conformers

    a = [0.00, 0.20, -0.30, 0.00]
    b = [-0.30, 0.20, -0.30, 0.00]
    c = [0.30, 0.20, -0.30, 0.00]  # |b - c| = 0.6 > 0.4, but both agree with a
    store = _store_with_charges(tmp_path, [a, b, c])
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1", "conf_2"]


def test_curate_conformers_drops_a_molecule_whose_conformers_all_disagree(tmp_path):
    from experiments.prepare_dash import curate_conformers

    store = _store_with_charges(
        tmp_path,
        [
            [0.00, 0.20, -0.30, 0.00],
            [1.00, 0.20, -0.30, 0.00],
            [2.00, 0.20, -0.30, 0.00],
        ],
    )
    curate_conformers(store)
    assert _kept_conf_ids(store) == []


def test_curate_conformers_drops_both_of_a_failing_pair(tmp_path):
    from experiments.prepare_dash import curate_conformers

    store = _store_with_charges(
        tmp_path, [[0.00, 0.20, -0.30, 0.00], [1.00, 0.20, -0.30, 0.00]]
    )
    curate_conformers(store)
    assert _kept_conf_ids(store) == []


def test_curate_conformers_drops_a_single_conformer_molecule(tmp_path):
    """A singleton has no pair to corroborate it. This never occurs in the
    real corpus (minimum 2 conformers per molecule) but the rule cannot
    admit an uncorroborated record, so the behaviour is pinned here."""
    from experiments.prepare_dash import curate_conformers

    store = _store_with_charges(tmp_path, [[0.10, 0.20, -0.30, 0.00]])
    curate_conformers(store)
    assert _kept_conf_ids(store) == []


def test_curate_conformers_groups_on_dash_id_not_chembl_id(tmp_path):
    """Half the corpus has no CHEMBL_ID; grouping on it would leave those
    molecules uncorroborated and delete them."""
    import pandas as pd
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    store = _store_with_charges(
        tmp_path, [base, [q + 0.01 for q in base]], dash_id="Rest_7"
    )
    df = pd.read_parquet(store / "molecules.parquet")
    assert df["chembl_id"].isna().all()
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1"]


def test_curate_conformers_is_idempotent(tmp_path):
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    outlier = [3.00, 0.20, -0.30, 0.00]
    store = _store_with_charges(tmp_path, [base, [q + 0.01 for q in base], outlier])
    first = curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1"]
    second = curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1"]
    assert "already curated" in second.lower() or second == first


def test_curate_conformers_writes_a_summary(tmp_path):
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    outlier = [3.00, 0.20, -0.30, 0.00]
    store = _store_with_charges(tmp_path, [base, [q + 0.01 for q in base], outlier])
    curate_conformers(store)
    summary = (store / "curation_summary.txt").read_text()
    assert "0.4" in summary
    assert "1" in summary  # one conformer removed


def test_prepare_store_curates_before_splitting(tmp_path):
    """The split must be computed on the curated population, so that no fold
    can inherit a record the criterion rejects."""
    import pandas as pd
    from experiments.prepare_dash import prepare_store

    sdf = tmp_path / "tiny.sdf"
    good = _TINY_SDF_DASH_ID
    outlier = _TINY_SDF_DASH_ID.replace(
        "-0.3034|0.3806|-0.3975|0.3204", "2.9000|0.3806|-0.3975|0.3204"
    )
    sdf.write_text(good + good.replace("Rest_2", "Rest_2") + outlier)
    prepare_store("s", stores_root=tmp_path / "stores", sdf_path=sdf, n_shards=1)

    df = pd.read_parquet(tmp_path / "stores" / "s" / "molecules.parquet")
    assert len(df) == 2, df[["dash_id", "conf_id"]]
    assert "split" in df.columns
    assert set(df["conf_id"]) == {"conf_0", "conf_1"}
    assert (tmp_path / "stores" / "s" / "curation_summary.txt").exists()


def test_prepare_store_refuses_a_store_split_before_curation_existed(tmp_path):
    """A store split from the uncurated population would end up with a split
    that predates its own contents; refuse rather than silently produce it."""
    import pytest
    from experiments.prepare_dash import prepare_store

    sdf = tmp_path / "tiny.sdf"
    sdf.write_text(_TINY_SDF_DASH_ID + _TINY_SDF_DASH_ID)
    stores = tmp_path / "stores"
    prepare_store("s", stores_root=stores, sdf_path=sdf, n_shards=1)
    (stores / "s" / "curation_summary.txt").unlink()  # simulate a legacy store

    with pytest.raises(RuntimeError, match="split before conformer curation"):
        prepare_store("s", stores_root=stores, sdf_path=sdf, n_shards=1)


def test_prepare_store_is_idempotent_once_curated_and_split(tmp_path):
    import pandas as pd
    from experiments.prepare_dash import prepare_store

    sdf = tmp_path / "tiny.sdf"
    sdf.write_text(_TINY_SDF_DASH_ID + _TINY_SDF_DASH_ID)
    stores = tmp_path / "stores"
    prepare_store("s", stores_root=stores, sdf_path=sdf, n_shards=1)
    before = pd.read_parquet(stores / "s" / "molecules.parquet")
    summary_before = (stores / "s" / "curation_summary.txt").read_text()
    prepare_store("s", stores_root=stores, sdf_path=sdf, n_shards=1)
    after = pd.read_parquet(stores / "s" / "molecules.parquet")
    assert len(before) == len(after)
    assert (stores / "s" / "curation_summary.txt").read_text() == summary_before


_DIVERSE_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "C1CCCCC1",  # cyclohexane
    "CC(C)=O",  # acetone
    "CN",  # methylamine
    "c1ccoc1",  # furan
    "CC#N",  # acetonitrile
    "C1CCNCC1",  # piperidine
    "CC(=O)N",  # acetamide
    "c1ccsc1",  # thiophene
    "CCN(CC)CC",  # triethylamine
    "OCC1CCCCC1",  # cyclohexylmethanol
    "c1ccc2ccccc2c1",  # naphthalene
    "CC(C)CC(C)C",  # 2,4-dimethylpentane
    "COC",  # dimethyl ether
    "CSC",  # dimethyl sulfide
    "FC(F)F",  # fluoroform
    "ClC(Cl)Cl",  # chloroform
    "BrCC",  # bromoethane
    "c1cc[nH]c1",  # pyrrole
    "C1CCOC1",  # tetrahydrofuran
    "CCCCCC",  # hexane
    "CC(C)(C)O",  # tert-butanol
    "c1ccc(cc1)O",  # phenol
    "c1ccc(cc1)N",  # aniline
    "CC(=O)OC",  # methyl acetate
    "NC(=O)N",  # urea
    "CCOC(=O)C",  # ethyl acetate
]


def _sdf_record_from_smiles(smiles: str, *, chembl_id: str, conf_id: str = "conf_00"):
    """One SDF record built from a SMILES string, in the same shape
    ``_chiral_sdf_text`` already uses: a real rdkit molblock plus the tag
    blocks ``_parse_one_record`` reads. Charges are a placeholder (index-
    scaled), not real MBIS output -- irrelevant here, since these tests are
    about clustering/splitting, not charge values.

    Carries a ``DASH_IDX`` alongside ``CHEMBL_ID``/``CONF_ID`` (derived from
    ``chembl_id`` one-to-one), matching the real corpus's own ChEMBL-derived
    half (``_parse_one_record``'s docstring: DASH_IDX is universal, present
    on every record) -- without it, ``curate_conformers``'s ``groupby
    ("dash_id")`` silently drops every row (pandas excludes an all-NaN
    group key), which a synthetic store with no DASH_IDX at all would hit
    on every single row."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    charges = [0.01 * (i + 1) for i in range(mol.GetNumAtoms())]
    molblock = Chem.MolToMolBlock(mol, kekulize=True)
    charge_str = "|".join(str(c) for c in charges)
    return (
        f"{molblock}"
        f">  <CHEMBL_ID>  (1)\n{chembl_id}\n\n"
        f">  <CONF_ID>  (1)\n{conf_id}\n\n"
        f">  <DASH_IDX>  (1)\nQMUGS500_{chembl_id}\n\n"
        f">  <MBIScharge>  (1)\n{charge_str}\n\n"
        f"$$$$\n"
    )


def _diverse_store(tmp_path, *, n_shards: int, train=0.9, val=0.0, test=0.1):
    """A store of ``_DIVERSE_SMILES``' worth of connectivity-distinct
    molecules (one conformer each), parsed and split with the given
    ``n_shards``. Returns the store dir."""
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

    records = [
        _sdf_record_from_smiles(smiles, chembl_id=f"CHEMBL{i}")
        for i, smiles in enumerate(_DIVERSE_SMILES)
    ]
    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)
    assign_splits(store_dir, train=train, val=val, test=test, n_shards=n_shards)
    return store_dir


def test_assign_splits_writes_cluster_and_shard_columns(tmp_path):
    import pandas as pd

    store_dir = _diverse_store(tmp_path, n_shards=5)

    df = pd.read_parquet(store_dir / "molecules.parquet")
    assert "cluster" in df.columns
    assert "shard" in df.columns
    assert df["cluster"].notna().all()
    assert df["shard"].notna().all()

    train_rows = df[df["split"] == "train"]
    test_rows = df[df["split"] == "test"]
    assert set(test_rows["shard"]) <= {"test"}
    assert set(train_rows["shard"]) <= {f"s{i:02d}" for i in range(5)}
    # Every requested shard actually got used -- assign_splits itself
    # raises if any came up empty, so a passing call already proves this,
    # but check it directly too.
    assert set(train_rows["shard"]) == {f"s{i:02d}" for i in range(5)}


def test_assign_splits_keeps_a_molecules_conformers_in_one_shard(tmp_path):
    """Multiple conformers of the same molecule must land in the same
    shard, the shard-level analogue of the split-level guarantee already
    covered by test_assign_splits_never_splits_a_chembl_id_across_splits."""
    import pandas as pd
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

    records = []
    for i, smiles in enumerate(_DIVERSE_SMILES):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _sdf_record_from_smiles(smiles, chembl_id=f"CHEMBL{i}", conf_id=conf)
            )
    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)
    assign_splits(store_dir, train=0.9, val=0.0, test=0.1, n_shards=5)

    df = pd.read_parquet(molecules_path)
    per_id_shards = df.groupby("chembl_id")["shard"].nunique()
    assert (per_id_shards == 1).all()


def test_assign_splits_rejects_n_shards_that_would_leave_one_empty(tmp_path):
    import pytest
    from experiments.prepare_dash import assign_splits, parse_dash_molecules

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(
        "".join(
            _sdf_record_from_smiles(smiles, chembl_id=f"CHEMBL{i}")
            for i, smiles in enumerate(_DIVERSE_SMILES)
        )
    )
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)

    with pytest.raises(ValueError, match="shard"):
        # 30 molecules, ~90% in train (~27) can't fill 1000 non-empty shards.
        assign_splits(store_dir, train=0.9, val=0.0, test=0.1, n_shards=1000)


def test_cluster_size_report_runs_against_a_parsed_store(tmp_path):
    from experiments.prepare_dash import cluster_size_report, parse_dash_molecules

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(
        "".join(
            _sdf_record_from_smiles(smiles, chembl_id=f"CHEMBL{i}")
            for i, smiles in enumerate(_DIVERSE_SMILES)
        )
    )
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    parse_dash_molecules(sdf_path, store_dir / "molecules.parquet")

    report = cluster_size_report(
        store_dir, train=0.9, test=0.1, candidate_n_shards=(1, 5, 1000)
    )
    assert "train molecules:" in report
    assert "clusters in train:" in report
    assert "N=1000" in report
    assert "empty_shards=" in report


def _two_conformer_records() -> str:
    """Two identical-charge conformers per molecule -- curate_conformers
    removes any single-conformer molecule outright (its own docstring), so
    a prepare_store test needs a pair per molecule to survive to the split
    stage at all."""
    records = []
    for i, smiles in enumerate(_DIVERSE_SMILES):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _sdf_record_from_smiles(smiles, chembl_id=f"CHEMBL{i}", conf_id=conf)
            )
    return "".join(records)


def test_prepare_store_defaults_to_a_90_10_split_with_no_val(tmp_path):
    import pandas as pd
    from experiments.prepare_dash import prepare_store

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(_two_conformer_records())
    stores = tmp_path / "stores"

    prepare_store("s", stores_root=stores, sdf_path=sdf_path, n_shards=5)

    df = pd.read_parquet(stores / "s" / "molecules.parquet")
    assert set(df["split"]) <= {"train", "test"}
    assert "cluster" in df.columns
    assert "shard" in df.columns


def test_prepare_store_refuses_a_pre_cv_redesign_split_lacking_cluster_and_shard(
    tmp_path,
):
    """A store split by the pre-redesign assign_splits (split only, no
    cluster/shard) cannot simply be re-split in place: Butina is not
    guaranteed bit-exact across two separate clustering runs, so
    back-filling cluster ids would not provably reproduce the existing
    split. Refuse loudly instead of silently reinterpreting it."""
    import pandas as pd
    import pytest
    from experiments.prepare_dash import prepare_store

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(_two_conformer_records())
    stores = tmp_path / "stores"
    prepare_store("s", stores_root=stores, sdf_path=sdf_path, n_shards=5)

    # Simulate a pre-redesign store: split only, cluster/shard dropped.
    molecules_path = stores / "s" / "molecules.parquet"
    df = pd.read_parquet(molecules_path)
    df.drop(columns=["cluster", "shard"]).to_parquet(molecules_path)

    with pytest.raises(RuntimeError, match=r"cluster.*shard"):
        prepare_store("s", stores_root=stores, sdf_path=sdf_path, n_shards=5)
