"""Fast-suite tests for prepare_dash.py's pure-logic pieces -- no download,
no real 8.3GB SDF needed. The real end-to-end parse/cluster/split path is
covered by test_prepare_dash_optional.py, gated on that file's
presence."""

from __future__ import annotations

from pathlib import Path

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


def _store_with_structures(tmp_path, groups, dash_id="Rest_1"):
    """A molecules.parquet whose rows share ONE ``dash_id`` but cover several
    structures -- the shape 3.57% of the real corpus has.

    ``groups`` is ``[(smiles, [charges, ...]), ...]``. Both SMILES used by
    the tests give 4 atoms after ``AddHs``, so cross-structure pairs are
    shape-compatible and really are compared by the old rule; if they were
    not, ``curate_conformers`` would skip the pair and the tests would pass
    for the wrong reason.
    """
    import pandas as pd
    from experiments.data import mol_to_blob
    from rdkit import Chem

    rows = []
    n = 0
    for smiles, per_conformer in groups:
        for charges in per_conformer:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            assert mol.GetNumAtoms() == len(charges), (smiles, mol.GetNumAtoms())
            for atom, q in zip(mol.GetAtoms(), charges, strict=True):
                atom.SetDoubleProp("MBIScharge", float(q))
            rows.append(
                {
                    "chembl_id": None,
                    "conf_id": f"conf_{n}",
                    "dash_id": dash_id,
                    "mol": mol_to_blob(mol),
                    "net_charge": 0.0,
                }
            )
            n += 1
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


def test_curate_conformers_does_not_read_swapped_symmetric_atoms_as_disagreement(
    tmp_path,
):
    """Formaldehyde's two hydrogens are one atom to every arm and have no
    correct pairing across conformers. Two conformers that differ only by
    which H carries which charge agree, however far apart those charges are;
    compared index by index they would be 0.5 e apart and both dropped."""
    from experiments.prepare_dash import curate_conformers

    store = _store_with_charges(
        tmp_path, [[0.1, -0.4, 0.0, 0.5], [0.1, -0.4, 0.5, 0.0]]
    )
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


def test_curate_conformers_keeps_a_single_conformer_structure(tmp_path):
    """Reversed deliberately. Under ``dash_id`` grouping a lone conformer
    never occurred, so removing it was free; under structure grouping 14,815
    structures have exactly one conformer of their own, and removing them
    would discard sound data for a reason unrelated to the MBIS convergence
    failure the criterion exists to catch. Nothing corroborates such a record
    and nothing contradicts it: it is unjudged, not failed."""
    from experiments.prepare_dash import curate_conformers

    store = _store_with_charges(tmp_path, [[0.10, 0.20, -0.30, 0.00]])
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0"]


def test_curate_conformers_keeps_a_solo_structure_inside_a_mixed_identifier(
    tmp_path,
):
    """The case the change exists for: one ``dash_id`` holding two
    structures, one of which was deposited with a single conformer whose
    charges agree with nothing else under that identifier.

    Under ``dash_id`` grouping it agreed with no sibling and was deleted.
    Under structure grouping it is alone in its own group and survives.
    """
    from experiments.prepare_dash import curate_conformers

    store = _store_with_structures(
        tmp_path,
        [
            ("C=O", [[2.00, 0.00, 0.00, 0.00]]),  # solo structure, far from the rest
            ("C=S", [[0.00, 0.00, 0.00, 0.00], [0.10, 0.00, 0.00, 0.00]]),
        ],
    )
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_0", "conf_1", "conf_2"]


def test_curate_conformers_does_not_corroborate_across_structures(tmp_path):
    """The other half of the key change. A structure whose own conformers all
    disagree must fail, even when a *different* structure under the same
    ``dash_id`` happens to agree with one of them.

    Old rule: conf_0 agrees with the C=S pair, so it survives on a
    cross-structure certificate. New rule: C=O is judged among its own
    conformers, which disagree by 1.0 e, so both go.
    """
    from experiments.prepare_dash import curate_conformers

    store = _store_with_structures(
        tmp_path,
        [
            ("C=O", [[0.00, 0.00, 0.00, 0.00], [1.00, 0.00, 0.00, 0.00]]),
            ("C=S", [[0.00, 0.00, 0.00, 0.00], [0.10, 0.00, 0.00, 0.00]]),
        ],
    )
    curate_conformers(store)
    assert _kept_conf_ids(store) == ["conf_2", "conf_3"]


def test_curate_conformers_summary_records_the_solo_structures(tmp_path):
    """The count belongs in the store's own summary, not only in a script:
    it is the number the manuscript's conditional invariant rests on."""
    from experiments.prepare_dash import curate_conformers

    store = _store_with_structures(
        tmp_path,
        [
            ("C=O", [[2.00, 0.00, 0.00, 0.00]]),
            ("C=S", [[0.00, 0.00, 0.00, 0.00], [0.10, 0.00, 0.00, 0.00]]),
        ],
    )
    summary = curate_conformers(store)
    assert "structures kept without a same-structure sibling: 1" in summary


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


def test_curate_conformers_skips_when_the_summary_matches_the_store(tmp_path):
    """The ordinary idempotent path: summary present and its recorded
    post-count matches the store, so curation is skipped."""
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    outlier = [3.00, 0.20, -0.30, 0.00]
    store = _store_with_charges(tmp_path, [base, [q + 0.01 for q in base], outlier])

    curate_conformers(store)
    second = curate_conformers(store)

    assert "already curated" in second.lower()


def test_curate_conformers_recurates_when_the_parquet_was_rebuilt_underneath(
    tmp_path,
):
    """The corruption path the fingerprint exists to close: the summary
    records a curation of a *previous* parquet, and the store has since
    been re-parsed (more rows, uncurated). Trusting the marker alone would
    leave an uncurated store claiming to be curated; the row-count check
    must notice and re-curate instead."""
    import pandas as pd
    from experiments.prepare_dash import curate_conformers

    base = [0.10, 0.20, -0.30, 0.00]
    outlier = [3.00, 0.20, -0.30, 0.00]
    store = _store_with_charges(tmp_path, [base, [q + 0.01 for q in base], outlier])

    curate_conformers(store)
    curated_rows = len(pd.read_parquet(store / "molecules.parquet"))
    assert curated_rows == 2  # the outlier was dropped
    summary_before = (store / "curation_summary.txt").read_text()

    # Simulate prepare_store re-parsing the SDF under a surviving summary.
    (tmp_path / "rebuilt").mkdir()
    rebuilt = _store_with_charges(
        tmp_path / "rebuilt", [base, [q + 0.01 for q in base], outlier]
    )
    (store / "molecules.parquet").write_bytes(
        (rebuilt / "molecules.parquet").read_bytes()
    )
    assert len(pd.read_parquet(store / "molecules.parquet")) == 3  # uncurated again

    result = curate_conformers(store)

    assert "already curated" not in result.lower()  # it re-ran, not skipped
    assert len(pd.read_parquet(store / "molecules.parquet")) == 2
    assert (store / "curation_summary.txt").read_text() == summary_before


def test_curated_conformer_count_parses_the_summary():
    """Both formats, deliberately. The count is the *fingerprint* the
    idempotency guard compares against the parquet's row count, so a summary
    it cannot parse makes the guard silently fall through to re-curating. The
    first text is the pre-structure-grouping wording, which real stores on
    disk still carry."""
    from experiments.prepare_dash import _curated_conformer_count

    older = (
        "conformer curation at threshold 0.4 e\n"
        "conformers: 1029785 -> 1027538 (2247 removed)\n"
        "molecules removed entirely: 86"
    )
    current = (
        "conformer curation at threshold 0.4 e\n"
        "conformers: 1029785 -> 1027538 (2247 removed)\n"
        "structures removed entirely: 86\n"
        "structures kept without a same-structure sibling: 14815"
    )
    assert _curated_conformer_count(older) == 1027538
    assert _curated_conformer_count(current) == 1027538
    assert _curated_conformer_count("no counts here") is None


def test_prepare_store_stop_before_split_leaves_the_store_unsplit(tmp_path):
    """--stop-before-split must produce exactly the state cluster-report
    reads: parsed and curated, with no split/cluster/shard columns."""
    import pandas as pd
    from experiments.prepare_dash import prepare_store

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(_two_conformer_records())
    stores = tmp_path / "stores"

    prepare_store("s", stores_root=stores, sdf_path=sdf_path, stop_before_split=True)

    df = pd.read_parquet(stores / "s" / "molecules.parquet")
    assert (stores / "s" / "curation_summary.txt").exists()
    assert "split" not in df.columns
    assert "cluster" not in df.columns
    assert "shard" not in df.columns
    assert not (stores / "s" / "split_summary.txt").exists()


def test_prepare_store_after_stop_before_split_can_still_split(tmp_path):
    """The two-phase path the workflow relies on: stop before the split to
    run cluster-report, then re-run without the flag to write the split --
    without re-parsing or re-curating."""
    import pandas as pd
    from experiments.prepare_dash import prepare_store

    sdf_path = tmp_path / "diverse.sdf"
    sdf_path.write_text(_two_conformer_records())
    stores = tmp_path / "stores"

    prepare_store("s", stores_root=stores, sdf_path=sdf_path, stop_before_split=True)
    curated = pd.read_parquet(stores / "s" / "molecules.parquet")
    summary_before = (stores / "s" / "curation_summary.txt").read_text()

    prepare_store("s", stores_root=stores, sdf_path=sdf_path, n_shards=5)

    df = pd.read_parquet(stores / "s" / "molecules.parquet")
    assert len(df) == len(curated)  # nothing re-parsed or re-curated
    assert (stores / "s" / "curation_summary.txt").read_text() == summary_before
    assert {"split", "cluster", "shard"} <= set(df.columns)
    assert set(df["split"]) <= {"train", "test"}


def _embedded(smiles):
    """A molecule with real 3D coordinates and stereo perceived from them."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def test_3d_perception_does_not_overwrite_a_declared_parity():
    """The bug this guards: gating on "any centre is unassigned" and then
    calling AssignStereochemistryFrom3D rewrites *every* centre, so one
    undeclared centre costs every declared one in the same molecule. On the
    real SDF that turned 64 undeclared centres into 160 perceived ones.

    The declared tag here is deliberately set against the geometry, so only
    preserving it -- not re-perceiving it -- can pass.
    """
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    tagged = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    mol = _embedded("C[C@H](O)[C@@H](N)C(=O)O")
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() in tagged]
    assert len(centers) == 2
    keep, clear = centers
    against_geometry = (
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        if mol.GetAtomWithIdx(keep).GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
        else Chem.ChiralType.CHI_TETRAHEDRAL_CW
    )
    mol.GetAtomWithIdx(keep).SetChiralTag(against_geometry)
    mol.GetAtomWithIdx(clear).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    _assign_stereo_if_needed(mol)

    assert mol.GetAtomWithIdx(keep).GetChiralTag() == against_geometry
    assert mol.GetAtomWithIdx(clear).GetChiralTag() in tagged


def test_an_unassigned_double_bond_is_perceived_from_3d():
    """A bond left STEREOANY used to survive untouched, because the gate only
    looked at tetrahedral centres. AssignStereochemistryFrom3D also declines to
    overrule an explicit STEREOANY, so clearing the mark first is what lets the
    coordinates speak.
    """
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    mol = _embedded("C/N=N/C")
    bond = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    declared = bond.GetStereo()
    bond.SetStereo(Chem.BondStereo.STEREOANY)

    _assign_stereo_if_needed(mol)

    got = next(
        b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    ).GetStereo()
    assert got not in {Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREONONE}
    assert got == declared


def test_two_records_of_one_structure_get_one_collapse_key():
    """The point of the bond half: collapse_key serialises what the Mol
    declares, so a STEREOANY in one record and a definite flag in another split
    a single structure into two molecules. On the test split that affected 143
    groups holding 865 conformers, 10.8% of the stereo-sensitive subset.
    """
    from experiments.collapse import collapse_key
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    specified, unspecified = _embedded("C/N=N/C"), _embedded("C/N=N/C")
    next(
        b for b in unspecified.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    ).SetStereo(Chem.BondStereo.STEREOANY)

    for mol in (specified, unspecified):
        _assign_stereo_if_needed(mol)

    assert collapse_key(specified) == collapse_key(unspecified)


def test_genuine_e_and_z_are_still_held_apart():
    """The complement of the test above: filling in unspecified geometry must
    not merge records that really do differ."""
    from experiments.collapse import collapse_key
    from experiments.prepare_dash import _assign_stereo_if_needed

    e_isomer, z_isomer = _embedded("C/N=N/C"), _embedded(r"C/N=N\C")
    for mol in (e_isomer, z_isomer):
        _assign_stereo_if_needed(mol)

    assert collapse_key(e_isomer) != collapse_key(z_isomer)


def test_a_fully_specified_record_is_left_alone():
    """Nothing to perceive means nothing is touched: deciding whether to act
    must not itself leave a mark."""
    from experiments.prepare_dash import _assign_stereo_if_needed

    mol = _embedded("F[C@H](Cl)Br")
    atoms = [(a.GetIdx(), a.GetChiralTag()) for a in mol.GetAtoms()]
    bonds = [
        (b.GetIdx(), b.GetStereo(), tuple(b.GetStereoAtoms())) for b in mol.GetBonds()
    ]

    _assign_stereo_if_needed(mol)

    assert [(a.GetIdx(), a.GetChiralTag()) for a in mol.GetAtoms()] == atoms
    assert [
        (b.GetIdx(), b.GetStereo(), tuple(b.GetStereoAtoms())) for b in mol.GetBonds()
    ] == bonds


# ---------------------------------------------------------------------------
# Stereo perception: the two defects found after the bond-stereo fix.
#
# Both use REAL store records, saved beside this file, and that is not
# incidental. Small symmetric molecules such as 1,4-dimethylcyclohexane keep
# their dependent stereocentres visible after ``AddHs``, so a toy input passes
# with or without the gate fix and tests nothing. ``Rest_95518`` (spiro/
# ring-fusion centres) and ``Rest_128717`` (a phosphorane) are the records the
# defects were measured on.
# ---------------------------------------------------------------------------

_DATA = Path(__file__).resolve().parent / "data"


def _fixture_mol(name):
    from rdkit import Chem

    mol = Chem.MolFromMolFile(str(_DATA / name), removeHs=False, sanitize=True)
    assert mol is not None, f"fixture {name} did not parse"
    return mol


def test_perception_gate_sees_dependent_stereocentres_under_explicit_hs():
    """The gate must consult coordinates for spiro/ring-fusion/bridgehead
    centres, whose ring branches are constitutionally identical.

    ``FindMolChiralCenters`` reports ``[]`` for these on an explicit-H
    molecule -- silently, an empty list rather than an error -- and the store
    parses with ``removeHs=False``, so asking on the explicit-H molecule left
    0.92% of potential centres unassigned corpus-wide. An unassigned centre
    makes ``collapse_key`` merge genuine diastereomers.
    """
    from experiments.prepare_dash import _assign_stereo_if_needed, _needs_perception
    from rdkit import Chem

    mol = _fixture_mol("dependent_stereocentres.mol")
    Chem.RemoveStereochemistry(mol)

    # The old gate, asked directly on the explicit-H molecule, sees nothing...
    blind = any(
        tag == "?"
        for _, tag in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    )
    assert not blind, "fixture no longer exercises the blind spot"

    # ...while the heavy-atom probe does, and perception then tags them.
    assert _needs_perception(mol)

    _assign_stereo_if_needed(mol)
    tagged = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetChiralTag()
        in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    ]
    assert len(tagged) == 2, f"expected both dependent centres tagged, got {tagged}"


def test_perception_gate_warns_when_it_cannot_strip_hydrogens(monkeypatch, caplog):
    """A record RDKit cannot strip must still be gated, loudly.

    The heavy-atom probe is what closes the dependent-stereocentre blind
    spot, so a record it cannot be built for is gated on the explicit-H
    molecule instead -- with the blind spot back. That degraded path is
    reachable on a malformed record and silently returns the same ``False``
    a clean molecule with nothing to perceive returns, so the warning is the
    only thing distinguishing them and is worth pinning.
    """
    import logging

    from rdkit import Chem

    from experiments import prepare_dash

    mol = _fixture_mol("dependent_stereocentres.mol")
    Chem.RemoveStereochemistry(mol)

    def _boom(_mol):
        raise ValueError("synthetic RemoveHs failure")

    monkeypatch.setattr(Chem, "RemoveHs", _boom)
    with caplog.at_level(logging.WARNING, logger=prepare_dash.logger.name):
        answer = prepare_dash._needs_perception(mol)

    # It falls back rather than propagating, and reports the same blindness
    # the un-patched gate is there to avoid -- which is why it must warn.
    assert answer is False
    assert "heavy-atom probe" in caplog.text


def test_non_tetrahedral_tags_are_cleared_so_conformers_share_one_key():
    """``AssignStereochemistryFrom3D`` gives pentavalent P and sulfonic S a
    ``CHI_TRIGONALBIPYRAMIDAL``/``CHI_SQUAREPLANAR`` tag whose permutation
    index varies between conformers of one molecule, so one structure got
    several ``collapse_key``s. ``mirror_mol`` inverts only the tetrahedral
    tags, so the enantiomer merge cannot rescue it either.
    """
    from experiments.collapse import collapse_key
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    keep = {
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    mol = _fixture_mol("phosphorane_tb_tag.mol")
    _assign_stereo_if_needed(mol)

    exotic = [
        (atom.GetIdx(), atom.GetSymbol(), str(atom.GetChiralTag()))
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() not in keep
    ]
    assert not exotic, f"non-tetrahedral tags survived: {exotic}"

    # A rotated copy is the same structure, so it must key the same. With the
    # TB tag present the permutation index moves and the keys diverge.
    from rdkit.Geometry import Point3D

    rotated = Chem.Mol(mol)
    conf = rotated.GetConformer()
    for i in range(rotated.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(-p.x, -p.y, p.z))  # a proper rotation
    _assign_stereo_if_needed(rotated)

    assert collapse_key(mol) == collapse_key(rotated)


def test_prepare_store_keeps_the_uncurated_parse_when_asked(tmp_path):
    """The uncurated parse exists only between parse and curation inside
    prepare_store, and it is the only state that can show whether the 0.4 e
    criterion deleted something it should not have -- the curated store, by
    construction, holds only survivors. Recovering it afterwards costs a full
    re-parse of the 8.3GB SDF."""
    import pandas as pd
    from experiments.prepare_dash import UNCURATED_PARQUET, prepare_store

    # One structure whose two conformers disagree, so curation really removes
    # rows and the two parquets differ.
    _store_with_charges(
        tmp_path,
        [[0.00, 0.00, 0.00, 0.00], [1.00, 0.00, 0.00, 0.00]],
    )
    sdf = tmp_path / "unused.sdf"  # parsing is skipped; the parquet is there
    sdf.write_text("")

    prepare_store(
        "store",
        stores_root=tmp_path,
        sdf_path=sdf,
        stop_before_split=True,
        keep_uncurated=True,
    )

    kept = tmp_path / "store" / UNCURATED_PARQUET
    assert kept.exists(), "the uncurated parse was not kept"
    assert len(pd.read_parquet(kept)) == 2
    assert len(pd.read_parquet(tmp_path / "store" / "molecules.parquet")) == 0


def test_prepare_store_refuses_to_mislabel_an_already_curated_store(tmp_path):
    """Copying a curated store to a file named 'uncurated' would be worse
    than having none: it reads as evidence about what curation deleted while
    holding only survivors."""
    from experiments.prepare_dash import UNCURATED_PARQUET, prepare_store

    _store_with_charges(
        tmp_path,
        [[0.00, 0.00, 0.00, 0.00], [0.10, 0.00, 0.00, 0.00]],
    )
    sdf = tmp_path / "unused.sdf"
    sdf.write_text("")
    common = {
        "stores_root": tmp_path,
        "sdf_path": sdf,
        "stop_before_split": True,
    }

    prepare_store("store", **common)  # curates, writes no copy
    assert not (tmp_path / "store" / UNCURATED_PARQUET).exists()

    prepare_store("store", keep_uncurated=True, **common)
    assert not (tmp_path / "store" / UNCURATED_PARQUET).exists()


def _embedded(smiles):
    """A 3D conformer, so AssignStereochemistryFrom3D has coordinates to read."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    return mol


def test_a_spurious_stereoany_mark_is_not_restored(tmp_path):
    """The molblock can flag a bond STEREOANY that is not stereogenic at all
    -- a terminal alkene such as OC=CH2, whose =CH2 end carries two
    hydrogens, so there is no E/Z to determine.

    The legacy FindPotentialStereoBonds gate used to clear such a flag as a
    side effect, and its rollback once put it straight back: 731 bonds in
    20,551 sampled store rows carried the flag for this reason, 3.34% of
    records. The coordinate probe that replaced that gate does not mutate, so
    the rule -- a flag the coordinates cannot settle is dropped -- is stated
    outright, and this pins it.
    """
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    mol = _embedded("C=CO")
    bond = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    bond.SetStereo(Chem.BondStereo.STEREOANY)

    _assign_stereo_if_needed(mol)

    assert mol.GetBondWithIdx(bond.GetIdx()).GetStereo() == (Chem.BondStereo.STEREONONE)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [("C/C=C/C", "STEREOE"), ("C/C=C\\C", "STEREOZ")],
)
def test_a_genuine_stereoany_bond_is_perceived_not_dropped(smiles, expected):
    """The other half, and the one that makes the fix safe to make. A bond
    that IS stereogenic is settled by the coordinate probe, so it reaches the
    perception path and is read off the coordinates. Dropping the flag must
    not become dropping the chemistry.
    """
    from experiments.prepare_dash import _assign_stereo_if_needed
    from rdkit import Chem

    mol = _embedded(smiles)
    bond = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert str(bond.GetStereo()) == expected, "embedding lost the configuration"
    bond.SetStereo(Chem.BondStereo.STEREOANY)  # as an unspecified molblock would

    _assign_stereo_if_needed(mol)

    assert str(mol.GetBondWithIdx(bond.GetIdx()).GetStereo()) == expected


def _store_from_mols(tmp_path, mols, dash_ids):
    """A molecules.parquet from explicit Mol objects, one row each."""
    import pandas as pd
    from experiments.data import mol_to_blob

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    pd.DataFrame(
        [
            {
                "chembl_id": None,
                "conf_id": f"conf_{i}",
                "dash_id": did,
                "mol": mol_to_blob(mol),
                "net_charge": 0.0,
            }
            for i, (mol, did) in enumerate(zip(mols, dash_ids, strict=True))
        ]
    ).to_parquet(store_dir / "molecules.parquet")
    return store_dir


def test_curate_conformers_groups_two_deposits_of_one_structure(tmp_path):
    """Structure grouping pools deposits, so two `dash_id`s holding the same
    molecule form ONE group and corroborate each other.

    The two records here carry identical chemistry with the atoms in a
    different order, which is the case `a.shape != b.shape` cannot see: 15 of
    400 sampled multi-deposit groups on the real store (~670 of 17,890) are
    like this. Compared by raw index the charges differ by 1.1 e and both
    records would be deleted as anomalous; aligned to the key's canonical
    order they are identical and both survive.
    """
    from experiments.prepare_dash import curate_conformers
    from rdkit import Chem

    first = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    # A spread well beyond the 0.4 e threshold, so mis-paired atoms cannot
    # agree by accident and the test really is differential.
    for atom in first.GetAtoms():
        atom.SetDoubleProp(
            "MBIScharge", {"C": -0.5, "O": 0.6, "H": 0.1}[atom.GetSymbol()]
        )

    # The same molecule as a second deposit would store it: heavy atoms last.
    order = sorted(
        range(first.GetNumAtoms()),
        key=lambda i: first.GetAtomWithIdx(i).GetSymbol() != "H",
    )
    second = Chem.RenumberAtoms(first, order)
    assert [a.GetSymbol() for a in first.GetAtoms()] != [
        a.GetSymbol() for a in second.GetAtoms()
    ], "the permutation did not change the atom ordering"

    store = _store_from_mols(tmp_path, [first, second], ["Rest_1", "Rest_2"])
    summary = curate_conformers(store)

    assert _kept_conf_ids(store) == ["conf_0", "conf_1"]
    # One group, so neither is the "deposited without a sibling" case.
    assert "structures kept without a same-structure sibling: 0" in summary


def test_achiral_fingerprints_match_the_deprecated_api():
    """The generator must reproduce AllChem.GetMorganFingerprintAsBitVect
    exactly, since the clustering built from these fingerprints defines the
    split every held-out number is measured against, and Butina is not
    bit-exact across runs -- a fingerprint that moved could not be reconciled
    after the fact.
    """
    import numpy as np
    from experiments.prepare_dash import _achiral_fingerprints
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    smiles = ["CCO", "c1ccccc1O", "C[C@H](N)C(=O)O", "C1CC2CCC1C2", "FC(F)(F)c1ccncc1"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]

    got = _achiral_fingerprints(mols)
    for i, mol in enumerate(mols):
        want = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=2048, useChirality=False
            ),
            want,
        )
        assert np.array_equal(got[i], want), f"fingerprint changed for {smiles[i]}"


# ---------------------------------------------------------------------------
# Stereo perception: pseudo-asymmetric double bonds.
#
# A C=N or C=C whose ring end carries two branches that differ only through
# stereocentres -- oximes and alkylidenes on tropanes, 9-azabicyclononanes and
# cis-2,6-disubstituted piperidines -- is stereogenic, and the rigorous CIP
# labeler gives it a lowercase e/z. The legacy FindPotentialStereoBonds gate
# never marked these, so their coordinates were never read: 225 store rows,
# 70 collapse groups. Nor can FindPotentialStereo stand in as the gate: it
# reports 108 of those 225 bonds as not stereogenic while they are unset, and
# as Specified once they are flagged. The fixtures are the real records the
# defect was measured on.
# ---------------------------------------------------------------------------


def _double_bonds(mol):
    from rdkit import Chem

    return [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
    ]


def _oxime_bond(mol):
    """The C=N bond of the record's oxime, the one bond these fixtures are about."""
    return next(
        i
        for i in _double_bonds(mol)
        if {
            mol.GetBondWithIdx(i).GetBeginAtom().GetSymbol(),
            mol.GetBondWithIdx(i).GetEndAtom().GetSymbol(),
        }
        == {"C", "N"}
    )


def _read_bonds(mol):
    """Bond indices the stereo featurisation yields a row for."""
    from sieve.io.rdkit_adapter import _stereo_bond_rows

    return {
        mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b, *_ in _stereo_bond_rows(mol)
    }


def test_a_bond_findpotentialstereo_cannot_see_while_unset_is_perceived():
    """``Rest_109172``: the oxime on a 9-azabicyclo[3.3.1]nonane.

    The premise is pinned first, because it is why the gate may not be
    ``FindPotentialStereo``: with the bond unset -- the state the store held
    it in -- that call does not report it at all, so a gate asking it would
    never read the coordinates.
    """
    from experiments.prepare_dash import _finalise_stereo
    from rdkit import Chem

    mol = _fixture_mol("oxime_bicyclic_pseudo_ez.mol")
    bond = _oxime_bond(mol)
    unset = Chem.Mol(mol)
    unset.GetBondWithIdx(bond).SetStereo(Chem.BondStereo.STEREONONE)
    reported = {
        int(e.centeredOn)
        for e in Chem.FindPotentialStereo(unset)
        if e.type == Chem.StereoType.Bond_Double
    }
    assert bond not in reported, "premise: FindPotentialStereo is blind to it unset"

    _finalise_stereo(mol)

    assert mol.GetBondWithIdx(bond).GetStereo() in {
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    }
    assert bond in _read_bonds(mol)


def test_conformers_of_opposite_geometry_are_held_apart():
    """``Rest_137421`` conformers 1 and 2 carry opposite oxime geometry -- the
    corpus never fixed it -- so they are two diastereomers, and the store had
    averaged their charges into one group."""
    from experiments.collapse import collapse_key
    from experiments.prepare_dash import _finalise_stereo

    a, b = _fixture_mol("oxime_conformer_a.mol"), _fixture_mol("oxime_conformer_b.mol")
    for mol in (a, b):
        _finalise_stereo(mol)

    assert collapse_key(a) != collapse_key(b)


def test_one_structure_from_two_sources_gets_one_key():
    """``QMUGS500_57675`` leaves the oxime unset; ``Rest_109895``, the same
    molecule from the other source, declares it. Perceiving the first must
    land it in the second's group rather than a group of its own."""
    from experiments.collapse import collapse_key
    from experiments.prepare_dash import _finalise_stereo

    undeclared = _fixture_mol("oxime_undeclared_source.mol")
    declared = _fixture_mol("oxime_declared_source.mol")
    for mol in (undeclared, declared):
        _finalise_stereo(mol)

    assert collapse_key(undeclared) == collapse_key(declared)


_STEREO_FIXTURES = [
    "dependent_stereocentres.mol",
    "phosphorane_tb_tag.mol",
    "oxime_bicyclic_pseudo_ez.mol",
    "oxime_conformer_a.mol",
    "oxime_conformer_b.mol",
    "oxime_undeclared_source.mol",
    "oxime_declared_source.mol",
]


@pytest.mark.parametrize("name", _STEREO_FIXTURES)
def test_the_store_form_is_its_own_fixed_point(name):
    """Finalising a stored ``Mol`` must reproduce it byte for byte.

    This is what lets the store be patched instead of rebuilt: re-running the
    finalisation over every stored row changes exactly the rows a fix
    touches, so the rows that differ *are* the fix. It fails if any step
    rewrites what an earlier pass wrote -- ``AssignStereochemistry`` turning
    the stored ``STEREOCIS``/``STEREOTRANS`` into CIP-derived
    ``STEREOE``/``STEREOZ`` and nothing turning them back, for one.
    """
    from experiments.data import blob_to_mol, mol_to_blob
    from experiments.prepare_dash import _finalise_stereo

    mol = _fixture_mol(name)
    _finalise_stereo(mol)
    once = mol_to_blob(mol)

    again = blob_to_mol(once)
    _finalise_stereo(again)
    assert mol_to_blob(again) == once


@pytest.mark.parametrize("name", _STEREO_FIXTURES)
def test_the_store_carries_only_the_local_bond_form(name):
    """``STEREOE``/``STEREOZ`` rank substituents arbitrarily far from the
    bond; ``STEREOCIS``/``STEREOTRANS`` name two neighbours. A corpus for a
    method built on locality stores the second."""
    from experiments.prepare_dash import _finalise_stereo
    from rdkit import Chem

    mol = _fixture_mol(name)
    _finalise_stereo(mol)

    assert {mol.GetBondWithIdx(i).GetStereo() for i in _double_bonds(mol)} <= {
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
    }


@pytest.mark.parametrize("name", _STEREO_FIXTURES)
def test_every_bond_the_coordinates_settle_reaches_the_featurisation(name):
    """The contract between the store and its reader.

    Whatever the record's coordinates can settle -- asked of a copy stripped
    of every bond flag, so the answer does not depend on what the molblock
    happened to declare -- the stored ``Mol`` must declare, and the stereo
    featurisation must read. The defect this closes lived in the gap between
    two definitions of "a stereo double bond", one in each module, and only a
    test spanning both can see such a gap.
    """
    from experiments.prepare_dash import _finalise_stereo
    from rdkit import Chem

    mol = _fixture_mol(name)
    _finalise_stereo(mol)

    probe = Chem.Mol(mol)
    for bond in probe.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.AssignStereochemistryFrom3D(probe)
    settled = {
        i
        for i in _double_bonds(probe)
        if probe.GetBondWithIdx(i).GetStereo()
        not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}
    }

    assert settled <= _read_bonds(mol)
