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
                _TINY_SDF.replace("CHEMBL185198", f"CHEMBL{i}").replace(
                    "conf_00", conf
                )
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
