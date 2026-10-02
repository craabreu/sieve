"""Fast-suite tests for the DASH subset stores: parsing into the staging store, the
shared split, the pair diagnostics and the curation -- no real SDF needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

_MOLBLOCK = """\

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
"""

_QMUGS_TAGS = """\
>  <CHEMBL_ID>  (1)
CHEMBL185198

>  <CONF_ID>  (1)
conf_00

>  <DFT:ESP_AT_NUCLEI>  (1)
-14.7|-18.3|-22.4|-1.1

>  <DFT:LOWDIN_CHARGES>  (1)
-0.1|0.2|-0.2|0.1

>  <DFT:MULLIKEN_CHARGES>  (1)
-0.2|0.3|-0.3|0.2

>  <GFN2:MULLIKEN_CHARGES>  (1)
-0.15|0.25|-0.25|0.15

>  <DFT:TOTAL_ENERGY>  (1)
-168.1234

>  <GFN2:TOTAL_ENERGY>  (1)
-12.3456

>  <DFT:DIPOLE>  (1)
0.1|0.2|0.3|0.37

>  <DASH_IDX>  (1)
QMUGS500_1

>  <MBIScharge>  (1)
-0.3034|0.3806|-0.3975|0.3204

$$$$
"""

_EXTRA_TAGS = """\
>  <MBIS_Energy>  (1)
-168.5

>  <XTB_Energy>  (1)
-12.4

>  <XTB_MulikenCharge>  (1)
-0.14|0.24|-0.24|0.14

>  <DASH_IDX>  (1)
Rest_2

>  <MBIScharge>  (1)
-0.3034|0.3806|-0.3975|0.3204

$$$$
"""


def test_subset_of_reads_the_prefix_and_refuses_others():
    from experiments.dash_subsets import subset_of

    assert subset_of("QMUGS500_7") == "QMugs"
    assert subset_of("Rest_7") == "Extra"
    with pytest.raises(ValueError):
        subset_of("Other_7")


def test_parse_staging_writes_aligned_rows_and_fields(tmp_path):
    from experiments.dash_subsets import parse_staging

    sdf = tmp_path / "x.sdf"
    sdf.write_text(_MOLBLOCK + _QMUGS_TAGS + _MOLBLOCK + _EXTRA_TAGS)
    n = parse_staging(sdf, tmp_path / "parsed.parquet", tmp_path / "fields.parquet")
    parsed = pd.read_parquet(tmp_path / "parsed.parquet")
    fields = pd.read_parquet(tmp_path / "fields.parquet")
    assert n == len(parsed) == len(fields) == 2
    assert parsed["subset"].tolist() == ["QMugs", "Extra"]
    assert parsed["collapse_key"].iat[0] == parsed["collapse_key"].iat[1]
    assert np.allclose(fields["esp"].iat[0], [-14.7, -18.3, -22.4, -1.1])
    assert fields["esp"].iat[1] is None
    assert np.allclose(fields["xtb"].iat[1], [-0.14, 0.24, -0.24, 0.14])
    assert fields["e_dft"].iat[0] == pytest.approx(-168.1234)
    assert np.isnan(fields["e_dft"].iat[1])
    assert fields["e_tpssh"].iat[1] == pytest.approx(-168.5)
    assert np.allclose(fields["dipole_dft"].iat[0], [0.1, 0.2, 0.3])


def test_parse_staging_skips_a_record_whose_field_length_is_wrong(tmp_path):
    from experiments.dash_subsets import parse_staging

    bad = _QMUGS_TAGS.replace("-14.7|-18.3|-22.4|-1.1", "-14.7|-18.3")
    sdf = tmp_path / "x.sdf"
    sdf.write_text(_MOLBLOCK + bad + _MOLBLOCK + _EXTRA_TAGS)
    assert parse_staging(sdf, tmp_path / "p.parquet", tmp_path / "f.parquet") == 1


def test_stratified_split_keeps_clusters_whole_and_balances_each_stratum():
    from experiments.dash_subsets import stratified_cluster_split

    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 400, size=4000)
    strata = np.where(rng.random(4000) < 0.5, "QMugs", "Extra")
    parts = stratified_cluster_split(clusters, strata, {"train": 0.9, "test": 0.1})
    train, test = set(clusters[parts["train"]]), set(clusters[parts["test"]])
    assert not train & test
    for s in ("QMugs", "Extra"):
        mine = strata == s
        share = np.isin(np.flatnonzero(mine), parts["test"]).mean()
        assert share == pytest.approx(0.1, abs=0.02)


def test_assign_split_gives_a_shared_structure_one_split_and_shard():
    from experiments.dash_subsets import assign_split

    keys = [f"k{i}" for i in range(40)] * 2
    subsets = ["QMugs"] * 40 + ["Extra"] * 40
    parsed = pd.DataFrame({"collapse_key": keys, "subset": subsets})
    cluster = np.array([i % 20 for i in range(40)] * 2)
    out = assign_split(parsed, cluster, np.ones(80, dtype=bool), n_shards=4)
    by_row = out.set_index("row")
    for i in range(40):
        assert by_row.loc[i, "split"] == by_row.loc[40 + i, "split"]
        assert by_row.loc[i, "shard"] == by_row.loc[40 + i, "shard"]
    assert set(out["shard"]) <= {"s00", "s01", "s02", "s03", "test"}


def test_assign_split_refuses_a_structure_spanning_clusters():
    from experiments.dash_subsets import assign_split

    parsed = pd.DataFrame({"collapse_key": ["a", "a"], "subset": ["QMugs", "QMugs"]})
    with pytest.raises(ValueError, match="span"):
        assign_split(parsed, np.array([0, 1]), np.ones(2, dtype=bool), n_shards=1)


def test_triangle_bounds_come_from_the_smallest_side():
    from experiments.dash_curation import triangle_bounds, triangles

    copies = pd.DataFrame(
        {
            "row_a": [0, 0, 1],
            "row_b": [1, 2, 2],
            "d_mbis": [0.01, 0.9, 0.95],
            "d_xtb": [0.0, 0.0, 0.0],
        }
    )
    tri = triangles(copies)
    assert len(tri) == 1
    assert tri["d1"].iat[0] == pytest.approx(0.01)
    agree, disagree = triangle_bounds(tri)
    assert agree == pytest.approx(0.01) and disagree == pytest.approx(0.01)


def test_copy_rule_trusts_a_witness_and_otherwise_the_esp_score():
    from experiments.dash_curation import copy_rule

    copies = pd.DataFrame(
        {
            "row_a": [0, 0, 1, 3],
            "row_b": [1, 2, 2, 4],
            "d_mbis": [0.01, 0.9, 0.95, 0.8],
        }
    )
    score = np.array([1.0, 1.0, 0.5, 2.0, 9.0])
    removed, info = copy_rule(copies, score, agree=0.05, disagree=0.2)
    assert removed == {2: "copy rule, witness", 4: "copy rule, ESP"}
    assert info["witnessed_pairs"] == 2 and info["esp_agrees_with_witness"] == 0


def test_esp_score_isolates_a_record_that_breaks_the_relation():
    from experiments.dash_curation import esp_score

    rng = np.random.default_rng(1)
    n_records, n_atoms = 400, 5
    z = np.tile([6, 1, 1, 8, 7], n_records)
    key = z.copy()
    q = rng.normal(0, 0.3, z.size)
    v = rng.normal(0, 0.5, z.size)
    esp = -10 + 0.4 * q + 0.5 * v + rng.normal(0, 0.01, z.size)
    esp[:n_atoms] += 1.0
    offsets = np.arange(0, z.size + 1, n_atoms)
    score = esp_score(z, key, q, v, esp, offsets)
    assert score[0] > 20 * np.median(score[1:])


def test_compare_pair_sees_a_mirror_image_as_a_copy():
    from experiments.dash_diagnostics import _mirror, compare_pair
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)O"))
    rdDistGeom.EmbedMolecule(mol, randomSeed=7)
    mirror = _mirror(mol)
    mols = [mol, mirror]
    heavy = [Chem.RemoveHs(m) for m in mols]
    mirrors = [_mirror(m) for m in heavy]
    charges = [
        {
            "mbis": np.arange(mol.GetNumAtoms(), dtype=float),
            "lowdin": None,
            "mulliken": None,
            "xtb": None,
        }
        for _ in mols
    ]
    energies = [
        {"e_tpssh": 1.0, "e_xtb": np.nan, "e_dft": np.nan, "e_gfn2": np.nan}
    ] * 2
    values = compare_pair(mols, heavy, mirrors, 0, 1, charges, energies)
    rmsd, is_mirror, d_mbis = values[0], values[1], values[2]
    assert rmsd < 1e-3 and is_mirror
    assert d_mbis == pytest.approx(0.0)
    assert values[-1] < 1e-6
