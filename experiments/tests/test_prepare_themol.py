"""Fast-suite tests for prepare_themol.py, on small HDF5 files written here in
THEMol's own layout -- no access to the real 31 GB subset needed.

A record is built from a stereo-specified SMILES: embedded in 3D, its atom
indices written as map numbers, so the SMILES carries them in canonical order
while the arrays follow index order, as in the real files. The *reported*
SMILES can then be swapped for a different stereoisomer's, or a
stereo-free one, while the geometry stays put.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
h5py = pytest.importorskip("h5py")


def _embedded(smiles: str, seed: int = 7):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, smiles
    return mol


def _mapped(mol, *, isomeric: bool = True) -> str:
    """``mol``'s SMILES with each atom's index + 1 as its map number."""
    from rdkit import Chem

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(copy, isomericSmiles=isomeric)


def _arrays(mol):
    import numpy as np

    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    coords = mol.GetConformer().GetPositions()
    charges = 0.01 * (np.arange(mol.GetNumAtoms()) + 1)
    return z, coords, charges


def _write_h5(path, records):
    """``records``: ``(uuid, mol, reported_smiles)`` triples, written as
    THEMol groups."""
    with h5py.File(path, "w") as f:
        for uuid, mol, smiles in records:
            z, coords, charges = _arrays(mol)
            g = f.create_group(uuid)
            g["atomic_numbers"] = z.reshape(-1, 1).astype("int32")
            g["coords"] = coords
            g["mapped_isomeric_smiles"] = smiles
            g["mapped_nonisomeric_smiles"] = _mapped(mol, isomeric=False)
            g["mbis_info/atomic_charge"] = charges.reshape(-1, 1)


def _build(mol, reported: str | None = None):
    from experiments.prepare_themol import build_record_mol

    z, coords, charges = _arrays(mol)
    return build_record_mol(z, coords, charges, reported or _mapped(mol))


def _cip(mol, idx: int) -> str:
    return mol.GetAtomWithIdx(idx).GetProp("_CIPCode")


def test_charges_and_coordinates_land_on_the_mapped_atoms():
    import numpy as np

    source = _embedded("N[C@@H](C)C(=O)O")
    mol, _, _ = _build(source)
    z, coords, charges = _arrays(source)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == z.tolist()
    assert np.allclose(mol.GetConformer().GetPositions(), coords)
    stored = [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()]
    assert np.allclose(stored, charges)


def test_a_consistent_record_agrees():
    source = _embedded("C/C=C/[C@@H](O)CC")
    _, verdict, counts = _build(source)
    assert verdict == "agree"
    assert counts == {"atom_common": 1, "bond_common": 1}


def test_a_mirrored_report_is_a_conflict_and_the_geometry_wins():
    source = _embedded("N[C@@H](C)C(=O)O")
    mirror = _embedded("N[C@H](C)C(=O)O")
    mol, verdict, counts = _build(source, _mapped(mirror))
    assert verdict == "conflict"
    assert counts == {"atom_common": 1}
    reference, _, _ = _build(source)
    assert _cip(mol, 1) == _cip(reference, 1)


def test_an_opposite_double_bond_is_a_conflict():
    # The SMILES side carries STEREOE/STEREOZ and the 3D side STEREOCIS/
    # STEREOTRANS, each relative to its own stereo atoms, so this is also the
    # test that the two conventions are reconciled.
    source = _embedded("Cl/C=C(\\C)CC")
    other = _embedded("Cl/C=C(/C)CC")
    _, verdict, counts = _build(source, _mapped(other))
    assert verdict == "conflict"
    assert counts == {"bond_common": 1}
    _, verdict, _ = _build(source)
    assert verdict == "agree"


def test_a_conflict_is_found_beside_an_element_only_one_side_specifies():
    # The reported SMILES leaves the double bond open and inverts the centre.
    # The bond is dropped from the comparison, and must not come back from
    # the single-bond directions its 3D perception left behind.
    from rdkit import Chem

    source = _embedded("C/C=C/[C@@H](O)CC")
    other = Chem.Mol(source)
    other.GetAtomWithIdx(3).InvertChirality()
    for bond in other.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    _, verdict, counts = _build(source, _mapped(other))
    assert verdict == "conflict"
    assert counts == {"atom_common": 1, "bond_perceived_only": 1}


def test_a_report_without_stereo_is_perceived_only():
    source = _embedded("C/C=C/[C@@H](O)CC")
    _, verdict, counts = _build(source, _mapped(source, isomeric=False))
    assert verdict == "perceived_only"
    assert counts["atom_perceived_only"] == 1
    assert counts["bond_perceived_only"] == 1


def test_inverting_a_symmetric_centre_is_not_a_conflict():
    # In spiro[5.5]undecane-3,9-diol, inverting one carbinol centre gives back
    # the same molecule, so an index-by-index comparison of parities would
    # report a conflict where there is none.
    from rdkit import Chem

    source = _embedded("O[C@H]1CC[C@]2(CC1)CC[C@H](O)CC2")
    other = Chem.Mol(source)
    other.GetAtomWithIdx(1).InvertChirality()
    assert Chem.MolToSmiles(other) == Chem.MolToSmiles(source)
    _, verdict, _ = _build(source, _mapped(other))
    assert verdict == "agree"


def test_a_centre_without_a_cip_label_is_still_counted():
    # A real THEMol record (016d2c483c475214840c8e11de079232): its reported
    # SMILES leaves the spiro carbon open, 3D perception tags it, and the CIP
    # labeler gives it no descriptor -- so a comparison of CIP labels would
    # not see it at all.
    source = _embedded("COC(=O)N1CC2(C1)O[C@@H]1C[C@@]2(C(N)=O)C1")
    mol, verdict, counts = _build(source)
    spiro = mol.GetAtomWithIdx(6)
    assert spiro.GetChiralTag() != spiro.GetChiralTag().CHI_UNSPECIFIED
    assert not spiro.HasProp("_CIPCode")
    assert verdict == "perceived_only"
    assert counts == {"atom_common": 2, "atom_perceived_only": 1}


def test_relative_configuration_without_cip_centres_is_compared():
    # cis- vs trans-1,4-dimethylcyclohexane: no chirality centre, only a
    # relative configuration, which canonical SMILES still distinguishes.
    source = _embedded("C[C@H]1CC[C@@H](C)CC1")
    trans = _embedded("C[C@H]1CC[C@H](C)CC1")
    _, verdict, _ = _build(source, _mapped(trans))
    assert verdict == "conflict"


def test_isotopes_are_kept():
    source = _embedded("[2H]C([2H])([2H])O")
    mol, verdict, _ = _build(source)
    assert sorted(a.GetIsotope() for a in mol.GetAtoms() if a.GetAtomicNum() == 1) == [
        0,
        2,
        2,
        2,
    ]
    assert verdict == "agree"


def test_mismatched_elements_are_rejected():
    from experiments.prepare_themol import build_record_mol

    source = _embedded("CCO")
    z, coords, charges = _arrays(source)
    z[0] = 7
    with pytest.raises(ValueError, match="atomic numbers"):
        build_record_mol(z, coords, charges, _mapped(source))


def test_parse_themol_shard_skips_a_bad_record_and_counts_the_rest(tmp_path):
    import pandas as pd
    from experiments.prepare_themol import parse_themol_shard

    good = _embedded("N[C@@H](C)C(=O)O")
    bad = _embedded("CCO")
    h5_path = tmp_path / "mbis_0.h5"
    _write_h5(
        h5_path,
        [("a", good, _mapped(good)), ("b", bad, _mapped(_embedded("CCN")))],
    )
    out = tmp_path / "part.parquet"
    totals = parse_themol_shard(h5_path, out)
    df = pd.read_parquet(out)
    assert df["themol_id"].tolist() == ["a"]
    assert df.loc[0, "stereo_check"] == "agree"
    assert df.loc[0, "h5_file"] == "mbis_0.h5"
    assert totals["skipped"] == 1
    assert totals["verdict_agree"] == 1


_DIVERSE_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "CC(C)=O",
    "CN",
    "c1ccoc1",
    "CC#N",
    "C1CCNCC1",
    "CC(=O)N",
    "c1ccsc1",
    "CCN(CC)CC",
    "OCC1CCCCC1",
    "c1ccc2ccccc2c1",
    "COC",
    "FC(F)F",
    "ClC(Cl)Cl",
    "c1cc[nH]c1",
    "C1CCOC1",
    "CC(C)(C)O",
    "c1ccc(cc1)O",
    "NC(=O)N",
    "N[C@@H](C)C(=O)O",
]


def _source_dir(tmp_path, *, reported=None):
    """Eight THEMol files, three molecules each, the molecules taken in
    order from ``_DIVERSE_SMILES``."""
    from experiments.prepare_themol import SHARD_FILES

    source = tmp_path / "MBIS"
    source.mkdir()
    for i, name in enumerate(SHARD_FILES):
        records = []
        for j, smiles in enumerate(_DIVERSE_SMILES[3 * i : 3 * i + 3]):
            mol = _embedded(smiles)
            records.append((f"{i}-{j}", mol, _mapped(mol)))
        _write_h5(source / name, records)
    return source


def test_parse_themol_keeps_file_order_whatever_the_workers(tmp_path):
    import pandas as pd
    from experiments.prepare_themol import parse_themol

    source = _source_dir(tmp_path)
    out = tmp_path / "molecules.parquet"
    totals = parse_themol(source, out, workers=3)
    df = pd.read_parquet(out)
    assert df["themol_id"].tolist() == [f"{i}-{j}" for i in range(8) for j in range(3)]
    assert totals["verdict_agree"] == len(_DIVERSE_SMILES)
    assert not (tmp_path / ".parts").exists()


def test_prepare_store_parses_checks_and_splits(tmp_path):
    import pandas as pd
    from experiments.prepare_themol import STEREO_SUMMARY, prepare_store

    source = _source_dir(tmp_path)
    stores = tmp_path / "stores"
    prepare_store("t", stores_root=stores, source_dir=source, n_shards=2, workers=2)
    store = stores / "t"
    df = pd.read_parquet(store / "molecules.parquet")
    assert {"split", "cluster", "shard", "stereo_check"} <= set(df.columns)
    assert set(df["split"]) == {"train", "test"}
    summary = (store / STEREO_SUMMARY).read_text()
    assert f"records: {len(_DIVERSE_SMILES)} (0 skipped)" in summary

    # Idempotent: a second call finds the store parsed and split.
    before = (store / "molecules.parquet").stat().st_mtime_ns
    prepare_store("t", stores_root=stores, source_dir=source, n_shards=2)
    assert (store / "molecules.parquet").stat().st_mtime_ns == before


def test_prepare_store_can_stop_before_the_split(tmp_path):
    import pyarrow.parquet as pq
    from experiments.prepare_themol import prepare_store

    source = _source_dir(tmp_path)
    stores = tmp_path / "stores"
    prepare_store("t", stores_root=stores, source_dir=source, stop_before_split=True)
    names = set(pq.ParquetFile(stores / "t" / "molecules.parquet").schema.names)
    assert "split" not in names


def _served(tmp_path, contents: dict[str, bytes]):
    """Files served from a ``file://`` base URL, which ``urllib`` opens the
    same way as the Hugging Face one, with the ``expected`` mapping that
    describes them."""
    import hashlib

    served = tmp_path / "served"
    served.mkdir()
    for name, data in contents.items():
        (served / name).write_bytes(data)
    expected = {
        name: (len(data), hashlib.sha256(data).hexdigest())
        for name, data in contents.items()
    }
    return served.as_uri(), expected


def test_download_themol_fetches_and_verifies(tmp_path):
    from experiments.prepare_themol import download_themol

    base_url, expected = _served(tmp_path, {"a.h5": b"alpha", "b.h5": b"beta"})
    dest = download_themol(tmp_path / "dest", base_url=base_url, expected=expected)
    assert (dest / "a.h5").read_bytes() == b"alpha"
    assert (dest / "b.h5").read_bytes() == b"beta"


def test_download_themol_keeps_a_file_of_the_expected_size(tmp_path):
    from experiments.prepare_themol import download_themol

    _, expected = _served(tmp_path, {"a.h5": b"alpha"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.h5").write_bytes(b"ALPHA")
    # An unreachable URL: the size check alone must decide.
    download_themol(dest, base_url="file:///nonexistent", expected=expected)
    assert (dest / "a.h5").read_bytes() == b"ALPHA"


def test_download_themol_rejects_and_removes_a_corrupted_file(tmp_path):
    from experiments.prepare_themol import download_themol

    base_url, expected = _served(tmp_path, {"a.h5": b"alpha"})
    expected["a.h5"] = (expected["a.h5"][0], "0" * 64)
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="sha256"):
        download_themol(dest, base_url=base_url, expected=expected)
    assert not (dest / "a.h5").exists()


def test_download_themol_rejects_an_incomplete_file(tmp_path):
    from experiments.prepare_themol import download_themol

    base_url, expected = _served(tmp_path, {"a.h5": b"alpha"})
    expected["a.h5"] = (10, expected["a.h5"][1])
    with pytest.raises(ValueError, match="incomplete"):
        download_themol(tmp_path / "dest", base_url=base_url, expected=expected)


def test_prepare_store_downloads_when_no_source_dir_is_given(tmp_path, monkeypatch):
    import experiments.prepare_themol as prepare_themol

    source = _source_dir(tmp_path)
    fetched = []

    def fake_download(dest_dir):
        fetched.append(dest_dir)
        return source

    monkeypatch.setattr(prepare_themol, "download_themol", fake_download)
    stores = tmp_path / "stores"
    prepare_themol.prepare_store("t", stores_root=stores, stop_before_split=True)
    assert fetched == [stores / "t"]
    assert (stores / "t" / "molecules.parquet").exists()

    # Once parsed, nothing is fetched again.
    prepare_themol.prepare_store("t", stores_root=stores, stop_before_split=True)
    assert len(fetched) == 1


def test_prepare_store_rejects_a_missing_source_dir(tmp_path):
    from experiments.prepare_themol import prepare_store

    with pytest.raises(ValueError, match="does not exist"):
        prepare_store("t", stores_root=tmp_path, source_dir=tmp_path / "nope")
