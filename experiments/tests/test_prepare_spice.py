"""Fast-suite tests for prepare_spice.py, on small HDF5 files written here in
SPICE's own layout (checked against modelforge's two-record SPICE sample):
``smiles``/``subset`` as one-element byte arrays, coordinates as float32 in
bohr with a ``units`` attribute, charges shaped ``(M, N, 1)``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
h5py = pytest.importorskip("h5py")

BOHR = 0.529177210903


def _embedded(smiles: str, n_conformers: int = 3, seed: int = 7):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=seed)
    assert len(ids) == n_conformers, smiles
    return mol


def _mapped(mol) -> str:
    from rdkit import Chem

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(copy)


def _charges(mol, n_conformers: int):
    import numpy as np

    base = 0.01 * (np.arange(mol.GetNumAtoms()) + 1)
    return np.stack([base + 0.001 * k for k in range(n_conformers)])


def _write_group(f, name, mol, *, subset="SPICE PubChem Set 1", charges="auto"):
    import numpy as np

    n_conf = mol.GetNumConformers()
    g = f.create_group(name)
    g["smiles"] = np.array([_mapped(mol).encode()], dtype=object)
    g["subset"] = np.array([subset.encode()], dtype=object)
    g["atomic_numbers"] = np.array(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int16
    )
    coords = np.stack([c.GetPositions() for c in mol.GetConformers()]) / BOHR
    g["conformations"] = coords.astype(np.float32)
    g["conformations"].attrs["units"] = "bohr"
    if charges is not None:
        q = _charges(mol, n_conf) if isinstance(charges, str) else charges
        g["mbis_charges"] = q[..., None].astype(np.float32)
        g["mbis_charges"].attrs["units"] = "elementary_charge"


def _parse(tmp_path, write):
    import pandas as pd
    from experiments.prepare_spice import parse_spice

    path = tmp_path / "spice.hdf5"
    with h5py.File(path, "w") as f:
        write(f)
    out = tmp_path / "molecules.parquet"
    totals = parse_spice(path, out, workers=2)
    return pd.read_parquet(out), totals


def test_one_row_per_conformer_with_coordinates_in_angstrom(tmp_path):
    import numpy as np
    from experiments.data import blob_to_mol

    mol = _embedded("N[C@@H](C)C(=O)O")
    df, totals = _parse(tmp_path, lambda f: _write_group(f, "ala", mol))
    assert df["conf_id"].tolist() == ["conf_0", "conf_1", "conf_2"]
    assert set(df["spice_id"]) == {"ala"}
    assert set(df["stereo_check"]) == {"agree"}
    stored = blob_to_mol(df["mol"].iat[1])
    expected = mol.GetConformer(1).GetPositions()
    assert np.allclose(stored.GetConformer().GetPositions(), expected, atol=1e-5)
    q = [a.GetDoubleProp("MBIScharge") for a in stored.GetAtoms()]
    assert np.allclose(q, _charges(mol, 3)[1], atol=1e-6)
    assert totals["groups_kept"] == 1


def test_multi_fragment_groups_are_left_out_and_counted_by_subset(tmp_path):
    def write(f):
        _write_group(f, "ethanol", _embedded("CCO"))
        _write_group(f, "dimer", _embedded("CCO.O"), subset="SPICE DES370K")

    df, totals = _parse(tmp_path, write)
    assert set(df["spice_id"]) == {"ethanol"}
    assert totals["multi_fragment_groups::SPICE DES370K"] == 1


def test_a_group_without_mbis_charges_is_left_out(tmp_path):
    def write(f):
        _write_group(f, "ethanol", _embedded("CCO"))
        _write_group(f, "methanol", _embedded("CO"), charges=None)

    df, totals = _parse(tmp_path, write)
    assert set(df["spice_id"]) == {"ethanol"}
    assert totals["groups_without_mbis"] == 1


def test_a_conformer_with_nonfinite_charges_is_left_out(tmp_path):
    import numpy as np

    mol = _embedded("CCO")
    q = _charges(mol, 3)
    q[1, 2] = np.nan
    df, totals = _parse(tmp_path, lambda f: _write_group(f, "ethanol", mol, charges=q))
    assert df["conf_id"].tolist() == ["conf_0", "conf_2"]
    assert totals["conformers_nonfinite_charges"] == 1


def test_coordinates_in_another_unit_are_refused(tmp_path):
    def write(f):
        _write_group(f, "ethanol", _embedded("CCO"))
        f["ethanol/conformations"].attrs["units"] = "angstrom"

    with pytest.raises(ValueError, match="expected bohr"):
        _parse(tmp_path, write)


def _serve(tmp_path, monkeypatch, data: bytes):
    import hashlib

    import experiments.prepare_spice as prepare_spice

    served = tmp_path / "served.hdf5"
    served.write_bytes(data)
    monkeypatch.setattr(prepare_spice, "EXPECTED_BYTES", len(data))
    monkeypatch.setattr(prepare_spice, "EXPECTED_MD5", hashlib.md5(data).hexdigest())
    return served.as_uri()


def test_download_fetches_and_verifies(tmp_path, monkeypatch):
    from experiments.prepare_spice import HDF5_FILENAME, download_spice_hdf5

    url = _serve(tmp_path, monkeypatch, b"spice")
    out = download_spice_hdf5(tmp_path / "dest", url=url)
    assert out == tmp_path / "dest" / HDF5_FILENAME
    assert out.read_bytes() == b"spice"


def test_download_keeps_a_file_of_the_expected_size(tmp_path, monkeypatch):
    from experiments.prepare_spice import HDF5_FILENAME, download_spice_hdf5

    _serve(tmp_path, monkeypatch, b"spice")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / HDF5_FILENAME).write_bytes(b"SPICE")
    download_spice_hdf5(dest, url="file:///nonexistent")
    assert (dest / HDF5_FILENAME).read_bytes() == b"SPICE"


def test_download_removes_a_corrupted_file(tmp_path, monkeypatch):
    import experiments.prepare_spice as prepare_spice

    url = _serve(tmp_path, monkeypatch, b"spice")
    monkeypatch.setattr(prepare_spice, "EXPECTED_MD5", "0" * 32)
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="md5"):
        prepare_spice.download_spice_hdf5(dest, url=url)
    assert not (dest / prepare_spice.HDF5_FILENAME).exists()


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
]


def _diverse_hdf5(tmp_path, *, bad: str | None = None):
    """``_DIVERSE_SMILES`` as three-conformer groups, and one multi-fragment
    group. ``bad`` names a group whose second conformer carries a failed-MBIS
    outlier on its first atom, which curation must remove."""
    path = tmp_path / "spice.hdf5"
    with h5py.File(path, "w") as f:
        for i, smiles in enumerate(_DIVERSE_SMILES):
            mol = _embedded(smiles)
            q = _charges(mol, 3)
            if bad == f"m{i:02d}":
                q[1, 0] += 3.0
            _write_group(f, f"m{i:02d}", mol, charges=q)
        _write_group(f, "pair", _embedded("CCO.O"), subset="SPICE DES370K")
    return path


def test_prepare_store_parses_curates_and_splits(tmp_path):
    import pandas as pd
    from experiments.prepare_dash import CURATION_SUMMARY, UNCURATED_PARQUET
    from experiments.prepare_spice import SUMMARY, prepare_store

    path = _diverse_hdf5(tmp_path, bad="m03")
    stores = tmp_path / "stores"
    prepare_store(
        "s",
        stores_root=stores,
        hdf5_path=path,
        n_shards=2,
        workers=3,
        keep_uncurated=True,
    )
    store = stores / "s"
    df = pd.read_parquet(store / "molecules.parquet")
    uncurated = pd.read_parquet(store / UNCURATED_PARQUET)
    assert len(uncurated) == 3 * len(_DIVERSE_SMILES)
    assert len(df) == len(uncurated) - 1
    assert "conf_1" not in set(df.loc[df["spice_id"] == "m03", "conf_id"])
    assert {"split", "cluster", "shard"} <= set(df.columns)
    assert df.groupby("spice_id")["split"].nunique().max() == 1
    assert "1 removed" in (store / CURATION_SUMMARY).read_text()
    assert "SPICE DES370K: 1" in (store / SUMMARY).read_text()

    # Idempotent: a second call finds the store curated and split.
    before = (store / "molecules.parquet").stat().st_mtime_ns
    prepare_store("s", stores_root=stores, hdf5_path=path, n_shards=2)
    assert (store / "molecules.parquet").stat().st_mtime_ns == before


def test_prepare_store_downloads_when_no_hdf5_path_is_given(tmp_path, monkeypatch):
    import experiments.prepare_spice as prepare_spice

    path = _diverse_hdf5(tmp_path)
    fetched = []

    def fake_download(dest_dir):
        fetched.append(dest_dir)
        return path

    monkeypatch.setattr(prepare_spice, "download_spice_hdf5", fake_download)
    stores = tmp_path / "stores"
    prepare_spice.prepare_store(
        "s", stores_root=stores, workers=2, stop_before_split=True
    )
    assert fetched == [stores / "s"]
    prepare_spice.prepare_store(
        "s", stores_root=stores, workers=2, stop_before_split=True
    )
    assert len(fetched) == 1


def test_prepare_store_rejects_a_missing_hdf5_path(tmp_path):
    from experiments.prepare_spice import prepare_store

    with pytest.raises(ValueError, match="does not exist"):
        prepare_store("s", stores_root=tmp_path, hdf5_path=tmp_path / "nope.hdf5")
