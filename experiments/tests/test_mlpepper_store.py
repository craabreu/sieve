"""Fast-suite tests for the MLPepper vacuum and water stores: decoding the view,
parsing into the staging store, curation, pairs, the shared split and the separation --
on a synthetic SQLite view shaped like the Zenodo one."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

pytest.importorskip("rdkit")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
msgpack = pytest.importorskip("msgpack")
zstandard = pytest.importorskip("zstandard")

BOHR = 0.529177210903
SPECS = ("wb97x-d/def2-tzvpp", "wb97x-d/def2-tzvpp/ddx-water")


def _blob(obj) -> bytes:
    return zstandard.ZstdCompressor().compress(msgpack.packb(obj, use_bin_type=True))


def _embedded(smiles: str, seed: int):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _molecule(mol, jitter: float, seed: int) -> dict:
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    xyz = mol.GetConformer().GetPositions() + rng.normal(
        0, jitter, (mol.GetNumAtoms(), 3)
    )
    return {
        "symbols": [a.GetSymbol() for a in mol.GetAtoms()],
        "geometry": (xyz / BOHR).ravel().tolist(),
        "connectivity": [
            [b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1.0] for b in mol.GetBonds()
        ],
        "molecular_charge": 0.0,
        "molecular_multiplicity": 1,
        "extras": {
            "canonical_isomeric_explicit_hydrogen_mapped_smiles": Chem.MolToSmiles(mol)
        },
    }


def _record(
    molecule: dict,
    molecule_id: int,
    *,
    status: str,
    energy: float,
    seed: int,
    water: bool,
):
    rng = np.random.default_rng(seed)
    n = len(molecule["symbols"])
    q = rng.normal(0, 0.3, n)
    q -= q.mean()
    properties = (
        {
            "return_energy": energy,
            "scf_dipole_moment": [0.1, 0.2, 0.3],
            "mbis charges": q.tolist(),
            "mulliken charges": (q * 1.1).tolist(),
            "lowdin charges": (q * 0.9).tolist(),
            **({"dd solvation energy": -0.01} if water else {}),
        }
        if status == "complete"
        else None
    )
    return {
        "status": status,
        "molecule_id": molecule_id,
        "molecule": molecule,
        "properties": properties,
        "compute_history": [{"provenance": {"version": "1.9"}}],
    }


# entry name, smiles, jitter seed, vacuum status
ENTRIES = [
    ("CCO", "CCO", 0, "complete"),
    ("CC(C)O-0", "CC(C)O", 1, "complete"),
    ("CC(C)O-1", "CC(C)O", 1, "complete"),  # same embedding and jitter: a copy
    ("CCN", "CCN", 3, "error"),
    ("CCCl", "CCCl", 4, "complete"),
    ("OCCO", "OCCO", 5, "complete"),
]


def _view(tmp_path):
    path = tmp_path / "view.sqlite"
    db = sqlite3.connect(path)
    db.executescript(
        "create table records (id integer primary key, status text not null, "
        "modified_on decimal not null, record blob not null);"
        "create table dataset_entries (name text primary key, entry blob not null);"
        "create table dataset_specifications (name text primary key, "
        "specification blob not null);"
        "create table dataset_records (entry_name text not null, "
        "specification_name text not null, record_id integer not null, "
        "primary key (entry_name, specification_name));"
    )
    for spec in SPECS:
        db.execute(
            "insert into dataset_specifications values (?, ?)",
            (spec, _blob({"name": spec, "specification": {"program": "psi4"}})),
        )
    rid = 100
    embedded = {}
    for k, (name, smiles, seed, vacuum_status) in enumerate(ENTRIES):
        mol = embedded.setdefault(smiles, _embedded(smiles, seed=11))
        molecule = _molecule(mol, 0.02, seed)
        db.execute(
            "insert into dataset_entries values (?, ?)",
            (
                name,
                _blob(
                    {"name": name, "attributes": {"canonical_isomeric_smiles": smiles}}
                ),
            ),
        )
        for spec, status, water in zip(
            SPECS, (vacuum_status, "complete"), (False, True), strict=True
        ):
            record = _record(
                molecule,
                1000 + k,
                status=status,
                energy=-100.0 - 0.001 * k - 0.5 * water,
                seed=100 * k + water,
                water=water,
            )
            db.execute(
                "insert into records values (?, ?, ?, ?)",
                (rid, status, 0.0, _blob(record)),
            )
            db.execute(
                "insert into dataset_records values (?, ?, ?)", (name, spec, rid)
            )
            rid += 1
    db.commit()
    db.close()
    return path


@pytest.fixture
def staged(tmp_path):
    from experiments.mlpepper_store import parse_staging

    path = _view(tmp_path)
    parsed_path, fields_path = tmp_path / "parsed.parquet", tmp_path / "fields.parquet"
    totals = parse_staging(path, parsed_path, fields_path, workers=2)
    return pd.read_parquet(parsed_path), pd.read_parquet(fields_path), totals


def test_parse_staging_reads_both_phases_and_flags_the_incomplete_entry(staged):
    parsed, fields, totals = staged
    assert len(parsed) == len(fields) == len(ENTRIES)
    assert parsed["entry"].tolist() == sorted(e[0] for e in ENTRIES)
    row = parsed.set_index("entry")
    frow = fields.set_index(parsed["entry"])
    assert frow.loc["CCN", "status_vacuum"] == "error"
    assert row.loc["CCN", "q_vacuum"] is None
    assert row.loc["CCN", "q_water"] is not None
    assert row["connectivity_agrees"].all()
    assert (row["molecule"] == row.index.str.replace(r"-\d$", "", regex=True)).all()
    assert row.loc["CC(C)O-0", "collapse_key"] == row.loc["CC(C)O-1", "collapse_key"]
    assert np.isfinite(frow["energy_water"]).all()
    assert np.isnan(frow.loc["CCN", "energy_vacuum"])
    assert totals["status_vacuum::error"] == 1 and totals["connectivity_disagrees"] == 0


def test_curate_removes_the_incomplete_entry_only(staged):
    from experiments.mlpepper_store import curate

    parsed, fields, _ = staged
    table, info = curate(parsed, fields)
    removed = parsed.loc[table["curation_step"] != "", "entry"].tolist()
    assert removed == ["CCN"]
    assert info["incomplete"] == 1 and info["geometry"] == 0
    assert info["kept"]["entries"] == len(ENTRIES) - 1


def test_diagnose_pairs_sees_the_copy(staged):
    from experiments.mlpepper_store import HARTREE, diagnose_pairs, pair_summary

    parsed, fields, _ = staged
    pairs = diagnose_pairs(parsed, fields, workers=2)
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    names = parsed["entry"].to_numpy()
    assert {names[pair["row_a"]], names[pair["row_b"]]} == {"CC(C)O-0", "CC(C)O-1"}
    assert pair["rmsd"] < 1e-6
    assert pair["heavy_atoms"] == 4
    e = fields["energy_vacuum"].to_numpy()
    assert pair["de_vacuum"] == pytest.approx(
        abs(e[pair["row_a"]] - e[pair["row_b"]]) * HARTREE
    )
    assert pair["d_mbis_vacuum"] >= 0 and pair["d_mbis_water"] >= 0
    summary = pair_summary(pairs)
    assert summary["n_copies"] == 1 and summary["copy_fraction"] == 1.0


def test_prepare_mlpepper_stores_builds_both_stores(tmp_path):
    from experiments.data import blob_to_mol
    from experiments.mlpepper_store import prepare_mlpepper_stores

    path = _view(tmp_path)
    out = prepare_mlpepper_stores(
        tmp_path / "stores", sqlite_path=path, n_shards=2, workers=2
    )
    assert set(out) == {"vacuum", "water"}
    staging = tmp_path / "stores" / "mlpepper-staging"
    for name in ("parsed", "record-fields", "clusters", "curation", "pairs", "split"):
        assert (staging / f"{name}.parquet").exists()
    parsed = pd.read_parquet(staging / "parsed.parquet").set_index("entry")
    stores = {
        phase: pd.read_parquet(
            tmp_path / "stores" / f"mlpepper-{phase}" / "molecules.parquet"
        )
        for phase in ("vacuum", "water")
    }
    for phase, store in stores.items():
        assert len(store) == len(ENTRIES) - 1 and "CCN" not in store["entry"].tolist()
        assert {"split", "shard", "cluster", "collapse_key", "n_collapsed"} <= set(
            store.columns
        )
        assert not any(c.startswith("q_") for c in store.columns)
        for entry, blob in zip(store["entry"], store["mol"], strict=True):
            q = [a.GetDoubleProp("MBIScharge") for a in blob_to_mol(blob).GetAtoms()]
            assert q == pytest.approx(parsed.loc[entry, f"q_{phase}"], abs=1e-6)
    assert stores["vacuum"][["entry", "split", "shard"]].equals(
        stores["water"][["entry", "split", "shard"]]
    )
    copies = stores["water"].set_index("entry").loc[["CC(C)O-0", "CC(C)O-1"]]
    assert copies["split"].nunique() == 1 and copies["n_collapsed"].tolist() == [2, 2]
    assert (tmp_path / "stores" / "mlpepper-water" / "diagnostics.json").exists()
    again = prepare_mlpepper_stores(tmp_path / "stores", sqlite_path=path, n_shards=2)
    assert again == out


def test_build_parser_prepare_mlpepper_stores_defaults():
    from experiments.cli import build_parser

    args = build_parser().parse_args(["prepare-mlpepper-stores"])
    assert args.sqlite_path is None
    assert args.n_shards == 50
    assert args.workers == 16
    assert args.limit_entries is None
