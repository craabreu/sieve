"""Fast-suite tests for the SPICE high-energy and low-energy stores: the generation
order, parsing into the staging store, curation, pairs, the shared split and the
separation -- on a synthetic HDF5 shaped like SPICE-2.0.1.hdf5."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
h5py = pytest.importorskip("h5py")

BOHR = 0.529177210903


def _embedded(smiles: str, seed: int = 7):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _group(f, name: str, smiles: str, n: int, *, seed: int = 0, subset: str = "S"):
    """A SPICE-like group of ``n`` conformations of ``smiles``: small jitters of one
    embedding, energies higher for the snapshots of the text order, random charges."""
    from experiments.spice_subsets import TEXT_ORDER
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    mol = _embedded(smiles, seed=seed + 1)
    xyz = mol.GetConformer().GetPositions()
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int16)
    g = f.create_group(name)
    g["smiles"] = np.array([Chem.MolToSmiles(mol)], dtype=object)
    g["subset"] = np.array([subset], dtype=object)
    g["atomic_numbers"] = z
    conf = (xyz[None] + rng.normal(0, 0.01, (n, len(z), 3))) / BOHR
    g.create_dataset("conformations", data=conf.astype(np.float32))
    g["conformations"].attrs["units"] = "bohr"
    q = rng.normal(0, 0.3, (n, len(z), 1))
    q -= q.mean(axis=1, keepdims=True)
    g["mbis_charges"] = q.astype(np.float32)
    snapshot = (TEXT_ORDER[:n] < 25) if n == 50 else np.zeros(n, bool)
    g["dft_total_energy"] = -100.0 + 0.05 * snapshot + rng.normal(0, 1e-3, n)
    g["dft_total_gradient"] = rng.normal(0, 0.01, (n, len(z), 3)).astype(np.float32)


def _hdf5(tmp_path, groups):
    path = tmp_path / "SPICE-2.0.1.hdf5"
    with h5py.File(path, "w") as f:
        for name, smiles, n, seed in groups:
            _group(f, name, smiles, n, seed=seed)
    return path


GROUPS = [
    ("m1", "CCO", 50, 0),
    ("m2", "CCN", 50, 1),
    ("m3", "CCCl", 50, 2),
    ("m4", "OCCO", 50, 3),
    ("m5", "CC(=O)O", 50, 4),
    ("m6", "CCCC", 50, 5),
    ("short", "CS", 49, 6),
    ("pair", "C.O", 50, 7),
]


def test_text_order_maps_positions_to_generation_indices():
    from experiments.spice_subsets import TEXT_ORDER, generation_indices, sampling_of

    assert TEXT_ORDER[:4].tolist() == [0, 1, 10, 11]
    assert TEXT_ORDER[12:14].tolist() == [2, 20]
    assert sorted(TEXT_ORDER.tolist()) == list(range(50))
    assert int((TEXT_ORDER < 25).sum()) == 25
    assert generation_indices(49) is None and generation_indices(100) is None
    assert [sampling_of(g) for g in (0, 24, 25, 49, -1)] == [
        "high_energy",
        "high_energy",
        "low_energy",
        "low_energy",
        "",
    ]


@pytest.fixture
def staged(tmp_path):
    from experiments.spice_subsets import parse_staging

    path = _hdf5(tmp_path, GROUPS)
    parsed_path, fields_path = tmp_path / "parsed.parquet", tmp_path / "fields.parquet"
    totals = parse_staging(path, parsed_path, fields_path, workers=2)
    return pd.read_parquet(parsed_path), pd.read_parquet(fields_path), totals


def test_parse_staging_labels_complete_groups_and_drops_multi_fragment(staged):
    from experiments.spice_subsets import TEXT_ORDER

    parsed, fields, totals = staged
    assert len(parsed) == len(fields) == 6 * 50 + 49
    assert "pair" not in set(parsed["spice_id"])
    assert totals["multi_fragment_groups::S"] == 1
    m1 = parsed[parsed["spice_id"] == "m1"]
    assert m1["generation"].tolist() == TEXT_ORDER.tolist()
    assert (m1["sampling"] == "high_energy").sum() == 25
    short = parsed[parsed["spice_id"] == "short"]
    assert (short["generation"] == -1).all() and (short["sampling"] == "").all()
    assert parsed["collapse_key"].notna().all() and (parsed["graph_smiles"] != "").all()
    assert np.isfinite(fields["energy"]).all()


def test_curate_removes_incomplete_molecules_then_geometry_flags(staged):
    from experiments.spice_subsets import curate, sampling_check

    parsed, fields, _ = staged
    parsed = parsed.copy()
    parsed.loc[3, "stretched_bond"] = True
    table, info = curate(parsed)
    step = table["curation_step"].to_numpy()
    assert (step[parsed["spice_id"] == "short"] == "incomplete").all()
    assert step[3] == "geometry"
    assert info["incomplete"] == {
        "molecules": 1,
        "conformations": 49,
        "groups_below_50": 1,
        "groups_above_50": 0,
    }
    assert info["kept"]["conformations"] == 6 * 50 - 1
    check = sampling_check(parsed, fields)
    assert check["relaxed_with_parent"] == 6 * 25 and check["lower_in_energy"] == 1.0


def test_draw_pairs_stays_within_one_kind_and_caps_the_count(staged):
    from experiments.spice_subsets import draw_pairs

    parsed, _, _ = staged
    kept = parsed[parsed["sampling"] != ""].reset_index(drop=True)
    groups = draw_pairs(kept, pairs_per_kind=10)
    assert len(groups) == 12
    kinds = kept["sampling"].to_numpy()
    for _, kind, a, b in groups:
        assert len(a) == 10
        assert (kinds[a] == kind).all() and (kinds[b] == kind).all()
        assert (a != b).all()


def test_pair_task_sees_a_mirror_image_as_identical():
    from experiments.dash_diagnostics import _mirror
    from experiments.data import mol_to_blob
    from experiments.spice_subsets import _pair_task

    mol = _embedded("CC(O)CN")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetDoubleProp("MBIScharge", 0.1 * atom.GetIdx())
    mirror = _mirror(mol)
    for atom, src in zip(mirror.GetAtoms(), mol.GetAtoms(), strict=True):
        atom.SetDoubleProp("MBIScharge", src.GetDoubleProp("MBIScharge"))
    rows = np.array([0, 1])
    blobs = [mol_to_blob(mol), mol_to_blob(mirror)]
    (pair,) = _pair_task(
        [(rows, "low_energy", rows[:1], rows[1:], blobs, np.array([-1.0, -1.0]))]
    )
    _, _, kind, rmsd, mirrored, d_mbis, de, heavy = pair
    assert kind == "low_energy" and mirrored
    assert rmsd < 1e-6 and d_mbis == pytest.approx(0.0) and de == 0.0
    assert heavy == 5


def test_split_keeps_both_kinds_of_a_molecule_together_and_stores_partition(staged):
    from experiments.dash_subsets import cluster_records
    from experiments.spice_subsets import assign_split, curate, separate

    parsed, _, _ = staged
    table, _ = curate(parsed)
    kept = (table["curation_step"] == "").to_numpy()
    cluster = cluster_records(parsed)
    split = assign_split(parsed, cluster, kept, n_shards=2)
    stores = separate(parsed, cluster, split)
    high, low = stores["high_energy"], stores["low_energy"]
    assert len(high) + len(low) == int(kept.sum())
    assert (high["sampling"] == "high_energy").all()
    shard = pd.concat([high, low]).groupby("spice_id")["shard"].nunique()
    assert (shard == 1).all()
    assert set(high["spice_id"]) == set(low["spice_id"])
    assert "short" not in set(high["spice_id"])
    for column in ("split", "shard", "cluster", "collapse_key", "n_collapsed"):
        assert column in high.columns
    assert "graph_smiles" not in high.columns


def test_prepare_spice_subsets_builds_both_stores(tmp_path):
    from experiments.spice_subsets import prepare_spice_subsets

    path = _hdf5(tmp_path, GROUPS)
    out = prepare_spice_subsets(
        tmp_path / "stores", hdf5_path=path, n_shards=2, workers=2
    )
    assert set(out) == {"high_energy", "low_energy"}
    staging = tmp_path / "stores" / "spice-staging"
    for name in ("parsed", "record-fields", "clusters", "curation", "pairs", "split"):
        assert (staging / f"{name}.parquet").exists()
    pairs = pd.read_parquet(staging / "pairs.parquet")
    parsed = pd.read_parquet(staging / "parsed.parquet")
    kinds = parsed["sampling"].to_numpy()
    assert (kinds[pairs["row_a"]] == kinds[pairs["row_b"]]).all()
    assert (kinds[pairs["row_a"]] == pairs["sampling"]).all()
    for kind in ("high-energy", "low-energy"):
        store = pd.read_parquet(
            tmp_path / "stores" / f"spice-{kind}" / "molecules.parquet"
        )
        assert len(store) == 6 * 25
        assert (tmp_path / "stores" / f"spice-{kind}" / "diagnostics.json").exists()
    again = prepare_spice_subsets(tmp_path / "stores", hdf5_path=path, n_shards=2)
    assert again == out


def test_build_parser_prepare_spice_subsets_defaults():
    from experiments.cli import build_parser

    args = build_parser().parse_args(["prepare-spice-subsets"])
    assert args.hdf5_path is None
    assert args.n_shards == 50
    assert args.workers == 16
    assert args.limit_groups is None
