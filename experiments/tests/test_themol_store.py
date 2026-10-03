"""Fast-suite tests for the curated THEMol store: the skeleton, the staging parse from
an uncurated store, the three curation channels, the skeleton clustering, the pairs
and the end-to-end build, on a synthetic uncurated store shaped like
``themol-mbis``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("rdkit")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("tqdm")


def _embedded(smiles: str, seed: int):
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0, smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def _record(
    uuid: str,
    smiles: str,
    *,
    seed: int,
    jitter: float = 0.02,
    charge_error: float = 0.0,
    stretch: bool = False,
) -> dict:
    """One row of an uncurated THEMol store: random charges adding up to the net
    charge (plus ``charge_error``), optionally one hydrogen moved far away."""
    from experiments.collapse import collapse_key
    from experiments.geometry import geometry_record
    from experiments.prepare_themol import build_record_mol, store_blob
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    mol = _embedded(smiles, seed=11)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    xyz = mol.GetConformer().GetPositions() + rng.normal(0, jitter, (len(z), 3))
    if stretch:
        xyz[np.flatnonzero(z == 1)[0]] += 3.0
    q = rng.normal(0, 0.3, len(z))
    q += (Chem.GetFormalCharge(mol) + charge_error - q.sum()) / len(z)
    built, verdict, _ = build_record_mol(z, xyz, q, Chem.MolToSmiles(mol))
    return {
        "themol_id": uuid,
        "h5_file": "mbis_0.h5",
        "smiles": Chem.MolToSmiles(mol),
        "mol": store_blob(built),
        "net_charge": float(Chem.GetFormalCharge(built)),
        "stereo_check": verdict,
        "collapse_key": collapse_key(built),
        "n_collapsed": 1,
        "n_molecules": 1,
        "n_enantiomer_forms": 1,
        **geometry_record(built),
    }


RECORDS: list[tuple[str, str, dict[str, Any]]] = [
    ("u-ethanol-a", "CCO", {}),
    ("u-ethanol-b", "CCO", {"seed": 1}),  # a duplicate molecule, a copy
    ("u-acetic", "CC(=O)O", {}),
    ("u-acetate", "CC(=O)[O-]", {}),  # same skeleton as acetic acid
    ("u-phosphonate", "CP(=O)([O-])[O-]", {}),  # multi_anion: -2 with P
    ("u-succinate", "[O-]C(=O)CC(=O)[O-]", {}),  # -2 without B/Si/P: kept
    ("u-citrate", "[O-]C(=O)CC(O)(CC(=O)[O-])C(=O)[O-]", {}),  # -3: multi_anion
    ("u-badsum", "CCN", {"charge_error": 0.5}),  # charge_sum
    ("u-stretched", "CCCl", {"stretch": True}),  # geometry
    ("u-sulfate", "[O-]S(=O)(=O)[O-]", {}),  # -2 with S only: kept
]


def _uncurated(tmp_path):
    rows = []
    for k, (uuid, smiles, kw) in enumerate(RECORDS):
        kwargs: dict[str, Any] = {"seed": k, **kw}
        rows.append(_record(uuid, smiles, **kwargs))
    path = tmp_path / "themol-mbis" / "molecules.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def staged(tmp_path):
    from experiments.themol_store import parse_staging

    parsed_path = tmp_path / "parsed.parquet"
    n = parse_staging(_uncurated(tmp_path), parsed_path, workers=2)
    parsed = pd.read_parquet(parsed_path)
    assert n == len(parsed) == len(RECORDS)
    return parsed


def test_skeleton_ignores_protonation_and_bond_orders():
    from experiments.themol_store import skeleton_fingerprints, skeleton_smiles
    from rdkit import Chem

    acid, base = (Chem.AddHs(Chem.MolFromSmiles(s)) for s in ("CC(=O)O", "CC(=O)[O-]"))
    assert skeleton_smiles(acid) == skeleton_smiles(base) == "CC(O)O"
    fps = skeleton_fingerprints([skeleton_smiles(acid), "CCO"])
    assert fps.shape == (2, 2048) and fps.sum() > 0


def test_parse_staging_adds_the_augmented_columns(staged):
    from experiments.themol_store import AUGMENTED

    row = staged.set_index("themol_id")
    assert set(AUGMENTED) <= set(staged.columns)
    assert not {"n_collapsed", "n_molecules"} & set(staged.columns)
    assert row.loc["u-acetic", "skeleton"] == row.loc["u-acetate", "skeleton"]
    assert (
        row.loc["u-phosphonate", "sensitive"]
        and not row.loc["u-succinate", "sensitive"]
    )
    assert abs(row.loc["u-badsum", "charge_sum"] - 0.5) < 1e-6
    assert abs(row.loc["u-succinate", "charge_sum"] + 2.0) < 1e-6
    assert (
        row.loc["u-ethanol-a", "canonical_smiles"]
        == row.loc["u-ethanol-b", "canonical_smiles"]
    )


def test_curate_applies_the_three_channels_in_order(staged):
    from experiments.themol_store import curate

    table, info = curate(staged)
    step = dict(zip(staged["themol_id"], table["curation_step"], strict=True))
    assert step["u-phosphonate"] == "multi_anion" and step["u-citrate"] == "multi_anion"
    assert step["u-badsum"] == "charge_sum" and step["u-stretched"] == "geometry"
    assert step["u-succinate"] == "" and step["u-sulfate"] == ""
    assert info["multi_anion"] == {"raised": 2, "removed": 2}
    assert info["kept"]["records"] == len(RECORDS) - 4
    assert info["kept"]["net_charge_at_or_below_-2"] == 2


def test_cluster_skeletons_joins_a_protonation_series(staged):
    from experiments.themol_store import cluster_skeletons

    cluster = dict(
        zip(
            staged["themol_id"],
            cluster_skeletons(staged["skeleton"], progress=False),
            strict=True,
        )
    )
    assert cluster["u-acetic"] == cluster["u-acetate"]
    assert cluster["u-ethanol-a"] == cluster["u-ethanol-b"]
    assert cluster["u-ethanol-a"] != cluster["u-citrate"]


def test_diagnose_pairs_finds_the_duplicate(staged):
    from experiments.themol_store import diagnose_pairs, pair_summary

    pairs = diagnose_pairs(staged, workers=2)
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    ids = staged["themol_id"].to_numpy()
    assert {ids[pair["row_a"]], ids[pair["row_b"]]} == {"u-ethanol-a", "u-ethanol-b"}
    assert pair["same_molecule"] and pair["rmsd"] < 0.17 and pair["d_mbis"] >= 0
    summary = pair_summary(pairs)
    assert summary["n_copies"] == 1 and summary["same_molecule"] == 1


def test_prepare_themol_curated_builds_the_store(tmp_path):
    from experiments.themol_store import prepare_themol_curated

    stores = tmp_path / "stores"
    uncurated = _uncurated(stores)
    path = prepare_themol_curated(
        stores, uncurated_path=uncurated, n_shards=2, workers=2
    )
    staging = stores / "themol-staging"
    for name in ("parsed", "clusters", "curation", "pairs", "split"):
        assert (staging / f"{name}.parquet").exists()
    store = pd.read_parquet(path)
    assert len(store) == len(RECORDS) - 4
    assert {
        "split",
        "shard",
        "cluster",
        "collapse_key",
        "n_collapsed",
        "stereo_check",
    } <= set(store.columns)
    assert not {
        "skeleton",
        "charge_sum",
        "sensitive",
        "graph_smiles",
        "canonical_smiles",
    } & set(store.columns)
    pair = store.set_index("themol_id").loc[["u-ethanol-a", "u-ethanol-b"]]
    assert pair["split"].nunique() == 1 and pair["n_collapsed"].tolist() == [2, 2]
    assert pair["n_molecules"].tolist() == [1, 1]
    assert (stores / "themol" / "curation_summary.txt").exists()
    again = prepare_themol_curated(stores, uncurated_path=uncurated, n_shards=2)
    assert again == path


def test_build_parser_prepare_themol_curated_defaults():
    from experiments.cli import build_parser

    args = build_parser().parse_args(["prepare-themol-curated"])
    assert args.uncurated_path is None and args.source_dir is None
    assert args.n_shards == 50
    assert args.workers == 16
    assert args.limit is None
