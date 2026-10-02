"""DASH/QMugs and DASH/Extra as two training-ready stores.

The deposited DASH SDF holds two record schemas: 518,669 ``QMUGS500_*`` records, which
carry the properties QMugs computed for the same geometry, and 511,116 ``Rest_*``
records. They are used as separate datasets, built in stages into a shared staging store
and two training stores::

    dash-staging/
        dashMoleculesSDF_v2.sdf parsed.parquet          every record: the training
        columns plus subset, collapse_key,
                                canonical_smiles and graph_smiles
        record-fields.parquet   per record, the deposited properties other than the MBIS
        charges clusters.parquet        Butina cluster of every record, one pass over
        the whole corpus pairs.parquet           every pair of records of one structure
        and subset (dash_diagnostics) structures.parquet      records and rotatable
        bonds per structure and subset curation.parquet        why each record is
        removed (empty when kept), and its ESP score split.parquet           split and
        shard of every surviving record diagnostics.json
    dash-qmugs/, dash-extra/
        molecules.parquet       training-ready; read by runner.load_molecule_set and
        cv.load_shards curation_summary.txt, split_summary.txt, diagnostics.json,
        built-at-commit.txt

Clustering, curation and the split all precede the separation: one Butina pass over the
unique achiral graphs of the corpus, and one cluster-level train/test split and shard
assignment over the surviving records, balanced per subset, so that a cluster -- and
with it every structure deposited in both subsets -- has the same split and shard in
both stores. Curation is defined per subset (``dash_curation``) and runs before the
split, so the split is computed on the survivors.

Every stage writes its own file and is skipped when that file exists; deleting a stage's
file (and those of the stages after it) reruns it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")

SUBSETS = {"QMUGS500_": "QMugs", "Rest_": "Extra"}
STORE_OF = {"QMugs": "dash-qmugs", "Extra": "dash-extra"}
STAGING = "dash-staging"
ATOM_FIELDS = {
    "esp": ("DFT:ESP_AT_NUCLEI",),
    "lowdin": ("DFT:LOWDIN_CHARGES",),
    "mulliken": ("DFT:MULLIKEN_CHARGES",),
    "xtb": ("GFN2:MULLIKEN_CHARGES", "XTB_MulikenCharge"),
}
RECORD_FIELDS = {
    "e_dft": "DFT:TOTAL_ENERGY",
    "e_gfn2": "GFN2:TOTAL_ENERGY",
    "e_tpssh": "MBIS_Energy",
    "e_xtb": "XTB_Energy",
}
VECTOR_FIELDS = {"dipole_dft": "DFT:DIPOLE", "dipole_gfn2": "GFN2:DIPOLE"}
TRAIN_FRACTION = 0.9
N_SHARDS = 50
BUTINA_CUTOFF = 0.65


# --------------------------------------------------------------------------
# Parsing into the staging store


def subset_of(dash_id: str | None) -> str:
    """``QMugs`` or ``Extra`` from the ``DASH_IDX`` prefix; any other prefix is
    refused."""
    for prefix, name in SUBSETS.items():
        if dash_id is not None and dash_id.startswith(prefix):
            return name
    raise ValueError(f"DASH_IDX {dash_id!r} belongs to neither subset")


def record_fields(mol: Any) -> dict[str, Any]:
    """The deposited properties of one SDF record that the paper's figures and the
    curation read,
    besides the MBIS charges: per-atom lists (None when absent), energies in hartree and
    dipole
    vectors (NaN when absent). A per-atom list whose length is not the atom count is
    refused."""
    n = mol.GetNumAtoms()
    out: dict[str, Any] = {}
    for key, names in ATOM_FIELDS.items():
        name = next((k for k in names if mol.HasProp(k)), None)
        values = None
        if name is not None:
            values = [float(x) for x in mol.GetProp(name).split("|")]
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} values for {n} atoms")
        out[key] = values
    for key, name in RECORD_FIELDS.items():
        out[key] = float(mol.GetProp(name)) if mol.HasProp(name) else float("nan")
    for key, name in VECTOR_FIELDS.items():
        out[key] = (
            [float(x) for x in mol.GetProp(name).split("|")][:3]
            if mol.HasProp(name)
            else None
        )
    return out


def staging_columns(mol: Any, dash_id: str | None) -> dict[str, str]:
    """Subset label, structure key and its canonical SMILES, and the achiral graph
    SMILES."""
    from rdkit import Chem

    from experiments.collapse import collapse_key_and_smiles

    key, canonical = collapse_key_and_smiles(mol)
    return {
        "subset": subset_of(dash_id),
        "collapse_key": key,
        "canonical_smiles": canonical,
        "graph_smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
    }


def _fields_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            *[(k, pa.list_(pa.float32())) for k in ATOM_FIELDS],
            *[(k, pa.float64()) for k in RECORD_FIELDS],
            *[(k, pa.list_(pa.float64())) for k in VECTOR_FIELDS],
        ]
    )


def parse_staging(
    sdf_path: Path, parsed_path: Path, fields_path: Path, *, batch: int = 50_000
) -> int:
    """Stream ``sdf_path`` into ``parsed_path`` and ``fields_path``, row-aligned;
    return the row count.

    Each accepted record gives a training row (``prepare_dash._parse_one_record``)
    extended by ``staging_columns``, and a row of ``record_fields``. Both files are
    promoted from temporary names
    only when the whole pass succeeds."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem

    from experiments.geometry import arrow_fields
    from experiments.prepare_dash import _parse_one_record

    schema = pa.schema(
        [
            ("chembl_id", pa.string()),
            ("conf_id", pa.string()),
            ("dash_id", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
            *arrow_fields(),
            ("subset", pa.string()),
            ("collapse_key", pa.string()),
            ("canonical_smiles", pa.string()),
            ("graph_smiles", pa.string()),
        ]
    )
    fschema = _fields_schema()
    tmp_parsed = parsed_path.with_suffix(parsed_path.suffix + ".tmp")
    tmp_fields = fields_path.with_suffix(fields_path.suffix + ".tmp")
    writers: list[Any] = []
    rows, fields = [], []
    n_written = n_skipped = 0
    counters: dict[str, int] = {}

    def flush() -> None:
        nonlocal rows, fields, n_written
        if not rows:
            return
        if not writers:
            writers.extend(
                [
                    pq.ParquetWriter(tmp_parsed, schema),
                    pq.ParquetWriter(tmp_fields, fschema),
                ]
            )
        writers[0].write_table(pa.Table.from_pylist(rows, schema=schema))
        writers[1].write_table(pa.Table.from_pylist(fields, schema=fschema))
        n_written += len(rows)
        rows, fields = [], []

    try:
        with open(sdf_path, "rb") as fh:
            for mol in Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=False):
                staging: dict = {}
                row = _parse_one_record(
                    mol, dash_conf_counters=counters, staging=staging
                )
                if row is None:
                    n_skipped += 1
                    continue
                rows.append({**row, **staging["columns"]})
                fields.append(staging["fields"])
                if len(rows) >= batch:
                    flush()
            flush()
    finally:
        for writer in writers:
            writer.close()
    if writers:
        tmp_parsed.replace(parsed_path)
        tmp_fields.replace(fields_path)
    logger.info(
        "parsed %d records (%d skipped) from %s", n_written, n_skipped, sdf_path
    )
    return n_written


# --------------------------------------------------------------------------
# Clustering, before separation


def cluster_records(parsed: Any) -> np.ndarray:
    """Butina cluster of every row, from one fingerprint per unique achiral graph."""
    from experiments._chalcedon.butina_cluster import butina_cluster
    from experiments.data import blob_to_mol
    from experiments.prepare_dash import _achiral_fingerprints

    graph = parsed["graph_smiles"]
    first = ~graph.duplicated(keep="first")
    unique = graph[first].to_numpy()
    fingerprints = _achiral_fingerprints(
        [blob_to_mol(b) for b in parsed.loc[first, "mol"]]
    )
    ids = butina_cluster(fingerprints, cutoff=BUTINA_CUTOFF)
    cluster = graph.map(
        dict(zip(unique.tolist(), ids.tolist(), strict=True))
    ).to_numpy()
    logger.info("%d unique graphs in %d clusters", len(unique), len(np.unique(ids)))
    return cluster.astype(np.int64)


# --------------------------------------------------------------------------
# Splitting, before separation


def stratified_cluster_split(
    cluster_ids: np.ndarray, strata: np.ndarray, fractions: dict[str, float]
) -> dict[str, np.ndarray]:
    """Points into named groups, whole clusters at a time, balanced within every
    stratum.

    Clusters are placed from largest to smallest, as in ``greedy_cluster_split``, each
    into the group whose deficit is largest when measured, stratum by stratum, on the
    strata the cluster holds and
    weighted by how many of its points fall in each. Returns ascending point indices
    per group."""
    names = list(fractions)
    target = np.array([fractions[k] for k in names], dtype=np.float64)
    if abs(target.sum() - 1) > 1e-6 or (target <= 0).any():
        raise ValueError("fractions must be positive and sum to 1")
    cl_names, cl = np.unique(cluster_ids, return_inverse=True)
    st_names, st = np.unique(strata, return_inverse=True)
    size = np.zeros((len(cl_names), len(st_names)))
    np.add.at(size, (cl, st), 1.0)
    totals = size.sum(axis=0)
    order = np.argsort(-size.sum(axis=1), kind="stable")
    counts = np.zeros((len(names), len(st_names)))
    group_of = np.empty(len(order), dtype=np.int64)
    for c in order:
        weight = size[c] / size[c].sum()
        deficit = ((target[:, None] - counts / totals[None, :]) * weight[None, :]).sum(
            axis=1
        )
        g = int(np.argmax(deficit))
        group_of[c] = g
        counts[g] += size[c]
    point_group = group_of[cl]
    return {name: np.flatnonzero(point_group == i) for i, name in enumerate(names)}


def assign_split(
    parsed: Any, cluster: np.ndarray, kept: np.ndarray, *, n_shards: int = N_SHARDS
) -> Any:
    """``split`` (train/test) and ``shard`` (``s00``.. on train, ``test`` on test) per
    surviving row.

    Points are the distinct (``collapse_key``, ``subset``) pairs of the survivors,
    stratified by subset, so each store's fractions are measured in its own structures;
    a structure in both subsets
    counts once in each. Every structure lies in one cluster, which is asserted."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "key": parsed["collapse_key"].to_numpy(),
            "subset": parsed["subset"].to_numpy(),
            "cluster": cluster,
        }
    )[kept]
    spread = df.groupby("key")["cluster"].nunique()
    if (spread > 1).any():
        raise ValueError(
            f"{int((spread > 1).sum())} structure(s) span more than one cluster"
        )
    points = df.drop_duplicates(["key", "subset"]).reset_index(drop=True)
    parts = stratified_cluster_split(
        points["cluster"].to_numpy(),
        points["subset"].to_numpy(),
        {"train": TRAIN_FRACTION, "test": 1 - TRAIN_FRACTION},
    )
    split_of_cluster = {}
    for name, idx in parts.items():
        split_of_cluster.update(
            dict.fromkeys(points["cluster"].to_numpy()[idx].tolist(), name)
        )
    train = points[points["cluster"].map(split_of_cluster) == "train"].reset_index(
        drop=True
    )
    shard_parts = stratified_cluster_split(
        train["cluster"].to_numpy(),
        train["subset"].to_numpy(),
        {f"s{i:02d}": 1 / n_shards for i in range(n_shards)},
    )
    shard_of_cluster = {}
    for name, idx in shard_parts.items():
        if len(idx) == 0:
            raise ValueError(f"shard {name} got no molecules; lower n_shards")
        shard_of_cluster.update(
            dict.fromkeys(train["cluster"].to_numpy()[idx].tolist(), name)
        )
    out = pd.DataFrame({"row": np.flatnonzero(kept)})
    out["split"] = pd.Series(cluster[kept]).map(split_of_cluster).to_numpy()
    out["shard"] = np.where(
        out["split"] == "train",
        pd.Series(cluster[kept]).map(shard_of_cluster).to_numpy(),
        out["split"],
    )
    if out[["split", "shard"]].isna().any().any():
        raise ValueError("some surviving row(s) got no split or shard")
    return out


# --------------------------------------------------------------------------
# Separation and reports


def _collapse_counts(store: Any) -> Any:
    grouped = store.groupby("collapse_key")
    store["n_collapsed"] = grouped["collapse_key"].transform("size").astype("int32")
    store["n_molecules"] = grouped["dash_id"].transform("nunique").astype("int32")
    store["n_enantiomer_forms"] = (
        grouped["canonical_smiles"].transform("nunique").astype("int32")
    )
    return store


def separate(
    parsed: Any, cluster: np.ndarray, curation: Any, split: Any
) -> dict[str, Any]:
    """The two training stores, as data frames keyed by subset."""
    df = parsed.drop(columns=["graph_smiles"]).copy()
    df["cluster"] = cluster
    df["esp_score"] = curation["esp_score"].to_numpy()
    df = df.iloc[split["row"].to_numpy()].reset_index(drop=True)
    df["split"] = split["split"].to_numpy()
    df["shard"] = split["shard"].to_numpy()
    stores = {}
    for subset in ("QMugs", "Extra"):
        store = df[df["subset"] == subset].reset_index(drop=True)
        if subset == "Extra":
            store = store.drop(columns=["esp_score"])
        stores[subset] = _collapse_counts(store).drop(columns=["canonical_smiles"])
    return stores


def split_summary(store: Any) -> str:
    """Conformers, structures and fractions per split, and the spread of the shard
    sizes."""
    summary = store.groupby("split").agg(
        n_conformers=("mol", "size"), n_structures=("collapse_key", "nunique")
    )
    summary["fraction"] = summary["n_conformers"] / len(store)
    train = store[store["split"] == "train"]
    shards = train.groupby("shard")["collapse_key"].nunique()
    return (
        summary.reindex(["train", "test"]).to_string()
        + f"\n\n{len(shards)} shards of train structures: "
        f"min {shards.min()}, max {shards.max()}, mean {shards.mean():.1f}\n"
    )


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def diagnostics(parsed: Any, pairs: Any, structures: Any) -> dict:
    """Copies and pair counts per subset, as quoted in the manuscript."""
    from experiments.dash_diagnostics import COPY_RMSD

    out = {}
    rot = structures.set_index(["collapse_key", "subset"])["rotatable_bonds"]
    for subset in ("QMugs", "Extra"):
        p = pairs[pairs["subset"] == subset]
        copies = p["rmsd"] < COPY_RMSD
        keys = list(
            zip(
                parsed["collapse_key"].to_numpy()[p["row_a"]],
                [subset] * len(p),
                strict=True,
            )
        )
        rigid = rot.reindex(keys).to_numpy() <= 2
        records = int(np.sum(parsed["subset"] == subset))
        merged = records - int(_copy_components(p[copies]))
        out[subset] = {
            "records": records,
            "structures": int(np.sum(structures["subset"] == subset)),
            "pairs": len(p),
            "copy_pairs": int(copies.sum()),
            "copy_fraction": float(copies.mean()),
            "copy_fraction_rigid": float(copies[rigid].mean()),
            "distinct_conformations": merged,
        }
    return out


def _copy_components(copies: Any) -> int:
    """Records that merging copies removes: records in copy pairs minus their
    connected components."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in copies[["row_a", "row_b"]].itertuples(index=False):
        parent[find(int(a))] = find(int(b))
    nodes = list(parent)
    return len(nodes) - len({find(x) for x in nodes})


# --------------------------------------------------------------------------
# Orchestration


def prepare_dash_subsets(
    stores_root: Path,
    *,
    sdf_path: Path | None = None,
    n_shards: int = N_SHARDS,
    workers: int = 16,
    staging: str = STAGING,
) -> dict[str, Path]:
    """Build (or finish building) the staging store and the two training stores;
    idempotent per stage."""
    import pandas as pd

    from experiments.dash_curation import curate
    from experiments.dash_diagnostics import diagnose_pairs
    from experiments.prepare_dash import download_dash_sdf

    root = Path(stores_root) / staging
    root.mkdir(parents=True, exist_ok=True)
    if sdf_path is None:
        sdf_path = download_dash_sdf(root)
    elif not Path(sdf_path).exists():
        raise ValueError(f"sdf_path {sdf_path} does not exist")

    parsed_path, fields_path = root / "parsed.parquet", root / "record-fields.parquet"
    if not (parsed_path.exists() and fields_path.exists()):
        parse_staging(Path(sdf_path), parsed_path, fields_path)
    parsed = pd.read_parquet(parsed_path)
    fields = pd.read_parquet(fields_path)
    if len(parsed) != len(fields):
        raise RuntimeError(
            "parsed.parquet and record-fields.parquet are not row-aligned"
        )

    clusters_path = root / "clusters.parquet"
    if not clusters_path.exists():
        pd.DataFrame({"cluster": cluster_records(parsed)}).to_parquet(clusters_path)
    cluster = pd.read_parquet(clusters_path)["cluster"].to_numpy()

    pairs_path, structures_path = root / "pairs.parquet", root / "structures.parquet"
    if not (pairs_path.exists() and structures_path.exists()):
        pairs, structures = diagnose_pairs(parsed, fields, workers=workers)
        pairs.to_parquet(pairs_path)
        structures.to_parquet(structures_path)
    pairs, structures = pd.read_parquet(pairs_path), pd.read_parquet(structures_path)

    curation_path, info_path = root / "curation.parquet", root / "diagnostics.json"
    if not (curation_path.exists() and info_path.exists()):
        table, info = curate(parsed, pairs, fields["esp"], workers=workers)
        table.to_parquet(curation_path)
        info = {
            "commit": _commit(),
            "pairs": diagnostics(parsed, pairs, structures),
            "curation": info,
        }
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    curation = pd.read_parquet(curation_path)
    info = json.loads(info_path.read_text())

    split_path = root / "split.parquet"
    if not split_path.exists():
        kept = (curation["curation_step"] == "").to_numpy()
        assign_split(parsed, cluster, kept, n_shards=n_shards).to_parquet(split_path)
    split = pd.read_parquet(split_path)

    out = {}
    for subset, store in separate(parsed, cluster, curation, split).items():
        store_dir = Path(stores_root) / STORE_OF[subset]
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / "molecules.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        store.to_parquet(tmp)
        tmp.replace(path)
        summary = split_summary(store)
        (store_dir / "split_summary.txt").write_text(summary)
        mine = (parsed["subset"] == subset).to_numpy()
        steps = curation.loc[mine, "curation_step"].value_counts()
        (store_dir / "curation_summary.txt").write_text(
            f"{int(mine.sum()):,} records, {len(store):,} kept\n"
            + steps.drop(labels=[""], errors="ignore").to_string()
            + "\n"
        )
        (store_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "commit": info["commit"],
                    "pairs": info["pairs"][subset],
                    "curation": info["curation"][subset],
                },
                indent=2,
            )
            + "\n"
        )
        (store_dir / "built-at-commit.txt").write_text(_commit() + "\n")
        logger.info("wrote %s (%d rows):\n%s", path, len(store), summary)
        out[subset] = path
    return out
