"""SPICE 2.0.1 high-energy and low-energy conformations as two training-ready stores.

SPICE generated 50 conformations per molecule (Eastman et al., Sci. Data 10, 11 (2023)):
25 snapshots of molecular dynamics at 500 K and, from each, a low-energy conformation
made by five iterations of L-BFGS minimisation and 1 ps of dynamics at 100 K. The
generation script stores the snapshots as conformations 0-24 and the conformation
relaxed from snapshot ``c`` as ``c + 25``; ``SPICE-2.0.1.hdf5`` lists them in the order
of those indices sorted as text (0, 1, 10, 11, ..., 19, 2, 20, ...), so in a group of 50
the position ``k`` holds generation index ``TEXT_ORDER[k]``. The two kinds were sampled
differently and are used as separate datasets, built in stages into a shared staging
store and two training stores, as ``dash_subsets`` does for DASH::

    spice-staging/
        parsed.parquet          every conformation of the single molecules: the training
                                columns of ``prepare_spice`` plus the deposited group
                                size, generation index, sampling, structure key, its
                                canonical SMILES and the achiral graph SMILES
        record-fields.parquet   per conformation, the deposited total energy and forces
        clusters.parquet        Butina cluster of every row, one pass over all molecules
        pairs.parquet           random pairs of conformations of one structure and kind
        curation.parquet        why each conformation is removed (empty when kept)
        split.parquet           split and shard of every surviving conformation
        spice_summary.txt, diagnostics.json
    spice-high-energy/, spice-low-energy/
        molecules.parquet       training-ready; read by runner.load_molecule_set and
                                cv.load_shards
        curation_summary.txt, split_summary.txt, diagnostics.json, built-at-commit.txt

Curation has two channels, applied before the split:

* ``incomplete``: every conformation of a molecule whose group does not hold exactly 50
  conformations, or of which the parse could not keep all 50. SPICE's downloader keeps
  only the calculations that completed and drops conformations with a force component
  above 1 hartree/bohr, so a smaller group lost conformations to one cause or the other;
  the one group of 100 holds two generations of one molecule. In either case the
  text-order mapping does not identify the kinds.
* ``geometry``: the conformations whose geometry contradicts their own graph (the
  ``stretched_bond`` and ``close_contact`` flags of ``experiments.geometry``), as for
  DASH.

Clustering and the split precede the separation: one cluster-level train/test split and
shard assignment over the survivors, so that the two kinds of a molecule -- which share
its graph and therefore its cluster -- fall in the same split and shard of both stores.

Every stage writes its own file and is skipped when that file exists; deleting a stage's
file (and those of the stages after it) reruns it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")

N_CONFORMATIONS = 50
N_SNAPSHOTS = 25
TEXT_ORDER = np.array(sorted(range(N_CONFORMATIONS), key=str))
KINDS = ("high_energy", "low_energy")
STORE_OF = {"high_energy": "spice-high-energy", "low_energy": "spice-low-energy"}
STAGING = "spice-staging"
STEPS = ("incomplete", "geometry")
N_SHARDS = 50
PAIRS_PER_KIND = 100
PAIR_SEED = 0
HARTREE = 627.5095  # kcal/mol
TASK_GROUPS = 400
PARQUET_BATCH_SIZE = 50_000


# --------------------------------------------------------------------------
# Generation order


def generation_indices(n_deposited: int) -> np.ndarray | None:
    """Generation index of each position of a group of ``n_deposited`` conformations,
    or None when the group does not hold exactly 50."""
    return TEXT_ORDER.copy() if n_deposited == N_CONFORMATIONS else None


def sampling_of(generation: int) -> str:
    """``high_energy`` for a snapshot (generation 0-24), ``low_energy`` for a
    relaxed conformation (25-49), ``""`` when unknown (negative)."""
    if generation < 0:
        return ""
    return KINDS[0] if generation < N_SNAPSHOTS else KINDS[1]


# --------------------------------------------------------------------------
# Parsing into the staging store


def _staging_schema() -> Any:
    import pyarrow as pa

    from experiments.prepare_spice import _schema

    return pa.schema(
        [
            *_schema(),
            ("n_deposited", pa.int32()),
            ("generation", pa.int32()),
            ("sampling", pa.string()),
            ("collapse_key", pa.string()),
            ("canonical_smiles", pa.string()),
            ("graph_smiles", pa.string()),
        ]
    )


def _fields_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("energy", pa.float64()),
            ("max_force", pa.float64()),
            ("rms_force", pa.float64()),
        ]
    )


def staging_rows(spice_id: str, group: Any) -> tuple[list[dict], list[dict], Counter]:
    """The staging rows and record fields of one HDF5 group, and its summary counts.

    The rows are ``prepare_spice._parse_one_group``'s, extended with the deposited group
    size, the generation index and sampling (-1 and ``""`` unless the group holds 50),
    the structure key with its canonical SMILES, and the achiral graph SMILES. The
    fields are the total energy (hartree) and the largest component and RMS of the
    force (hartree/bohr), NaN when not deposited."""
    from rdkit import Chem

    from experiments.collapse import collapse_key_and_smiles
    from experiments.data import blob_to_mol
    from experiments.prepare_spice import _parse_one_group

    rows, counts = _parse_one_group(spice_id, group)
    if not rows:
        return [], [], counts
    n_deposited = int(group["conformations"].shape[0])
    generation = generation_indices(n_deposited)
    nan = np.full(n_deposited, np.nan)
    energy = group["dft_total_energy"][()] if "dft_total_energy" in group else nan
    if "dft_total_gradient" in group:
        grad = group["dft_total_gradient"][()].astype(np.float64)
        max_force = np.abs(grad).reshape(n_deposited, -1).max(axis=1)
        rms_force = np.sqrt((grad**2).sum(axis=2).mean(axis=1))
    else:
        max_force = rms_force = nan
    fields = []
    for row in rows:
        k = int(row["conf_id"].removeprefix("conf_"))
        mol = blob_to_mol(row["mol"])
        key, canonical = collapse_key_and_smiles(mol)
        g = int(generation[k]) if generation is not None else -1
        row.update(
            n_deposited=n_deposited,
            generation=g,
            sampling=sampling_of(g),
            collapse_key=key,
            canonical_smiles=canonical,
            graph_smiles=Chem.MolToSmiles(mol, isomericSmiles=False),
        )
        fields.append(
            {
                "energy": float(energy[k]),
                "max_force": float(max_force[k]),
                "rms_force": float(rms_force[k]),
            }
        )
    return rows, fields, counts


def _parse_chunk(
    hdf5_path: Path, names: list[str], parsed_out: Path, fields_out: Path
) -> Counter:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    schema, fschema = _staging_schema(), _fields_schema()
    totals: Counter = Counter()
    rows: list[dict] = []
    fields: list[dict] = []
    with (
        h5py.File(hdf5_path, "r") as f,
        pq.ParquetWriter(parsed_out, schema) as writer,
        pq.ParquetWriter(fields_out, fschema) as fwriter,
    ):

        def flush() -> None:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            fwriter.write_table(pa.Table.from_pylist(fields, schema=fschema))
            rows.clear()
            fields.clear()

        for name in names:
            r, fl, counts = staging_rows(name, f[name])
            totals.update(counts)
            rows.extend(r)
            fields.extend(fl)
            if len(rows) >= PARQUET_BATCH_SIZE:
                flush()
        if rows:
            flush()
    return totals


def parse_staging(
    hdf5_path: Path,
    parsed_path: Path,
    fields_path: Path,
    *,
    workers: int = 16,
    limit_groups: int | None = None,
) -> Counter:
    """Parse ``hdf5_path`` into ``parsed_path`` and ``fields_path``, row-aligned and in
    the file's group order, and return the summary counts of
    ``prepare_spice.spice_summary``.

    As in ``prepare_spice.parse_spice``, the groups are cut into contiguous blocks
    parsed by spawned processes into part files, which are concatenated in block order;
    both outputs are promoted from temporary names only when the whole parse succeeds.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import h5py
    import pyarrow.parquet as pq

    with h5py.File(hdf5_path, "r") as f:
        names = list(f.keys())
    if limit_groups is not None:
        names = names[:limit_groups]
    blocks = [list(b) for b in np.array_split(np.array(names, dtype=object), workers)]
    blocks = [b for b in blocks if b]

    parts_dir = parsed_path.parent / ".parts"
    parts_dir.mkdir(exist_ok=True)
    parts = [
        (parts_dir / f"parsed-{i:03d}.parquet", parts_dir / f"fields-{i:03d}.parquet")
        for i in range(len(blocks))
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(_parse_chunk, hdf5_path, block, p, fl)
            for block, (p, fl) in zip(blocks, parts, strict=True)
        ]
        results = [future.result() for future in futures]
    totals: Counter = Counter()
    for counts in results:
        totals.update(counts)

    for column, (out, schema) in enumerate(
        ((parsed_path, _staging_schema()), (fields_path, _fields_schema()))
    ):
        tmp = out.with_suffix(out.suffix + ".tmp")
        with pq.ParquetWriter(tmp, schema) as writer:
            for part in parts:
                source = pq.ParquetFile(part[column])
                for i in range(source.num_row_groups):
                    writer.write_table(source.read_row_group(i))
    for out in (parsed_path, fields_path):
        out.with_suffix(out.suffix + ".tmp").replace(out)
    for pair in parts:
        for part in pair:
            part.unlink()
    parts_dir.rmdir()
    return totals


# --------------------------------------------------------------------------
# Pair diagnostics


def draw_pairs(
    parsed: Any, *, pairs_per_kind: int = PAIRS_PER_KIND, seed: int = PAIR_SEED
) -> list[tuple[np.ndarray, str, np.ndarray, np.ndarray]]:
    """``(rows, kind, a, b)`` per structure and kind: at most ``pairs_per_kind`` pairs
    of its conformations, drawn uniformly without replacement (all of them when it has
    fewer), as row indices into ``parsed``. Conformations of unknown kind are left
    out."""
    rng = np.random.default_rng(seed)
    known = parsed["sampling"].to_numpy() != ""
    groups = parsed[known].groupby(["collapse_key", "sampling"], sort=True).indices
    index = np.flatnonzero(known)
    out = []
    for (_, kind), positions in groups.items():
        rows = index[positions]
        if len(rows) < 2:
            continue
        iu, ju = np.triu_indices(len(rows), 1)
        pick = np.sort(rng.choice(len(iu), min(pairs_per_kind, len(iu)), replace=False))
        out.append((rows, kind, rows[iu[pick]], rows[ju[pick]]))
    return out


PAIR_COLUMNS = (
    "row_a",
    "row_b",
    "sampling",
    "rmsd",
    "mirror",
    "d_mbis",
    "de",
    "heavy_atoms",
)


def _pair_task(items: list[tuple]) -> list[tuple]:
    """Pairs of a batch of ``(rows, kind, a, b, blobs, energies)``: heavy-atom RMSD
    (best of as deposited and mirror image), sorted MBIS charge discrepancy over
    equivalence classes, and absolute energy difference (kcal/mol)."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolAlign

    from experiments.collapse import aligned_values
    from experiments.dash_diagnostics import _mirror, _sorted_discrepancy
    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    out = []
    for rows, kind, a, b, blobs, energies in items:
        where = {int(r): i for i, r in enumerate(rows)}
        mols = [blob_to_mol(blob) for blob in blobs]
        heavy = [Chem.RemoveHs(m) for m in mols]
        mirrors = [_mirror(m) for m in heavy]
        for atom_mol in mols:
            for atom in atom_mol.GetAtoms():
                atom.SetDoubleProp("_index", float(atom.GetIdx()))
        for ra, rb in zip(a, b, strict=True):
            i, j = where[int(ra)], where[int(rb)]
            proper = rdMolAlign.GetBestRMS(
                Chem.Mol(heavy[i]), Chem.Mol(heavy[j]), maxMatches=10000
            )
            mirror = rdMolAlign.GetBestRMS(
                Chem.Mol(heavy[i]), Chem.Mol(mirrors[j]), maxMatches=10000
            )
            index, orbit = aligned_values([mols[i], mols[j]], "_index", stereo=True)
            match = index[1].astype(int)
            qa = np.array([x.GetDoubleProp("MBIScharge") for x in mols[i].GetAtoms()])
            qb = np.array([x.GetDoubleProp("MBIScharge") for x in mols[j].GetAtoms()])
            out.append(
                (
                    int(ra),
                    int(rb),
                    kind,
                    min(proper, mirror),
                    bool(mirror < proper),
                    _sorted_discrepancy(qa, qb[match], orbit),
                    abs(energies[i] - energies[j]) * HARTREE,
                    heavy[i].GetNumAtoms(),
                )
            )
    return out


def diagnose_pairs(
    parsed: Any,
    fields: Any,
    *,
    workers: int = 16,
    pairs_per_kind: int = PAIRS_PER_KIND,
    seed: int = PAIR_SEED,
) -> Any:
    """The pairs of ``draw_pairs`` with the columns ``PAIR_COLUMNS``. Workers are
    spawned; a calling script needs an ``if __name__ == "__main__"`` guard."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    energy = fields["energy"].to_numpy()
    blobs = parsed["mol"].to_numpy()
    items = []
    for _rows, kind, a, b in draw_pairs(
        parsed, pairs_per_kind=pairs_per_kind, seed=seed
    ):
        used = np.unique(np.concatenate([a, b]))
        items.append((used, kind, a, b, [blobs[r] for r in used], energy[used]))
    tasks = [items[k : k + TASK_GROUPS] for k in range(0, len(items), TASK_GROUPS)]
    rows: list[tuple] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for part in pool.map(_pair_task, tasks):
            rows += part
    pairs = pd.DataFrame(rows, columns=pd.Index(PAIR_COLUMNS))
    logger.info("compared %d pairs in %d structure-kind groups", len(pairs), len(items))
    return pairs


# --------------------------------------------------------------------------
# Curation


def curate(parsed: Any) -> tuple[Any, dict]:
    """``curation_step`` per row (``""`` when kept), and the counts per channel and
    kind."""
    import pandas as pd

    n_parsed = parsed.groupby("spice_id")["spice_id"].transform("size").to_numpy()
    incomplete = (parsed["n_deposited"].to_numpy() != N_CONFORMATIONS) | (
        n_parsed != N_CONFORMATIONS
    )
    geometry = (parsed["stretched_bond"] | parsed["close_contact"]).to_numpy()
    step = np.full(len(parsed), "", dtype=object)
    step[geometry] = "geometry"
    step[incomplete] = "incomplete"
    table = pd.DataFrame({"curation_step": step})
    sizes = parsed.groupby("spice_id")["n_deposited"].first()
    info: dict[str, Any] = {
        "molecules": len(sizes),
        "conformations": len(parsed),
        "incomplete": {
            "molecules": int(parsed.loc[incomplete, "spice_id"].nunique()),
            "conformations": int(incomplete.sum()),
            "groups_below_50": int((sizes < N_CONFORMATIONS).sum()),
            "groups_above_50": int((sizes > N_CONFORMATIONS).sum()),
        },
    }
    kinds = parsed["sampling"].to_numpy()
    for name, mask in (
        ("stretched_bond", parsed["stretched_bond"].to_numpy()),
        ("close_contact", parsed["close_contact"].to_numpy()),
        ("geometry", geometry),
    ):
        mask = mask & ~incomplete
        info[name] = {
            "molecules": int(parsed.loc[mask, "spice_id"].nunique()),
            "conformations": int(mask.sum()),
            **{k: int((mask & (kinds == k)).sum()) for k in KINDS},
        }
    kept = step == ""
    info["kept"] = {
        "molecules": int(parsed.loc[kept, "spice_id"].nunique()),
        "conformations": int(kept.sum()),
        **{k: int((kept & (kinds == k)).sum()) for k in KINDS},
    }
    return table, info


def sampling_check(parsed: Any, fields: Any) -> dict:
    """How often a low-energy conformation of a complete molecule lies below the
    snapshot it was relaxed from, the check that the text-order mapping is right."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "spice_id": parsed["spice_id"].to_numpy(),
            "generation": parsed["generation"].to_numpy(),
            "energy": fields["energy"].to_numpy(),
        }
    )
    df = df[df["generation"] >= 0]
    parents = df.assign(generation=df["generation"] + N_SNAPSHOTS)
    pairs = df.merge(
        parents, on=["spice_id", "generation"], suffixes=("_child", "_parent")
    )
    lower = pairs["energy_child"] < pairs["energy_parent"]
    return {
        "relaxed_with_parent": len(pairs),
        "lower_in_energy": float(lower.mean()),
    }


# --------------------------------------------------------------------------
# Separation and reports


def _collapse_counts(store: Any) -> Any:
    grouped = store.groupby("collapse_key")
    store["n_collapsed"] = grouped["collapse_key"].transform("size").astype("int32")
    store["n_molecules"] = grouped["spice_id"].transform("nunique").astype("int32")
    store["n_enantiomer_forms"] = (
        grouped["canonical_smiles"].transform("nunique").astype("int32")
    )
    return store


def assign_split(
    parsed: Any, cluster: np.ndarray, kept: np.ndarray, *, n_shards: int = N_SHARDS
) -> Any:
    """``dash_subsets.assign_split`` with the kind as the stratum, and a check that
    every molecule lies in one split and shard."""
    import pandas as pd

    from experiments.dash_subsets import assign_split as assign

    frame = pd.DataFrame(
        {
            "collapse_key": parsed["collapse_key"].to_numpy(),
            "subset": parsed["sampling"].to_numpy(),
        }
    )
    out = assign(frame, cluster, kept, n_shards=n_shards)
    ids = parsed["spice_id"].to_numpy()[out["row"].to_numpy()]
    spread = out.groupby(ids)["shard"].nunique()
    if (spread > 1).any():
        raise ValueError(f"{int((spread > 1).sum())} molecule(s) span two shards")
    return out


def separate(parsed: Any, cluster: np.ndarray, split: Any) -> dict[str, Any]:
    """The two training stores, as data frames keyed by kind."""
    df = parsed.drop(columns=["graph_smiles", "n_deposited"]).copy()
    df["cluster"] = cluster
    df = df.iloc[split["row"].to_numpy()].reset_index(drop=True)
    df["split"] = split["split"].to_numpy()
    df["shard"] = split["shard"].to_numpy()
    if (df["sampling"] == "").any():
        raise ValueError("a surviving conformation has no kind")
    stores = {}
    for kind in KINDS:
        store = df[df["sampling"] == kind].reset_index(drop=True)
        stores[kind] = _collapse_counts(store).drop(columns=["canonical_smiles"])
    if sum(len(s) for s in stores.values()) != len(df):
        raise RuntimeError("the two stores do not partition the survivors")
    return stores


def pair_summary(pairs: Any, *, min_heavy: int = 4) -> dict:
    """Pair counts and quantiles per kind, as quoted in the manuscript's SI."""
    out = {}
    for kind in KINDS:
        p = pairs[pairs["sampling"] == kind]
        big = p[p["heavy_atoms"] >= min_heavy]
        out[kind] = {
            "pairs": len(p),
            "pairs_at_least_4_heavy": len(big),
            "rmsd_below_0.17": float(np.mean(big["rmsd"] < 0.17)),
            "rmsd_median": float(big["rmsd"].median()),
            "d_mbis_median": float(p["d_mbis"].median()),
            "d_mbis_max": float(p["d_mbis"].max()),
            "de_median": float(p["de"].median()),
            "de_max": float(p["de"].max()),
        }
    return out


# --------------------------------------------------------------------------
# Orchestration


def prepare_spice_subsets(
    stores_root: Path,
    *,
    hdf5_path: Path | None = None,
    n_shards: int = N_SHARDS,
    workers: int = 16,
    staging: str = STAGING,
    limit_groups: int | None = None,
) -> dict[str, Path]:
    """Build (or finish building) the staging store and the two training stores;
    idempotent per stage. Without ``hdf5_path``, the file is downloaded into the
    staging store (37 GB)."""
    import pandas as pd

    from experiments.dash_subsets import _commit, cluster_records, split_summary
    from experiments.prepare_spice import download_spice_hdf5, spice_summary

    root = Path(stores_root) / staging
    root.mkdir(parents=True, exist_ok=True)

    parsed_path, fields_path = root / "parsed.parquet", root / "record-fields.parquet"
    if not (parsed_path.exists() and fields_path.exists()):
        if hdf5_path is None:
            hdf5_path = download_spice_hdf5(root)
        elif not Path(hdf5_path).exists():
            raise ValueError(f"hdf5_path {hdf5_path} does not exist")
        totals = parse_staging(
            Path(hdf5_path),
            parsed_path,
            fields_path,
            workers=workers,
            limit_groups=limit_groups,
        )
        (root / "spice_summary.txt").write_text(spice_summary(totals) + "\n")
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

    curation_path = root / "curation.parquet"
    if not curation_path.exists():
        table, _ = curate(parsed)
        table.to_parquet(curation_path)
    curation = pd.read_parquet(curation_path)
    kept = (curation["curation_step"] == "").to_numpy()

    pairs_path = root / "pairs.parquet"
    if not pairs_path.exists():
        rows = np.flatnonzero(kept)
        pairs = diagnose_pairs(
            parsed.iloc[rows].reset_index(drop=True),
            fields.iloc[rows].reset_index(drop=True),
            workers=workers,
        )
        _reindex_rows(pairs, rows).to_parquet(pairs_path)
    pairs = pd.read_parquet(pairs_path)

    info_path = root / "diagnostics.json"
    if not info_path.exists():
        _, cinfo = curate(parsed)
        info = {
            "commit": _commit(),
            "curation": cinfo,
            "sampling": sampling_check(parsed, fields),
            "pairs": pair_summary(pairs),
        }
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    info = json.loads(info_path.read_text())

    split_path = root / "split.parquet"
    if not split_path.exists():
        assign_split(parsed, cluster, kept, n_shards=n_shards).to_parquet(split_path)
    split = pd.read_parquet(split_path)

    out = {}
    for kind, store in separate(parsed, cluster, split).items():
        store_dir = Path(stores_root) / STORE_OF[kind]
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / "molecules.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        store.to_parquet(tmp)
        tmp.replace(path)
        summary = split_summary(store)
        (store_dir / "split_summary.txt").write_text(summary)
        c = info["curation"]
        (store_dir / "curation_summary.txt").write_text(
            f"{c['molecules']:,} single molecules, "
            f"{c['conformations']:,} conformations\n"
            f"incomplete (removed before the kinds are told apart): "
            f"{c['incomplete']['molecules']:,} molecules, "
            f"{c['incomplete']['conformations']:,} conformations\n"
            f"geometry, {kind}: {c['geometry'][kind]:,} conformations\n"
            f"kept, {kind}: {len(store):,} conformations of "
            f"{store['spice_id'].nunique():,} molecules\n"
        )
        (store_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "commit": info["commit"],
                    "curation": info["curation"],
                    "sampling": info["sampling"],
                    "pairs": info["pairs"][kind],
                },
                indent=2,
            )
            + "\n"
        )
        (store_dir / "built-at-commit.txt").write_text(_commit() + "\n")
        logger.info("wrote %s (%d rows):\n%s", path, len(store), summary)
        out[kind] = path
    return out


def _reindex_rows(pairs: Any, rows: np.ndarray) -> Any:
    """Map pair rows from positions in the kept subset back to rows of ``parsed``."""
    pairs = pairs.copy()
    pairs["row_a"] = rows[pairs["row_a"].to_numpy()]
    pairs["row_b"] = rows[pairs["row_b"].to_numpy()]
    return pairs
