"""MLPepper v1.1 in vacuum and in water as two training-ready stores.

MLPepper RECAP Optimized Fragments v1.1 (Adams et al.; Zenodo 10.5281/zenodo.15801339,
CC BY 4.0) is distributed as a view of a QCArchive singlepoint dataset: one SQLite file
holding 75,097 entries, each computed under two specifications on the same molecule
object, that is, on the same geometry: ``wb97x-d/def2-tzvpp`` in vacuum and the same
level of theory in ddX-PCM water. Records, entries and specifications are
zstd-compressed msgpack blobs (``records.record``, ``dataset_entries.entry``,
``dataset_specifications.specification``), joined by ``dataset_records``. The two phases
are used as separate datasets over the same entries, built in stages into a shared
staging store and two training stores, as ``spice_subsets`` does for SPICE's two kinds
of conformation::

    mlpepper-staging/
        parsed.parquet          one row per entry: the store Mol built from the mapped
                                SMILES and the shared geometry, the stereo check,
                                whether the deposited connectivity equals the SMILES
                                graph, the geometry columns, the structure key with its
                                canonical SMILES, the achiral graph SMILES, and the MBIS
                                charges of both phases (null for a record that did not
                                complete)
        record-fields.parquet   per entry and phase: status, total energy, ddX solvation
                                energy (water), SCF dipole, Mulliken and Lowdin charges
        clusters.parquet        Butina cluster of every entry, one pass over all graphs
        pairs.parquet           every pair of entries of one structure
        curation.parquet        why each entry is removed (empty when kept)
        split.parquet           split and shard of every surviving entry
        mlpepper_summary.txt, diagnostics.json
    mlpepper-vacuum/, mlpepper-water/
        molecules.parquet       training-ready, the Mol of each row carrying that
                                phase's charges as ``MBIScharge``
        curation_summary.txt, split_summary.txt, diagnostics.json, built-at-commit.txt

Curation has two channels, applied before the split, and an entry is removed from both
stores or from neither, since the phases share the geometry:

* ``incomplete``: an entry without a complete record in both phases. Five vacuum records
  ended in Psi4's ``could not converge MBIS`` after a converged SCF, so no unconverged
  charges were deposited for them.
* ``geometry``: the entries whose geometry contradicts their own graph (the
  ``stretched_bond`` and ``close_contact`` flags of ``experiments.geometry``), as for
  DASH. No MLPepper entry raises either flag.

Exact duplicate conformers (two entries of one molecule with a heavy-atom RMSD below
``COPY_RMSD``, 0.01 angstrom, the minimum of MLPepper's pair density) are kept: their
labels agree, so they only weight the minimum they share; ``diagnostics.json`` records
their share.

Clustering and the split precede the separation: one cluster-level train/test split and
shard assignment over the surviving entries, so that an entry has the same split and
shard in both stores. Every stage writes its own file and is skipped when that file
exists; deleting a stage's file (and those of the stages after it) reruns it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")

SQLITE_FILENAME = "MLPepper-RECAP-Optimized-Fragments-v1.1_singlepoint_view.sqlite"
DOWNLOAD_URL = (
    "https://zenodo.org/api/records/15801339/files/" + SQLITE_FILENAME + "/content"
)
EXPECTED_BYTES = 5_107_335_168
EXPECTED_MD5 = "8182af6fa1bff7e55b0d030a21bc7ac7"
CHUNK_SIZE = 1 << 20  # 1 MiB

PHASE_OF = {"wb97x-d/def2-tzvpp": "vacuum", "wb97x-d/def2-tzvpp/ddx-water": "water"}
PHASES = ("vacuum", "water")
STORE_OF = {"vacuum": "mlpepper-vacuum", "water": "mlpepper-water"}
STAGING = "mlpepper-staging"
STEPS = ("incomplete", "geometry")
N_SHARDS = 50
COPY_RMSD = 0.01  # angstrom
BOHR_TO_ANGSTROM = 0.529177210903  # CODATA 2018
HARTREE = 627.5095  # kcal/mol
ENTRIES_PER_TASK = 250
TASK_GROUPS = 400


# --------------------------------------------------------------------------
# Download


def download_mlpepper_sqlite(dest_dir: Path, *, url: str = DOWNLOAD_URL) -> Path:
    """Stream ``url`` to ``dest_dir / SQLITE_FILENAME``, verifying size and md5
    against the values Zenodo publishes; a correctly sized file already present is
    kept without hashing, as in ``prepare_spice.download_spice_hdf5``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / SQLITE_FILENAME
    if out_path.exists() and out_path.stat().st_size == EXPECTED_BYTES:
        logger.info("%s already present with the expected size; skipping", out_path)
        return out_path
    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            md5.update(chunk)
    if out_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"downloaded {out_path.name}: {out_path.stat().st_size} bytes, expected "
            f"{EXPECTED_BYTES}; download incomplete or corrupted"
        )
    if md5.hexdigest() != EXPECTED_MD5:
        out_path.unlink()
        raise ValueError(
            f"downloaded {out_path.name} md5 {md5.hexdigest()} != expected "
            f"{EXPECTED_MD5}; download corrupted"
        )
    logger.info("downloaded %s (md5 %s)", out_path, EXPECTED_MD5)
    return out_path


# --------------------------------------------------------------------------
# Reading the view


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)


def decode_blob(blob: bytes) -> Any:
    """A zstd-compressed msgpack blob of the view, decoded."""
    import msgpack
    import zstandard

    return msgpack.unpackb(
        zstandard.ZstdDecompressor().decompress(blob), raw=False, strict_map_key=False
    )


def entry_names(sqlite_path: Path) -> list[str]:
    with _connect(sqlite_path) as db:
        return [
            name
            for (name,) in db.execute("select name from dataset_entries order by name")
        ]


def _records_of(
    db: sqlite3.Connection, names: list[str]
) -> dict[str, dict[str, tuple]]:
    """``{entry: {phase: (status, decoded record)}}`` for ``names``."""
    marks = ",".join("?" * len(names))
    out: dict[str, dict[str, tuple]] = {n: {} for n in names}
    query = (
        "select r.entry_name, r.specification_name, s.status, s.record "
        "from dataset_records r join records s on s.id = r.record_id "
        f"where r.entry_name in ({marks})"
    )
    for entry, spec, status, blob in db.execute(query, names):
        out[entry][PHASE_OF[spec]] = (status, decode_blob(blob))
    return out


def _entries_of(db: sqlite3.Connection, names: list[str]) -> dict[str, dict]:
    marks = ",".join("?" * len(names))
    query = f"select name, entry from dataset_entries where name in ({marks})"
    return {name: decode_blob(blob) for name, blob in db.execute(query, names)}


# --------------------------------------------------------------------------
# Parsing into the staging store


def _staging_schema() -> Any:
    import pyarrow as pa

    from experiments.geometry import arrow_fields

    return pa.schema(
        [
            ("entry", pa.string()),
            ("molecule", pa.string()),
            ("smiles", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
            ("stereo_check", pa.string()),
            ("connectivity_agrees", pa.bool_()),
            *arrow_fields(),
            ("collapse_key", pa.string()),
            ("canonical_smiles", pa.string()),
            ("graph_smiles", pa.string()),
            *[(f"q_{p}", pa.list_(pa.float32())) for p in PHASES],
        ]
    )


def _fields_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            *[(f"status_{p}", pa.string()) for p in PHASES],
            *[(f"energy_{p}", pa.float64()) for p in PHASES],
            ("solvation_energy_water", pa.float64()),
            *[(f"dipole_{p}", pa.list_(pa.float64())) for p in PHASES],
            *[(f"mulliken_{p}", pa.list_(pa.float32())) for p in PHASES],
            *[(f"lowdin_{p}", pa.list_(pa.float32())) for p in PHASES],
        ]
    )


def _list(values: Any) -> list[float] | None:
    if values is None:
        return None
    return np.asarray(values, dtype=float).ravel().tolist()


def staging_row(
    entry: str, attributes: dict, records: dict[str, tuple]
) -> tuple[dict | None, dict, Counter]:
    """The staging row and record fields of one entry, and its summary counts.

    The Mol is built by ``prepare_themol.build_record_mol`` from the molecule object of
    a complete record (the phases share it), carrying that phase's charges; the charges
    of each complete phase are kept as list columns. An entry with no complete record is
    left out and counted."""
    from rdkit import Chem

    from experiments.collapse import collapse_key_and_smiles
    from experiments.geometry import geometry_record
    from experiments.prepare_themol import build_record_mol, store_blob

    counts: Counter = Counter(entries=1)
    fields: dict[str, Any] = {}
    charges: dict[str, list[float] | None] = {}
    complete = []
    for phase in PHASES:
        status, record = records.get(phase, ("missing", {}))
        prop = record.get("properties") or {}
        fields[f"status_{phase}"] = status
        fields[f"energy_{phase}"] = float(prop.get("return_energy") or np.nan)
        fields[f"dipole_{phase}"] = _list(prop.get("scf_dipole_moment"))
        fields[f"mulliken_{phase}"] = _list(prop.get("mulliken charges"))
        fields[f"lowdin_{phase}"] = _list(prop.get("lowdin charges"))
        q = _list(prop.get("mbis charges"))
        charges[phase] = q if status == "complete" and q is not None else None
        counts[f"status_{phase}::{status}"] += 1
        if charges[phase] is not None:
            complete.append(phase)
    fields["solvation_energy_water"] = float(
        (records.get("water", (None, {}))[1].get("properties") or {}).get(
            "dd solvation energy"
        )
        or np.nan
    )
    if not complete:
        counts["entries_without_charges"] += 1
        return None, fields, counts

    phase = "water" if "water" in complete else complete[0]
    molecule = records[phase][1]["molecule"]
    if len({records[p][1].get("molecule_id") for p in complete}) > 1:
        counts["entries_with_different_molecules"] += 1
    table = Chem.GetPeriodicTable()
    z = np.array([table.GetAtomicNumber(s) for s in molecule["symbols"]])
    xyz = (
        np.asarray(molecule["geometry"], dtype=float).reshape(-1, 3) * BOHR_TO_ANGSTROM
    )
    smiles = molecule["extras"]["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
    try:
        mol, verdict, stereo_counts = build_record_mol(
            z, xyz, np.asarray(charges[phase]), smiles
        )
    except ValueError as exc:
        logger.warning("%s: %s; skipping", entry, exc)
        counts["skipped"] += 1
        return None, fields, counts
    counts.update(stereo_counts)
    counts[f"verdict_{verdict}"] += 1
    graph = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    }
    deposited = {
        tuple(sorted((int(a), int(b))))
        for a, b, *_ in (molecule.get("connectivity") or [])
    }
    agrees = graph == deposited
    counts["connectivity_disagrees"] += not agrees
    net = float(Chem.GetFormalCharge(mol))
    counts["charge_disagrees"] += net != float(molecule.get("molecular_charge", 0.0))
    key, canonical = collapse_key_and_smiles(mol)
    row = {
        "entry": entry,
        "molecule": attributes.get("canonical_isomeric_smiles", entry),
        "smiles": smiles,
        "mol": store_blob(mol),
        "net_charge": net,
        "stereo_check": verdict,
        "connectivity_agrees": agrees,
        **geometry_record(mol),
        "collapse_key": key,
        "canonical_smiles": canonical,
        "graph_smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
        **{f"q_{p}": charges[p] for p in PHASES},
    }
    return row, fields, counts


def _parse_task(
    sqlite_path: Path, names: list[str]
) -> tuple[list[dict], list[dict], Counter]:
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    rows, fields = [], []
    totals: Counter = Counter()
    with _connect(sqlite_path) as db:
        records = _records_of(db, names)
        entries = _entries_of(db, names)
    for name in names:
        attributes = entries.get(name, {}).get("attributes") or {}
        row, f, counts = staging_row(name, attributes, records[name])
        totals.update(counts)
        if row is not None:
            rows.append(row)
            fields.append(f)
    return rows, fields, totals


def parse_staging(
    sqlite_path: Path,
    parsed_path: Path,
    fields_path: Path,
    *,
    workers: int = 16,
    limit_entries: int | None = None,
) -> Counter:
    """Parse the view into ``parsed_path`` and ``fields_path``, row-aligned and in
    entry-name order, and return the summary counts. Workers are spawned, each
    opening its own read-only connection."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    names = entry_names(sqlite_path)
    if limit_entries is not None:
        names = names[:limit_entries]
    chunks = [
        names[k : k + ENTRIES_PER_TASK] for k in range(0, len(names), ENTRIES_PER_TASK)
    ]
    rows: list[dict] = []
    fields: list[dict] = []
    totals: Counter = Counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for r, f, counts in pool.map(_parse_task, [sqlite_path] * len(chunks), chunks):
            rows.extend(r)
            fields.extend(f)
            totals.update(counts)
    for out, data, schema in (
        (parsed_path, rows, _staging_schema()),
        (fields_path, fields, _fields_schema()),
    ):
        tmp = out.with_suffix(out.suffix + ".tmp")
        pq.write_table(pa.Table.from_pylist(data, schema=schema), tmp)
        tmp.replace(out)
    return totals


def mlpepper_summary(totals: Counter) -> str:
    lines = [f"{totals['entries']:,} entries"]
    for phase in PHASES:
        statuses = {
            k.split("::", 1)[1]: v
            for k, v in totals.items()
            if k.startswith(f"status_{phase}::")
        }
        lines.append(
            f"{phase}: " + ", ".join(f"{s} {n:,}" for s, n in sorted(statuses.items()))
        )
    for key in (
        "entries_without_charges",
        "entries_with_different_molecules",
        "skipped",
        "connectivity_disagrees",
        "charge_disagrees",
    ):
        lines.append(f"{key}: {totals.get(key, 0):,}")
    verdicts = {k[8:]: v for k, v in totals.items() if k.startswith("verdict_")}
    lines.append(
        "stereo check: " + ", ".join(f"{k} {v:,}" for k, v in sorted(verdicts.items()))
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Curation


def curate(parsed: Any, fields: Any) -> tuple[Any, dict]:
    """``curation_step`` per entry (``""`` when kept), and the counts per channel."""
    import pandas as pd

    incomplete = np.zeros(len(parsed), dtype=bool)
    for phase in PHASES:
        incomplete |= (fields[f"status_{phase}"] != "complete").to_numpy()
        incomplete |= parsed[f"q_{phase}"].isna().to_numpy()
    geometry = (parsed["stretched_bond"] | parsed["close_contact"]).to_numpy()
    step = np.full(len(parsed), "", dtype=object)
    step[geometry] = "geometry"
    step[incomplete] = "incomplete"
    info: dict[str, Any] = {
        "entries": len(parsed),
        "molecules": int(parsed["molecule"].nunique()),
        "structures": int(parsed["collapse_key"].nunique()),
        "incomplete": int(incomplete.sum()),
        "geometry": int((geometry & ~incomplete).sum()),
        "stretched_bond": int(
            (parsed["stretched_bond"].to_numpy() & ~incomplete).sum()
        ),
        "close_contact": int((parsed["close_contact"].to_numpy() & ~incomplete).sum()),
    }
    kept = step == ""
    info["kept"] = {
        "entries": int(kept.sum()),
        "molecules": int(parsed.loc[kept, "molecule"].nunique()),
        "structures": int(parsed.loc[kept, "collapse_key"].nunique()),
    }
    return pd.DataFrame({"curation_step": step}), info


# --------------------------------------------------------------------------
# Pair diagnostics

PAIR_COLUMNS = (
    "row_a",
    "row_b",
    "rmsd",
    "mirror",
    *[f"d_mbis_{p}" for p in PHASES],
    *[f"de_{p}" for p in PHASES],
    "heavy_atoms",
)


def _pair_task(items: list[tuple]) -> list[tuple]:
    """All pairs of each group ``(rows, blobs, charges, energies)``: heavy-atom RMSD
    (best of as deposited and mirror image), sorted MBIS charge discrepancy over
    equivalence classes and absolute energy difference (kcal/mol) in each phase."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolAlign

    from experiments.collapse import aligned_values
    from experiments.dash_diagnostics import _mirror, _sorted_discrepancy
    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    out = []
    for rows, blobs, charges, energies in items:
        mols = [blob_to_mol(blob) for blob in blobs]
        heavy = [Chem.RemoveHs(m) for m in mols]
        mirrors = [_mirror(m) for m in heavy]
        for mol in mols:
            for atom in mol.GetAtoms():
                atom.SetDoubleProp("_index", float(atom.GetIdx()))
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                proper = rdMolAlign.GetBestRMS(
                    Chem.Mol(heavy[i]), Chem.Mol(heavy[j]), maxMatches=10000
                )
                mirror = rdMolAlign.GetBestRMS(
                    Chem.Mol(heavy[i]), Chem.Mol(mirrors[j]), maxMatches=10000
                )
                index, orbit = aligned_values([mols[i], mols[j]], "_index", stereo=True)
                match = index[1].astype(int)
                d = [
                    _sorted_discrepancy(charges[i][p], charges[j][p][match], orbit)
                    for p in PHASES
                ]
                de = [abs(energies[i][p] - energies[j][p]) * HARTREE for p in PHASES]
                out.append(
                    (
                        int(rows[i]),
                        int(rows[j]),
                        min(proper, mirror),
                        bool(mirror < proper),
                        *d,
                        *de,
                        heavy[i].GetNumAtoms(),
                    )
                )
    return out


def diagnose_pairs(parsed: Any, fields: Any, *, workers: int = 16) -> Any:
    """Every pair of entries sharing a ``collapse_key``, with the columns
    ``PAIR_COLUMNS``. Workers are spawned; a calling script needs an
    ``if __name__ == "__main__"`` guard."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    blobs = parsed["mol"].to_numpy()
    q = {p: parsed[f"q_{p}"].to_numpy() for p in PHASES}
    e = {p: fields[f"energy_{p}"].to_numpy() for p in PHASES}
    items = []
    for rows in parsed.groupby("collapse_key", sort=False).indices.values():
        if len(rows) < 2:
            continue
        items.append(
            (
                rows,
                [blobs[r] for r in rows],
                [{p: np.asarray(q[p][r], dtype=float) for p in PHASES} for r in rows],
                [{p: float(e[p][r]) for p in PHASES} for r in rows],
            )
        )
    tasks = [items[k : k + TASK_GROUPS] for k in range(0, len(items), TASK_GROUPS)]
    rows_out: list[tuple] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for part in pool.map(_pair_task, tasks):
            rows_out += part
    pairs = pd.DataFrame(rows_out, columns=pd.Index(PAIR_COLUMNS))
    logger.info("compared %d pairs in %d structures", len(pairs), len(items))
    return pairs


def pair_summary(pairs: Any, *, copy_rmsd: float = COPY_RMSD) -> dict:
    """Pair counts and quantiles, for copies (below ``copy_rmsd``) and distinct
    conformers, per phase."""
    copies = pairs["rmsd"].to_numpy() < copy_rmsd
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "copy_rmsd": copy_rmsd,
        "n_copies": int(copies.sum()),
        "copy_fraction": float(copies.mean()) if len(pairs) else float("nan"),
    }
    for name, mask in (("copies", copies), ("distinct", ~copies)):
        sub = pairs[mask]
        out[name] = {}
        for p in PHASES:
            out[name][p] = {
                "d_mbis_median": float(sub[f"d_mbis_{p}"].median()),
                "d_mbis_max": float(sub[f"d_mbis_{p}"].max()),
                "de_median": float(sub[f"de_{p}"].median()),
                "de_max": float(sub[f"de_{p}"].max()),
            }
    return out


# --------------------------------------------------------------------------
# Split, separation and reports


def assign_split(
    parsed: Any, cluster: np.ndarray, kept: np.ndarray, *, n_shards: int = N_SHARDS
) -> Any:
    """``dash_subsets.assign_split`` over the kept entries, with a single stratum."""
    import pandas as pd

    from experiments.dash_subsets import assign_split as assign

    frame = pd.DataFrame(
        {"collapse_key": parsed["collapse_key"].to_numpy(), "subset": "all"}
    )
    return assign(frame, cluster, kept, n_shards=n_shards)


def _collapse_counts(store: Any) -> Any:
    grouped = store.groupby("collapse_key")
    store["n_collapsed"] = grouped["collapse_key"].transform("size").astype("int32")
    store["n_molecules"] = grouped["molecule"].transform("nunique").astype("int32")
    store["n_enantiomer_forms"] = (
        grouped["canonical_smiles"].transform("nunique").astype("int32")
    )
    return store


def _with_charges(blob: bytes, charges: Any) -> bytes:
    from experiments.data import blob_to_mol, mol_to_blob

    mol = blob_to_mol(blob)
    for atom, q in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", float(q))
    return mol_to_blob(mol)


def separate(parsed: Any, cluster: np.ndarray, split: Any) -> dict[str, Any]:
    """The two training stores, as data frames keyed by phase: the same surviving
    entries, each store's Mol carrying that phase's charges."""
    df = parsed.drop(columns=["graph_smiles"]).copy()
    df["cluster"] = cluster
    df = df.iloc[split["row"].to_numpy()].reset_index(drop=True)
    df["split"] = split["split"].to_numpy()
    df["shard"] = split["shard"].to_numpy()
    stores = {}
    for phase in PHASES:
        store = df.copy()
        store["mol"] = [
            _with_charges(blob, q)
            for blob, q in zip(store["mol"], store[f"q_{phase}"], strict=True)
        ]
        store = store.drop(columns=[f"q_{p}" for p in PHASES])
        stores[phase] = _collapse_counts(store).drop(columns=["canonical_smiles"])
    return stores


def split_summary(store: Any) -> str:
    from experiments.dash_subsets import split_summary as summary

    return summary(store)


# --------------------------------------------------------------------------
# Orchestration


def prepare_mlpepper_stores(
    stores_root: Path,
    *,
    sqlite_path: Path | None = None,
    n_shards: int = N_SHARDS,
    workers: int = 16,
    staging: str = STAGING,
    limit_entries: int | None = None,
) -> dict[str, Path]:
    """Build (or finish building) the staging store and the two training stores;
    idempotent per stage. Without ``sqlite_path``, the view is downloaded into the
    staging store (5.1 GB)."""
    import pandas as pd

    from experiments.dash_subsets import _commit, cluster_records

    root = Path(stores_root) / staging
    root.mkdir(parents=True, exist_ok=True)

    parsed_path, fields_path = root / "parsed.parquet", root / "record-fields.parquet"
    if not (parsed_path.exists() and fields_path.exists()):
        if sqlite_path is None:
            sqlite_path = download_mlpepper_sqlite(root)
        elif not Path(sqlite_path).exists():
            raise ValueError(f"sqlite_path {sqlite_path} does not exist")
        totals = parse_staging(
            Path(sqlite_path),
            parsed_path,
            fields_path,
            workers=workers,
            limit_entries=limit_entries,
        )
        (root / "mlpepper_summary.txt").write_text(mlpepper_summary(totals) + "\n")
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
        table, _ = curate(parsed, fields)
        table.to_parquet(curation_path)
    kept = (pd.read_parquet(curation_path)["curation_step"] == "").to_numpy()

    pairs_path = root / "pairs.parquet"
    if not pairs_path.exists():
        rows = np.flatnonzero(kept)
        pairs = diagnose_pairs(
            parsed.iloc[rows].reset_index(drop=True),
            fields.iloc[rows].reset_index(drop=True),
            workers=workers,
        )
        pairs["row_a"] = rows[pairs["row_a"].to_numpy()]
        pairs["row_b"] = rows[pairs["row_b"].to_numpy()]
        pairs.to_parquet(pairs_path)
    pairs = pd.read_parquet(pairs_path)

    info_path = root / "diagnostics.json"
    if not info_path.exists():
        _, cinfo = curate(parsed, fields)
        info = {"commit": _commit(), "curation": cinfo, "pairs": pair_summary(pairs)}
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    info = json.loads(info_path.read_text())

    split_path = root / "split.parquet"
    if not split_path.exists():
        assign_split(parsed, cluster, kept, n_shards=n_shards).to_parquet(split_path)
    split = pd.read_parquet(split_path)

    out = {}
    for phase, store in separate(parsed, cluster, split).items():
        store_dir = Path(stores_root) / STORE_OF[phase]
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / "molecules.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        store.to_parquet(tmp)
        tmp.replace(path)
        summary = split_summary(store)
        (store_dir / "split_summary.txt").write_text(summary)
        c = info["curation"]
        (store_dir / "curation_summary.txt").write_text(
            f"{c['entries']:,} entries of {c['molecules']:,} molecules, "
            f"{c['structures']:,} structures\n"
            f"incomplete (a record of either phase did not complete): "
            f"{c['incomplete']:,}\n"
            f"geometry: {c['geometry']:,}\n"
            f"kept, {phase}: {len(store):,} entries of "
            f"{store['molecule'].nunique():,} molecules\n"
        )
        (store_dir / "diagnostics.json").write_text(json.dumps(info, indent=2) + "\n")
        (store_dir / "built-at-commit.txt").write_text(_commit() + "\n")
        logger.info("wrote %s (%d rows):\n%s", path, len(store), summary)
        out[phase] = path
    return out
