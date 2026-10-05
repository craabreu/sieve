"""THEMol's MBIS subset, curated and split, as a training-ready store.

``prepare_themol`` parses the eight HDF5 files of THEMol's MBIS subset (3,082,151
records, one geometry each) into an uncurated store, ``themol-mbis``, with the stereo
check, the structure key and the geometry columns. This module takes that parse (or
makes one) through the stages of the other datasets' builders, into a staging store and
one training store::

    themol-staging/
        parsed.parquet          the uncurated parse plus, from each stored Mol: the sum
                                of its charges, whether it contains B, Si or P, its
                                canonical isomeric SMILES, the achiral graph SMILES and
                                the heavy-atom skeleton SMILES
        clusters.parquet        Butina cluster of every record, from one fingerprint per
                                unique skeleton
        curation.parquet        why each record is removed (empty when kept)
        pairs.parquet           every pair of records of one structure
        split.parquet           split and shard of every surviving record
        diagnostics.json
    themol/
        molecules.parquet       training-ready; read by runner.load_molecule_set and
                                cv.load_shards
        curation_summary.txt, split_summary.txt, diagnostics.json, built-at-commit.txt

Curation has three channels, in this order of precedence, all decided in the
manuscript's SI (sieve_paper, "Diagnostics and Curation of the Other Datasets", THEMol):

* ``charge_sum``: a record whose MBIS charges do not add up to its net charge within
  ``CHARGE_SUM_TOLERANCE``. One record fails, a neutral molecule whose charges add up to
  -0.84 e; every other record agrees to within 3.3e-3 e.
* ``multi_anion``: a record with a net charge of ``MAX_NET_CHARGE_ANY`` (-3) or below,
  or of ``MAX_NET_CHARGE_ANION`` (-2) if it contains one of ``SENSITIVE_ELEMENTS`` (B,
  Si, P). THEMol computed its charges in vacuum with def2-TZVPD, a basis with diffuse
  functions; within THEMol's own protonation series, the charge of a P, Si or B centre
  drops by 0.09-0.17 e on the first deprotonation but by 0.3-0.4 e on the second,
  whereas C, N and S change smoothly, and on the structures shared with MLPepper the
  phosphorus dianions differ from MLPepper's vacuum charges by 0.3-1.1 e while the
  dicarboxylates agree to within 0.15 e.
* ``geometry``: the records whose geometry contradicts their own graph (the
  ``stretched_bond`` and ``close_contact`` flags of ``experiments.geometry``), as for
  DASH; mostly B-O and C-Si bonds that fell apart, and intramolecular O..Si and B..N
  contacts the graph does not bond.

Duplicates (one molecule under several uuids) and copies (records of one structure with
a heavy-atom RMSD below ``COPY_RMSD``, 0.17 angstrom, the minimum of THEMol's pair
density, as for DASH) are kept: their labels agree.

The clusters come from a Butina pass over unique heavy-atom skeletons -- the graph
without hydrogens, formal charges, bond orders, aromaticity and stereochemistry -- so
that a protonation series, which THEMol enumerates, shares a cluster by construction.
Two million skeletons take a few hours with the vendored implementation (quadratic;
measured 138 s for 200k on 32 cores), which is why every stage writes its own file and
is skipped when that file exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")

STAGING = "themol-staging"
STORE = "themol"
UNCURATED_STORE = "themol-mbis"
STEPS = ("charge_sum", "multi_anion", "geometry")
CHARGE_SUM_TOLERANCE = 0.01  # e
MAX_NET_CHARGE_ANION = -2  # removed from here down when sensitive
MAX_NET_CHARGE_ANY = -3  # removed from here down whatever the elements
SENSITIVE_ELEMENTS = (5, 14, 15)  # B, Si, P
COPY_RMSD = 0.17  # angstrom
BUTINA_CUTOFF = 0.65
N_SHARDS = 50
BATCH_ROWS = 50_000
TASK_ROWS = 2_000
TASK_GROUPS = 400
COUNT_COLUMNS = ("n_collapsed", "n_molecules", "n_enantiomer_forms")
AUGMENTED = ("charge_sum", "sensitive", "canonical_smiles", "graph_smiles", "skeleton")


# --------------------------------------------------------------------------
# Skeletons


def skeleton_mol(mol: Any) -> Any:
    """``mol`` without hydrogens, formal charges, bond orders, aromaticity, isotopes,
    atom maps and stereochemistry: what a protonation series has in common."""
    from rdkit import Chem

    rw = Chem.RWMol(Chem.RemoveAllHs(mol, sanitize=False))
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetIsAromatic(False)
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return rw.GetMol()


def skeleton_smiles(mol: Any) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(skeleton_mol(mol))


def skeleton_fingerprints(smiles: list[str], *, radius: int = 2, n_bits: int = 2048):
    """Dense Morgan fingerprints of skeleton SMILES, which parse only unsanitized."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        out[i] = generator.GetFingerprintAsNumPy(mol)
    return out


# --------------------------------------------------------------------------
# Parsing into the staging store


def augmented_record(blob: bytes) -> dict[str, Any]:
    """The ``AUGMENTED`` columns of one stored Mol."""
    from rdkit import Chem

    from experiments.collapse import collapse_key_and_smiles
    from experiments.data import blob_to_mol

    mol = blob_to_mol(blob)
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    _, canonical = collapse_key_and_smiles(mol)
    return {
        "charge_sum": float(
            sum(atom.GetDoubleProp("MBIScharge") for atom in mol.GetAtoms())
        ),
        "sensitive": any(v in SENSITIVE_ELEMENTS for v in z),
        "canonical_smiles": canonical,
        "graph_smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
        "skeleton": skeleton_smiles(mol),
    }


def _augment_task(blobs: list[bytes]) -> list[dict[str, Any]]:
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    return [augmented_record(b) for b in blobs]


def _augmented_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("charge_sum", pa.float64()),
            ("sensitive", pa.bool_()),
            ("canonical_smiles", pa.string()),
            ("graph_smiles", pa.string()),
            ("skeleton", pa.string()),
        ]
    )


def parse_staging(
    uncurated_path: Path,
    parsed_path: Path,
    *,
    workers: int = 16,
    limit: int | None = None,
) -> int:
    """Write ``parsed_path``: the rows of the uncurated store at ``uncurated_path``
    (without the collapse counts, which are recomputed at the end) plus the
    ``AUGMENTED`` columns, streamed in batches of ``BATCH_ROWS``. Returns the number of
    rows. Workers are spawned."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(uncurated_path)
    keep = [n for n in source.schema_arrow.names if n not in COUNT_COLUMNS]
    schema = pa.schema(
        [source.schema_arrow.field(n) for n in keep] + list(_augmented_schema())
    )
    tmp = parsed_path.with_suffix(parsed_path.suffix + ".tmp")
    rows = 0
    context = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool,
        pq.ParquetWriter(tmp, schema) as writer,
    ):
        for batch in source.iter_batches(batch_size=BATCH_ROWS, columns=keep):
            if limit is not None and rows >= limit:
                break
            if limit is not None and rows + batch.num_rows > limit:
                batch = batch.slice(0, limit - rows)
            blobs = batch.column("mol").to_pylist()
            tasks = [blobs[k : k + TASK_ROWS] for k in range(0, len(blobs), TASK_ROWS)]
            augmented = [r for part in pool.map(_augment_task, tasks) for r in part]
            extra = pa.Table.from_pylist(augmented, schema=_augmented_schema())
            table = pa.Table.from_batches([batch])
            for name in extra.column_names:
                table = table.append_column(name, extra.column(name))
            writer.write_table(table)
            rows += batch.num_rows
            logger.info("augmented %d records", rows)
    tmp.replace(parsed_path)
    return rows


# --------------------------------------------------------------------------
# Clustering


def cluster_skeletons(
    skeletons: Any, *, cutoff: float = BUTINA_CUTOFF, progress: bool = True
) -> np.ndarray:
    """Butina cluster of every record, from one fingerprint per unique skeleton."""
    import pandas as pd

    from experiments._chalcedon.butina_cluster import butina_cluster

    skeletons = pd.Series(np.asarray(skeletons))
    first = ~skeletons.duplicated(keep="first")
    unique = skeletons[first].to_numpy()
    logger.info("fingerprinting %d unique skeletons", len(unique))
    fingerprints = skeleton_fingerprints(unique.tolist())
    logger.info("clustering %d skeletons at %.2f", len(unique), cutoff)
    ids = butina_cluster(fingerprints, cutoff=cutoff, progress=progress)
    cluster = skeletons.map(dict(zip(unique.tolist(), ids.tolist(), strict=True)))
    logger.info("%d unique skeletons in %d clusters", len(unique), len(np.unique(ids)))
    return cluster.to_numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Curation


def curate(parsed: Any) -> tuple[Any, dict]:
    """``curation_step`` per record (``""`` when kept), and the counts per channel."""
    import pandas as pd

    net = parsed["net_charge"].to_numpy()
    flags = {
        "charge_sum": np.abs(parsed["charge_sum"].to_numpy() - net)
        > CHARGE_SUM_TOLERANCE,
        "multi_anion": (net <= MAX_NET_CHARGE_ANY)
        | ((net <= MAX_NET_CHARGE_ANION) & parsed["sensitive"].to_numpy()),
        "geometry": (parsed["stretched_bond"] | parsed["close_contact"]).to_numpy(),
    }
    step = np.full(len(parsed), "", dtype=object)
    for channel in reversed(STEPS):
        step[flags[channel]] = channel
    info: dict[str, Any] = {
        "records": len(parsed),
        "molecules": int(parsed["canonical_smiles"].nunique()),
        "structures": int(parsed["collapse_key"].nunique()),
    }
    for channel in STEPS:
        info[channel] = {
            "raised": int(flags[channel].sum()),
            "removed": int((step == channel).sum()),
        }
    kept = step == ""
    info["kept"] = {
        "records": int(kept.sum()),
        "molecules": int(parsed.loc[kept, "canonical_smiles"].nunique()),
        "structures": int(parsed.loc[kept, "collapse_key"].nunique()),
        "net_charge_at_or_below_-2": int((kept & (net <= MAX_NET_CHARGE_ANION)).sum()),
    }
    return pd.DataFrame({"curation_step": step}), info


# --------------------------------------------------------------------------
# Pair diagnostics

PAIR_COLUMNS = (
    "row_a",
    "row_b",
    "same_molecule",
    "rmsd",
    "mirror",
    "d_mbis",
    "heavy_atoms",
)


def _pair_task(items: list[tuple]) -> list[tuple]:
    """All pairs of each group ``(rows, blobs, molecules)``: heavy-atom RMSD (best of as
    deposited and mirror image) and sorted MBIS charge discrepancy over equivalence
    classes."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolAlign

    from experiments.collapse import aligned_values
    from experiments.dash_diagnostics import _mirror, _sorted_discrepancy
    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    out = []
    for rows, blobs, molecules in items:
        mols = [blob_to_mol(blob) for blob in blobs]
        charges = [
            np.array([a.GetDoubleProp("MBIScharge") for a in m.GetAtoms()])
            for m in mols
        ]
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
                out.append(
                    (
                        int(rows[i]),
                        int(rows[j]),
                        molecules[i] == molecules[j],
                        min(proper, mirror),
                        bool(mirror < proper),
                        _sorted_discrepancy(charges[i], charges[j][match], orbit),
                        heavy[i].GetNumAtoms(),
                    )
                )
    return out


def diagnose_pairs(parsed: Any, *, workers: int = 16) -> Any:
    """Every pair of records sharing a ``collapse_key``, with the columns
    ``PAIR_COLUMNS``; ``same_molecule`` compares canonical isomeric SMILES. Workers
    are spawned; a calling script needs an ``if __name__ == "__main__"`` guard."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    blobs = parsed["mol"].to_numpy()
    molecules = parsed["canonical_smiles"].to_numpy()
    items = []
    for rows in parsed.groupby("collapse_key", sort=False).indices.values():
        if len(rows) < 2:
            continue
        items.append((rows, [blobs[r] for r in rows], [molecules[r] for r in rows]))
    tasks = [items[k : k + TASK_GROUPS] for k in range(0, len(items), TASK_GROUPS)]
    rows_out: list[tuple] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for part in pool.map(_pair_task, tasks):
            rows_out += part
    pairs = pd.DataFrame(rows_out, columns=pd.Index(PAIR_COLUMNS))
    logger.info("compared %d pairs in %d structures", len(pairs), len(items))
    return pairs


def pair_summary(
    pairs: Any, *, copy_rmsd: float = COPY_RMSD, min_heavy: int = 4
) -> dict:
    copies = pairs["rmsd"].to_numpy() < copy_rmsd
    big = pairs["heavy_atoms"].to_numpy() >= min_heavy
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "same_molecule": int(pairs["same_molecule"].sum()),
        "pairs_at_least_4_heavy": int(big.sum()),
        "copy_rmsd": copy_rmsd,
        "n_copies": int(copies.sum()),
        "copy_fraction": float(copies.mean()) if len(pairs) else float("nan"),
    }
    for name, mask in (("copies", copies), ("distinct", ~copies)):
        sub = pairs[mask]
        out[name] = {
            "pairs": int(mask.sum()),
            "same_molecule": int(sub["same_molecule"].sum()),
            "d_mbis_median": float(sub["d_mbis"].median()),
            "d_mbis_99.9th": float(sub["d_mbis"].quantile(0.999))
            if len(sub)
            else float("nan"),
            "d_mbis_max": float(sub["d_mbis"].max()),
        }
    return out


# --------------------------------------------------------------------------
# Split and store


def assign_split(
    parsed: Any, cluster: np.ndarray, kept: np.ndarray, *, n_shards: int = N_SHARDS
) -> Any:
    """``dash_subsets.assign_split`` over the kept records, with a single stratum."""
    import pandas as pd

    from experiments.dash_subsets import assign_split as assign

    frame = pd.DataFrame(
        {"collapse_key": parsed["collapse_key"].to_numpy(), "subset": "all"}
    )
    return assign(frame, cluster, kept, n_shards=n_shards)


def training_store(parsed: Any, cluster: np.ndarray, split: Any) -> Any:
    """The surviving records with ``cluster``, ``split``, ``shard`` and the collapse
    counts, without the staging-only columns."""
    df = parsed.drop(
        columns=["graph_smiles", "skeleton", "charge_sum", "sensitive"]
    ).copy()
    df["cluster"] = cluster
    df = df.iloc[split["row"].to_numpy()].reset_index(drop=True)
    df["split"] = split["split"].to_numpy()
    df["shard"] = split["shard"].to_numpy()
    grouped = df.groupby("collapse_key")
    df["n_collapsed"] = grouped["collapse_key"].transform("size").astype("int32")
    df["n_molecules"] = grouped["canonical_smiles"].transform("nunique").astype("int32")
    df["n_enantiomer_forms"] = df["n_molecules"]
    return df.drop(columns=["canonical_smiles"])


# --------------------------------------------------------------------------
# Orchestration


def prepare_themol_curated(
    stores_root: Path,
    *,
    uncurated_path: Path | None = None,
    source_dir: Path | None = None,
    n_shards: int = N_SHARDS,
    workers: int = 16,
    staging: str = STAGING,
    store: str = STORE,
    limit: int | None = None,
) -> Path:
    """Build (or finish building) the staging store and the training store;
    idempotent per stage. The uncurated parse is read from ``uncurated_path``
    (default: the ``themol-mbis`` store under ``stores_root``) and, when that does not
    exist, made first by ``prepare_themol.parse_themol`` from the HDF5 files of
    ``source_dir``, downloaded when not given."""
    import pandas as pd

    from experiments.dash_subsets import _commit, split_summary

    root = Path(stores_root) / staging
    root.mkdir(parents=True, exist_ok=True)

    parsed_path = root / "parsed.parquet"
    if not parsed_path.exists():
        if uncurated_path is None:
            uncurated_path = Path(stores_root) / UNCURATED_STORE / "molecules.parquet"
        uncurated_path = Path(uncurated_path)
        if not uncurated_path.exists():
            from experiments.prepare_themol import download_themol, parse_themol

            uncurated_path.parent.mkdir(parents=True, exist_ok=True)
            if source_dir is None:
                source_dir = download_themol(uncurated_path.parent)
            parse_themol(Path(source_dir), uncurated_path, workers=min(workers, 8))
        n = parse_staging(uncurated_path, parsed_path, workers=workers, limit=limit)
        logger.info("staged %d records", n)
    parsed = pd.read_parquet(parsed_path)

    clusters_path = root / "clusters.parquet"
    if not clusters_path.exists():
        cluster = cluster_skeletons(parsed["skeleton"].to_numpy())
        pd.DataFrame({"cluster": cluster}).to_parquet(clusters_path)
    cluster = pd.read_parquet(clusters_path)["cluster"].to_numpy()

    curation_path = root / "curation.parquet"
    if not curation_path.exists():
        table, _ = curate(parsed)
        table.to_parquet(curation_path)
    kept = (pd.read_parquet(curation_path)["curation_step"] == "").to_numpy()

    pairs_path = root / "pairs.parquet"
    if not pairs_path.exists():
        rows = np.flatnonzero(kept)
        pairs = diagnose_pairs(
            parsed.iloc[rows].reset_index(drop=True), workers=workers
        )
        pairs["row_a"] = rows[pairs["row_a"].to_numpy()]
        pairs["row_b"] = rows[pairs["row_b"].to_numpy()]
        pairs.to_parquet(pairs_path)
    pairs = pd.read_parquet(pairs_path)

    info_path = root / "diagnostics.json"
    if not info_path.exists():
        _, cinfo = curate(parsed)
        info = {
            "commit": _commit(),
            "clusters": len(np.unique(cluster)),
            "curation": cinfo,
            "pairs": pair_summary(pairs),
        }
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    info = json.loads(info_path.read_text())

    split_path = root / "split.parquet"
    if not split_path.exists():
        assign_split(parsed, cluster, kept, n_shards=n_shards).to_parquet(split_path)
    split = pd.read_parquet(split_path)

    store_dir = Path(stores_root) / store
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / "molecules.parquet"
    frame = training_store(parsed, cluster, split)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)
    summary = split_summary(frame)
    (store_dir / "split_summary.txt").write_text(summary)
    c = info["curation"]
    (store_dir / "curation_summary.txt").write_text(
        f"{c['records']:,} records of {c['molecules']:,} molecules, "
        f"{c['structures']:,} structures\n"
        + "".join(
            f"{channel}: raised by {c[channel]['raised']:,}, "
            f"removes {c[channel]['removed']:,}\n"
            for channel in STEPS
        )
        + f"kept: {c['kept']['records']:,} records of "
        f"{c['kept']['molecules']:,} molecules, "
        f"{c['kept']['net_charge_at_or_below_-2']:,} with a net charge of "
        f"{MAX_NET_CHARGE_ANION} or below\n"
    )
    (store_dir / "diagnostics.json").write_text(json.dumps(info, indent=2) + "\n")
    (store_dir / "built-at-commit.txt").write_text(_commit() + "\n")
    logger.info("wrote %s (%d rows):\n%s", path, len(frame), summary)
    return path
