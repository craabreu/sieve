"""Build a store from SPICE 2.0.1 (Eastman et al., J. Chem. Theory Comput. 20,
8583 (2024); Zenodo 10.5281/zenodo.10975225, CC0): ωB97M-D3(BJ)/def2-TZVPPD
MBIS charges on many conformers per molecule.

Built from the same parts as the other two stores. The record builder is
``prepare_themol.build_record_mol``: SPICE, like THEMol, stores a canonical
isomeric explicit-hydrogen atom-mapped SMILES whose map numbers index the
coordinate rows, so the graph is read from it, the stereochemistry is
perceived from each conformer's own coordinates, and the report is checked
against the perception (``stereo_check``). Curation and the split are
``prepare_dash``'s, since SPICE, like DASH, has several conformers per
molecule to judge a charge against.

**Only single molecules are kept.** About half of SPICE's subsets are
multi-molecule systems -- DES370K dimers, amino-acid--ligand dimers, ion
pairs, water clusters, solvated amino acids and solvated PubChem molecules --
whose MBIS charges are polarised by the partner molecules, which no graph
featurisation of one molecule can see. They are recognised by their SMILES
having more than one fragment, not by subset name, and are counted per
subset in ``spice_summary.txt``. That also keeps curation sound: for a
multi-fragment system ``collapse_key`` is only a composition, so unrelated
clusters of the same composition would be pooled and their atoms compared
under an arbitrary pairing.

A group without ``mbis_charges`` (MBIS did not converge) is skipped, as is a
conformer whose charges are not all finite. Coordinates are stored in bohr
and converted to Å.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DOWNLOAD_URL = "https://zenodo.org/api/records/10975225/files/SPICE-2.0.1.hdf5/content"
EXPECTED_BYTES = 37_479_271_148
EXPECTED_MD5 = "bfba2224b6540e1390a579569b475510"
HDF5_FILENAME = "SPICE-2.0.1.hdf5"
BOHR_TO_ANGSTROM = 0.529177210903  # CODATA 2018
ID_COLUMNS = ("spice_id",)
SUMMARY = "spice_summary.txt"
CHUNK_SIZE = 1 << 20  # 1 MiB
PARQUET_BATCH_SIZE = 50_000

logger = logging.getLogger("experiments")


def download_spice_hdf5(dest_dir: Path, *, url: str = DOWNLOAD_URL) -> Path:
    """Stream ``url`` to ``dest_dir / HDF5_FILENAME``, verifying size and md5
    against the values Zenodo publishes. Idempotent in the way
    ``prepare_dash.download_dash_sdf`` is: a correctly sized file already
    present is kept without hashing (an md5 pass over 37 GB on every call
    would be needlessly slow), and a corrupted download is caught by md5 the
    next time one is actually fetched."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / HDF5_FILENAME

    if out_path.exists() and out_path.stat().st_size == EXPECTED_BYTES:
        logger.info(
            "%s already present with the expected size; skipping download", out_path
        )
        return out_path

    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            md5.update(chunk)

    actual_bytes = out_path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            f"downloaded {out_path.name}: {actual_bytes} bytes, expected "
            f"{EXPECTED_BYTES}; download incomplete or corrupted"
        )
    actual_md5 = md5.hexdigest()
    if actual_md5 != EXPECTED_MD5:
        out_path.unlink()
        raise ValueError(
            f"downloaded {out_path.name} md5 {actual_md5} != expected "
            f"{EXPECTED_MD5}; download corrupted"
        )
    logger.info("downloaded %s (md5 %s)", out_path, actual_md5)
    return out_path


def _scalar_string(dataset: Any) -> str:
    """SPICE stores ``smiles`` and ``subset`` as one-element byte arrays."""
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _n_fragments(smiles: str) -> int:
    """Connected components of the SMILES graph: one per '.'-separated part,
    counted on the parsed molecule rather than on the string."""
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("SMILES did not parse")
    return len(Chem.GetMolFrags(mol))


def _parse_one_group(spice_id: str, group: Any) -> tuple[list[dict], Counter]:
    """The rows of one HDF5 group, one per usable conformer, and the counts
    they contribute to the summary. Nothing here raises for a bad record: a
    handful of malformed ones should not abort a multi-million-conformer
    parse, so each is counted under the reason it was left out."""
    from rdkit import Chem

    from experiments.geometry import geometry_record
    from experiments.prepare_themol import build_record_mol, store_blob

    counts: Counter = Counter(groups=1)
    smiles = _scalar_string(group["smiles"])
    subset = _scalar_string(group["subset"]) if "subset" in group else ""
    try:
        n_fragments = _n_fragments(smiles)
    except ValueError:
        counts["groups_unparsed"] += 1
        return [], counts
    if n_fragments != 1:
        counts[f"multi_fragment_groups::{subset}"] += 1
        return [], counts
    if "mbis_charges" not in group:
        counts["groups_without_mbis"] += 1
        return [], counts

    units = group["conformations"].attrs.get("units", "bohr")
    if isinstance(units, bytes):
        units = units.decode()
    if units != "bohr":
        raise ValueError(f"{spice_id}: conformations in {units!r}, expected bohr")

    z = group["atomic_numbers"][()]
    conformations = group["conformations"][()] * BOHR_TO_ANGSTROM
    charges = group["mbis_charges"][()][..., 0]

    rows: list[dict] = []
    for k, (coords, q) in enumerate(zip(conformations, charges, strict=True)):
        if not np.all(np.isfinite(q)):
            counts["conformers_nonfinite_charges"] += 1
            continue
        try:
            mol, verdict, stereo_counts = build_record_mol(z, coords, q, smiles)
        except ValueError as exc:
            logger.warning("%s conformer %d: %s; skipping", spice_id, k, exc)
            counts["skipped"] += 1
            continue
        counts.update(stereo_counts)
        counts[f"verdict_{verdict}"] += 1
        rows.append(
            {
                "spice_id": spice_id,
                "conf_id": f"conf_{k}",
                "subset": subset,
                "smiles": smiles,
                "mol": store_blob(mol),
                "net_charge": float(Chem.GetFormalCharge(mol)),
                "stereo_check": verdict,
                **geometry_record(mol),
            }
        )
    counts["groups_kept"] += bool(rows)
    return rows, counts


def _schema():
    import pyarrow as pa

    from experiments.geometry import arrow_fields

    return pa.schema(
        [
            ("spice_id", pa.string()),
            ("conf_id", pa.string()),
            ("subset", pa.string()),
            ("smiles", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
            ("stereo_check", pa.string()),
            *arrow_fields(),
        ]
    )


def parse_spice_chunk(hdf5_path: Path, names: list[str], out_path: Path) -> Counter:
    """Parse the groups ``names`` of ``hdf5_path``, in that order, into one
    parquet file, and return their summary counts. Several of these run at
    once on one file, each opening it read-only."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    schema = _schema()
    totals: Counter = Counter()
    batch: list[dict] = []
    with h5py.File(hdf5_path, "r") as f, pq.ParquetWriter(out_path, schema) as writer:
        for name in names:
            rows, counts = _parse_one_group(name, f[name])
            totals.update(counts)
            batch.extend(rows)
            if len(batch) >= PARQUET_BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    return totals


def parse_spice(
    hdf5_path: Path,
    out_path: Path,
    *,
    workers: int = 16,
    limit_groups: int | None = None,
) -> Counter:
    """Parse ``hdf5_path`` into ``out_path`` and return the summary counts.

    The groups, in the file's own order, are cut into ``workers`` contiguous
    blocks, each parsed by its own process into a part file; the parts are
    then concatenated in block order, so the row order is the file's whatever
    the scheduling. Nothing appears at ``out_path`` until the whole parse has
    succeeded. ``limit_groups`` parses only the first that many groups.

    Workers are spawned, not forked, as in ``prepare_themol.parse_themol``;
    a script calling this needs an ``if __name__ == "__main__"`` guard.
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

    parts_dir = out_path.parent / ".parts"
    parts_dir.mkdir(exist_ok=True)
    parts = [parts_dir / f"block-{i:03d}.parquet" for i in range(len(blocks))]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(parse_spice_chunk, hdf5_path, block, part)
            for block, part in zip(blocks, parts, strict=True)
        ]
        results = [future.result() for future in futures]

    totals: Counter = Counter()
    for counts in results:
        totals.update(counts)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with pq.ParquetWriter(tmp_path, _schema()) as writer:
        for part in parts:
            source = pq.ParquetFile(part)
            for i in range(source.num_row_groups):
                writer.write_table(source.read_row_group(i))
    tmp_path.replace(out_path)
    for part in parts:
        part.unlink()
    parts_dir.rmdir()
    return totals


def spice_summary(totals: Counter) -> str:
    """The text of ``spice_summary.txt``: which groups were kept and why the
    rest were not, then ``prepare_themol.stereo_summary`` over the kept
    conformers."""
    from experiments.prepare_themol import stereo_summary

    lines = [
        f"groups: {totals['groups']}",
        f"  kept (single molecule with MBIS charges): {totals['groups_kept']}",
        f"  without mbis_charges: {totals['groups_without_mbis']}",
        f"  unparsable SMILES: {totals['groups_unparsed']}",
    ]
    multi = sorted(
        (k.split("::", 1)[1], v)
        for k, v in totals.items()
        if k.startswith("multi_fragment_groups::")
    )
    lines.append(f"  multi-fragment: {sum(v for _, v in multi)}")
    lines += [f"    {subset}: {count}" for subset, count in multi]
    lines.append(
        f"conformers with non-finite charges: {totals['conformers_nonfinite_charges']}"
    )
    lines.append("")
    stereo = (
        stereo_summary(totals)
        .replace("records", "conformers")
        .replace("mapped_isomeric_smiles", "SPICE's smiles")
    )
    return "\n".join(lines) + "\n\n" + stereo


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    hdf5_path: Path | None = None,
    train: float = 0.9,
    val: float = 0.0,
    test: float = 0.1,
    n_shards: int = 25,
    workers: int = 16,
    limit_groups: int | None = None,
    stop_before_split: bool = False,
    keep_uncurated: bool = False,
) -> None:
    """Ensure ``store_name`` is downloaded, parsed, curated, and has
    ``split``/``cluster``/``shard`` columns, idempotently at each stage, as
    ``prepare_dash.prepare_store`` does. If ``hdf5_path`` is given it is
    parsed instead of downloading a fresh copy into the store directory (a
    ``ValueError`` is raised if it does not exist).

    Curation is ``prepare_dash.curate_conformers`` and the split is
    ``prepare_dash.assign_splits`` keyed on ``spice_id``, the HDF5 group name;
    ``keep_uncurated`` and ``stop_before_split`` mean what they mean there,
    including the refusal to curate a store whose split predates curation.
    Unlike ``prepare_dash``, the download is skipped once the store is
    parsed: 37 GB is not worth re-fetching for a parse that is done.
    """
    import pyarrow.parquet as pq

    from experiments.prepare_dash import (
        CURATION_SUMMARY,
        UNCURATED_PARQUET,
        assign_splits,
        curate_conformers,
    )

    store_dir = stores_root / store_name
    store_dir.mkdir(parents=True, exist_ok=True)
    molecules_path = store_dir / "molecules.parquet"

    if hdf5_path is not None and not hdf5_path.exists():
        raise ValueError(f"hdf5_path {hdf5_path} does not exist")

    if not molecules_path.exists():
        if hdf5_path is None:
            hdf5_path = download_spice_hdf5(store_dir)
        totals = parse_spice(
            hdf5_path, molecules_path, workers=workers, limit_groups=limit_groups
        )
        summary = spice_summary(totals)
        (store_dir / SUMMARY).write_text(summary + "\n")
        logger.info("parsed %s:\n%s", store_name, summary)
    else:
        logger.info("%s already parsed; skipping", molecules_path)

    schema_names = set(pq.ParquetFile(molecules_path).schema.names)
    has_split = "split" in schema_names
    already_curated = (store_dir / CURATION_SUMMARY).exists()
    if {"split", "cluster", "shard"} <= schema_names and already_curated:
        logger.info("%s already curated and split; nothing to do", molecules_path)
        return
    if has_split and not already_curated:
        raise RuntimeError(
            f"{molecules_path} was split before it was curated. Curating now "
            f"would drop rows the split was computed from; delete the store "
            f"and rebuild it."
        )

    if keep_uncurated:
        uncurated_path = store_dir / UNCURATED_PARQUET
        if uncurated_path.exists():
            logger.info("%s already kept; not overwriting", uncurated_path)
        elif already_curated:
            logger.warning(
                "%s is already curated; cannot keep an uncurated copy "
                "without re-parsing, so none is written",
                molecules_path,
            )
        else:
            shutil.copy2(molecules_path, uncurated_path)
            logger.info("kept the uncurated parse at %s", uncurated_path)

    logger.info("curated %s:\n%s", store_name, curate_conformers(store_dir))

    if stop_before_split:
        logger.info("%s is parsed and curated; stopping before the split", store_name)
        return

    summary_text = assign_splits(
        store_dir,
        train=train,
        val=val,
        test=test,
        n_shards=n_shards,
        id_columns=ID_COLUMNS,
    )
    (store_dir / "split_summary.txt").write_text(summary_text + "\n")
    logger.info("wrote split for %s:\n%s", store_name, summary_text)
