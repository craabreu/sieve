"""Download DASH's published training SDF, parse it (streaming, never
loading the whole 8.3GB file into memory), and cluster+split it.

Mirrors cosmo_experiments/sieve_experiments/prepare_store.py's own
download-verify-idempotent shape and its download/split separation
(``download_chaos_store``/``split_chaos_store``/``prepare_store`` there ->
``download_dash_sdf``/``parse_dash_molecules``/``assign_splits``/
``prepare_store`` here), adapted for a plain (non-zip) file download and a
streaming SDF parse instead of a cosmolayer SegmentStore load.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DOWNLOAD_URL = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "4e827dd2-65a0-4305-9118-480ef5fce0b5/content"
)
EXPECTED_BYTES = 8_278_301_584
EXPECTED_MD5 = "305f521c6b422546bdf09c1e87eb922d"
SDF_FILENAME = "dashMoleculesSDF_v2.sdf"
# The ETH Research Collection server 403s a request carrying urllib's default
# User-Agent (python-urllib/x.y) -- matches the UA the original
# download_dash_molecules.sh bash script already had to set for the same
# reason (see that script's own comment: the plain bitstream-content URL
# still needs a browser-shaped UA even though it isn't the HTML app-shell
# page that 500s under wget).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

CHUNK_SIZE = 1 << 20  # 1 MiB
PARQUET_BATCH_SIZE = 50_000  # rows buffered before each parquet write

logger = logging.getLogger("charge_experiments")


def download_dash_sdf(dest_dir: Path, *, url: str = DOWNLOAD_URL) -> Path:
    """Stream ``url`` to ``dest_dir / SDF_FILENAME``, verifying size and md5
    against the published values. Idempotent: does nothing but log if a
    correctly-sized file is already present (a full md5 pass over 8.3GB on
    every call would be needlessly slow; size is a fast first check, and a
    truncated/corrupted re-download is caught by md5 the next time this
    function actually re-downloads)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / SDF_FILENAME

    if out_path.exists() and out_path.stat().st_size == EXPECTED_BYTES:
        logger.info(
            "%s already present with the expected size; skipping download", out_path
        )
        return out_path

    md5 = hashlib.md5()
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response, out_path.open("wb") as f:
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


def _assign_stereo_if_needed(mol: Any) -> None:
    """If ``mol`` has an unassigned stereocenter (not already fully
    specified by the molblock's own parity bits), perceive stereo from its
    own 3D coordinates. Mutates ``mol`` in place, matching
    ``Chem.AssignStereochemistry``'s own convention."""
    from rdkit import Chem

    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if any(tag == "?" for _, tag in centers):
        Chem.AssignStereochemistryFrom3D(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _parse_one_record(mol: Any) -> dict[str, Any] | None:
    """Extract one row's worth of data from an already-parsed rdkit ``Mol``
    (one ``ForwardSDMolSupplier`` record). Returns ``None`` (and logs a
    warning) for a record missing ``MBIScharge``/``CHEMBL_ID``/``CONF_ID``
    or whose atom count disagrees with its ``MBIScharge`` count, rather than
    raising -- a handful of malformed records should not abort an
    hours-long parse of an 8.3GB file."""
    from charge_experiments.data import mol_to_blob

    if mol is None:
        return None
    for required in ("CHEMBL_ID", "CONF_ID", "MBIScharge"):
        if not mol.HasProp(required):
            logger.warning("record missing %s property; skipping", required)
            return None

    try:
        charges = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
    except ValueError:
        logger.warning(
            "MBIScharge could not be parsed as floats; skipping (chembl_id=%s)",
            mol.GetProp("CHEMBL_ID"),
        )
        return None
    if len(charges) != mol.GetNumAtoms():
        logger.warning(
            "MBIScharge has %d values but molecule has %d atoms; skipping "
            "(chembl_id=%s)",
            len(charges),
            mol.GetNumAtoms(),
            mol.GetProp("CHEMBL_ID"),
        )
        return None

    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)

    _assign_stereo_if_needed(mol)

    from rdkit import Chem

    return {
        "chembl_id": mol.GetProp("CHEMBL_ID"),
        "conf_id": mol.GetProp("CONF_ID"),
        "mol": mol_to_blob(mol),
        "net_charge": float(Chem.GetFormalCharge(mol)),
    }


def parse_dash_molecules(sdf_path: Path, out_path: Path) -> None:
    """Stream-parse ``sdf_path`` (never loading it whole into memory) into
    ``out_path``, a parquet file with columns ``chembl_id, conf_id, mol,
    net_charge`` (no ``split`` column yet -- see ``assign_splits``). Written
    in batches via a ``pyarrow.parquet.ParquetWriter`` so peak memory is
    bounded by ``PARQUET_BATCH_SIZE`` rows, not the whole (multi-million-row)
    dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem

    schema = pa.schema(
        [
            ("chembl_id", pa.string()),
            ("conf_id", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
        ]
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    n_written = 0
    n_skipped = 0
    try:
        with open(sdf_path, "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=False)
            for mol in supplier:
                row = _parse_one_record(mol)
                if row is None:
                    n_skipped += 1
                    continue
                batch.append(row)
                if len(batch) >= PARQUET_BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, schema)
                    writer.write_table(table)
                    n_written += len(batch)
                    batch = []
            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, schema)
                writer.write_table(table)
                n_written += len(batch)
    finally:
        if writer is not None:
            writer.close()

    # Only promote the temp file to out_path once the writer has closed
    # successfully -- an interrupted/failed parse (crash, OOM, a raised
    # exception) leaves nothing at out_path, so prepare_store's own
    # idempotency check never mistakes a truncated file for a finished one.
    if writer is not None:
        tmp_path.replace(out_path)

    logger.info(
        "parsed %d records (%d skipped) from %s", n_written, n_skipped, sdf_path
    )


def _achiral_fingerprints(mols: list[Any], *, radius: int = 2, n_bits: int = 2048):
    """Dense achiral Morgan fingerprint matrix, one row per mol -- the input
    shape ``_chalcedon.butina_cluster`` expects."""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    out = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=False
        )
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def assign_splits(
    store_dir: Path, *, train: float = 0.8, val: float = 0.1, test: float = 0.1
) -> str:
    """Compute (or refresh) the ``split`` column on ``store_dir /
    'molecules.parquet'`` and overwrite it in place; return the summary
    text. Clustering fingerprints come from each unique ``chembl_id``'s
    first-seen conformer only (any one conformer's connectivity suffices --
    clustering is graph-level, computed achiral so different stereoisomers
    of the same 2D graph land in the same cluster). Splits are then assigned
    per-cluster via the vendored ``greedy_cluster_split`` and joined back
    onto every row by ``chembl_id``, so a molecule's conformers/
    stereoisomers never span two splits. Loads the entire parsed store into
    memory at once (via ``pd.read_parquet``) rather than streaming, since by
    this point it is the much-smaller already-parsed parquet, not the raw
    8.3GB SDF that ``parse_dash_molecules`` streams. The fraction targets
    are cluster/chembl_id-level, so the resulting row-level
    ``split_summary.txt`` fractions may diverge somewhat from the requested
    train/val/test fractions if conformer counts per chembl_id vary."""
    if abs(train + val + test - 1) >= 1e-6:
        raise ValueError("the fractions must sum to 1")
    import pandas as pd

    from charge_experiments._chalcedon.butina_cluster import butina_cluster
    from charge_experiments._chalcedon.greedy_cluster_split import (
        greedy_cluster_split,
    )
    from charge_experiments.data import blob_to_mol

    molecules_path = store_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path)

    first_seen = df.drop_duplicates(subset="chembl_id", keep="first")
    unique_chembl_ids = first_seen["chembl_id"].to_numpy()
    first_mols = [blob_to_mol(b) for b in first_seen["mol"]]
    fingerprints = _achiral_fingerprints(first_mols)

    cluster_ids = butina_cluster(fingerprints, cutoff=0.65)
    split_by_index = greedy_cluster_split(
        cluster_ids, fractions={"train": train, "val": val, "test": test}
    )
    chembl_to_split: dict[str, str] = {}
    for split_name, indices in split_by_index.items():
        for i in indices:
            chembl_to_split[unique_chembl_ids[i]] = split_name

    df["split"] = df["chembl_id"].map(chembl_to_split)

    unmapped = df[df["split"].isna()]
    if not unmapped.empty:
        n_rows = len(unmapped)
        n_ids = unmapped["chembl_id"].nunique()
        raise ValueError(
            f"{n_rows} row(s) ({n_ids} chembl_id(s)) were not assigned a "
            "split by clustering"
        )

    summary = (
        df.groupby("split")
        .agg(n_conformers=("chembl_id", "size"), n_chembl_ids=("chembl_id", "nunique"))
        .reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_conformers"] / len(df)
    summary_text = summary.to_string()

    tmp_path = molecules_path.with_suffix(molecules_path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(molecules_path)
    return summary_text


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    sdf_path: Path | None = None,
) -> None:
    """Ensure ``store_name`` is downloaded, parsed, and has a ``split``
    column. Idempotent at each stage, mirroring
    cosmo_experiments/sieve_experiments/prepare_store.py's own
    ``prepare_store``. If ``sdf_path`` is given, it is used directly for
    parsing instead of downloading a fresh copy via ``download_dash_sdf``
    (a ``ValueError`` is raised if it does not exist)."""
    store_dir = stores_root / store_name
    store_dir.mkdir(parents=True, exist_ok=True)

    if sdf_path is not None:
        if not sdf_path.exists():
            raise ValueError(f"sdf_path {sdf_path} does not exist")
    else:
        sdf_path = download_dash_sdf(store_dir)

    molecules_path = store_dir / "molecules.parquet"
    if not molecules_path.exists():
        parse_dash_molecules(sdf_path, molecules_path)
    else:
        logger.info("%s already parsed; skipping", molecules_path)

    import pyarrow.parquet as pq

    already_split = "split" in pq.ParquetFile(molecules_path).schema.names
    if already_split:
        logger.info("%s already has a split column; nothing to do", molecules_path)
        return

    summary_text = assign_splits(store_dir, train=train, val=val, test=test)
    (store_dir / "split_summary.txt").write_text(summary_text + "\n")
    logger.info("wrote split for %s:\n%s", store_name, summary_text)
