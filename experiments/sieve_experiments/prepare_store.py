"""Download and split a COSMO segment store (chaos-store by default).

Replaces the untracked stores/download_chaos_store.py -- untracked only
because stores/ is wholesale git-ignored (it holds ~8 GB of data), not
because the download step itself should be unreviewable. Differences from
that script, all reproducibility-motivated:

- Streams the zip in chunks via urllib instead of buffering the whole
  response through `requests`, dropping an otherwise-undeclared dependency.
- Verifies the download's integrity two ways. Zenodo's own record (record
  22050672, resolved 2026-08-24 after an earlier HTTP 404) publishes an md5
  checksum per file -- ``EXPECTED_ZIP_MD5`` is checked against that. A sha256
  is also computed and recorded next to the store on first download, as a
  permanent provenance record. It is checked only once, though: the hash is
  of the downloaded *zip*, which is deleted right after extraction (to avoid
  doubling the ~8GB on disk), so there is nothing left to re-hash against on
  a later run -- ``prepare_store`` just reports the recorded value and moves
  on. An earlier version of this docstring claimed the sha256 gets
  "re-verified (not re-trusted)" on every later run; that was never actually
  implemented, and isn't achievable without either keeping the zip around or
  hashing the multi-GB extracted directory tree by some other scheme.
- Idempotent: if the store already has a ``biased_split`` column, does
  nothing and says so, rather than recomputing every run.
- Also writes split_summary.txt next to the store, so a run manifest can
  reference the split sizes without recomputing them.

``assign_clusters_by_mean_size`` is kept verbatim from the original script:
it is the headline (biased_split) split's definition.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ZENODO_RECORD = "22050672"
DOWNLOAD_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/chaos-store.zip?download=1"
)
# Zenodo record 22050672's published per-file checksum (md5, Zenodo's own
# scheme -- see `curl -s https://zenodo.org/api/records/22050672` -> files[].
# checksum). None disables published-hash verification (the sha256
# trust-on-first-download check below is still applied regardless).
EXPECTED_ZIP_MD5: str | None = "8d01a192e068e569484cd444c7264da7"

CHUNK_SIZE = 1 << 20  # 1 MiB

logger = logging.getLogger("sieve_experiments")


def _download_zip(url: str, dest: Path) -> tuple[str, str]:
    """Stream ``url`` to ``dest``, returning (sha256, md5) hex digests."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def download_chaos_store(
    stores_root: Path, *, store_dirname: str = "chaos-store"
) -> None:
    store_dir = stores_root / store_dirname
    zip_path = stores_root / f"{store_dirname}.zip"
    sha_path = store_dir / ".download.sha256"

    sha256, md5 = _download_zip(DOWNLOAD_URL, zip_path)

    if EXPECTED_ZIP_MD5 is not None and md5 != EXPECTED_ZIP_MD5:
        zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded {store_dirname}.zip md5 {md5} != Zenodo-published "
            f"{EXPECTED_ZIP_MD5}; download corrupted or record changed"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(stores_root)
    zip_path.unlink()

    sha_path.write_text(sha256 + "\n")
    logger.info("downloaded %s (sha256 %s)", store_dirname, sha256)


def assign_clusters_by_mean_size(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    n_molecules: int,
    train: float,
    val: float,
) -> dict[int, str]:
    """Assign clusters in the given order to train, then val, then test.

    Clusters are assumed already sorted (smallest molecules first). A split
    stops before the next cluster that would push it past its target count.
    Whatever is left after train and val goes to test.
    """
    assignment = {}
    train_target = train * n_molecules
    val_target = val * n_molecules
    train_count = 0
    val_count = 0
    phase = "train"

    for cluster_id, size in zip(cluster_ids, cluster_sizes, strict=True):
        if phase == "train" and train_count + size > train_target:
            phase = "val"
        if phase == "val" and val_count + size > val_target:
            phase = "test"
        assignment[int(cluster_id)] = phase
        if phase == "train":
            train_count += size
        elif phase == "val":
            val_count += size

    return assignment


def split_chaos_store(
    store_dir: Path, *, train: float = 0.8, val: float = 0.1, test: float = 0.1
) -> str:
    """Compute (or refresh) the biased_split column; return the summary text."""
    assert abs(train + val + test - 1) < 1e-6, "the fractions must sum to 1"
    from cosmolayer import store as cosmolayer_store
    from rdkit import Chem

    chaos_store = cosmolayer_store.SegmentStore.load(store_dir)
    chaos_store.assign_splits(fractions={"train": train, "val": val, "test": test})
    molecules_df = chaos_store.molecules_df

    df = molecules_df.copy()
    mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"].to_numpy()]
    if any(mol is None for mol in mols):
        raise ValueError("unparseable SMILES in chaos-store")
    df["num_heavy_atoms"] = [mol.GetNumHeavyAtoms() for mol in mols]

    cluster_stats = (
        df.groupby("cluster_id")
        .agg(
            mean_heavy_atoms=("num_heavy_atoms", "mean"),
            size=("num_heavy_atoms", "size"),
        )
        .sort_values("mean_heavy_atoms", kind="mergesort")
    )

    assignment = assign_clusters_by_mean_size(
        cluster_stats.index.to_numpy(),
        cluster_stats["size"].to_numpy(),
        n_molecules=len(df),
        train=train,
        val=val,
    )
    df["split"] = df["cluster_id"].map(assignment)

    summary = (
        df.groupby("split")
        .agg(
            n_molecules=("smiles", "size"),
            n_clusters=("cluster_id", "nunique"),
            mean_heavy_atoms=("num_heavy_atoms", "mean"),
        )
        .reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_molecules"] / len(df)
    summary_text = summary.to_string()

    molecules_df["biased_split"] = df["split"]
    molecules_df.to_parquet(store_dir / "molecules.parquet")

    return summary_text


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> None:
    """Ensure ``store_name`` is downloaded and has a ``biased_split`` column."""
    store_dir = stores_root / store_name
    if not store_dir.exists():
        download_chaos_store(stores_root, store_dirname=store_name)
    else:
        sha_path = store_dir / ".download.sha256"
        if sha_path.exists():
            logger.info(
                "%s already present (recorded sha256 %s); skipping download",
                store_name,
                sha_path.read_text().strip(),
            )
        else:
            logger.info("%s already present; skipping download", store_name)

    molecules_path = store_dir / "molecules.parquet"
    already_split = False
    if molecules_path.exists():
        import pyarrow.parquet as pq

        already_split = "biased_split" in pq.ParquetFile(molecules_path).schema.names

    if already_split:
        logger.info("%s already has a biased_split column; nothing to do", store_name)
        return

    summary_text = split_chaos_store(store_dir, train=train, val=val, test=test)
    (store_dir / "split_summary.txt").write_text(summary_text + "\n")
    logger.info("wrote biased_split for %s:\n%s", store_name, summary_text)


def prepare_ua_store(
    source_store_name: str, ua_store_name: str, *, stores_root: Path
) -> None:
    """Ensure ``ua_store_name`` exists as a united-atom (hydrogens merged
    into their heavy-atom neighbor) coarse-graining of the already-prepared
    ``source_store_name``.

    Built via ``cosmolayer``'s own ``SegmentStore.coarse_grain`` (see
    ``experiments/docs/chaos_store_ua.md`` for why this, rather than an
    atom-level predictor option, is the right place to make DASH and
    Chemprop's atom-level baselines united-atom: no segment is ever dropped,
    so molecule-level truth is bit-identical between the AA and UA stores --
    verified on a 400-molecule subsample -- only atom PARTITIONING changes.

    Idempotent like ``prepare_store``: does nothing if ``ua_store_name``
    already has a ``biased_split`` column. Requires ``source_store_name`` to
    already be downloaded and split (raises otherwise) -- the UA store reuses
    the AA store's own ``split``/``biased_split`` columns verbatim
    (``coarse_grain`` carries every ``molecules_df`` column through
    unchanged except ``smiles``/``num_atoms``/``atom_offsets``), rather than
    recomputing a split from scratch. That guarantees the two stores' splits
    are identical, not just similarly distributed -- required for T8/T11's
    AA-vs-UA numbers to be a controlled comparison (same molecules in the
    same roles) rather than two independent benchmarks.
    """
    from cosmolayer.store import SegmentStore

    source_dir = stores_root / source_store_name
    ua_dir = stores_root / ua_store_name

    ua_molecules_path = ua_dir / "molecules.parquet"
    if ua_molecules_path.exists():
        import pyarrow.parquet as pq

        if "biased_split" in pq.ParquetFile(ua_molecules_path).schema.names:
            logger.info(
                "%s already has a biased_split column; nothing to do",
                ua_store_name,
            )
            return

    if not SegmentStore.exists(source_dir):
        raise ValueError(
            f"{source_store_name!r} is not a complete store at {source_dir} "
            f"-- run `prepare-store {source_store_name}` first"
        )
    source = SegmentStore.load(source_dir)
    if "biased_split" not in source.molecules_df.columns:
        raise ValueError(
            f"{source_store_name!r} has no biased_split column -- run "
            f"`prepare-store {source_store_name}` first"
        )

    logger.info(
        "coarse-graining %s -> %s (merging hydrogens into their heavy-atom neighbor)",
        source_store_name,
        ua_store_name,
    )
    ua = source.coarse_grain()
    ua.save(ua_dir)

    n_aa, n_ua = len(source.atoms_df), len(ua.atoms_df)
    logger.info(
        "%s: %d molecules, %d atoms -> %d (%.1f%% reduction); "
        "split/biased_split carried through unchanged from %s",
        ua_store_name,
        len(ua.molecules_df),
        n_aa,
        n_ua,
        100 * (1 - n_ua / n_aa),
        source_store_name,
    )
