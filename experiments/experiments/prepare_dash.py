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

logger = logging.getLogger("experiments")


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


def _parse_one_record(
    mol: Any, *, dash_conf_counters: dict[str, int]
) -> dict[str, Any] | None:
    """Extract one row's worth of data from an already-parsed rdkit ``Mol``
    (one ``ForwardSDMolSupplier`` record). Returns ``None`` (and logs a
    warning) for a record missing an identity (see below) or ``MBIScharge``,
    or whose atom count disagrees with its ``MBIScharge`` count, rather than
    raising -- a handful of malformed records should not abort an
    hours-long parse of an 8.3GB file.

    ``DASH_IDX`` is the corpus's *universal* molecule identifier and is read
    unconditionally: a full-file tag scan finds one on all 1,029,785 records,
    taking 348,935 distinct values -- exactly the corpus's unique-molecule
    count. Its prefix encodes which of the DASH paper's four sources
    (arXiv:2305.15981: QMugs, a prior paper's training set, lead-like ChEMBL
    v30, organic liquids) a molecule came from, and there are exactly two:
    518,669 ``QMUGS500_*`` records and 511,116 ``Rest_*`` records.

    ``CHEMBL_ID``/``CONF_ID`` are *supplementary*, present on the 518,669
    ``QMUGS500_*`` records only -- QMugs is itself derived from ChEMBL, so
    those molecules carry both identities, and the two keys agree 1:1
    (176,969 unique ``CHEMBL_ID``s against 176,969 unique ``QMUGS500_*``
    ids). This function therefore populates ``dash_id`` for every row it
    accepts and ``chembl_id``/``conf_id`` additionally where the record
    offers them; earlier revisions treated the two identities as mutually
    exclusive (an ``if``/``elif``) and so dropped ``dash_id`` on half the
    corpus. A ``Rest_*`` record has no ``CONF_ID``, so one is synthesized
    (``dash_conf_counters``, keyed by ``DASH_IDX``, hands out sequential
    ``"conf_N"`` labels per group in file order) -- ``conf_id`` is purely
    informational downstream (no predictor/metric reads it back), so a
    synthesized label is exactly as good as a real one for that purpose.

    A record carrying neither identity is skipped; see ``assign_splits`` for
    how the two columns are reconciled into one clustering key.
    """
    from experiments.data import mol_to_blob

    if mol is None:
        return None
    if not mol.HasProp("MBIScharge"):
        logger.warning("record missing MBIScharge property; skipping")
        return None

    dash_id: str | None = mol.GetProp("DASH_IDX") if mol.HasProp("DASH_IDX") else None
    has_chembl_id = mol.HasProp("CHEMBL_ID")
    if has_chembl_id:
        if not mol.HasProp("CONF_ID"):
            logger.warning(
                "record has CHEMBL_ID (%s) but missing CONF_ID; skipping",
                mol.GetProp("CHEMBL_ID"),
            )
            return None
        chembl_id: str | None = mol.GetProp("CHEMBL_ID")
        conf_id = mol.GetProp("CONF_ID")
        identity = chembl_id
    elif dash_id is not None:
        chembl_id = None
        count = dash_conf_counters.get(dash_id, 0)
        conf_id = f"conf_{count}"
        dash_conf_counters[dash_id] = count + 1
        identity = dash_id
    else:
        logger.warning("record has neither CHEMBL_ID nor DASH_IDX; skipping")
        return None

    try:
        charges = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
    except ValueError:
        logger.warning(
            "MBIScharge could not be parsed as floats; skipping (id=%s)", identity
        )
        return None
    if len(charges) != mol.GetNumAtoms():
        logger.warning(
            "MBIScharge has %d values but molecule has %d atoms; skipping (id=%s)",
            len(charges),
            mol.GetNumAtoms(),
            identity,
        )
        return None

    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)

    _assign_stereo_if_needed(mol)

    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    # Unconditional, unlike _assign_stereo_if_needed's own 3D-perception
    # step: a record whose stereo was already fully specified by the
    # molblock's own parity bits skips that conditional branch entirely,
    # but _CIPCode is set by AssignStereochemistry/AssignCIPLabels, not by
    # molblock parsing itself -- so it would be missing for that (common)
    # case if this call were nested inside _assign_stereo_if_needed's own
    # "only if unassigned" branch. AssignCIPLabels needs
    # AssignStereochemistry's own ChiralTag perception to already have run
    # (it labels tagged centers, it doesn't discover them) -- cheap to
    # call again here even when _assign_stereo_if_needed already ran it,
    # since re-running is idempotent.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(mol)

    net_charge = float(Chem.GetFormalCharge(mol))

    # ForwardSDMolSupplier auto-attaches every ">  <TAG>" block in the
    # record as a mol-level property -- not just the three read above, but
    # every GFN2:*/DFT:* quantum-chemistry field too (energies, dipoles,
    # bond orders, Mulliken/Loewdin charges, ...). mol_to_blob's
    # PropertyPickleOptions.MolProps would otherwise serialize all of them
    # into the stored blob, unused, bloating every row several-fold beyond
    # what this series actually needs (chembl_id/conf_id/net_charge already
    # live in their own parquet columns; nothing reads them back off the
    # Mol). Clear every mol-level property before serializing -- atom-level
    # MBIScharge (set just above) is unaffected, it lives on the Atom
    # objects, not the Mol's own property dict.
    for name in list(mol.GetPropNames()):
        mol.ClearProp(name)

    # Set *after* the clear-props loop above, not before -- CIP_LABELED_PROP
    # is a mol-level marker (this loop clears exactly those), unlike
    # MBIScharge/_CIPCode, which live on the Atom objects and are
    # unaffected by it. Tells sieve.io.rdkit_adapter's own
    # _ensure_cip_labels that the rigorous rdCIPLabeler already ran here,
    # so featurization never needs to recompute it.
    from sieve.io.rdkit_adapter import CIP_LABELED_PROP

    mol.SetBoolProp(CIP_LABELED_PROP, True)

    return {
        "chembl_id": chembl_id,
        "conf_id": conf_id,
        "dash_id": dash_id,
        "mol": mol_to_blob(mol),
        "net_charge": net_charge,
    }


def parse_dash_molecules(sdf_path: Path, out_path: Path) -> None:
    """Stream-parse ``sdf_path`` (never loading it whole into memory) into
    ``out_path``, a parquet file with columns ``chembl_id, conf_id, dash_id,
    mol, net_charge`` (no ``split`` column yet -- see ``assign_splits``).
    ``dash_id`` is set on every row, ``chembl_id``/``conf_id`` additionally
    on the ``QMUGS500_*`` cohort whose records carry them (see
    ``_parse_one_record``'s docstring for why the SDF has two record
    schemas). Written in batches via a ``pyarrow.parquet.ParquetWriter`` so
    peak memory is bounded by ``PARQUET_BATCH_SIZE`` rows, not the whole
    (multi-million-row) dataset -- ``dash_conf_counters`` is the one piece of
    state carried across the whole streaming pass, and it's small (one int
    per unique ``DASH_IDX``, not per row)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem

    schema = pa.schema(
        [
            ("chembl_id", pa.string()),
            ("conf_id", pa.string()),
            ("dash_id", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
        ]
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    n_written = 0
    n_skipped = 0
    dash_conf_counters: dict[str, int] = {}
    try:
        with open(sdf_path, "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=False)
            for mol in supplier:
                row = _parse_one_record(mol, dash_conf_counters=dash_conf_counters)
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


def cluster_size_report(
    store_dir: Path,
    *,
    train: float = 0.9,
    test: float = 0.1,
    candidate_n_shards: tuple[int, ...] = (10, 25, 50, 100),
) -> str:
    """Diagnostic, read-only: report the Butina cluster size distribution
    within what would become the train split, and the shard balance
    ``greedy_cluster_split`` would achieve for each candidate ``n_shards`` --
    the evidence ``assign_splits``'s own ``n_shards`` should be chosen from,
    not a value picked in advance. Recomputes the same clustering/split
    ``assign_splits`` performs (deterministically, so its numbers match
    whatever ``assign_splits`` is then actually called with); it writes
    nothing and never touches the store.

    The binding constraint on ``n_shards`` is the largest Butina cluster: a
    single cluster larger than one shard's target (``train_molecules /
    n_shards``) pushes that shard over target no matter what, and can even
    leave some other shard empty (``greedy_cluster_split`` returns an empty
    array for a split that never gets picked -- silent, not an error). Pick
    the largest ``n_shards`` for which every candidate below reports zero
    empty shards and a tight min/max spread."""
    if abs(train + test - 1) >= 1e-6:
        raise ValueError("train and test must sum to 1")
    import pandas as pd

    from experiments._chalcedon.butina_cluster import butina_cluster
    from experiments._chalcedon.greedy_cluster_split import greedy_cluster_split
    from experiments.data import blob_to_mol

    molecules_path = store_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path, columns=["chembl_id", "dash_id", "mol"])
    mol_key = df["dash_id"].fillna(df["chembl_id"])
    first_seen_mask = ~mol_key.duplicated(keep="first")
    first_mols = [blob_to_mol(b) for b in df.loc[first_seen_mask, "mol"]]
    fingerprints = _achiral_fingerprints(first_mols)
    cluster_ids = butina_cluster(fingerprints, cutoff=0.65)

    split_by_index = greedy_cluster_split(
        cluster_ids, fractions={"train": train, "test": test}
    )
    train_cluster_ids = cluster_ids[split_by_index["train"]]
    n_train = len(train_cluster_ids)

    _, inverse = np.unique(train_cluster_ids, return_inverse=True)
    sizes = np.bincount(inverse)

    lines = [
        f"train molecules: {n_train}",
        f"clusters in train: {len(sizes)}",
        f"largest cluster: {int(sizes.max())} molecule(s) "
        f"({sizes.max() / n_train:.4%} of train)",
    ]
    for q in (50, 90, 99, 99.9):
        lines.append(f"  p{q} cluster size: {int(np.percentile(sizes, q))}")

    lines.append("")
    lines.append("achieved shard balance by candidate n_shards:")
    for n in candidate_n_shards:
        fractions = {f"s{i:02d}": 1 / n for i in range(n)}
        shard_by_index = greedy_cluster_split(train_cluster_ids, fractions=fractions)
        counts = np.array([len(v) for v in shard_by_index.values()])
        n_empty = int((counts == 0).sum())
        lines.append(
            f"  N={n:>4}: min={counts.min()} max={counts.max()} "
            f"mean={counts.mean():.1f} std={counts.std():.1f} "
            f"empty_shards={n_empty}"
        )
    return "\n".join(lines) + "\n"


def assign_splits(
    store_dir: Path,
    *,
    train: float = 0.9,
    val: float = 0.0,
    test: float = 0.1,
    n_shards: int = 25,
) -> str:
    """Compute (or refresh) the ``split``/``cluster``/``shard`` columns on
    ``store_dir / 'molecules.parquet'`` and overwrite it in place; return
    the summary text. Clustering fingerprints come from each unique
    molecule's first-seen conformer only (any one conformer's connectivity
    suffices -- clustering is graph-level, computed achiral so different
    stereoisomers of the same 2D graph land in the same cluster). Splits
    are then assigned per-cluster via the vendored ``greedy_cluster_split``
    and joined back onto every row by molecule identity, so a molecule's
    conformers/stereoisomers never span two splits.

    ``val`` defaults to ``0.0`` -- the production series is a plain 90/10
    train/test split, per the CV redesign (repeated cross-validation on
    train supersedes a held-out validation split). A non-zero ``val`` is
    still honored (existing tests pass one explicitly), but a zero-or-less
    value is *omitted* from the fractions dict passed to
    ``greedy_cluster_split`` rather than passed through as ``0.0`` --
    that function rejects any non-positive target.

    ``n_shards`` divides the *train* molecules only (never test) into that
    many further cluster-clean partitions, named ``s00``..``s{n_shards-1}``,
    via a second ``greedy_cluster_split`` call restricted to train's own
    cluster ids -- same clustering pass as the split above, not a separate
    one (Butina is float32 and not guaranteed to reproduce boundary
    decisions bit-for-bit across two separate runs, so the shard ids must
    come from the same ``cluster_ids`` array the split itself used). These
    are the shards a CV scheme fits once and reassembles via merging
    (design doc: redesign-cv-shards) -- see ``cluster_size_report`` for how
    to choose ``n_shards`` before calling this.

    A row's identity is ``dash_id`` -- the universal key, set on every row a
    current ``_parse_one_record`` accepts (see its docstring) -- falling back
    to ``chembl_id`` for stores written before ``dash_id`` was populated on
    the ChEMBL-identified half. The two orderings group identically wherever
    both are present (they agree 1:1), so this fallback changes no existing
    store's split; it only lets a legacy store still be split. That coalesced
    key (``mol_key`` below) is what clustering, splitting, and this
    function's own uniqueness/grouping all operate on -- ``chembl_id``/
    ``dash_id`` themselves stay in the output purely as provenance, never
    read back for grouping elsewhere.

    Loads the entire parsed store into memory at once (via
    ``pd.read_parquet``) rather than streaming, since by this point it is
    the much-smaller already-parsed parquet, not the raw 8.3GB SDF that
    ``parse_dash_molecules`` streams. The fraction targets are
    cluster/mol_key-level, so the resulting row-level ``split_summary.txt``
    fractions may diverge somewhat from the requested train/val/test
    fractions if conformer counts per molecule vary."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    fractions = {"train": train, "test": test}
    if val > 0:
        fractions["val"] = val
    if abs(sum(fractions.values()) - 1) >= 1e-6:
        raise ValueError("the fractions must sum to 1")
    import pandas as pd

    from experiments._chalcedon.butina_cluster import butina_cluster
    from experiments._chalcedon.greedy_cluster_split import (
        greedy_cluster_split,
    )
    from experiments.data import blob_to_mol

    molecules_path = store_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path)
    mol_key = df["dash_id"].fillna(df["chembl_id"])

    first_seen_mask = ~mol_key.duplicated(keep="first")
    unique_keys = mol_key[first_seen_mask].to_numpy()
    first_mols = [blob_to_mol(b) for b in df.loc[first_seen_mask, "mol"]]
    fingerprints = _achiral_fingerprints(first_mols)

    cluster_ids = butina_cluster(fingerprints, cutoff=0.65)
    split_by_index = greedy_cluster_split(cluster_ids, fractions=fractions)
    key_to_split: dict[str, str] = {}
    for split_name, indices in split_by_index.items():
        for i in indices:
            key_to_split[unique_keys[i]] = split_name

    df["split"] = mol_key.map(key_to_split)
    df["cluster"] = mol_key.map(
        dict(zip(unique_keys.tolist(), cluster_ids.tolist(), strict=True))
    ).astype("Int64")

    unmapped = df[df["split"].isna()]
    if not unmapped.empty:
        n_rows = len(unmapped)
        n_ids = mol_key[df["split"].isna()].nunique()
        raise ValueError(
            f"{n_rows} row(s) ({n_ids} molecule(s)) were not assigned a "
            "split by clustering"
        )

    # Shard assignment: restricted to the train molecules' own cluster ids
    # (index space is into `train_positions`, not into `unique_keys` --
    # greedy_cluster_split returns positions into whatever array it was
    # given, here the train-only subset, so the index has to be translated
    # back through train_positions before it means a row in unique_keys).
    train_positions = np.flatnonzero(
        np.array([key_to_split[k] == "train" for k in unique_keys])
    )
    train_cluster_ids = cluster_ids[train_positions]
    shard_fractions = {f"s{i:02d}": 1 / n_shards for i in range(n_shards)}
    shard_by_index = greedy_cluster_split(train_cluster_ids, fractions=shard_fractions)

    empty_shards = [name for name, idxs in shard_by_index.items() if len(idxs) == 0]
    if empty_shards:
        raise ValueError(
            f"{len(empty_shards)} of {n_shards} shard(s) got no molecules "
            f"({empty_shards}); lower n_shards or see cluster_size_report"
        )

    key_to_shard: dict[str, str] = {}
    for shard_name, idxs in shard_by_index.items():
        for i in idxs:
            key_to_shard[unique_keys[train_positions[i]]] = shard_name

    # Every row starts out labelled by its own split (so a val/test row's
    # shard reads "val"/"test", matching the "literal 'test' on test rows"
    # convention exactly when there is no val split at all); only train
    # rows are then overwritten with their own s00.. shard id.
    df["shard"] = df["split"]
    train_row_mask = (df["split"] == "train").to_numpy()
    df.loc[train_row_mask, "shard"] = mol_key[train_row_mask].map(key_to_shard)
    if df.loc[train_row_mask, "shard"].isna().any():
        raise ValueError("some train row(s) were not assigned a shard")

    shard_counts = (
        mol_key[train_row_mask].groupby(df.loc[train_row_mask, "shard"]).nunique()
    )
    logger.info(
        "n_shards=%d train molecule counts: min=%d max=%d mean=%.1f",
        n_shards,
        int(shard_counts.min()),
        int(shard_counts.max()),
        float(shard_counts.mean()),
    )

    summary = (
        df.groupby("split").agg(n_conformers=("mol", "size")).reindex(list(fractions))
    )
    summary["n_molecules"] = (
        mol_key.groupby(df["split"]).nunique().reindex(list(fractions))
    )
    summary["fraction"] = summary["n_conformers"] / len(df)
    summary_text = summary.to_string()

    tmp_path = molecules_path.with_suffix(molecules_path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(molecules_path)
    return summary_text


CURATION_THRESHOLD = 0.4
CURATION_SUMMARY = "curation_summary.txt"


def curate_conformers(store_dir: Path, *, threshold: float = CURATION_THRESHOLD) -> str:
    """Drop conformers whose MBIS charges disagree with *every* sibling, and
    overwrite ``molecules.parquet`` in place; return the summary text.

    This is the DASH paper's own conformer criterion -- "we used the
    difference between the partial charge of the same atom in the three
    conformers to discard conformers with differences larger than 0.4 e" --
    applied here at parse time, before the split. It is a reconstruction,
    not a port: the published DASH-tree code implements only their
    *other* filter, a per-element charge-range check
    (``tree_develop/tree_constructor.py::_check_charges``), which is gated
    behind a ``sanitize_charges=False`` default, runs at tree-construction
    time, and drops zero of the released corpus's 43,248,638 atoms -- its
    bounds (C within (-2, 4), for instance) are wide enough to admit the
    +2.975 e aromatic carbon that motivated this function. No
    implementation of the 0.4 e criterion appears anywhere in that repo,
    and 2,292 molecules of the distributed SDF violate it, for reasons
    this code cannot determine.

    **The rule.** For each molecule, compare every *pair* of conformers
    atom by atom; a pair agrees when no atom's charge differs by more than
    ``threshold``. A conformer is removed exactly when it agrees with none
    of its siblings.

    Stated the other way round: a failed MBIS partition is wrong in its
    own particular way, so it disagrees with everything and is identified
    without a tie-break. Charges that vary smoothly with geometry leave
    every conformer agreeing with at least one neighbour, so a molecule
    spread along a continuum is kept whole -- the A-B and A-C agree while
    B-C does not case, where deleting either B or C would be arbitrary
    (141 molecules of the real corpus). Requiring instead that *all* pairs
    agree would discard those molecules for being smoothly variable.

    Survivors come in pairs by construction, so a molecule ends with
    0, 2 or 3 conformers and never a lone one; no separate guard is
    needed for that. A molecule with a single conformer in the *input*
    has no pair to corroborate it and is therefore removed -- which never
    occurs in the real corpus (minimum 2 conformers per molecule), but is
    pinned by test rather than left to chance.

    Grouping is by ``dash_id``, the only identity present on every record:
    49.6% of the corpus has no ``CHEMBL_ID``, and grouping on that would
    leave half the molecules uncorroborated and delete them.

    On the real store this removes 2,247 of 1,029,785 conformers (0.218%)
    and 86 molecules outright. Idempotent: skips when
    ``curation_summary.txt`` already exists.
    """
    import pandas as pd

    from experiments.data import blob_to_mol

    summary_path = store_dir / CURATION_SUMMARY
    if summary_path.exists():
        text = summary_path.read_text().strip()
        logger.info("%s already curated; skipping", store_dir)
        return f"already curated\n{text}"

    molecules_path = store_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path)
    charges = [
        np.array(
            [a.GetDoubleProp("MBIScharge") for a in blob_to_mol(b).GetAtoms()],
            dtype=np.float64,
        )
        for b in df["mol"]
    ]

    keep = np.zeros(len(df), dtype=bool)
    n_molecules_dropped = 0
    for _, positions in df.groupby("dash_id", sort=False).groups.items():
        rows = list(positions)
        survivors: set[int] = set()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = charges[rows[i]], charges[rows[j]]
                if a.shape != b.shape:
                    # Same molecule id with differing atom counts cannot be
                    # compared atom-by-atom; treat the pair as disagreeing
                    # rather than crashing or silently broadcasting.
                    continue
                if float(np.abs(a - b).max()) <= threshold:
                    survivors.update((rows[i], rows[j]))
        if not survivors:
            n_molecules_dropped += 1
        for r in survivors:
            keep[r] = True

    n_before = len(df)
    df.loc[keep].reset_index(drop=True).to_parquet(molecules_path)
    summary_text = (
        f"conformer curation at threshold {threshold} e\n"
        f"conformers: {n_before} -> {int(keep.sum())} "
        f"({n_before - int(keep.sum())} removed)\n"
        f"molecules removed entirely: {n_molecules_dropped}"
    )
    summary_path.write_text(summary_text + "\n")
    return summary_text


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    train: float = 0.9,
    val: float = 0.0,
    test: float = 0.1,
    n_shards: int = 25,
    sdf_path: Path | None = None,
) -> None:
    """Ensure ``store_name`` is downloaded, parsed, curated, and has
    ``split``/``cluster``/``shard`` columns. Idempotent at each stage,
    mirroring cosmo_experiments/sieve_experiments/prepare_store.py's own
    ``prepare_store``. If ``sdf_path`` is given, it is used directly for
    parsing instead of downloading a fresh copy via ``download_dash_sdf``
    (a ``ValueError`` is raised if it does not exist).

    ``curate_conformers`` runs *between* parse and split, which is the whole
    point of doing it here rather than at fitting time: the split is then
    computed on the population that survives curation, so no fold inherits a
    record the criterion rejects, and nothing downstream has to remember to
    filter. A store parsed before curation existed has no
    ``curation_summary.txt`` and is curated in place on the next call --
    which changes its row set, so any split already written from the
    uncurated population is stale. That case is refused rather than silently
    producing a store whose split predates its own contents.

    ``n_shards`` is threaded straight to ``assign_splits`` (see its own
    docstring): it divides train into that many cluster-clean shards, the
    unit a CV scheme fits once and reassembles by merging. Choose it via
    ``cluster_size_report`` against a parsed-but-not-yet-split store, not
    blind."""
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

    schema_names = set(pq.ParquetFile(molecules_path).schema.names)
    has_split = "split" in schema_names
    has_full_split = {"split", "cluster", "shard"} <= schema_names
    already_curated = (store_dir / CURATION_SUMMARY).exists()
    if has_full_split and already_curated:
        logger.info("%s already curated and split; nothing to do", molecules_path)
        return
    if has_split and not has_full_split:
        raise RuntimeError(
            f"{molecules_path} has a 'split' column but not the newer "
            f"'cluster'/'shard' columns -- it was split before the CV "
            f"redesign added them. Delete the store and rebuild it (a "
            f"pre-CV-redesign split cannot be back-filled with cluster ids "
            f"that reproduce it, since Butina is not guaranteed bit-exact "
            f"across separate runs)."
        )
    if has_split and not already_curated:
        raise RuntimeError(
            f"{molecules_path} was split before conformer curation existed. "
            f"Curating now would drop rows the split was computed from, "
            f"leaving a stale split. Delete the store and rebuild it, or "
            f"curate and re-run assign_splits deliberately."
        )

    # curate_conformers writes CURATION_SUMMARY itself; writing it again
    # here would stamp its own "already curated" prefix into the file on a
    # re-run.
    logger.info("curated %s:\n%s", store_name, curate_conformers(store_dir))

    summary_text = assign_splits(
        store_dir, train=train, val=val, test=test, n_shards=n_shards
    )
    (store_dir / "split_summary.txt").write_text(summary_text + "\n")
    logger.info("wrote split for %s:\n%s", store_name, summary_text)
