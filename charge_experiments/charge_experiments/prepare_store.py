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


def _parse_one_record(
    mol: Any, *, dash_conf_counters: dict[str, int]
) -> dict[str, Any] | None:
    """Extract one row's worth of data from an already-parsed rdkit ``Mol``
    (one ``ForwardSDMolSupplier`` record). Returns ``None`` (and logs a
    warning) for a record missing an identity (see below) or ``MBIScharge``,
    or whose atom count disagrees with its ``MBIScharge`` count, rather than
    raising -- a handful of malformed records should not abort an
    hours-long parse of an 8.3GB file.

    The real SDF turns out to hold two distinct record schemas, confirmed
    against the DASH paper's own stated dataset composition (arXiv:2305.15981):
    the training set was assembled from four sources -- QMugs, a prior
    paper's training set, lead-like ChEMBL v30 molecules, and organic
    liquids -- and only the ChEMBL-sourced third of records carry a
    ``CHEMBL_ID``/``CONF_ID`` pair. The other three sources' records instead
    carry a ``DASH_IDX`` property (e.g. ``"Rest_2"``) which plays the exact
    same per-molecule grouping role ``CHEMBL_ID`` does -- confirmed
    empirically: ~3 rows share each ``DASH_IDX`` value, the same "a few
    conformers per molecule" pattern ``CHEMBL_ID`` rows show. Those records
    have no ``CONF_ID`` at all, so one is synthesized here
    (``dash_conf_counters``, keyed by ``DASH_IDX``, hands out sequential
    ``"conf_N"`` labels per group in file order) -- ``conf_id`` is purely
    informational downstream (no predictor/metric reads it back), so a
    synthesized label is exactly as good as a real one for that purpose.
    A row's ``chembl_id``/``dash_id`` columns are populated from whichever
    scheme its own record used; the other is left ``None`` -- see
    ``assign_splits`` for how the two are reconciled into one clustering key.
    """
    from charge_experiments.data import mol_to_blob

    if mol is None:
        return None
    if not mol.HasProp("MBIScharge"):
        logger.warning("record missing MBIScharge property; skipping")
        return None

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
        dash_id: str | None = None
        identity = chembl_id
    elif mol.HasProp("DASH_IDX"):
        chembl_id = None
        dash_id = mol.GetProp("DASH_IDX")
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
    Exactly one of ``chembl_id``/``dash_id`` is set per row (see
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


def assign_splits(
    store_dir: Path, *, train: float = 0.8, val: float = 0.1, test: float = 0.1
) -> str:
    """Compute (or refresh) the ``split`` column on ``store_dir /
    'molecules.parquet'`` and overwrite it in place; return the summary
    text. Clustering fingerprints come from each unique molecule's
    first-seen conformer only (any one conformer's connectivity suffices --
    clustering is graph-level, computed achiral so different stereoisomers
    of the same 2D graph land in the same cluster). Splits are then assigned
    per-cluster via the vendored ``greedy_cluster_split`` and joined back
    onto every row by molecule identity, so a molecule's conformers/
    stereoisomers never span two splits.

    A row's identity is ``chembl_id`` when set, else ``dash_id`` (exactly
    one is set per row -- see ``_parse_one_record``'s docstring for why the
    store has two identity schemes). That coalesced key (``mol_key`` below)
    is what clustering, splitting, and this function's own uniqueness/
    grouping all operate on -- ``chembl_id``/``dash_id`` themselves stay in
    the output purely as provenance, never read back for grouping elsewhere.

    Loads the entire parsed store into memory at once (via
    ``pd.read_parquet``) rather than streaming, since by this point it is
    the much-smaller already-parsed parquet, not the raw 8.3GB SDF that
    ``parse_dash_molecules`` streams. The fraction targets are
    cluster/mol_key-level, so the resulting row-level ``split_summary.txt``
    fractions may diverge somewhat from the requested train/val/test
    fractions if conformer counts per molecule vary."""
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
    mol_key = df["chembl_id"].fillna(df["dash_id"])

    first_seen_mask = ~mol_key.duplicated(keep="first")
    unique_keys = mol_key[first_seen_mask].to_numpy()
    first_mols = [blob_to_mol(b) for b in df.loc[first_seen_mask, "mol"]]
    fingerprints = _achiral_fingerprints(first_mols)

    cluster_ids = butina_cluster(fingerprints, cutoff=0.65)
    split_by_index = greedy_cluster_split(
        cluster_ids, fractions={"train": train, "val": val, "test": test}
    )
    key_to_split: dict[str, str] = {}
    for split_name, indices in split_by_index.items():
        for i in indices:
            key_to_split[unique_keys[i]] = split_name

    df["split"] = mol_key.map(key_to_split)

    unmapped = df[df["split"].isna()]
    if not unmapped.empty:
        n_rows = len(unmapped)
        n_ids = mol_key[df["split"].isna()].nunique()
        raise ValueError(
            f"{n_rows} row(s) ({n_ids} molecule(s)) were not assigned a "
            "split by clustering"
        )

    summary = (
        df.groupby("split")
        .agg(n_conformers=("mol", "size"))
        .reindex(["train", "val", "test"])
    )
    summary["n_molecules"] = (
        mol_key.groupby(df["split"]).nunique().reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_conformers"] / len(df)
    summary_text = summary.to_string()

    tmp_path = molecules_path.with_suffix(molecules_path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(molecules_path)
    return summary_text


def subsample_store(
    source_store: str,
    dest_store: str,
    *,
    stores_root: Path,
    n_molecules: int = 50_000,
    conformers_per_molecule: int = 1,
    seed: int = 0,
) -> str:
    """Build a smaller, independent store by subsampling molecules from an
    already-split ``source_store``, reproducing that source's own real
    train/val/test fractions (measured directly from it -- never assumed to
    be 80/10/10) -- the scientifically-sound alternative to ``--limit``'s
    literal row-prefix slice (``runner.load_molecule_set``), which is
    neither a random sample nor split-proportional (see the harness's own
    docs on why: a small ``--limit`` window can badly over/under-represent
    a split, even leave one empty, since it just takes however many of the
    original SDF's own first-N rows happen to carry that label).

    Operates purely on ``source_store``'s own parquet columns (``chembl_id``/
    ``dash_id``/``split``) -- never deserializes a single ``Mol`` blob, so
    this stays fast even against the full, 1M+-row store.

    ``n_molecules`` is a *target* total across all three splits, distributed
    to each split proportionally to that split's own real share of the
    source store's molecule count (rounded, then clamped to however many
    molecules that split actually has -- clamping is logged, not an error,
    since a small source store can't always supply the requested count).

    ``conformers_per_molecule`` is a *cap*, not a floor: a molecule with
    fewer conformers than requested contributes all of its own (no
    padding/repeats); one with more has exactly that many sampled uniformly
    at random, without replacement -- so a molecule's conformers never span
    two splits (inherited directly from the source's own per-molecule
    split assignment) and a selected molecule is never over-represented
    beyond what was asked for.

    Both molecule selection and conformer selection use
    ``np.random.default_rng(seed)``, so the same ``seed`` reproduces the
    same subsample. Writes ``dest_store/molecules.parquet`` (with the same
    schema, ``split`` column included) and a ``split_summary.txt`` -- the
    subsample's own *actually achieved* counts/fractions, for transparency
    against the requested target. Returns that summary text.
    """
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1")
    if conformers_per_molecule < 1:
        raise ValueError("conformers_per_molecule must be >= 1")

    import pandas as pd

    source_path = stores_root / source_store / "molecules.parquet"
    df = pd.read_parquet(source_path)
    if "split" not in df.columns:
        raise ValueError(
            f"{source_path} has no split column; run prepare_store on "
            f"{source_store!r} first"
        )

    mol_key = df["chembl_id"].fillna(df["dash_id"])
    split_col = df["split"].to_numpy()
    # One groupby pass gives every molecule's own row positions (not row
    # labels -- .indices, unlike .groups, is positional, which is exactly
    # what .iloc needs below) -- O(n_rows), not O(n_molecules * n_rows).
    positions_by_key = df.groupby(mol_key).indices

    keys_by_split: dict[str, list[str]] = {}
    for key, positions in positions_by_key.items():
        keys_by_split.setdefault(split_col[positions[0]], []).append(str(key))
    total_molecules = len(positions_by_key)

    rng = np.random.default_rng(seed)
    selected_positions: list[np.ndarray] = []
    for split_name in ("train", "val", "test"):
        keys = keys_by_split.get(split_name, [])
        if not keys:
            continue
        target = round(n_molecules * len(keys) / total_molecules)
        n_pick = min(target, len(keys))
        if n_pick < target:
            logger.warning(
                "%s split of %r only has %d molecule(s), fewer than the "
                "%d requested; using all of them",
                split_name,
                source_store,
                len(keys),
                target,
            )
        picked = rng.choice(len(keys), size=n_pick, replace=False)
        for i in picked:
            positions = positions_by_key[keys[i]]
            if len(positions) > conformers_per_molecule:
                positions = rng.choice(
                    positions, size=conformers_per_molecule, replace=False
                )
            selected_positions.append(positions)

    all_positions = (
        np.sort(np.concatenate(selected_positions))
        if selected_positions
        else np.array([], dtype=np.int64)
    )
    out_df = df.iloc[all_positions].reset_index(drop=True)

    dest_dir = stores_root / dest_store
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "molecules.parquet"
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    out_df.to_parquet(tmp_path)
    tmp_path.replace(dest_path)

    out_mol_key = out_df["chembl_id"].fillna(out_df["dash_id"])
    summary = (
        out_df.groupby("split")
        .agg(n_conformers=("mol", "size"))
        .reindex(["train", "val", "test"])
    )
    summary["n_molecules"] = (
        out_mol_key.groupby(out_df["split"]).nunique().reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_conformers"] / len(out_df)
    summary_text = summary.to_string()
    (dest_dir / "split_summary.txt").write_text(summary_text + "\n")

    logger.info(
        "subsampled %r -> %r: %d molecule(s), %d conformer(s)\n%s",
        source_store,
        dest_store,
        out_mol_key.nunique(),
        len(out_df),
        summary_text,
    )
    return summary_text


def _to_united_atom(mol: Any) -> tuple[Any, int, int]:
    """Remove ``mol``'s own hydrogens via ``Chem.RemoveHs`` (rdkit's own
    default judgment of which H's are safe to strip -- see module docstring
    for what "safe" means: not stereo-defining, no isotope/query, not
    bridging, ...), adding each actually-removed H's own ``MBIScharge`` onto
    the single heavy atom it was bonded to. An H rdkit declines to remove is
    left in place, its own charge untouched -- never forced out. Total
    charge is conserved exactly: every removed H's charge lands on exactly
    one heavy atom, never dropped.

    Uses a scratch atom-map-number tag (cleared again before returning) to
    recover, for every atom surviving ``RemoveHs``, which original atom
    index it was -- the only way to tell which specific H's were removed
    vs. kept, since ``RemoveHs`` doesn't report that directly. Returns
    ``(ua_mol, n_removed, n_kept)``.
    """
    from rdkit import Chem

    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # 0 means "unset" in rdkit

    h_charge: dict[int, float] = {}
    h_heavy_neighbor: dict[int, int] = {}
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() == 1:
            idx = atom.GetIdx()
            h_charge[idx] = atom.GetDoubleProp("MBIScharge")
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                h_heavy_neighbor[idx] = neighbors[0].GetIdx()

    ua_mol = Chem.RemoveHs(work)

    surviving_orig_by_new_idx = {
        atom.GetIdx(): atom.GetAtomMapNum() - 1 for atom in ua_mol.GetAtoms()
    }
    surviving_orig = set(surviving_orig_by_new_idx.values())

    bonus: dict[int, float] = {}
    n_removed = 0
    n_kept = 0
    for h_idx, charge in h_charge.items():
        if h_idx in surviving_orig:
            n_kept += 1
            continue
        heavy_idx = h_heavy_neighbor.get(h_idx)
        if heavy_idx is None:
            # No single heavy neighbor (a bridging or isolated H) -- rdkit
            # does not remove these by default, so this branch should be
            # unreachable, but treat it as "kept" defensively rather than
            # silently drop a charge with nowhere documented to go.
            n_kept += 1
            continue
        bonus[heavy_idx] = bonus.get(heavy_idx, 0.0) + charge
        n_removed += 1

    for atom in ua_mol.GetAtoms():
        atom.SetAtomMapNum(0)
        add = bonus.get(surviving_orig_by_new_idx[atom.GetIdx()])
        if add:
            atom.SetDoubleProp("MBIScharge", atom.GetDoubleProp("MBIScharge") + add)

    return ua_mol, n_removed, n_kept


def to_united_atom_store(
    source_store: str, dest_store: str, *, stores_root: Path
) -> None:
    """Build a united-atom (heavy-atom-only, where rdkit allows it) version
    of an already-prepared ``source_store``: every conformer's ``Mol`` goes
    through ``_to_united_atom`` (see its own docstring for the redistribution
    rule and rdkit's "refuses to remove" cases). ``chembl_id``/``conf_id``/
    ``dash_id``/``net_charge``/``split`` are copied through unchanged --
    this is a different chemical representation of the exact same
    conformers, not a re-split or re-sample, so a molecule's split
    assignment is untouched. ``net_charge`` (a molblock-level ``M CHG`` sum,
    not an atom-level quantity) needs no adjustment either.

    Streams the source parquet in ``PARQUET_BATCH_SIZE``-row batches (read
    and write both), so peak memory stays bounded regardless of the source
    store's own size -- the full store's ~1M rows never load into memory at
    once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from charge_experiments.data import blob_to_mol, mol_to_blob

    source_path = stores_root / source_store / "molecules.parquet"
    source_file = pq.ParquetFile(source_path)
    schema = source_file.schema_arrow

    dest_dir = stores_root / dest_store
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "molecules.parquet"
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    n_conformers = 0
    n_h_removed = 0
    n_h_kept = 0
    writer: pq.ParquetWriter | None = None
    try:
        for record_batch in source_file.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            out_rows = []
            for row in record_batch.to_pylist():
                mol = blob_to_mol(row["mol"])
                ua_mol, removed, kept = _to_united_atom(mol)
                n_h_removed += removed
                n_h_kept += kept
                row = dict(row)
                row["mol"] = mol_to_blob(ua_mol)
                out_rows.append(row)
            table = pa.Table.from_pylist(out_rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, schema)
            writer.write_table(table)
            n_conformers += len(out_rows)
    finally:
        if writer is not None:
            writer.close()

    if writer is not None:
        tmp_path.replace(dest_path)

    source_summary = stores_root / source_store / "split_summary.txt"
    if source_summary.exists():
        (dest_dir / "split_summary.txt").write_text(
            f"(same molecules/splits as {source_store!r})\n\n"
            + source_summary.read_text()
        )

    logger.info(
        "united-atom store %r -> %r: %d conformer(s), %d H removed, "
        "%d H kept (rdkit declined)",
        source_store,
        dest_store,
        n_conformers,
        n_h_removed,
        n_h_kept,
    )


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
