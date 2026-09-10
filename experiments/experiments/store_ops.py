"""Dataset-agnostic operations on an already-prepared store: draw smaller
representative stores from a big one (``subsample_store``), divide one into
disjoint folds (``partition_store``), or emit a heavy-atom-only version
(``to_united_atom_store``). Every one of these acts on the store parquet's
own columns, so any dataset whose prep script emits that format inherits
them -- see experiments/README.md's "adding a dataset".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")


def subsample_store(
    source_store: str,
    dest_store: str,
    *,
    stores_root: Path,
    n_molecules: int = 50_000,
    conformers_per_molecule: int = 1,
    seed: int = 0,
    n_stores: int = 1,
) -> str | list[str]:
    """Build one or more smaller, independent stores by subsampling molecules
    from an already-split ``source_store``, reproducing that source's own real
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

    ``n_stores`` > 1 builds that many stores at once, drawing *without
    replacement across all of them*: every split is shuffled exactly once
    and handed out in contiguous blocks, so no molecule ever lands in two
    of the stores and each store still gets the source's own split mix.
    They are named ``dest_store-1`` ... ``dest_store-n``, and the return
    value is the list of their summaries; with the default ``n_stores=1``
    the single store keeps the bare ``dest_store`` name and one summary
    string is returned. Since disjoint stores can't be clamped
    independently, a request that the source can't fill (``n_stores`` times
    a split's own target exceeding what that split has) raises before any
    store is written, rather than silently shrinking the later ones -- the
    single-store case keeps clamping instead, as there is nothing to
    starve.

    ``conformers_per_molecule`` is a *cap*, not a floor: a molecule with
    fewer conformers than requested contributes all of its own (no
    padding/repeats); one with more has exactly that many sampled uniformly
    at random, without replacement -- so a molecule's conformers never span
    two splits (inherited directly from the source's own per-molecule
    split assignment) and a selected molecule is never over-represented
    beyond what was asked for.

    Both molecule selection and conformer selection use
    ``np.random.default_rng(seed)``, so the same ``seed`` reproduces the
    same subsample. Writes ``<store>/molecules.parquet`` (with the same
    schema, ``split`` column included) and a ``split_summary.txt`` -- the
    subsample's own *actually achieved* counts/fractions, for transparency
    against the requested target.
    """
    if n_molecules < 1:
        raise ValueError("n_molecules must be >= 1")
    if conformers_per_molecule < 1:
        raise ValueError("conformers_per_molecule must be >= 1")
    if n_stores < 1:
        raise ValueError("n_stores must be >= 1")

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

    # Every split's own per-store molecule count, resolved up front so that
    # an unfillable multi-store request fails before anything is written.
    picks_per_split: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        keys = keys_by_split.get(split_name, [])
        if not keys:
            continue
        target = round(n_molecules * len(keys) / total_molecules)
        if n_stores > 1:
            if n_stores * target > len(keys):
                raise ValueError(
                    f"{split_name} split of {source_store!r} has only "
                    f"{len(keys)} molecule(s), too few for {n_stores} "
                    f"disjoint store(s) of {target} molecule(s) each "
                    f"({n_stores * target} needed); lower --n-stores or "
                    f"--n-molecules"
                )
            picks_per_split[split_name] = target
            continue
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
        picks_per_split[split_name] = n_pick

    rng = np.random.default_rng(seed)
    # One store's worth of selected row positions per entry.
    selected_positions: list[list[np.ndarray]] = [[] for _ in range(n_stores)]
    for split_name, n_pick in picks_per_split.items():
        keys = keys_by_split[split_name]
        # A single shuffle handed out in contiguous blocks *is* the
        # draw-without-replacement across stores -- no store can be dealt a
        # molecule another one already holds.
        order = rng.permutation(len(keys))
        for store_index in range(n_stores):
            block = order[store_index * n_pick : (store_index + 1) * n_pick]
            for i in block:
                positions = positions_by_key[keys[i]]
                if len(positions) > conformers_per_molecule:
                    positions = rng.choice(
                        positions, size=conformers_per_molecule, replace=False
                    )
                selected_positions[store_index].append(positions)

    summaries = [
        _write_subsample(
            df,
            positions,
            source_store=source_store,
            dest_dir=stores_root
            / (dest_store if n_stores == 1 else f"{dest_store}-{store_index + 1}"),
        )
        for store_index, positions in enumerate(selected_positions)
    ]
    return summaries[0] if n_stores == 1 else summaries


def partition_store(
    source_store: str,
    dest_prefix: str,
    *,
    stores_root: Path,
    n_stores: int,
    conformers_per_molecule: int | None = None,
    seed: int = 0,
) -> str | list[str]:
    """Exhaustively partition *every* molecule in ``source_store`` into
    ``n_stores`` disjoint stores -- unlike ``subsample_store``, nothing is
    left unused: each molecule lands in exactly one destination store.
    Each store still reproduces the source's own real train/val/test
    fractions (measured directly, same convention as ``subsample_store``),
    but molecule assignment is exhaustive division rather than a
    proportional target with clamping -- each split's own molecule list is
    shuffled once and divided into ``n_stores`` near-equal contiguous
    blocks (sizes differ by at most one, via ``divmod``), so every
    molecule in that split is used and none is used twice.

    ``conformers_per_molecule`` defaults to ``None`` here (unlike
    ``subsample_store``'s default of ``1``): unlimited, every conformer of
    every selected molecule is kept. Passing an explicit value caps it the
    same way ``subsample_store`` does (uniform sampling without
    replacement when a molecule has more than the cap).

    Named ``dest_prefix-1`` ... ``dest_prefix-n_stores``, mirroring
    ``subsample_store``'s own ``n_stores`` convention exactly, including
    ``n_stores=1`` keeping the bare ``dest_prefix`` name and returning one
    summary string instead of a list.

    Idempotent as a whole (not store-by-store, unlike ``prepare_store``'s
    own per-stage idempotency): if every destination store already has a
    ``molecules.parquet``, this returns their existing ``split_summary.txt``
    contents straight off disk without touching ``source_store`` at all --
    safe to re-run as part of a larger reproduction script. Assignment is
    computed jointly across all ``n_stores`` in one pass (each split is
    shuffled once and divided as a whole), so a *partially* complete set
    -- e.g. one destination missing after an interrupted prior run -- is
    not treated as done: the entire partition reruns and every destination
    is rewritten, the same as a fresh call.
    """
    if n_stores < 1:
        raise ValueError("n_stores must be >= 1")
    if conformers_per_molecule is not None and conformers_per_molecule < 1:
        raise ValueError("conformers_per_molecule must be >= 1 or None (uncapped)")

    dest_dirs = [
        stores_root / (dest_prefix if n_stores == 1 else f"{dest_prefix}-{i + 1}")
        for i in range(n_stores)
    ]
    if all((d / "molecules.parquet").exists() for d in dest_dirs):
        logger.info(
            "every destination of %r (%d store(s)) already exists; skipping",
            dest_prefix,
            n_stores,
        )
        existing_summaries = [
            (d / "split_summary.txt").read_text().removesuffix("\n") for d in dest_dirs
        ]
        return existing_summaries[0] if n_stores == 1 else existing_summaries

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
    positions_by_key = df.groupby(mol_key).indices

    keys_by_split: dict[str, list[str]] = {}
    for key, positions in positions_by_key.items():
        keys_by_split.setdefault(split_col[positions[0]], []).append(str(key))

    rng = np.random.default_rng(seed)
    selected_positions: list[list[np.ndarray]] = [[] for _ in range(n_stores)]
    for split_name in ("train", "val", "test"):
        keys = keys_by_split.get(split_name, [])
        if not keys:
            continue
        order = rng.permutation(len(keys))
        base, remainder = divmod(len(keys), n_stores)
        start = 0
        for store_index in range(n_stores):
            size = base + (1 if store_index < remainder else 0)
            block = order[start : start + size]
            start += size
            for i in block:
                positions = positions_by_key[keys[i]]
                if (
                    conformers_per_molecule is not None
                    and len(positions) > conformers_per_molecule
                ):
                    positions = rng.choice(
                        positions, size=conformers_per_molecule, replace=False
                    )
                selected_positions[store_index].append(positions)

    summaries = [
        _write_subsample(df, positions, source_store=source_store, dest_dir=dest_dir)
        for dest_dir, positions in zip(dest_dirs, selected_positions, strict=True)
    ]
    return summaries[0] if n_stores == 1 else summaries


def _write_subsample(
    df: Any,
    selected_positions: list[np.ndarray],
    *,
    source_store: str,
    dest_dir: Path,
) -> str:
    """Write the rows at ``selected_positions`` (one array per selected
    molecule) out as ``dest_dir``'s own store, plus its ``split_summary.txt``
    -- the subsample's *actually achieved* counts/fractions. Returns that
    summary text."""
    all_positions = (
        np.sort(np.concatenate(selected_positions))
        if selected_positions
        else np.array([], dtype=np.int64)
    )
    out_df = df.iloc[all_positions].reset_index(drop=True)

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
        dest_dir.name,
        out_mol_key.nunique(),
        len(out_df),
        summary_text,
    )
    return summary_text


def _to_united_atom(
    mol: Any, *, atom_property: str = "MBIScharge"
) -> tuple[Any, int, int]:
    """Remove ``mol``'s own hydrogens via ``Chem.RemoveHs`` (rdkit's own
    default judgment of which H's are safe to strip -- see module docstring
    for what "safe" means: not stereo-defining, no isotope/query, not
    bridging, ...), adding each actually-removed H's own value of
    ``atom_property`` onto the single heavy atom it was bonded to. An H
    rdkit declines to remove is left in place, its own value untouched --
    never forced out. The total is conserved exactly: every removed H's
    value lands on exactly one heavy atom, never dropped. ``atom_property``
    defaults to ``MBIScharge`` (this series' original, still most common,
    target) but works identically for any run's own configured
    ``target.atom_property`` -- the redistribution rule has nothing
    charge-specific about it.

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

    h_value: dict[int, float] = {}
    h_heavy_neighbor: dict[int, int] = {}
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() == 1:
            idx = atom.GetIdx()
            h_value[idx] = atom.GetDoubleProp(atom_property)
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
    for h_idx, value in h_value.items():
        if h_idx in surviving_orig:
            n_kept += 1
            continue
        heavy_idx = h_heavy_neighbor.get(h_idx)
        if heavy_idx is None:
            # No single heavy neighbor (a bridging or isolated H) -- rdkit
            # does not remove these by default, so this branch should be
            # unreachable, but treat it as "kept" defensively rather than
            # silently drop a value with nowhere documented to go.
            n_kept += 1
            continue
        bonus[heavy_idx] = bonus.get(heavy_idx, 0.0) + value
        n_removed += 1

    for atom in ua_mol.GetAtoms():
        atom.SetAtomMapNum(0)
        add = bonus.get(surviving_orig_by_new_idx[atom.GetIdx()])
        if add:
            atom.SetDoubleProp(atom_property, atom.GetDoubleProp(atom_property) + add)

    return ua_mol, n_removed, n_kept


def to_united_atom_store(
    source_store: str,
    dest_store: str,
    *,
    stores_root: Path,
    atom_property: str = "MBIScharge",
) -> None:
    """Build a united-atom (heavy-atom-only, where rdkit allows it) version
    of an already-prepared ``source_store``: every conformer's ``Mol`` goes
    through ``_to_united_atom`` (see its own docstring for the
    redistribution rule and rdkit's "refuses to remove" cases), folding each
    removed H's own value of ``atom_property`` onto its heavy-atom neighbor.
    ``atom_property`` defaults to ``MBIScharge`` -- pass a different name to
    match another dataset's own ``target.atom_property``. Every other
    column (``chembl_id``/``conf_id``/``dash_id``/``net_charge``/``split``,
    plus any dataset-specific identifier column) is copied through
    unchanged -- this is a different chemical representation of the exact
    same conformers, not a re-split or re-sample, so a molecule's split
    assignment is untouched, and a molecule-level column like ``net_charge``
    (not an atom-level quantity) needs no adjustment either.

    Streams the source parquet in ``PARQUET_BATCH_SIZE``-row batches (read
    and write both), so peak memory stays bounded regardless of the source
    store's own size -- the full store's ~1M rows never load into memory at
    once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.data import blob_to_mol, mol_to_blob
    from experiments.prepare_dash import PARQUET_BATCH_SIZE

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
                ua_mol, removed, kept = _to_united_atom(
                    mol, atom_property=atom_property
                )
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
