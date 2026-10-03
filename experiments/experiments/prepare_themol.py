"""Build a store from THEMol's MBIS subset (ByteDance-Seed/THEMol on Hugging
Face; arXiv:2605.14973): PBE0/def2-TZVPD MBIS charges, DZVP for iodine, on
3,082,151 molecules with one geometry each.

The counterpart of ``prepare_dash``, and deliberately built on its stereo
machinery: ``_finalise_stereo`` perceives stereochemistry from the record's
own coordinates and writes the rigorous CIP labels, so a THEMol ``Mol`` and a
DASH ``Mol`` reach featurisation in the same state. Two things differ.

**The graph comes from a SMILES, the stereo from the geometry.** An HDF5
record carries atomic numbers, coordinates, charges and an atom-mapped SMILES,
but no bonds, so connectivity, bond orders and formal charges are read from
``mapped_isomeric_smiles`` and its atoms renumbered by map number onto the
HDF5 rows. That SMILES also *reports* a stereochemistry. It is removed before
perception, so what the store holds is what the geometry the charges were
computed on says, and the report is compared against it afterwards
(``stereo_check``). The isomeric SMILES is used rather than the non-isomeric
one because it also carries isotopes -- deuterated records are common, and
the non-isomeric SMILES silently turns ``[2H]`` into ``H``.

**There is no curation stage.** ``prepare_dash.curate_conformers`` judges an
atom against the same atom in sibling conformers; THEMol stores one geometry
per molecule, so the criterion has nothing to compare.

The eight HDF5 files (31 GB) are downloaded from Hugging Face when no local
copy is given, pinned to one revision of the repository and verified against
its published sizes and SHA-256 digests (``download_themol``). The data is
licensed CC BY-NC 4.0.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# The repository revision the sizes and digests below were read from, so that
# a later push to the dataset cannot change what a rebuild downloads.
REVISION = "e029fa677b2638b29efa5930a1173d4e41abcab8"
DOWNLOAD_BASE_URL = (
    f"https://huggingface.co/datasets/ByteDance-Seed/THEMol/resolve/{REVISION}/MBIS"
)
# (bytes, SHA-256) per file, from the Hugging Face tree API at REVISION.
EXPECTED_FILES = {
    "mbis_0.h5": (
        4_135_202_860,
        "c307229b4a7c47654a51203dbbf51e93951de817fa04ecd41cbb6603efc3d5a2",
    ),
    "mbis_1.h5": (
        4_145_984_100,
        "d2a6b469fec5730626230244c98c56f28c77777d8932e8093837e8947a623452",
    ),
    "mbis_2.h5": (
        3_556_407_192,
        "b10313baf25ecc9b2b98d03cc01d012353c719fb0f3837093d361b34133e913d",
    ),
    "mbis_3.h5": (
        3_549_593_228,
        "b364c682c2d871951c80db3d0c5daac5aebfe6cc09596e29ed0a64e4d0adc557",
    ),
    "mbis_4.h5": (
        3_553_728_336,
        "6bf24bcd99def7544e4875bee9a7a808abe2bb1b2120d81b9f36eb23f28f3549",
    ),
    "mbis_5.h5": (
        3_550_420_692,
        "7b76913e297d8495d4d5ae019f69b5b42982a364fd8f13cc0bf202b4361de836",
    ),
    "mbis_6.h5": (
        3_548_145_724,
        "3841f123e4445019d5f786b23024d7470abf6fdf5ce0678c47201e150dc47b7e",
    ),
    "mbis_7.h5": (
        3_549_135_012,
        "aa7b3d568e282d1ac55ea0fdf6c6865e99dedc2a3e975b2892a1a3fa2a1f3766",
    ),
}
SHARD_FILES = tuple(EXPECTED_FILES)
EXPECTED_RECORDS = 3_082_151
ID_COLUMNS = ("themol_id",)
STEREO_SUMMARY = "stereo_check_summary.txt"
PARQUET_BATCH_SIZE = 50_000
CHUNK_SIZE = 1 << 20  # 1 MiB

logger = logging.getLogger("experiments")


def download_themol(
    dest_dir: Path,
    *,
    base_url: str = DOWNLOAD_BASE_URL,
    expected: dict[str, tuple[int, str]] = EXPECTED_FILES,
) -> Path:
    """Stream each file in ``expected`` from ``base_url`` into ``dest_dir``,
    verifying size and SHA-256 against the published values, and return
    ``dest_dir``. Idempotent in the way ``prepare_dash.download_dash_sdf`` is:
    a file already present with the expected size is kept without hashing
    (a full SHA-256 pass over 31 GB on every call would be needlessly slow),
    and a corrupted download is caught by its digest the next time one is
    actually fetched.

    ``resolve/<revision>`` redirects to Hugging Face's CDN, which
    ``urllib`` follows on its own; the repository is not gated, so no token
    is needed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, (expected_bytes, expected_sha256) in expected.items():
        out_path = dest_dir / name
        if out_path.exists() and out_path.stat().st_size == expected_bytes:
            logger.info(
                "%s already present with the expected size; skipping download",
                out_path,
            )
            continue

        sha256 = hashlib.sha256()
        with (
            urllib.request.urlopen(f"{base_url}/{name}") as response,
            out_path.open("wb") as f,
        ):
            while chunk := response.read(CHUNK_SIZE):
                f.write(chunk)
                sha256.update(chunk)

        actual_bytes = out_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"downloaded {name}: {actual_bytes} bytes, expected "
                f"{expected_bytes}; download incomplete or corrupted"
            )
        actual_sha256 = sha256.hexdigest()
        if actual_sha256 != expected_sha256:
            out_path.unlink()
            raise ValueError(
                f"downloaded {name} sha256 {actual_sha256} != expected "
                f"{expected_sha256}; download corrupted"
            )
        logger.info("downloaded %s (sha256 %s)", out_path, actual_sha256)
    return dest_dir


def _mol_from_mapped_smiles(smiles: str) -> Any:
    """Parse an atom-mapped SMILES with its hydrogens kept, atoms renumbered
    so that map number ``k`` becomes index ``k - 1`` -- the HDF5 row order --
    and the map numbers then cleared, since they would otherwise enter the
    canonical SMILES that ``collapse_key`` builds."""
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("SMILES did not parse")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if sorted(maps) != list(range(1, len(maps) + 1)):
        raise ValueError("atom map numbers are not 1..N")
    order = [0] * len(maps)
    for idx, number in enumerate(maps):
        order[number - 1] = idx
    mol = Chem.RenumberAtoms(mol, order)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _stereo_elements(mol: Any) -> tuple[set[int], set[int]]:
    """Indices of the tetrahedral centres and the double bonds ``mol``
    specifies. Non-tetrahedral tags are left out: ``_finalise_stereo``
    clears them on the perceived side, so they could only ever appear on the
    reported one."""
    from rdkit import Chem

    tetrahedral = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    unset = {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}
    atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() in tetrahedral}
    bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetStereo() not in unset}
    return atoms, bonds


def _restricted_smiles(mol: Any, atoms: set[int], bonds: set[int]) -> str:
    """Canonical SMILES of ``mol`` carrying only the stereo of ``atoms`` and
    ``bonds``.

    Built by putting those elements back onto the bare graph, not by clearing
    the rest off ``mol``, because clearing leaves residue the SMILES writer
    reads. Cached stereo ranks let a perceived spiro centre whose tag had been
    reset still shape the output, so the restricted molecules differed on the
    very element the restriction had removed. Bond directions let a double
    bond whose flag had been reset be re-derived, so a report with no stereo
    at all read as a conflict with its own geometry. Clearing the directions
    as well cannot fix that, since the writer then drops every E/Z and an E
    and a Z record compare equal.

    So the graph is stripped bare, the chosen tags and double-bond flags are
    set on it, and the single-bond directions are regenerated from those flags
    alone. A flag is set as cis/trans relative to the bond's stereo atoms;
    ``STEREOE``/``STEREOZ`` read that way, since RDKit's stereo atoms for
    them are the CIP-ranked highest neighbours -- checked on 1,142 real
    reported bonds, each E/Z agreeing with the cis/trans that 3D perception
    gave the same bond, and each disagreeing once inverted.
    """
    from rdkit import Chem

    cis = {Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ}
    tags = {i: mol.GetAtomWithIdx(i).GetChiralTag() for i in atoms}
    flags = {}
    for i in bonds:
        bond = mol.GetBondWithIdx(i)
        refs = tuple(bond.GetStereoAtoms())
        if len(refs) == 2:
            is_cis = bond.GetStereo() in cis
            flags[i] = (
                Chem.BondStereo.STEREOCIS if is_cis else Chem.BondStereo.STEREOTRANS,
                refs,
            )

    bare = Chem.Mol(mol)
    bare.RemoveAllConformers()
    Chem.RemoveStereochemistry(bare)
    bare.ClearComputedProps()
    for atom in bare.GetAtoms():
        for name in ("_CIPRank", "_CIPCode", "_ChiralityPossible"):
            atom.ClearProp(name)
        atom.SetChiralTag(tags.get(atom.GetIdx(), Chem.ChiralType.CHI_UNSPECIFIED))
    for bond in bare.GetBonds():
        bond.ClearProp("_CIPCode")
        bond.SetBondDir(Chem.BondDir.NONE)
        if bond.GetIdx() in flags:
            stereo, refs = flags[bond.GetIdx()]
            bond.SetStereoAtoms(*refs)
            bond.SetStereo(stereo)
    Chem.SetDoubleBondNeighborDirections(bare)
    return Chem.MolToSmiles(bare)


def stereo_check(perceived: Any, reported: Any) -> tuple[str, Counter]:
    """Compare the stereochemistry perceived from 3D with the one the SMILES
    reports; both ``Mol``s must come from one parse, so that an index names
    the same atom or bond in each.

    The two sides can differ in two ways, and they are told apart. One
    specifies an element the other leaves open: the geometry settles what the
    SMILES does not say, or the SMILES states what the geometry does not
    settle. Or both specify the same elements and describe different
    stereoisomers. Only the second is a contradiction, so each side is first
    restricted to the elements both specify, and the restricted molecules are
    compared by canonical isomeric SMILES.

    Canonical SMILES rather than index-by-index parities, because the
    question is whether the two describe one stereoisomer, and a symmetric
    molecule can be one stereoisomer under two different sets of parities:
    in spiro[5.5]undecane-3,9-diol, inverting one carbinol centre gives back
    the same molecule. And canonical SMILES rather than CIP labels, because a
    chiral tag need not carry one: the carbinol centres of that same diol are
    tagged and have no CIP descriptor, so a comparison of labels would not
    see them at all.

    ``perceived_only`` does not by itself mean the report is incomplete. On
    the THEMol record 016d2c483c475214840c8e11de079232 the extra tag sits on
    a spiro carbon whose two azetidine branches are exchanged by a symmetry
    of the ring; inverting it gives back the same molecule, so the centre is
    not stereogenic and the report was right to leave it open.

    Returns the verdict and per-element counts. The verdict is ``conflict``
    if the restricted molecules differ, otherwise ``agree`` when both sides
    specify the same elements, or else whichever side specified more:
    ``perceived_only``, ``reported_only``, or ``both_only`` when each has
    elements the other lacks. The counts are ``{atom,bond}_{common,
    perceived_only,reported_only}``.
    """
    p_atoms, p_bonds = _stereo_elements(perceived)
    r_atoms, r_bonds = _stereo_elements(reported)
    common_atoms, common_bonds = p_atoms & r_atoms, p_bonds & r_bonds

    counts: Counter = Counter()
    for kind, p, r in (("atom", p_atoms, r_atoms), ("bond", p_bonds, r_bonds)):
        counts[f"{kind}_common"] += len(p & r)
        counts[f"{kind}_perceived_only"] += len(p - r)
        counts[f"{kind}_reported_only"] += len(r - p)

    perceived_extra = counts["atom_perceived_only"] + counts["bond_perceived_only"]
    reported_extra = counts["atom_reported_only"] + counts["bond_reported_only"]
    if _restricted_smiles(perceived, common_atoms, common_bonds) != (
        _restricted_smiles(reported, common_atoms, common_bonds)
    ):
        verdict = "conflict"
    elif perceived_extra and reported_extra:
        verdict = "both_only"
    elif perceived_extra:
        verdict = "perceived_only"
    elif reported_extra:
        verdict = "reported_only"
    else:
        verdict = "agree"
    return verdict, +counts


def build_record_mol(
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    charges: np.ndarray,
    mapped_smiles: str,
) -> tuple[Any, str, Counter]:
    """One store ``Mol`` from one HDF5 record's arrays, with its stereo check.

    The graph is ``mapped_smiles``'s, renumbered onto the HDF5 rows; the
    atomic numbers must then agree row for row, which is what makes the
    coordinates and charges belong to the atoms they are attached to. The
    reported stereochemistry is copied aside and removed -- tags, bond flags
    and the bond directions a SMILES parse leaves on single bonds, any of
    which would otherwise be read back as a declaration -- so that
    ``_finalise_stereo`` perceives every element from the coordinates.

    Raises ``ValueError`` for a record that cannot be built.
    """
    from rdkit import Chem

    from experiments.prepare_dash import _finalise_stereo

    mol = _mol_from_mapped_smiles(mapped_smiles)
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if z != [int(v) for v in atomic_numbers]:
        raise ValueError("SMILES atoms do not match the record's atomic numbers")
    if len(charges) != len(z) or len(coords) != len(z):
        raise ValueError("array lengths do not match the atom count")

    reported = Chem.Mol(mol)
    Chem.AssignStereochemistry(reported, cleanIt=True, force=True)

    Chem.RemoveStereochemistry(mol)
    conformer = Chem.Conformer(len(z))
    conformer.SetPositions(np.asarray(coords, dtype=np.float64))
    conformer.Set3D(True)
    mol.AddConformer(conformer, assignId=True)
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", float(charge))

    _finalise_stereo(mol)
    verdict, counts = stereo_check(mol, reported)
    return mol, verdict, counts


def store_blob(mol: Any) -> bytes:
    """``mol`` serialised as the store holds it. As in ``prepare_dash``: the
    mol-level properties a parse left behind are dropped, and the marker that
    the rigorous CIP labeler has already run is set. Mutates ``mol``."""
    from experiments.data import mol_to_blob
    from sieve.io.rdkit_adapter import CIP_LABELED_PROP

    for name in list(mol.GetPropNames()):
        mol.ClearProp(name)
    mol.SetBoolProp(CIP_LABELED_PROP, True)
    return mol_to_blob(mol)


def _parse_one_group(themol_id: str, group: Any, h5_name: str) -> tuple[dict, Counter]:
    """One parquet row from one HDF5 group, plus the counts it contributes to
    the stereo summary. A record that cannot be built returns an empty row
    and a ``skipped`` count rather than raising, as in ``prepare_dash``: a
    malformed record should not abort a three-million-record parse."""
    from rdkit import Chem

    from experiments.geometry import geometry_record

    smiles = group["mapped_isomeric_smiles"][()].decode()
    try:
        mol, verdict, counts = build_record_mol(
            group["atomic_numbers"][:, 0],
            group["coords"][:],
            group["mbis_info/atomic_charge"][:, 0],
            smiles,
        )
    except ValueError as exc:
        logger.warning("%s: %s; skipping", themol_id, exc)
        return {}, Counter(skipped=1)

    counts[f"verdict_{verdict}"] += 1
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        counts["isotope_labelled"] += 1
    net_charge = float(Chem.GetFormalCharge(mol))

    geometry = geometry_record(mol)
    row = {
        "themol_id": themol_id,
        "h5_file": h5_name,
        "smiles": smiles,
        "mol": store_blob(mol),
        "net_charge": net_charge,
        "stereo_check": verdict,
        **geometry,
    }
    return row, counts


def _schema():
    import pyarrow as pa

    from experiments.geometry import arrow_fields

    return pa.schema(
        [
            ("themol_id", pa.string()),
            ("h5_file", pa.string()),
            ("smiles", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
            ("stereo_check", pa.string()),
            *arrow_fields(),
        ]
    )


def parse_themol_shard(
    h5_path: Path, out_path: Path, *, limit: int | None = None
) -> Counter:
    """Parse one HDF5 file into one parquet file, in the file's own group
    order, and return its summary counts. ``limit`` stops after that many
    groups, for a quick build against the real data."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    schema = _schema()
    totals: Counter = Counter()
    batch: list[dict] = []
    with h5py.File(h5_path, "r") as f, pq.ParquetWriter(out_path, schema) as writer:
        for n, themol_id in enumerate(f):
            if limit is not None and n >= limit:
                break
            row, counts = _parse_one_group(themol_id, f[themol_id], h5_path.name)
            totals.update(counts)
            if row:
                batch.append(row)
            if len(batch) >= PARQUET_BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    return totals


def stereo_summary(totals: Counter) -> str:
    """The text of ``stereo_check_summary.txt``: records by verdict, then
    stereo elements by which side specifies them."""
    n_records = sum(v for k, v in totals.items() if k.startswith("verdict_"))
    lines = [
        "stereochemistry perceived from 3D vs reported in mapped_isomeric_smiles",
        f"records: {n_records} ({totals['skipped']} skipped)",
        f"isotope-labelled records: {totals['isotope_labelled']}",
        "",
        "records by verdict:",
    ]
    verdicts = ("agree", "perceived_only", "reported_only", "both_only", "conflict")
    for verdict in verdicts:
        count = totals[f"verdict_{verdict}"]
        share = count / n_records if n_records else 0.0
        lines.append(f"  {verdict:<15} {count:>9} ({share:.4%})")
    lines.append("")
    lines.append("stereo elements (centres, double bonds) by who specifies them:")
    for kind in ("atom", "bond"):
        for side in ("common", "perceived_only", "reported_only"):
            lines.append(f"  {kind} {side:<15} {totals[f'{kind}_{side}']:>9}")
    return "\n".join(lines)


def parse_themol(
    source_dir: Path,
    out_path: Path,
    *,
    workers: int = len(SHARD_FILES),
    limit_per_shard: int | None = None,
) -> Counter:
    """Parse every HDF5 file under ``source_dir`` into ``out_path``, one
    worker process per file, and return the combined summary counts.

    Each worker writes its own part file; the parts are then concatenated in
    ``SHARD_FILES`` order, so the row order does not depend on which worker
    finished first. As in ``prepare_dash.parse_dash_molecules``, nothing
    appears at ``out_path`` until the whole parse has succeeded.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow.parquet as pq

    paths = [source_dir / name for name in SHARD_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"{source_dir} lacks {missing}")

    parts_dir = out_path.parent / ".parts"
    parts_dir.mkdir(exist_ok=True)
    parts = [parts_dir / f"{p.stem}.parquet" for p in paths]
    # Spawned, not forked: by now this process runs BLAS and Arrow threads,
    # and forking a threaded process can deadlock the child. Spawning
    # re-imports the calling script in each worker, so a script calling this
    # needs an ``if __name__ == "__main__"`` guard; ``python -m experiments``
    # has one.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(parse_themol_shard, p, part, limit=limit_per_shard)
            for p, part in zip(paths, parts, strict=True)
        ]
        results = [future.result() for future in futures]

    totals: Counter = Counter()
    for path, counts in zip(paths, results, strict=True):
        logger.info("parsed %s: %s", path.name, dict(counts))
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

    n_records = sum(v for k, v in totals.items() if k.startswith("verdict_"))
    if limit_per_shard is None and n_records + totals["skipped"] != EXPECTED_RECORDS:
        logger.warning(
            "read %d records, expected %d",
            n_records + totals["skipped"],
            EXPECTED_RECORDS,
        )
    return totals


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    source_dir: Path | None = None,
    train: float = 0.9,
    val: float = 0.0,
    test: float = 0.1,
    n_shards: int = 25,
    workers: int = len(SHARD_FILES),
    limit_per_shard: int | None = None,
    stop_before_split: bool = False,
) -> None:
    """Ensure ``store_name`` is downloaded, parsed, and has
    ``split``/``cluster``/``shard`` columns, idempotently at each stage. If
    ``source_dir`` is given, its HDF5 files are parsed instead of downloading
    a fresh copy into the store directory via ``download_themol`` (a
    ``ValueError`` is raised if it does not exist).

    The split is ``prepare_dash.assign_splits`` keyed on ``themol_id``, the
    record's UUID. Its Butina clustering is quadratic in the number of
    molecules and THEMol has nine times DASH's, so ``stop_before_split`` is
    the way to inspect the parse, and ``cluster-report`` the way to choose
    ``n_shards``, before paying for it.
    """
    import pyarrow.parquet as pq

    from experiments.prepare_dash import assign_splits

    store_dir = stores_root / store_name
    store_dir.mkdir(parents=True, exist_ok=True)
    molecules_path = store_dir / "molecules.parquet"

    if source_dir is not None and not source_dir.exists():
        raise ValueError(f"source_dir {source_dir} does not exist")

    if not molecules_path.exists():
        # Unlike prepare_dash, the download is skipped once the store is
        # parsed: 31 GB is not worth re-fetching for a parse that is done.
        if source_dir is None:
            source_dir = download_themol(store_dir)
        totals = parse_themol(
            source_dir,
            molecules_path,
            workers=workers,
            limit_per_shard=limit_per_shard,
        )
        summary = stereo_summary(totals)
        (store_dir / STEREO_SUMMARY).write_text(summary + "\n")
        logger.info("parsed %s:\n%s", store_name, summary)
    else:
        logger.info("%s already parsed; skipping", molecules_path)

    if {"split", "cluster", "shard"} <= set(
        pq.ParquetFile(molecules_path).schema.names
    ):
        logger.info("%s already split; nothing to do", molecules_path)
        return
    if stop_before_split:
        logger.info("%s is parsed; stopping before the split as asked", store_name)
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
