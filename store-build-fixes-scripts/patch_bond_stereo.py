"""Patch the DASH store for the pseudo-asymmetric bond-stereo fix (f43b4c4),
in place of a rebuild.

The fix changes 225 of 1,027,555 rows: a stereogenic double bond whose
configuration the record's own coordinates settle, left unset by the old
gate. A rebuild would re-parse the 8.3 GB SDF, re-curate and re-split, and
leave a store that cannot be diffed cleanly against the one the experiments
ran on. This instead re-finalises every stored ``Mol`` through the same
``_finalise_stereo`` the parse now uses -- which is a byte-for-byte fixed
point on every row the fix does not touch -- and recomputes only what is
derived from the blobs:

1. ``mol``, for the rows whose blob changes;
2. curation, replayed with ``curate_conformers`` on a patched copy of
   ``molecules.parquet.uncurated``, because curation groups by
   ``collapse_key`` and the fix moves rows between keys;
3. ``collapse_key`` and its three counts, through ``collapse_columns``, the
   same code ``annotate_collapse`` runs;
4. ``floor-components.json``, for the train shards whose rows changed.

Nothing else reads the blobs' bond stereo: the split, cluster and shard come
from achiral fingerprints, and no code table in the store names a stereo
attribute (checked below rather than assumed).

Two modes. By default everything is built in ``<store>/bondfix-staging/``
and a report is written there; the store is not touched. ``--apply`` then
moves the staged files into the store, keeping each original beside it with
a ``.pre-bondfix`` suffix, the naming the earlier fixes used. The staging run
refuses to go on if curation would decide differently, since adding rows back
needs a split and shard for them -- a decision, not a patch.

Usage, from the repository root:

    python store-build-fixes-scripts/patch_bond_stereo.py            # stage
    python store-build-fixes-scripts/patch_bond_stereo.py --apply    # swap in
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "experiments")

from experiments.collapse import floor_components
from experiments.cv import load_shards, store_identity
from experiments.data import mol_to_blob
from experiments.prepare_dash import (
    CURATION_SUMMARY,
    UNCURATED_PARQUET,
    _finalise_stereo,
    curate_conformers,
)
from experiments.store_ops import collapse_columns

from sieve.io.rdkit_adapter import CIP_LABELED_PROP

RDLogger.DisableLog("rdApp.*")

STORE = "dash-molecules"
STORES_ROOT = Path("experiments/stores")
STAGING = "bondfix-staging"
SUFFIX = ".pre-bondfix"
FLOORS = "floor-components.json"
REPORT = "bondfix-report.json"
COLLAPSE_COLUMNS = ["collapse_key", "n_collapsed", "n_molecules", "n_enantiomer_forms"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refinalise(blob: bytes) -> bytes:
    """A stored blob as the fixed parse would have written it: the tail of
    ``_parse_one_record`` from ``_finalise_stereo`` on. Stored ``Mol``s carry
    no mol-level property but the marker, so clearing and re-setting it
    reproduces the parse exactly."""
    mol = Chem.Mol(blob)
    _finalise_stereo(mol)
    for name in list(mol.GetPropNames()):
        mol.ClearProp(name)
    mol.SetBoolProp(CIP_LABELED_PROP, True)
    return mol_to_blob(mol)


def _bond_changes(old: bytes, new: bytes) -> list[dict]:
    a, b = Chem.Mol(old), Chem.Mol(new)
    if [x.GetChiralTag() for x in a.GetAtoms()] != [
        x.GetChiralTag() for x in b.GetAtoms()
    ]:
        raise AssertionError("the fix is not meant to move any chiral tag")
    out = []
    for x, y in zip(a.GetBonds(), b.GetBonds(), strict=True):
        before = (x.GetStereo(), tuple(x.GetStereoAtoms()))
        after = (y.GetStereo(), tuple(y.GetStereoAtoms()))
        if before != after:
            out.append(
                {
                    "bond": y.GetIdx(),
                    "atoms": "=".join(
                        (y.GetBeginAtom().GetSymbol(), y.GetEndAtom().GetSymbol())
                    ),
                    "before": str(x.GetStereo()),
                    "after": str(y.GetStereo()),
                    "cip": y.GetPropsAsDict(includePrivate=True).get("_CIPCode"),
                }
            )
    return out


def _work(chunk: tuple[int, list[bytes]]) -> list[tuple[int, bytes, list[dict]]]:
    start, blobs = chunk
    out = []
    for k, blob in enumerate(blobs):
        new = refinalise(blob)
        if new == blob:
            continue
        if refinalise(new) != new:
            raise AssertionError(f"row {start + k}: not a fixed point after patching")
        out.append((start + k, new, _bond_changes(blob, new)))
    return out


def refinalise_column(
    blobs: list[bytes], processes: int
) -> dict[int, tuple[bytes, list]]:
    step = 1000
    chunks = [(i, blobs[i : i + step]) for i in range(0, len(blobs), step)]
    changed: dict[int, tuple[bytes, list]] = {}
    with mp.Pool(processes) as pool:
        for part in pool.imap(_work, chunks):
            for row, new, bonds in part:
                changed[row] = (new, bonds)
    return changed


def check_code_tables(store_dir: Path) -> list[str]:
    """No code table may name a stereo attribute; if one did, its codes would
    have to be rebuilt too, and this script does not do that."""
    tables = sorted(store_dir.glob("sieve-codes*.json"))
    for path in tables:
        spec = json.loads(path.read_text())
        names = set(spec.get("attribute_codes", {})) | set(spec.get("edge_codes", {}))
        stereo = sorted(
            n for n in names if "stereo" in n or "chiral" in n or "cis" in n
        )
        if stereo:
            raise SystemExit(f"{path.name} codes {stereo}; rebuild it, do not patch")
    return [p.name for p in tables]


def compare_groups(old: pd.DataFrame, new: pd.DataFrame, rows: set[int]) -> list[dict]:
    """Every collapse group a changed row belonged to before or after, and
    what happened to it: ``renamed`` (same members, new key), ``split``,
    ``merged``, or ``regrouped`` for anything less tidy."""
    touched_old = set(old.loc[sorted(rows), "collapse_key"])
    old_members = {
        k: frozenset(v)
        for k, v in old.groupby("collapse_key").groups.items()
        if k in touched_old
    }
    new_by_key = new.groupby("collapse_key").groups
    # Looked up for every successor, not only the keys changed rows land on:
    # a group can hold unchanged rows that keep the old key.
    new_members = {k: frozenset(v) for k, v in new_by_key.items()}

    out = []
    for key, members in sorted(old_members.items()):
        successors = sorted({new.at[i, "collapse_key"] for i in members})
        joined = set().union(*(new_members[k] for k in successors)) - members
        if len(successors) == 1 and not joined:
            kind = "renamed" if successors[0] != key else "unchanged"
        elif len(successors) > 1 and not joined:
            kind = "split"
        elif len(successors) == 1 and joined:
            kind = "merged"
        else:
            kind = "regrouped"
        out.append(
            {
                "kind": kind,
                "old_key": key,
                "new_keys": successors,
                "rows": sorted(int(i) for i in members),
                "rows_joined_from_other_groups": sorted(int(i) for i in joined),
                "dash_ids": sorted(set(old.loc[sorted(members), "dash_id"])),
                "split": sorted(set(old.loc[sorted(members), "split"])),
                "shard": sorted(set(old.loc[sorted(members), "shard"])),
            }
        )
    return out


def stage(store_dir: Path, processes: int) -> None:
    staging = store_dir / STAGING
    if staging.exists():
        raise SystemExit(f"{staging} exists; remove it to stage again")
    (staging / STORE).mkdir(parents=True)

    codes = check_code_tables(store_dir)
    log(f"code tables checked, none reads stereo: {codes}")

    log("reading the store")
    old = pd.read_parquet(store_dir / "molecules.parquet")
    log(f"re-finalising {len(old):,} stored rows")
    changed = refinalise_column(old["mol"].tolist(), processes)
    log(f"{len(changed)} rows change")

    new = old.copy()
    for row, (blob, _) in changed.items():
        new.at[row, "mol"] = blob

    # Curation, replayed from the uncurated parse on the real code path.
    log("replaying curation on a patched copy of the uncurated parse")
    uncurated = pd.read_parquet(store_dir / UNCURATED_PARQUET)
    changed_u = refinalise_column(uncurated["mol"].tolist(), processes)
    for row, (blob, _) in changed_u.items():
        uncurated.at[row, "mol"] = blob
    replay_dir = staging / "curation-replay"
    replay_dir.mkdir()
    uncurated.to_parquet(replay_dir / "molecules.parquet")
    del uncurated
    curation_text = curate_conformers(replay_dir)
    kept = pd.read_parquet(
        replay_dir / "molecules.parquet", columns=["dash_id", "conf_id"]
    )
    kept_ids = set(zip(kept["dash_id"], kept["conf_id"], strict=True))
    store_ids = set(zip(old["dash_id"], old["conf_id"], strict=True))
    curation = {
        "summary": curation_text,
        "restored": sorted(kept_ids - store_ids),
        "removed": sorted(store_ids - kept_ids),
        "uncurated_rows_changed": len(changed_u),
    }
    shutil.rmtree(replay_dir)
    if curation["restored"] or curation["removed"]:
        (staging / REPORT).write_text(json.dumps({"curation": curation}, indent=2))
        raise SystemExit(
            f"curation decides differently: {len(curation['restored'])} row(s) "
            f"restored, {len(curation['removed'])} removed. That needs a split "
            f"and shard for the restored rows, which is a decision, not a patch. "
            f"See {staging / REPORT}."
        )
    log(f"curation keeps the same {len(kept_ids):,} rows")

    log("recomputing collapse columns")
    new = collapse_columns(new)
    rows = set(changed)
    groups = compare_groups(old, new, rows)
    column_changes = {
        c: int((old[c] != new[c]).sum()) for c in COLLAPSE_COLUMNS if c in old.columns
    }

    log("writing the staged store")
    new.to_parquet(staging / STORE / "molecules.parquet")

    # Floors: only a train shard whose rows changed can move.
    moved = sorted(
        {s for g in groups for s in g["shard"] if g["kind"] != "unchanged"} - {"test"}
    )
    floors = json.loads((store_dir / FLOORS).read_text())
    if moved:
        log(f"recomputing floor components for {moved}")
        by_shard = load_shards(STORE, moved, stores_root=staging)
        before = {s: floors[s] for s in moved}
        for s in moved:
            floors[s] = floor_components(by_shard[s])
        floor_deltas = {
            s: {k: floors[s][k] - before[s][k] for k in floors[s]} for s in moved
        }
    else:
        floor_deltas = {}
    (staging / FLOORS).write_text(json.dumps(floors, indent=2, sort_keys=True))
    (staging / CURATION_SUMMARY).write_text(curation_text + "\n")

    identity_before = store_identity(STORE, stores_root=STORES_ROOT)
    identity_after = store_identity(STORE, stores_root=staging)

    report = {
        "rows_changed": [
            {
                "row": int(row),
                **{
                    c: (None if pd.isna(old.at[row, c]) else str(old.at[row, c]))
                    for c in ("dash_id", "conf_id", "chembl_id", "split", "shard")
                },
                "bonds": bonds,
            }
            for row, (_, bonds) in sorted(changed.items())
        ],
        "groups": groups,
        "group_kinds": pd.Series([g["kind"] for g in groups]).value_counts().to_dict(),
        "column_changes": column_changes,
        "floor_deltas": floor_deltas,
        "curation": curation,
        "code_tables_checked": codes,
        "store_digest": {
            "before": identity_before["store_digest"],
            "after": identity_after["store_digest"],
        },
    }
    (staging / REPORT).write_text(json.dumps(report, indent=2, default=str))
    summarise(report)
    log(f"staged in {staging}; nothing in the store was touched. --apply swaps it in.")


def summarise(report: dict) -> None:
    rows = report["rows_changed"]
    print()
    print(
        f"rows changed: {len(rows)}  by split: "
        f"{pd.Series([r['split'] for r in rows]).value_counts().to_dict()}"
    )
    print(f"groups touched: {len(report['groups'])}  kinds: {report['group_kinds']}")
    for g in report["groups"]:
        if g["kind"] not in {"renamed", "unchanged"}:
            print(
                f"  {g['kind']:9s} {g['split']} {g['shard']} dash_ids={g['dash_ids']} "
                f"rows={g['rows']} joined={g['rows_joined_from_other_groups']}"
            )
    print(f"collapse-column changes (rows): {report['column_changes']}")
    print(f"floor deltas: {json.dumps(report['floor_deltas'])}")
    same = report["store_digest"]["before"] == report["store_digest"]["after"]
    print(
        f"store digest {'unchanged' if same else 'CHANGES'}: "
        f"{report['store_digest']['before']} -> {report['store_digest']['after']}"
    )
    if not same:
        print(
            "  every cached model's sidecar names the old digest, so each will be "
            "refused as stale and refitted on next use"
        )
    print()


def apply(store_dir: Path) -> None:
    staging = store_dir / STAGING
    if not (staging / REPORT).exists():
        raise SystemExit(f"nothing staged in {staging}; run without --apply first")
    report = json.loads((staging / REPORT).read_text())
    if "rows_changed" not in report:
        raise SystemExit("the staging run stopped early; see its report")
    moves = [
        (staging / STORE / "molecules.parquet", store_dir / "molecules.parquet"),
        (staging / FLOORS, store_dir / FLOORS),
        (staging / CURATION_SUMMARY, store_dir / CURATION_SUMMARY),
    ]
    for _, dest in moves:
        backup = dest.with_name(dest.name + SUFFIX)
        if backup.exists():
            raise SystemExit(f"{backup} exists; this patch was already applied")
    for src, dest in moves:
        dest.rename(dest.with_name(dest.name + SUFFIX))
        src.rename(dest)
        log(f"{dest.name} replaced; original kept as {dest.name}{SUFFIX}")
    (staging / REPORT).rename(store_dir / REPORT)
    shutil.rmtree(staging)
    log(f"applied; report kept at {store_dir / REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="swap the staged files in")
    parser.add_argument("--processes", type=int, default=32)
    args = parser.parse_args()
    store_dir = STORES_ROOT / STORE
    if args.apply:
        apply(store_dir)
    else:
        stage(store_dir, args.processes)


if __name__ == "__main__":
    main()
