"""Compare a rebuilt DASH store against the staged bond-stereo patch.

The staged patch (``patch_bond_stereo.py``, in
``experiments/stores/dash-molecules/bondfix-staging/``) is what the current
store becomes under the fix and nothing else. A rebuild that reproduces it
therefore proves the rebuild changed exactly the fix; one that does not says
where else it moved, before any experiment is rerun on it.

Checked, keyed by ``(dash_id, conf_id)`` so row order cannot matter:

* the row set, allowing exactly the rows the staging's curation replay
  restores or removes;
* every column of ``molecules.parquet``, the ``mol`` blobs byte for byte.
  ``cluster`` is compared as a partition, since a rebuild may number the
  same clusters differently; ``split`` and ``shard`` by label, since the
  fold plan names shards;
* ``floor-components.json``, per shard;
* ``curation_summary.txt`` and ``split_summary.txt``, as text;
* the rebuilt ``molecules.parquet.uncurated`` against the old one: exactly
  the patch's changed rows may differ, and they must equal the staged blobs.

Exits 0 when everything matches, 1 otherwise, printing what differs.

Usage, from the repository root:

    python store-build-fixes-scripts/compare_rebuild.py dash-molecules-rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

STORES = Path("experiments/stores")
CURRENT = STORES / "dash-molecules"
STAGED = CURRENT / "bondfix-staging"
KEY = ["dash_id", "conf_id"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    dup = df.duplicated(KEY)
    if dup.any():
        raise SystemExit(f"{path}: {int(dup.sum())} duplicate {KEY} rows")
    return df.set_index(KEY).sort_index()


def _examples(index: pd.Index, n: int = 5) -> str:
    shown = ", ".join(f"{d}/{c}" for d, c in index[:n])
    return shown + (f" (+{len(index) - n} more)" if len(index) > n else "")


# Counts over a collapse group, which a row curation restored or removed
# changes for its whole group.
_GROUP_COUNTS = {"n_collapsed", "n_molecules", "n_enantiomer_forms"}


def compare_rows(
    rebuilt: pd.DataFrame, staged: pd.DataFrame, restored: set, removed: set
) -> list[str]:
    """``restored``/``removed``: the rows the staging's curation replay says
    the rebuild keeps or drops differently from the store it patched. Those
    rows are expected in one store only, and their groups' counts differ."""
    problems = []
    only_r = set(rebuilt.index.difference(staged.index))
    only_s = set(staged.index.difference(rebuilt.index))
    if only_r != restored or only_s != removed:
        extra_r, extra_s = sorted(only_r - restored), sorted(only_s - removed)
        missing = sorted((restored - only_r) | (removed - only_s))
        in_rebuild, in_staged = (
            _examples(pd.Index(extra_r)),
            _examples(pd.Index(extra_s)),
        )
        problems.append(
            f"row sets differ beyond curation's expected changes: "
            f"{len(extra_r)} unexpected in the rebuild [{in_rebuild}], "
            f"{len(extra_s)} unexpected in the staged store [{in_staged}], "
            f"{len(missing)} expected but absent"
        )
    touched_keys = set(rebuilt.loc[sorted(only_r), "collapse_key"]) | set(
        staged.loc[sorted(only_s), "collapse_key"]
    )
    common = rebuilt.index.intersection(staged.index)
    r, s = rebuilt.loc[common], staged.loc[common]
    untouched = ~r["collapse_key"].isin(touched_keys).values

    columns = sorted(set(r.columns) | set(s.columns))
    for column in columns:
        if column not in r.columns or column not in s.columns:
            problems.append(f"column {column!r} is in only one of the two stores")
            continue
        if column == "cluster":
            # Same partition iff the label pairs define a bijection.
            pairs = pd.DataFrame({"r": r[column].values, "s": s[column].values})
            if (
                pairs.groupby("r")["s"].nunique().max() > 1
                or pairs.groupby("s")["r"].nunique().max() > 1
            ):
                problems.append("cluster: the rebuild partitions rows differently")
            continue
        a, b = r[column], s[column]
        differ = ~((a == b) | (a.isna() & b.isna()))
        if column in _GROUP_COUNTS:
            differ &= untouched
        if differ.any():
            problems.append(
                f"{column}: {int(differ.sum())} row(s) differ "
                f"[{_examples(r.index[differ.values])}]"
            )
    return problems


def compare_floors(rebuilt: Path, staged: Path, skip: set[str]) -> list[str]:
    """Per shard, except the shards in ``skip``: those hold rows curation
    restored or removed, which only the rebuild's floors can count."""
    a, b = json.loads(rebuilt.read_text()), json.loads(staged.read_text())
    if set(a) != set(b):
        return [f"floor-components shards differ: {sorted(set(a) ^ set(b))}"]
    problems = []
    for shard in sorted(set(a) - skip):
        for field in sorted(set(a[shard]) | set(b[shard])):
            x, y = a[shard].get(field), b[shard].get(field)
            if x is None or y is None or abs(x - y) > 1e-9 * max(1.0, abs(y)):
                problems.append(f"floor {shard}/{field}: {x} (rebuild) vs {y} (staged)")
    return problems


def compare_text(name: str, rebuilt: Path, reference: Path) -> list[str]:
    if rebuilt.read_text() != reference.read_text():
        return [f"{name} differs from {reference}"]
    return []


def compare_uncurated(rebuilt_dir: Path, staged: pd.DataFrame) -> list[str]:
    report = json.loads((STAGED / "bondfix-report.json").read_text())
    patched = {(r["dash_id"], r["conf_id"]) for r in report["rows_changed"]}
    old = _load(CURRENT / "molecules.parquet.uncurated")["mol"]
    new = _load(rebuilt_dir / "molecules.parquet.uncurated")["mol"]
    if not old.index.equals(new.index):
        return ["uncurated row sets differ between the rebuild and the old parse"]
    differ = set(old.index[(old != new).values])
    problems = []
    if differ != patched:
        problems.append(
            f"uncurated: {len(differ)} rows differ from the old parse, the patch "
            f"changed {len(patched)}; {len(differ - patched)} unexpected, "
            f"{len(patched - differ)} expected but unchanged"
        )
    wrong = [k for k in differ & patched if new.loc[k] != staged.at[k, "mol"]]
    if wrong:
        problems.append(
            f"uncurated: {len(wrong)} patched rows differ from the staged blob"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "store", help="the rebuilt store's name under experiments/stores"
    )
    args = parser.parse_args()
    rebuilt_dir = STORES / args.store
    if not (STAGED / "bondfix-report.json").exists():
        raise SystemExit(f"no staged patch in {STAGED}; run patch_bond_stereo.py first")

    rebuilt = _load(rebuilt_dir / "molecules.parquet")
    staged = _load(STAGED / "dash-molecules" / "molecules.parquet")
    curation = json.loads((STAGED / "bondfix-report.json").read_text())["curation"]
    restored = {tuple(x) for x in curation["restored"]}
    removed = {tuple(x) for x in curation["removed"]}
    problems = compare_rows(rebuilt, staged, restored, removed)
    skip = set(rebuilt.loc[sorted(restored & set(rebuilt.index)), "shard"]) | set(
        staged.loc[sorted(removed & set(staged.index)), "shard"]
    )
    problems += compare_floors(
        rebuilt_dir / "floor-components.json", STAGED / "floor-components.json", skip
    )
    if restored or removed:
        print(
            f"expected from curation: {len(restored)} row(s) restored, "
            f"{len(removed)} removed; their groups' counts and the floors of "
            f"shards {sorted(skip)} are not compared"
        )
    problems += compare_text(
        "curation_summary.txt",
        rebuilt_dir / "curation_summary.txt",
        STAGED / "curation_summary.txt",
    )
    problems += compare_text(
        "split_summary.txt",
        rebuilt_dir / "split_summary.txt",
        CURRENT / "split_summary.txt",
    )
    problems += compare_uncurated(rebuilt_dir, staged)

    print(f"compared {len(rebuilt):,} rebuilt rows against {len(staged):,} staged")
    if problems:
        print(f"{len(problems)} difference(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("identical: the rebuild is the current store plus exactly the fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
