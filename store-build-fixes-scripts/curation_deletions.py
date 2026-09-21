"""Did the old 0.4 e criterion delete anything it should not have?

``curation_risk.py`` measures the *curated* store, so it can only describe
survivors.  This script answers the complementary question, and needs the
**uncurated** parse to do it -- the state that exists only between
``parse_dash_molecules`` and ``curate_conformers``, and which
``prepare-store --keep-uncurated`` now keeps as
``molecules.parquet.uncurated``.

It replays the OLD rule (group by ``dash_id``; a conformer survives when it
agrees within 0.4 e with at least one sibling) and classifies what that rule
removed.  Stereo perception changes no charge and no atom count, so replaying
it on a freshly parsed store must reproduce the recorded 2,247 conformers and
86 identifiers exactly; a mismatch means the harness is not faithful and the
counts below should not be trusted.

``curation-key-fix.md`` section 4's three questions:

  1. of the removed conformers, how many sit in an identifier covering more
     than one structure?
  2. of those, how many agree with every conformer of their OWN structure and
     disagree only across a structure boundary?  Those are wrongful deletions
     -- records the new grouping keeps and the old one threw away.
  3. of the identifiers removed entirely, how many are mixed identifiers in
     which each structure held a single conformer?  Those may be two sound
     molecules discarded for failing to corroborate each other.

Usage:  python store-build-fixes-scripts/curation_deletions.py <parquet>
Run from the repository root.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from rdkit import RDLogger

sys.path.insert(0, "experiments")
from experiments.collapse import collapse_key  # noqa: E402
from experiments.data import blob_to_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")

THRESHOLD = 0.4


def main(path: str) -> None:
    df = pd.read_parquet(path, columns=["dash_id", "mol"])
    print(f"uncurated rows: {len(df):,}")

    mols = [blob_to_mol(b) for b in df["mol"]]
    charges = [
        np.array(
            [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()], dtype=np.float64
        )
        for mol in mols
    ]
    keys = [collapse_key(mol) for mol in mols]
    df = df.assign(key=keys)

    def agrees(i: int, j: int) -> bool:
        a, b = charges[i], charges[j]
        if a.shape != b.shape:
            return False
        return float(np.abs(a - b).max()) <= THRESHOLD

    removed: list[int] = []
    dropped_ids: list[str] = []
    for dash_id, positions in df.groupby("dash_id", sort=False).groups.items():
        rows = list(positions)
        survivors: set[int] = set()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if agrees(rows[i], rows[j]):
                    survivors.update((rows[i], rows[j]))
        if not survivors:
            dropped_ids.append(dash_id)
        removed.extend(r for r in rows if r not in survivors)

    print(f"\nold rule: removed {len(removed):,} conformer(s), "
          f"{len(dropped_ids):,} identifier(s) entirely")
    print("  (expected 2,247 and 86 on the real corpus -- a mismatch means "
          "this replay is not faithful)")

    n_structs = df.groupby("dash_id")["key"].nunique()
    mixed = set(n_structs[n_structs > 1].index)

    # --- question 1 --------------------------------------------------------
    in_mixed = [r for r in removed if df.at[r, "dash_id"] in mixed]
    print(f"\n1. removed conformers inside a mixed identifier: {len(in_mixed):,}"
          f" of {len(removed):,}")

    # --- question 2 --------------------------------------------------------
    by_key: dict[tuple, list[int]] = {}
    for r in range(len(df)):
        by_key.setdefault((df.at[r, "dash_id"], df.at[r, "key"]), []).append(r)

    wrongful = []
    for r in in_mixed:
        own = [s for s in by_key[(df.at[r, "dash_id"], df.at[r, "key"])] if s != r]
        if own and all(agrees(r, s) for s in own):
            wrongful.append(r)
    print(f"2. of those, agreeing with EVERY conformer of their own structure "
          f"(wrongful deletions): {len(wrongful):,}")
    for r in wrongful[:10]:
        print(f"     {df.at[r, 'dash_id']}  row {r}")

    # --- question 3 --------------------------------------------------------
    all_solo = [
        d
        for d in dropped_ids
        if d in mixed
        and all(
            len(rows) == 1
            for (dash_id, _), rows in by_key.items()
            if dash_id == d
        )
    ]
    print(f"3. identifiers dropped entirely that are mixed with every "
          f"structure holding one conformer: {len(all_solo):,} of "
          f"{len(dropped_ids):,}")
    for d in all_solo[:10]:
        print(f"     {d}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
