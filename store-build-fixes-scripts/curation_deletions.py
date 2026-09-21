"""Did the old 0.4 e criterion delete anything it should not have?

``curation_risk.py`` measures the *curated* store, so it can only describe
survivors.  This answers the complementary question from the **uncurated**
parse -- the state ``prepare-store --keep-uncurated`` leaves as
``molecules.parquet.uncurated``.

The two removal sets are obtained differently on purpose:

* the **old** ``dash_id`` rule is replayed here, and must reproduce its
  recorded 2,247 conformers / 86 identifiers exactly.  Stereo perception moves
  no charge, no atom count and no atom order, so a mismatch means something
  unrelated changed in the parse and none of the counts below can be trusted.
* the **new** structure-keyed rule is NOT reimplemented.  The curated store is
  its own output, so ``uncurated - curated`` *is* its removal set, with no
  second implementation to keep faithful.

Charges are held twice, and which one is used matters:

* ``raw`` -- record order, as the old rule compared them.
* ``aligned`` -- the key's canonical order (``collapse._canonical_order``), as
  the new rule compares them.  Cross-deposit pairs need not share an atom
  ordering, so classifying a rescued record with raw charges would compare one
  atom against another and find no partner.

``curation-key-fix.md`` section 4, as corrected by
``curation-audit-correction.md``:

  1. of the conformers the old rule removed, how many sit in an identifier
     covering more than one structure?
  2. how many were **wrongly deleted** -- rescued by the new rule *and*
     holding a same-structure conformer they agree with?  Reported with
     whether that partner sits in another deposit, which is what makes the
     deletion attributable to the old grouping rather than to the threshold.
  3. of the identifiers dropped entirely, how many are mixed identifiers in
     which every structure held a single conformer?

It also reports the records the new rule *newly* deletes.  Nothing asked for
them, and they are the evidence that the change tightened the criterion rather
than merely loosening it.

Usage:
    python store-build-fixes-scripts/curation_deletions.py <uncurated> <curated>
Run from the repository root.
"""

from __future__ import annotations

import collections
import sys

import numpy as np
import pandas as pd
from rdkit import RDLogger

sys.path.insert(0, "experiments")
from experiments.collapse import _canonical_order, collapse_key
from experiments.data import blob_to_mol

RDLogger.DisableLog("rdApp.*")

THRESHOLD = 0.4


def main(uncurated: str, curated: str) -> None:
    unc = pd.read_parquet(uncurated, columns=["dash_id", "conf_id", "mol"])
    cur = pd.read_parquet(curated, columns=["dash_id", "conf_id"])
    print(f"uncurated rows: {len(unc):,}   curated rows: {len(cur):,}")

    raw: list[np.ndarray] = []
    aligned: list[np.ndarray] = []
    keys: list[str] = []
    for blob in unc["mol"]:
        mol = blob_to_mol(blob)
        key = collapse_key(mol)
        q = np.array(
            [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()], dtype=np.float64
        )
        raw.append(q)
        aligned.append(q[_canonical_order(mol, key)])
        keys.append(key)

    def agrees(charges: list[np.ndarray], i: int, j: int) -> bool:
        a, b = charges[i], charges[j]
        if a.shape != b.shape:
            return False
        return float(np.abs(a - b).max()) <= THRESHOLD

    # --- the old rule, replayed ------------------------------------------
    old_removed: set[int] = set()
    dropped_ids: list[str] = []
    for dash_id, positions in unc.groupby("dash_id", sort=False).groups.items():
        rows = list(positions)
        survivors: set[int] = set()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if agrees(raw, rows[i], rows[j]):
                    survivors.update((rows[i], rows[j]))
        if not survivors:
            dropped_ids.append(dash_id)
        old_removed.update(r for r in rows if r not in survivors)

    ok = len(old_removed) == 2247 and len(dropped_ids) == 86
    print(
        f"\nold rule replayed: removed {len(old_removed):,} conformer(s), "
        f"{len(dropped_ids):,} identifier(s) entirely "
        f"-- {'MATCHES the record (2,247 / 86)' if ok else '*** MISMATCH ***'}"
    )
    if not ok:
        raise SystemExit(
            "the replay does not reproduce the recorded figures; something "
            "unrelated changed in the parse and no count below is trustworthy"
        )

    # --- the new rule, observed rather than reimplemented ------------------
    kept = set(zip(cur["dash_id"], cur["conf_id"], strict=True))
    new_removed = {
        i
        for i, pair in enumerate(zip(unc["dash_id"], unc["conf_id"], strict=True))
        if pair not in kept
    }
    rescued = old_removed - new_removed
    newly_deleted = new_removed - old_removed
    print(f"new rule removed (observed): {len(new_removed):,}")
    print(f"  rescued       (old deleted, new keeps): {len(rescued):,}")
    print(f"  newly deleted (old kept, new deletes) : {len(newly_deleted):,}")
    print(
        f"  net change in curation_summary.txt: "
        f"{len(new_removed) - len(old_removed):+,} "
        f"-- a difference of two larger movements, not a small correction"
    )

    by_key: dict[str, list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    n_structs = unc.assign(key=keys).groupby("dash_id")["key"].nunique()
    mixed = set(n_structs[n_structs > 1].index)

    # --- question 1 --------------------------------------------------------
    in_mixed = [r for r in old_removed if unc.at[r, "dash_id"] in mixed]
    print(
        f"\n1. removed conformers inside a mixed identifier: "
        f"{len(in_mixed):,} of {len(old_removed):,}"
    )

    # --- question 2 --------------------------------------------------------
    wrongful, cross_only, unjudged = [], 0, 0
    for r in sorted(rescued):
        partners = [s for s in by_key[keys[r]] if s != r and agrees(aligned, r, s)]
        if not partners:
            unjudged += 1
            continue
        wrongful.append(r)
        if all(unc.at[s, "dash_id"] != unc.at[r, "dash_id"] for s in partners):
            cross_only += 1
    print(
        f"2. WRONGLY DELETED -- rescued and holding an agreeing same-structure "
        f"conformer: {len(wrongful):,}"
    )
    print(
        f"     of those, every agreeing partner is in ANOTHER deposit: "
        f"{cross_only:,}  <- attributable to the grouping, not the threshold"
    )
    print(
        f"     remaining rescued, with no agreeing same-structure partner: "
        f"{unjudged:,}  (kept as unjudged solo structures by policy)"
    )
    for r in wrongful[:10]:
        print(f"       {unc.at[r, 'dash_id']} {unc.at[r, 'conf_id']}")

    # --- question 3 --------------------------------------------------------
    per_id_key: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i in range(len(unc)):
        per_id_key[(unc.at[i, "dash_id"], keys[i])].append(i)
    all_solo = [
        d
        for d in dropped_ids
        if d in mixed
        and all(len(v) == 1 for (dash_id, _), v in per_id_key.items() if dash_id == d)
    ]
    print(
        f"3. identifiers dropped entirely that are mixed with every structure "
        f"holding one conformer: {len(all_solo):,} of {len(dropped_ids):,}"
    )
    for d in all_solo[:10]:
        print(f"     {d}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
