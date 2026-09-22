"""Every figure the manuscript's Dataset and Curation sections quote, measured
on the store as built.

The repository is ground truth: the paper reports what is measured here, so
this script exists to produce those numbers in one pass rather than have them
re-derived by hand each time the store is rebuilt.

Usage:
    python store-build-fixes-scripts/paper_numbers.py <curated> <uncurated>
Run from the repository root.
"""

from __future__ import annotations

import collections
import sys

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Descriptors, inchi

sys.path.insert(0, "experiments")
from experiments.collapse import _canonical_order, collapse_key
from experiments.data import blob_to_mol

RDLogger.DisableLog("rdApp.*")
THRESHOLD = 0.4


def head(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(curated: str, uncurated: str) -> None:
    cur = pd.read_parquet(
        curated, columns=["dash_id", "split", "cluster", "shard", "collapse_key", "mol"]
    )

    head("sec:the dash charges corpus -- composition")
    print(f"curated conformers : {len(cur):,}")
    print(f"identifiers        : {cur.dash_id.nunique():,}")
    print(f"structures         : {cur.collapse_key.nunique():,}")

    n_atoms = 0
    max_mw = 0.0
    elements: set[str] = set()
    for blob in cur["mol"]:
        mol = blob_to_mol(blob)
        n_atoms += mol.GetNumAtoms()
        max_mw = max(max_mw, Descriptors.MolWt(mol))
        elements.update(a.GetSymbol() for a in mol.GetAtoms())
    print(f"atoms              : {n_atoms:,}")
    print(f"elements ({len(elements)})      : {', '.join(sorted(elements))}")
    print(f"max molar mass     : {max_mw:.1f} g/mol")

    head("sec:the dash charges corpus -- identifier vs structure")
    per = cur.groupby("dash_id")["collapse_key"].nunique()
    mixed = per[per > 1]
    print(f"identifiers covering >1 structure: {len(mixed):,} "
          f"({len(mixed) / len(per):.2%})")
    print(f"  conformers in them             : "
          f"{cur[cur.dash_id.isin(mixed.index)].shape[0]:,}")

    # InChI-layer classification of the mixed identifiers.
    kinds: collections.Counter = collections.Counter()
    for did in mixed.index:
        rows = cur[cur.dash_id == did]
        keys = {}
        for blob in rows["mol"]:
            mol = blob_to_mol(blob)
            keys[collapse_key(mol)] = inchi.MolToInchi(mol)
        vals = [v for v in keys.values() if v]
        if len(set(vals)) <= 1:
            kinds["same InChI (mobile H / tautomer)"] += 1
            continue
        layers = [dict(p.split("=", 1) if "=" in p else (p[0], p[1:])
                       for p in v.split("/")[1:] if p) for v in vals]
        t = len({lay.get("t") for lay in layers}) > 1
        b = len({lay.get("b") for lay in layers}) > 1
        kinds["both /t and /b" if (t and b) else
              "diastereomers (/t)" if t else
              "E/Z isomers (/b)" if b else "other layer"] += 1
    for k, v in kinds.most_common():
        print(f"  {k:36}: {v:,}")

    head("sec:cluster-clean splitting -- repeated structures")
    g = cur.groupby("collapse_key")["dash_id"].nunique()
    multi = g[g > 1]
    n_ids = cur[cur.collapse_key.isin(multi.index)].dash_id.nunique()
    print(f"structures under >1 identifier: {len(multi):,}, "
          f"accounting for {n_ids:,} identifiers")
    tr = set(cur[cur.split == "train"].collapse_key)
    te = set(cur[cur.split == "test"].collapse_key)
    print(f"structures spanning the split : {len(tr & te)}  (must be 0)")

    head("Table 1 -- composition of the curated corpus and its split")
    print(f"{'Split':6} {'Conformers':>12} {'Identifiers':>12} "
          f"{'Structures':>12} {'Fraction':>10}")
    for s in ("train", "test"):
        d = cur[cur.split == s]
        print(f"{s:6} {len(d):>12,} {d.dash_id.nunique():>12,} "
              f"{d.collapse_key.nunique():>12,} {len(d) / len(cur):>10.4f}")
    print(f"{'Total':6} {len(cur):>12,} {cur.dash_id.nunique():>12,} "
          f"{cur.collapse_key.nunique():>12,} {1.0:>10.4f}")

    head("sec:cluster-clean splitting -- clusters and shards")
    train = cur[cur.split == "train"]
    per_mol = train.drop_duplicates("dash_id")
    sizes = per_mol.groupby("cluster").size()
    print(f"train molecules : {len(per_mol):,}")
    print(f"clusters        : {len(sizes):,}")
    print(f"largest cluster : {sizes.max():,} ({sizes.max() / len(per_mol):.4%})")
    print(f"  median {int(sizes.median())}, p90 {int(sizes.quantile(.9))}, "
          f"p99 {int(sizes.quantile(.99))}")
    sh = per_mol.groupby("shard").size()
    print(f"shards          : {len(sh)}  balance {sh.min():,}-{sh.max():,}, "
          f"empty {int((sh == 0).sum())}")

    head("sec:curation of anomalous conformers")
    unc = pd.read_parquet(uncurated, columns=["dash_id", "conf_id", "mol"])
    aligned: list[np.ndarray] = []
    keys: list[str] = []
    for blob in unc["mol"]:
        mol = blob_to_mol(blob)
        key = collapse_key(mol)
        q = np.array(
            [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()], dtype=np.float64
        )
        aligned.append(q[_canonical_order(mol, key)])
        keys.append(key)

    def agrees(i: int, j: int) -> bool:
        a, b = aligned[i], aligned[j]
        return a.shape == b.shape and float(np.abs(a - b).max()) <= THRESHOLD

    violating = dropped_whole = partial = smooth = solo = 0
    n_removed = 0
    sizes_kept: collections.Counter = collections.Counter()
    for _, pos in unc.assign(k=keys).groupby("k", sort=False).groups.items():
        rows = list(pos)
        if len(rows) == 1:
            solo += 1
            sizes_kept[1] += 1
            continue
        survivors: set[int] = set()
        any_disagree = False
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if agrees(rows[i], rows[j]):
                    survivors.update((rows[i], rows[j]))
                else:
                    any_disagree = True
        removed = len(rows) - len(survivors)
        n_removed += removed
        sizes_kept[len(survivors)] += 1
        if any_disagree:
            violating += 1
            if not survivors:
                dropped_whole += 1
            elif removed:
                partial += 1
            else:
                smooth += 1

    print(f"uncurated conformers                : {len(unc):,}")
    print(f"removed                             : {n_removed:,} "
          f"({n_removed / len(unc):.3%})")
    print(f"remaining                           : {len(unc) - n_removed:,}")
    print(f"structures with a disagreeing pair  : {violating:,}")
    print(f"  resolved by removing some         : {partial:,}")
    print(f"  dropped entirely                  : {dropped_whole:,}")
    print(f"  kept whole (smooth continuum)     : {smooth:,}")
    print(f"structures deposited with one conformer (kept, unjudged): {solo:,}")
    print("surviving conformers per structure  : "
          + ", ".join(f"{k}:{v:,}" for k, v in sorted(sizes_kept.items())))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
