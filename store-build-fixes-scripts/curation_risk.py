"""Does the 0.4 e criterion compare conformers across different structures?

Grouping is by ``dash_id``, which is not a structure key.  Two harms follow:
a record whose only siblings are a different structure can be deleted for
disagreeing with chemistry rather than with itself (false positive), and a
genuinely failed record can be corroborated by a different molecule (false
negative).  Both are measured here on the curated store.
"""
from __future__ import annotations
import collections
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms
sys.path.insert(0, "experiments")
from experiments.collapse import _strip_stereo
from experiments.data import blob_to_mol
RDLogger.DisableLog("rdApp.*")

THRESHOLD = 0.4
path = sys.argv[1]
df = pd.read_parquet(path, columns=["dash_id", "collapse_key", "mol"])
per_id = df.groupby("dash_id")["collapse_key"].nunique()
mixed = set(per_id[per_id > 1].index)
sub = df[df["dash_id"].isin(mixed)].reset_index(drop=True)
print(f"identifiers covering >1 structure: {len(mixed):,}; conformers: {len(sub):,}")

# A structure represented by a single conformer inside a mixed identifier has
# no same-structure sibling: at curation time it could only be kept by
# agreeing with a different molecule, or else removed.
sizes = sub.groupby(["dash_id", "collapse_key"]).size()
lonely = sizes[sizes == 1]
print(f"structures with no same-structure sibling in their identifier: {len(lonely):,}")
print(f"  ... spread over {lonely.index.get_level_values(0).nunique():,} identifiers")

# Do cross-structure conformers actually agree within the threshold?
rng = np.random.default_rng(0)
ids = sorted(mixed)
sample = [ids[i] for i in rng.choice(len(ids), size=min(400, len(ids)), replace=False)]
agree_cross = disagree_cross = 0
agree_same = disagree_same = 0
skipped = 0
for did in sample:
    rows = sub[sub["dash_id"] == did]
    mols = {i: blob_to_mol(b) for i, b in zip(rows.index, rows["mol"], strict=True)}
    keys = dict(zip(rows.index, rows["collapse_key"], strict=True))
    order, charges = {}, {}
    frames = {}
    for i, m in mols.items():
        s = _strip_stereo(m)
        o = np.argsort(np.asarray(CanonicalRankAtoms(s)))
        order[i] = o
        frames[i] = (Chem.MolToSmiles(s),)
        charges[i] = np.array([m.GetAtomWithIdx(int(a)).GetDoubleProp("MBIScharge") for a in o])
    idx = list(mols)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if frames[i] != frames[j] or charges[i].shape != charges[j].shape:
                skipped += 1
                continue
            worst = float(np.abs(charges[i] - charges[j]).max())
            same_structure = keys[i] == keys[j]
            if same_structure:
                if worst <= THRESHOLD: agree_same += 1
                else: disagree_same += 1
            else:
                if worst <= THRESHOLD: agree_cross += 1
                else: disagree_cross += 1

print(f"\nsampled {len(sample)} mixed identifiers, pairs compared "
      f"({skipped} skipped for a differing stripped graph):")
print(f"  same-structure pairs : {agree_same:5d} agree, {disagree_same:5d} disagree")
print(f"  cross-structure pairs: {agree_cross:5d} agree, {disagree_cross:5d} disagree")
tot = agree_cross + disagree_cross
if tot:
    print(f"\n  => {100*agree_cross/tot:.1f}% of cross-structure pairs pass the 0.4 e test,")
    print(f"     i.e. one molecule is certifying another as non-anomalous.")
