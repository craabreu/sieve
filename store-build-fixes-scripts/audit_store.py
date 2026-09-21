"""What is still wrong in the built store, corpus-wide."""
from __future__ import annotations
import collections, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
sys.path.insert(0, "experiments")
from experiments.data import blob_to_mol
RDLogger.DisableLog("rdApp.*")

path, stride = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(path, columns=["mol", "split", "dash_id", "collapse_key"])
df = df.iloc[::stride].reset_index(drop=True)
print(f"sampled {len(df):,} of every {stride} rows\n")

TET = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
tags = collections.Counter()
bond_stereo = collections.Counter()
exotic_rows, anyrows, conflict_rows, conflict_atoms = 0, 0, 0, 0
exotic_elems = collections.Counter()
exotic_ids, any_ids = set(), set()

for i in range(len(df)):
    m = blob_to_mol(df["mol"][i])
    xyz = np.asarray(m.GetConformer().GetPositions())
    has_exotic = has_any = has_conflict = False
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        tags[str(t)] += 1
        if t not in TET and str(t) != "CHI_UNSPECIFIED":
            has_exotic = True
            exotic_elems[f"{a.GetSymbol()} deg={a.GetDegree()} {t}"] += 1
        if t in TET and a.GetDegree() == 4:
            # Calibrated against RDKit at 390/390: the parity is the
            # determinant of the first three neighbour vectors taken from
            # the central atom, with neighbours in bond order.
            nb = [n.GetIdx() for n in a.GetNeighbors()]
            v = xyz[nb[:3]] - xyz[a.GetIdx()]
            vol = float(np.dot(np.cross(v[0], v[1]), v[2]))
            # RDKit's own convention: CCW is positive in bond order.
            want = 1 if t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else -1
            if abs(vol) > 0.5 and np.sign(vol) != want:
                has_conflict = True
                conflict_atoms += 1
    for b in m.GetBonds():
        s = str(b.GetStereo())
        bond_stereo[s] += 1
        if s == "STEREOANY":
            has_any = True
    exotic_rows += has_exotic; anyrows += has_any; conflict_rows += has_conflict
    if has_exotic:
        exotic_ids.add(str(df["dash_id"][i]))
    if has_any:
        any_ids.add(str(df["dash_id"][i]))

n = len(df)
print("--- chiral tags seen ---")
for k, v in tags.most_common():
    if k != "CHI_UNSPECIFIED":
        print(f"  {v:8d}  {k}")
print(f"\n--- exotic (non-tetrahedral) tags ---\n  rows carrying one: {exotic_rows} ({100*exotic_rows/n:.3f}%), "
      f"{len(exotic_ids)} distinct molecules")
for k, v in exotic_elems.most_common(10):
    print(f"    {v:6d}  {k}")
print(f"\n--- STEREOANY bonds remaining ---\n  rows carrying one: {anyrows} ({100*anyrows/n:.3f}%), "
      f"{len(any_ids)} distinct molecules")
for k, v in bond_stereo.most_common():
    if k != "STEREONONE":
        print(f"    {v:8d}  {k}")
print(f"\n--- declared parity vs coordinates (4-coordinate, |vol|>0.5) ---")
print(f"  rows with >=1 conflict: {conflict_rows} ({100*conflict_rows/n:.3f}%), atoms: {conflict_atoms}")
