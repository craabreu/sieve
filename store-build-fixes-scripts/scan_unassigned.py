"""Records holding a potential stereocentre the store left unassigned."""
from __future__ import annotations
import collections, sys
import pandas as pd
from rdkit import Chem, RDLogger
sys.path.insert(0, "experiments")
from experiments.data import blob_to_mol
RDLogger.DisableLog("rdApp.*")
TET = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
df = pd.read_parquet(sys.argv[1], columns=["mol", "split", "dash_id"])
stride = int(sys.argv[2]); df = df.iloc[::stride].reset_index(drop=True)
rows = unassigned_rows = 0
centres = unassigned_centres = 0
env = collections.Counter(); ids = set()
for i in range(len(df)):
    m = blob_to_mol(df["mol"][i]); rows += 1
    probe = Chem.Mol(m)
    for a in probe.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    try:
        probe = Chem.RemoveHs(probe)
    except Exception:
        continue
    hit = False
    for el in Chem.FindPotentialStereo(probe):
        if el.type != Chem.StereoType.Atom_Tetrahedral:
            continue
        num = probe.GetAtomWithIdx(int(el.centeredOn)).GetAtomMapNum()
        if not num:
            continue
        centres += 1
        a = m.GetAtomWithIdx(num - 1)
        if a.GetChiralTag() not in TET:
            unassigned_centres += 1; hit = True
            ring = m.GetRingInfo().NumAtomRings(num - 1)
            env[f"{a.GetSymbol()} deg={a.GetDegree()} rings={ring}"] += 1
    if hit:
        unassigned_rows += 1; ids.add(str(df["dash_id"][i]))
print(f"scanned {rows:,} rows (every {stride})")
print(f"potential tetrahedral centres (heavy-atom probe): {centres:,}")
print(f"  of which the store leaves unassigned          : {unassigned_centres:,} "
      f"({100*unassigned_centres/max(centres,1):.2f}%)")
print(f"rows holding >=1 such centre: {unassigned_rows:,} ({100*unassigned_rows/rows:.2f}%), "
      f"{len(ids):,} distinct molecules")
for k, v in env.most_common(8):
    print(f"   {v:6d}  {k}")
