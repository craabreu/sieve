"""How many records carry a non-tetrahedral chiral tag, and do they split keys?"""
from __future__ import annotations
import collections, sys
import pandas as pd
from rdkit import Chem, RDLogger
sys.path.insert(0, "experiments")
from experiments.data import blob_to_mol
RDLogger.DisableLog("rdApp.*")
OK = {Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
      Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
df = pd.read_parquet(sys.argv[1], columns=["mol", "split", "dash_id", "collapse_key"])
stride = int(sys.argv[2]); df = df.iloc[::stride].reset_index(drop=True)
kinds = collections.Counter(); keys_by_id = collections.defaultdict(set)
rows = 0; ids = set()
for i in range(len(df)):
    m = blob_to_mol(df["mol"][i])
    ex = [(a.GetSymbol(), a.GetDegree(), str(a.GetChiralTag())) for a in m.GetAtoms()
          if a.GetChiralTag() not in OK]
    if not ex:
        continue
    rows += 1; ids.add(str(df["dash_id"][i]))
    keys_by_id[str(df["dash_id"][i])].add(str(df["collapse_key"][i]))
    for e in ex:
        kinds[f"{e[0]} deg={e[1]} {e[2]}"] += 1
print(f"scanned {len(df):,} rows (every {stride}) of the whole store")
print(f"rows with a non-tetrahedral tag: {rows} ({100*rows/len(df):.4f}%), "
      f"{len(ids)} distinct molecules")
for k, v in kinds.most_common():
    print(f"   {v:5d}  {k}")
split = {d: ks for d, ks in keys_by_id.items() if len(ks) > 1}
print(f"molecules whose sampled conformers already take >1 collapse_key: {len(split)} of {len(ids)}")
