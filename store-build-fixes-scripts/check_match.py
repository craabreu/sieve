"""Does the stripped canonical ranking actually match atoms across members?"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms
sys.path.insert(0, "experiments")
from experiments.collapse import _group_indices, _strip_stereo
from experiments.data import blob_to_mol
RDLogger.DisableLog("rdApp.*")

df = pd.read_parquet(sys.argv[1], columns=["mol", "split", "collapse_key"])
df = df[df["split"] == "test"].reset_index(drop=True)
mols = [blob_to_mol(b) for b in df["mol"]]
keys = [str(k) for k in df["collapse_key"]]
stripped = [_strip_stereo(m) for m in mols]
orders = {i: np.argsort(np.asarray(CanonicalRankAtoms(s))) for i, s in enumerate(stripped)}
blind = _group_indices([Chem.MolToSmiles(s) for s in stripped])
subset = {g: r for g, r in blind.items() if len({keys[i] for i in r}) > 1}

def frame(mol, order):
    pos = np.empty(mol.GetNumAtoms(), dtype=np.int64); pos[order] = np.arange(len(order))
    elems = tuple(mol.GetAtomWithIdx(int(a)).GetSymbol() for a in order)
    bonds = tuple(sorted((min(int(pos[b.GetBeginAtomIdx()]), int(pos[b.GetEndAtomIdx()])),
                          max(int(pos[b.GetBeginAtomIdx()]), int(pos[b.GetEndAtomIdx()])),
                          str(b.GetBondType())) for b in mol.GetBonds()))
    return elems, bonds

bad = 0
for g, rows in subset.items():
    frames = {frame(mols[i], orders[i]) for i in rows}
    if len(frames) > 1:
        bad += 1
print(f"subset groups: {len(subset)}   groups whose members do NOT share one matched frame: {bad}"
      f"  ({100*bad/len(subset):.1f}%)")
