from __future__ import annotations
import pathlib
import sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms
sys.path.insert(0, "experiments")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiments.collapse import _group_indices, _strip_stereo
from experiments.data import blob_to_mol
from rebuild_stereo_numbers import geometry_signature, potential_centres
RDLogger.DisableLog("rdApp.*")
df = pd.read_parquet(sys.argv[1], columns=["mol", "split", "collapse_key", "dash_id"])
df = df[df["split"] == "test"].reset_index(drop=True)
mols = [blob_to_mol(b) for b in df["mol"]]
keys = [str(k) for k in df["collapse_key"]]
stripped = [_strip_stereo(m) for m in mols]
orders = {i: np.argsort(np.asarray(CanonicalRankAtoms(s))) for i, s in enumerate(stripped)}
blind = _group_indices([Chem.MolToSmiles(s) for s in stripped])
subset = {g: r for g, r in blind.items() if len({keys[i] for i in r}) > 1}
for g, rows in subset.items():
    sigs = {geometry_signature(mols[i], orders[i], potential_centres(stripped[i])) for i in rows}
    ok = len(sigs) == 1
    if not ok and len({s[1] for s in sigs}) == 1:
        c = {s[0] for s in sigs}; b = next(iter(c))
        ok = c <= {b, tuple((p, -v) for p, v in b)}
    if not ok or "P" not in g:
        continue
    reps = {}
    for i in rows:
        reps.setdefault(keys[i], i)
    print("=" * 90)
    print(f"dash_id={df['dash_id'][list(reps.values())[0]]} keys={len(reps)} conformers={len(rows)}")
    for k, i in reps.items():
        print(f"  key: {k[:110]}")
        print(f"     potential centres: {sorted(potential_centres(stripped[i]))}")
        print(f"     chirality vector : {geometry_signature(mols[i], orders[i], potential_centres(stripped[i]))[0]}")
        tagged = [(a.GetIdx(), a.GetSymbol(), str(a.GetChiralTag()).replace('CHI_TETRAHEDRAL_',''))
                  for a in mols[i].GetAtoms() if str(a.GetChiralTag()) != "CHI_UNSPECIFIED"]
        print(f"     tagged atoms     : {tagged}")
