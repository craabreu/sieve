"""Dump the groups my test still cannot adjudicate."""
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
    if not ok:
        continue
    reps = {}
    for i in rows:
        reps.setdefault(keys[i], i)
    reps = list(reps.values())
    def tags(i):
        m, o = mols[i], orders[i]
        p = np.empty(m.GetNumAtoms(), dtype=np.int64); p[o] = np.arange(len(o))
        return {int(p[a.GetIdx()]): str(a.GetChiralTag()) for a in m.GetAtoms()
                if str(a.GetChiralTag()) != "CHI_UNSPECIFIED"}
    t0 = tags(reps[0]); diff = set()
    for i in reps[1:]:
        t1 = tags(i)
        diff |= {p for p in set(t0) | set(t1) if t0.get(p) != t1.get(p)}
    m0, o0 = mols[reps[0]], orders[reps[0]]
    covered = potential_centres(stripped[reps[0]])
    unflagged = [p for p in diff if int(o0[p]) not in covered]
    if not unflagged:
        continue
    print("=" * 100)
    print(f"dash_id={df['dash_id'][reps[0]]}  keys={len(reps)}  conformers={len(rows)}")
    for p in sorted(diff):
        idx = int(o0[p]); at = m0.GetAtomWithIdx(idx)
        nbr = [n.GetSymbol() for n in at.GetNeighbors()]
        ring = [r for r in m0.GetRingInfo().AtomRings() if idx in r]
        print(f"  pos {p}: {at.GetSymbol()} deg={at.GetDegree()} nbrs={nbr} "
              f"flagged_potential={idx in covered} in_ring={[len(r) for r in ring]} "
              f"tags={[tags(i).get(p, '-') for i in reps]}")
        # is the declared parity consistent with the coordinates?
        for i in reps:
            m, o = mols[i], orders[i]
            pp = np.empty(m.GetNumAtoms(), dtype=np.int64); pp[o] = np.arange(len(o))
            a = int(o[p]); xyz = np.asarray(m.GetConformer().GetPositions())
            nb = sorted((n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors()), key=lambda j: pp[j])
            v = xyz[nb[1:]] - xyz[nb[0]]
            print(f"      key rep {i}: signed volume {float(np.dot(np.cross(v[0], v[1]), v[2])):+8.3f}")
    for i in reps[:3]:
        print(f"  SMILES: {Chem.MolToSmiles(mols[i])[:150]}")
