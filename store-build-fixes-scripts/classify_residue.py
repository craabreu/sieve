"""Split the residual groups into ones my test was blind to and real conflicts."""
from __future__ import annotations
import pathlib
import collections, sys
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

verdict = collections.Counter(); detail = collections.Counter(); rows_by = collections.Counter()
for g, rows in subset.items():
    sigs = {geometry_signature(mols[i], orders[i], potential_centres(stripped[i])) for i in rows}
    ok = len(sigs) == 1
    if not ok and len({s[1] for s in sigs}) == 1:
        chirs = {s[0] for s in sigs}; base = next(iter(chirs))
        ok = chirs <= {base, tuple((p, -v) for p, v in base)}
    if not ok:
        continue
    reps = {}
    for i in rows:
        reps.setdefault(keys[i], i)
    reps = list(reps.values())
    m0, o0 = mols[reps[0]], orders[reps[0]]
    pos0 = np.empty(m0.GetNumAtoms(), dtype=np.int64); pos0[o0] = np.arange(len(o0))
    covered = potential_centres(stripped[reps[0]])
    def tags(i):
        m, o = mols[i], orders[i]
        p = np.empty(m.GetNumAtoms(), dtype=np.int64); p[o] = np.arange(len(o))
        return {int(p[a.GetIdx()]): str(a.GetChiralTag()) for a in m.GetAtoms()
                if str(a.GetChiralTag()) != "CHI_UNSPECIFIED"}
    t0 = tags(reps[0]); diff = set()
    for i in reps[1:]:
        t1 = tags(i)
        diff |= {p for p in set(t0) | set(t1) if t0.get(p) != t1.get(p)}
    kinds = set()
    for p in diff:
        at = m0.GetAtomWithIdx(int(o0[p]))
        in_test = at.GetDegree() == 4 and int(o0[p]) in covered
        if in_test:
            # Does each member's own declared tag agree with its own
            # coordinates?  Convention calibrated at 390/390: determinant of
            # the first three neighbour vectors from the centre, bond order.
            import numpy as _np
            agree = []
            for i in reps:
                m, o = mols[i], orders[i]
                pp = _np.empty(m.GetNumAtoms(), dtype=_np.int64); pp[o] = _np.arange(len(o))
                ai = int(o[p]); a = m.GetAtomWithIdx(ai)
                t = str(a.GetChiralTag())
                if t not in ("CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"):
                    agree.append("untagged"); continue
                xyz = _np.asarray(m.GetConformer().GetPositions())
                nb = [n.GetIdx() for n in a.GetNeighbors()][:3]
                v = xyz[nb] - xyz[ai]
                vol = float(_np.dot(_np.cross(v[0], v[1]), v[2]))
                want = 1 if t == "CHI_TETRAHEDRAL_CCW" else -1
                agree.append("ok" if _np.sign(vol) == want else f"MISMATCH({vol:+.2f},{t})")
            print(f"  group {g[:40]}... pos {p}: {agree}")
        tagset = {tags(i).get(p, "-") for i in reps}
        exotic = any("TB" in t or "SP" in t or "OH" in t for t in tagset)
        if exotic:
            kinds.add("exotic tag (TB/SP/OH) -- perception garbage")
        elif not in_test:
            kinds.add(f"{at.GetSymbol()}, degree {at.GetDegree()} -- outside my 4-coordinate test")
        else:
            kinds.add(f"{at.GetSymbol()}, degree 4 -- declared parity contradicts coordinates")
    v = ("real conflict" if any("contradicts" in k for k in kinds)
         else "perception garbage" if any("garbage" in k for k in kinds)
         else "blind spot in my test")
    verdict[v] += 1; rows_by[v] += len(rows)
    for k in kinds:
        detail[k] += 1

print(f"{'verdict':28s} {'groups':>7s} {'conformers':>11s}")
for k, v in verdict.most_common():
    print(f"{k:28s} {v:7d} {rows_by[k]:11d}")
print()
for k, v in detail.most_common():
    print(f"  {v:4d}  {k}")
