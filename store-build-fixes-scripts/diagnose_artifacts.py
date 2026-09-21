"""What separates the collapse keys of groups that are geometrically identical?"""
from __future__ import annotations
import pathlib
import collections, json, sys
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms
sys.path.insert(0, "experiments")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiments.collapse import _group_indices, _strip_stereo
from experiments.data import blob_to_mol
from rebuild_stereo_numbers import geometry_signature, potential_centres
RDLogger.DisableLog("rdApp.*")

df = pd.read_parquet(sys.argv[1], columns=["mol", "split", "collapse_key", "dash_id", "chembl_id"])
df = df[df["split"] == "test"].reset_index(drop=True)
mols = [blob_to_mol(b) for b in df["mol"]]
keys = [str(k) for k in df["collapse_key"]]
stripped = [_strip_stereo(m) for m in mols]
orders = {i: np.argsort(np.asarray(CanonicalRankAtoms(s))) for i, s in enumerate(stripped)}
blind = _group_indices([Chem.MolToSmiles(s) for s in stripped])
subset = {g: r for g, r in blind.items() if len({keys[i] for i in r}) > 1}

artifacts = []
for g, rows in subset.items():
    sigs = {geometry_signature(mols[i], orders[i], potential_centres(stripped[i])) for i in rows}
    if len(sigs) == 1:
        artifacts.append((g, rows)); continue
    if len({s[1] for s in sigs}) == 1:
        chirs = {s[0] for s in sigs}; base = next(iter(chirs))
        if chirs <= {base, tuple((p, -v) for p, v in base)}:
            artifacts.append((g, rows))

print(f"artifact groups: {len(artifacts)}  conformers: {sum(len(r) for _, r in artifacts)}\n")

def annot(mol, order):
    pos = np.empty(mol.GetNumAtoms(), dtype=np.int64); pos[order] = np.arange(len(order))
    atoms = {int(pos[a.GetIdx()]): str(a.GetChiralTag()) for a in mol.GetAtoms()
             if str(a.GetChiralTag()) != "CHI_UNSPECIFIED"}
    bonds = {}
    for b in mol.GetBonds():
        st = str(b.GetStereo())
        if st != "STEREONONE":
            k = tuple(sorted((int(pos[b.GetBeginAtomIdx()]), int(pos[b.GetEndAtomIdx()]))))
            bonds[k] = st
    return atoms, bonds

kind = collections.Counter(); bond_pairs = collections.Counter()
flavours = collections.Counter(); atom_env = collections.Counter()
examples = []
for g, rows in artifacts:
    by_key = {}
    for i in rows:
        by_key.setdefault(keys[i], i)
    reps = list(by_key.values())
    a0, b0 = annot(mols[reps[0]], orders[reps[0]])
    atom_diff, bond_diff = set(), set()
    for i in reps[1:]:
        a1, b1 = annot(mols[i], orders[i])
        atom_diff |= {p for p in set(a0) | set(a1) if a0.get(p) != a1.get(p)}
        bond_diff |= {p for p in set(b0) | set(b1) if b0.get(p) != b1.get(p)}
    k = ("both" if atom_diff and bond_diff else "bond stereo only" if bond_diff
         else "atom parity only" if atom_diff else "no declared difference")
    kind[k] += 1
    m, o = mols[reps[0]], orders[reps[0]]
    for p in bond_diff:
        i0, i1 = int(o[p[0]]), int(o[p[1]])
        e = tuple(sorted((m.GetAtomWithIdx(i0).GetSymbol(), m.GetAtomWithIdx(i1).GetSymbol())))
        bt = str(m.GetBondBetweenAtoms(i0, i1).GetBondType())
        bond_pairs[f"{e[0]}{'=' if bt=='DOUBLE' else '~'}{e[1]} ({bt.lower()})"] += 1
        vals = {annot(mols[i], orders[i])[1].get(p, "STEREONONE") for i in reps}
        flavours[" vs ".join(sorted(vals))] += 1
    for p in atom_diff:
        at = m.GetAtomWithIdx(int(o[p]))
        nN = sum(1 for n in at.GetNeighbors() if n.GetAtomicNum() > 1)
        atom_env[f"{at.GetSymbol()}, {at.GetDegree()} nbrs ({nN} heavy)"] += 1
    if len(examples) < 8:
        examples.append({"n_keys": len(by_key), "n_conf": len(rows), "kind": k,
                         "dash_id": str(df['dash_id'][reps[0]]),
                         "keys": [Chem.MolToSmiles(mols[i]) for i in reps][:3]})

for name, c in [("what differs in the declarations", kind), ("differing bonds", bond_pairs),
                ("declared values in conflict", flavours), ("differing atom centres", atom_env)]:
    print(f"--- {name} ---")
    for k, v in c.most_common(12):
        print(f"  {v:5d}  {k}")
    print()
print("--- examples ---")
for e in examples:
    print(json.dumps(e))
