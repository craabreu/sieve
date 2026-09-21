"""Rebuild stereochemistry.md's test-split numbers (S2, S2.0, subset share).

Runs against an arbitrary molecules.parquet so the pre- and post-stereofix
stores can be compared on one definition.  The labelling-artifact test of
S2.0 is done from *geometry* (signed volumes and torsions) rather than from
declared flags, so the fix -- which changes the flags -- cannot move the
criterion itself.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms

sys.path.insert(0, "experiments")

from experiments.collapse import (  # noqa: E402
    _group_indices,
    _strip_stereo,
    _within_group_sse,
    mirror_mol,
)
from experiments.data import MoleculeSet, blob_to_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ATOM_PROP = "MBIScharge"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def floors(mset: MoleculeSet, keyed_groups, blind_groups, orders) -> dict:
    n = float(mset.n_atoms)
    sse_k = _within_group_sse(mset, keyed_groups, lambda i: orders[i])
    sse_b = _within_group_sse(mset, blind_groups, lambda i: orders[i])
    return {
        "n_atoms": n,
        "keyed": float(np.sqrt(sse_k / n)),
        "blind": float(np.sqrt(sse_b / n)),
        "sse_keyed": sse_k,
        "sse_blind": sse_b,
    }


def potential_centres(stripped):
    """Positions of graph-determined potential tetrahedral stereocentres.

    Two traps here.  Restricting to potential centres matters at all because
    a symmetric centre such as a CH2 has a definite signed volume whose sign
    depends on which of two equivalent hydrogens the canonical ranking put
    first, so including it manufactures differences that are not
    stereochemical.  And the probe has to run on the heavy-atom graph:
    explicit hydrogens suppress RDKit's detection of *dependent* (para-)
    stereocentres -- the spiro, ring-fusion and acetal carbons whose two ring
    branches are constitutionally identical -- which is silent, returning an
    empty list rather than an error.
    """
    probe = Chem.Mol(stripped)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    try:
        probe = Chem.RemoveHs(probe)
    except Exception:
        probe = Chem.Mol(stripped)
    out = set()
    for el in Chem.FindPotentialStereo(probe):
        if el.type == Chem.StereoType.Atom_Tetrahedral:
            num = probe.GetAtomWithIdx(int(el.centeredOn)).GetAtomMapNum()
            if num:
                out.add(num - 1)
    return out


def geometry_signature(mol, order, centres=None):
    """(chirality signs, cis/trans classes) in the group's shared positions.

    ``order`` maps position -> atom index, so every member of a group is
    described in one frame and the vectors compare elementwise.
    """
    conf = mol.GetConformer()
    xyz = np.asarray(conf.GetPositions())
    pos_of = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    pos_of[order] = np.arange(len(order))

    chir = []
    for p, a_idx in enumerate(order):
        atom = mol.GetAtomWithIdx(int(a_idx))
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) not in (3, 4) or (centres is not None and int(a_idx) not in centres):
            continue
        nbrs.sort(key=lambda j: pos_of[j])
        # Degree 3 is a lone-pair stereocentre (a sulfoxide, a sulfilimine).
        # RDKit has no lone-pair pseudoatom; it defines the parity with the
        # central atom as the fourth vertex, which is where the lone pair
        # points, so the same signed volume works with c in place of n0.
        v = xyz[nbrs[1:]] - xyz[nbrs[0]] if len(nbrs) == 4 else xyz[nbrs] - xyz[int(a_idx)]
        vol = float(np.dot(np.cross(v[0], v[1]), v[2]))
        chir.append((p, int(np.sign(vol)) if abs(vol) > 1e-6 else 0))

    geo = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        na = sorted(
            (n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()),
            key=lambda j: pos_of[j],
        )
        nb = sorted(
            (n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != a.GetIdx()),
            key=lambda j: pos_of[j],
        )
        if not na or not nb:
            continue
        p1, p2, p3, p4 = xyz[na[0]], xyz[a.GetIdx()], xyz[b.GetIdx()], xyz[nb[0]]
        b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
            continue
        cos = float(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
        key = (int(pos_of[a.GetIdx()]), int(pos_of[b.GetIdx()]))
        geo.append((key, "cis" if cos > 0 else "trans"))
    return tuple(chir), tuple(sorted(geo))


def main(parquet: str, tag: str, limit: int | None = None) -> None:
    log(f"{tag}: reading {parquet}")
    df = pd.read_parquet(parquet, columns=["mol", "split", "collapse_key"])
    df = df[df["split"] == "test"].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    log(f"{tag}: {len(df):,} test conformers; deserialising")
    mols = [blob_to_mol(b) for b in df["mol"]]
    keys = [str(k) for k in df["collapse_key"]]
    mset = MoleculeSet(mols=mols, atom_property=ATOM_PROP, ids={"collapse_key": keys})
    log(f"{tag}: {mset.n_atoms:,} atoms; stripping stereo and ranking")

    stripped = [_strip_stereo(m) for m in mols]
    blind_keys = [Chem.MolToSmiles(s) for s in stripped]
    orders = {i: np.argsort(np.asarray(CanonicalRankAtoms(s))) for i, s in enumerate(stripped)}
    log(f"{tag}: grouping")

    blind_groups = _group_indices(blind_keys)
    keyed_groups = _group_indices(keys)

    whole = floors(mset, keyed_groups, blind_groups, orders)
    log(f"{tag}: whole split keyed={whole['keyed']:.6f} blind={whole['blind']:.6f}")

    # The stereo-sensitive subset: blind groups covering >1 collapse key.
    subset_groups = {
        g: rows for g, rows in blind_groups.items() if len({keys[i] for i in rows}) > 1
    }
    subset_rows = sorted(r for rows in subset_groups.values() for r in rows)
    mask = np.zeros(len(mols), dtype=bool)
    mask[subset_rows] = True
    sub = mset.select(mask)
    remap = {old: new for new, old in enumerate(subset_rows)}
    sub_orders = {remap[i]: orders[i] for i in subset_rows}
    sub_keyed = {
        k: [remap[i] for i in rows if i in remap] for k, rows in keyed_groups.items()
    }
    sub_keyed = {k: v for k, v in sub_keyed.items() if v}
    sub_blind = {g: [remap[i] for i in rows] for g, rows in subset_groups.items()}
    subset = floors(sub, sub_keyed, sub_blind, sub_orders)
    log(
        f"{tag}: subset {len(subset_groups):,} groups, {len(subset_rows):,} conformers "
        f"({100 * len(subset_rows) / len(mols):.2f}%) keyed={subset['keyed']:.6f} "
        f"blind={subset['blind']:.6f}"
    )

    # S2.0: groups whose members are geometrically indistinguishable.
    log(f"{tag}: geometry test over {len(subset_groups):,} groups")
    centres_of = {i: potential_centres(stripped[i]) for rows in subset_groups.values() for i in rows}
    artifact, artifact_rows = [], 0
    for g, rows in subset_groups.items():
        sigs = set()
        for i in rows:
            chir, geo = geometry_signature(mols[i], orders[i], centres_of[i])
            sigs.add((chir, geo))
        if len(sigs) == 1:
            artifact.append(g)
            artifact_rows += len(rows)
        elif len({s[1] for s in sigs}) == 1:
            # same double-bond geometry everywhere; check chirality up to a
            # global reflection, which collapse_key already merges.
            chirs = {s[0] for s in sigs}
            base = next(iter(chirs))
            flipped = tuple((p, -v) for p, v in base)
            if chirs <= {base, flipped}:
                artifact.append(g)
                artifact_rows += len(rows)

    log(f"{tag}: {len(artifact):,} artifact groups, {artifact_rows:,} conformers")

    # Corrected keyed floor: artifact groups merged into one key.
    merged_keyed = dict(sub_keyed)
    for g in artifact:
        rows = [remap[i] for i in subset_groups[g]]
        for k in list(merged_keyed):
            if set(merged_keyed[k]) & set(rows):
                del merged_keyed[k]
        merged_keyed[f"__merged__{g}"] = rows
    sse_corr = _within_group_sse(sub, merged_keyed, lambda i: sub_orders[i])
    keyed_corr = float(np.sqrt(sse_corr / sub.n_atoms))

    out = {
        "tag": tag,
        "parquet": parquet,
        "n_conformers": len(mols),
        "n_atoms": whole["n_atoms"],
        "whole_split": {"keyed": whole["keyed"], "blind": whole["blind"]},
        "blind_groups": len(blind_groups),
        "subset": {
            "groups": len(subset_groups),
            "conformers": len(subset_rows),
            "share": len(subset_rows) / len(mols),
            "keyed": subset["keyed"],
            "blind": subset["blind"],
            "gap": subset["blind"] - subset["keyed"],
            "keyed_artifacts_merged": keyed_corr,
            "gap_corrected": subset["blind"] - keyed_corr,
        },
        "artifacts": {
            "groups": len(artifact),
            "conformers": artifact_rows,
            "group_share": len(artifact) / max(len(subset_groups), 1),
            "conformer_share": artifact_rows / max(len(subset_rows), 1),
        },
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(f"/tmp/stereo-numbers-{tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
