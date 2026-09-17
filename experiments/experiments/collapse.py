"""Which molecules this series' model premise makes equivalent.

A molecule's conformers, an exactly duplicated structure, and an enantiomer
are indistinguishable to every arm here: they share one graph, and no
featurization in this series reads chirality. Diastereomers and E/Z isomers
are NOT equivalent -- they are measurably different molecules (spec section 1)
and merging them would define the target as a mean over different chemistry.

See docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md section 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import CanonicalRankAtoms

from experiments.data import MoleculeSet

_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def mirror_mol(mol: Any) -> Any:
    """A copy with every tetrahedral centre inverted, bond stereo untouched.

    Bond stereo is deliberately preserved: a mirror image has the opposite
    handedness at every stereocentre but the same E/Z geometry, which is what
    makes E/Z isomers survive the key while enantiomers do not.
    """
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == _CW:
            atom.SetChiralTag(_CCW)
        elif tag == _CCW:
            atom.SetChiralTag(_CW)
    return out


def collapse_key(mol: Any) -> str:
    """The equivalence class of ``mol``, as a canonical string.

    ``min`` over the molecule and its mirror picks the same representative of
    an enantiomer pair from either side, so the two share a key without any
    pairwise comparison.
    """
    return min(Chem.MolToSmiles(mol), Chem.MolToSmiles(mirror_mol(mol)))


def _canonical_order(mol: Any, key: str) -> np.ndarray:
    """Positions in the key's canonical order -> atom indices in ``mol``.

    A member whose own canonical SMILES is not the key is its mirror, so the
    ranking is taken on the mirrored copy; chiral-tag inversion does not move
    atom indices, so the resulting order applies to ``mol`` unchanged.
    """
    ranked = mol if Chem.MolToSmiles(mol) == key else mirror_mol(mol)
    return np.argsort(np.asarray(CanonicalRankAtoms(ranked)))


def collapse_molecule_set(
    mset: MoleculeSet, *, weight_by_collapse: bool = False
) -> MoleculeSet:
    """One conformer per collapse key, carrying that key's mean target.

    For FITTING only. Held-out sets are never collapsed: averaging a test
    target would encode the equivalence premise into the metric, so the metric
    could no longer detect the premise being false (spec section 2).

    ``weight_by_collapse`` repeats each representative ``n_collapsed`` times,
    reproducing the uncollapsed, atom-weighted fit. It exists for the
    migration check and is not the recommended setting.
    """
    keys = mset.ids.get("collapse_key")
    if keys is None:
        raise ValueError(
            "MoleculeSet has no 'collapse_key'; run `experiments "
            "annotate-collapse <store>` first"
        )
    dash = mset.ids.get("dash_id") or [""] * mset.n_conformers
    conf = mset.ids.get("conf_id") or [""] * mset.n_conformers

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(str(k), []).append(i)

    mols: list[Any] = []
    ids: dict[str, list[Any]] = {name: [] for name in mset.ids}
    values: list[float] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda i: (str(dash[i]), str(conf[i])))
        orders = [_canonical_order(mset.mols[i], key) for i in members]
        stacked = np.stack(
            [
                np.array(
                    [
                        mset.mols[i]
                        .GetAtomWithIdx(int(a))
                        .GetDoubleProp(mset.atom_property)
                        for a in order
                    ]
                )
                for i, order in zip(members, orders, strict=True)
            ]
        )
        mean = stacked.mean(axis=0)

        rep_i = members[0]
        rep = Chem.Mol(mset.mols[rep_i])
        for position, atom_idx in enumerate(orders[0]):
            rep.GetAtomWithIdx(int(atom_idx)).SetDoubleProp(
                mset.atom_property, float(mean[position])
            )
        repeats = len(members) if weight_by_collapse else 1
        for _ in range(repeats):
            mols.append(rep)
            for name in mset.ids:
                ids[name].append(mset.ids[name][rep_i])
            if mset.molecule_value is not None:
                values.append(float(mset.molecule_value[rep_i]))

    return MoleculeSet(
        mols=mols,
        atom_property=mset.atom_property,
        molecule_property=mset.molecule_property,
        molecule_value=(None if mset.molecule_value is None else np.array(values)),
        ids=ids,
        split=None,
    )
