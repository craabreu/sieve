"""Which molecules this series' model premise makes equivalent.

A molecule's conformers, an exactly duplicated structure, and an enantiomer
are indistinguishable to every arm here: they share one graph, and no
featurization in this series reads chirality. Diastereomers and E/Z isomers
are NOT equivalent -- they are measurably different molecules (spec section 1)
and merging them would define the target as a mean over different chemistry.

See docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md section 3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


def _strip_stereo(mol: Any) -> Any:
    """A copy with every stereo annotation removed, atom indices untouched.

    ``RemoveStereochemistry`` edits flags in place and neither adds, removes
    nor reorders atoms, so an atom order derived from the stripped copy
    applies unchanged to the original -- which is what lets the floor below
    read charges off the real molecule while matching atoms through the
    stripped one.
    """
    out = Chem.Mol(mol)
    Chem.RemoveStereochemistry(out)
    return out


def _within_group_sse(
    mset: MoleculeSet,
    groups: dict[str, list[int]],
    order_of: Callable[[int], np.ndarray],
) -> float:
    """Sum of squared deviations from each group's own per-position mean.

    ``order_of`` maps a row index to the atom order that puts its atoms in
    the group's shared canonical positions -- the whole content of "these
    molecules correspond atom-for-atom".
    """
    sse = 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        orders = [order_of(i) for i in members]
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
        sse += float(((stacked - stacked.mean(axis=0)) ** 2).sum())
    return sse


def _group_indices(keys: Sequence[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(str(k), []).append(i)
    return groups


def held_out_floor(mset: MoleculeSet) -> float:
    """RMS of the within-key deviations across a held-out set.

    Rows sharing a collapse key are identical to every arm in this series, so
    any spread among their targets is error no graph-based model can avoid.
    Returns 0.0 when no key repeats, and 0.0 when the set carries no
    ``collapse_key`` at all.

    This is the *stereo-aware* floor: the key separates diastereomers and E/Z
    isomers, so their scatter falls between groups and is not counted here.
    See ``held_out_floor_stereo_blind`` for the complement.
    """
    keys = mset.ids.get("collapse_key")
    if keys is None or mset.n_atoms == 0:
        return 0.0
    groups = _group_indices(keys)
    sse = _within_group_sse(
        mset, groups, lambda i: _canonical_order(mset.mols[i], str(keys[i]))
    )
    return float(np.sqrt(sse / mset.n_atoms))


def held_out_floor_stereo_blind(mset: MoleculeSet) -> float:
    """The same RMS, grouped on the stereo-stripped graph instead.

    No arm in this series reads stereochemistry -- Sieve carries ``element``
    with no edge attributes, DASH's tuple is ``(element, degree,
    formal_charge, aromatic, num_h)``, and HOSE codes carry bond order but
    not bond stereo. Diastereomers and E/Z isomers therefore land in one
    class and receive one prediction, so their target scatter is irreducible
    for these featurizations exactly as conformer scatter is -- but
    ``held_out_floor`` cannot see it, because ``collapse_key`` deliberately
    holds them apart.

    Grouping on the stereo-stripped canonical SMILES is what the arms
    actually collide, so this is the honest floor for them. Its groups are
    supersets of ``held_out_floor``'s, so it can only be larger.

    Needs no ``collapse_key``: it derives its own grouping from the
    molecules, and therefore works on any held-out set.
    """
    if mset.n_atoms == 0:
        return 0.0
    stripped = [_strip_stereo(m) for m in mset.mols]
    groups = _group_indices([Chem.MolToSmiles(s) for s in stripped])
    # A stereo-stripped molecule is its own mirror, so there is no "key or
    # its mirror?" question here -- the rank is taken directly.
    orders = {
        i: np.argsort(np.asarray(CanonicalRankAtoms(stripped[i])))
        for members in groups.values()
        if len(members) > 1
        for i in members
    }
    sse = _within_group_sse(mset, groups, lambda i: orders[i])
    return float(np.sqrt(sse / mset.n_atoms))


def held_out_floors(mset: MoleculeSet) -> dict[str, float]:
    """Both floors, computed once.

    Callers should compute this once per held-out set and reuse it, rather
    than once per run: a fold's held-out set is shared by every depth and
    every variant scored against it, and the calculation costs ~21 s on a
    real fold -- which across Study B's 175 runs over 25 distinct held-out
    sets is the difference between ~2 h and ~17 min.
    """
    return {
        "floor/rmse": held_out_floor(mset),
        "floor/rmse_stereo_blind": held_out_floor_stereo_blind(mset),
    }
