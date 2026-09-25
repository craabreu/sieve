"""Which molecules this series' model premise makes equivalent.

A molecule's conformers, an exactly duplicated structure, and an enantiomer
are indistinguishable to every arm here: they share one graph, and no
featurization in this series reads chirality. Diastereomers and E/Z isomers
are NOT equivalent -- they are measurably different molecules (spec section 1)
and merging them would define the target as a mean over different chemistry.

See docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md section 3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import CanonicalRankAtoms

from experiments.data import MoleculeSet
from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

logger = logging.getLogger(__name__)

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

    A tie-broken ranking: among symmetric atoms the order follows the input
    order, so it is not an alignment (see ``aligned_values``). Kept because
    the store-build-fixes scripts reproduce numbers computed with it.

    A member whose own canonical SMILES is not the key is its mirror, so the
    ranking is taken on the mirrored copy; chiral-tag inversion does not move
    atom indices, so the resulting order applies to ``mol`` unchanged.
    """
    ranked = mol if Chem.MolToSmiles(mol) == key else mirror_mol(mol)
    return np.argsort(np.asarray(CanonicalRankAtoms(ranked)))


# --------------------------------------------------------------------------
# Symmetry and alignment
#
# Which atoms of a group correspond, and which atoms are the same atom to every
# arm, are both questions about automorphisms, and are answered with them
# rather than with canonical ranks. ``CanonicalRankAtoms(breakTies=False)``
# gives refinement classes, not orbits: measured against exact orbits over all
# 1,027,555 store rows, with ``includeChirality=True`` it splits equivalent
# atoms on 17,986 rows -- the C2-related halves of (R,R)-2,3-butanediol -- and
# merges distinct ones on 57; without chirality it merges a molecule's
# aziridine and piperazine nitrogens, which no refinement can tell apart. With
# ``breakTies=True`` it is an ordering, and among tied atoms the order follows
# the input atom order.
# --------------------------------------------------------------------------

# Compared through SubstructMatchParameters.atomProperties. A Mol used as a
# query matches elements and bond orders but not hydrogen counts, and a
# neutral query atom matches any charge, so on the heavy-atom graph a
# benzimidazole's [nH] would match its n.
_ATOM_INVARIANTS = ("num_hs", "formal_charge", "isotope", "radicals")

# Heavy-atom automorphisms per molecule before the orbits are abandoned. None
# of 20,552 sampled store rows needed more than a small fraction of this;
# enumerating hydrogens instead, 4,267 of them exceeded it.
AUTOMORPHISM_CAP = 10_000


def _heavy_graph(mol: Any, *, stereo: bool) -> tuple[Any, list[int]]:
    """``mol`` without its removable hydrogens, carrying the invariants the
    match must respect, and the original index of each atom it kept.

    Which hydrogens survive ``RemoveHs`` is read back rather than assumed: it
    keeps those it must, such as a stereo reference. Without ``stereo`` the
    copy is stripped, so no stereo annotation can reach the match.
    """
    tagged = Chem.Mol(mol)
    if not stereo:
        Chem.RemoveStereochemistry(tagged)
    for atom in tagged.GetAtoms():
        atom.SetIntProp("orig_idx", atom.GetIdx())
    heavy = Chem.RemoveHs(tagged, sanitize=False)
    for atom in heavy.GetAtoms():
        atom.SetIntProp("num_hs", atom.GetTotalNumHs())
        atom.SetIntProp("formal_charge", atom.GetFormalCharge())
        atom.SetIntProp("isotope", atom.GetIsotope())
        atom.SetIntProp("radicals", atom.GetNumRadicalElectrons())
    return heavy, [a.GetIntProp("orig_idx") for a in heavy.GetAtoms()]


def _match_params(
    *, stereo: bool, all_matches: bool = False
) -> Chem.SubstructMatchParameters:
    params = Chem.SubstructMatchParameters()
    params.useChirality = stereo
    params.atomProperties = list(_ATOM_INVARIANTS)
    if all_matches:
        params.uniquify = False
        params.maxMatches = AUTOMORPHISM_CAP
    return params


def _images(heavy: Any, *, stereo: bool) -> tuple[Any, ...]:
    """Where a stereo-preserving map may send the molecule: onto itself, and,
    with stereo, onto its mirror too -- MBIS charges are invariant under
    reflection, and a collapse group pools enantiomers."""
    return (heavy, mirror_mol(heavy)) if stereo else (heavy,)


def symmetry_orbits(mol: Any, *, stereo: bool) -> np.ndarray:
    """One label per atom; atoms share a label exactly when some automorphism
    maps one onto the other -- preserving stereo, directly or through the
    mirror, when ``stereo`` is set.

    Found by exhaustive self-matching of the heavy-atom graph. Hydrogens then
    join their parent's orbit: the ones on one atom are always
    interchangeable, and enumerating them is what makes exhaustive matching
    explode. Past ``AUTOMORPHISM_CAP`` each atom is left its own class, which
    can only withhold averaging, never merge atoms that differ.
    """
    heavy, kept = _heavy_graph(mol, stereo=stereo)
    n = heavy.GetNumAtoms()
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    params = _match_params(stereo=stereo, all_matches=True)
    for image in _images(heavy, stereo=stereo):
        matches = image.GetSubstructMatches(heavy, params)
        if len(matches) >= AUTOMORPHISM_CAP:
            logger.warning(
                "%s has more than %d automorphisms; its atoms are left "
                "unmerged, so their symmetry is not averaged",
                Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False)),
                AUTOMORPHISM_CAP,
            )
            return np.arange(mol.GetNumAtoms())
        for match in matches:
            for i, j in enumerate(match):
                a, b = find(i), find(int(j))
                if a != b:
                    parent[a] = b

    labels = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    for k, idx in enumerate(kept):
        labels[idx] = find(k)
    kept_set = set(kept)
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in kept_set:
            (host,) = atom.GetNeighbors()
            labels[atom.GetIdx()] = n + labels[host.GetIdx()]
    return labels


def _graph_signature(mol: Any, *, stereo: bool) -> bytes:
    """The molecule's labelled graph as bytes, in its own atom order: RDKit's
    pickle of a copy without conformers or properties, which keeps atoms,
    bonds, charges, hydrogen counts and stereo. Two rows with equal
    signatures are related by the identity."""
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    if not stereo:
        Chem.RemoveStereochemistry(graph)
    return graph.ToBinary(Chem.PropertyPickleOptions.NoProps)


def _isomorphism(
    reference: Any, member: Any, *, stereo: bool, reference_signature: bytes
) -> np.ndarray:
    """``out[i]`` is the atom of ``member`` that corresponds to atom ``i`` of
    ``reference``: an isomorphism that preserves stereo, directly or through
    the mirror, when ``stereo`` is set.

    Any one will do. Values are then averaged over orbits, and two
    isomorphisms differ by an automorphism, which maps every orbit onto
    itself. A removed hydrogen is paired with one on the corresponding host.
    """
    # Most groups are one deposit's conformers, which share their atom order;
    # for those the identity is an isomorphism and no search is needed.
    if _graph_signature(member, stereo=stereo) == reference_signature:
        return np.arange(reference.GetNumAtoms())

    ref_heavy, ref_kept = _heavy_graph(reference, stereo=stereo)
    mem_heavy, mem_kept = _heavy_graph(member, stereo=stereo)
    params = _match_params(stereo=stereo)
    match: tuple[int, ...] = ()
    for image in _images(mem_heavy, stereo=stereo):
        match = image.GetSubstructMatch(ref_heavy, params)
        if match:
            break
    if not match or reference.GetNumAtoms() != member.GetNumAtoms():
        raise ValueError(
            "rows grouped as one molecule are not isomorphic: "
            f"{Chem.MolToSmiles(reference)} vs {Chem.MolToSmiles(member)}"
        )

    out = np.full(reference.GetNumAtoms(), -1, dtype=np.int64)
    for i, j in enumerate(match):
        out[ref_kept[i]] = mem_kept[int(j)]
    ref_kept_set, mem_kept_set = set(ref_kept), set(mem_kept)
    for atom in reference.GetAtoms():
        if atom.GetIdx() in ref_kept_set:
            continue
        (host,) = atom.GetNeighbors()
        partners = [
            nb.GetIdx()
            for nb in member.GetAtomWithIdx(int(out[host.GetIdx()])).GetNeighbors()
            if nb.GetIdx() not in mem_kept_set
        ]
        taken = set(out[out >= 0].tolist())
        free = [h for h in partners if h not in taken]
        if not free:
            raise ValueError("hydrogens do not correspond between group members")
        out[atom.GetIdx()] = free[0]
    return out


def aligned_values(
    mols: Sequence[Any], atom_property: str, *, stereo: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``(values, orbit)`` for one group of rows of the same molecule.

    ``values[m, i]`` is member ``m``'s ``atom_property`` on the atom that
    corresponds to atom ``i`` of the first member, the reference, and
    ``orbit[i]`` numbers that atom's symmetry orbit ``0..k-1``. Anything
    computed per orbit from these -- a mean, a scatter, a sorted list -- is
    independent of atom order and of which isomorphism aligned the members.
    """
    reference = mols[0]
    signature = _graph_signature(reference, stereo=stereo)
    maps = [np.arange(reference.GetNumAtoms())] + [
        _isomorphism(reference, m, stereo=stereo, reference_signature=signature)
        for m in mols[1:]
    ]
    values = np.stack(
        [
            np.array(
                [m.GetAtomWithIdx(int(a)).GetDoubleProp(atom_property) for a in atoms]
            )
            for m, atoms in zip(mols, maps, strict=True)
        ]
    )
    _, orbit = np.unique(symmetry_orbits(reference, stereo=stereo), return_inverse=True)
    return values, orbit


def orbit_means(values: np.ndarray, orbit: np.ndarray) -> np.ndarray:
    """Each atom's value pooled over its orbit and over every member."""
    sums = np.bincount(orbit, weights=values.sum(axis=0))
    counts = np.bincount(orbit) * values.shape[0]
    return (sums / counts)[orbit]


def collapse_molecule_set(
    mset: MoleculeSet, *, weight_by_collapse: bool = False
) -> MoleculeSet:
    """One conformer per collapse key, carrying that key's mean target: each
    atom's value averaged over the key's rows and over the atom's symmetry
    orbit (``aligned_values``, ``orbit_means``).

    For FITTING only. Held-out sets are never collapsed: averaging a test
    target would encode the equivalence premise into the metric, so the metric
    could no longer detect the premise being false (spec section 2).

    ``weight_by_collapse`` repeats each representative ``n_collapsed`` times,
    reproducing the uncollapsed, atom-weighted fit. It exists for the
    migration check and is not the recommended setting.

    Each atom also carries ``<atom_property>__within_sse`` and ``__within_n``,
    the scatter the averaging removed (within-structure-variance spec 3.1).
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
        aligned, orbit = aligned_values(
            [mset.mols[i] for i in members], mset.atom_property, stereo=True
        )
        # The class mean, pooled over the members and over each symmetry
        # orbit: atoms of one orbit are the same atom to every arm, so they
        # get one target, and no pairing of them can change it.
        target = orbit_means(aligned, orbit)
        # What the averaging removes, kept beside the target so the fit can
        # add it back to the predictive variance (within-structure-variance
        # spec 3.1): each atom's squared deviations from that mean, summed over
        # the members. Summed over atoms and keys this is floor_components'
        # `sse`, and the member counts sum to its `n_atoms`.
        sse = ((aligned - target) ** 2).sum(axis=0)
        # A weighted collapse repeats the representative, so each copy carries
        # its share and the sums stay the same.
        repeats = len(members) if weight_by_collapse else 1
        n_share = len(members) / repeats

        rep_i = members[0]
        rep = Chem.Mol(mset.mols[rep_i])
        for atom_idx, value in enumerate(target):
            atom = rep.GetAtomWithIdx(atom_idx)
            atom.SetDoubleProp(mset.atom_property, float(value))
            atom.SetDoubleProp(
                mset.atom_property + WITHIN_SSE_SUFFIX, float(sse[atom_idx]) / repeats
            )
            atom.SetDoubleProp(mset.atom_property + WITHIN_N_SUFFIX, n_share)
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

    The floor definition before symmetry orbits (``_orbit_sse``), kept because
    the store-build-fixes scripts reproduce numbers computed with it.

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


def _orbit_sse(
    mols: Sequence[Any],
    atom_property: str,
    groups: dict[str, list[int]],
    *,
    stereo: bool,
) -> float:
    """Sum of squared deviations from each atom's orbit mean, pooled over a
    group's rows (``orbit_means``).

    Atoms of one orbit get one prediction from every arm, so their scatter is
    as irreducible as the scatter across a group's conformers -- and it is
    there in a group of one, which therefore counts.
    """
    sse = 0.0
    for members in groups.values():
        values, orbit = aligned_values(
            [mols[i] for i in members], atom_property, stereo=stereo
        )
        sse += float(((values - orbit_means(values, orbit)) ** 2).sum())
    return sse


def _group_indices(keys: Sequence[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(str(k), []).append(i)
    return groups


def _stereo_blind_groups(mset: MoleculeSet) -> dict[str, list[int]]:
    """Rows grouped on the stereo-stripped canonical SMILES -- what the
    stereo-blind arms actually collide."""
    return _group_indices([Chem.MolToSmiles(_strip_stereo(m)) for m in mset.mols])


def held_out_floor(mset: MoleculeSet) -> float:
    """RMS of the within-key deviations across a held-out set.

    Rows sharing a collapse key are identical to every arm in this series, and
    so are the atoms of one symmetry orbit, so any spread among their targets
    is error no graph-based model can avoid. Deviations are taken from each
    atom's orbit mean over the group (``_orbit_sse``), so a group of one still
    counts. Returns 0.0 when the set carries no ``collapse_key`` at all.

    This is the *stereo-aware* floor: the key separates diastereomers and E/Z
    isomers, so their scatter falls between groups and is not counted here.
    See ``held_out_floor_stereo_blind`` for the complement.
    """
    keys = mset.ids.get("collapse_key")
    if keys is None or mset.n_atoms == 0:
        return 0.0
    sse = _orbit_sse(mset.mols, mset.atom_property, _group_indices(keys), stereo=True)
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
    sse = _orbit_sse(
        mset.mols, mset.atom_property, _stereo_blind_groups(mset), stereo=False
    )
    return float(np.sqrt(sse / mset.n_atoms))


def floor_components(mset: MoleculeSet) -> dict[str, float]:
    """The additive pieces a floor is made of: two SSEs and an atom count.

    A floor is ``sqrt(SSE / n_atoms)``, and both terms are plain sums over
    rows -- so the pieces for a union of shards are the pieces of each shard
    added together, and a fold's floor never needs recomputing from
    molecules. That is only valid because no group spans a shard, which is
    not an assumption: ``annotate_collapse`` refuses a ``collapse_key`` group
    that straddles one, and the stereo-blind grouping (coarser, so the one
    that could break it) was measured on the real corpus at 0 straddling
    across split, cluster and shard alike.
    """
    keyed = mset.ids.get("collapse_key")
    sse_keyed = 0.0
    if keyed is not None and mset.n_atoms:
        sse_keyed = _orbit_sse(
            mset.mols, mset.atom_property, _group_indices(keyed), stereo=True
        )

    sse_blind = 0.0
    if mset.n_atoms:
        sse_blind = _orbit_sse(
            mset.mols, mset.atom_property, _stereo_blind_groups(mset), stereo=False
        )

    return {
        "sse": sse_keyed,
        "sse_stereo_blind": sse_blind,
        "n_atoms": float(mset.n_atoms),
    }


def floors_from_components(parts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Combine per-shard ``floor_components`` into one held-out set's floors."""
    sse = sse_blind = n_atoms = 0.0
    for part in parts:
        sse += part["sse"]
        sse_blind += part["sse_stereo_blind"]
        n_atoms += part["n_atoms"]
    if not n_atoms:
        return {"floor/rmse": 0.0, "floor/rmse_stereo_blind": 0.0}
    return {
        "floor/rmse": float(np.sqrt(sse / n_atoms)),
        "floor/rmse_stereo_blind": float(np.sqrt(sse_blind / n_atoms)),
    }


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
