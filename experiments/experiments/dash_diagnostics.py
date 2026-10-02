"""Pairs of records of one structure in one DASH subset, and what they disagree on.

Every pair of records that share a ``collapse_key`` and a ``subset`` is compared by

* the heavy-atom RMSD after optimal superposition (``rdMolAlign.GetBestRMS``), minimised
  over the atom maps that molecular symmetry allows and over the reflection of the
  second record, since mirror-image conformations carry identical MBIS charges; a pair
  below ``COPY_RMSD`` holds two copies of one conformation;
* the charge discrepancy of the manuscript: the charges of each equivalence class are
  sorted in both records and the largest absolute difference between corresponding
  entries is kept, for the MBIS charges and for every population-charge set the pair
  carries;
* the absolute difference of every total energy the pair carries (kcal/mol);
* the largest difference between the lengths of corresponding bonds (angstrom), bonds
  grouped by the pair of equivalence classes of their atoms and sorted within each
  group.

One atom map per pair (``collapse.aligned_values`` on the atom indices) serves every
per-atom quantity. The rotatable bonds of each structure (RDKit's default definition,
hydrogens removed) are recorded beside the pairs, for the strata of the pair-RMSD
figure.

Ported from sieve_paper's ``data_analysis/pair_rmsd.py``, ``pair_charges.py``,
``pair_energies.py`` and ``pair_bond_lengths.py``, whose numbers it reproduces.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("experiments")

COPY_RMSD = 0.17  # angstrom
HARTREE = 627.5095  # kcal/mol
CHARGE_SETS = ("mbis", "lowdin", "mulliken", "xtb")
ENERGIES = ("e_tpssh", "e_xtb", "e_dft", "e_gfn2")
PAIR_COLUMNS = (
    "row_a",
    "row_b",
    "subset",
    "rmsd",
    "mirror",
    *(f"d_{k}" for k in CHARGE_SETS),
    *(f"de_{k[2:]}" for k in ENERGIES),
    "max_bond_difference",
)
TASK_GROUPS = 400


def _mirror(mol: Any) -> Any:
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    for k in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(k)
        conf.SetAtomPosition(k, Point3D(-p.x, p.y, p.z))
    return mol


def _sorted_discrepancy(
    a: np.ndarray | None, b: np.ndarray | None, orbit: np.ndarray
) -> float:
    """Largest difference between the sorted values of each orbit; NaN when either
    side is absent."""
    if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    worst = 0.0
    for o in np.unique(orbit):
        cols = orbit == o
        worst = max(worst, float(np.max(np.abs(np.sort(a[cols]) - np.sort(b[cols])))))
    return worst


def compare_pair(
    mols: list[Any],
    heavy: list[Any],
    mirrors: list[Any],
    i: int,
    j: int,
    charges: list[dict[str, np.ndarray | None]],
    energies: list[dict[str, float]],
) -> tuple:
    """Every quantity of ``PAIR_COLUMNS`` after ``row_a, row_b, subset`` for records
    ``i`` and ``j``."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    from experiments.collapse import aligned_values

    proper = rdMolAlign.GetBestRMS(
        Chem.Mol(heavy[i]), Chem.Mol(heavy[j]), maxMatches=10000
    )
    mirror = rdMolAlign.GetBestRMS(
        Chem.Mol(heavy[i]), Chem.Mol(mirrors[j]), maxMatches=10000
    )

    pair = [Chem.Mol(mols[i]), Chem.Mol(mols[j])]
    for mol in pair:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("_index", float(atom.GetIdx()))
    index, orbit = aligned_values(pair, "_index", stereo=True)
    match = index[1].astype(int)

    d = []
    for k in CHARGE_SETS:
        other = charges[j][k]
        matched = None if other is None else other[match]
        d.append(_sorted_discrepancy(charges[i][k], matched, orbit))
    de = [abs(energies[i][k] - energies[j][k]) * HARTREE for k in ENERGIES]

    xa, xb = (
        pair[0].GetConformer().GetPositions(),
        pair[1].GetConformer().GetPositions(),
    )
    groups: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    for bond in pair[0].GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lo, hi = sorted((int(orbit[u]), int(orbit[v])))
        la, lb = groups.setdefault((lo, hi), ([], []))
        la.append(float(np.linalg.norm(xa[u] - xa[v])))
        lb.append(float(np.linalg.norm(xb[match[u]] - xb[match[v]])))
    bond = max(
        (float(np.max(np.abs(np.sort(p) - np.sort(q)))) for p, q in groups.values()),
        default=0.0,
    )
    return (min(proper, mirror), bool(mirror < proper), *d, *de, bond)


def _task(task: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """Pairs and rotatable bonds for a batch of groups ``(rows, subset, blobs,
    charges, energies)``."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    pairs, rotors = [], []
    for rows, subset, blobs, charges, energies in task:
        mols = [blob_to_mol(b) for b in blobs]
        for mol, c in zip(mols, charges, strict=True):
            c["mbis"] = np.array(
                [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()]
            )
        heavy = [Chem.RemoveHs(m) for m in mols]
        rotors.append(
            (
                int(rows[0]),
                subset,
                int(rdMolDescriptors.CalcNumRotatableBonds(heavy[0])),
            )
        )
        mirrors = [_mirror(m) for m in heavy]
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                values = compare_pair(mols, heavy, mirrors, i, j, charges, energies)
                pairs.append((int(rows[i]), int(rows[j]), subset, *values))
    return pairs, rotors


def diagnose_pairs(parsed: Any, fields: Any, *, workers: int = 16) -> tuple[Any, Any]:
    """``(pairs, structures)`` for a staging store.

    ``parsed`` holds ``mol``, ``subset`` and ``collapse_key`` per row and ``fields`` the
    per-atom population charges and per-record energies of
    ``dash_subsets.record_fields``, row-aligned. ``pairs`` has the columns
    ``PAIR_COLUMNS``; ``structures`` one row per ``collapse_key`` and ``subset`` with
    the number of records and the rotatable bonds. Workers are spawned, as in
    ``geometry.annotate_geometry``; a calling script needs an ``if __name__ ==
    "__main__"`` guard.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    groups = parsed.groupby(["collapse_key", "subset"], sort=False).indices
    items = []
    for (_, subset), rows in groups.items():
        if len(rows) < 2:
            continue
        charges = [
            {k: _array(fields, k, r) for k in CHARGE_SETS if k != "mbis"} for r in rows
        ]
        energies = [{k: _scalar(fields, k, r) for k in ENERGIES} for r in rows]
        items.append(
            (rows, subset, [parsed["mol"].iat[r] for r in rows], charges, energies)
        )
    tasks = [items[k : k + TASK_GROUPS] for k in range(0, len(items), TASK_GROUPS)]
    pairs, rotors = [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for p, r in pool.map(_task, tasks):
            pairs += p
            rotors += r
    pairs = pd.DataFrame(pairs, columns=pd.Index(PAIR_COLUMNS))
    sizes = (
        parsed.groupby(["collapse_key", "subset"], sort=False)
        .size()
        .rename("records")
        .reset_index()
    )
    rot = pd.DataFrame(rotors, columns=pd.Index(["row", "subset", "rotatable_bonds"]))
    rot["collapse_key"] = parsed["collapse_key"].to_numpy()[rot["row"].to_numpy()]
    structures = sizes.merge(
        rot[["collapse_key", "subset", "rotatable_bonds"]],
        how="left",
        on=["collapse_key", "subset"],
    )
    singles = structures["rotatable_bonds"].isna()
    if singles.any():
        first = parsed.groupby(["collapse_key", "subset"], sort=False).head(1)
        first = first.set_index(["collapse_key", "subset"])["mol"]
        structures.loc[singles, "rotatable_bonds"] = [
            _rotatable(first.loc[(k, s)])
            for k, s in structures.loc[singles, ["collapse_key", "subset"]].itertuples(
                index=False
            )
        ]
    structures["rotatable_bonds"] = structures["rotatable_bonds"].astype("int32")
    logger.info("compared %d pairs in %d structures", len(pairs), len(structures))
    return pairs, structures


def _rotatable(blob: bytes) -> int:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from experiments.data import blob_to_mol

    return int(rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveHs(blob_to_mol(blob))))


def _array(fields: Any, key: str, row: int) -> np.ndarray | None:
    value = fields[key].iat[row]
    return np.asarray(value, dtype=np.float64) if value is not None else None


def _scalar(fields: Any, key: str, row: int) -> float:
    value = fields[key].iat[row]
    return float("nan") if value is None or np.isnan(value) else float(value)
