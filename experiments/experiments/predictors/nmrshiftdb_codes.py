"""HOSE codes generated as NMRShiftDB generates them, by its own Java code.

NMRShiftDB keeps, for every atom of its database, two codes at six spheres
(``hosecode.spheres`` of its configuration): the plain code of CDK's
``HOSECodeGenerator`` and a code with ring sizes from its own
``ExtendedHOSECodeGenerator`` (``SubmittingData.saveMolecule``). A prediction
generates one plain code of the query atom and strips it one sphere at a time
(``PredictionTool.generalPredict``). All three codes come from
``experiments/nmrshiftdb/NmrshiftdbCodes.java``, run against the jars of
NMRShiftDB revision 2699, which ``experiments/nmrshiftdb/setup.sh`` fetches and
builds; nothing here re-implements a generator.

Two things about the molecules are ours, not NMRShiftDB's. Aromaticity is
RDKit's, as on the other HOSE arms, passed as flags on atoms and bonds; and no
bond carries a wedge, so the extended generator writes no stereo descriptors.
And where ``PredictionTool`` codes a heavy atom on the molecule with its
hydrogens made implicit, it reads that atom at the index it has in the molecule
*with* hydrogens, which is the same atom only when every hydrogen comes after
every heavy atom, as in NMRShiftDB's own molecules. The driver reads it at its
index among the heavy atoms, so the code is that of the right atom whatever
the order of the atoms.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: Spheres of the codes NMRShiftDB stores (``hosecode.spheres=6`` in
#: ``NMRShiftDB.properties``), and so the most a query can use.
STORED_SPHERES: int = 6

_DEFAULT_HOME = Path(__file__).resolve().parents[2] / "external" / "nmrshiftdb2-r2699"


def nmrshiftdb_home() -> Path:
    """Where ``setup.sh`` built the generator: ``$NMRSHIFTDB_HOME`` if set,
    else ``experiments/external/nmrshiftdb2-r2699``."""
    home = Path(os.environ.get("NMRSHIFTDB_HOME", _DEFAULT_HOME))
    if not (home / "classes" / "NmrshiftdbCodes.class").is_file():
        raise FileNotFoundError(
            f"no NMRShiftDB code generator under {home}; run "
            "experiments/nmrshiftdb/setup.sh first"
        )
    return home


def _bond_order(bond: Any) -> int:
    from rdkit import Chem

    kind = bond.GetBondType()
    if bond.GetIsAromatic() or kind == Chem.BondType.AROMATIC:
        # The generator reads an aromatic bond by its flag alone; the order is
        # never consulted, so any is as good as the Kekule one.
        return 1
    orders = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }
    if kind not in orders:
        raise ValueError(f"bond type {kind} has no NMRShiftDB order")
    return orders[kind]


def block(mol: Any) -> str:
    """The driver's input for one molecule: atoms with element, formal charge
    and aromatic flag, bonds with 0-based atoms, order and aromatic flag. It
    holds everything the generators read, so two molecules with the same block
    get the same codes, atom by atom. Every hydrogen must be an explicit
    atom."""
    for atom in mol.GetAtoms():
        if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
            raise ValueError(
                "every hydrogen must be an explicit atom, as on the "
                "molecules NMRShiftDB codes"
            )
    lines = [f"{mol.GetNumAtoms()} {mol.GetNumBonds()}"]
    lines.extend(
        f"{a.GetSymbol()} {a.GetFormalCharge()} {int(a.GetIsAromatic())}"
        for a in mol.GetAtoms()
    )
    lines.extend(
        f"{b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} {_bond_order(b)} "
        f"{int(b.GetIsAromatic())}"
        for b in mol.GetBonds()
    )
    return "\n".join(lines) + "\n"


def _run(blocks: Sequence[str], query_spheres: int, home: Path) -> list[list[str]]:
    classpath = ":".join(
        [str(home / "classes"), *sorted(str(p) for p in (home / "lib").glob("*.jar"))]
    )
    proc = subprocess.run(
        [
            str(home / "jdk" / "bin" / "java"),
            "-Xss16m",
            "-cp",
            classpath,
            "NmrshiftdbCodes",
            str(STORED_SPHERES),
            str(query_spheres),
        ],
        input="".join(blocks).encode(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"NmrshiftdbCodes failed ({proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[-2000:]}"
        )
    n_fields = 3 if query_spheres > 0 else 2
    rows = [line.split("\t") for line in proc.stdout.decode().split("\n")[:-1]]
    n_atoms = sum(int(b.split(" ", 1)[0]) for b in blocks)
    if len(rows) != n_atoms or any(len(r) != n_fields for r in rows):
        raise RuntimeError(
            f"NmrshiftdbCodes returned {len(rows)} rows for {n_atoms} atoms"
        )
    return rows


def nmrshiftdb_codes(
    mols: Sequence[Any], query_spheres: int = 0, jobs: int | None = None
) -> list[list[str]]:
    """Per atom, in the flattened atom order of ``mols``: the plain stored
    code, the stored code with ring sizes, and, when ``query_spheres`` is
    positive, the plain query code at that many spheres. An empty string is a
    code the generator could not produce.

    Molecules with the same input block, such as the conformers of one
    structure, are coded once. ``jobs`` Java processes share the distinct
    blocks (default ``$NMRSHIFTDB_JOBS``, else 1)."""
    if not 0 <= query_spheres <= STORED_SPHERES:
        raise ValueError(
            f"query_spheres must lie in 0..{STORED_SPHERES}, got {query_spheres}"
        )
    home = nmrshiftdb_home()
    blocks = [block(m) for m in mols]
    if not blocks:
        return []
    distinct = list(dict.fromkeys(blocks))
    jobs = max(
        1, int(jobs if jobs is not None else os.environ.get("NMRSHIFTDB_JOBS", 1))
    )
    jobs = min(jobs, len(distinct))
    bounds = [len(distinct) * i // jobs for i in range(jobs + 1)]
    chunks = [distinct[bounds[i] : bounds[i + 1]] for i in range(jobs)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(lambda c: _run(c, query_spheres, home), chunks))
    rows_of: dict[str, list[list[str]]] = {}
    flat = iter(row for part in parts for row in part)
    for b in distinct:
        rows_of[b] = [next(flat) for _ in range(int(b.split(" ", 1)[0]))]
    return [row for b in blocks for row in rows_of[b]]
