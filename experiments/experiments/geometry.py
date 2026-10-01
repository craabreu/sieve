"""Whether a stored conformer's geometry agrees with its own molecular graph.

The charges of a record belong to its geometry, but every model reads its graph, so a
geometry that contradicts the graph pairs correct charges with the wrong molecule. No
check that reads the charges against the density can see that -- the calculation itself
is healthy -- so it is tested on the geometry alone, against covalent radii (RDKit's
periodic table):

* **Stretched bond**: a declared bond longer than ``STRETCHED`` times the sum of the
  covalent radii of its atoms. No QMugs-derived DASH geometry exceeds 1.12; SPICE 2.0.1
  already removed the conformers whose bonds broke during generation.
* **Close contact**: two heavy atoms that the graph does not bond, closer than the
  graph allows. Three or more bonds apart, the bound is ``CONTACT_FAR`` times the
  covalent sum, below which no QMugs-derived DASH geometry falls (minimum 1.17). Two
  bonds apart (a 1,3 pair, across a shared neighbour), angles around P, Si, B and S put
  healthy pairs well below that, so the bound is ``CONTACT_13``, about the distance of a
  single bond between them: the geometry then holds a three-membered ring the graph does
  not declare. Electron-deficient boron cages legitimately hold boron atoms at bonding
  distance across the cage and are flagged too.

The values are written as columns, never used to delete rows here: ``max_bond_ratio``,
``min_contact_13``, ``min_contact_far`` (NaN where the molecule has no such pair), and
the flags ``stretched_bond`` and ``close_contact``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

STRETCHED = 1.25
CONTACT_13 = 1.05
CONTACT_FAR = 1.15
GEOMETRY_COLUMNS = (
    "max_bond_ratio",
    "min_contact_13",
    "min_contact_far",
    "stretched_bond",
    "close_contact",
)

logger = logging.getLogger("experiments")


def geometry_record(mol: Any) -> dict[str, Any]:
    """The geometry columns of one stored ``Mol`` (explicit H, one conformer)."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    xyz = mol.GetConformer().GetPositions()
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    radius = np.array([table.GetRcovalent(int(v)) for v in z])
    ratio = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) / (
        radius[:, None] + radius[None]
    )

    pairs = np.array(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=int
    ).reshape(-1, 2)
    max_bond = float(ratio[pairs[:, 0], pairs[:, 1]].max()) if len(pairs) else math.nan

    # Topological distance; atoms in different fragments are 1e8 apart, i.e. "far".
    topo = Chem.GetDistanceMatrix(mol)
    heavy = z > 1
    upper = np.triu(heavy[:, None] & heavy[None, :], 1)

    def smallest(selection: np.ndarray) -> float:
        return float(ratio[selection].min()) if selection.any() else math.nan

    min_13 = smallest(upper & (topo == 2))
    min_far = smallest(upper & (topo >= 3))
    return {
        "max_bond_ratio": max_bond,
        "min_contact_13": min_13,
        "min_contact_far": min_far,
        "stretched_bond": bool(max_bond > STRETCHED),
        "close_contact": bool(min_13 < CONTACT_13 or min_far < CONTACT_FAR),
    }


def arrow_fields() -> list:
    """The geometry columns as pyarrow fields, for the builders' parquet schemas."""
    import pyarrow as pa

    return [
        ("max_bond_ratio", pa.float64()),
        ("min_contact_13", pa.float64()),
        ("min_contact_far", pa.float64()),
        ("stretched_bond", pa.bool_()),
        ("close_contact", pa.bool_()),
    ]


def _records(blobs: list[bytes]) -> list[dict[str, Any]]:
    from rdkit import rdBase

    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    return [geometry_record(blob_to_mol(blob)) for blob in blobs]


def annotate_geometry(
    store: str, *, stores_root: Path, workers: int = 16
) -> dict[str, int]:
    """Add (or recompute) the geometry columns of an existing store, in place.

    For stores built before the builders wrote these columns. The parquet is rewritten
    through a temporary file, so an interrupted run leaves the store as it was.
    Workers are spawned, as in the builders; a calling script needs an
    ``if __name__ == "__main__"`` guard.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    path = Path(stores_root) / store / "molecules.parquet"
    df = pd.read_parquet(path)
    blobs = df["mol"].tolist()
    chunk = 5000
    pieces = [blobs[k : k + chunk] for k in range(0, len(blobs), chunk)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        records = [r for part in pool.map(_records, pieces) for r in part]
    values = pd.DataFrame(records, index=df.index)
    for column in GEOMETRY_COLUMNS:
        df[column] = values[column]
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return {
        "rows": len(df),
        "stretched_bond": int(df["stretched_bond"].sum()),
        "close_contact": int(df["close_contact"].sum()),
        "either": int((df["stretched_bond"] | df["close_contact"]).sum()),
    }
