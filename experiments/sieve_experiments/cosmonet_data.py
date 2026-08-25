"""Convert a Sieve ``MoleculeSet`` into COSMO-NET-Paper's CSAC-style CSV.

``DMPNN-Train-pSigma.py`` (experiments/external/COSMO-NET-Paper) reads a CSV
with an exact column layout: some metadata columns, then
``data.columns[10:]`` sliced *positionally* as the 51 sigma-profile values --
only ``CanonicalSMILES`` and ``CATEGORY`` are referenced by name anywhere in
the training script, everything else in the first 10 columns is cosmetic.
See experiments/pins.toml's ``[cosmonet]`` notes for how this was found
(deepchem's DMPNNModel import failure, the pretrained-checkpoint caveat,
the dependency pin) and confirmed (grid match, positional slicing).

``DEFAULT_GRID`` (51 points, sigma in [-0.025, 0.025], bin width 0.001) is
identical to COSMO-NET's own grid (see the design doc and data.py's own
comment on ``DEFAULT_GRID``), so ``MoleculeSet.mol_profile`` values transfer
to the sigma columns as-is -- no rescaling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sieve_experiments.data import DEFAULT_GRID, MoleculeSet

_METADATA_COLUMNS = [
    "ID",
    "FORMULA",
    "CAS",
    "NAME",
    "CanonicalSMILES",
    "INCHI",
    "INCHIKEY",
    "CATEGORY",
    "area [A^2]",
    "volume [A^3]",
]

# Sieve's own split names -> COSMO-NET's CATEGORY vocabulary (splitter 4:
# "Stratified from CATEGORY column").
_CATEGORY_BY_SPLIT = {"train": "Training", "val": "Validation", "test": "Test"}


def category_labels(
    masks: dict[str, NDArray[np.bool_]], n_molecules: int
) -> NDArray[np.str_]:
    """Map Sieve's train/val/test boolean masks onto COSMO-NET's CATEGORY strings.

    Raises if any molecule is in zero or more than one split -- COSMO-NET's
    CATEGORY column assumes a strict partition of every row.
    """
    labels = np.full(n_molecules, "", dtype=object)
    membership = np.zeros(n_molecules, dtype=np.int64)
    for split, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != n_molecules:
            raise ValueError(
                f"{split!r} mask has {len(mask)} entries, expected {n_molecules}"
            )
        labels[mask] = _CATEGORY_BY_SPLIT[split]
        membership += mask.astype(np.int64)
    if np.any(membership != 1):
        bad = int(np.sum(membership != 1))
        raise ValueError(
            f"{bad}/{n_molecules} molecules are in zero or more than one split; "
            "COSMO-NET's CATEGORY column requires a strict partition"
        )
    return labels.astype(str)


def _sigma_column_names(grid) -> list[str]:
    return [str(round(float(v), 3)) for v in grid.values]


def write_cosmonet_csv(
    mset: MoleculeSet, category: NDArray[np.str_] | list[str], out_path: Path
) -> None:
    """Write ``mset`` (all molecules, any split, any order) as a COSMO-NET CSV.

    ``category`` labels each molecule "Training"/"Validation"/"Test", aligned
    to ``mset.smiles`` -- build it with ``category_labels`` from Sieve's own
    split masks. Non-target metadata columns (ID/FORMULA/CAS/NAME/INCHI/
    INCHIKEY) are filled with placeholders: COSMO-NET's training script never
    reads them by content, only by position/count.
    """
    if mset.mol_profile is None:
        raise ValueError(
            "mset.mol_profile is required (unnormalized molecule sigma profile)"
        )
    if not np.allclose(mset.grid.values, DEFAULT_GRID.values):
        raise ValueError("mset.grid does not match COSMO-NET's expected 51-point grid")
    n = mset.n_molecules
    category = list(category)
    if len(category) != n:
        raise ValueError(f"category has {len(category)} entries, expected {n}")

    mol_area = mset.mol_area if mset.mol_area is not None else np.full(n, np.nan)
    metadata = pd.DataFrame(
        {
            "ID": np.arange(n),
            "FORMULA": [""] * n,
            "CAS": [""] * n,
            "NAME": [""] * n,
            "CanonicalSMILES": list(mset.smiles),
            "INCHI": [""] * n,
            "INCHIKEY": [""] * n,
            "CATEGORY": category,
            "area [A^2]": np.asarray(mol_area, dtype=np.float64),
            "volume [A^3]": np.full(n, np.nan),
        },
        columns=pd.Index(_METADATA_COLUMNS),
    )
    sigma = pd.DataFrame(
        mset.mol_profile, columns=pd.Index(_sigma_column_names(mset.grid))
    )
    out = pd.concat(
        [metadata.reset_index(drop=True), sigma.reset_index(drop=True)], axis=1
    )
    out.to_csv(Path(out_path), index=False)
