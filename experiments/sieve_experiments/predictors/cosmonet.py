"""COSMO-NET (DMPNN) baseline: a lookup against an externally trained
checkpoint's own saved predictions, not a live re-inference.

Training itself happens outside this harness entirely, in a separate
Python 3.10 venv with its own torch/tensorflow/deepchem stack
(experiments/external/COSMO-NET-Paper/.venv) -- incompatible with the main
project's env (Python 3.12, no CUDA deps) and deliberately excluded from the
uv workspace (see pins.toml's ``[dash_tree]`` gotcha, which applies equally
here). See pins.toml's ``[cosmonet]`` notes for the training command
(``Training/DMPNN/DMPNN-Train-pSigma.py --splitter 4``, i.e. driven by a
CATEGORY column built from one of our own split columns via
``cosmonet_data.category_labels``).

That script's own ``save_fold_outputs`` already runs the trained model's
``predict()`` (correctly inverse-transformed through the fitted
``MinMaxTransformer``) on *every* molecule in the source CSV -- train, val,
and test alike -- and writes it to ``Results/pSigma-{model}.csv``. This
predictor is therefore a SMILES-keyed lookup against that file, not a
process that runs the model itself:

- For any molecule that was in the CSV the checkpoint was trained from, the
  lookup is numerically identical to what live inference would give -- it
  *is* that inference's own output, not an approximation of it.
- It does NOT generalize to molecules outside that CSV: there is no live
  model object here, so a genuinely novel SMILES raises an error rather
  than producing a prediction. Live re-inference (via a subprocess into the
  cosmonet venv) would be needed to lift that limitation -- not built here,
  since nothing in this milestone predicts on out-of-store molecules yet.

CATEGORY/split-column guard: the CSV's ``CATEGORY`` labels are frozen at
whichever ``split_column`` built them for that particular training run
(``--splitter 4`` reads them, but doesn't know or care which Sieve split
they came from). Pointing a run at a predictions CSV built from a
*different* split_column than the run's own ``data.split_column`` would
silently let molecules that were "Training"/"Validation" during the
checkpoint's own training count as this run's held-out "test" set --
undetectable from the metrics alone. ``fit_molecules``/``predict_molecules``
re-certify every molecule's CATEGORY against what this run expects, every
run, the same check done manually and reported to the user for the
2026-08-25 chaos-store pass (0 mismatches out of 53079).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sieve_experiments.data import DEFAULT_GRID, MoleculeSet, SigmaGridSpec
from sieve_experiments.predictors import register
from sieve_experiments.predictors.base import MoleculePredictor, Prediction

_PROFILE = dict[str, NDArray[np.float64]]
_CATEGORY = dict[str, str]


def _load_predictions(
    csv_path: Path, grid: SigmaGridSpec
) -> tuple[_PROFILE, _CATEGORY]:
    """Read a COSMO-NET ``pSigma-*.csv`` into SMILES-keyed profile + CATEGORY lookups.

    Same column convention as ``cosmonet_data.write_cosmonet_csv`` produces
    as input (and this file is COSMO-NET's own output, in the same shape):
    a ``CanonicalSMILES`` column, a ``CATEGORY`` column, and one column per
    grid point, named by its rounded sigma value.
    """
    sigma_cols = [str(round(float(v), 3)) for v in grid.values]
    wanted_cols = {"CanonicalSMILES", "CATEGORY", *sigma_cols}
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted_cols)
    dupes = df["CanonicalSMILES"][df["CanonicalSMILES"].duplicated()]
    if len(dupes):
        raise ValueError(
            f"duplicate SMILES in {csv_path}, cannot look up predictions "
            f"unambiguously: {dupes.iloc[0][:60]!r}"
        )
    values = df[sigma_cols].to_numpy(dtype=np.float64)
    profile_by_smiles = dict(zip(df["CanonicalSMILES"], values, strict=True))
    category_by_smiles = dict(zip(df["CanonicalSMILES"], df["CATEGORY"], strict=True))
    return profile_by_smiles, category_by_smiles


class CosmonetPredictor(MoleculePredictor):
    name = "cosmonet"

    def __init__(
        self,
        *,
        predictions_csv: str,
        store: str,
        scheme: str,
        grid: SigmaGridSpec = DEFAULT_GRID,
    ) -> None:
        self.predictions_csv = predictions_csv
        self.store = store
        self.scheme = scheme
        self.grid = grid
        self._by_smiles: _PROFILE | None = None
        self._category_by_smiles: _CATEGORY | None = None

    def _certify_category(self, mset: MoleculeSet, expected: str, role: str) -> None:
        assert self._category_by_smiles is not None
        bad = [s for s in mset.smiles if self._category_by_smiles.get(s) != expected]
        if bad:
            raise ValueError(
                f"{len(bad)}/{mset.n_molecules} {role} molecules are missing "
                f"from, or not labeled {expected!r} in, {self.predictions_csv}'s "
                f"CATEGORY column, e.g. {bad[0][:60]!r} -- that CSV's CATEGORY "
                "must have been built from this run's own split_column, or "
                "this run's held-out set may include molecules the checkpoint "
                "was itself trained on (see module docstring)"
            )

    def fit_molecules(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del rng  # nothing to fit -- see module docstring
        self._by_smiles, self._category_by_smiles = _load_predictions(
            Path(self.predictions_csv), self.grid
        )
        self._certify_category(train, "Training", "train")
        self._certify_category(val, "Validation", "val")

    def predict_molecules(self, test: MoleculeSet) -> Prediction:
        if self._by_smiles is None or self._category_by_smiles is None:
            raise RuntimeError("fit_molecules must be called before predict_molecules")
        self._certify_category(test, "Test", "test")
        mol_profile = np.stack([self._by_smiles[s] for s in test.smiles])
        mol_area = mol_profile.sum(axis=1)
        # Same convention as cosmolayer's own charge column: sigma-weighted
        # sum of the unnormalized profile, i.e. Sum(sigma * p(sigma)) --
        # verified equal to atom_table.charges bit-for-bit in data.py.
        mol_charge_raw = mol_profile @ self.grid.values
        return Prediction(
            mol_profile=mol_profile, mol_area=mol_area, mol_charge_raw=mol_charge_raw
        )


def _build(params: Mapping[str, Any]) -> CosmonetPredictor:
    return CosmonetPredictor(**params)


register("cosmonet", _build)
