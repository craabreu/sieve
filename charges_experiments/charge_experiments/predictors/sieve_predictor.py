"""A basic Sieve charge predictor: this project's own hierarchical
regressogram (``sieve.fit``/``sieve.predict``) wired onto per-atom
``MBIScharge`` prediction directly via
``sieve.io.rdkit_adapter.from_rdkit``'s ``y_from_atom_prop`` option -- no
separate, position-aligned ``y`` array, since the target already rides on
each conformer's own ``Mol`` (see the design spec's "Sieve's target
ingestion" decision). No SMILES, no atom-map-order recovery: ``mset.mols``
are already the canonical, deserialized ``Mol`` objects, so
``node_order=None`` (natural atom order) is exactly right.

Attribute set mirrors DASH's own atom feature tuple (see
predictors/dash.py's docstring and cosmo_experiments' own precedent),
``max_wl_depth``/``minimum_support`` are starting values, deliberately not
tuned -- see the design spec's "Out of scope" list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction

DEFAULT_ATTRIBUTES = ("element", "degree", "formal_charge", "aromatic", "num_h")


def _build_config(
    train_mols: list[Any],
    *,
    attributes: tuple[str, ...],
    target_dim: int,
    max_wl_depth: int,
    minimum_support: int,
    shrinkage_strength: float | None,
) -> Any:
    """Learn ``attribute_codes``/``edge_codes`` from the training corpus and
    freeze them into a ``SieveConfig``."""
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes

    codes, edge_codes = build_codes(train_mols, attributes)
    return SieveConfig(
        target_dim=target_dim,
        attribute_levels=(attributes,),
        attribute_codes=codes,
        edge_codes=edge_codes,
        max_wl_depth=max_wl_depth,
        minimum_support=minimum_support,
        shrinkage_strength=shrinkage_strength,
    )


def _batch_for(mols: list[Any], config: Any, *, with_target: bool) -> Any:
    """Build a ``NodeBatch`` for ``mols`` under an already-fitted
    ``config``. ``node_order`` is left ``None``: each ``Mol``'s own atom
    order is already this series' canonical order."""
    from sieve.io.rdkit_adapter import from_rdkit

    return from_rdkit(
        mols, config=config, y_from_atom_prop="MBIScharge" if with_target else None
    )


class SievePredictor:
    """A basic Sieve charge baseline: one attribute level, a handful of
    Weisfeiler-Lehman refinement rounds, no shrinkage -- the first,
    deliberately unengineered Sieve baseline for this series."""

    name: ClassVar[str] = "sieve"

    def __init__(
        self,
        *,
        attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
        max_wl_depth: int = 3,
        minimum_support: int = 1,
        shrinkage_strength: float | None = None,
    ) -> None:
        self.attributes = tuple(attributes)
        self.max_wl_depth = max_wl_depth
        self.minimum_support = minimum_support
        self.shrinkage_strength = shrinkage_strength
        self._config: Any = None
        self._model: Any = None

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        import sieve

        self._config = _build_config(
            train.mols, attributes=self.attributes, target_dim=1,
            max_wl_depth=self.max_wl_depth, minimum_support=self.minimum_support,
            shrinkage_strength=self.shrinkage_strength,
        )
        batch = _batch_for(train.mols, self._config, with_target=True)
        self._model = sieve.fit(batch, self._config)

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._model is None or self._config is None:
            raise RuntimeError("fit must be called before predict")
        import sieve

        batch = _batch_for(test.mols, self._config, with_target=False)
        atom_charge_2d = sieve.predict(self._model, batch)
        atom_charge_arr = np.asarray(atom_charge_2d, dtype=np.float64)
        atom_charge: NDArray[np.float64] = atom_charge_arr[:, 0]
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> SievePredictor:
    return SievePredictor(**params)


register("sieve", _build)
