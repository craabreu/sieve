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

``predict_raw``/``save_model_state``/``load_model_state`` mirror
predictors/dash.py's own nested-runs support (see
docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md):
``predict_raw`` uses ``sieve.predict_detailed`` rather than the plain
``sieve.predict`` wrapper -- the two compute identically (``predict`` is
literally ``predict_detailed(...).value``), so this costs nothing extra,
and it additionally exposes each atom's class ``variance`` as a real
``atom_std`` (``sqrt(variance)``, NaN wherever ``support == 1`` -- sieve's
own "no spread observed" case, not invented here), rather than a filler
value. Only ``normalize.equal_weighted_normalize`` is wired into this
series' own nested example config for now (see
configs/sieve-nested-charge-example.yaml's ``children`` list) -- std_weighted
normalization is left for a follow-up once this ``atom_std`` has been
checked against real data, not because ``predict_raw`` itself is missing
anything std_weighted would need. ``save_model_state``/``load_model_state``
delegate directly to ``sieve.SieveModel.save``/``.load`` (a single ``.npz``,
already self-describing via its own ``format_version``/``schema_version``
guards) -- no bespoke serialization needed here, unlike
predictors/dash.py's own per-node stats table.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from charge_experiments.data import MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction, RawPrediction

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

    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        if self._model is None or self._config is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_raw"
            )
        import sieve

        batch = _batch_for(test.mols, self._config, with_target=False)
        detailed = sieve.predict_detailed(self._model, batch)
        atom_charge = np.asarray(detailed.value, dtype=np.float64)[:, 0]
        atom_std = np.sqrt(np.asarray(detailed.variance, dtype=np.float64)[:, 0])
        return RawPrediction(atom_charge=atom_charge, atom_std=atom_std)

    def predict(self, test: MoleculeSet) -> Prediction:
        return Prediction(atom_charge=self.predict_raw(test).atom_charge)

    def save_model_state(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("fit must be called before save_model_state")
        self._model.save(path)

    def load_model_state(self, path: str | Path) -> None:
        """Skips fit()'s own sieve.fit() call entirely -- SieveModel.load
        reconstructs both the fitted model and the SieveConfig it was fit
        with (attribute_codes/edge_codes included), so this predictor is
        immediately ready for predict_raw()."""
        import sieve

        self._model = sieve.SieveModel.load(path)
        self._config = self._model.config


def _build(params: Mapping[str, Any]) -> SievePredictor:
    return SievePredictor(**params)


register("sieve", _build)
