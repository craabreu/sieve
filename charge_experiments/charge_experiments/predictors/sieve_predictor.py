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
    attribute_levels: tuple[tuple[str, ...], ...] | None = None,
    neighbor_depth: int | None = None,
    target_dim: int,
    max_wl_depth: int,
    minimum_support: int,
    shrinkage_strength: float | None,
    n_jobs: int | None = None,
) -> Any:
    """Learn ``attribute_codes``/``edge_codes`` from the training corpus and
    freeze them into a ``SieveConfig``.

    ``attribute_levels`` defaults to ``None``, meaning "one level, every
    attribute in ``attributes``" -- this series' original, still-default
    shape. Passing an explicit grouping is what makes ``neighbor_depth``
    (sieve's coarse-neighbor-attribute mechanism, design.md 3.6) usable at
    all here: a single level admits no valid coarsening depth. ``attributes``
    itself is only read as a fallback in that case; a caller supplying
    ``attribute_levels`` need not keep the two in sync -- the flat name list
    ``build_codes`` needs is derived from ``attribute_levels`` directly.

    ``n_jobs`` parallelizes ``build_codes``'s own vocabulary-discovery pass
    -- profiling showed this and ``from_rdkit`` (``_batch_for``, below) are
    ~96% of a real fit, against ~4% for ``sieve.fit`` itself.
    """
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes

    levels = attribute_levels if attribute_levels is not None else (attributes,)
    flat = [name for group in levels for name in group]
    codes, edge_codes = build_codes(train_mols, flat, n_jobs=n_jobs)
    return SieveConfig(
        target_dim=target_dim,
        attribute_levels=levels,
        attribute_codes=codes,
        edge_codes=edge_codes,
        max_wl_depth=max_wl_depth,
        neighbor_depth=neighbor_depth,
        minimum_support=minimum_support,
        shrinkage_strength=shrinkage_strength,
    )


def _batch_for(
    mols: list[Any], config: Any, *, with_target: bool, n_jobs: int | None = None
) -> Any:
    """Build a ``NodeBatch`` for ``mols`` under an already-fitted
    ``config``. ``node_order`` is left ``None``: each ``Mol``'s own atom
    order is already this series' canonical order."""
    from sieve.io.rdkit_adapter import from_rdkit

    return from_rdkit(
        mols,
        config=config,
        y_from_atom_prop="MBIScharge" if with_target else None,
        n_jobs=n_jobs,
    )


class SievePredictor:
    """A basic Sieve charge baseline: one attribute level by default, a
    handful of Weisfeiler-Lehman refinement rounds, no shrinkage -- the
    first, deliberately unengineered Sieve baseline for this series.

    ``attribute_levels``/``neighbor_depth`` are opt-in (both default to the
    single-level, no-coarsening shape above) -- passing a graded
    ``attribute_levels`` is what makes ``neighbor_depth`` usable at all
    (design.md 3.6): a single level admits no valid coarsening depth.

    ``n_jobs`` parallelizes featurization (``build_codes``/``from_rdkit``),
    which profiling showed is ~96% of a real fit -- ``sieve.fit`` itself is
    ~4%, so this is deliberately not where ``n_jobs`` is spent. ``None``
    (the default) keeps today's sequential behavior exactly.

    ``report_loo`` opts into a leave-one-out pass over the training split
    (``predict_loo_raw``), which ``runner`` scores as ``train_loo/*``. Off by
    default: it costs a second featurization of train (~38% on a real run),
    and a default that changed the shape of metrics.json mid-series would
    break comparability with runs already recorded.
    """

    name: ClassVar[str] = "sieve"

    def __init__(
        self,
        *,
        attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
        attribute_levels: tuple[tuple[str, ...], ...] | None = None,
        neighbor_depth: int | None = None,
        max_wl_depth: int = 3,
        minimum_support: int = 1,
        shrinkage_strength: float | None = None,
        n_jobs: int | None = None,
        report_loo: bool = False,
    ) -> None:
        self.attributes = tuple(attributes)
        self.attribute_levels = (
            None if attribute_levels is None else tuple(attribute_levels)
        )
        self.neighbor_depth = neighbor_depth
        self.max_wl_depth = max_wl_depth
        self.minimum_support = minimum_support
        self.shrinkage_strength = shrinkage_strength
        self.n_jobs = n_jobs
        self.report_loo = report_loo
        self._config: Any = None
        self._model: Any = None
        # Accumulated wall time spent in build_codes/from_rdkit (featurization)
        # across this predictor instance's calls -- separate from sieve.fit/
        # predict_detailed themselves. Public so runner.py can surface it as
        # its own metric: profiling showed featurization is ~96% of a real
        # fit+predict pass and time/fit_s/time/predict_s do not separate it
        # out, so that split was invisible in every recorded run. Reset at
        # the start of fit() (one run's worth), accumulated (not reset)
        # across predict_raw() calls, since a run predicts test/train/val as
        # up to three separate passes, each re-featurizing.
        self.last_featurize_s: float = 0.0

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        import time

        import sieve

        self.last_featurize_s = 0.0
        t0 = time.perf_counter()
        self._config = _build_config(
            train.mols,
            attributes=self.attributes,
            attribute_levels=self.attribute_levels,
            neighbor_depth=self.neighbor_depth,
            target_dim=1,
            max_wl_depth=self.max_wl_depth,
            minimum_support=self.minimum_support,
            shrinkage_strength=self.shrinkage_strength,
            n_jobs=self.n_jobs,
        )
        batch = _batch_for(
            train.mols, self._config, with_target=True, n_jobs=self.n_jobs
        )
        self.last_featurize_s += time.perf_counter() - t0
        self._model = sieve.fit(batch, self._config)

    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        if self._model is None or self._config is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_raw"
            )
        import time

        import sieve

        t0 = time.perf_counter()
        batch = _batch_for(
            test.mols, self._config, with_target=False, n_jobs=self.n_jobs
        )
        self.last_featurize_s += time.perf_counter() - t0
        detailed = sieve.predict_detailed(self._model, batch)
        atom_charge = np.asarray(detailed.value, dtype=np.float64)[:, 0]
        atom_std = np.sqrt(np.asarray(detailed.variance, dtype=np.float64)[:, 0])
        return RawPrediction(atom_charge=atom_charge, atom_std=atom_std)

    def predict_loo_raw(self, train: MoleculeSet) -> RawPrediction:
        """Leave-one-out prediction for the *training* split (design.md 10.3).

        A training node contributes its own target to its class mean, so any
        in-sample score is optimistic -- at minimum_support=1 and large depth
        it approaches perfect recall. ``sieve.predict_loo`` subtracts that
        contribution before the support check and treats a class with one
        member as unsupported, so the node backs off to its parent instead of
        recalling itself.

        **Train-only, and not structurally enforceable.** LOO computes
        ``(cnt*mean - y_node) / (cnt - 1)``; for a val or test node that
        subtracts a value which was never in the class mean, so the result is
        corrupt rather than merely uninformative. Every MoleculeSet in this
        series carries MBIScharge on its Mols, so a val set would satisfy
        ``predict_loo``'s only guard (``batch.y is not None``) and return
        quietly wrong numbers. The parameter is named ``train`` and
        ``runner`` calls this for the train split only.
        """
        if self._model is None or self._config is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_loo_raw"
            )
        import time

        import sieve

        t0 = time.perf_counter()
        batch = _batch_for(
            train.mols, self._config, with_target=True, n_jobs=self.n_jobs
        )
        self.last_featurize_s += time.perf_counter() - t0
        detailed = sieve.predict_loo(self._model, batch)
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
