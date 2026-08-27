"""DASH-tree charge predictor using the tree's own **published** charge
statistics, with **zero training on our own data** and **no invented
fallback behavior**.

Unlike ``predictors/dash.py``'s ``DASHChargePredictor`` -- which
re-populates the tree with a brand-new per-node statistic derived from our
own train split, because that's what Stage A's own precedent requires for
a target the tree was never built to predict (sigma-profile, in
cosmo_experiments) -- this series' target, ``MBIScharge``, is *exactly*
the property DASH-tree's own default, shipped tree data already holds:
every node's ``tree.default_value_column`` (``"result"``) is the paper's
own published per-node MBIS charge mean, populated once, by the paper's
own authors, from their own full training run -- not something this
predictor derives. ``fit`` is therefore a genuine no-op (it never reads
``train``'s own atom charges at all -- see the optional test that proves
this by fitting on two wildly different synthetic train sets and checking
``predict`` gives bit-identical output either way).

**Does not call ``DASHTree.get_molecules_partial_charges``.** That method
-- the one documented, public charge-prediction entry point -- has a real,
confirmed bug at the pinned commit: its own ``get_property_noNAN`` treats
``matched_node_path[0]`` as *both* the branch index (consumed once,
correctly) *and*, erroneously, as a candidate node id in its own walk loop
(``for atom in reversed(matched_node_path)`` iterates the whole list,
including that same first element) -- even though its own docstring says
the list should hold "node ids ... in the order of the traversal", not the
branch index. When the walk exhausts every real node id without finding a
non-NaN value, it then does ``df.iloc[branch_idx]`` -- indexing a
per-branch dataframe by a value from an entirely different index space --
and crashes with ``IndexError: single positional indexer is out-of-bounds``.
Measured directly against 20 small, ordinary molecules (alkanes/alcohols,
squarely within DASH's own vocabulary): 18 crashed this way, 2 hit the
separate, already-documented vocabulary-gap failure
(``init_neighbor_dict`` raising on an out-of-vocabulary atom feature
tuple) -- so the published API fails outright on the large majority of
ordinary small molecules at this pinned commit, not as a rare edge case.
That failure rate is specific to that tiny, atypical synthetic fixture,
though, not a real-world number: run against the actual store's own
conformers (--limit 2000, 4322+22998+5980 test/train/val atoms), this
predictor's own bypass reports zero NaN and R2=0.997 -- real DASH-corpus
molecules match at well-populated tree nodes far more often than eight
small alkanes/alcohols happen to.

Bypassing it is not "inventing a fallback": this predictor reuses
``predictors/dash.py``'s own ``_atom_paths``/``predict_via_data_storage_walk``
(already correct, already tested -- the same deepest -> shallowest,
first-non-NaN-wins walk ``get_property_noNAN``'s own docstring describes)
pointed at the tree's own pre-existing ``default_value_column`` instead of
a freshly-populated one, with ``fallback_charge=nan`` -- i.e. genuinely no
substitution at all when nothing along a path is populated, matching
``get_property_noNAN``'s own *documented* (if buggy-in-practice) intent.
An unmatched atom's ``atom_charge`` entry is ``float("nan")``, never a
value we chose on DASH's behalf. ``metrics.regression_metrics`` is
NaN-aware precisely so this predictor's own faithfulness doesn't silently
poison a whole run's aggregate MAE/RMSE/R2 -- see that module's own
docstring. ``match_stats`` (from ``_atom_paths``) still counts unmatched
atoms/molecules -- bookkeeping about coverage, not a predicted value.

Deliberately does not apply DASH's own ``std_weighted`` charge-conservation
renormalization (the residual-redistribution step
``get_molecules_partial_charges`` layers on top of its raw per-atom
values, using each molecule's own formal charge): this predictor returns
the tree's raw, unreconciled ``"result"`` values, same as
``DASHChargePredictor``'s own current output shape -- consistent between
the two DASH baselines, not a new inconsistency introduced here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction
from charge_experiments.predictors.dash import (
    LiteralTreeChargeProperties,
    _atom_paths,
    predict_via_data_storage_walk,
)

logger = logging.getLogger("charge_experiments")


class DASHPretrainedChargePredictor:
    """DASH-tree charge baseline, exactly as published: the default tree's
    own ``"result"`` node statistics, no re-training, no invented
    missing-value fallback. See module docstring.

    ``preload`` defaults to True for the same reason
    ``DASHChargePredictor`` does (see its own docstring's GOTCHA 1 note in
    cosmo_experiments/pins.toml): the pinned commit's on-demand loading
    raises on every H atom.

    ``max_depth``/``attention_threshold`` default to the paper's own tuned
    values (16 / 5.2 -- see cosmo_experiments/pins.toml's PAPER PARAMETERS
    note) -- the same tree-matching hyperparameters ``DASHChargePredictor``
    uses, since matching itself (``match_new_atom``) is unaffected by the
    ``get_molecules_partial_charges`` bug this predictor works around.
    """

    name: ClassVar[str] = "dash_pretrained"

    def __init__(
        self,
        *,
        max_depth: int = 16,
        attention_threshold: float = 5.2,
        tree_folder_path: str | None = None,
        preload: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.attention_threshold = attention_threshold
        self.tree_folder_path = tree_folder_path
        self.preload = preload
        self.match_stats: dict[str, int] = {}
        self._tree: Any = None

    def _load_tree(self) -> Any:
        if self._tree is None:
            from serenityff.charge.tree.dash_tree import DASHTree

            kwargs: dict[str, Any] = {"preload": self.preload, "verbose": False}
            if self.tree_folder_path is not None:
                kwargs["tree_folder_path"] = self.tree_folder_path
            self._tree = DASHTree(**kwargs)
        return self._tree

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        """Loads the default tree; reads nothing from ``train``/``val``.
        Their presence in this signature is the ``Predictor`` protocol's
        own shape, not something this predictor uses -- see the optional
        test that proves ``predict`` is identical regardless of what
        ``train`` contains."""
        del train, val, rng
        self._load_tree()

    def predict(self, test: MoleculeSet) -> Prediction:
        tree = self._load_tree()
        paths, stats = _atom_paths(
            test,
            tree,
            max_depth=self.max_depth,
            attention_threshold=self.attention_threshold,
        )
        self.match_stats = stats
        if stats["n_unmatched_atoms"]:
            logger.warning(
                "DASH (pretrained) could not match %d/%d atoms (%d/%d molecules "
                "rejected outright); these are reported as NaN, not guessed at",
                stats["n_unmatched_atoms"],
                stats["n_atoms"],
                stats["n_unmatched_molecules"],
                stats["n_conformers"],
            )
        props = LiteralTreeChargeProperties(
            charge_column=tree.default_value_column, fallback_charge=float("nan")
        )
        atom_charge: NDArray[np.float64] = predict_via_data_storage_walk(
            tree, paths, props
        )
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> DASHPretrainedChargePredictor:
    return DASHPretrainedChargePredictor(**params)


register("dash_pretrained", _build)
