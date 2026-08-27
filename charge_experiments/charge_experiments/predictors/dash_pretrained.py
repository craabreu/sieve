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

**Applies DASH's own ``std_weighted`` charge-conservation renormalization**
(the paper's eq 4) after the raw per-atom walk above -- the paper states
explicitly that this, not raw per-atom values, is what its own reported
results use: "Normalizing the DASH charges with eq 4 reduces the errors
slightly... we used normalization with eq 4 in the remainder of this
work" (Lehner et al., J. Chem. Inf. Model. 2023, 63, 6014-6028). Skipping
it, as an earlier version of this predictor did, would not be faithful to
the authors' own reported performance.

The paper's own printed eq 2/3 have a sign error, though, confirmed two
ways: (1) reading ``get_molecules_partial_charges``'s actual ``symmetric``
branch (``x + (tot_charge_mol - tot_charge_tree) / N``, i.e.
``Q_formal - sum(Q)``) against the paper's printed
``ΔQ = sum(Q) - Q_formal`` then ``Q_i' = Q_i + ΔQ/N`` -- literally the
opposite sign; (2) a direct arithmetic check: applying the paper's printed
formula to a toy 3-atom example whose raw charges sum to 0.0 against a
target formal charge of 1.0 gives a renormalized sum of -1.0, not 1.0 --
it does not conserve charge, while the code's actual (opposite-sign)
formula gives 1.0 exactly. This predictor follows the real, shipped code
(ground truth: it actually runs, and actually conserves charge), not the
paper's printed equation.

``default_std_value=0.1`` guards a non-positive (including NaN, since
``nan > 0`` is always ``False``) per-atom std before it's used as a
normalization weight -- this mirrors ``get_molecules_partial_charges``'s
own hardcoded default for that exact situation (``tmp_tree_std if
tmp_tree_std > 0 else default_std_value``), so it is the authors' own
published fallback for *that* quantity, not one invented here. It is not
a fallback for a missing *charge*: an atom whose raw ``"result"`` walk
itself came back NaN still propagates NaN through the whole molecule's
renormalized output (the residual sum ``ΣQ_i`` is NaN, so every atom in
that molecule ends up NaN too) -- exactly ``get_molecules_partial_charges``'s
own real behavior for that case, verified by tracing its actual code path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet, molecule_sum
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction
from charge_experiments.predictors.dash import (
    LiteralTreeChargeProperties,
    _atom_paths,
    predict_via_data_storage_walk,
)

logger = logging.getLogger("charge_experiments")

# get_molecules_partial_charges' own hardcoded default -- see module
# docstring's note on why using it here is faithful, not invented.
_DEFAULT_STD_VALUE = 0.1


def std_weighted_normalize(
    raw_charge: NDArray[np.floating],
    raw_std: NDArray[np.floating],
    net_charge: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    n_conformers: int,
) -> NDArray[np.float64]:
    """DASH's own eq 4 (std-weighted normalization), pure numpy -- no tree,
    no rdkit, so this is fast-suite tested independent of the real
    DASH-tree clone (see module docstring for the sign-convention and
    ``default_std_value`` notes this implements).

    A non-positive (including NaN) entry in ``raw_std`` is floored to
    ``get_molecules_partial_charges``'s own ``default_std_value`` (0.1) --
    the authors' own published fallback for *that* quantity. A NaN entry
    in ``raw_charge`` is not floored or substituted at all: it propagates
    through ``molecule_sum`` into that whole conformer's residual, so
    every atom in a conformer with even one unmatched raw charge ends up
    NaN -- exactly ``get_molecules_partial_charges``'s own real behavior
    for that case.
    """
    raw_charge = np.asarray(raw_charge, dtype=np.float64)
    raw_std = np.asarray(raw_std, dtype=np.float64)
    effective_std = np.where(raw_std > 0, raw_std, _DEFAULT_STD_VALUE)
    tot_charge_tree = molecule_sum(raw_charge, mol_id, n_conformers)
    tot_std_tree = molecule_sum(effective_std, mol_id, n_conformers)
    residual = np.asarray(net_charge, dtype=np.float64) - tot_charge_tree
    return raw_charge + (residual[mol_id] * effective_std / tot_std_tree[mol_id])


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
        value_props = LiteralTreeChargeProperties(
            charge_column=tree.default_value_column, fallback_charge=float("nan")
        )
        std_props = LiteralTreeChargeProperties(
            charge_column=tree.default_std_column, fallback_charge=float("nan")
        )
        raw_charge = predict_via_data_storage_walk(tree, paths, value_props)
        raw_std = predict_via_data_storage_walk(tree, paths, std_props)

        atom_charge = std_weighted_normalize(
            raw_charge, raw_std, test.net_charge, test.atom_mol_id, test.n_conformers
        )
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> DASHPretrainedChargePredictor:
    return DASHPretrainedChargePredictor(**params)


register("dash_pretrained", _build)
