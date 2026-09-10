"""DASH-tree charge predictor: DASH-tree's published topology
(``DASHTree.match_new_atom``, unmodified) with a back-off step reproducing
``DASHTree.get_property_noNAN``'s own missing-value fallback (deepest ->
shallowest, first populated node wins, else the global mean) -- ported from
cosmo_experiments/sieve_experiments/predictors/dash.py, adapted to a
**scalar** target (this series' own ``MBIScharge``, not a 51-bin profile)
and to this series' Mol-blob store (no atom-map-order/SMILES bookkeeping:
``MoleculeSet.mols`` are already-parsed ``Mol`` objects in their own atom
order, so tree-matching iterates them directly).

Two layers, deliberately split so the algorithm is testable without either
optional dependency (see experiments/tests/test_predictor_dash.py
for the pure-logic layer; the real-tree/real-rdkit layer is
_optional-tested only, in test_predictor_dash_optional.py):

- ``populate_tree_with_charge_property``/``predict_via_data_storage_walk``
  -- pure numpy + pandas over pre-computed tree paths and an already-loaded
  ``DASHTree``'s own storage.
- ``DASHChargePredictor`` -- wires those onto real atoms: rdkit for
  iterating each conformer's own atoms and ``DASHTree.match_new_atom`` for
  the tree path.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from experiments.data import REPO_ROOT, MoleculeSet
from experiments.predictors import register
from experiments.predictors.base import Prediction, RawPrediction
from experiments.tree_artifact import (
    LiteralTreeChargeProperties,
    TreeNodeStats,
    apply_node_stats,
    compute_node_stats,
    load_node_stats,
    save_node_stats,
)

logger = logging.getLogger("experiments")

PathKey = tuple[int, int]
NodePath = list[PathKey]

# pins.toml's [dash_tree]: a plain git clone, cloned independently of
# cosmo_experiments' own copy (see Task 9) -- see that pins.toml entry for
# why (no shared harness code between series).
_DASH_TREE_ROOT = REPO_ROOT / "experiments" / "external" / "DASH-tree"
if _DASH_TREE_ROOT.exists() and str(_DASH_TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASH_TREE_ROOT))


def populate_tree_with_charge_property(
    tree: Any, paths: list[NodePath], atom_value: NDArray[np.floating]
) -> LiteralTreeChargeProperties:
    """Populate an already-loaded ``DASHTree``'s own storage with our own
    per-node mean/std ``MBIScharge`` over every node on every atom's path --
    a thin wrapper over tree_artifact's own ``compute_node_stats``/
    ``apply_node_stats``, kept for its existing callers/tests. Returns only
    the mean props (the std props are also written onto
    ``tree.data_storage``, just not returned here -- ``DASHChargePredictor.
    fit()`` below calls ``compute_node_stats``/``apply_node_stats`` directly
    instead, to keep both)."""
    stats = compute_node_stats(paths, atom_value)
    mean_props, _std_props = apply_node_stats(tree, stats)
    return mean_props


def predict_via_data_storage_walk(
    tree: Any, paths: list[NodePath], props: LiteralTreeChargeProperties
) -> NDArray[np.float64]:
    """Predict by walking each atom's matched path deepest -> shallowest
    directly against ``tree.data_storage`` and using the first node whose
    row is populated -- the same fallback ``DASHTree.get_property_noNAN``
    itself implements."""
    n = len(paths)
    predicted = np.empty(n, dtype=np.float64)

    arrays: dict[int, NDArray[np.float64]] = {}
    for branch_idx in {path[0][0] for path in paths if path}:
        branch_df = tree.data_storage[branch_idx]
        if props.charge_column in branch_df.columns:
            arrays[branch_idx] = branch_df[props.charge_column].to_numpy(
                dtype=np.float64
            )

    for i, path in enumerate(paths):
        value = None
        arr = arrays.get(path[0][0]) if path else None
        if arr is not None:
            for _, node_id in reversed(path):
                candidate = arr[node_id]
                if not np.isnan(candidate):
                    value = candidate
                    break
        predicted[i] = props.fallback_charge if value is None else value

    return predicted


def predict_raw_from_paths(
    tree: Any,
    paths: list[NodePath],
    mean_props: LiteralTreeChargeProperties,
    std_props: LiteralTreeChargeProperties,
    *,
    max_depth: int,
) -> RawPrediction:
    """Derive a ``max_depth``-capped prediction from ``paths`` already
    walked at some depth >= ``max_depth``, by truncating each atom's own
    path to its first ``max_depth`` entries before backoff.

    Equivalent to re-walking ``DASHTree.match_new_atom`` with
    ``max_depth=max_depth`` from scratch, but far cheaper: the walk
    (``match_new_atom``, this module's real cost) is a strict-prefix
    relationship in depth -- an atom's depth-*k* path is exactly the first
    *k* entries of any deeper walk for that same atom, since matching
    stops early rather than taking a different route. Node stats are
    unaffected by which depth ``fit()`` used either
    (``tree_artifact.compute_node_stats`` accumulates over every node on
    every atom's own path, not just its deepest one), so one fit at the
    deepest depth needed already covers every shallower depth's own node
    statistics too. Together this means ``fit`` + one walk per split, done
    once at the sweep's own maximum depth, is enough to derive every
    shallower depth's prediction -- see ``experiments.dash_depth_sweep``,
    which is what actually exploits this rather than the ordinary
    one-depth-per-run ``fit``/``predict_raw``.

    A standalone function, not a ``DASHChargePredictor`` method: it needs
    no tree-matching of its own, so it is testable with the same
    ``_FakeTree`` pattern the rest of this module's pure-logic tests use.
    """
    truncated = [path[:max_depth] for path in paths]
    atom_value = predict_via_data_storage_walk(tree, truncated, mean_props)
    atom_std = predict_via_data_storage_walk(tree, truncated, std_props)
    return RawPrediction(atom_value=atom_value, atom_std=atom_std)


def _default_neighbor_dict_factory(mol: Any, af: Any) -> Any:
    from serenityff.charge.tree.dash_tools import init_neighbor_dict

    return init_neighbor_dict(mol, af=af)


def _atom_paths(
    mset: MoleculeSet,
    tree: Any,
    *,
    max_depth: int,
    attention_threshold: float,
    neighbor_dict_factory: Any = _default_neighbor_dict_factory,
) -> tuple[list[NodePath], dict[str, int]]:
    """``DASHTree.match_new_atom`` for every atom in ``mset``, in each
    conformer's own atom order (no atom-map-order decoding needed here --
    ``mset.mols`` are already-parsed ``Mol`` objects in their canonical
    order). Two failure modes are tolerated and counted, mirroring
    cosmo_experiments' own ``_atom_paths``: the whole molecule, when
    ``init_neighbor_dict`` raises (one out-of-vocabulary atom feature tuple
    takes the whole molecule down); a single atom, when ``match_new_atom``
    itself raises."""
    paths: list[NodePath] = []
    n_unmatched_atoms = 0
    n_unmatched_molecules = 0

    for mol in mset.mols:
        n_atoms = mol.GetNumAtoms()
        try:
            neighbor_dict = neighbor_dict_factory(mol, tree.atom_feature_type)
        except Exception:
            paths.extend([] for _ in range(n_atoms))
            n_unmatched_molecules += 1
            n_unmatched_atoms += n_atoms
            continue

        for j in range(n_atoms):
            try:
                raw = tree.match_new_atom(
                    j,
                    mol,
                    max_depth=max_depth,
                    attention_threshold=attention_threshold,
                    neighbor_dict=neighbor_dict,
                )
                path = [(raw[0], node_id) for node_id in raw[1:]]
            except Exception:
                path = []
                n_unmatched_atoms += 1
            paths.append(path)

    stats = {
        "n_atoms": len(paths),
        "n_conformers": mset.n_conformers,
        "n_unmatched_atoms": n_unmatched_atoms,
        "n_unmatched_molecules": n_unmatched_molecules,
    }
    return paths, stats


class DASHChargePredictor:
    """DASH-tree charge baseline: published topology + our own per-node
    MBIScharge mean and missing-value back-off. See module docstring.

    ``preload`` defaults to True (see cosmo_experiments/pins.toml's GOTCHA 1
    -- on-demand loading has an ordering bug that raises on every H atom at
    the pinned commit).
    """

    name: ClassVar[str] = "dash"

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
        self.match_stats: dict[str, dict[str, int]] = {}
        self._tree: Any = None
        self._stats: TreeNodeStats | None = None
        self._mean_props: LiteralTreeChargeProperties | None = None
        self._std_props: LiteralTreeChargeProperties | None = None

    def _load_tree(self) -> Any:
        if self._tree is None:
            from serenityff.charge.tree.dash_tree import DASHTree

            kwargs: dict[str, Any] = {"preload": self.preload, "verbose": False}
            if self.tree_folder_path is not None:
                kwargs["tree_folder_path"] = self.tree_folder_path
            self._tree = DASHTree(**kwargs)
        return self._tree

    def match_paths(self, mset: MoleculeSet, *, split: str) -> list[NodePath]:
        """``DASHTree.match_new_atom`` for every atom in ``mset``, walked to
        ``self.max_depth`` -- the expensive step (real tree traversal), and
        the one worth caching across a depth sweep: a path walked at
        ``self.max_depth`` already contains every shallower depth's own
        path as a prefix (see ``predict_raw_from_paths``), so a caller
        sweeping multiple depths should walk once at the deepest value
        needed and reuse the result, not call this once per depth."""
        tree = self._load_tree()
        paths, stats = _atom_paths(
            mset,
            tree,
            max_depth=self.max_depth,
            attention_threshold=self.attention_threshold,
        )
        self.match_stats[split] = stats
        if stats["n_unmatched_atoms"]:
            logger.warning(
                "DASH could not match %d/%d %s atoms "
                "(%d/%d conformers rejected outright)",
                stats["n_unmatched_atoms"],
                stats["n_atoms"],
                split,
                stats["n_unmatched_molecules"],
                stats["n_conformers"],
            )
        return paths

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        paths = self.match_paths(train, split="train")
        self._stats = compute_node_stats(paths, train.atom_target)
        self._mean_props, self._std_props = apply_node_stats(self._tree, self._stats)

    def predict_raw_at_depth(
        self, paths: list[NodePath], *, max_depth: int
    ) -> RawPrediction:
        """``predict_raw_from_paths`` bound to this predictor's own fitted
        tree/props -- the public seam a depth-sweep caller uses to derive
        a shallower-than-``self.max_depth`` prediction from paths already
        walked once (via ``match_paths``), without reaching into this
        instance's own ``_tree``/``_mean_props``/``_std_props``. See
        ``predict_raw_from_paths``'s own docstring for why truncation is
        exactly equivalent to a fresh, shallower walk."""
        if self._mean_props is None or self._std_props is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_raw_at_depth"
            )
        return predict_raw_from_paths(
            self._tree, paths, self._mean_props, self._std_props, max_depth=max_depth
        )

    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        if self._mean_props is None or self._std_props is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_raw"
            )
        paths = self.match_paths(test, split="test")
        atom_value = predict_via_data_storage_walk(self._tree, paths, self._mean_props)
        atom_std = predict_via_data_storage_walk(self._tree, paths, self._std_props)

        # match_stats' own n_unmatched_atoms (set in match_paths) only counts
        # atoms whose path-matching itself failed -- a strict undercount of
        # the real NaN rate now that there is no invented fallback: an
        # atom's own path can match successfully yet still return NaN if
        # nothing along that hierarchy was ever populated (the exact case
        # DASH's own get_property_noNAN returns NaN for). Tracked here so
        # match_stats -- the one place a run's manifest.json reports
        # coverage -- reflects the true final NaN rate, mirroring
        # predictors/dash_pretrained.py's own convention.
        n_final_nan_atoms = int(np.isnan(atom_value).sum())
        stats = self.match_stats["test"]
        stats["n_final_nan_atoms"] = n_final_nan_atoms
        if n_final_nan_atoms:
            logger.warning(
                "DASH predicted NaN for %d/%d atoms (%d unmatched by path, "
                "%d matched but with nothing populated along the "
                "hierarchy); these are reported as NaN, not guessed at",
                n_final_nan_atoms,
                stats["n_atoms"],
                stats["n_unmatched_atoms"],
                n_final_nan_atoms - stats["n_unmatched_atoms"],
            )
        return RawPrediction(atom_value=atom_value, atom_std=atom_std)

    def predict(self, test: MoleculeSet) -> Prediction:
        return Prediction(atom_value=self.predict_raw(test).atom_value)

    def save_model_state(self, path: str | Path) -> None:
        """See ``NormalizablePredictor``'s own docstring
        (predictors/base.py) for why this method name is predictor-agnostic
        rather than ``save_tree_stats``."""
        if self._stats is None:
            raise RuntimeError("fit must be called before save_model_state")
        save_node_stats(self._stats, path)

    def load_model_state(self, path: str | Path) -> None:
        """Loads the tree (fast -- reads the pinned clone's own data files,
        no atom matching) and applies a previously-saved stats artifact,
        skipping fit()'s own expensive match_new_atom walk over train
        entirely."""
        self._load_tree()
        self._stats = load_node_stats(path)
        self._mean_props, self._std_props = apply_node_stats(self._tree, self._stats)


def _build(params: Mapping[str, Any]) -> DASHChargePredictor:
    return DASHChargePredictor(**params)


register("dash", _build)
