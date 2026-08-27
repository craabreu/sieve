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
optional dependency (see charges_experiments/tests/test_charge_predictor_dash.py
for the pure-logic layer; the real-tree/real-rdkit layer is
_optional-tested only, in test_charge_predictor_dash_optional.py):

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
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction

logger = logging.getLogger("charge_experiments")

PathKey = tuple[int, int]
NodePath = list[PathKey]

# pins.toml's [dash_tree]: a plain git clone, cloned independently of
# cosmo_experiments' own copy (see Task 9) -- see that pins.toml entry for
# why (no shared harness code between series).
_DASH_TREE_ROOT = REPO_ROOT / "charges_experiments" / "external" / "DASH-tree"
if _DASH_TREE_ROOT.exists() and str(_DASH_TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASH_TREE_ROOT))


@dataclass(frozen=True)
class LiteralTreeChargeProperties:
    """What ``populate_tree_with_charge_property`` writes onto a
    ``DASHTree``'s own ``data_storage``, and what
    ``predict_via_data_storage_walk`` needs to read it back."""

    charge_column: str
    fallback_charge: float


def populate_tree_with_charge_property(
    tree: Any, paths: list[NodePath], atom_charge: NDArray[np.floating]
) -> LiteralTreeChargeProperties:
    """Populate an already-loaded ``DASHTree``'s own storage with our own
    per-node mean ``MBIScharge`` over every node on every atom's path. A
    node with zero matching atoms gets no entry and stays ``NaN`` --
    exactly DASH's own ``get_property_noNAN`` missing-value semantics."""
    n = len(paths)
    if len(atom_charge) != n:
        raise ValueError("paths and atom_charge must have the same length")
    atom_charge = np.asarray(atom_charge, dtype=np.float64)
    charge_column = "dash_charge_mean"

    charge_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}
    for path, charge in zip(paths, atom_charge, strict=True):
        for key in path:
            charge_sum[key] = charge_sum.get(key, 0.0) + float(charge)
            count[key] = count.get(key, 0) + 1

    by_branch: dict[int, list[int]] = {}
    for branch_idx, node_id in count:
        by_branch.setdefault(branch_idx, []).append(node_id)

    for branch_idx, node_ids in by_branch.items():
        df = tree.data_storage[branch_idx]
        n_rows = len(df)
        node_ids_arr = np.array(node_ids, dtype=np.int64)
        means = np.array(
            [
                charge_sum[(branch_idx, nid)] / count[(branch_idx, nid)]
                for nid in node_ids
            ]
        )
        values = np.full(n_rows, np.nan)
        values[node_ids_arr] = means
        df[charge_column] = values

    return LiteralTreeChargeProperties(
        charge_column=charge_column, fallback_charge=float(atom_charge.mean())
    )


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
        self._props: LiteralTreeChargeProperties | None = None

    def _load_tree(self) -> Any:
        if self._tree is None:
            from serenityff.charge.tree.dash_tree import DASHTree

            kwargs: dict[str, Any] = {"preload": self.preload, "verbose": False}
            if self.tree_folder_path is not None:
                kwargs["tree_folder_path"] = self.tree_folder_path
            self._tree = DASHTree(**kwargs)
        return self._tree

    def _paths_for(self, mset: MoleculeSet, *, split: str) -> list[NodePath]:
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
        paths = self._paths_for(train, split="train")
        self._props = populate_tree_with_charge_property(
            self._tree, paths, train.atom_charge
        )

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._props is None:
            raise RuntimeError("fit must be called before predict")
        paths = self._paths_for(test, split="test")
        atom_charge = predict_via_data_storage_walk(self._tree, paths, self._props)
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> DASHChargePredictor:
    return DASHChargePredictor(**params)


register("dash", _build)
