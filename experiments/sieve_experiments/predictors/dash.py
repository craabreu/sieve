"""DASH Stage A: the published DASH-tree topology, our own per-node
statistics fit on the training split, predicted by support-based back-off --
the same algorithm Sieve itself uses, applied to someone else's hierarchy
(design.md's "DASH" decision).

Two layers, deliberately split so the algorithm is testable without either
optional dependency:

- ``fit_backoff``/``predict_backoff`` -- pure numpy over pre-computed tree
  paths. No rdkit, no DASH-tree clone. Fast-suite tested
  (tests/test_experiment_predictor_dash.py).
- ``DASHBackoffPredictor`` -- wires those onto real atoms: RDKit for the
  atom-index mapping (same convention as
  ``src/sieve/io/cosmolayer_adapter.py``) and DASHTree.match_new_atom for the
  tree path. Needs the DASH-tree clone (pins.toml's ``[dash_tree]``) and
  atom-level truth loaded straight from the store (``data.load_atom_truth``,
  since ``load_molecule_set`` never populates it). Optional-data tested only.

A path is a list of ``(branch_idx, node_id)`` pairs, root (shallowest) first,
deepest last -- the converted form of DASHTree.match_new_atom's raw
``[branch_idx, 0, node_id_1, node_id_2, ...]`` return value.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sieve_experiments.data import REPO_ROOT, MoleculeSet, load_atom_truth
from sieve_experiments.predictors import register
from sieve_experiments.predictors.base import AtomPrediction, AtomPredictor

logger = logging.getLogger("sieve_experiments")

PathKey = tuple[int, int]
NodePath = list[PathKey]

# pins.toml's [dash_tree]: a plain git clone, not pip-installed (the default
# tree data may not travel with a wheel -- see pins.toml's notes). Adding it
# to sys.path here is harmless even when the clone is absent; only actually
# importing `serenityff` (done lazily, in _load_tree) requires it to exist.
_DASH_TREE_ROOT = REPO_ROOT / "experiments" / "external" / "DASH-tree"
if _DASH_TREE_ROOT.exists() and str(_DASH_TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASH_TREE_ROOT))


@dataclass(frozen=True)
class NodeStat:
    count: int
    profile: NDArray[np.float64]
    area: float
    charge: float
    charge_std: float


@dataclass(frozen=True)
class BackoffStats:
    nodes: dict[PathKey, NodeStat]
    fallback: NodeStat


def fit_backoff(
    paths: list[NodePath],
    atom_profile: NDArray[np.floating],
    atom_area: NDArray[np.floating],
    atom_charge: NDArray[np.floating],
    *,
    minimum_support: int,
) -> BackoffStats:
    """Accumulate count/mean per ``(branch_idx, node_id)`` over every node on
    every atom's path (not just the leaf), pruning nodes with fewer than
    ``minimum_support`` supporting atoms. ``fallback`` is the unconditional
    mean over all training atoms -- always available, used when an atom's
    path retains no node at all (a fresh branch, or every node pruned)."""
    n = len(paths)
    if not (len(atom_profile) == len(atom_area) == len(atom_charge) == n):
        raise ValueError(
            "paths, atom_profile, atom_area and atom_charge must have the "
            "same length (one entry per atom)"
        )
    atom_profile = np.asarray(atom_profile, dtype=np.float64)
    atom_area = np.asarray(atom_area, dtype=np.float64)
    atom_charge = np.asarray(atom_charge, dtype=np.float64)

    profile_sum: dict[PathKey, NDArray[np.float64]] = {}
    area_sum: dict[PathKey, float] = {}
    charge_sum: dict[PathKey, float] = {}
    charge_sq_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}

    for path, profile, area, charge in zip(
        paths, atom_profile, atom_area, atom_charge, strict=True
    ):
        for key in path:
            if key not in count:
                count[key] = 0
                profile_sum[key] = np.zeros_like(profile)
                area_sum[key] = 0.0
                charge_sum[key] = 0.0
                charge_sq_sum[key] = 0.0
            count[key] += 1
            profile_sum[key] += profile
            area_sum[key] += area
            charge_sum[key] += charge
            charge_sq_sum[key] += charge * charge

    nodes: dict[PathKey, NodeStat] = {}
    for key, c in count.items():
        if c < minimum_support:
            continue
        mean_charge = charge_sum[key] / c
        variance = max(charge_sq_sum[key] / c - mean_charge * mean_charge, 0.0)
        nodes[key] = NodeStat(
            count=c,
            profile=profile_sum[key] / c,
            area=area_sum[key] / c,
            charge=mean_charge,
            charge_std=max(np.sqrt(variance), 1e-6),
        )

    global_mean_charge = float(np.mean(atom_charge))
    fallback = NodeStat(
        count=n,
        profile=atom_profile.mean(axis=0),
        area=float(np.mean(atom_area)),
        charge=global_mean_charge,
        charge_std=max(float(np.std(atom_charge)), 1e-6),
    )
    return BackoffStats(nodes=nodes, fallback=fallback)


def predict_backoff(paths: list[NodePath], stats: BackoffStats) -> AtomPrediction:
    """Walk each atom's path deepest -> shallowest, use the first retained
    node's stats, else ``stats.fallback``."""
    n = len(paths)
    profile_dim = stats.fallback.profile.shape[0]
    atom_profile = np.empty((n, profile_dim), dtype=np.float64)
    atom_area = np.empty(n, dtype=np.float64)
    atom_charge = np.empty(n, dtype=np.float64)
    atom_charge_std = np.empty(n, dtype=np.float64)

    for i, path in enumerate(paths):
        chosen = stats.fallback
        for key in reversed(path):
            if key in stats.nodes:
                chosen = stats.nodes[key]
                break
        atom_profile[i] = chosen.profile
        atom_area[i] = chosen.area
        atom_charge[i] = chosen.charge
        atom_charge_std[i] = chosen.charge_std

    return AtomPrediction(
        atom_profile=atom_profile,
        atom_area=atom_area,
        atom_charge=atom_charge,
        atom_charge_std=atom_charge_std,
    )


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
    """RDKit + DASHTree.match_new_atom for every atom in ``mset``, in
    ``mset``'s own atom order, plus a count of what could not be matched.

    Atom order mirrors ``src/sieve/io/cosmolayer_adapter.py``'s
    atom-mapped-SMILES convention: atom-mapped SMILES are numbered in COSMO
    file order, so the RDKit index matching flat position ``j`` is
    ``order[j]`` where ``order`` sorts atoms by their map number.
    ``tests/test_experiment_predictor_dash_optional.py`` checks that mapping
    against the store's own element column, the way the adapter's
    ``check_alignment`` does -- a transposed mapping still yields finite
    metrics, so it needs a real guard.

    The neighbor dict is built **once per molecule** and passed into every
    ``match_new_atom`` call for that molecule. ``match_new_atom`` otherwise
    rebuilds it from the whole molecule on every call, which is
    O(n_atoms^2) per molecule (measured ~8.5x slower on chaos-store);
    DASH's own ``_get_allAtoms_nodePaths`` hoists it for the same reason.

    Two failure modes are tolerated and **counted** (never silent) -- an
    unmatched atom just falls back to the global mean in
    ``predict_backoff``, which is the same degradation DASH's own
    ``get_molecules_partial_charges`` accepts, but a baseline that quietly
    predicted the global mean for part of the test set would be impossible
    to interpret:

    - the whole molecule, when ``init_neighbor_dict`` raises. It runs over
      every atom, so one atom whose feature tuple is outside DASH's
      published vocabulary (boron, Si, Ge, Sb, Te all appear in
      chaos-store) takes the entire molecule down with it. This is the
      dominant mode: ~4% of chaos-store molecules.
    - a single atom, when ``match_new_atom`` itself raises.
    """
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False

    paths: list[NodePath] = []
    n_unmatched_atoms = 0
    n_unmatched_molecules = 0

    for smi, n_atoms in zip(mset.smiles, mset.num_atoms, strict=True):
        mol = Chem.MolFromSmiles(smi, params)
        if mol is None:
            raise ValueError(f"unparseable SMILES: {smi[:60]}")
        order = np.argsort([a.GetAtomMapNum() for a in mol.GetAtoms()])
        if len(order) != n_atoms:
            raise ValueError(
                f"atom count mismatch for {smi[:60]}: rdkit parsed "
                f"{len(order)}, expected {n_atoms}"
            )

        try:
            neighbor_dict = neighbor_dict_factory(mol, tree.atom_feature_type)
        except Exception:
            # One out-of-vocabulary atom kills the whole molecule; still
            # emit one (empty) path per atom so the rollup stays aligned.
            paths.extend([] for _ in range(int(n_atoms)))
            n_unmatched_molecules += 1
            n_unmatched_atoms += int(n_atoms)
            continue

        for j in range(n_atoms):
            rdkit_idx = int(order[j])
            try:
                raw = tree.match_new_atom(
                    rdkit_idx,
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
        "n_molecules": mset.n_molecules,
        "n_unmatched_atoms": n_unmatched_atoms,
        "n_unmatched_molecules": n_unmatched_molecules,
    }
    return paths, stats


class DASHBackoffPredictor(AtomPredictor):
    """Stage A DASH baseline: DASH-tree's published topology, our own
    per-node statistics, support-based back-off at predict time.

    ``store``/``scheme`` are needed here (not just in the run config's
    ``data`` section) because atom-level truth for the training split has to
    be loaded independently -- ``load_molecule_set`` only ever supplies
    molecule-level truth (see data.py's module docstring).

    ``preload`` defaults to True (loads the ~300MB default tree fully into
    memory, ~10s one-time cost): the pinned DASH-tree commit's on-demand
    loading (``preload=False``) has an ordering bug that raises on every H
    atom (``_get_init_layer`` needs ``tree_storage[branch_idx]`` for the
    H-connected heavy atom's expansion before ``match_new_atom``'s own
    lazy-load runs).

    ``match_stats`` holds ``_atom_paths``' unmatched counts from the last
    ``fit_atoms``/``predict_atoms`` call. The runner copies them into the run
    manifest, so how much of a split fell back to the global mean is always
    on the record -- see ``_atom_paths`` for the two failure modes.
    """

    name = "dash_backoff"

    def __init__(
        self,
        *,
        store: str,
        scheme: str,
        max_depth: int = 16,
        attention_threshold: float = 10,
        minimum_support: int = 5,
        charge_reconciliation: str = "std_weighted",
        stores_root: str | None = None,
        tree_folder_path: str | None = None,
        preload: bool = True,
    ) -> None:
        self.store = store
        self.scheme = scheme
        self.max_depth = max_depth
        self.attention_threshold = attention_threshold
        self.minimum_support = minimum_support
        self.charge_reconciliation = charge_reconciliation
        self.stores_root = stores_root
        self.tree_folder_path = tree_folder_path
        self.preload = preload
        self.match_stats: dict[str, dict[str, int]] = {}
        self._tree: Any = None
        self._stats: BackoffStats | None = None

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
                "DASH could not match %d/%d %s atoms (%d/%d molecules "
                "rejected outright); these fall back to the global mean",
                stats["n_unmatched_atoms"],
                stats["n_atoms"],
                split,
                stats["n_unmatched_molecules"],
                stats["n_molecules"],
            )
        return paths

    def fit_atoms(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng  # nothing to tune here; minimum_support is a fixed config value
        kwargs: dict[str, Any] = {}
        if self.stores_root is not None:
            kwargs["stores_root"] = Path(self.stores_root)
        atom_profile, atom_area, atom_charge = load_atom_truth(
            self.store,
            scheme=self.scheme,
            smiles=train.smiles,
            num_atoms=train.num_atoms,
            **kwargs,
        )
        paths = self._paths_for(train, split="train")
        self._stats = fit_backoff(
            paths,
            atom_profile,
            atom_area,
            atom_charge,
            minimum_support=self.minimum_support,
        )

    def predict_atoms(self, test: MoleculeSet) -> AtomPrediction:
        if self._stats is None:
            raise RuntimeError("fit_atoms must be called before predict_atoms")
        paths = self._paths_for(test, split="test")
        return predict_backoff(paths, self._stats)


def _build(params: Mapping[str, Any]) -> DASHBackoffPredictor:
    return DASHBackoffPredictor(**params)


register("dash_backoff", _build)
