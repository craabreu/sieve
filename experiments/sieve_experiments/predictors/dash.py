"""DASH Stage A: DASH-tree's published topology (``DASHTree.match_new_atom``,
unmodified) with a back-off step that reproduces DASH's own real
prediction-time fallback -- ``DASHTree.get_property_noNAN``'s deepest ->
shallowest walk, first populated node wins, else the global mean over all
training atoms. No support threshold: DASH's own code has none
(``Node.prune()``, the one piece of its code that would implement a count
threshold, is dead -- confirmed via grep, no caller anywhere in the
DASH-tree repo except its own commented-out recursive self-call).

Populated with our own sigma-profile data, since DASH's own tree ships only
MBIS partial charges: there is no DASH function that populates an *existing*
tree with a new property -- DASH-properties' own follow-up paper gets new
atomic properties onto the published topology by re-matching every atom and
averaging, the same shape this module's own accumulation takes. The
property is the plain, raw (undecomposed) profile -- the plainest possible
choice, and the only one DASH's own real algorithm would recognize; a
shape/location/area decomposition was tried and measured to cost accuracy
relative to this simpler design (full comparison and the two retired
back-off variants: ``experiments/docs/dash.md``).

Two layers, deliberately split so the algorithm is testable without either
optional dependency:

- ``populate_tree_with_sigma_properties``/``predict_via_data_storage_walk``
  -- pure numpy + pandas over pre-computed tree paths and an already-loaded
  ``DASHTree``'s own storage. No rdkit, no live DASH-tree matching needed for
  their own unit tests (a fake tree-like object stands in). Fast-suite
  tested (experiments/tests/test_experiment_predictor_dash.py).
- ``DASHPredictor`` -- wires those onto real atoms: RDKit for the atom-index
  mapping (same convention as ``src/sieve/io/cosmolayer_adapter.py``) and
  ``DASHTree.match_new_atom`` for the tree path. Needs the DASH-tree clone
  (pins.toml's ``[dash_tree]``) and atom-level truth loaded straight from the
  store (``data.load_atom_truth``, since ``load_molecule_set`` never
  populates it). Optional-data tested only.

A path is a list of ``(branch_idx, node_id)`` pairs, root (shallowest) first,
deepest last -- the converted form of ``DASHTree.match_new_atom``'s raw
``[branch_idx, 0, node_id_1, node_id_2, ...]`` return value.
``predict_via_data_storage_walk`` converts back to that raw format itself
where needed (see its own docstring).

``predict_via_data_storage_walk`` reimplements ``get_property_noNAN``'s own
walk rather than calling it directly, purely for speed (calling it live is
~130x slower at full-store scale) -- verified bit-for-bit equivalent to a
literal-call prototype's own output before adopting it. Full performance
story and the two real bugs found building it: ``experiments/docs/dash.md``.
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
class LiteralTreeProperties:
    """What ``populate_tree_with_sigma_properties`` writes onto a
    ``DASHTree``'s own ``data_storage``, and what
    ``predict_via_data_storage_walk`` needs to read it back: the profile
    bin column names (in bin order) and the charge_std column name, plus a
    global fallback for an atom no node on its path was ever populated for
    (including an atom ``_atom_paths`` never matched at all, ``path ==
    []``). ``sigma_values`` is carried through so area/charge can be
    derived from the predicted profile, the same convention every other
    "raw"-style predictor in this project uses.
    """

    profile_columns: list[str]
    charge_std_column: str
    fallback_profile: NDArray[np.float64]
    fallback_charge_std: float
    sigma_values: NDArray[np.float64]


def populate_tree_with_sigma_properties(
    tree: Any,
    paths: list[NodePath],
    atom_profile: NDArray[np.floating],
    *,
    sigma_values: NDArray[np.floating],
) -> LiteralTreeProperties:
    """Populate an already-loaded ``DASHTree``'s own storage with our own
    sigma-profile statistics: the plain (raw, undecomposed) mean profile per
    node over every node on every atom's path. ``charge`` is derived from
    the profile (``atom_profile @ sigma_values``) purely so its per-node std
    can be computed for reconciliation weighting -- charge itself is never
    predicted as its own quantity.

    Writes each new property as a plain new column directly onto
    ``tree.data_storage[branch_idx]`` (an ordinary, freely-mutable pandas
    ``DataFrame``), at row position == node id -- verified true for every
    node the topology defines, for every loaded branch. A node with zero
    matching atoms simply gets no entry and stays ``NaN`` by construction --
    exactly DASH's own ``get_property_noNAN``'s missing-value semantics.
    There is no ``minimum_support`` concept here at all: DASH's own code has
    none (``Node.prune()``, the one piece of code that would implement a
    count threshold, is dead -- confirmed never called from the real
    tree-building pipeline).
    """
    n = len(paths)
    if len(atom_profile) != n:
        raise ValueError("paths and atom_profile must have the same length")
    atom_profile = np.asarray(atom_profile, dtype=np.float64)
    sigma_values = np.asarray(sigma_values, dtype=np.float64)
    if atom_profile.shape[1] != len(sigma_values):
        raise ValueError(
            f"atom_profile has {atom_profile.shape[1]} bins but sigma_values "
            f"has {len(sigma_values)}"
        )
    n_bins = atom_profile.shape[1]
    profile_columns = [f"sigma_bin_{i}" for i in range(n_bins)]
    charge_std_column = "sigma_charge_std"

    # Same integral load_atom_truth's own charge column already is -- see
    # data.py::load_atom_truth. Only used here to compute a per-node std for
    # reconciliation; never predicted as its own quantity.
    atom_charge = atom_profile @ sigma_values

    profile_sum: dict[PathKey, NDArray[np.float64]] = {}
    charge_sum: dict[PathKey, float] = {}
    charge_sq_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}

    for path, profile, charge in zip(paths, atom_profile, atom_charge, strict=True):
        for key in path:
            if key not in count:
                count[key] = 0
                profile_sum[key] = np.zeros(n_bins)
                charge_sum[key] = 0.0
                charge_sq_sum[key] = 0.0
            count[key] += 1
            profile_sum[key] += profile
            charge_sum[key] += charge
            charge_sq_sum[key] += charge * charge

    by_branch: dict[int, list[int]] = {}
    for branch_idx, node_id in count:
        by_branch.setdefault(branch_idx, []).append(node_id)

    for branch_idx, node_ids in by_branch.items():
        df = tree.data_storage[branch_idx]
        n_rows = len(df)
        node_ids_arr = np.array(node_ids, dtype=np.int64)
        c = np.array([count[(branch_idx, nid)] for nid in node_ids], dtype=np.float64)

        profile_stack = np.stack([profile_sum[(branch_idx, nid)] for nid in node_ids])
        profile_means = profile_stack / c[:, None]
        for b, col in enumerate(profile_columns):
            values = np.full(n_rows, np.nan)
            values[node_ids_arr] = profile_means[:, b]
            df[col] = values

        charge_means = np.array([charge_sum[(branch_idx, nid)] for nid in node_ids]) / c
        charge_sq_means = (
            np.array([charge_sq_sum[(branch_idx, nid)] for nid in node_ids]) / c
        )
        variance = np.maximum(charge_sq_means - charge_means**2, 0.0)
        values = np.full(n_rows, np.nan)
        values[node_ids_arr] = np.sqrt(variance)
        df[charge_std_column] = values

    return LiteralTreeProperties(
        profile_columns=profile_columns,
        charge_std_column=charge_std_column,
        fallback_profile=atom_profile.mean(axis=0),
        fallback_charge_std=max(float(atom_charge.std()), 1e-6),
        sigma_values=sigma_values,
    )


def predict_via_data_storage_walk(
    tree: Any, paths: list[NodePath], props: LiteralTreeProperties
) -> AtomPrediction:
    """Predict by walking each atom's matched path deepest -> shallowest
    directly against ``tree.data_storage`` and using the first node whose
    row is populated -- the same fallback ``DASHTree.get_property_noNAN``
    itself implements, applied to a whole profile row at once instead of
    one scalar call per bin (see the module docstring for why this is a
    reimplementation rather than a live call).

    A node's profile row and its charge_std are always written together, by
    ``populate_tree_with_sigma_properties`` (never partially) -- so checking
    whether ``charge_std`` is present for a node tells you whether its whole
    row is, and one column check decides the whole row.

    Converts each touched branch's relevant columns to a plain numpy array
    ONCE (``df[cols].to_numpy()``), outside the per-atom walk, rather than
    re-selecting those columns from the DataFrame on every path step --
    ~1300x faster than re-selecting per lookup (``experiments/docs/dash.md``
    has the measurement).
    """
    n = len(paths)
    n_bins = len(props.profile_columns)
    cols = [*props.profile_columns, props.charge_std_column]
    atom_profile = np.empty((n, n_bins), dtype=np.float64)
    atom_charge_std = np.empty(n, dtype=np.float64)

    # A branch with zero training atoms never gets these new columns
    # written at all (populate_tree_with_sigma_properties only touches
    # branches that appear in its own accumulation) -- a real edge case
    # found only at full-store scale (a --limit 5000 probe never happened
    # to hit a test atom in such a branch; the full store did). Treated the
    # same as any other "nothing here" case: skip the branch, every atom
    # matching it falls straight through to the global fallback below.
    arrays: dict[int, NDArray[np.float64]] = {}
    for branch_idx in {path[0][0] for path in paths if path}:
        branch_df = tree.data_storage[branch_idx]
        if all(col in branch_df.columns for col in cols):
            arrays[branch_idx] = branch_df[cols].to_numpy(dtype=np.float64)

    for i, path in enumerate(paths):
        row = None
        arr = arrays.get(path[0][0]) if path else None
        if arr is not None:
            for _, node_id in reversed(path):
                candidate = arr[node_id]
                if not np.isnan(candidate[-1]):
                    row = candidate
                    break
        if row is None:
            atom_profile[i] = props.fallback_profile
            atom_charge_std[i] = props.fallback_charge_std
        else:
            atom_profile[i] = row[:-1]
            atom_charge_std[i] = row[-1]

    atom_area = atom_profile.sum(axis=1)
    atom_charge = atom_profile @ props.sigma_values
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
    ``experiments/tests/test_experiment_predictor_dash_optional.py`` checks that mapping
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
    ``predict_via_data_storage_walk``, which is the same degradation DASH's
    own ``get_molecules_partial_charges`` accepts, but a baseline that
    quietly predicted the global mean for part of the test set would be
    impossible to interpret:

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


class _DASHTreeMixin:
    """Shared ``DASHTree`` loading and atom-matching bookkeeping for
    ``DASHPredictor``: tree-loading kwargs, ``match_stats`` recording, and
    the two-failure-mode warning (see ``_atom_paths``'s docstring).

    Instances must set ``max_depth``/``attention_threshold``/``preload``/
    ``tree_folder_path``/``match_stats``/``_tree`` themselves (in their own
    ``__init__``) -- this mixin only supplies the methods, not the state.
    """

    max_depth: int
    attention_threshold: float
    preload: bool
    tree_folder_path: str | None
    match_stats: dict[str, dict[str, int]]
    _tree: Any

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


class DASHPredictor(_DASHTreeMixin, AtomPredictor):
    """Stage A DASH baseline: DASH-tree's published topology
    (``match_new_atom``, unmodified) with a back-off step that reproduces
    ``DASHTree.get_property_noNAN``'s own missing-value fallback semantics
    (deepest -> shallowest, first populated node wins, else the global
    mean) -- see ``populate_tree_with_sigma_properties``/
    ``predict_via_data_storage_walk``'s own docstrings for exactly what
    that means and what's unavoidably ours (there is no DASH function that
    populates an existing tree with a new property -- confirmed, not
    assumed). Uses the plain raw (undecomposed) profile as its property --
    see the module docstring for why.

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

    ``charge_reconciliation`` defaults to ``"std_weighted"``, algebraically
    identical to ``DASHTree.get_molecules_partial_charges``'s own formula
    (verified). ``charge_std_floor`` defaults to **0.1**, matching that same
    function's own ``default_std_value`` -- not the ``1e-12`` every other
    predictor in this project uses (see ``reconcile_charge``'s docstring).
    """

    name = "dash"

    def __init__(
        self,
        *,
        store: str,
        scheme: str,
        max_depth: int = 16,
        attention_threshold: float = 5.2,
        charge_reconciliation: str = "std_weighted",
        charge_std_floor: float = 0.1,
        stores_root: str | None = None,
        tree_folder_path: str | None = None,
        preload: bool = True,
    ) -> None:
        self.store = store
        self.scheme = scheme
        self.max_depth = max_depth
        self.attention_threshold = attention_threshold
        self.charge_reconciliation = charge_reconciliation
        self.charge_std_floor = charge_std_floor
        self.stores_root = stores_root
        self.tree_folder_path = tree_folder_path
        self.preload = preload
        self.match_stats: dict[str, dict[str, int]] = {}
        self._tree: Any = None
        self._props: LiteralTreeProperties | None = None

    def fit_atoms(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng  # nothing to tune -- DASH's own algorithm has no
        # hyperparameter to fit here beyond the tree-matching ones
        kwargs: dict[str, Any] = {}
        if self.stores_root is not None:
            kwargs["stores_root"] = Path(self.stores_root)
        atom_profile, _, _ = load_atom_truth(
            self.store,
            scheme=self.scheme,
            smiles=train.smiles,
            num_atoms=train.num_atoms,
            **kwargs,
        )
        tree = self._load_tree()
        paths = self._paths_for(train, split="train")
        self._props = populate_tree_with_sigma_properties(
            tree, paths, atom_profile, sigma_values=train.grid.values
        )

    def predict_atoms(self, test: MoleculeSet) -> AtomPrediction:
        if self._props is None:
            raise RuntimeError("fit_atoms must be called before predict_atoms")
        tree = self._load_tree()
        paths = self._paths_for(test, split="test")
        return predict_via_data_storage_walk(tree, paths, self._props)


def _build(params: Mapping[str, Any]) -> DASHPredictor:
    return DASHPredictor(**params)


register("dash", _build)
