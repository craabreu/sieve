"""DASH Stage A: the published DASH-tree topology, our own per-node
statistics fit on the training split, predicted by support-based back-off --
the same algorithm Sieve itself uses, applied to someone else's hierarchy
(design.md's "DASH" decision).

Two layers, deliberately split so the algorithm is testable without either
optional dependency:

- ``fit_backoff``/``predict_backoff`` -- pure numpy over pre-computed tree
  paths. No rdkit, no DASH-tree clone. Fast-suite tested
  (experiments/tests/test_experiment_predictor_dash.py).
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


VALID_LOCATION_MODES = ("sigma", "charge")
VALID_PROFILE_MODES = ("decomposed", "raw")


@dataclass(frozen=True)
class NodeStat:
    count: int
    shape: NDArray[np.float64]
    area: float
    location: float
    charge: float
    charge_std: float


@dataclass(frozen=True)
class BackoffStats:
    nodes: dict[PathKey, NodeStat]
    fallback: NodeStat
    sigma_values: NDArray[np.float64]


def _atom_location(charge: float, area: float) -> float:
    """The sigma-centroid of an atom's own profile: Sigma(sigma*profile)/area,
    computed as charge/area since ``load_atom_truth``'s (scheme-averaged)
    atom_charge already *is* that integral exactly -- see
    data.py::load_atom_truth and
    experiments/tests/test_experiment_store.py::
    test_load_atom_truth_charge_is_scheme_consistent_not_raw."""
    return charge / area if area > 0 else 0.0


def _shift(
    values: NDArray[np.float64], sigma_values: NDArray[np.float64], delta: float
) -> NDArray[np.float64]:
    """``values`` (as a function of ``sigma_values``) shifted right by
    ``delta``, via linear interpolation; zero outside the grid's range.

    Used in both directions: a negative delta re-centers an atom's own
    profile onto zero before it's averaged into a node's shape template (fit
    time); a positive delta shifts the averaged template back out to a
    predicted location (predict time).
    """
    return np.interp(sigma_values - delta, sigma_values, values, left=0.0, right=0.0)


def fit_backoff(
    paths: list[NodePath],
    atom_profile: NDArray[np.floating],
    atom_area: NDArray[np.floating],
    atom_charge: NDArray[np.floating],
    *,
    minimum_support: int,
    sigma_values: NDArray[np.floating],
) -> BackoffStats:
    """Accumulate count/mean per ``(branch_idx, node_id)`` over every node on
    every atom's path (not just the leaf), pruning nodes with fewer than
    ``minimum_support`` supporting atoms. ``fallback`` is the unconditional
    mean over all training atoms -- always available, used when an atom's
    path retains no node at all (a fresh branch, or every node pruned).

    Decomposes each atom into shape/location/magnitude before averaging,
    rather than bin-wise-averaging raw unnormalized profiles directly:
    atoms sharing a tree node rarely sit at exactly the same sigma-centroid
    ("location"), and averaging their raw profiles as-is smears/widens the
    result by however much those locations spread (measured ~5-35% width
    inflation on real chaos-store tree-node groups). Instead, each atom's
    own profile is shifted so its own centroid sits at zero and divided by
    its own area (a "shape" -- location- and scale-invariant), *then*
    averaged; ``NodeStat.location``/``.charge``/``.area`` are the plain
    means of each atom's own (location, charge, area), aggregated
    separately. ``predict_backoff`` reconstructs a prediction by shifting
    the averaged shape back out to a predicted location and scaling by a
    predicted area -- see its docstring for the sigma-vs-charge location
    choice.
    """
    n = len(paths)
    if not (len(atom_profile) == len(atom_area) == len(atom_charge) == n):
        raise ValueError(
            "paths, atom_profile, atom_area and atom_charge must have the "
            "same length (one entry per atom)"
        )
    atom_profile = np.asarray(atom_profile, dtype=np.float64)
    atom_area = np.asarray(atom_area, dtype=np.float64)
    atom_charge = np.asarray(atom_charge, dtype=np.float64)
    sigma_values = np.asarray(sigma_values, dtype=np.float64)
    if atom_profile.shape[1] != len(sigma_values):
        raise ValueError(
            f"atom_profile has {atom_profile.shape[1]} bins but sigma_values "
            f"has {len(sigma_values)}"
        )

    locations = np.array(
        [_atom_location(c, a) for c, a in zip(atom_charge, atom_area, strict=True)]
    )
    shapes = np.stack(
        [
            _shift(p, sigma_values, -loc) / a if a > 0 else np.zeros_like(p)
            for p, a, loc in zip(atom_profile, atom_area, locations, strict=True)
        ]
    )

    shape_sum: dict[PathKey, NDArray[np.float64]] = {}
    area_sum: dict[PathKey, float] = {}
    location_sum: dict[PathKey, float] = {}
    charge_sum: dict[PathKey, float] = {}
    charge_sq_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}

    for path, shape, area, loc, charge in zip(
        paths, shapes, atom_area, locations, atom_charge, strict=True
    ):
        for key in path:
            if key not in count:
                count[key] = 0
                shape_sum[key] = np.zeros_like(shape)
                area_sum[key] = 0.0
                location_sum[key] = 0.0
                charge_sum[key] = 0.0
                charge_sq_sum[key] = 0.0
            count[key] += 1
            shape_sum[key] += shape
            area_sum[key] += area
            location_sum[key] += loc
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
            shape=shape_sum[key] / c,
            area=area_sum[key] / c,
            location=location_sum[key] / c,
            charge=mean_charge,
            charge_std=max(np.sqrt(variance), 1e-6),
        )

    fallback = NodeStat(
        count=n,
        shape=shapes.mean(axis=0),
        area=float(np.mean(atom_area)),
        location=float(np.mean(locations)),
        charge=float(np.mean(atom_charge)),
        charge_std=max(float(np.std(atom_charge)), 1e-6),
    )
    return BackoffStats(nodes=nodes, fallback=fallback, sigma_values=sigma_values)


def predict_backoff(
    paths: list[NodePath], stats: BackoffStats, *, location_mode: str = "charge"
) -> AtomPrediction:
    """Walk each atom's path deepest -> shallowest, use the first retained
    node's stats, else ``stats.fallback``; reconstruct a profile by shifting
    that node's shape template out to a predicted location and scaling by
    its predicted area.

    ``location_mode`` picks how the scalar location is obtained from the
    chosen ``NodeStat`` (both keep ``area`` as the plain mean of atom
    areas):

    - ``"charge"`` (default): ``location = charge / area``, both means of
      the *same* additive quantities charge reconciliation already relies
      on (``roll_up``/``reconcile_charge``) -- the natural way to combine
      an intensive quantity (sigma is a density) across a heterogeneous
      population is total/total, not an average of ratios.
    - ``"sigma"``: ``location`` is the plain mean of each atom's own
      sigma-centroid, computed independently of area. Differs from
      ``"charge"`` whenever areas and locations both vary within a node
      (mean(charge)/mean(area) != mean(charge/area) in general) -- see
      test_location_mode_charge_vs_sigma_diverge_on_heterogeneous_areas.
    """
    if location_mode not in VALID_LOCATION_MODES:
        raise ValueError(
            f"location_mode must be one of {VALID_LOCATION_MODES}, "
            f"got {location_mode!r}"
        )
    n = len(paths)
    profile_dim = stats.fallback.shape.shape[0]
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

        area = chosen.area
        if location_mode == "charge":
            location = _atom_location(chosen.charge, area)
            charge = chosen.charge
        else:  # "sigma"
            location = chosen.location
            charge = location * area

        atom_profile[i] = area * _shift(chosen.shape, stats.sigma_values, location)
        atom_area[i] = area
        atom_charge[i] = charge
        atom_charge_std[i] = chosen.charge_std

    return AtomPrediction(
        atom_profile=atom_profile,
        atom_area=atom_area,
        atom_charge=atom_charge,
        atom_charge_std=atom_charge_std,
    )


@dataclass(frozen=True)
class RawNodeStat:
    count: int
    profile: NDArray[np.float64]
    charge_std: float


@dataclass(frozen=True)
class RawBackoffStats:
    nodes: dict[PathKey, RawNodeStat]
    fallback: RawNodeStat
    sigma_values: NDArray[np.float64]


def fit_backoff_raw(
    paths: list[NodePath],
    atom_profile: NDArray[np.floating],
    *,
    minimum_support: int,
    sigma_values: NDArray[np.floating],
) -> RawBackoffStats:
    """``fit_backoff``'s counterpart with the shape/location/area
    decomposition removed: each node's stat is just the plain bin-wise mean
    of its members' *raw, unnormalized* profiles -- no shift-to-zero-centroid,
    no divide-by-area. area and charge are never fit as separate quantities
    here; both are only ever derived from a predicted profile after the
    fact (``predict_backoff_raw``), the same convention molecule-level
    profile predictors use (``CosmonetPredictor``, ``prediction_from_profile``
    in ``predictors/chemprop_dmpnn.py``): ``area = profile.sum()``,
    ``charge = profile @ sigma_values``.

    ``fit_backoff``'s own docstring measured 5-35% width inflation from
    bin-wise-averaging raw profiles directly, across real chaos-store
    tree-node groups, which is exactly why that decomposition exists. This
    variant exists to *measure* that cost end to end (does the extra
    machinery actually buy better molecule-level metrics, or does it wash
    out after roll-up?), not because it is expected to win.

    ``atom_charge`` (needed only for each node's ``charge_std``, which
    ``predict_backoff_raw`` copies through for ``reconcile_charge``'s
    std-weighted mode) is derived as ``atom_profile @ sigma_values`` rather
    than taken as a separate input -- exactly the integral
    ``load_atom_truth``'s own scheme-averaged ``atom_charge`` already *is*
    (see ``_atom_location``'s docstring), so nothing is lost by not
    threading a second array through.
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
    atom_charge = atom_profile @ sigma_values

    profile_sum: dict[PathKey, NDArray[np.float64]] = {}
    charge_sum: dict[PathKey, float] = {}
    charge_sq_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}

    for path, profile, charge in zip(paths, atom_profile, atom_charge, strict=True):
        for key in path:
            if key not in count:
                count[key] = 0
                profile_sum[key] = np.zeros_like(profile)
                charge_sum[key] = 0.0
                charge_sq_sum[key] = 0.0
            count[key] += 1
            profile_sum[key] += profile
            charge_sum[key] += charge
            charge_sq_sum[key] += charge * charge

    nodes: dict[PathKey, RawNodeStat] = {}
    for key, c in count.items():
        if c < minimum_support:
            continue
        mean_charge = charge_sum[key] / c
        variance = max(charge_sq_sum[key] / c - mean_charge * mean_charge, 0.0)
        nodes[key] = RawNodeStat(
            count=c,
            profile=profile_sum[key] / c,
            charge_std=max(np.sqrt(variance), 1e-6),
        )

    fallback = RawNodeStat(
        count=n,
        profile=atom_profile.mean(axis=0),
        charge_std=max(float(np.std(atom_charge)), 1e-6),
    )
    return RawBackoffStats(nodes=nodes, fallback=fallback, sigma_values=sigma_values)


def predict_backoff_raw(
    paths: list[NodePath], stats: RawBackoffStats
) -> AtomPrediction:
    """``predict_backoff``'s counterpart for ``RawBackoffStats``: walk each
    atom's path deepest -> shallowest like ``predict_backoff``, but predict
    the chosen node's raw mean profile directly, with no location/area
    reconstruction step -- the node mean already carries whatever magnitude
    its members had. ``atom_area``/``atom_charge`` are derived from that
    predicted profile (``sum``, ``profile @ sigma_values``), not fit
    separately -- see ``fit_backoff_raw``'s docstring.
    """
    n = len(paths)
    profile_dim = stats.fallback.profile.shape[0]
    atom_profile = np.empty((n, profile_dim), dtype=np.float64)
    atom_charge_std = np.empty(n, dtype=np.float64)

    for i, path in enumerate(paths):
        chosen = stats.fallback
        for key in reversed(path):
            if key in stats.nodes:
                chosen = stats.nodes[key]
                break
        atom_profile[i] = chosen.profile
        atom_charge_std[i] = chosen.charge_std

    atom_area = atom_profile.sum(axis=1)
    atom_charge = atom_profile @ stats.sigma_values
    return AtomPrediction(
        atom_profile=atom_profile,
        atom_area=atom_area,
        atom_charge=atom_charge,
        atom_charge_std=atom_charge_std,
    )


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
    node over every node on every atom's path -- the same minimal
    accumulation ``fit_backoff_raw`` already does, deliberately with no
    shape/location decomposition (a choice of property, not an algorithmic
    deviation from DASH -- see the module docstring). ``charge`` is derived
    from the profile (``atom_profile @ sigma_values``) purely so its
    per-node std can be computed for reconciliation weighting -- charge
    itself is never predicted as its own quantity.

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
    # fit_backoff_raw's docstring. Only used here to compute a per-node std
    # for reconciliation; never predicted as its own quantity.
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
    one scalar call per bin (an earlier version of this function called
    ``get_property_noNAN`` 52 times per atom; measured at ~47
    microseconds/call, ~39 minutes extrapolated for a full-store predict --
    replaced with this direct version for speed).

    A node's profile row and its charge_std are always written together, by
    ``populate_tree_with_sigma_properties`` (never partially) -- so checking
    whether ``charge_std`` is present for a node tells you whether its whole
    row is, and one column check decides the whole row.

    Converts each touched branch's relevant columns to a plain numpy array
    ONCE (``df[cols].to_numpy()``), outside the per-atom walk, rather than
    re-selecting those columns from the DataFrame on every path step.
    Measured directly: repeated ``df[cols].iloc[node_id]`` costs ~467
    microseconds/call (pandas column-selection overhead dominates, worse
    than calling ``get_property_noNAN`` itself) -- indexing a precomputed
    numpy array costs ~0.35 microseconds/call, ~1300x faster. A first
    version of this function did the slow thing; caught by an actual timing
    run (a `--limit 5000` probe that didn't finish in 10 minutes, where it
    should have taken seconds), not assumed correct from line count alone.
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


class _DASHTreeMixin:
    """Shared ``DASHTree`` loading and atom-matching bookkeeping for both
    ``DASHBackoffPredictor`` and ``DASHLiteralPredictor`` -- identical
    tree-loading kwargs, ``match_stats`` recording, and the two-failure-mode
    warning (see ``_atom_paths``'s docstring). A pure extraction: no
    behavior change for ``DASHBackoffPredictor``, which used to define these
    two methods itself.

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


class DASHBackoffPredictor(_DASHTreeMixin, AtomPredictor):
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

    ``location_mode`` picks how ``predict_backoff`` derives a predicted
    atom's sigma-location -- ``"charge"`` (default) or ``"sigma"``. See
    ``predict_backoff``'s docstring. Ignored when ``profile_mode="raw"``.

    ``profile_mode`` picks the fit/predict algorithm:

    - ``"decomposed"`` (default): ``fit_backoff``/``predict_backoff`` -- the
      shape/location/area decomposition, i.e. area and charge are fit as
      their own per-node quantities.
    - ``"raw"``: ``fit_backoff_raw``/``predict_backoff_raw`` -- plain
      bin-wise averaging of raw, unnormalized profiles per node, no
      decomposition; area and charge are only ever derived from the
      predicted profile (sum, sigma-weighted sum), never fit separately.
      See ``fit_backoff_raw``'s docstring for why this variant exists.
    """

    name = "dash_backoff"

    def __init__(
        self,
        *,
        store: str,
        scheme: str,
        max_depth: int = 16,
        attention_threshold: float = 5.2,
        minimum_support: int = 5,
        charge_reconciliation: str = "std_weighted",
        location_mode: str = "charge",
        profile_mode: str = "decomposed",
        stores_root: str | None = None,
        tree_folder_path: str | None = None,
        preload: bool = True,
    ) -> None:
        if profile_mode not in VALID_PROFILE_MODES:
            raise ValueError(
                f"profile_mode must be one of {VALID_PROFILE_MODES}, "
                f"got {profile_mode!r}"
            )
        self.store = store
        self.scheme = scheme
        self.max_depth = max_depth
        self.attention_threshold = attention_threshold
        self.minimum_support = minimum_support
        self.charge_reconciliation = charge_reconciliation
        self.location_mode = location_mode
        self.profile_mode = profile_mode
        self.stores_root = stores_root
        self.tree_folder_path = tree_folder_path
        self.preload = preload
        self.match_stats: dict[str, dict[str, int]] = {}
        self._tree: Any = None
        self._stats: BackoffStats | RawBackoffStats | None = None

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
        if self.profile_mode == "raw":
            # atom_area/atom_charge loaded above are never fit separately in
            # raw mode -- see profile_mode's docstring.
            self._stats = fit_backoff_raw(
                paths,
                atom_profile,
                minimum_support=self.minimum_support,
                sigma_values=train.grid.values,
            )
        else:
            self._stats = fit_backoff(
                paths,
                atom_profile,
                atom_area,
                atom_charge,
                minimum_support=self.minimum_support,
                sigma_values=train.grid.values,
            )

    def predict_atoms(self, test: MoleculeSet) -> AtomPrediction:
        if self._stats is None:
            raise RuntimeError("fit_atoms must be called before predict_atoms")
        paths = self._paths_for(test, split="test")
        if isinstance(self._stats, RawBackoffStats):
            return predict_backoff_raw(paths, self._stats)
        return predict_backoff(paths, self._stats, location_mode=self.location_mode)


class DASHLiteralPredictor(_DASHTreeMixin, AtomPredictor):
    """Literal-DASH baseline: DASH's own published topology
    (``match_new_atom``, unmodified) with a back-off step that reproduces
    ``DASHTree.get_property_noNAN``'s own missing-value fallback semantics
    (deepest -> shallowest, first populated node wins, else the global
    mean) -- see ``populate_tree_with_sigma_properties``/
    ``predict_via_data_storage_walk``'s own docstrings for exactly what
    "literal" means here and what's unavoidably ours (there is no DASH
    function that populates an existing tree with a new property --
    confirmed, not assumed).

    Uses the plain raw (undecomposed) profile as its property -- the only
    property this predictor supports. ``DASHBackoffPredictor``'s
    "decomposed" shape/location/area mode is a choice of *which* property to
    populate nodes with, not an algorithmic deviation from DASH (DASH itself
    never decomposes anything -- see the module docstring), so it has no
    place in a literal reproduction; "raw" is the plainest choice, adding
    nothing DASH's own algorithm doesn't already do.

    ``charge_reconciliation`` defaults to ``"std_weighted"``, algebraically
    identical to ``DASHTree.get_molecules_partial_charges``'s own formula
    (verified). ``charge_std_floor`` defaults to **0.1**, matching that same
    function's own ``default_std_value`` -- not the ``1e-12`` every other
    predictor in this project uses (see ``reconcile_charge``'s docstring).

    ``store``/``scheme``/``max_depth``/``attention_threshold``/``preload``/
    ``stores_root``/``tree_folder_path``/``match_stats`` mean exactly what
    they do on ``DASHBackoffPredictor`` -- see there. There is no
    ``minimum_support`` or ``location_mode`` here: DASH's own algorithm has
    neither concept.
    """

    name = "dash_literal"

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


def _build(params: Mapping[str, Any]) -> DASHBackoffPredictor:
    return DASHBackoffPredictor(**params)


def _build_literal(params: Mapping[str, Any]) -> DASHLiteralPredictor:
    return DASHLiteralPredictor(**params)


register("dash_backoff", _build)
register("dash_literal", _build_literal)
