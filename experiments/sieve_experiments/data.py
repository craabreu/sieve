"""Molecule/atom data for the experiment harness.

``MoleculeSet``, ``SigmaGridSpec``, ``molecule_sum`` and ``select`` are pure
numpy -- no cosmolayer, no rdkit -- so they are importable and testable
without any optional dependency (see experiments/tests/test_experiment_data.py
and the ``synthetic_molecule_set`` fixture in experiments/tests/helpers.py).

``load_molecule_set`` is the one function here that touches the real store:
it imports cosmolayer and rdkit locally, not at module level, so importing
this module never requires them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CACHE_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES_ROOT = REPO_ROOT / "stores"
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments" / "cache"


@dataclass(frozen=True)
class SigmaGridSpec:
    """A local stand-in for cosmolayer's ``SigmaGrid``.

    Deliberately not a re-export: this lets metrics.py and the smoke test
    import the grid shape without depending on cosmolayer being installed.
    """

    max_abs_sigma: float
    num_points: int

    @property
    def bin_width(self) -> float:
        return (2.0 * self.max_abs_sigma) / (self.num_points - 1)

    @property
    def values(self) -> NDArray[np.float64]:
        return np.linspace(-self.max_abs_sigma, self.max_abs_sigma, self.num_points)


# The cosmolayer default grid (51 points, -0.025..+0.025 e/A^2), identical
# across all three averaging schemes and identical to COSMO-NET's own grid --
# see docs/superpowers/specs/2026-08-24-baseline-experiment-harness-design.md.
DEFAULT_GRID = SigmaGridSpec(max_abs_sigma=0.025, num_points=51)


def molecule_sum(
    per_atom: NDArray[np.floating], mol_id: NDArray[np.int64], n_molecules: int
) -> NDArray[np.float64]:
    """Sum per-atom rows into per-molecule rows.

    A plain sum, never an average and never a normalization: unnormalized
    atom-level quantities (area, charge, unnormalized profile bins) partition
    the molecule-level quantity exactly (design.md 11.4).
    """
    per_atom = np.asarray(per_atom, dtype=np.float64)
    shape = (n_molecules,) if per_atom.ndim == 1 else (n_molecules, per_atom.shape[1])
    out = np.zeros(shape, dtype=np.float64)
    np.add.at(out, mol_id, per_atom)
    return out


@dataclass(frozen=True)
class MoleculeSet:
    """One split's worth of molecules, atoms aligned to store order.

    Molecule-level truth fields (``mol_*``) and atom-level truth fields
    (``atom_*``) are each optional and independent -- a synthetic set built
    for a test may supply only some of them.
    """

    smiles: list[str]
    num_atoms: NDArray[np.int64]
    net_charge: NDArray[np.float64]
    grid: SigmaGridSpec = DEFAULT_GRID
    mol_profile: NDArray[np.float64] | None = None
    mol_area: NDArray[np.float64] | None = None
    mol_charge: NDArray[np.float64] | None = None
    atom_profile: NDArray[np.floating] | None = None
    atom_area: NDArray[np.float64] | None = None
    atom_charge: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        n = len(self.smiles)
        if len(self.num_atoms) != n:
            raise ValueError("num_atoms must have one entry per molecule")
        if len(self.net_charge) != n:
            raise ValueError("net_charge must have one entry per molecule")

    @property
    def n_molecules(self) -> int:
        return len(self.smiles)

    @property
    def n_atoms(self) -> int:
        return int(np.sum(self.num_atoms))

    @property
    def atom_mol_id(self) -> NDArray[np.int64]:
        """Molecule index of each atom, e.g. [0,0,0,1,1,2,...]."""
        return np.repeat(np.arange(self.n_molecules), self.num_atoms)

    @property
    def screening_charge(self) -> NDArray[np.float64]:
        """The target any sigma-derived "charge" quantity should reconcile
        to or be scored against -- ``-net_charge``, not ``net_charge``.

        ``net_charge`` is the molecule's own (solute) formal charge. Every
        other "charge" field in this codebase (``atom_charge``,
        ``mol_charge_raw``, ``atom_table.charges``, ``Sum(sigma * profile)``)
        is the COSMO *screening* charge: the charge the dielectric
        continuum induces on the cavity surface, which opposes the enclosed
        solute charge (COSMO conductor-screening theory) -- confirmed
        empirically on chaos-store: molecules with ``net_charge == +1``
        average ``Sum(sigma * mol_profile) == -1.005``, not ``+1``
        (correlation -0.997 against ``net_charge``, +0.997 against this
        property). Reconciling or scoring a sigma-derived charge against
        raw ``net_charge`` silently flips the sign for every charged
        molecule.
        """
        return -self.net_charge

    def select(self, mol_mask: NDArray[np.bool_]) -> MoleculeSet:
        """The sub-set of molecules (and their atoms) where ``mol_mask`` is True.

        The only place a split mask is applied: every other function operates
        on whatever ``MoleculeSet`` it is handed.
        """
        mol_mask = np.asarray(mol_mask, dtype=bool)
        atom_mask = np.repeat(mol_mask, self.num_atoms)

        def mol_slice(x):
            return None if x is None else np.asarray(x)[mol_mask]

        def atom_slice(x):
            return None if x is None else np.asarray(x)[atom_mask]

        return MoleculeSet(
            smiles=[s for s, keep in zip(self.smiles, mol_mask, strict=True) if keep],
            num_atoms=np.asarray(self.num_atoms)[mol_mask],
            net_charge=np.asarray(self.net_charge)[mol_mask],
            grid=self.grid,
            mol_profile=mol_slice(self.mol_profile),
            mol_area=mol_slice(self.mol_area),
            mol_charge=mol_slice(self.mol_charge),
            atom_profile=atom_slice(self.atom_profile),
            atom_area=atom_slice(self.atom_area),
            atom_charge=atom_slice(self.atom_charge),
        )


def select_atoms_by_smiles(
    full_smiles: list[str],
    full_num_atoms: NDArray[np.int64],
    atom_arrays: dict[str, NDArray],
    *,
    wanted_smiles: list[str],
    wanted_num_atoms: NDArray[np.int64],
) -> dict[str, NDArray]:
    """Join per-atom arrays (laid out in full-store order) onto a wanted
    molecule order/subset, by SMILES -- never by position.

    ``load_molecule_set`` never populates ``MoleculeSet.atom_*`` (atom-level
    truth is deliberately not cached, per data.py's module docstring), so a
    predictor that needs it (DASH, later Sieve) loads the full store's atom
    truth itself and re-aligns it onto its own train/test split this way --
    the same "join by SMILES, not by position" idiom the design doc calls
    for on COSMO-NET's output (design.md risk #4).
    """
    index_by_smiles: dict[str, int] = {}
    for i, smi in enumerate(full_smiles):
        if smi in index_by_smiles:
            raise ValueError(
                f"duplicate SMILES in store, cannot join safely: {smi[:60]}"
            )
        index_by_smiles[smi] = i

    offsets = np.concatenate([[0], np.cumsum(full_num_atoms)])
    out_parts: dict[str, list[NDArray]] = {k: [] for k in atom_arrays}
    for smi, n in zip(wanted_smiles, wanted_num_atoms, strict=True):
        if smi not in index_by_smiles:
            raise KeyError(f"SMILES not found in store: {smi[:60]}")
        i = index_by_smiles[smi]
        if full_num_atoms[i] != n:
            raise ValueError(
                f"atom count mismatch for {smi[:60]}: store has "
                f"{full_num_atoms[i]}, expected {n}"
            )
        start, end = offsets[i], offsets[i + 1]
        for k, arr in atom_arrays.items():
            out_parts[k].append(arr[start:end])

    return {k: np.concatenate(parts, axis=0) for k, parts in out_parts.items()}


def _cache_key(store_dir: Path, scheme: str) -> str:
    import cosmolayer

    meta_path = store_dir / "metadata.json"
    meta_bytes = meta_path.read_bytes() if meta_path.exists() else b""
    payload = {
        "metadata_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "scheme": scheme,
        "cosmolayer_version": getattr(cosmolayer, "__version__", "unknown"),
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def load_molecule_set(
    store_name: str,
    *,
    scheme: str,
    split_column: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    stores_root: Path = DEFAULT_STORES_ROOT,
) -> tuple[MoleculeSet, dict[str, NDArray[np.bool_]]]:
    """Load molecule-level truth (profile, area, charge) plus split masks.

    Molecule-level truth is cached to ``cache_dir`` (~22 MB), keyed on the
    store's metadata, the scheme, and the cosmolayer version, so repeated
    runs against the same store/scheme pair pay the cosmolayer cost once.
    Atom-level truth is deliberately NOT cached here -- see the harness
    design doc for why -- and is left to the caller (a predictor's own
    ``fit``/``predict``) to compute from the store directly, if it needs it.
    """
    from cosmolayer.store import SegmentStore
    from rdkit import Chem

    store_dir = stores_root / store_name
    store = SegmentStore.load(store_dir)
    df = store.molecules_df

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{store_name}_{scheme}_molecules.npz"
    key = _cache_key(store_dir, scheme)

    cached = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["cache_key"].item()) == key:
                cached = {k: data[k] for k in data.files}

    if cached is not None:
        mol_area = cached["mol_area"]
        mol_profile = cached["mol_profile"]
        mol_charge = cached["mol_charge"]
        net_charge = cached["net_charge"]
    else:
        table = store.compute_molecule_sigma_profiles(scheme=scheme)
        mol_area = np.asarray(table.areas, dtype=np.float64)
        mol_profile = np.asarray(table.profiles, dtype=np.float64) * mol_area[:, None]
        mol_charge = np.asarray(table.charges, dtype=np.float64)

        params = Chem.SmilesParserParams()
        params.removeHs = False
        net_charge = np.array(
            [Chem.GetFormalCharge(Chem.MolFromSmiles(s, params)) for s in df.smiles],
            dtype=np.float64,
        )
        np.savez(
            cache_path,
            cache_key=np.array(key),
            mol_profile=mol_profile,
            mol_area=mol_area,
            mol_charge=mol_charge,
            net_charge=net_charge,
        )

    if limit is not None:
        keep = slice(0, limit)
        df = df.iloc[keep].reset_index(drop=True)
        mol_profile = mol_profile[keep]
        mol_area = mol_area[keep]
        mol_charge = mol_charge[keep]
        net_charge = net_charge[keep]

    mset = MoleculeSet(
        smiles=list(df.smiles),
        num_atoms=df.num_atoms.to_numpy(),
        net_charge=net_charge,
        grid=DEFAULT_GRID,
        mol_profile=mol_profile,
        mol_area=mol_area,
        mol_charge=mol_charge,
    )
    masks = {name: (df[split_column] == name).to_numpy() for name in splits}
    return mset, masks


def load_atom_truth(
    store_name: str,
    *,
    scheme: str,
    smiles: list[str],
    num_atoms: NDArray[np.int64],
    stores_root: Path = DEFAULT_STORES_ROOT,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Atom-level truth (profile, area, charge) for exactly the molecules in
    ``smiles``/``num_atoms`` (typically a ``MoleculeSet.select()`` split),
    aligned to that order.

    Deliberately not cached (~400 MB full-store; see design.md's Data
    section) -- recomputed from the store every call. This is what an
    ``AtomPredictor`` (DASH, later Sieve) calls itself in ``fit_atoms`` /
    ``predict_atoms`` when it needs atom-level truth, since
    ``load_molecule_set`` never populates it.
    """
    from cosmolayer.store import SegmentStore

    store = SegmentStore.load(stores_root / store_name)
    df = store.molecules_df
    full_num_atoms = df.num_atoms.to_numpy()

    atom_table = store.compute_atom_sigma_profiles(scheme=scheme)
    atom_area = np.asarray(atom_table.areas, dtype=np.float64)
    atom_profile = (
        np.asarray(atom_table.profiles, dtype=np.float64) * atom_area[:, None]
    )
    # atom_table.charges is area*sigma computed from this SAME averaged
    # table -- not store.charges/store.atom_indices, which are the raw,
    # pre-averaging segment charges. cosmo-sac-2010's sigma-averaging kernel
    # redistributes charge across atom boundaries, so raw per-atom charge
    # and this averaged-profile-consistent charge are genuinely different
    # quantities (~60% relative gap on chaos-store), not just noisy
    # estimates of the same one. See
    # test_load_atom_truth_charge_is_scheme_consistent_not_raw.
    atom_charge = np.asarray(atom_table.charges, dtype=np.float64)

    full_atom_arrays = {
        "atom_profile": atom_profile,
        "atom_area": atom_area,
        "atom_charge": atom_charge,
    }
    selected = select_atoms_by_smiles(
        list(df.smiles),
        full_num_atoms,
        full_atom_arrays,
        wanted_smiles=smiles,
        wanted_num_atoms=num_atoms,
    )
    return selected["atom_profile"], selected["atom_area"], selected["atom_charge"]
