"""Molecule/atom data for the charges experiment harness.

``MoleculeSet``, ``molecule_sum``, ``mol_to_blob``/``blob_to_mol`` are pure
rdkit + numpy -- no pandas, no network -- so they are importable and
testable without touching the real (8.3GB source / parsed parquet) store.
See experiments/tests/test_data.py and the
``synthetic_molecule_set`` fixture in experiments/tests/helpers.py.

Unlike cosmo_experiments' MoleculeSet, there is no SMILES field anywhere: a
conformer's target (``MBIScharge``) rides directly on its own RDKit ``Mol``
as a real atom property, and the store persists the serialized ``Mol``
itself -- see the design spec's "Store row format" decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES_ROOT = REPO_ROOT / "experiments" / "stores"
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments" / "cache"

# AtomProps carries MBIScharge (set via atom.SetDoubleProp); MolProps is
# cheap to include too and covers any future mol-level property. Chiral
# tags and 3D conformer coordinates are intrinsic Atom/Mol fields, not
# properties, so they survive ToBinary()/Chem.Mol() regardless of
# propertyFlags -- passed explicitly here anyway, rather than relying on
# rdkit's own global pickle-property default, so this doesn't silently
# break if that default ever changes upstream.
#
# PrivateProps is required too, not just AtomProps/MolProps: rdkit treats
# any underscore-prefixed property name as "private" and silently drops it
# from ToBinary()'s output unless PrivateProps is explicitly OR'd in --
# confirmed empirically (a round trip without it produces a Mol with an
# *empty* prop dict for such names, no error). MBIScharge survives without
# this flag only because its name happens not to start with "_"; RDKit's
# own CIP labels (``_CIPCode`` and friends) and this codebase's own
# ``_sieve_rigorous_cip_labeled`` marker (see prepare_dash.py /
# sieve.io.rdkit_adapter.CIP_LABELED_PROP) both need it.


def mol_to_blob(mol: Any) -> bytes:
    """Serialize ``mol`` to bytes, preserving atom/bond/mol properties (which
    is where ``MBIScharge`` and the CIP-label props live) and stereo/chiral
    tags (intrinsic, always preserved).

    ``BondProps`` matters as much as ``AtomProps``: ``rdCIPLabeler`` writes
    the canonical E/Z descriptor of a double bond to that *bond's* own
    ``_CIPCode``, which sieve's ``bond_stereo`` attribute reads. Omitting the
    bit dropped those silently -- ``prepare_store`` ran the labeler, the
    mol-level ``CIP_LABELED_PROP`` marker survived the round trip, so
    ``_ensure_cip_labels`` short-circuited on load and never recomputed, and
    every stored molecule reported ``bond_stereo == "none"``. Stores written
    before this fix must be rebuilt for that attribute to carry any signal.
    """
    from rdkit import Chem

    return mol.ToBinary(
        Chem.PropertyPickleOptions.AtomProps
        | Chem.PropertyPickleOptions.BondProps
        | Chem.PropertyPickleOptions.MolProps
        | Chem.PropertyPickleOptions.PrivateProps
    )


def blob_to_mol(blob: bytes) -> Any:
    """Deserialize a blob written by ``mol_to_blob`` back into a ``Mol``."""
    from rdkit import Chem

    return Chem.Mol(blob)


def molecule_sum(
    per_atom: NDArray[np.floating], mol_id: NDArray[np.int64], n_molecules: int
) -> NDArray[np.float64]:
    """Sum per-atom values into per-conformer rows. A plain sum: the
    conformer's own net charge is the sum of its atoms' real partial
    charges, no averaging or normalization involved."""
    per_atom = np.asarray(per_atom, dtype=np.float64)
    out = np.zeros(n_molecules, dtype=np.float64)
    np.add.at(out, mol_id, per_atom)
    return out


@dataclass(frozen=True)
class MoleculeSet:
    """One split's worth of conformers. Each entry in ``mols`` is one
    conformer's own RDKit ``Mol``, its atoms carrying ``atom_property`` as a
    real double property -- there is no separate, position-aligned target
    array to keep in sync.

    ``molecule_property``/``molecule_value`` are the optional per-molecule
    total the atom values should sum to (the DASH series' ``net_charge``);
    both are ``None`` for a dataset with no such constraint.

    ``ids`` carries every remaining store column as per-conformer
    provenance -- the DASH stores' ``chembl_id``/``conf_id``/``dash_id``,
    some other dataset's own keys. Nothing in the harness groups or
    clusters by them; they are written straight through to
    ``predictions.npz``.
    """

    mols: list[Any]
    atom_property: str
    molecule_property: str | None = None
    molecule_value: NDArray[np.float64] | None = None
    ids: Mapping[str, list[str | None]] = field(default_factory=dict)
    split: list[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.mols)
        if self.molecule_value is not None and len(self.molecule_value) != n:
            raise ValueError("molecule_value must have one entry per conformer")
        if (self.molecule_property is None) != (self.molecule_value is None):
            raise ValueError(
                "molecule_property and molecule_value must be set together"
            )
        for key, values in self.ids.items():
            if len(values) != n:
                raise ValueError(f"ids[{key!r}] must have one entry per conformer")
        if self.split is not None and len(self.split) != n:
            raise ValueError("split must have one entry per conformer")

    @property
    def n_conformers(self) -> int:
        return len(self.mols)

    @property
    def num_atoms(self) -> NDArray[np.int64]:
        return np.array([m.GetNumAtoms() for m in self.mols], dtype=np.int64)

    @property
    def n_atoms(self) -> int:
        return int(self.num_atoms.sum()) if self.mols else 0

    @property
    def atom_mol_id(self) -> NDArray[np.int64]:
        """Conformer index of each atom, e.g. [0,0,0,1,1,2,...]."""
        return np.repeat(np.arange(self.n_conformers), self.num_atoms)

    @property
    def atom_target(self) -> NDArray[np.float64]:
        """Per-atom ground truth for ``atom_property``, flattened across
        every conformer's own atom order."""
        if not self.mols:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([self._mol_target(m) for m in self.mols])

    def _mol_target(self, mol: Any) -> NDArray[np.float64]:
        out = np.empty(mol.GetNumAtoms(), dtype=np.float64)
        for i, atom in enumerate(mol.GetAtoms()):
            if not atom.HasProp(self.atom_property):
                raise KeyError(
                    f"atom {i} of a stored conformer has no property "
                    f"{self.atom_property!r} -- the store was prepared for a "
                    "different target"
                )
            out[i] = atom.GetDoubleProp(self.atom_property)
        return out

    def select(self, mol_mask: NDArray[np.bool_]) -> MoleculeSet:
        """The sub-set of conformers where ``mol_mask`` is True. The only
        place a split mask is applied."""
        mol_mask = np.asarray(mol_mask, dtype=bool)
        idx = np.flatnonzero(mol_mask)
        return MoleculeSet(
            mols=[self.mols[i] for i in idx],
            atom_property=self.atom_property,
            molecule_property=self.molecule_property,
            molecule_value=(
                None
                if self.molecule_value is None
                else np.asarray(self.molecule_value)[mol_mask]
            ),
            ids={k: [v[i] for i in idx] for k, v in self.ids.items()},
            split=None if self.split is None else [self.split[i] for i in idx],
        )
