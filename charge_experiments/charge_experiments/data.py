"""Molecule/atom data for the charges experiment harness.

``MoleculeSet``, ``molecule_sum``, ``mol_to_blob``/``blob_to_mol`` are pure
rdkit + numpy -- no pandas, no network -- so they are importable and
testable without touching the real (8.3GB source / parsed parquet) store.
See charge_experiments/tests/test_charge_data.py and the
``synthetic_molecule_set`` fixture in charge_experiments/tests/helpers.py.

Unlike cosmo_experiments' MoleculeSet, there is no SMILES field anywhere: a
conformer's target (``MBIScharge``) rides directly on its own RDKit ``Mol``
as a real atom property, and the store persists the serialized ``Mol``
itself -- see the design spec's "Store row format" decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES_ROOT = REPO_ROOT / "charge_experiments" / "stores"
DEFAULT_CACHE_DIR = REPO_ROOT / "charge_experiments" / "cache"

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
# ``_sieve_rigorous_cip_labeled`` marker (see prepare_store.py /
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
    conformer's own RDKit ``Mol``, atoms carrying ``MBIScharge`` as a real
    double property -- there is no separate, position-aligned target array
    to keep in sync (contrast cosmo_experiments' ``MoleculeSet``, which
    carries ``mol_profile``/``atom_profile`` as arrays parallel to
    ``smiles``).

    The source SDF turns out to hold two record schemas (see
    ``prepare_store._parse_one_record``'s docstring): ChEMBL-sourced rows
    carry a ``CHEMBL_ID``, the rest carry a ``DASH_IDX`` instead. Exactly
    one of ``chembl_id``/``dash_id`` is set per conformer, the other
    ``None`` -- both are carried through purely as provenance; nothing in
    this harness groups/clusters by them at this point (that already
    happened once, at ``prepare_store`` time, and is baked into ``split``).
    """

    chembl_id: list[str | None]
    conf_id: list[str]
    mols: list[Any]
    net_charge: NDArray[np.float64]
    dash_id: list[str | None]
    split: list[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.mols)
        if len(self.chembl_id) != n:
            raise ValueError("chembl_id must have one entry per conformer")
        if len(self.conf_id) != n:
            raise ValueError("conf_id must have one entry per conformer")
        if len(self.net_charge) != n:
            raise ValueError("net_charge must have one entry per conformer")
        if len(self.dash_id) != n:
            raise ValueError("dash_id must have one entry per conformer")
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
    def atom_charge(self) -> NDArray[np.float64]:
        """Per-atom ``MBIScharge`` ground truth, flattened across every
        conformer's own atom order."""
        if not self.mols:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(
            [
                np.array(
                    [a.GetDoubleProp("MBIScharge") for a in m.GetAtoms()],
                    dtype=np.float64,
                )
                for m in self.mols
            ]
        )

    def select(self, mol_mask: NDArray[np.bool_]) -> MoleculeSet:
        """The sub-set of conformers where ``mol_mask`` is True. The only
        place a split mask is applied."""
        mol_mask = np.asarray(mol_mask, dtype=bool)
        idx = np.flatnonzero(mol_mask)
        return MoleculeSet(
            chembl_id=[self.chembl_id[i] for i in idx],
            conf_id=[self.conf_id[i] for i in idx],
            mols=[self.mols[i] for i in idx],
            net_charge=np.asarray(self.net_charge)[mol_mask],
            dash_id=[self.dash_id[i] for i in idx],
            split=None if self.split is None else [self.split[i] for i in idx],
        )
