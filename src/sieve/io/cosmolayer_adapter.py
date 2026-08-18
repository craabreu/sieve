"""cosmolayer SegmentStore -> AtomBatch (design.md 11.3, 11.4)."""
from __future__ import annotations

import numpy as np

from sieve.batch import AtomBatch, check_alignment
from sieve.config import SieveConfig


def from_segment_store(store, target="area", *, config: SieveConfig,
                       scheme="cosmo-rs") -> tuple[AtomBatch, np.ndarray]:
    """Build a batch and a test mask from a cosmolayer segment store.

    ``sigma_profile`` targets are converted to **area** units. The store's
    native ``SigmaProfileTable.profiles`` are area *fractions* summing to 1,
    with the scale held separately in ``.areas``; design.md 11.4 requires
    unnormalised areas, which is ``profiles * areas[:, None]``. Getting this
    wrong produces plausible numbers and no error.
    """
    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit

    df = store.molecules_df
    ai = np.asarray(store.atom_indices)
    n_atoms = int(ai.max()) + 1

    if target == "area":
        y = np.bincount(ai, weights=np.asarray(store.areas),
                        minlength=n_atoms)[:, None]
    elif target == "charge":
        y = np.bincount(ai, weights=np.asarray(store.charges),
                        minlength=n_atoms)[:, None]
    elif target == "sigma_profile":
        table = store.compute_atom_sigma_profiles(scheme=scheme)
        y = np.asarray(table.profiles, np.float64) * np.asarray(table.areas)[:, None]
    else:
        raise ValueError(f"unknown target {target!r}")

    params = Chem.SmilesParserParams()
    params.removeHs = False          # COSMO tables carry explicit hydrogens
    mols, orders = [], []
    for smi in df.smiles:
        mol = Chem.MolFromSmiles(smi, params)
        if mol is None:
            raise ValueError(f"unparseable SMILES in store: {smi[:60]}")
        # Atom-mapped SMILES are numbered onto the COSMO file's 0-based order.
        orders.append(np.argsort([a.GetAtomMapNum() for a in mol.GetAtoms()]))
        mols.append(mol)

    batch = from_rdkit(mols, y=y, config=config, atom_order=orders)

    # design.md 7.5 / 11.3: the check that actually catches reordering.
    pt = Chem.GetPeriodicTable()
    expected = np.array([pt.GetAtomicNumber(s) for s in store.atoms_df["element"]],
                        np.int64)
    check_alignment(batch, df.num_atoms.to_numpy(), expected)

    is_test = np.repeat((df.split == "test").to_numpy(), df.num_atoms.to_numpy())
    return batch, is_test
