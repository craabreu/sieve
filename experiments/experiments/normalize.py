"""Sum-constraint schemes: redistribute a predictor's raw, unnormalized
per-atom values so each molecule's atoms sum to a known per-molecule total
(``config.TargetCfg.molecule_property``). Kept independent of any predictor
so a run can apply one to an already-computed raw prediction without
re-matching or re-fitting anything (see ``config.ExperimentCfg.
normalization`` and ``runner._predict``).

Both schemes come from DASH's own post-hoc charge conservation (the paper's
eq. 4 and ``get_molecules_partial_charges``'s ``symmetric`` branch), but
neither is charge-specific: they need only a per-atom value, a per-atom
std, and a per-molecule total.

Every entry in ``NORMALIZERS`` shares one signature, ``(raw_value, raw_std,
molecule_value, mol_id, n_conformers) -> atom_value``, even though
``equal_weighted_normalize`` ignores ``raw_std`` entirely -- this lets
calling code stay normalization-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from experiments.data import molecule_sum

# get_molecules_partial_charges' own hardcoded default -- see
# std_weighted_normalize's own docstring, and predictors/dash_pretrained.py's
# module docstring, for why using it here is faithful, not invented.
_DEFAULT_STD_VALUE = 0.1


def std_weighted_normalize(
    raw_value: NDArray[np.floating],
    raw_std: NDArray[np.floating],
    molecule_value: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    n_conformers: int,
) -> NDArray[np.float64]:
    """DASH's own eq 4 (std-weighted normalization), pure numpy -- no tree,
    no rdkit. Verified against ``get_molecules_partial_charges``'s actual
    ``symmetric`` branch, not the paper's printed eq 2/3 (which has a sign
    error -- see predictors/dash_pretrained.py's own module docstring for
    both proofs).

    A non-positive (including NaN) entry in ``raw_std`` is floored to
    ``get_molecules_partial_charges``'s own ``default_std_value`` (0.1) --
    the authors' own published fallback for *that* quantity. A NaN entry in
    ``raw_value`` is not floored or substituted: it propagates through
    ``molecule_sum`` into that whole conformer's residual, so every atom in
    a conformer with even one unmatched raw value ends up NaN.
    """
    raw_value = np.asarray(raw_value, dtype=np.float64)
    raw_std = np.asarray(raw_std, dtype=np.float64)
    effective_std = np.where(raw_std > 0, raw_std, _DEFAULT_STD_VALUE)
    tot_value_tree = molecule_sum(raw_value, mol_id, n_conformers)
    tot_std_tree = molecule_sum(effective_std, mol_id, n_conformers)
    residual = np.asarray(molecule_value, dtype=np.float64) - tot_value_tree
    return raw_value + (residual[mol_id] * effective_std / tot_std_tree[mol_id])


def equal_weighted_normalize(
    raw_value: NDArray[np.floating],
    raw_std: NDArray[np.floating],
    molecule_value: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    n_conformers: int,
) -> NDArray[np.float64]:
    """A simpler sum-constraint scheme: spread each conformer's
    residual equally across its own atoms, ignoring ``raw_std`` entirely
    (accepted only for signature parity with ``std_weighted_normalize`` --
    see module docstring). A NaN ``raw_value`` on any atom propagates to
    the whole conformer the same way ``std_weighted_normalize``'s does, via
    ``molecule_sum``.
    """
    del raw_std
    raw_value = np.asarray(raw_value, dtype=np.float64)
    tot_value_tree = molecule_sum(raw_value, mol_id, n_conformers)
    residual = np.asarray(molecule_value, dtype=np.float64) - tot_value_tree
    n_atoms_per_mol = molecule_sum(np.ones_like(raw_value), mol_id, n_conformers)
    return raw_value + residual[mol_id] / n_atoms_per_mol[mol_id]


NORMALIZERS: dict[
    str,
    Callable[
        [
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.int64],
            int,
        ],
        NDArray[np.float64],
    ],
] = {
    "std_weighted": std_weighted_normalize,
    "equal_weighted": equal_weighted_normalize,
}
