"""Which molecules this series' model premise makes equivalent.

A molecule's conformers, an exactly duplicated structure, and an enantiomer
are indistinguishable to every arm here: they share one graph, and no
featurization in this series reads chirality. Diastereomers and E/Z isomers
are NOT equivalent -- they are measurably different molecules (spec section 1)
and merging them would define the target as a mean over different chemistry.

See docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md section 3.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def mirror_mol(mol: Any) -> Any:
    """A copy with every tetrahedral centre inverted, bond stereo untouched.

    Bond stereo is deliberately preserved: a mirror image has the opposite
    handedness at every stereocentre but the same E/Z geometry, which is what
    makes E/Z isomers survive the key while enantiomers do not.
    """
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == _CW:
            atom.SetChiralTag(_CCW)
        elif tag == _CCW:
            atom.SetChiralTag(_CW)
    return out


def collapse_key(mol: Any) -> str:
    """The equivalence class of ``mol``, as a canonical string.

    ``min`` over the molecule and its mirror picks the same representative of
    an enantiomer pair from either side, so the two share a key without any
    pairwise comparison.
    """
    return min(Chem.MolToSmiles(mol), Chem.MolToSmiles(mirror_mol(mol)))
