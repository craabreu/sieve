"""RDKit -> AtomBatch (design.md 11.2)."""

from __future__ import annotations

import numpy as np

from sieve.batch import AtomBatch
from sieve.config import SieveConfig

_ATTRS = {
    "element": lambda a: a.GetSymbol(),
    "degree": lambda a: str(a.GetDegree()),
    "hybridization": lambda a: str(a.GetHybridization()),
    "aromatic": lambda a: str(a.GetIsAromatic()),
    "formal_charge": lambda a: str(a.GetFormalCharge()),
    "num_h": lambda a: str(a.GetTotalNumHs()),
    "chirality": lambda a: str(a.GetChiralTag()),
}


def build_codes(mols, attributes):
    """Learn dense code tables from a corpus.

    Every attribute reserves one code above its observed maximum for unseen
    categories: an unknown value must fail to match at level 0 and back off,
    never silently collide with a seen one.
    """
    codes = {}
    for name in attributes:
        fn = _ATTRS[name]
        seen = sorted({fn(a) for m in mols for a in m.GetAtoms()})
        codes[name] = {v: i for i, v in enumerate(seen)}
    edge_codes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    return codes, edge_codes


def from_rdkit(mols, y=None, *, config: SieveConfig, atom_order=None) -> AtomBatch:
    flat = [a for g in config.attribute_levels for a in g]
    n = sum(m.GetNumAtoms() for m in mols)
    node_attrs = np.zeros((n, len(flat)), np.int64)
    elements = np.zeros(n, np.int64)
    graph_id = np.zeros(n, np.int64)
    src, dst, attr = [], [], []
    off = 0
    for gi, mol in enumerate(mols):
        order = (
            list(range(mol.GetNumAtoms()))
            if atom_order is None
            else list(atom_order[gi])
        )
        inv = np.empty(mol.GetNumAtoms(), np.int64)
        inv[order] = np.arange(mol.GetNumAtoms())
        for local, idx in enumerate(order):
            a = mol.GetAtomWithIdx(int(idx))
            g = off + local
            for j, name in enumerate(flat):
                table = config.attribute_codes[name]
                unknown = max(table.values()) + 1 if table else 0
                node_attrs[g, j] = table.get(_ATTRS[name](a), unknown)
            elements[g] = a.GetAtomicNum()
            graph_id[g] = gi
        for b in mol.GetBonds():
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            c = config.edge_codes.get(str(b.GetBondType()), 0)
            src += [u, v]
            dst += [v, u]
            attr += [c, c]
        off += mol.GetNumAtoms()
    return AtomBatch(
        node_attrs=node_attrs,
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.array(attr, np.int64),
        graph_id=graph_id,
        y=y,
        elements=elements,
    )


def from_smiles(smiles, y=None, *, config: SieveConfig) -> AtomBatch:
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles, mols, strict=True) if m is None]
        raise ValueError(f"unparseable SMILES: {bad[:3]}")
    return from_rdkit(mols, y, config=config)
