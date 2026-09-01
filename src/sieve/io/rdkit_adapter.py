"""RDKit -> NodeBatch (design.md 11.2)."""

from __future__ import annotations

import numpy as np

from sieve.batch import NodeBatch
from sieve.config import SieveConfig

_ATTRS = {
    "element": lambda a: a.GetSymbol(),
    "degree": lambda a: str(a.GetDegree()),
    "hybridization": lambda a: str(a.GetHybridization()),
    "aromatic": lambda a: str(a.GetIsAromatic()),
    "formal_charge": lambda a: str(a.GetFormalCharge()),
    "num_h": lambda a: str(a.GetTotalNumHs()),
    # RDKit's raw ChiralTag (CW/CCW) is defined relative to the atom's own
    # neighbor traversal order -- not canonical, so the same physical
    # stereocenter can get a different raw tag purely from incidental atom
    # numbering. _CIPCode (R/S, or r/s for pseudo-asymmetric centers) is
    # RDKit's own canonical, order-independent descriptor instead. It is
    # not set by molblock parsing alone; see from_rdkit's own fallback
    # computation below for callers that reach here without it already set
    # (e.g. from_smiles, or any Mol built outside charge_experiments'
    # prepare_store.py, which sets it once at data-prep time).
    "chirality": lambda a: a.GetPropsAsDict().get("_CIPCode", "none"),
}


def build_codes(mols, attributes):
    """Learn dense code tables from a corpus.

    Every attribute reserves one code above its observed maximum for unseen
    categories: an unknown value must fail to match at level 0 and back off,
    never silently collide with a seen one.
    """
    if "chirality" in attributes:
        # Same fallback from_rdkit applies -- needed here too, since a
        # fresh fit's own code-table discovery (this function) typically
        # runs on the training mols *before* from_rdkit ever sees them.
        # Without it, an un-prepped corpus would silently discover only
        # "none" as the observed chirality vocabulary.
        for m in mols:
            _ensure_cip_labels(m)
    codes = {}
    for name in attributes:
        fn = _ATTRS[name]
        seen = sorted({fn(a) for m in mols for a in m.GetAtoms()})
        codes[name] = {v: i for i, v in enumerate(seen)}
    edge_codes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    return codes, edge_codes


# Public so a caller that pre-computes CIP labels itself (e.g.
# charge_experiments' own prepare_store.py, at data-prep time) can set the
# same marker and skip _ensure_cip_labels' own recomputation -- the whole
# point of storing labels ahead of time.
CIP_LABELED_PROP = "_sieve_rigorous_cip_labeled"


def _ensure_cip_labels(mol) -> None:
    """Fallback for a ``Mol`` that reaches ``from_rdkit`` without
    pre-computed, *rigorous* ``_CIPCode`` atom props (e.g. built via
    ``from_smiles``, or from any caller other than ``charge_experiments``'s
    own ``prepare_store.py``, which sets these once at data-prep time).

    Gated on a dedicated mol-level marker, not on ``atom.HasProp("_CIPCode")``
    -- ``Chem.MolFromSmiles``'s own default sanitization already sets a
    *legacy* ``_CIPCode``, which can genuinely disagree with
    ``rdCIPLabeler``'s rigorous one for pseudo-asymmetric centers (verified:
    a bridged-bicyclic case gave legacy 'S'/'S' vs. rigorous 'r'/'s' for the
    same two atoms) -- checking prop presence alone would silently accept
    the legacy value and never call the rigorous labeler. The marker is a
    plain mol-level bool prop, so it survives the same ``mol_to_blob``
    round trip ``_CIPCode`` itself does.
    """
    if mol.HasProp(CIP_LABELED_PROP):
        return
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
    mol.SetBoolProp(CIP_LABELED_PROP, True)


def from_rdkit(
    mols,
    y=None,
    *,
    config: SieveConfig,
    node_order=None,
    y_from_atom_prop: str | None = None,
) -> NodeBatch:
    if y is not None and y_from_atom_prop is not None:
        raise ValueError("pass either y or y_from_atom_prop, not both")

    flat = [a for g in config.attribute_levels for a in g]
    n = sum(m.GetNumAtoms() for m in mols)
    node_attrs = np.zeros((n, len(flat)), np.int64)
    elements = np.zeros(n, np.int64)
    graph_id = np.zeros(n, np.int64)
    y_out = np.zeros((n, 1), np.float64) if y_from_atom_prop is not None else None
    src, dst, attr = [], [], []
    off = 0
    needs_cip = "chirality" in flat
    # Hoisted out of the per-atom loop below: all three are loop-invariant,
    # and the loop runs once per atom *per attribute* -- at 1.69M atoms and 5
    # attributes that is 8.4M repetitions of work that never changes.
    # `max(table.values())` alone profiled at ~10% of this function. The
    # plain-dict copy matters too: config.attribute_codes is a nested
    # MappingProxyType (config._freeze_mappings), whose lookup is measurably
    # slower than a dict's.
    getters = [_ATTRS[name] for name in flat]
    tables = [dict(config.attribute_codes[name]) for name in flat]
    unknowns = [max(t.values()) + 1 if t else 0 for t in tables]
    edge_codes = dict(config.edge_codes)
    for gi, mol in enumerate(mols):
        if needs_cip:
            _ensure_cip_labels(mol)
        order = (
            list(range(mol.GetNumAtoms()))
            if node_order is None
            else list(node_order[gi])
        )
        inv = np.empty(mol.GetNumAtoms(), np.int64)
        inv[order] = np.arange(mol.GetNumAtoms())
        for local, idx in enumerate(order):
            a = mol.GetAtomWithIdx(int(idx))
            g = off + local
            for j in range(len(flat)):
                node_attrs[g, j] = tables[j].get(getters[j](a), unknowns[j])
            elements[g] = a.GetAtomicNum()
            graph_id[g] = gi
            if y_out is not None:
                y_out[g, 0] = a.GetDoubleProp(y_from_atom_prop)
        for b in mol.GetBonds():
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            c = edge_codes.get(str(b.GetBondType()), 0)
            src += [u, v]
            dst += [v, u]
            attr += [c, c]
        off += mol.GetNumAtoms()
    return NodeBatch(
        node_attrs=node_attrs,
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.array(attr, np.int64),
        graph_id=graph_id,
        y=y if y is not None else y_out,
        elements=elements,
    )


def from_smiles(smiles, y=None, *, config: SieveConfig) -> NodeBatch:
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles, mols, strict=True) if m is None]
        raise ValueError(f"unparseable SMILES: {bad[:3]}")
    return from_rdkit(mols, y, config=config)
