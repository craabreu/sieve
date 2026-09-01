"""RDKit -> NodeBatch (design.md 11.2)."""

from __future__ import annotations

import functools
import itertools

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

from sieve.batch import NodeBatch, concat_batches
from sieve.config import SieveConfig


@functools.lru_cache(maxsize=1)
def _periodic_table():
    from rdkit import Chem

    return Chem.GetPeriodicTable()


def _period(a) -> str:
    return str(_periodic_table().GetRow(a.GetAtomicNum()))


# Row lengths of the long-form periodic table (period 1 through 7): 2 for the
# s-only first row, 8 while the d-block is absent, 18 once it appears, 32
# once the f-block appears too. First-atomic-number-of-each-period follows
# by cumulative sum.
_PERIOD_LENGTHS = (2, 8, 8, 18, 18, 32, 32)
_PERIOD_STARTS = tuple(itertools.accumulate((1, *_PERIOD_LENGTHS[:-1])))


def _group(a) -> str:
    """IUPAC group (1-18) from atomic number and period, computed rather than
    tabulated: within a period, position and group coincide once every block
    present in that period's row is accounted for.

    Position 1/2 (s-block) is always group 1/2, except period 1 (H, He),
    where He's own position (2) is not its group -- it is a noble gas placed
    in group 18 by convention, not by any block-filling logic, so period 1
    is a hardcoded exception rather than an instance of the general rule.
    Elsewhere, a period of length 8 or 18 has no gap: position p beyond the
    s-block maps to group p + (18 - length) (verified against the full
    118-element table this replaced). A period of length 32 inserts a
    14/15-wide f-block with no well-defined group between the s- and
    d-blocks (positions 3-17, i.e. the lanthanides/actinides) -- those atoms
    fall back to "none", the same sentinel "chirality" uses for its own
    undefined case, which also covers atomic number 0 (RDKit's dummy atom).
    """
    z = a.GetAtomicNum()
    if z <= 0:
        return "none"
    period = _periodic_table().GetRow(z)
    start, length = _PERIOD_STARTS[period - 1], _PERIOD_LENGTHS[period - 1]
    p = z - start + 1
    if period == 1:
        return "1" if p == 1 else "18"
    if length == 32 and 3 <= p <= 17:
        return "none"
    return str(p if p <= 2 else p + (18 - length))


_ATTRS = {
    "element": lambda a: a.GetSymbol(),
    "degree": lambda a: str(a.GetDegree()),
    "hybridization": lambda a: str(a.GetHybridization()),
    "aromatic": lambda a: str(a.GetIsAromatic()),
    "formal_charge": lambda a: str(a.GetFormalCharge()),
    "num_h": lambda a: str(a.GetTotalNumHs()),
    "period": _period,
    "group": _group,
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


def _serialize_mol(mol) -> bytes:
    """Explicit, all-properties serialization for shipping a ``Mol`` across a
    process boundary.

    RDKit's *global* default pickle-property setting is ``NoProps`` -- naive
    pickling (which is what joblib's default ``loky`` backend does to task
    arguments) silently drops every atom/mol property. ``MBIScharge``-style
    targets would fail loudly on the worker side (``GetDoubleProp`` raises on
    a missing prop); the ``chirality`` attribute would not -- it reads
    ``.GetPropsAsDict().get("_CIPCode", "none")`` and would silently become
    ``"none"`` for every atom instead of raising. ``AllProps`` (rather than
    naming ``AtomProps``/``MolProps``/``PrivateProps`` individually) is used
    deliberately: it is the literal union of every pickle-property bit,
    confirmed to include the "private" bit that underscore-prefixed props
    like ``_CIPCode`` need, so it cannot omit a category the way an
    enumerated list can.
    """
    from rdkit import Chem

    return mol.ToBinary(Chem.PropertyPickleOptions.AllProps)


def _deserialize_mol(blob: bytes):
    from rdkit import Chem

    return Chem.Mol(blob)


def _chunk_boundaries_by_atom_count(mols: list, n_chunks: int) -> list[tuple[int, int]]:
    """Molecule-index boundaries splitting ``mols`` into ``n_chunks`` pieces
    with roughly equal total atom count.

    Splitting on molecule *count* alone unbalances workers badly when the
    corpus mixes small and large molecules (a real property of both the
    COSMO and DASH corpora), so this splits on cumulative atom count via
    binary search instead.
    """
    n = len(mols)
    n_chunks = max(1, min(n_chunks, n))
    if n_chunks == 1:
        return [(0, n)]
    counts = np.fromiter((m.GetNumAtoms() for m in mols), dtype=np.int64, count=n)
    cum = np.concatenate([[0], np.cumsum(counts)])
    total = int(cum[-1])
    targets = [total * k / n_chunks for k in range(1, n_chunks)]
    cuts = {int(np.searchsorted(cum, t)) for t in targets}
    bounds = sorted({0, n, *(c for c in cuts if 0 < c < n)})
    return list(itertools.pairwise(bounds))


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

    Mutates ``mol`` in place, which matters under ``n_jobs``: called from a
    worker process, the mutation lands on that worker's own deserialized
    copy and never returns to the caller's original ``Mol`` -- correct (the
    worker's own featurization still sees the labels), just double work if
    the same corpus is featurized again without having been pre-labeled by
    ``prepare_store.py``.
    """
    if mol.HasProp(CIP_LABELED_PROP):
        return
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
    mol.SetBoolProp(CIP_LABELED_PROP, True)


def _discover_codes_chunk(blobs: list[bytes], attributes) -> dict[str, set[str]]:
    """Worker body for ``build_codes``: the per-attribute *set* of observed
    values for one chunk of (already CIP-labeled, if needed) molecules.

    Returns raw sets, not code tables -- assigning dense integer codes needs
    the union across every chunk first, so numbering happens once in the
    parent after all chunks return (deterministic regardless of how many
    chunks there are or which finishes first).
    """
    mols = [_deserialize_mol(b) for b in blobs]
    return {
        name: {_ATTRS[name](a) for m in mols for a in m.GetAtoms()}
        for name in attributes
    }


def build_codes(mols, attributes, *, n_jobs: int | None = None):
    """Learn dense code tables from a corpus.

    Every attribute reserves one code above its observed maximum for unseen
    categories: an unknown value must fail to match at level 0 and back off,
    never silently collide with a seen one.

    ``n_jobs`` parallelizes the per-attribute value-discovery pass, which is
    a set union -- commutative and associative, so chunking and merging sets
    from workers gives the exact same discovered vocabulary as the
    sequential pass, just found in a different order (codes are assigned
    from the sorted union, so the *numbering* is unaffected by chunk count
    or completion order too). The CIP-labeling pre-pass below always runs
    sequentially in this process first, so workers never need to compute or
    return labels themselves -- only from_rdkit's own worker path has that
    concern.
    """
    if "chirality" in attributes:
        # Same fallback from_rdkit applies -- needed here too, since a
        # fresh fit's own code-table discovery (this function) typically
        # runs on the training mols *before* from_rdkit ever sees them.
        # Without it, an un-prepped corpus would silently discover only
        # "none" as the observed chirality vocabulary. Runs before any
        # chunking/dispatch, so every worker's serialized blob already
        # carries the labels.
        for m in mols:
            _ensure_cip_labels(m)

    if n_jobs is None or n_jobs == 1 or len(mols) < 2:
        seen_by_attr = {
            name: {_ATTRS[name](a) for m in mols for a in m.GetAtoms()}
            for name in attributes
        }
    else:
        n_chunks = 4 * effective_n_jobs(n_jobs)
        bounds = _chunk_boundaries_by_atom_count(mols, n_chunks)
        blob_chunks = [[_serialize_mol(m) for m in mols[s:e]] for s, e in bounds]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_discover_codes_chunk)(blobs, attributes) for blobs in blob_chunks
        )
        seen_by_attr = {
            name: set().union(*(r[name] for r in results)) for name in attributes
        }

    codes = {
        name: {v: i for i, v in enumerate(sorted(seen_by_attr[name]))}
        for name in attributes
    }
    edge_codes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    return codes, edge_codes


def _from_rdkit_sequential(
    mols,
    y,
    *,
    config: SieveConfig,
    node_order,
    y_from_atom_prop: str | None,
) -> NodeBatch:
    """The actual per-molecule featurization loop, unchanged by ``n_jobs``.

    Runs once per chunk (including the single, whole-corpus "chunk" when
    ``n_jobs`` is not set); ``from_rdkit`` is the thin dispatcher around it.
    """
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


def _from_rdkit_worker(
    blobs: list[bytes],
    y_chunk,
    config: SieveConfig,
    node_order_chunk,
    y_from_atom_prop: str | None,
) -> NodeBatch:
    mols = [_deserialize_mol(b) for b in blobs]
    return _from_rdkit_sequential(
        mols,
        y_chunk,
        config=config,
        node_order=node_order_chunk,
        y_from_atom_prop=y_from_atom_prop,
    )


def from_rdkit(
    mols,
    y=None,
    *,
    config: SieveConfig,
    node_order=None,
    y_from_atom_prop: str | None = None,
    n_jobs: int | None = None,
) -> NodeBatch:
    """Featurize a corpus of RDKit ``Mol``s into one ``NodeBatch``.

    ``n_jobs`` chunks ``mols`` by cumulative atom count
    (``_chunk_boundaries_by_atom_count``) and featurizes each chunk in a
    separate worker via ``joblib``, reassembling with ``concat_batches``.
    Dispatched and concatenated in chunk order regardless of ``n_jobs`` or
    which worker finishes first, so output is byte-identical to the
    sequential path at any ``n_jobs`` -- verified directly, not just argued
    (see ``tests/test_rdkit_adapter.py``'s determinism tests).

    Molecules are shipped to workers as explicit, all-properties-preserving
    blobs (``_serialize_mol``/``_deserialize_mol``), never as naively-pickled
    live ``Mol`` objects -- see ``_serialize_mol``'s own docstring for why
    that distinction is a correctness requirement, not an optimization.
    """
    if y is not None and y_from_atom_prop is not None:
        raise ValueError("pass either y or y_from_atom_prop, not both")

    if n_jobs is None or n_jobs == 1 or len(mols) < 2:
        return _from_rdkit_sequential(
            mols,
            y,
            config=config,
            node_order=node_order,
            y_from_atom_prop=y_from_atom_prop,
        )

    n_chunks = 4 * effective_n_jobs(n_jobs)
    bounds = _chunk_boundaries_by_atom_count(mols, n_chunks)

    # y, when a real array, is per-*node* -- sliced by cumulative atom count,
    # not by molecule index, unlike node_order (per-molecule, sliced
    # positionally) and the mols themselves.
    if y is not None:
        counts = np.fromiter(
            (m.GetNumAtoms() for m in mols), dtype=np.int64, count=len(mols)
        )
        atom_cum = np.concatenate([[0], np.cumsum(counts)])
    y_chunks = (
        [y[atom_cum[s] : atom_cum[e]] for s, e in bounds]
        if y is not None
        else [None] * len(bounds)
    )
    order_chunks = (
        [node_order[s:e] for s, e in bounds]
        if node_order is not None
        else [None] * len(bounds)
    )
    blob_chunks = [[_serialize_mol(m) for m in mols[s:e]] for s, e in bounds]

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_from_rdkit_worker)(
            blobs, y_chunk, config, order_chunk, y_from_atom_prop
        )
        for blobs, y_chunk, order_chunk in zip(
            blob_chunks, y_chunks, order_chunks, strict=True
        )
    )
    return concat_batches(parts)


def from_smiles(smiles, y=None, *, config: SieveConfig) -> NodeBatch:
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles, mols, strict=True) if m is None]
        raise ValueError(f"unparseable SMILES: {bad[:3]}")
    return from_rdkit(mols, y, config=config)
