# Design: Flexible edge attribute schema

**Status:** approved, ready for implementation planning
**Date:** 2026-09-02
**Scope:** make edge attributes configurable the way atom attributes already are — a named
registry of bond attribute providers, corpus-discovered code tables, and a configurable
`edge_attributes` tuple. Breaking change to `SieveConfig`, `NodeBatch` and the saved-model
format.

## Background

Atom attributes are fully configurable: `SieveConfig.attribute_levels` names them,
`attribute_codes` holds a per-attribute `{value: code}` table discovered from the corpus, and
`rdkit_adapter` resolves each name through the `_ATTRS` (per-atom) or `_MOL_ATTRS`
(whole-molecule precompute) registry. Adding an attribute means adding one registry entry.

Edge attributes have none of that. `NodeBatch.edge_attr` is a 1-D `(n_edges,)` array,
`SieveConfig.edge_codes` is a single flat `Mapping[str, int]`, and `build_codes` hardcodes it:

```python
edge_codes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}   # rdkit_adapter.py:375
```

Anything else — `DATIVE`, `UNSPECIFIED` — maps to `0` with no signal that it happened. There is
no way to add bond conjugation or ring membership short of editing the refinement path.

`design.md` already specifies the feature this document designs, and the code never implemented
it. §11.1 declares `edge_attrs: np.ndarray  # (n_edges, n_edge_attr) int64`; §10.1 declares
`edge_attributes: tuple[str, ...] = ("bond_type",)`. Part of this work is therefore closing a
gap between `design.md` and the code rather than inventing a new interface.

## Decisions

| Question | Decision |
|---|---|
| Graded edge levels, like `attribute_levels`? | **No — a flat set.** Edge attributes contribute at full resolution at every level, which is what `design.md` §3.6 already assumes ("edge attributes stay at full resolution in both arms"). Grading edges is a separate, larger change to the refinement chain. |
| Where do several edge attributes collapse into the one integer `refine` needs? | **In `refine`, from the config** (see "The collapse"). Storage stays 2-D and per-attribute; the collapse is not baked into the batch. |
| Which attributes ship first? | `bond_type` (default), `conjugated`, and three ring attributes. Bond `stereo` (E/Z) is deferred. |
| May `edge_attributes` be empty? | **Yes.** `()` is a supported control arm giving pure-topology refinement. |
| Saved-model back-compat? | **Clean break.** `FORMAT_VERSION` 2 → 3; loading a v2 model raises. |
| Naming for edge ring attributes | **Prefixed** — `bond_min_ring_size`, not `min_ring_size`. |

### Why `stereo` is deferred

Rigorous bond stereo wants `rdCIPLabeler`/`FindPotentialStereo` rather than the raw parse-time
tag, for the same reason atom `chirality` does — the raw tag is defined relative to incidental
atom numbering. That subtlety took atom `chirality` two commits to get right and would make this
spec about two things. It lands as a `_MOL_BOND_ATTRS` provider in a follow-up, needing no
further interface change.

### Why bond `aromatic` is not included

It is very nearly redundant with `bond_type == AROMATIC`. YAGNI; it can be added later as a
one-line registry entry.

## 1. Bond attribute registries

In `src/sieve/io/rdkit_adapter.py`, mirroring `_ATTRS` / `_MOL_ATTRS`:

```python
_BOND_ATTRS = {                                   # f(bond) -> str
    "bond_type":  lambda b: str(b.GetBondType()),
    "conjugated": lambda b: str(b.GetIsConjugated()),
}

_MOL_BOND_ATTRS = {                               # f(mol) -> Sequence[str], bond-index order
    "bond_in_ring":              _bond_in_ring,
    "bond_min_ring_size":        _bond_min_ring_size,         # RingInfo.MinBondRingSize
    "bond_num_ring_memberships": _bond_num_ring_memberships,  # RingInfo.NumBondRings
}
```

`_MOL_BOND_ATTRS` providers return one value per bond in RDKit **bond**-index order (the atom
registry's providers use atom-index order). The three ring providers call `Chem.GetSymmSSSR`,
whose result RDKit caches on the `Mol`, so they share the perception the atom-side ring
attributes already trigger.

Sentinels follow the atom-side convention established by `min_ring_size` and
`num_ring_memberships`: `bond_min_ring_size` returns `"none"` for a non-ring bond (a ring size
of zero is meaningless), while `bond_num_ring_memberships` returns `"0"` (a count of zero is
meaningful).

A name lives in exactly one of the two registries, and the edge registries are a separate
namespace from the atom ones — hence the `bond_` prefix on the ring attributes, so that a config
line and a grep both stay unambiguous.

Note that `bond_in_ring` is exactly derivable from either of the other two
(`bond_num_ring_memberships != "0"`, `bond_min_ring_size != "none"`). It is kept because it is
the cheapest and most commonly wanted of the three, but a config should choose one ring
attribute rather than stacking redundant ones — they multiply the edge alphabet without adding
information.

## 2. Config

```python
edge_attributes: tuple[str, ...] = ("bond_type",)
edge_codes: Mapping[str, Mapping[str, int]]      # name -> {value: code}
```

`__post_init__` additions:

- `edge_attributes` **may** be empty. The zero-width check at `config.py:45` stays in force for
  `attribute_levels` and must not be extended to edges.
- `set(edge_attributes) == set(edge_codes)`, raising on either direction of mismatch — a named
  attribute with no table, or a table for an unnamed attribute, are both config drift.

`_freeze_mappings`, `__getstate__` and `__setstate__` nest one level deeper for `edge_codes`,
matching what they already do for `attribute_codes`.

New and changed properties:

```python
@property
def edge_radices(self) -> tuple[int, ...]:
    """Alphabet size per edge attribute, including its reserved unknown code."""
    return tuple(len(self.edge_codes[n]) + 1 for n in self.edge_attributes)

@property
def n_edge_types(self) -> int:
    """Size of the collapsed edge alphabet. 1 when no edge attributes are configured."""
    return math.prod(self.edge_radices)
```

`n_edge_types` keeps its name and its meaning ("edge-code alphabet size") but is now a product
over per-attribute radices rather than `max(edge_codes.values()) + 1`.

## 3. The collapse

**This is the load-bearing decision of the design.**

`refine.py:105` needs one integer per edge:

```python
pair = base[csr.dst] * n_edge_types + csr.attr
```

The obvious way to collapse an `(n_edges, n_edge_attr)` row into that integer is `dense_rows`,
which already exists in `refine.py`. **That is wrong here.** `dense_rows` assigns ids from the
values present *in the batch it is given*. Fit and predict run on different batches, so a
predict batch that happens to contain no triple bonds would renumber the alphabet, and every
class id would silently mean something different from what was fitted. That is precisely the
failure mode `schema_version` (§9.2) exists to prevent, arriving through a different door.

The collapse must therefore be a pure function of the **config**, which is shared between fit
and predict by construction. Mixed-radix, computed once in `refine` before the level loop:

```python
edge_code = np.zeros(csr.attr.shape[0], np.int64)
for j, r in enumerate(config.edge_radices):
    edge_code = edge_code * r + csr.attr[:, j]
```

The per-level hot path is then unchanged from today:

```python
pair = base[csr.dst] * n_edge_types + edge_code
```

Two consequences worth stating explicitly:

- **The empty case needs no special-casing.** `edge_attributes == ()` gives `edge_radices == ()`,
  so `n_edge_types == 1` and `edge_code` is all zeros, making `pair == base[csr.dst]`. The
  refinement becomes pure topology, which is the intended control arm.
- **The range check improves.** `refine.py:93` currently checks a single flat code against
  `[0, n_edge_types)` and reports a bare integer. It becomes a per-column check of
  `csr.attr[:, j]` against `edge_radices[j]`, and the error can name the offending attribute and
  value.

Alphabet growth is not a practical concern: bond vocabularies are tiny (bond type ~5,
conjugated 2, ring membership 2, ring size ~6), so even all five configured together stays in
the low hundreds.

## 4. `NodeBatch`

`edge_attr: (n_edges,)` becomes `edge_attrs: (n_edges, n_edge_attr)`, taking the name and shape
`design.md` §11.1 already specifies.

| Site | Change |
|---|---|
| `_check_shapes` | assert 2-D with `shape[0] == n_edges`; a 1-D array must raise, not broadcast |
| `__getitem__` | `self.edge_attrs[keep]` — already correct for 2-D |
| `csr()` | `CSRLayout.attr = self.edge_attrs[order]`, now 2-D |
| `concat_batches` | `np.concatenate(..., axis=0)` |
| `_check_edges` | untouched — it never reads attributes |

An empty schema produces a well-formed `(n_edges, 0)` array, not `None`.

## 5. Adapter

`build_codes` gains an `edge_attributes` parameter and discovers edge vocabularies from the
corpus through the same path node attributes use: `_observed_values`-style dispatch across the
two edge registries, sorted union, dense codes from 0, one reserved code above the maximum for
unseen values, and the same `n_jobs` chunking (the union is commutative and associative, so
chunk count does not affect the numbering).

This removes the hardcoded table and fixes the latent bug it caused: an unusual bond type no
longer collides silently onto code `0`.

`_from_rdkit_sequential` fills the `(n_edges, n_edge_attr)` array. `_MOL_BOND_ATTRS` providers
are evaluated once per molecule in the outer loop, exactly as `mol_vals` already does for atoms,
and indexed by **raw** bond index so that a permuted `node_order` stays correct. Both directions
of an edge receive the same attribute row.

## 6. Serialization

`FORMAT_VERSION` 2 → 3. `SieveModel.load` refuses a v2 file loudly rather than guessing, per
`design.md` §9.2. `model.py` persists `edge_attributes` alongside the nested `edge_codes`.

`schema_version`'s payload gains `edge_attributes` and nests `edge_codes`. Any model fitted
before this change is therefore unmergeable with any fitted after — which is correct, since
their edge alphabets genuinely mean different things.

## 7. Testing

- Registry membership and separation: an edge attribute name resolves in exactly one of
  `_BOND_ATTRS` / `_MOL_BOND_ATTRS`, and edge names do not leak into `_ATTRS`.
- `edge_attributes=()` yields pure-topology refinement, **and** demonstrably different classes
  from `("bond_type",)` on a corpus where bond order matters.
- An unseen bond type receives the reserved unknown code rather than colliding with a seen one.
- Two molecules differing only in conjugation fall in different classes under
  `("bond_type", "conjugated")` and the same class under `("bond_type",)`.
- **Fit on one batch, predict on another whose edge vocabulary is a strict subset, and assert
  class ids agree.** This is the test that fails under the `dense_rows` collapse rejected in §3;
  it is the reason that section exists and must not be dropped from the plan.
- `bond_min_ring_size` on a fused system reports the smaller ring; `bond_num_ring_memberships`
  reports 2 for a fusion bond.
- `__getitem__` and `concat_batches` preserve 2-D `edge_attrs`, including the `(n_edges, 0)`
  case.
- Loading a `format_version: 2` model raises.
- Existing `n_jobs` determinism tests extended to cover edge-attribute discovery.

## 8. `design.md`

§9.2, §10.1 and §11.1 describe this feature as though it exists; they are updated to match what
is built. Added: the flat-set decision and its rationale, and `n_edge_types` as a product over
`edge_radices` with §3's reasoning for why the collapse is config-determined.

## Out of scope

- Bond `stereo` (E/Z) — follow-up, no interface change needed.
- Graded edge attribute levels — would restructure the refinement chain.
- Any migration path for `format_version: 2` models.
