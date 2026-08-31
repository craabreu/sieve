# Sieve — Design Update v1

**Status:** correction note, supersedes specific sections of `design.md` listed below
**Date:** 2026-08-19
**Scope:** brings §5.3, §7.2, §9.2, and §10.1 of `design.md` into line with the code as of
commit `fac7664` (branch `perf/batch-validation`). No new decisions are made here; this is a
historical record of what changed and why `design.md`'s text no longer matches the
implementation.
**Relationship to other documents:** `design.md` remains the primary reference. Where this note
and `design.md` disagree, this note wins for the sections it lists, until those sections are
folded back into `design.md` directly.

---

## 1. Row deduplication now uses a 64-bit key, not a void-view byte compare (supersedes §7.2)

**What §7.2 says.** Signature rows are deduplicated by viewing each row as one `np.void` scalar
and calling `np.unique` on the 1-D view, with `return_index=True` to get representative rows. The
section's table reports 0.27 s for this method against 0.70 s for `np.unique(..., axis=0)`.

**What the code does now** (`src/sieve/dedupe.py`):

1. Each row is mixed down to a single `uint64` — FNV-1a across the row's 8-byte lanes, then a
   splitmix64 finalizer to spread bits across the whole word (the accumulation loop alone mixes
   poorly at the low end of the word).
2. Rows are grouped by that key with `np.unique(keys, return_inverse=True)`, **without**
   `return_index` — representative rows are recovered by scattering `m` into `rows[inv]` instead,
   which skips the stable sort that `return_index` requires (~28% cheaper).
3. Because equal keys are only *evidence* of equal rows, not proof, the result is checked with one
   vectorized comparison (`rows[labels] == mat`). If it fails — a genuine key collision — the
   function falls back to the exact byte-view path described in old §7.2, unchanged.

```python
def dense_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.ascontiguousarray(mat)
    if m.dtype.itemsize != 8:
        return _dense_rows_exact(m)          # the key mixer reads 8-byte lanes
    labels, rows = _group(_row_keys(m), m)
    if not np.array_equal(rows[labels], m):
        return _dense_rows_exact(m)           # collision: fall back to exact byte compare
    return labels, rows
```

**Why:** sorting one `uint64` per row hits numpy's type-specialized sort path over an eighth of
the data that the void-row sort touches, and dedupe was already measured as the dominant cost in
§7.2 (51 ms of a ~55 ms per-level total). This is a further win on top of that section's original
2.6× finding, not a different algorithm — void-view byte comparison is still the correctness
fallback.

**Ids are no longer ordered by first occurrence.** §7.2's version numbered classes by
`return_index` order (first appearance in the row order). The keyed version numbers by ascending
*key*, which has no relationship to row content or appearance order. `dense_rows`'s contract is
now explicitly: `unique_rows[labels] == mat` holds, but the numbering itself is unspecified and no
caller may depend on it. (`merge.py`'s `_lookup_rows` builds its own ordering where it needs one —
see §2 below.)

**A consequence for §9.** §9's "vocab is stored as an `(n_classes, width)` array, row *i* is class
*i*" claim still holds — it says nothing about numbering order, so it needs no correction.

---

## 2. Merge no longer goes through a Python dict (supersedes §5.3)

**What §5.3 says.** `merge_level` builds a Python `dict[bytes, int]` vocabulary, iterates
`B.vocab.items()`, and looks up/inserts hash keys one at a time in Python:

```python
vocab = dict(A.vocab)
remap = np.empty(len(B.vocab), np.int32)
for h, bid in B.vocab.items():
    cid = vocab.get(h)
    ...
```

This was already in tension with §9's later, and settled, decision that the vocabulary is stored
as a plain array (not `dict[bytes, int]`) — §5.3 was never updated after §9 was written.

**What the code does now** (`src/sieve/merge.py`): merging is entirely array-based, with no
per-row Python loop.

1. `_translate` rewrites B's signature rows into A's id space for the previous level
   (`remap_prev`), handling the WL case (bond-encoded neighbor pairs) and out-of-vocabulary
   neighbors (a dedicated `_OOV_NEIGHBOR = -2` sentinel, distinct from the `-1` pad sentinel, so an
   OOV neighbor can never be silently mistaken for "one fewer neighbor").
2. `_widen` pads A's and B's signature rows to a common width, **left**-padding WL rows (not
   right-padding, which was the old, buggy-in-spirit behavior described nowhere in §5.3 — that
   section predates variable-width batches merging at all).
3. `_lookup_rows` finds each of B's (translated, widened) rows inside A's table using the same
   64-bit row key as `dense_rows` (§1 above): sort A's keys, binary-search each of B's keys into
   that order, and verify every claimed hit against the real row bytes. A repeated key *within*
   A's own table (impossible in a correctly deduplicated vocabulary unless two distinct rows
   collided) is detected up front and routes the whole call through the exact byte-compare
   fallback.
4. Rows of B not found in A are minted as new classes via `dense_rows` on the unmatched subset.
5. Moments are combined with the same weighted-average form §5.2 already specifies — that part of
   §5.3 is unchanged and correct.
6. The parent-agreement invariant (§5.3's third bullet, "assert rather than overwrite") is now a
   real `raise AssertionError`, not Python's `assert` — deliberately, since `assert` compiles away
   under `python -O` and this is a correctness invariant, not a debug aid.

**Why the array-based version replaced the dict version:** performance (avoiding a Python-level
loop over classes, which numbers in the hundreds of thousands to millions per §3.2) and consistency
with §9's array-only storage — the dict never should have coexisted with §9's decision.

**What did not change:** everything else in §5.3 — pinning A's ids, remapping only B's, the
no-branch handling of B-only classes, the ordering requirement between reading `delta` and
overwriting `mean`/`msd` — still holds and is still visible in the current code (`delta = b.mean -
mean[i]` precedes the update, exactly as prescribed).

---

## 3. `SieveConfig` fields (supersedes the pseudocode in §9.2 and §10.1)

**What §9.2 and §10.1 show:**

```python
edge_attributes: tuple[str, ...] = ("bond_type",)
```

and a stored JSON field `"n_levels": 6`.

**What the code has** (`src/sieve/config.py`):

```python
@dataclass(frozen=True)
class SieveConfig:
    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]
    attribute_codes: Mapping[str, Mapping[str, int]]   # not in §9.2/§10.1 at all
    edge_codes: Mapping[str, int]                       # replaces edge_attributes
    max_wl_depth: int
    neighbor_depth: int | None = None                   # was neighbor_schema; see §5 below
    n_min: int = 1
    alpha: float | None = None
    chunk_size: int | None = None

    @property
    def n_levels(self) -> int:                          # derived, not stored
        return len(self.attribute_levels) + self.max_wl_depth
```

Three differences:

- **`edge_codes` replaces `edge_attributes`.** The config no longer just names which edge
  attributes exist (`("bond_type",)`); it carries the actual name → integer code mapping, the same
  role `attribute_codes` plays for node attributes. This is what `n_bond` (`max(edge_codes.values())
  + 1`) and the WL encoding in `refine.py` (`pair = prev[dst] * n_bond + csr.attr`) are built on.
- **`attribute_codes` is a real field**, entirely absent from §9.2/§10.1's pseudocode. It is what
  lets an adapter (§11.2) map a category name to a dense code and lets `schema_version` (§9.2)
  actually cover "everything that changes what a class means" — without it, two configs with
  different node-attribute encodings but the same `attribute_levels` names would incorrectly be
  considered mergeable.
- **`n_levels` is a computed property, not stored state.** `SieveModel.save()` does not write an
  `n_levels` key; `load()` does not read one. §9.2's JSON sample listing it as a top-level field
  describes a shape the serializer has never produced. `n_bond` is likewise a derived property,
  also absent from both pseudocode blocks.

Both `edge_codes` and `attribute_codes` are part of `schema_version` (config.py `schema_version`
property), matching §9.2's stated intent that the digest covers "everything that affects what a
class means" — that intent is correct, only the field list illustrating it was wrong.

---

## 4. What was checked and found still accurate

For the record, since this note exists to separate stale text from current text: §1–§4, §6, §8,
§11.3, §11.4, and the merge *algebra* in §5.2 (as opposed to its §5.3 implementation sketch) were
spot-checked against `refine.py`, `level.py`, `predict.py`, and `merge.py` and match the current
code. Only the three items above were found to have drifted.

---

## 5. Coarse neighbor attributes implemented (§3.6, open question 6)

§3.6 measured coarsening neighbors to element-only on cosmobase (no targets) and left it
unadopted. The mechanism is now implemented, parametrized as `neighbor_depth: int | None` — an
index into `attribute_levels` rather than a separate named schema (`neighbor_schema` above), so
any prefix of the declared attribute order can be the coarse base, not just "element."
`SieveConfig.level_kinds`/`level_parents`/`neighbor_source`/`backoff_path` are the single source
of truth for the resulting level DAG; `refine`/`merge`/`predict` all read them rather than
re-deriving level shape from indices.

The mechanism was reproduced on real `charge_experiments` data (`dash-molecules-50k`,
`MBIScharge` targets, `minimum_support=5` held fixed): element-only neighbors raised mean matched
level from 4.144 to 4.301 and the share of atoms matching at depth ≥3 from 94.3% to 97.3% — the
same direction §3.6 found on cosmobase's coverage-only measurement. **The accuracy question item
8 actually asks — MAE at the matched class, coarse vs. full — is still open**; this only confirms
the mechanism does what it claims, not that the trade is worth it for any given target.
