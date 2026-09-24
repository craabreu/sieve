# Cis/Trans Geometry as a Refinement-Time Edge Code

**Date:** 2026-09-22
**Status:** implemented, then revised: §5.4 (the fold of the stereo code into every WL
level) is replaced by `2026-09-23-stereo-refines-the-blind-class-design.md` §3–§6.
**Scope:** the E/Z half of stereo-aware featurisation, and only that. The tetrahedral
half is deliberately deferred; see §9.
**Supersedes, in part:** `cis-trans-geometry.md`, which states the construction without
the interface. Where this document and that one differ, §7 lists the differences and
this one wins.
**Related:** `stereochemistry.md` holds the evidence and the rejected alternatives.
`docs/superpowers/specs/2026-09-18-radius-resolved-chirality-design.md` is the earlier
chirality proposal, several of whose claims do not hold (`stereochemistry.md` §8).

---

## 1. What this builds, and why this half first

A double bond's two sp2 atoms gain one distinction: whether the substituents that rank
highest at each end lie on the same side. The code appears at the radius where those
substituents first become distinguishable and is absent everywhere else.

**Why E/Z before handedness**, against the order `stereochemistry.md` §9 assumes. E/Z
survives reflection, so this half needs no mirror quotient, no post-fit stage over the
vocabulary, and no change to `merge.py`, `dedupe.py` or `predict.py`. It is strictly the
smaller increment, it shares the content-rank machinery the tetrahedral half will need,
and on the rebuilt store it is worth nearly as much: the stereo-sensitive subset divides
967 groups / 3,734 conformers (40.3%) E/Z-only against 884 / 4,806 (51.9%)
tetrahedral-only, with 125 / 717 differing in both.

**Why build at all, given the measured headroom is small.** Not on accuracy grounds.
`stereochemistry.md` §2.1 is the argument: `collapse_key` deliberately separates
diastereomers and E/Z isomers because they are different chemistry, and every arm is
then fitted on those separated targets with features structurally unable to express the
separation. That is the fit contradicting its own premise, and it holds whatever the
RMSE says. The two free non-local bound arms may run in parallel as evidence; they do
not gate this work.

## 2. Decisions taken

| decision | choice | why |
|---|---|---|
| carrier | ternary edge code on the double bond | rides the existing `divmod`; a trailing signature column collides with `_widen`'s left-padding (`stereochemistry.md` §6.5) |
| ordering source | content-rank Merkle fingerprint | the only measured-zero ordering; dense ids are a coin flip at 47.8% (`§6.5.1`) |
| what is stored | nothing | the trit is consumed into the pair encoding and discarded, so no value can go stale when `merge` re-sorts |
| relation source | `Chem.FindPotentialStereo` | one local API on both entry paths; see §4 |
| radius rule | no code before *k* = 2 | see §5 |
| interface | an optional `NodeBatch` array | see §3 |

## 3. `NodeBatch.stereo_bonds`

A new optional field, following the precedent of the existing optional `elements`:

```python
stereo_bonds: np.ndarray | None = None   # (n_stereo, 7) int64
```

Each row is `[a, b, a1, a2, b1, b2, cis]`: the two sp2 atoms, `a`'s two controlling
substituents, `b`'s two, and whether `a1` and `b1` lie on the same side. Absent slots
(an imine end, a single substituent) hold `-1`. Indices are batch-global. `None` for
every batch built today, so nothing existing changes and `_check_edges` never sees it.

Naming both substituents at each end, rather than only RDKit's two reference atoms, is
what makes the per-round work a lookup: once the winner at each end is known, the code
follows without reconstructing the pairing.

**Validated in `__post_init__`**, beside the existing checks, all vectorised: atom
indices in range; `a`–`b` present as an edge in both directions; each controlling atom
adjacent to the end it belongs to; `-1` only in the `a2`/`b2` slots; `cis ∈ {0, 1}`.

Note that `cis` here is the stored *relation* between `a1` and `b1`, a boolean. It is
not the emitted code, which is the ternary `{none, cis, trans}` of §5.4 and depends on
which substituents win at the current radius.
Every one of these is a corpus bug that would otherwise produce a plausible wrong code
rather than an error.

## 4. The adapter reads annotations, never coordinates

`experiments/experiments/prepare_dash.py` is the single place 3D coordinates are
consulted, and it writes what it perceives into the stored `Mol`. The adapter does not
repeat that work. Three reasons, in decreasing order of how much they would cost to get
wrong: `main.tex`'s claim that no predictor sees coordinates depends on it; perceiving
at featurisation time would make features conformer-dependent and break the
"function of the molecular graph alone" premise the whole method rests on; and the
store's annotations are already the curated ones.

**Use `Chem.FindPotentialStereo`, not `bond.GetStereo()`.** The two entry paths represent
the same fact differently, *measured 2026-09-22*:

| path | `GetStereo()` |
|---|---|
| the store, after `prepare_dash` | `STEREOCIS`/`STEREOTRANS` — local |
| `from_smiles` | `STEREOE`/`STEREOZ` — **CIP-derived** |

Reading `GetStereo()`, as `cis-trans-geometry.md` §2.1 specifies, therefore works on the
store and silently imports unbounded-depth CIP information through `from_smiles` — the
exact non-locality this construction exists to avoid, entering on the path every unit
test uses. `FindPotentialStereo` returns `Bond_Cis`/`Bond_Trans` relative to its
`controllingAtoms` on both paths, never E/Z, and it is unaffected by the explicit
hydrogens that blind RDKit's *atom* stereo perception.

A bond whose `specified` is `Unspecified` yields no row and therefore no code. On the
rebuilt store that is **190 of 7,213 potentially stereogenic double bonds (2.6%)**;
`STEREOANY`, which was 3.44% of records before `29cc341`, is now entirely absent. The
190 are not yet characterised — §10.

Ends carrying more than two substituents are skipped. Cumulenes, whose stereochemistry
is axial rather than per-bond, are out of scope.

## 5. The per-round computation

### 5.1 The content rank

`fp_0(v) = mix(node_attrs row of v)`, then per level, mirroring the WL round's own shape:

```
nb   = mix(fp_prev[csr.dst], edge_code)
pad  = full((n, max_deg), SENTINEL); pad[csr.src, csr.slot] = nb; pad.sort(axis=1)
fp   = mix(fp_prev, pad)
```

using `dedupe._row_keys`' mixer. A fingerprint `fp_j` is a function of the atom's radius-*j*
environment and nothing else, so two atoms with the same environment share it in any
batch, and adding molecules to the batch cannot change it. That is the property class
ids do not have: `dense_rows` numbers them by hashing the row, and its docstring states
that no caller may rely on that numbering.

**The fingerprint folds the static `edge_code`, never the stereo trit.** This is a
requirement, not an optimisation. A stereo-aware ordering would let the code reorder the
substituents it is ranking, so remapping or mirroring could change which substituent
wins, and the merge would disagree. It is a one-character mistake with no error and a
failure visible only as a handful of atoms differing after a merge; §8 makes it a test.

Cost: one extra pad-and-sort per level, roughly doubling a WL round, and only when
`config.stereo` is non-empty.

### 5.2 Resolving the code

Indices, since the rest of this section depends on them: *k* is the WL round, counted
from 1 over the WL levels only; `fp_j` is the fingerprint after *j* WL rounds, with
`fp_0` the one built from the attribute levels alone (radius 0). At WL round *k* the
code is read from `fp_j` with `j = k - 2`, which by §5.3 is never negative because no
code is emitted before *k* = 2.

Then, per listed bond:

1. At each end, take the controlling substituent with the larger `fp_j`.
2. If an end's two controlling substituents have equal `fp_j`, the code is `none` — the
   bond is not distinguishable at this radius and the feature defers.
3. Otherwise the code is `cis` as stored, flipped exactly when one winner — but not
   both — differs from the stored first controlling atom.

All of it is gathers and boolean algebra over the `(n_stereo, 7)` array. The two
directed edge positions of each listed bond are resolved once, before the round loop,
since the bond set never changes.

### 5.3 The radius rule: no code before *k* = 2

The gather reaches distance 2, so honesty needs `j = k - 2`, which does not exist for
*k* = 1. **The feature emits nothing at *k* = 1.**

This is not a conservative choice; it is what the chemistry is. The cis/trans relation is
a fact about a four-atom span, substituent–C=C–substituent. At radius 1 an sp2 atom's
environment is itself and its bonded neighbours, and the substituent on the far sp2 atom
is two bonds away, outside it. A radius-1 class carrying a cis/trans code would be
asserting something about an atom its own neighbourhood does not contain — the same
objection this document raises against `_CIPCode`, differing only in degree, and it
would break the property that a level-*k* class is a function of the radius-*k*
neighbourhood, which the depth sweep and the backoff chain both read.

Clamping to `max(k-2, 0)` instead, which the measurements in `cis-trans-geometry.md` §4
evidently did, does not buy reach. It buys latency: groups recorded there as separating
at radius 1 separate at radius 2 instead, and the cumulative curve is unchanged from
radius 2 onward.

Note the asymmetry this creates with the half that comes next: the tetrahedral code
gathers at distance 1, reads `fp_{k-1}`, and can therefore fire honestly from *k* = 1.

### 5.4 Emission and fold

`stereo_code` is 0 everywhere, 1 (`cis`) or 2 (`trans`) on the two directed halves of
each resolved bond, 3 reserved for the unknown. Then

```
full = edge_code * stereo_radix + stereo_code      # stereo_radix = prod(stereo_radices), 4 here
pair = base[csr.dst] * n_edge_types + full
```

One extra multiply-add in the expression `refine` already evaluates. Nothing else in the
round changes.

## 6. Configuration, and why the digest matters

`SieveConfig` gains `stereo: tuple[str, ...] = ()`, valid member `"cis_trans"` now and
`"tetrahedral"` later — a tuple rather than a boolean because the second track is coming.

Each track contributes a radix of 4: `none`, `cis`, `trans`, and the reserved unknown.
`edge_radices` keeps meaning the static adapter columns; `stereo_radices` joins it, with
`n_edge_types = prod(edge_radices) * prod(stereo_radices)`. The modulus stays constant
across levels and across shards, so `merge._translate`'s `divmod` and `predict`'s use of
it are untouched: `_translate` remaps only the label half and keeps the bond half opaque
before re-sorting.

**`stereo` enters the `schema_version` payload** as `list(self.stereo)`. This is not
bookkeeping. A code computed inside `refine` is otherwise invisible to the digest, so a
stereo model and a stereo-blind one would hash identically and `check_mergeable` would
let them merge — silently, producing a model whose classes mean two different things.
With `stereo=()` the digest is byte-identical to today's, so every fitted model on disk
stays valid.

`config.stereo` naming a track while `batch.stereo_bonds is None` raises, mirroring the
existing `node_attrs` width check. The converse is permitted and ignored: `from_rdkit`
should not have to know the config.

## 7. Differences from `cis-trans-geometry.md`

| that document | this design | why |
|---|---|---|
| read the relation from `bond.GetStereo()` | read it from `Chem.FindPotentialStereo` | `GetStereo()` returns CIP-derived E/Z on the `from_smiles` path (§4) |
| the code may fire at *k* = 1 (§4's radius table) | nothing before *k* = 2 | radius honesty (§5.3); the table's first row moves to radius 2 |
| `STEREOANY` bonds are a coverage limit | `STEREOANY` no longer exists in the store | measured after `29cc341`; the limit is now 190 `Unspecified` bonds |
| the relation is read from stored conformers (§7.2) | read from the graph, through the store's annotations | that path is now the only one |

`cis-trans-geometry.md`'s separation and cost figures — 291 of 449 groups, +0.26 pt in
low-support atoms at level 3 — were measured on the pre-rebuild store and are not
re-measured here. They are indicative, not current.

## 8. Test plan

Written before the code, grouped by what each would catch.

**Separation, hand-checkable.** `trans`/`cis`-2-butene separate, at *k* = 2 and not
before, which pins §5.3's rule and not merely the feature. The same for
`trans`/`cis`-1,2-dichloroethene. An identical pair never separates. 2-methyl-2-butene,
a non-stereogenic double bond, never separates.

**The invariance that defines this half.** 60 mirrored pairs, zero separated.

**The merge monoid, through the real machinery.** `fit(A ∪ B) == merge(fit(A), fit(B))`
on 40 E/Z pairs over 2 and 3 shards, comparing class counts per level *and* predictions,
using `sieve.fit`, `SieveModel.merge` and `predict` themselves rather than a
reimplementation. This is the protocol of `stereochemistry.md` §6.6.1, and the only one
that has caught the batch-locality failures.

**Batch-composition invariance.** The same molecule refined beside different companions
receives the same code, asserted directly on the emitted codes rather than inferred from
class counts. This is the test that fails if the fingerprint ever sees the stereo trit.

**Schema and merge safety.** `stereo=()` leaves the digest byte-identical;
`stereo=("cis_trans",)` changes it; `check_mergeable` refuses the two against each other.

**Adapter.** A SMILES-built `C/C=C/C` and the same molecule from the store yield the same
relation — the case where `GetStereo()` would import CIP. `Unspecified` yields no row.
Ends with more than two substituents are skipped. The `(n_stereo, 7)` validation rejects
an out-of-range index and a non-adjacent controlling atom.

## 9. Out of scope

The tetrahedral half, which needs the mirror quotient (`stereochemistry.md` §6.3), a
post-fit stage over the fitted vocabulary, and the parity convention of §4 there. It
reuses §5.1's fingerprint unchanged, which is the reason to build this half first.

Also out of scope: the two non-local bound arms, cumulenes, axial stereochemistry, and
any change to `collapse_key` or to the curation.

## 10. Open items

1. **The 190 `Unspecified` double bonds are uncharacterised.** Likely ring double bonds
   whose geometry is constrained rather than declared. Worth knowing before
   "2.6% of stereogenic bonds are outside the feature's reach" appears in a paper.
2. **`cis-trans-geometry.md`'s figures are pre-rebuild.** Separation, cost and the radius
   table should be re-measured on the rebuilt store, under §5.3's rule.
3. **The combined edge alphabet is not specified.** When the tetrahedral half lands, both
   tracks multiply `n_edge_types` by 4, and the mirror involution must negate the
   chirality trit while leaving this one fixed. That mapping belongs in the tetrahedral
   spec, not here, but it is a known gap between the two.
