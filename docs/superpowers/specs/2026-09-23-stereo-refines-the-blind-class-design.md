# Stereo Refines the Stereo-Blind Class

**Date:** 2026-09-23
**Status:** design, awaiting review; not implemented
**Revises:** `2026-09-22-cis-trans-featurisation-design.md`. §5.4 there (the fold of the
stereo code into every WL level) is replaced by §3–§6 here. §3–§5.3 there (the batch
array, the adapter, the content rank, the resolution rule, the radius rule) are kept
unchanged and are what this document builds on.
**Scope:** the cis/trans track. §9 says why the construction carries over to the
tetrahedral one.

---

## 1. Why the fused construction fails

In the fused construction the cis/trans trit rides on the double bond's edge code in
every WL round from *k* = 2, so each class at radius *k* is the stereo-aware class, and
there is one chain. `_search` walks that chain and stops at the first miss. When the
stereo-aware class at radius *k* has no training atom, the atom is answered at radius
*k* − 1, although the stereo-blind class at radius *k*, the one the incumbent would have
used, existed and was supported. It no longer exists anywhere in the fused model.

Measured on the rebuilt store, repeat 0, radius 5, element continuation without
shrinkage, five folds:

| | incumbent | fused cis/trans |
|---|---|---|
| RMSE, all atoms | 0.019935 | 0.019973 |
| MAE, all atoms | 0.010943 | 0.010938 |
| RMSE within 2 bonds of a stereogenic C=C/C=N | 0.032820 | 0.033474 |

MAE improves in four folds of five, RMSE worsens: the typical atom gains and a tail
loses. The tail is concentrated in push–pull ylidenes. In
`CCN1/C(=C2/OC(=S)N(C)C2=O)Sc2ccccc21`, the carbon of the exocyclic bond matches a
radius-2 blind class of support 1 whose mean is −0.096 e against a reference of −0.137 e;
the fused model splits that class by geometry, finds no training atom of this geometry,
and answers from radius 1 (support 158,353, mean +0.347 e). The error goes from 0.04 to
0.48 e. Fragmentation as such is small: classes grow by 5.4% at radius 2 and under 2% at
radius 4–5, and the fraction of near-bond atoms matched to radius 5 falls from 41.6% to
39.3%. The code itself is canonical: 300 molecules renumbered three times each change
no atom's class at any radius.

Note on the figures above: they were measured before the fix of §4.1. Fold 0's held-out
batch contains a molecule with a five-coordinate atom, so its fingerprints were computed
at a different width; recomputed in batches without one, fold 0's near-bond RMSE is
0.032745, not 0.034721. The training shards carried the same defect, so the fused models'
codes were partly inconsistent across shards. The defect touches the six molecules with a
degree-5 atom and whichever batches contain them, and the §2 emulation improves every
fold, including the four whose held-out batches had none; the figures are therefore kept
as measured rather than repeated, and Study D (§11) measures the fixed track.

## 2. What replaces it

The stereo-aware class is a refinement of the stereo-blind class *at the same radius*,
and it is used only when it exists. An atom is always answered at the incumbent's
radius, from the incumbent's class or from a refinement of it.

Emulated post hoc on the two fitted fold models (§1's setting), by answering an atom from
the fused model only when a stereo bond is within reach at the incumbent's matched
radius *k*\* and the fused model matched at *k*\* or deeper:

| metric | incumbent | emulated | change | 95% interval, paired, 4 df |
|---|---|---|---|---|
| RMSE, all atoms | 0.019935 | 0.019884 | −5.0 × 10⁻⁵ | [−5.5, −4.6] × 10⁻⁵ |
| MAE, all atoms | 0.010943 | 0.010919 | −2.4 × 10⁻⁵ | [−2.6, −2.2] × 10⁻⁵ |
| RMSE, conformers with E/Z | 0.021822 | 0.021426 | −4.0 × 10⁻⁴ | [−4.3, −3.6] × 10⁻⁴ |
| RMSE, within 2 bonds | 0.032820 | 0.031881 | −9.4 × 10⁻⁴ | [−10.2, −8.6] × 10⁻⁴ |
| RMSE, within 1 bond | 0.038845 | 0.037359 | −1.5 × 10⁻³ | [−1.61, −1.36] × 10⁻³ |

Every fold improves on every metric. The emulation lacks §5's estimator for aware
classes and approximates "within reach" by distance, so these are the target the
implementation must approach, not its specification.

## 3. One vocabulary per level, with class kinds

Each level holds a single vocabulary containing two kinds of classes:

- **blind classes**, computed exactly as a stereo-blind model computes them;
- **aware classes**, computed as the fused construction computes them (the kept §5.1–§5.3
  of the revised document: content rank, resolution, the *k* − 2 radius rule).

A class carries a two-bit **kind**: `BLIND`, `AWARE`, or both. Classes are deduplicated
by signature as today, and an aware class whose signature equals a blind class's is the
same class, flagged both. That is the common case: an atom with no stereo bond within
reach at radius *k* has the same aware and blind signature, since its trit is 0 and its
parent and neighbour ids are themselves both-flagged. Aware-only classes exist only where
stereo changes the environment, so the vocabulary grows only there, and a model fitted
with `stereo=()` has no aware-only class and is today's model.

Each level also stores **`blind_of`**, mapping every class to the blind class of the same
atom at the same radius. It is the identity on both-flagged classes.

## 4. Fitting

For each level *k*, `refine` produces two label arrays per atom:

- the **aware** label: the fused round, parent and neighbours from the aware labels at
  *k* − 1, the trit on double-bond edges;
- the **blind** label: the same round with the trit forced to 0 and parent and
  neighbours from the **blind** labels at *k* − 1.

The blind label must be computed recursively in this way. Dropping only the current
round's trit would leave stereo information inherited through the ids of level *k* − 1,
and the "blind" class would not be the incumbent's.

Encoding: blind and aware signatures share `full = edge_code * stereo_radix +
stereo_code`, with `stereo_code = 0` for every blind edge. The two therefore coincide
exactly when they should, and the modulus is unchanged, so `merge._translate`'s `divmod`
is untouched.

Each atom's sufficient statistics accrue to its aware class and, when different, to its
blind class. A blind class's statistics are therefore exactly those a stereo-blind fit
would compute, and the model need never be re-fitted blind.

### 4.1 The content rank is batch-invariant (fixed on `main`, PR #39)

Before PR #39, `stereo.content_ranks` built `fp_j` by mixing the atom's padded, sorted
neighbour row as `width = max(csr.max_deg, 1)` separate columns. `max_deg` is the largest degree anywhere in
the batch, so the number of values mixed into every fingerprint depends on the batch, and
its docstring's claim that adding molecules cannot change a fingerprint fails whenever the
added molecule raises that maximum. Fingerprint order can then change, the winning
substituent at an end can flip, and the code flips with it.

Measured: `QMUGS500_27594`'s prediction changes by 0.055 e when a single molecule with a
five-coordinate phosphorus, `COP12(OC)NC(=O)O[C@]1(C(F)(F)F)c1ccccc1O2`, joins its batch.
Six molecules of the store have a degree-5 atom and none a higher one, which is why the
defect was confined to fold 0 and escaped the batch-composition test, whose companions all
had degree 4.

The fix makes the width a property of the method, not of the batch: every neighbour row
pads to `CONTENT_RANK_WIDTH = 8`, and `content_ranks` refuses a batch with an atom of
higher degree rather than truncating it. An order-independent aggregate, the degree mixed
with a wrapping sum of the neighbours' mixed hashes, would remove the cap but changes the
mixer; the fixed width is the smaller change and keeps the kept §5.1 of the revised spec
otherwise intact. The fused `element-ct-eb` shards and cache entries fitted before the fix
have been deleted. This design inherits the fixed content rank unchanged.

## 5. Estimation

**Continuation and τ² are computed over blind children only.** A class's children are the
next-level classes flagged `BLIND` whose parent is that class. Where the chains diverge,
one training atom sits in two children of a common parent, its aware class and its blind
class; counting both would count it twice. Restricting to blind children makes every
blind class's continuation estimate, and every τ² on the backoff path, identical to the
incumbent's. `continuation._child_of_level` already takes a class's children from one
designated level; it gains the kind filter.

**An aware-only class** has no children on the path, so its base estimate is its pooled
mean. Under a shrinkage rule it is shrunk toward the estimate of its blind counterpart
`blind_of[c]`, with weight `m_c / (m_c + α_k)`. Under `empirical_bayes`, α_k comes from
a per-level τ²_{aware,k}: the one-way ANOVA of aware-class means about their blind
counterpart, over blind classes that have at least two aware refinements, debiased for
sampling noise exactly as `sibling_variance` does for the backoff path. Without a
shrinkage rule, the aware class answers with its pooled mean.

## 6. Prediction

At each radius, from the deepest down: the query's aware class if it exists in the
model, then its blind class, then radius *k* − 1.

Because a blind class exists wherever one of its aware refinements does, the first radius
at which anything matches is the incumbent's matched radius *k*\*. The rule is
therefore implemented as the existing bottom-up walk over **blind** labels, which yields
*k*\* and the incumbent's estimate, followed by one lookup of the query's aware class at
*k*\*. If that class exists, the atom is answered by §5's aware estimate; otherwise by the
incumbent's. The aware lookup can only succeed if the query's aware parent and neighbour
classes at *k*\* − 1 also exist, since its signature refers to them; a miss anywhere
below simply falls back to blind.

`Predictions.matched_level` keeps meaning the position on the blind path. A new
boolean `stereo_refined` reports whether the aware class answered.

**Guarantee.** An atom with no stereo bond in reach, or whose aware class at *k*\* is
absent, receives exactly the incumbent's prediction.

## 7. Configuration and schema

`SieveConfig.stereo` keeps its meaning. `schema_version` gains a construction marker,
`"stereo_construction": "refines_blind"`, whenever `stereo` is non-empty. Fused models,
which anyone can still fit from `main`, carry no marker and hash differently, so
`check_mergeable` refuses to merge them with models built under this design; with
`stereo=()` the digest is byte-identical to today's.

The Level arrays gain `kind` (uint8) and `blind_of` (int64). Both are rebuilt by
`dedupe` and remapped by `merge` like `parent`; `kind` combines under merge by bitwise
OR.

## 8. Test plan

Written before the code.

1. **Blind equivalence, the test that makes the design trustworthy.** Fit a corpus with
   and without `stereo`. The blind-flagged classes of the first equal the classes of the
   second, level by level, up to relabelling: signatures, counts, sums, mean squared
   deviations, parents. Continuation estimates and τ² agree. `predict` agrees exactly for
   every atom whose `stereo_refined` is false.
2. **Fallback exactness.** Train on one E/Z isomer only and query the other: the prediction
   equals the stereo-blind model's. The ylidene of §1 is a ready fixture.
3. **Refinement.** Train on both isomers of 2-butene: they separate at *k* = 2 and not
   before (the kept radius rule), and each is answered by its aware class.
4. **Recursive blinding.** An atom one bond from a stereogenic bond has a blind label at
   *k* = 3 equal to the stereo-blind model's, which fails if only the current round's
   trit is dropped.
5. **Batch invariance, with a degree above the batch's own.** PR #39 added the
   fingerprint and code forms (a path beside a degree-5 star; the adapter's codes beside
   the five-coordinate phosphorus; a degree above `CONTENT_RANK_WIDTH` refused). This
   design adds the prediction form, `predict` identical alone and beside that molecule,
   and the shard form: two shards that share a molecule give it the same aware and blind
   labels although only one of them contains a higher-degree molecule.
6. **Kept from the revised spec:** mirror invariance; the merge monoid
   `fit(A ∪ B) == merge(fit(A), fit(B))` over 2 and 3 shards, now also on `kind` and
   `blind_of`; batch-composition invariance; renumbering invariance; the adapter tests.
7. **Schema.** A fused model, a model under this design and a stereo-blind model have
   three distinct digests, and `check_mergeable` refuses each pair.

## 9. The tetrahedral track

The construction carries over. Blinding the tetrahedral trit yields the blind class in
the same way, and the kind flags generalise to one bit per track. The mirror quotient
the tetrahedral half needs acts on aware classes only: it negates the chirality trit and
leaves the cis/trans trit fixed, and blind classes are fixed by it, so blinding commutes
with the quotient. The tetrahedral spec should state this explicitly and extend §8.1's
equivalence test to both tracks at once.

## 10. Open items

1. **No re-measurement before Study D.** §1's proof of principle and §2's emulation ran
   before the §4.1 fix and are not repeated (see the note in §1); Study D measures the
   fixed track directly.
2. **Storage.** The aware-only fraction per level should be measured on the full training
   split once implemented; §3 predicts it tracks the ~12% of atoms in conformers with an
   E/Z bond, shrinking with radius as reach grows only as *k* − 2.
3. **The subset metric.** Study D's primary metric, RMSE on atoms within two bonds of a
   stereogenic double bond, belongs in the CV scorer rather than in post-hoc scripts, and
   predictions are saved today only at each study's selected depth.
4. **The floor-derived "ceilings" are not bounds.** They measure only the scatter between
   stereoisomers of one structure present in the corpus. §2's emulation exceeds them
   because a stereo-aware model also transfers what it learns about cis and trans
   environments to held-out molecules whose stereoisomers never appear. Study D should
   report effect sizes against the incumbent, not against those figures.

## 11. Validation: Study D

Two arms, element continuation with and without the cis/trans track under this design, at
radius 5, five repeats of five folds, paired. Primary metric: RMSE within two bonds of a
stereogenic double bond. Secondary: corpus-wide RMSE and MAE, as the no-regression check.
The target is §2's emulated effect, measured before the §4.1 fix; a result materially
below it would mean §5's estimator or the reach rule costs more than the emulation
assumed.
