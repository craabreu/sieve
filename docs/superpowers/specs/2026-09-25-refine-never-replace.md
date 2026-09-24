# Refine, Never Replace: How Sieve Takes On New Information

**Date:** 2026-09-25
**Status:** design note (principle); no code change by itself
**Generalises:** `2026-09-23-stereo-refines-the-blind-class-design.md` (design D) and
`2026-09-24-tetrahedral-handedness-design.md`, whose mechanics are stated here once for
any property.
**Supersedes, as intended route:** `neighbor_depth` (design.md 3.6), which becomes one
instance of §4 below.

---

## 1. The principle

The element-only WL chain is the backbone. It refines on elements over bare
connectivity, with no edge attributes, so it is local and resonance-invariant by
construction. Any further property of an atom or a bond enters as a **refinement of a
backbone class at the same radius**, never as a replacement for it.

A property that replaces the backbone changes every class it touches, so a weak or noisy
property degrades the whole chain. Study C measured this: hybridization and aromaticity
fed through WL cost +3.95% RMSE. A property that refines the backbone can only change the
answers of atoms whose refined class is consulted. Everything else keeps the backbone's
answer exactly.

## 2. What refinement guarantees

Each property ("track") adds an **aware** row per atom beside the **blind** (backbone)
row. Both are deduplicated into one vocabulary per level (design D §3):

- **The incumbent is embedded exactly.** Blind classes and their counts, means and
  variances are bit-identical to the backbone fit. Continuation and τ² count blind
  children only, so every blind estimate is the backbone's.
- **Backoff runs within a radius before across radii.** Prediction walks the blind chain
  to the backbone's matched radius *k*\*, then tries the aware class at *k*\*. A missing
  refinement costs the property, not a whole radius.
- **Statistics stay additive, and merge stays a monoid.** Refinements are more classes in
  the same vocabulary, linked by `kind` and `blind_of` (and `mirror_of` where a quotient
  applies).
- **An atom no refinement reaches is untouched.** Its aware and blind rows coincide.

What refinement does **not** guarantee is that a consulted aware class answers better
than its blind class. That is an estimation question (§6), and it is where every track
succeeds or fails.

## 3. The contract a track must state

A track is admissible only if its specification answers all of these:

1. **Carrier.** Where the code enters the aware row: a digit of the edge code (both
   stereo tracks), or the atom's own row.
2. **Honest radius.** The first round at which the code is a fact about the atom's
   radius-*k* environment. cis/trans reads the fingerprint at *k* − 2, because the far
   substituent is two bonds away. Handedness reads it at *k* − 1, because its ranked
   neighbours are one bond away.
3. **Ordering source.** Anything the code ranks must be ranked by the stereo-blind
   content fingerprint, fixed width and batch-invariant (PR #39), never by class ids,
   which are batch-local.
4. **Quotient.** Any symmetry the target has and the code breaks. Handedness breaks
   mirror symmetry, which MBIS charges have, so it needs the M row set and pooling
   through `mirror_of`. cis/trans survives reflection and needs none.
5. **Informativeness.** When an aware row carries nothing the quotient keeps, its class
   must not answer. Under the mirror quotient, a single centre's sign is erased, so the
   aware class is consulted only where two or more centres or a cis/trans code have
   reached the row (`stereo.advance_reach`).
6. **Locality and invariance of the input itself.** The property must be computable from
   a bounded neighbourhood, and invariant to whatever the backbone is invariant to
   (atom numbering, batch composition, resonance form). CIP descriptors fail locality.
   Aromaticity fails locality (a Hückel test over a whole fused ring system) and
   resonance invariance. Hybridization fails resonance invariance.

## 4. Neighbour visibility, and what replaces `neighbor_depth`

Design D's aware row takes its parent and neighbours from aware classes, so a refinement
propagates. A track may instead **blind its neighbours**: the aware row carries the
atom's own property, and the neighbours' backbone (element-only) classes.

That is what `neighbor_depth` does with a coarse element chain and `LEVEL_WL_PAIR`
levels. As a track, it needs none of that machinery: no second level kind, no
`neighbor_source`, no special cases in `merge._translate`, `predict` or
`truncate_model`, and no conflict with the stereo tracks. It also gains the
guarantees of §2, which `neighbor_depth` never had.

Study C's only winning arm (element + hybridization + aromaticity, WL on element,
−0.53% RMSE, 25/25) is this construction. It is not adopted, because its attributes fail
§3.6, not because of the mechanism. `neighbor_depth` should be removed. An atom-level
property that passes §3.6, if one is ever wanted, should be built as a
neighbour-blinded track.

## 5. Several tracks: chain or lattice

With several tracks, blinding one at a time gives intermediate classes. Two tracks give
a lattice: both, cis/trans only, handedness only, neither.

- **The current choice** is two chains, aware (all tracks) and blind (none). It is the
  simplest, and its cost is confined to atoms reached by more than one track. There, a
  missing all-track class falls back to the backbone rather than to a one-track class.
  Measured: 0.51% of atoms are within two bonds of both an E/Z bond and a tetrahedral
  centre, which is 15.4% of the E/Z-near atoms. Study E's `near_ez2` check shows no
  measurable cost.
- **A fixed blinding order** (a chain of chains) embeds each earlier model exactly, at
  one extra row set per track.
- **A lattice** backs off over subsets of tracks. The closest precedent is factored
  language models with generalised parallel backoff (Bilmes and Kirchhoff, 2003), which
  back off over subsets of word factors rather than one fixed history. It sits beside
  the Kneser–Ney connection Sieve already draws. *Citation to be verified before use.*

Adopt more than two chains only when a measured overlap cost justifies it.

## 6. The failure mode: thin aware classes

Every track fails in the same way when its signal is weak. The aware class answers with
less support than its blind class, and the model trusts it too much.

Study E measured this at depth 5 on one fold. Among aware-only classes, the median
support is one training structure at levels 3–5 (68% singletons among the τ² group
members at level 5). Yet empirical Bayes gives them weights of 0.64–0.86, because
τ²_aware is 6–7 times the atom noise. Most of that spread is between single training
structures, which is variation beyond the radius that does not transfer to new
molecules, not the property's signal. cis/trans shows a τ²_aware of similar size but
also carries a real E/Z effect, and its gain relies on those low-support classes: a
minimum support of 10 for aware answers would give up more than half of Study D's gain.

Two consequences:

- **A support threshold is not the fix.** It helps a weak track and hurts a strong one.
- **The fix belongs in the framework:** in how τ²_aware is estimated, for example by
  separating within-structure from between-structure spread, or in how an aware class is
  shrunk toward its blind counterpart. Solved once, it serves every track. A smaller,
  separate defect: the noise term uses each class's own `msd/N`, which is zero for a
  one-atom class. The level-pooled σ̄²/N removes about 10% of τ²_aware. The incumbent's
  `sibling_variance` has the same form.

## 7. Evidence so far

| track | construction | result near the property | corpus-wide |
|---|---|---|---|
| cis/trans (Study D) | trit on the double bond, *k* − 2 | −2.53 to −2.66% RMSE, 25/25 | −0.22 to −0.24% |
| handedness (Study E) | trit on centre out-edges, *k* − 1, mirror quotient | +0.26 to +0.45%, 0/25 | +0.08 to +0.15% |
| handedness + informativeness rule (Study E rerun) | as above, lone centres blind | RMSE +0.06 to +0.07%, 0/25; MAE −0.02 to −0.04%, 21/25 | RMSE +0.02%, 0/25 |
| own hyb + arom, blinded neighbours (Study C) | `neighbor_depth` | — | −0.53%, 25/25 (continuation) |

Ranges span the two estimators (continuation, and continuation with empirical Bayes).

cis/trans is the positive case, handedness the instructive negative. With the informativeness rule, handedness moves the error distribution rather than its size: MAE improves slightly while RMSE worsens slightly, the signature of §6, where most atoms gain a little and a few thin classes cost a lot. Near an E/Z bond, under empirical Bayes, it helps (−0.05%, 25/25). The one-chain
simplification had no measurable cost in either study.

## 8. Open items

1. The estimation fix of §6, tested on both stereo tracks at once.
2. A pruning pass before shipping: remove `neighbor_depth`, `continuation_recursive`
   (indistinguishable from `continuation`), `shrinkage_weight="diversity"` (measured
   loss) and the tetrahedral track (no net gain). Tag the commit before pruning so every
   study stays reproducible.
3. Whether §1 becomes the organising idea of the paper's discussion of extending Sieve.
