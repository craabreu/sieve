# Tetrahedral Handedness in the Aware Chain

**Date:** 2026-09-24
**Status:** design, awaiting review; not implemented
**Builds on:** `2026-09-23-stereo-refines-the-blind-class-design.md` (design D), whose
aware and blind chains, union vocabulary, `blind_of`, estimation and prediction are kept
unchanged. That spec's §9 anticipated this track.
**Evidence:** `stereochemistry.md` on the unmerged `stereochemistry-findings` branch, §4
(local parity), §6.2–§6.6 (constructions measured on 2026-09-20), §3 (reachability).

---

## 1. What this adds

Design D's aware class carries the cis/trans trit. This track adds a second trit, the
handedness of a tetrahedral centre, **to the same aware class**. Blinding still removes
every stereo code at once. There are two chains, aware and blind, as today, plus a third
row set **M**, each atom's aware row in the enantiomer of its molecule, which makes the
model exactly mirror-invariant.

**The assumption it encodes.** Mirror invariance is correct only for an achiral
observable. MBIS partial charges are one; optical rotation would not be. Enabling
`"tetrahedral"` therefore implies the quotient, with no switch, because no target in this
repository needs one.

Configuration: `stereo=("cis_trans", "tetrahedral")`. A track list without
`"tetrahedral"` behaves exactly as today. `SieveConfig.__post_init__` normalises `stereo`
to `STEREO_TRACKS` order and rejects duplicates, so `("tetrahedral", "cis_trans")` and
`("cis_trans", "tetrahedral")` are one configuration with one digest; `("cis_trans",)`
keeps a digest byte-identical to `main`'s.

Simplicity is the design criterion. An intermediate chain, cis/trans-aware but
handedness-blind, was considered and dropped (§8).

## 2. The handedness code

### 2.1 Centres and the adapter

A centre is any atom RDKit tags `CHI_TETRAHEDRAL_CW` or `CHI_TETRAHEDRAL_CCW`, whatever
its element. Allenes, atropisomers, square-planar and other non-tetrahedral tags are out
of scope. An untagged atom is never a centre.

The adapter emits `NodeBatch.stereo_centres`, the counterpart of `stereo_bonds`: one row
per centre, `(centre, n0, n1, n2, n3, parity)` in batch-global indices.

- `n0..n3` are the neighbours **in the order the tag refers to**, read from
  `atom.GetBonds()` and written explicitly, so nothing relies on the CSR order matching
  the bond order.
- `parity` is +1 for CCW and −1 for CW.
- A centre with three graph neighbours has a virtual fourth, an implicit hydrogen or a
  lone pair, marked `n3 = -1`. Measured 2026-09-24 over five SMILES orderings of alanine:
  RDKit's tag reads exactly as if the implicit hydrogen were appended last, which is where
  `AddHs` puts it. Any fixed position would do: it flips every such sign alike, which the
  ranking and the mirror both respect.
- A tagged atom with fewer than three or more than four graph neighbours is not a centre
  and gets no code.

The store, measured 2026-09-24: 1,059,780 tagged atoms over 1,027,598 conformers, all
tetrahedral and all with explicit hydrogens. 1,047,477 are four-coordinate carbon; the
rest are P (6,044), three-coordinate S (5,208, sulfoxides and sulfinyl groups, where the
virtual neighbour is the lone pair), four-coordinate S (683), N (309) and B (59). None
has fewer than three or more than four neighbours.

No coordinates are read. Centres come from local tags, not from `FindPotentialStereo`,
whose stereogenicity test uses RDKit's whole-molecule ranking; a tagged atom that is not
stereogenic at some radius simply gets `none` there (§2.2).

**Validated in `NodeBatch.__post_init__`**, vectorised, beside `_check_stereo_bonds`:
indices in range; `-1` only in `n3`; every present `n_i` adjacent to the centre; the
number of present `n_i` equal to the centre's degree, so every graph neighbour is named;
each centre listed at most once; `parity ∈ {−1, +1}`. **Carried through
`NodeBatch.__getitem__` and `concat_batches`** as `stereo_bonds` is: filtered and
reindexed on slicing, offset and all-or-none on concat.

### 2.2 The code at WL round k

At round *k*, rank the centre's neighbours by the stereo-blind content fingerprint at
radius *k* − 1 (`stereo.content_ranks`, fixed-width since PR #39). A virtual hydrogen
takes the fixed value 0, which sorts first.

- If any two ranked values tie, the code is `none`: the centre is not resolvable at this
  radius. This also covers atoms that are tagged but not stereogenic.
- Otherwise the code is `parity × sgn(π)`, where π sorts the neighbours by fingerprint;
  `+1 → CODE_PLUS`, `−1 → CODE_MINUS`.

Radius *k* − 1 is honest: the ranked neighbours sit one bond away, so their radius-(*k*−1)
environments lie inside the centre's radius-*k* one. The code can therefore fire from
round 1, unlike cis/trans, whose far substituents need *k* − 2.

**Two radii per round.** With both tracks on, round *k* reads `fp_{k-1}` (tetrahedral) and
`fp_{k-2}` (cis/trans). `content_ranks` is a generator consumed strictly in order and is
asked today for `max(n_wl - 2, 0)` rounds. With the tetrahedral track it must yield
`n_wl - 1`, and `refine` keeps one fingerprint of history. Otherwise the cis/trans code
silently shifts by one radius whenever both tracks are on; test 1 (Embedding), which
compares predictions and not only partitions, is what catches it.

The fingerprint folds only the static edge code, never a stereo code (as for cis/trans).
So the ranking is identical in a molecule and its mirror image, and mirroring negates
exactly the handedness codes.

### 2.3 Where it goes

The code rides on **every out-edge of the centre**, folded into the edge code beside the
cis/trans trit: `full = (edge_code * STEREO_RADIX + ct_code) * STEREO_RADIX + tet_code`,
with `stereo_radices` gaining one `STEREO_RADIX` factor. `n_edge_types` grows by that
factor, and `merge._translate`'s `divmod` is unchanged. The in-edges carry the
neighbour's own code, so a centre's handedness enters its own row directly and its
neighbours' rows through its label one round later.

### 2.4 Known limits

- **Pseudo-asymmetric centres (r/s) never resolve.** Their two enantiomorphic ligands
  have equal stereo-blind fingerprints and tie.
- **Automorphism ties.** About 3.2% of the stereo-sensitive conformers are unreachable
  by any local scheme (`stereochemistry.md` §3, §7.3).
- **Declared tags only.** A centre the store leaves unassigned is invisible, as any
  unperceived E/Z bond is to the cis/trans track.

## 3. The mirror row set M

At each WL level, `refine` builds three rows per atom:

- **blind:** both trits 0, parent and neighbours from blind classes one level down
  (unchanged);
- **aware:** both trits, parent and neighbours from aware classes (unchanged, now with
  the handedness trit);
- **M:** the aware row of the atom's enantiomer. The handedness codes are negated, the
  cis/trans codes are kept, and parent and neighbours come from **M** classes one level
  down. At attribute levels, M is the label itself.

All three are deduplicated together into the level's one vocabulary. As in design D,
only rows that differ from the row below them are stacked: an M row differs from its
aware row only where a handedness code is in reach, and an aware row that equals its
blind row has an M row equal to both.

**Kinds and maps.** An M class is an aware class (`KIND_AWARE`); it never coincides with
a blind class, since `M(a) = b` would force `a = M(b) = b`. Each level gains
**`mirror_of`**, beside `blind_of`:

- the identity on blind classes and on achiral aware classes;
- `mirror_of[c] = M(c)` otherwise.

It is well defined for the inductive reason `blind_of` is, it is an involution, and it
commutes with blinding: `blind_of[M(c)] = blind_of[c]`.

**Rejected quotients**, for the record:

| alternative | why not |
|---|---|
| post-fit pooling of *c* and M(*c*) where both exist, predict falling back to M(*c*) | breaks the monoid: a pair split across shards is never pooled, and re-pooling after merge double-counts |
| the same, pooled at prediction time | sound, but needs a mirrored translation chain in `predict` and pooled views in shrinkage and continuation |
| half-hits, weight ½ per hand | same means, half the support: chiral classes shrink harder, fall below `minimum_support` early, and `count` turns float |
| fitting on a materialised mirror batch (`sieve-openff`'s 2026-09-24 spec) | the same classes and statistics as M, but refines every chiral molecule twice and still needs `mirror_of` for τ² (§4) |
| mirror-invariant at source, via products of signs | an atom that is not itself a centre has no reference sign |

## 4. Statistics

- **Blind classes** reduce over every atom in atom order, as now, so they stay
  bit-identical to the stereo-blind fit.
- **An aware-only class** reduces over the atoms whose aware class is *c* and the atoms
  whose M class is *c*. Each training atom therefore counts toward its aware class *a*
  and toward M(*a*); where *a* = M(*a*), an achiral environment, it counts once.

Consequently *c* and M(*c*) always hold identical count, mean and variance. That is the
fit on the corpus plus every molecule's enantiomer, without materialising any enantiomer.

**`aware_variance` counts each mirror orbit once.** A mirror pair is two identical
refinements of one blind class. Counting both would add a zero-variance sibling and bias
τ²_aware downward, so only one member of each orbit enters (either will do, since their
statistics are identical).

Everything else in design D §5 is unchanged: continuation and τ² over blind children
only, aware-only classes pooled and shrunk toward `blind_of`.

## 5. Prediction

Unchanged from design D §6: the walk over blind labels to *k*\*, then one lookup of the
query's aware class at *k*\*.

**Mirror invariance, exactly.** An enantiomer query has aware class M(*a*) wherever the
original has *a*. Every training atom minted both *a* and M(*a*) with identical
statistics, so the two queries receive identical predictions atom for atom. That holds
when only one hand, or neither, was in training, and across any sharding, since the
M rows are built per atom inside each shard.

A query does not need its own M rows; `refine` may skip them outside fitting. Building
them anyway is correct and only costs time.

## 6. Merge, schema, persistence

- `mirror_of` is remapped and checked exactly like `blind_of`
  (`merge.merge_level`); kinds still combine by OR.
- `schema_version` already digests `stereo`, so a cis/trans-only model and a
  both-tracks model hash differently and never merge. The
  `stereo_construction: refines_blind` marker is unchanged.
- `save`/`load` add `level_{k}_mirror_of` beside `level_{k}_blind_of`, only where
  present.
- `predict_loo` and `experiments.analytic.sieve_train_stats` stay refused for stereo
  models.

## 7. Tests

Written before the code. Most fixtures come from **tag manipulation**, which isolates
each track's effect without a second code path:

1. **Embedding.** A both-tracks fit on molecules with every chiral tag removed equals the
   cis/trans-only fit, in partition, statistics and predictions (up to the larger edge
   modulus, which splits nothing).
2. **Mirror invariance.** Inverting every chiral tag (CW ↔ CCW, E/Z untouched) turns each
   molecule into its enantiomer.
   - A fit on the inverted corpus has the same partition and statistics as the original.
   - Predictions on inverted queries equal predictions on the originals, including when
     training held only one hand.
3. **Diastereomers separate, enantiomers do not.** In 2,3-dibromobutane, (*R*,*R*) and
   meso (*R*,*S*) separate once both centres are within reach; (*R*,*R*) and (*S*,*S*)
   pool; the meso form equals its own mirror.
4. **Self-mirror classes count once.** In meso-2,3-dibromobutane, an atom whose row names
   the two centres' classes *a* and M(*a*) is unchanged by reflection. Every class with
   `mirror_of[c] == c` has a count equal to its number of training atoms, which fails if
   an atom's M row is counted onto the same class twice.
5. **Enantiomers pool.** (*R*)- and (*S*)-alanine share statistics: the class and its
   mirror hold identical count, mean and variance.
6. **The radius rule**, under an **element-only** attribute config, stated in the test:
   `fp_0` hashes the whole attribute row, so with ring, aromatic or degree-like attributes
   alanine's methyl and carboxyl carbons already differ at radius 0 and the test would
   pass or fail for a reason unrelated to the rule. CHFClBr resolves at round 1;
   alanine, whose two carbon neighbours tie at radius 0, resolves at round 2 and not
   before.
7. **Virtual neighbour, pinned.** A centre written with implicit H gives the same code
   across the five SMILES orderings of §2.1. Hand-checked cases, a sulfoxide and an
   implicit-H carbon, assert the specific code (`CODE_PLUS` or `CODE_MINUS`), not only
   invariance, which pins the virtual neighbour's convention. A sulfoxide's aware class
   and its mirror's are distinct classes holding identical statistics.
8. **Renumbering.** Refining a molecule under a permuted `node_order` leaves its codes
   and classes unchanged; this fails if `parity` does not follow the stored `n0..n3` order
   after the adapter's `inv[]` permutation.
9. **Validation.** `stereo_centres` rejects a non-adjacent neighbour, `-1` outside `n3`, a
   repeated centre, a row naming fewer neighbours than the degree, and a parity outside
   {−1, +1}; slicing and `concat_batches` carry it.
10. **Ties defer.** A tagged atom with two identical substituents gets `none` at every
   radius.
11. **Blind equivalence** (design D §8.1) and **the merge monoid**, now also on
   `mirror_of`, over two and three shards.
12. **Batch invariance.** Codes and predictions are identical beside the five-coordinate
   phosphorus molecule.
13. **Schema.** cis/trans-only, both-tracks and stereo-blind models have three distinct
    digests; both spellings of the two-track config share one; save/load round-trips
    `mirror_of`.

## 8. Rejected: an intermediate chain

Blinding the handedness trit first would give a third chain, cis/trans-aware but
handedness-blind, which is exactly Study D's aware class. Prediction could then fall
back from both-aware to cis/trans-aware to blind at *k*\*, and the cis/trans model would
be embedded atom for atom.

It was dropped for simplicity. The cost is confined to atoms within reach of both a
stereogenic double bond and a tetrahedral centre, which lose their cis/trans answer
whenever the both-aware class is missing. On 100,000 train conformers (2026-09-24),
0.51% of atoms are within two bonds of both, but that is 15.4% of the atoms within two
bonds of an E/Z bond, where Study D's gain lives. Study E's report therefore includes the
E/Z subsets (§9): a regression on `near_ez2` against the cis/trans arm measures this
cost, and the chain can be added later as a refinement if it does.

## 9. Validation: Study E

Two arms at the selected depth, five repeats of *K* folds, both estimators (continuation
with and without empirical-Bayes shrinkage):

- **incumbent:** Study D's cis/trans arm, reused, not re-run;
- **new:** element continuation with both tracks, `element-ctt-eb` shard fits.

Subsets are added to `experiments.stereo_subsets` beside the E/Z ones: `has_tet`
(conformers with a tagged tetrahedral centre), `near_tet2` and `near_tet1` (within two
and one bonds of one). The mask table is rebuilt and every paired run re-scored.

- **Primary metric:** RMSE on `near_tet2`.
- **Checks:** RMSE on `near_ez2`, which must not regress (§8), and corpus-wide RMSE and
  MAE (no regression).

## 10. Open items

1. **How often the code fires.** 21.7% of sampled atoms are within two bonds of a tagged
   centre, but a tagged atom whose neighbours tie never fires. Measure the fraction of
   centres resolved at each radius once implemented; it sets the expectation for §9.
2. **Cost.** The M rows add at most one row per atom within reach of a resolved centre.
   Measure the aware-only fraction per level, as for design D §10.2.
3. **Collapse keys.** `collapse_key` inherits RDKit's canonical SMILES stereo perception.
   That population is unchanged here; this track only reads the tags the store already
   carries.
