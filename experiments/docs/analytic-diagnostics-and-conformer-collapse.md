# Analytic diagnostics from stored statistics, and collapsing conformers

Two findings from 2026-09-17, recorded together because the second removes the
main caveat on the first.

Everything below was measured on the real corpus unless it says otherwise.
Where a claim is derived rather than measured, it says so, and where it was
checked numerically, the check is named.

## 1. The training error is already on disk

A CV run scores its training split by re-predicting it (`--score-train`). That
is unnecessary: for every predictor in this harness the training error is a
closed-form function of the statistics the fit already stores.

**The argument.** Every predictor here answers an atom by backing off from its
deepest class until one exists with enough support. A *training* atom's
deepest class always exists -- the atom itself contributed to its count -- so
with `minimum_support = 1` the loop stops immediately and the prediction is the
mean of the training atoms sharing that class. The training error is therefore
pure within-class variance:

    SSE_train = sum_c ( ss_c - sum_c^2 / n_c ) = sum_c n_c * var_c

Both forms are per-class accumulators, and both merge additively across shards.

**Sieve and DASH already store what is needed.** `LevelStats.msd` is the
per-class population variance -- `merge.py`'s update
`msd = wA*msd_A + wB*msd_B + wA*wB*delta^2` is Chan's parallel-variance
formula -- so `SSE = sum(count * msd)`. DASH's `TreeNodeStats` carries `std`
per node, giving `SSE = sum(count * std^2)`. Neither needs a refit. HOSE
stores `sum` and `count` only and would need one more column.

**Verified, not asserted.** Against the `train/rmse` that Study A recorded with
`--score-train`, for `sieve-element-pooled`, repeat 0 fold 0, on the cached
training model `element-eb-w10-n50-k5/r0-f0.npz` (31,048,227 training atoms):

| depth | classes | analytic | recorded | difference |
|---:|---:|---:|---:|---:|
| 3 | 565,025 | 0.018423 | 0.018423 | 0 |
| 5 | 3,217,930 | 0.011147 | 0.011147 | 1.7e-18 |
| 6 | 4,414,817 | 0.010645 | 0.010645 | 1.7e-18 |

The analytic pass takes about a second and loads no molecules.

**What it does not give.** MAE and quantiles are not recoverable from first and
second moments. Nothing about held-out atoms is available -- test error,
coverage and backoff rates all require the test molecules to be featurized.
`sum_constraint/*` needs per-molecule aggregation the class tables do not
carry. The identity also fails for any variant with `minimum_support > 1`
(`sieve-element-continuation-cutoff`), where training atoms do back off.

**A caveat worth keeping.** This is an identity on the fitted statistics, not
an independent measurement. If `predict` ever disagreed with the stored
statistics, the analytic curve would agree with the bug. `--score-train`
should stay available as an occasional cross-check rather than be deleted.

## 2. Everything else the same statistics contain

Computed from one cached model in ~2 s, per refinement level:

| level | classes | train RMSE | eta^2 | atoms in classes < 12 | LOO RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 0.234348 | 0.46215 | 0.00% | 0.234348 |
| 1 | 506 | 0.077973 | 0.94046 | 0.00% | 0.077974 |
| 2 | 32,915 | 0.033757 | 0.98884 | 0.23% | 0.033786 |
| 3 | 565,025 | 0.018423 | 0.99668 | 5.94% | 0.018730 |
| 4 | 1,798,456 | 0.012769 | 0.99840 | 20.19% | 0.013768 |
| 5 | 3,217,930 | 0.011147 | 0.99878 | 36.78% | **0.012903** |
| 6 | 4,414,817 | 0.010645 | 0.99889 | 51.04% | 0.012925 |
| 7 | 5,331,980 | 0.010430 | 0.99893 | 62.07% | 0.013058 |
| 8 | 5,994,669 | 0.010307 | 0.99896 | 70.15% | 0.013170 |
| 9 | 6,454,421 | 0.010230 | 0.99898 | 75.75% | 0.013253 |
| 10 | 6,766,536 | 0.010179 | 0.99899 | 79.52% | 0.013309 |

**eta^2 at every level, free.** Level 0 gives 0.46215 against the 0.4601 this
notebook's own table reports for `element`, computed there with dedicated
full-store passes. It is a byproduct of any fit, at every refinement level
rather than only at level 0.

**Fragmentation, measured at depth for the first time.** The Study C analysis
argued fragmentation from level-0 class counts (11 for `element`, 27 for
`element+hybridization+aromatic`, 33 for `element+num_ring_memberships`). The
support distribution measures it directly: the share of training atoms sitting
in classes with fewer than 12 atoms runs 0.23% at level 2, 36.78% at level 5,
79.52% at level 10.

**There are no singleton classes at any level**, because conformers put a floor
of 2-3 atoms under every class. That is the empirical content of framing
`minimum_support = 12` as "at least four molecules".

## 3. Leave-one-out, and why it is optimistic

For a class of size `n` and mean `mu`, the LOO prediction for atom `i` is
`(n*mu - y_i)/(n-1)`, and the LOO error sums in closed form:

    SSE_loo = sum_c n_c^3 * var_c / (n_c - 1)^2      (classes with n_c >= 2)

**It reproduces the shape of the CV depth curve.** Train RMSE falls
monotonically to depth 10 -- it can only fall. LOO turns up after level 5:
0.012903 at 5, 0.012925 at 6, 0.013309 at 10. Its argmin is 5, which is the
depth Study A selected by the plateau rule over a ~1.2 h run; the CV minimum
is at 6 with depths 4-10 tied. The analytic version costs seconds.

**Its absolute level is not trustworthy, and the reason is quantified.**
Conformers of one molecule share a class, so leaving one atom out leaves its
siblings behind, and they are near-perfect predictors: within-conformer
variance is 0.131% of the total, putting a floor of
`sqrt(0.00131 * 0.102108) = 0.01157` under any LOO estimate here. The measured
LOO minimum of 0.012903 is **1.12x that floor**, and 1.55x *below* the CV value
of 0.020000. At depth, atom-level LOO measures little except conformer noise.

## 4. Leave-one-molecule-out closes in the same way

The honest unit is the molecule. For a class holding molecules `m` with `k_m`
conformers, sum `S_m` and sum of squares `Q_m`:

    v_m = (S - S_m) / (n - k_m)
    SSE = sum_m [ Q_m - 2*v_m*S_m + k_m*v_m^2 ]

The denominator varies with `k_m`, which blocks a naive collapse into moments,
but it is *constant within a group of equal conformer count*. Grouping by `k`:

    SSE = sum_k [ Q_k - (2/d_k)*(S*S_k - T_k)
                       + (k/d_k^2)*(M_k*S^2 - 2*S*S_k + T_k) ],   d_k = n - k

with, per class and per `k`: `M_k` molecules, `S_k = sum S_m`, `T_k = sum S_m^2`.
Brute force against this closed form agreed to ~1e-15 over four random trials
with mixed conformer counts.

**Storage is not a problem**, which was checked rather than assumed: at level
10 there are 4.6 atoms per class, so one or two distinct `k` values per class;
shallow levels carry many `k` values but few classes.

**Cluster leakage remains.** This corpus is split by Butina cluster, not by
molecule, so even leave-one-molecule-out is optimistic against the real CV
number -- a molecule's cluster-mates stay in its class. Leave-one-cluster-out
uses the identical grouping trick, but cluster sizes are large and variable
(the largest holds 3,733 molecules), so the accumulator set widens. The
correction is expected to be much smaller than the conformer one, since
cluster-mates are different molecules while conformers are the same molecule.
That expectation has not been measured.

## 5. Collapsing conformers

Treating the dataset as molecules, with targets averaged over conformers,
removes the leakage at its source: atom-level LOO then *is*
leave-one-molecule-out, with no grouping machinery at all.

**The imbalance objection is real in principle and negligible here.** Measured
on the train split, keyed on `dash_id`: 313,964 molecules, 924,714 conformers,
mean 2.95, range **2 to 3** -- 94.5% have exactly three and 5.5% have two.

Two independent reasons the imbalance does not bite:

- *Optimal weights barely differ.* Under a random-effects model the
  minimum-variance weight for a molecule mean is `1/(sigma_b^2 + sigma_w^2/k)`.
  Since within-conformer variance is 0.131% of the total, a third conformer
  buys almost no precision: across the whole population the optimal weights
  span **0.022%**. Equal weighting per molecule is effectively optimal.
- *Atom weighting costs almost nothing either.* The effective molecular sample
  size under atom weighting is 312,103 of 313,964, or **99.4%**.

**The positive case is stronger than the defensive one.** Under atom weighting
a three-conformer molecule counts 1.5x a two-conformer one in every class mean,
for no informational reason; collapsing removes that artifact. It makes
`minimum_support` count molecules, which is what this notebook meant when it
explained 12 as "at least four molecules". And it makes the free analytic LOO
honest.

**Comparability with the per-conformer literature is preserved exactly.** Every
arm here is graph-based and therefore predicts identically across a molecule's
conformers, so for a prediction `v` at atom position `p` of molecule `m`:

    SSE_atoms = sum_mp W_mp  +  sum_mp k_mp * (ubar_mp - v)^2

`W_mp` is the within-conformer sum of squares, model-independent and computed
once; the second term is the `k`-weighted collapsed SSE. A collapsed run
therefore reconstructs the per-conformer RMSE that DASH's paper reports,
provided `k` is carried per unit and used as a weight.

**It is about 3x cheaper everywhere**, since 924,714 conformers become 313,964
units. On the measured HOSE timings that turns radii 6-8 from ~9 h into ~3 h,
and every Study A and Study B gains the same factor.

**Data note.** `chembl_id` is null for 449,776 train rows; grouping on it
silently produces one 449,776-conformer "molecule". `dash_id` is complete and
is the correct key.

## 6. Corpus properties found while checking the above

These came out of smoke tests run against the collapse design. They are
recorded because each one changes what a number in this repository means, and
none of them is visible from the store's schema.

### `chembl_id` is null on 449,776 train rows

Grouping conformers by `chembl_id` silently fuses every null into a single
449,776-conformer "molecule", which is how the first conformer-count
measurement produced a mean of 5.69 and a maximum of 449,776. `dash_id` is
complete on every row and is the correct molecule key. Nothing downstream
uses `chembl_id`, so no result was affected; the trap is for the next person
who reaches for the obvious column.

### 7,780 molecules are duplicated, and none of them leaks

Taking one conformer per `dash_id` and comparing isomeric canonical SMILES
over the whole store:

| | |
|---|---:|
| conformers scanned | 1,027,538 |
| distinct `dash_id` | 348,849 |
| distinct canonical SMILES | 340,905 |
| SMILES with more than one `dash_id` | 7,780 (15,724 molecules) |
| group sizes | 2: 7,633, 3: 132, 4: 13, 5: 2 |
| groups straddling a **split** | **0** |
| groups straddling a **cluster** | **0** |
| groups straddling a **shard** | **0** |

So about 2.2% of `dash_id`s are redundant, and every duplicate group sits
entirely inside one split, one Butina cluster and one shard. **No fold in any
study was tested on a molecule whose copy it had trained on.**

That is not luck. Identical molecules have identical fingerprints, so Butina
necessarily assigns them to one cluster, and both the split and the sharding
are by whole cluster. The zeros are what a correct cluster-clean partition
must produce, which is exactly what makes the check worth running: a non-zero
would have meant the partition was broken.

Two consequences, both small. The effective number of distinct structures is
~2.2% below the `dash_id` count, so the "313,964 train molecules" quoted above
is really ~307k distinct structures. And a duplicated molecule contributes
about six conformers to its class rather than three, inflating that class's
support -- the same arbitrary weighting that collapsing conformers removes,
and an argument for keying any leave-one-molecule-out on the canonical SMILES
rather than on `dash_id`.

### A `dash_id` is not a structure: 2.14% hold more than one

Found while annotating the real store for the first time. Keying every row
rather than one row per `dash_id` gives *more* keys than there are `dash_id`s,
which can only happen if a molecule's own conformers key differently:

| | |
|---|---:|
| distinct `dash_id` | 348,849 |
| collapse keys, one conformer per `dash_id` | 332,618 |
| collapse keys, **every** conformer | 339,521 |
| `dash_id`s whose conformers disagree | **7,452 (2.14%)** |
| conformers involved | 22,140 |

Classified by InChI layer, which is the authoritative tool here -- comparing
stereo-stripped SMILES strings is not, because changing E/Z geometry also
reorders RDKit's canonical traversal, so stripping `/` and `\` leaves two
strings that differ for a reason that has nothing to do with connectivity:

| | | |
|---|---:|---:|
| stereocentre configuration (diastereomers) | 5,676 | 76.2% |
| E/Z double-bond stereo | 1,038 | 13.9% |
| identical InChI (mobile-H / tautomer normalization) | 707 | 9.5% |
| both E/Z and stereocentre | 31 | 0.4% |
| **different connectivity** | **0** | **0.0%** |

No record has a different molecular graph under one `dash_id`, which is the
reassuring half. The other half is that one `dash_id` routinely covers
chemically distinct species -- 5,676 of them cover diastereomers, which have
different energies and different charges.

`collapse_key` handles this correctly without special-casing: it groups by
structure, so such a molecule's three conformers become two or three units
rather than one. That is the whole of the 332,618 -> 339,521 difference.

**Two consequences for what is written above.** Section 5's proposal to
average conformers keyed on `dash_id` would, for these 5,676, average across
diastereomers -- exactly the "mean over different chemistry" the collapse
design exists to prevent. And section 8's open question of whether
leave-one-molecule-out should key on `dash_id` or on canonical SMILES is
answered: `dash_id` is not a structure key, and `collapse_key` is.

These counts are properties of the molecules alone, not of any partition, so
they are unchanged by a re-clustering of the corpus.

### Stereochemistry: only enantiomers are interchangeable

Stripping stereochemistry collapses 345,865 molecules onto 319,477 flat
graphs, so **7.6% of the corpus is distinguished from something else by
stereo alone** -- more than the 2.2% that are exact duplicates.

Whether that matters depends on which kind of stereoisomerism. Charges were
averaged over each molecule's conformers first, hydrogens folded onto their
heavy neighbour, and corresponding atoms matched through the canonical
ranking of the stereo-stripped graph, which is a valid correspondence exactly
because those graphs are identical. A pair counts as enantiomers iff
inverting every tetrahedral centre of one reproduces the other.

| pair type | pairs | median | mean | p95 | excess over control |
|---|---:|---:|---:|---:|---:|
| **enantiomers (control)** | 8,264 | 0.00780 | 0.00897 | 0.01920 | -- |
| diastereomers | 2,342 | 0.01047 | 0.01199 | 0.02610 | 0.00698 |
| E/Z only | 9,189 | 0.01093 | 0.01201 | 0.02215 | 0.00766 |
| mixed | 558 | 0.01077 | 0.01185 | 0.02235 | 0.00743 |

**Enantiomers are the only valid control**, and separating them matters. A
first pass grouped every chirality difference together and measured 0.00952;
that bucket mixed mirror images with diastereomers, which are *not* mirror
images and may legitimately differ. The clean control is **0.00780**, and
every effect measured against it is correspondingly larger than that first
pass reported.

The control also sits below the 0.00945 predicted from conformer averaging
alone, so the prediction was an overestimate -- plausibly because folding
hydrogen charges onto heavy atoms sums correlated quantities. The measured
control, not the prediction, is the floor to use.

**Both diastereomers and E/Z show real, comparable excesses**, about 0.0070
and 0.0077 per-atom RMS. So the intuition that chirality is irrelevant while
cis/trans matters is only half right: enantiomers are interchangeable,
everything else is not, and diastereomers differ nearly as much as geometric
isomers do.

**No arm in this series can see any of it.** Sieve here carries `element` on
nodes with no edge attributes, DASH's tuple is
`(element, degree, formal_charge, aromatic, num_h)`, and HOSE codes carry bond
order but not bond stereo. Stereoisomers therefore land in the same class and
receive one prediction, so roughly half the pair difference is incurred on
each. Being common to all three arms, it cannot change their ranking -- it is
a shared floor, like the conformational one.

Two limits on reading these numbers. The penalty is only paid where **both**
isomers are present and collide in a class, which is the ~5% of molecules with
a stereoisomer twin in the corpus, so the corpus-wide contribution is much
smaller than 0.0077 and has not been measured. And diastereomers differ in 3D
shape, so part of their excess may be that two diastereomers' conformer
ensembles are not comparable samples rather than an electronic difference;
this measurement cannot separate those.

The lever exists if it is ever worth pulling: `bond_stereo` is in the RDKit
adapter's edge attributes, and `charge_experiments` holds an
`element-bondstereo-10fold` run from the earlier harness.

## 7. What this enables

**Implemented** (`docs/superpowers/specs/2026-09-17-analytic-training-metrics-
design.md`, `experiments/analytic.py`): `run_sieve_cv` and `run_hose_cv` now
record `train/rmse`, `train/r2`, `train/eta2`, `train/matched_fraction` and
the support distribution on every run, computed from the fitted model's own
stored statistics -- no extra fit, no extra walk, no molecules loaded.
`--score-train` was dropped from Study A's Sieve and HOSE steps and from
Study C's stage-1 curve, which had skipped it outright to afford stage 2's
repeats; both now get the train curve for free. A standalone
`analytic-curve` command reads the same numbers from any saved state.

**DASH was measured and excluded, not merely left out.** Its per-node
statistics describe the atoms passing *through* a node, and at a shallower
depth some of those atoms are answered at a shallower, *terminal* node whose
own moments the stored columns cannot isolate. On a real DASH shard the
terminal group is 94% of training atoms at depth 16 -- the depth Study A
selects -- so no analytic shortcut exists there, and DASH keeps
`--score-train` as its only route to a training curve.

**HOSE's curve is exact only at the baseline `n_min=1`**, where a training
atom's own deepest key always answers it and no backoff occurs; generalizing
to `n_min>1` would mean implementing prefix backoff over the key tables for
a method nobody runs. It also needed one new stored column (`sumsq` per
key), so a HOSE state saved before this landed still loads and predicts
exactly as before, but raises rather than fabricate an analytic number.

Still open, not yet implemented:

- Select depth from the analytic LOO curve and use CV to confirm rather than to
  discover.
- Sweep `shrinkage_strength` against analytic LOO, which is closed-form in the
  shrinkage parameter, instead of one CV per value. This notebook already
  names that sweep as the experiment most likely to move test R^2 off its
  plateau. It should be validated against one real CV point first: LOO's
  leakage may interact with shrinkage differently than with depth.
- The per-conformer reconstruction under `--collapse`
  (`train/rmse_per_conformer`, spec section 5): the identity is derived, but
  it needs a second, k-weighted shard-fit artifact whose class signatures
  align with the unweighted fit's, doubling shard-fit storage and merge
  time. Deferred pending a decision on that cost; `train/rmse` in collapsed
  units is still correct and already recorded for a `--collapse` run.

## 8. Open

- Leave-one-cluster-out: derived but not implemented, and its size against
  leave-one-molecule-out is unmeasured.
- ~~The analytic identity for `class_estimator="continuation"` should hold...
  still untested.~~ Resolved: `experiments/analytic.py`'s general backoff
  walk covers `continuation` and every `shrinkage_weight` rule, each checked
  against `sieve.predict`'s own output rather than assumed. Verifying it
  this way caught two real bugs in the walk itself -- see the
  analytic-training-metrics design spec and its implementation plan.
- Whether the collapsed dataset changes any ranking in Study B. It should not,
  since the removed term is a common additive constant across graph-based arms,
  but that is an argument rather than a measurement.
- The corpus-wide cost of stereo blindness. The per-pair excess is measured
  (~0.0077 for E/Z, ~0.0070 for diastereomers) but the fraction of the total
  error it accounts for is not, because the penalty is paid only where both
  isomers collide in one class.
- Whether the diastereomer excess is electronic or an artifact of two
  diastereomers' conformer ensembles not being comparable samples. Averaging
  2-3 conformers does not settle it.
- Whether leave-one-molecule-out should key on `dash_id` or on canonical
  SMILES. The 7,780 duplicate groups leak between copies under the former.
