# Fit-time collapse of equivalent molecules

Specification for collapsing molecules that are equivalent under this series'
model premise, at fit time rather than in the store, and for the annotation
that makes it cheap.

Everything cited as measured was measured on the real `dash-molecules` corpus
on 2026-09-17; the measurements themselves are recorded in
`experiments/docs/analytic-diagnostics-and-conformer-collapse.md`.

## 1. The problem

The corpus counts a molecule once per conformer, and it counts some molecules
more than once:

- **Conformers.** 924,714 training rows for 313,964 molecules, 2 or 3
  conformers each (94.5% have three). Conformers of one molecule share a
  graph, so every arm in this series predicts them identically; the extra
  rows carry almost no information (within-conformer variance is 0.131% of
  total) but do reweight every class mean.
- **Exact duplicates.** 7,780 canonical SMILES carry more than one `dash_id`,
  covering 15,724 molecules (~2.2%).
- **Enantiomers.** Measured interchangeable: enantiomer pairs differ by
  0.00780 per-atom RMS, which is the noise floor of that measurement. No arm
  here can see chirality in any case.

The consequence is an arbitrary weighting. A three-conformer molecule counts
1.5x a two-conformer one; a duplicated structure counts twice again. None of
that reflects information content.

**Diastereomers and E/Z isomers are NOT equivalent** and must not be merged.
They differ by 0.00698 and 0.00766 per-atom RMS above the enantiomer control.
No arm can distinguish them either, but that is a deficiency of the
featurizations, not a licence to average their targets: merging them would
define the target as a mean over chemically different molecules and hide a
real error as a smaller one.

## 2. Why fit time, and not in the store

Collapsing is licensed by the model premise -- the members are
indistinguishable to the model, so it would learn their mean anyway.
**That argument holds at training time and fails at test time.** A collapsed
test target encodes the premise into the metric, so the metric can no longer
detect the premise being false: if enantiomers did differ, the collapsed
target would absorb that difference as "not an error".

Therefore the collapse is a property of *fitting*, never of the data. Test
rows stay one per conformer and are scored individually. A held-out enantiomer
pair produces two rows with identical features and whatever targets they
actually have, and the model eats the difference as error. The premise is
checked rather than assumed.

Three consequences follow, all of them simplifications:

- Reported metrics remain **per conformer**, directly comparable to the
  published DASH numbers, with no reconstruction step.
- No second store, and no averaged-charge row carrying one member's 3D
  geometry -- a mismatch that would be harmless for graph-based arms and a
  trap for anything using coordinates.
- `split`, `cluster`, `shard` and `dash_id` keep their current meaning, so the
  CV machinery is untouched.

## 3. The equivalence key

For each row, with the molecule as stored:

    collapse_key = min( canonical_smiles(mol), canonical_smiles(mirror(mol)) )

where `mirror` inverts every tetrahedral chiral tag (`CHI_TETRAHEDRAL_CW` and
`CHI_TETRAHEDRAL_CCW` exchanged) and leaves bond stereo untouched.

One rule performs exactly the three intended merges:

- conformers of one `dash_id` share a key by construction, having one graph;
- exact duplicates share it, having one canonical SMILES;
- enantiomers share it, because `min` selects the same representative of the
  mirror pair from either side.

And it performs no others: diastereomers differ in canonical SMILES and are
not each other's mirror, so their keys differ; E/Z isomers likewise, since
`mirror` preserves bond stereo.

The key is a string, computed once per row.

## 4. Store annotation

A new op adds four columns to an existing store, in place. Adding columns in
place is the established pattern here -- `prepare-store --n-shards` adds
`cluster` and `shard` to a store that already exists.

| column | type | meaning |
|---|---|---|
| `collapse_key` | string | as defined in section 3 |
| `n_collapsed` | int32 | rows in this row's key group, corpus-wide |
| `n_molecules` | int32 | distinct `dash_id` in the group |
| `n_enantiomer_forms` | int32 | 1, or 2 when both handednesses are present |

`n_collapsed` is the headline count. The breakdown distinguishes a
multi-conformer molecule from a genuinely redundant one, which are different
curation facts that the total alone conflates.

The op is idempotent: a store already carrying the columns is skipped. It
costs one canonical-SMILES pass, about ten minutes for ~1M rows, paid once
rather than per fit.

**It asserts what was verified rather than assuming it.** If any key group
straddles a `split`, a `cluster` or a `shard`, the op fails and names the
group. Measured today, none does -- identical molecules share a fingerprint,
so Butina cannot separate them, and both the split and the sharding are by
whole cluster. Making it an assertion means a future corpus that breaks the
property stops, rather than silently averaging across a fold boundary.

Because no group straddles a shard, a group lies entirely inside train or
entirely inside test for every fold, so `n_collapsed` is fold-invariant and
can be stored once.

## 5. Fitting

Before `fit`, the harness groups the training rows by `collapse_key` and
builds its `NodeBatch` from one representative graph per key, with the mean of
that key's targets. Default weighting is **one key, one vote**.

`--weight-by-collapse` instead weights each key by its `n_collapsed`,
reproducing today's atom-weighted fit. It exists for the migration check in
section 9 and is not the recommended setting.

### Representative and atom correspondence

"The mean of that key's targets" needs an atom correspondence, and it is not
the identity: two records of the same structure may order their atoms
differently, and an enantiomer's atoms certainly do not correspond
positionally to its mirror partner's.

- **Representative**: the row with the lowest `(dash_id, conf_id)` in the
  group. Any member would serve, since all share a graph; the ordering is
  fixed so that a refit reproduces the same batch.
- **Correspondence**: for each member, take whichever of the molecule or its
  mirror has canonical SMILES equal to `collapse_key`, and order its atoms by
  `Chem.CanonicalRankAtoms` of that molecule. Every member then presents its
  atoms in one canonical order, and targets are averaged position-wise in it.
  The correspondence is valid precisely because all members share that
  canonical form.
- **Hydrogens** are kept, not folded: the targets are per-atom charges and the
  featurizations see explicit H. The canonical ranking covers them.

The representative's own handedness is arbitrary and irrelevant, since no
featurization in this series reads chirality. It is recorded in the batch only
as a graph.

Nothing in `sieve`, in DASH's `compute_node_stats` or in HOSE's tables
changes. The collapse happens in batch construction, inside the experiments
harness.

### Relation to fractional weighting

An equivalent-looking alternative is to keep every row and accumulate
`1/n_collapsed` instead of 1. For the **means** it is exactly equivalent: all
rows of a key share a graph, so they land in one class, the key contributes
total weight 1, and the weighted class mean equals the mean of the key means.

For the **variances** it is not. Expanding the weighted second moment,

    msd_frac = mean_k( W_k / n_k )  +  var_k( ybar_k )

where `W_k` is the within-key sum of squares. The fractional form carries the
conformational noise inside `msd`; the collapsed form treats each key as one
clean observation. That difference reaches empirical-Bayes alpha, the
predictive variance and the analytic train/LOO diagnostics, so the two produce
*different fitted models* under shrinkage.

Fit-time collapse is chosen because it needs no weighted-moment machinery in
`sieve` core, `merge.py`, DASH's stats or HOSE's tables -- and because
fractional weighting still featurizes all 924,714 rows when 313,964 carry the
information. Fractional weighting is nevertheless the more general mechanism,
and is what a future cluster-level or duplicate-down-weighting scheme would
need; the `msd` distinction above is the thing to get right when building it.

## 6. What changes downstream

- **`minimum_support` counts distinct structures.** The present value of 12,
  chosen as "at least four molecules" because of the ~3 conformers, becomes
  **4**. This is a semantic change, not a rescaling of an arbitrary knob.
- **Empirical-Bayes alpha is re-estimated** from the new counts. Shrinkage
  arms must be re-tuned rather than ported.
- **Class supports fall by roughly 3x**, so the support distribution in the
  diagnostics document is re-measured rather than translated.
- **Cost.** Fits process ~313,964 units instead of 924,714 rows, a 3x
  reduction. Prediction is unchanged, since the held-out side is not
  collapsed. A fold holds out 10 of 50 shards, so the overall saving is about
  **2x**, not 3x.
- **Existing results are not invalidated.** Studies A, B and C used a
  different and defensible weighting. They are not directly comparable to
  post-collapse numbers, and re-running them is a separate decision.

## 7. The held-out floor diagnostic

Per fold, the within-key scatter on the **held-out** side is the irreducible
error for that fold: the part no graph-based arm can avoid, because the
members are identical to any such model. It is available at no cost from the
rows already being scored.

Reported beside RMSE, it turns the equivalence premise into a measured number
per study rather than a standing assumption. A drift upward is the signal
that the premise is failing on new data.

## 8. Testing

- `collapse_key` merges conformers, exact duplicates and enantiomers, and
  leaves diastereomers and E/Z isomers distinct. Unit tests over small
  hand-written SMILES, one per case.
- Fit-time collapse and fractional weighting give identical class means under
  `class_estimator="pooled"`. This is the equivalence that justifies the
  design and it should be pinned, not argued.
- The straddle assertion fires on a store where a group spans two shards.
- `--weight-by-collapse` on a collapsed fit reproduces the uncollapsed fit's
  class means exactly.
- End-to-end: a CV run on a tiny sharded store with collapse enabled, checking
  that held-out rows remain one per conformer.

## 9. Migration check

Before any result is re-run, one arm is fitted with `--weight-by-collapse` and
must reproduce that arm's current Study B number exactly. That single check
validates the annotation, the grouping and the representative selection at
once. Only then is the default weighting used for new work.

## 10. Out of scope

- Re-running Studies A, B and C on the collapsed premise. A separate decision
  with its own cost.
- Any collapse of diastereomers or E/Z isomers. Section 1 explains why.
- Cluster-level or other non-uniform weighting. Section 5 records what it
  would need.
- Capturing stereochemistry in the featurizations. `bond_stereo` exists in the
  RDKit adapter and `charge_experiments` holds an `element-bondstereo-10fold`
  run from the earlier harness; that is a featurization question, not a
  curation one.
