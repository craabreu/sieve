# Analytic training metrics — design

**Status:** proposed
**Implements:** `experiments/docs/analytic-diagnostics-and-conformer-collapse.md`, sections 1–3 and 7
**Date:** 2026-09-17

## 1. What this is

A fitted model already stores, per class, the count, mean and mean-squared
deviation of the training atoms it holds. That is enough to reproduce the
training error *exactly* — and the training R², η², the support distribution
and the leave-one-out error with it — without loading a molecule or running a
walk.

The argument, in one line: a training atom's own class exists at every level
it reaches, because the atom itself contributed to it, so the backoff search
is decided entirely by the stored counts. Whichever class answers, the
prediction is that class's stored estimate, and the error against that class's
own stored moments is algebra.

The diagnostics note measured this to 1.7e-18 against a real `--score-train`
run before any of this existed. This document is about turning that
measurement into a capability.

**The payoff** is the note's section 7: Study A's `--score-train` is recorded
as taking Sieve's run from ~15 min to ~1.2 h, for a number that is free; and
Study C's five arms, which were run without train curves precisely because of
that cost, can have them retroactively from cached models.

## 2. Scope

| predictor | analytic train + LOO | refit | state change |
|---|---|---|---|
| Sieve | yes | no | none |
| HOSE | yes | yes | `sumsq` per key |
| DASH | **no** | no | none |

### Why DASH is excluded

DASH's `TreeNodeStats` holds `count`/`mean`/`std` for the atoms passing
*through* each node. At depth *d* a training atom is answered at the last node
of `path[:d]` — which is a node at depth *d* if its path is long enough, and
its own **terminal** node if the path ended higher. For the first group the
existing columns are exactly right. For the second they are not: the node's
statistics describe every atom through it, including those that carried on
deeper, and the terminating subset cannot be separated after the fact.

Measured on `fit-dash-s00` (717,145 training atoms, mean path length 11.42),
with node depth taken as BFS depth over `tree_storage[b][node][5]` — the
children list `_pick_subgraph_expansion_node` itself walks — and validated by
0 mismatches over 591 real `match_new_atom` path steps:

| depth | answered at that depth | terminal group above it |
|---:|---:|---:|
| 4 | 99.88% | 0.12% |
| 6 | 98.44% | 1.56% |
| 8 | 93.70% | 6.30% |
| 10 | 78.99% | 21.01% |
| 12 | 49.01% | 50.99% |
| 16 | 5.65% | **94.35%** |

Study A ran DASH's curve to depth 16 and selected 16, where 94% of training
atoms sit in the group the stored statistics cannot describe. Making DASH
exact would need `compute_node_stats` to also accumulate terminal-atom
moments, and therefore a refit. **Decision: DASH gets nothing.** It keeps
`--score-train` as its route to a training curve, and the `--score-train`
flag therefore stays in the workflow's DASH steps.

This is recorded rather than silently dropped because the measurement is the
only reason it is not worth doing, and a future reader will otherwise ask.

### Why HOSE's is restricted to `n_min = 1`

`HoseState` stores `sum` and `count` per key and no second moment, so nothing
analytic is possible today. Adding `sumsq` is pure storage — it changes no
prediction — and makes the training error exact at the baseline setting,
where the deepest key always answers a training atom and no backoff occurs.

Generalizing to `n_min > 1` would mean implementing prefix backoff over the
key tables. That is a method nobody runs: `hose-charge-example.yaml` sets
`n_min: 1` and every study used it. **The analytic path raises for any other
`n_min` rather than generalizing.** Analysing a model that is not the
baseline is worse than not analysing it.

## 3. The general form

The note's section 1 derives `SSE = Σ_c n_c·var_c`, which holds only when
`minimum_support = 1`, the estimator is pooled and nothing shrinks. The
implementation uses the general form instead, of which that is one branch.

Walk the backoff chain from the deepest level *up*, exactly as
`sieve.predict._search` walks it down. Carry per class the triple
`(n, Σy, Σy²)`, initialized at the deepest level from `count`, `mean` and
`msd` (`Σy² = n(msd + mean²)`). At each level a class either has the support
the config demands — in which case every atom still carrying it is answered
there —

```
SSE += Σ_c ( Q_c − 2·v_c·S_c + n_c·v_c² )
```

— or it does not, and its `(n, Σy, Σy²)` are folded into its parent class and
tried one level up. Atoms that run out of levels are answered by the global
mean.

`v_c` is `shrinkage.shrunk_means(model)[k][c]`. That one call is correct for
every reading of the class tables: with no shrinkage it returns the class's
own estimate (pooled or continuation) for populated classes, and under every
shrinkage rule `predict` provably lands on the same class-indexed value.

The expanded form, rather than the centered `Σ n_c·var_c`, is what makes this
general: under shrinkage, backoff or LOO the prediction is not the class's own
mean, and only the expansion accepts an arbitrary `v`.

**Leave-one-out** is the same walk with two changes. A class must clear
`minimum_support + 1` to answer, and the residual of atom *i* against the
leave-one-out mean of an *N*-atom class is `N/(N−1)` times its ordinary
residual — so the group's contribution is scaled by `N²/(N−1)²` and centered
on the class's own mean. LOO is refused for `continuation` and for shrinkage,
the same cases `sieve.predict_loo` refuses, and for the same reason: the
correction needs a child class identity this walk does not carry.

### Why truncation gives the whole curve

WL refinement never looks ahead, so a depth-*D* fit's levels `0..d` are
bit-for-bit what a native depth-*d* fit stored (`cv.truncate_model`, verified
bit-for-bit against native fits). One merged model therefore yields every
depth at the cost of one pass over the class tables per depth — about a second
each on the real corpus.

## 4. What is reported

Per depth: `n_classes`, `n_atoms`, `sse`, `rmse`, `r2`, `eta2`,
`matched_fraction`, and the share of training atoms in classes below each of
several support thresholds (2, 4, 12, 50).

`eta2` is a property of the partition — the share of total variance lying
between classes at the deepest level. `r2` is a property of the fitted
estimator, `1 − SSE/TSS`. They coincide exactly when nothing shrinks and every
atom matches its deepest class, and diverge as soon as either is false. Both
are reported because the divergence is the interesting part: it is how much
the estimator gives away relative to the partition it was handed.

`matched_fraction` is the share of training atoms answered by some level
rather than by the global mean. At `minimum_support = 1` it is 1.0 by
construction, and anything else means the walk is wrong.

## 5. Interaction with `--collapse`

A fit with `--collapse` trains on one representative per equivalence key, so
its analytic training error is in **collapsed units** — not comparable to an
uncollapsed run, nor to the per-conformer figures the DASH paper reports.

Both numbers are recorded, named so the difference cannot be missed:

- `train/rmse` — what the fit actually saw and optimized.
- `train/rmse_per_conformer` — reconstructed, via the note's section 5
  identity. Every arm here is graph-based and predicts identically across a
  molecule's conformers, so for a prediction `v` at atom position `p` of unit
  `m`:

      SSE_atoms = Σ W_mp  +  Σ k_mp·(ū_mp − v)²

  `W` is the within-key sum of squares: model-independent, and
  `collapse_molecule_set` already stacks each key's members, so it can return
  it as a scalar at no extra cost.

The second term needs the training moments **k-weighted** but scored against
the **unweighted** fit's values. A collapsed fit stores only unweighted
moments, so `Σ(ū−v)²` is what falls out, not `Σk(ū−v)²`.

The way through is that the partition is purely structural — it depends on the
graphs, not on the weights — so a `weight_by_collapse=True` fit has *identical
classes* and differs only in its moment tables. The reconstruction therefore
reads `v` from the real fit and `(n, Σy, Σy²)` from a companion weighted fit
over the same collapsed molecules, and is exact.

**Sieve only.** For Sieve the companion fit is seconds. For HOSE it would mean
regenerating every code a second time, which is the expensive half of that
arm; doing it properly would need a weight-aware accumulation inside
`HoseLookupPredictor.fit`, i.e. a change to the baseline. A collapsed HOSE run
records `train/rmse` in collapsed units and no reconstruction. Revisit if the
comparison is ever wanted.

## 6. Artifact compatibility

`HoseState` gains `sumsq` per key. `load_hose_state` accepts a file without
the column and leaves it `None`; `merge_hose_states` propagates `None` if
either side lacks it (rather than inventing zeros, which would read as "no
spread" and silently corrupt any later curve); the analytic entry point raises
a `ValueError` naming the file and saying a refit is needed.

So every existing HOSE artifact stays loadable, mergeable and scoreable
exactly as today. Only the new capability requires the refit.

## 7. Surface

**Module.** `experiments/experiments/analytic.py` — one backoff walk
parameterized by `(loo, weighted)`, a `TrainStats` record, and
`sieve_train_stats` / `sieve_curve` / `hose_train_stats` / `hose_curve`
adapters that feed it per-class `(n, Σy, Σy²)` and per-class values.

**CLI.** `analytic-curve --predictor {sieve,hose} <state.npz>` with
`--depths`/`--radii`, `--loo`, and `--out` for a CSV. Reads one file and
nothing else — no store, no molecules, no tree.

**CV.** `_write_cv_run` records `train/{rmse,r2,eta2,matched_fraction}` plus
the support fractions, computed analytically, for Sieve and HOSE runs;
`train/rmse_per_conformer` additionally under `--collapse` for Sieve.

**Workflow.** `cv_charges.sh` stops passing `--score-train` in the Sieve and
HOSE study steps. **The DASH steps keep it** — it is now DASH's only route to
a training curve.

## 8. Testing

Every analytic number is asserted equal to brute-force predicting the training
set with the real predictor — never against a stored constant. The closed form
is an identity on the stored statistics, so a test comparing it to anything
other than an independent prediction would pass even if the identity were
wrong.

Coverage: each `class_estimator`; each `shrinkage_weight`; `minimum_support`
of 1, 2, 3 and 8, so that training atoms genuinely back off; `neighbor_depth`,
so the coarse chain is skipped exactly as `predict` skips it; and the
degenerate cases (empty model, single class).

One test asserts the analytic result equals what `--score-train` records on a
small fit. That is the cross-check the note argues for, and the reason
`--score-train` is kept rather than deleted: the analytic curve is an identity
on the fitted statistics, so if `predict` ever disagreed with what the fit
stored, the analytic curve would agree with the bug.

## 9. Out of scope

- DASH, for the reason measured in section 2.
- HOSE at `n_min > 1`, and prefix backoff generally.
- `train/rmse_per_conformer` for HOSE.
- MAE and quantiles, which are not recoverable from first and second moments.
- Anything about held-out atoms — test error, coverage and backoff rates all
  need the test molecules featurized.
- `sum_constraint/*`, which needs per-molecule aggregation the class tables do
  not carry.
- Re-running any study. Landing this changes no published number; it changes
  what a future run records.

## 10. Open

- Leave-one-cluster-out, derived in the note's section 4 but not implemented.
- Sweeping `shrinkage_strength` against analytic LOO, which is closed-form in
  the shrinkage parameter. The note names it as the experiment most likely to
  move test R² off its plateau, and warns it should be validated against one
  real CV point first, since LOO's conformer leakage may interact with
  shrinkage differently than with depth.
