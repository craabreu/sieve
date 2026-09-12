# Design: cluster-clean-shard cross-validation for model comparison

**Status:** implemented (code + tests); not yet run against the real corpus
**Date:** 2026-09-12
**Scope:** `experiments/experiments/prepare_dash.py` (store split), `cv.py`
(new), `compare.py` (new), `predictors/sieve_predictor.py`,
`predictors/dash.py`, `tree_artifact.py`, `store_ops.py`, `config.py`,
`cli.py`, both workflow scripts. No change to `sieve`'s core (`src/sieve/**`).

## Context

Conformer curation (`prepare_dash.curate_conformers`, commit `d7ff3f1`)
changed the store, so every number under `experiments/runs/` had to be
recomputed anyway -- the cheapest moment to also change how the runs
themselves are produced (see `experiments/docs/harness-friction-
observations.md`).

The scientific goal is **model comparison via Tukey HSD**, following Ash,
Wognum, Rodríguez-Pérez et al. (*JCIM* 2025, 65(18), 9398–9411,
doi:10.1021/acs.jcim.5c01609, `AshWognum2025ModelComparison`): repeated
cross-validation, repeated-measures ANOVA, post hoc Tukey HSD, ≥25 samples,
and deliberately **no variance correction** for CV's known fold-dependence --
their answer is design (5×5 repeated k-fold, validated against Bates et al.'s
nested CV), not a Nadeau–Bengio-style adjustment.

Three things rule out simply running the old 10-fold sweep 25 times:

1. **Clustering.** Molecules cannot be split across train/eval at random.
   `_chalcedon.greedy_cluster_split` is deterministic, so re-seeding it to get
   a "different" fold assignment barely moves anything -- the large clusters
   land in a near-fixed LPT-scheduled pattern, only the small ones move.
   Repeats built that way would understate partition variance, which is
   exactly what a repeated-CV variance estimate needs to be real.
2. **Partial fits are shards.** Both methods' fitted state is mergeable
   sufficient statistics: DASH's `tree_artifact.TreeNodeStats`
   (`merge_node_stats`, an outer join on the *externally published* tree's
   own `(branch_idx, node_id)` keys -- no remapping needed) and Sieve's
   `SieveModel` (`sieve.merge.merge_models`/`fold`, id-remapped, balanced-tree
   reduction -- design.md §5). A k-fold training model is therefore `merge(k-1
   shards)`, not a fresh fit.
3. Consequently the number of CV samples stops costing fits. It becomes a
   purely statistical choice, freeing repeats/folds to be picked for
   statistical soundness rather than compute.

## Decisions

### 1. N cluster-clean shards, chosen from measured evidence

`prepare_dash.assign_splits` now writes three columns in one clustering pass
(the split and the shard ids must share one `butina_cluster` call -- it is
float32 and not guaranteed bit-exact across two separate runs):

- `split`: `{"train", "test"}` only, 90/10 by default (`val` optional, kept
  only for backward compatibility with existing tests/configs -- non-positive
  fractions are omitted from `greedy_cluster_split`'s own dict rather than
  passed as zero, since it rejects non-positive targets).
- `cluster`: the raw Butina cluster id, kept as provenance/for later
  diagnostics.
- `shard`: `s00`..`s{N-1}` on train rows (a second `greedy_cluster_split`
  restricted to train's own cluster ids), the row's own split name
  otherwise.

`N` (`--n-shards`, default 25) is a resolution knob, not a statistical one --
it changes neither the 80%-ish training fraction nor the number of CV samples
(`k` × repeats), only how finely shard boundaries track cluster boundaries.
Its ceiling is the largest Butina cluster: pushed past one shard's target
(`train / N`), that cluster forces its shard over target and can leave
another shard empty (`greedy_cluster_split` returns an empty array for a
split that never gets picked -- legal, silent). `cluster_size_report`
recomputes the same clustering and reports, for each candidate `N`, the
achieved shard balance and empty-shard count, so `N` is chosen from that
evidence (`experiments cluster-report`) rather than fixed in advance.

`prepare_store`'s already-split sentinel is now
`{"split", "cluster", "shard"} ⊆ schema.names`; a store with `split` alone
(pre-redesign) is refused with a clear message rather than silently
reinterpreted, since a second, separate clustering pass is not guaranteed to
reproduce the first's cluster ids.

### 2. Freeze the Sieve vocabulary before sharding

`SievePredictor.fit`'s own `build_codes` assigns each attribute value a dense
rank over whatever *that call's* training molecules contain. A shard is
~1/N of train; two shards' own vocabularies can disagree on what integer
code means what value, and `attribute_codes` feeds `SieveConfig.
schema_version`, which `check_mergeable` compares exactly -- a shifted
element table refuses to merge outright. With this series' own
`attributes: [element], edge_attributes: []`, the entire schema is that one
table.

Fix: `SievePredictor(codes_path=...)` loads a frozen `attribute_codes`/
`edge_codes` (`sieve_predictor.save_codes`/`load_codes`, plain JSON) instead
of calling `build_codes`, and `experiments build-sieve-codes` computes it
once over the whole train split before any shard is fit. `minimum_support`/
`class_estimator`/`shrinkage_weight` must also match across shards -- they
are deliberately excluded from `schema_version` (prediction-time, not
model-shape), so a mismatch would merge silently and wrongly.

### 3. A cached-batch seam for Sieve prediction

A `NodeBatch` carries no depth information at all -- `sieve.predict._search`
calls `refine(batch, model.config)` itself -- so one batch, built once, is
valid for `predict_raw_from_batch` against every depth's own model, as long
as they share `attribute_codes` (guaranteed by §2).
`SievePredictor.build_predict_batch`/`predict_raw_from_batch` split what
`predict_raw` used to do in one call, so a depth sweep featurizes an eval set
once rather than once per depth (featurization measured at ~96% of a fit).

Sieve still fits one shard set **per depth**, unlike DASH: under
`class_estimator="continuation"`, a class's estimate depends on whether its
own level is the model's *deepest* one, so a shallow config is not a
truncation of a deep one, and the two are not mergeable either (`max_wl_depth`
feeds `schema_version`). This is cheap regardless, since `N` shards is one
pass over train no matter how many depths are fit.

### 4. Assembly: permute shards, not the splitter

Randomization is a permutation of the already-built shards
(`cv.permute_into_folds`, seeded by the repeat number), not a re-run of
`greedy_cluster_split` -- which, being deterministic, would barely move the
large clusters between "repeats." Each repeat's `k` groups (`k=5`) give `k`
CV samples; each sample's training model is the merge of the other `k-1`
groups, computed by `cv.leave_one_group_out` via a prefix/suffix scan over
the `k` group-level merges (`O(k)` merges total, not `O(k)` redone from
scratch per held-out group) -- generic over `T` and the two artifacts'
own merge functions (`tree_artifact.merge_node_stats` /
`sieve.merge.merge_models`), with `None` standing in for "the merge of zero
groups" (neither artifact has a natural identity worth building just for
this).

`tree_artifact.merge_node_stats` was rewritten vectorized (`np.unique` over
stacked `(branch_idx, node_id)` keys plus a segmented Chan combination)
rather than a Python `dict` keyed by boxed tuples -- the dict form was the
dominant cost once the CV scheme's assembled models approach the whole
corpus's ~7M populated nodes. `sieve.merge.fold` already does a balanced-tree
reduction and needed no such change.

**DASH stale-state trap.** `apply_node_stats` mutates a `DASHTree`'s own
`tree.data_storage` in place, and only iterates branches present in the *new*
stats -- so reusing one loaded tree across CV samples (to avoid re-reading it
from disk, `DASHTree(preload=True)` being the expensive part of constructing
a predictor) would silently carry a previous sample's values on any branch
the new sample's model doesn't populate. Fixed with `apply_node_stats(...,
reset_existing=True)`, which first NaNs both columns across every branch that
already carries them; `DASHChargePredictor.load_model_state` always passes
it. Verified directly (`test_run_dash_cv_does_not_leak_state_across_repeats`,
and the unit-level `test_apply_node_stats_reset_existing_clears_a_branch_
absent_from_new_stats`).

### 5. Score twice, raw and normalized, from one raw prediction

`cv._score_raw_and_normalized` scores an evaluation both unnormalized and
with the method's own post-hoc normalization (`std_weighted` for DASH,
`equal_weighted` for Sieve) -- the normalized family lands under a `norm/`
prefix in the same `metrics.json`. Normalization is a pure post-hoc transform
of the same raw prediction, so the second scoring costs nothing extra; it
turns normalization from a fixed prior choice into something the study
itself can report on.

### 6. Two studies

- **Study A (depth selection).** One repeat (`k=5`, seed 0), no ANOVA, no
  Tukey -- a per-depth curve, read off the normalized MAE (the form each
  method is actually deployed in), raw reported alongside.
- **Study B (model comparison).** Four further repeats (seeds 1–4; Study A's
  own repeat/partition is reused as the first of five) at each method's own
  selected depth → 25 samples per method, feeding `compare.py`.

The competing methods are DASH plus one or more named Sieve configurations
(each independently depth-selected by Study A) -- `cv.py`'s driver functions
take a `config_label`/`predictor_params` pair per Sieve configuration, so
adding one is a parameter, not a code change. Which configurations enter
Study B is left open by design.

### 7. `compare.py`: scipy only, no statsmodels

`read_cv_table` pivots CV runs into a `(samples, methods)` matrix keyed by
`(repeat, fold)` as the repeated-measures "subject" -- valid because
`permute_into_folds(shard_ids, k=k, seed=repeat)` is deterministic in `(n, k,
seed)`, not in which predictor called it, so every method's sample at a given
`(repeat, fold)` is scored against the *same* held-out molecules. Raises
rather than silently comparing mismatched samples if that pairing does not
hold. `repeated_measures_anova` (`SS_error = SS_total - SS_method -
SS_subject`) and `tukey_hsd` (`scipy.stats.studentized_range`) follow
directly; `write_tukey_plot` renders one CI per pairwise comparison with a
provenance caption (store, run count) -- addressing harness-friction-
observations.md's "figures carry no provenance."

Runs are read via the *existing* `aggregate.read_runs_from_dirs`/
`flatten_params`: `cv.py`'s own run manifests carry a `"config"` block shaped
like an ordinary run's (`predictor.name`, `run.tags.*`, plus a `cv.*` block
of repeat/fold/method/depth/normalization), so `summarize`/`sweep` read a CV
run exactly like any other, with no changes to either.

### 8. One final held-out evaluation

Once, at the end, outside both studies: merge all `N` shards (at the
selected depth) and predict the untouched 10% test split -- the headline
number, never part of the CV or the Tukey test.

## Batch-id / experiment-name conventions

- Shard fits (predict nothing): `fit-dash-{shard}`,
  `fit-sieve-{config_label}-w{depth}-{shard}`, under a shared `cv-shard-fits`
  experiment (idempotency sentinel: the shard's own `tree_stats.npz`, not a
  `metrics.json`).
- CV samples: `r{repeat}-f{fold}-{method}-w{depth}`, one run directory each,
  idempotent via `metrics.json`. Repeat/fold/method/depth are also recorded
  as *structured* fields (`manifest["config"]["cv"]`/`["run"]["tags"]`), not
  parsed back out of the batch_id string.

## What is not yet done

Implemented and unit/integration-tested against small synthetic stores
(`experiments/tests/test_cv.py`, `test_cv_optional.py`, `test_compare.py`) --
including the load-bearing exactness claim itself (a CV sample's assembled
model predicts identically to a direct fit on the same molecule union, for
both predictors) -- but **not yet run against the real corpus**: that means
deleting/rebuilding `experiments/stores/dash-molecules` under the new
90/10 + shard split (a store the existing 10-fold partitions and every run
under `experiments/runs/` currently depend on), choosing `N` from
`cluster-report`'s real output, and then running both workflow scripts'
shard-fit/Study-A/Study-B stages for real -- each a long-running, resource-
heavy operation deferred to a deliberate follow-up rather than done as a
side effect of this implementation pass.
