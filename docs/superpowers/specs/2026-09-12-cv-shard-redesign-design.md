# Design: cluster-clean-shard cross-validation for model comparison

**Status:** implemented (code + tests); not yet run against the real corpus
**Date:** 2026-09-12
**Scope:** `experiments/experiments/prepare_dash.py` (store split), `cv.py`
(new), `compare.py` (new), `predictors/sieve_predictor.py`,
`predictors/dash.py`, `tree_artifact.py`, `store_ops.py`, `config.py`,
`cli.py`, the workflow scripts. No change to `sieve`'s core (`src/sieve/**`).

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

**Correction (this claim was wrong as first written).** An earlier version
of this section said Sieve must fit one shard set *per depth*, because under
`class_estimator="continuation"` a shallow config "is not a truncation of a
deep one". What is actually true is narrower: a shallow depth's *prediction*
cannot be read out of a deep model's *output*, since the deepest level is
read differently from the rest. The *stored statistics* are another matter —
WL refinement never looks ahead, so a depth-*D* fit's levels `0..d` are
bit-identical to a native depth-*d* fit's. `cv.truncate_model` rebuilds the
config at `max_wl_depth=d` and slices the levels, reproducing a native fit
exactly: same `schema_version`, identical predictions at every depth,
pinned by `test_truncate_model_matches_a_native_fit_at_every_depth`.

So **both** predictors fit one shard set, at the deepest depth needed, and
derive every shallower depth — DASH by truncating its prefix-nested paths,
Sieve by truncating the merged model's levels. The ordering constraint is
real and was the part the original reasoning got right: `max_wl_depth` feeds
`schema_version`, so truncation must come *after* merging, never before.
Merging first is also cheaper, since one merge then serves every depth.

The one exception is **DASH depth 1**, which truncation gets measurably
wrong (mae 0.0937 vs 0.1315): `_get_init_layer` redirects a hydrogen to its
heavy neighbour and consumes a depth unit before `max_depth` is checked, so
an H atom's depth-1 and depth-2 requests resolve to the same path.
`dash_depth_sweep._MIN_DERIVABLE_DEPTH` has encoded this since before the
redesign; `run_dash_cv` now refuses depths below it rather than scoring them
wrongly.

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
both predictors).

## Follow-up: the workflow, and what running it for real turned up

The two per-predictor scripts were replaced by a single
`experiments/workflows/cv_charges.sh`, because under this design the two
series share one store, one shard partition and one fold assignment per
repeat -- and that sharing is exactly what makes their samples pairable in
`compare.py`'s repeated-measures design, so describing the procedure twice
risked the pairing silently drifting apart.

It is built as guarded steps (`step <name> <guard> -- <command>`) whose
guards check the **real artifact** -- a parquet's columns, a run's
`metrics.json`, a shard's `tree_stats.npz` -- never a side marker
recording that something once ran. Each guard is re-evaluated *after* its
step, so a step that silently no-ops fails loudly instead of leaving an
artifact that misrepresents itself. `CV_UNTIL=<step>` stops after a named
step, which the procedure genuinely needs: Study A's depth curve has to be
read by a person before Study B can be told which depth to fix.

Two defects surfaced while rebuilding the real corpus, both now fixed:

- **`curate_conformers` trusted its own marker.** The skip fired on
  `curation_summary.txt` merely existing, which records that curation once
  ran -- a different claim from "this parquet is curated". Deleting
  `molecules.parquet` while leaving the summary made `prepare_store`
  re-parse (uncurated) and then skip curation, splitting a store that
  claimed a curation it never received. The skip now also compares the
  summary's own recorded post-count against the parquet's actual row
  count, and re-curates on a mismatch.
- **The parsed-and-curated-but-unsplit state had no supported way to
  exist.** Both workflow scripts told the reader to run `cluster-report`
  against such a store, but `prepare-store` always ran through to
  `assign_splits`; reaching it meant calling the module's stages by hand.
  `--stop-before-split` makes it a first-class, idempotent step.
- **`predictions.npz` was specified opt-in and shipped opt-out.** The
  `save_predictions` flag existed on the writer but was never threaded
  through the drivers, so it defaulted to on: ~14GB for Study A and ~7.5GB
  for Study B, most of it duplication (everything but `atom_target_pred` is
  identical across the depths of one (repeat, fold)). Now threaded, default
  off, and enabled by the workflow for Study B alone -- the one selected
  depth per method, which is the set a per-atom error analysis actually
  reads.
- **The workflow was written sequential, discarding the old scripts' own
  parallel dispatch.** The pre-redesign scripts dispatched one process per
  fold with `xargs -P`, with concurrency defaults justified by measured RSS
  ("a single fold peaked around 8GB"; "~37GB at depth 6 with n_jobs=8, so
  the default is deliberately far below the fold sweep's"), and the rule
  that dispatch and `n_jobs` are alternatives, never both. Rewriting from
  scratch lost all of it: DASH's 50 shard fits ran sequentially in 53.8
  minutes on a 64-core box where a filled dispatch queue is 1-2 minutes.
  Restored, with `--shard` on both shard-fit commands as the dispatch seam.
  design.md 5.5's warning about process overhead does not apply at this
  granularity -- it concerns pools spun up per `fit()` call, where startup
  rivals a sub-second fit; a shard fit is ~65s.
