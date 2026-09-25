# Study F: Calibration of Sieve's Predictive Variance

**Date:** 2026-09-25
**Status:** design, awaiting review; not implemented
**Builds on:** `2026-09-25-within-structure-variance-design.md` (form B, merged in PR #47).

---

## 1. Purpose

The paper needs one reproducible study of Sieve's uncertainty, run by the workflow like Studies A–E. This spec is that study, Study F, which measures two things:

- **Calibration:** how well Sieve's predictive variance matches its errors.
- **Normalisation:** what the variance buys when it weights the per-molecule charge correction.

Both are measured for the shipped form B and for four ablations, each of which removes one of form B's ingredients.

Everything here was measured once already, in scratchpad scripts, on 2026-09-25 (spec above, §1, and the phase-1 evaluation). Study F turns that into tested code, a workflow step and a committed report.

It is **Sieve only**: DASH's σ is out of scope. It produces a **text report only**, with no figures or LaTeX.

## 2. What is measured

### 2.1 Samples

These are the Study B incumbent's runs: experiment `$SIEVE_STUDY_B`, method `sieve-element-continuation-eb`, at depth `$SIEVE_SELECTED_DEPTH`. That is 5 repeats × K = 5 folds = 25 samples, from the train split only; the test split is never read.

For each run, Study F loads the fold model the run itself was scored with. It is truncated to the run's depth and read with continuation + EB. It is given `predictive_variance=True` and the pooled σ²_w of its training shards from the floor cache (`_attach_training_floors`, with `collapse` taken from the run's manifest).

Study F then predicts the run's held-out conformers once. **The predictions must equal the run's `predictions.npz` to 1e-12, and atoms must line up by `(dash_id, conf_id)`.** Otherwise the step refuses, since it would be describing a different model from the one Study B scored.

### 2.2 Arms

Each arm is a per-atom σ², computed from the same prediction. Unmatched atoms take the model's global fallback, `global_msd + σ²_w`, in the arms that include σ²_w, and `global_msd` otherwise.

| arm | within-class | α^v | selection (a) | mean estimation | σ²_w |
|---|---|---:|---:|---|---|
| `form_b` (shipped) | yes | 10 | 0.5 | yes | yes |
| `no_sigma2_w` | yes | 10 | 0.5 | yes | **no** |
| `alpha_v_30` | yes | **30** | 0.5 | yes | yes |
| `no_selection` | yes | 10 | **0** | yes | yes |
| `no_estimation` | yes | 10 | 0.5 | **no** | yes |

The variance before this work, the old three-term form, is not a separate arm. It is `no_sigma2_w` at α^v = 30, and the report says so. Its numbers are already in the within-structure spec.

### 2.3 Metrics

These are computed per arm and per sample. Let z = e/σ, where e is the prediction minus the reference:

- `nll`: the Gaussian mean ½(log 2πσ² + z²);
- `ez2`: the mean of z²;
- `cov50`, `cov90`, `cov95`, `cov99`: the share of atoms with |z| below the normal quantile for each nominal level;
- `ez2_k{k}`: the mean of z² among atoms whose deepest matched radius k\* = k, for every k from 0 to the depth. It is absent where no atom matched at k;
- `rho`: the Pearson correlation, over all atoms, between |e| and σ² after each is converted to a percentile rank within its own conformer. This is how well σ orders a molecule's atoms, which is what normalisation relies on;
- `norm_rmse`, `norm_mae`: RMSE and MAE after `normalize.variance_weighted_normalize` with this arm's σ, and each molecule's total constrained to its `molecule_value`.

Per sample, independent of arm, Study F also records `equal/norm_rmse` and `equal/norm_mae` (`equal_weighted_normalize`), plus `n_atoms` and `sigma2_w`.

Sidecar keys are `calibration/<arm>/<metric>` and `calibration/equal/<metric>`. None of these can collide with `metrics.json`.

## 3. Components

### 3.1 `sieve.uncertainty.variance_terms` (library)

```python
def variance_terms(model, *, alpha_v=ALPHA_V, alpha_t=ALPHA_T) -> list[dict[str, np.ndarray]]
```

It returns one dict per level, with keys `within`, `selection`, `estimation` and `within_structure`, each broadcastable to `(n_classes, d)`.

`predictive_variance` becomes `within + selection_weight * selection + estimation + within_structure`, built from these terms, and it must stay **bit-identical** to its current output. This is the only library change. An ablation is then a sum of terms, not a flag per term.

### 3.2 `experiments/experiments/calibration.py` (new)

This module mirrors `stereo_subsets.py`.

- `METRICS_FILE = "calibration_metrics.json"` and `ARMS`, the table in §2.2 as data.
- `calibration_metrics(e, sigma2, k_star, conf_index, mu, molecule_value) -> dict[str, float]`. It is pure NumPy, with no model involved, which keeps it unit-testable on synthetic arrays.
- `arm_variances(model, prediction) -> dict[str, np.ndarray]`: the per-atom σ² of every arm, from `variance_terms` and the prediction's `matched_level` and `class_id`.
- `score_run(run_dir, *, store, stores_root, runs_root, model_cache) -> dict`. It does §2.1's loading, the prediction-equality check, the arms and the metrics.
- `missing_scores` and `score_runs`, with the same contract as Study D's. A selected run is stale if it has no sidecar, if the sidecar is older than its `predictions.npz`, or if the sidecar lacks any arm's keys.
- `calibration_report(runs_root, *, experiment, method, depth, k) -> str`. This is the text report of §4.

**Fold models.** `run_sieve_cv` assembles them inline today: first the model cache, then the shard fits merged with `leave_one_group_out`. That assembly is factored out into `experiments.cv.sieve_train_models(...)`, which returns the K fold models of one repeat. `run_sieve_cv` and Study F both call it, and `run_sieve_cv`'s behaviour and outputs are unchanged. `_attach_training_floors` stays at `run_sieve_cv`'s call site, and Study F calls it too.

### 3.3 `aggregate.read_runs_from_dirs`

It merges `calibration_metrics.json` beside `subset_metrics.json`, under the same clash rule: a key in both a sidecar and `metrics.json` is refused. After that, `compare.read_cv_table` reads `calibration/<arm>/<metric>` like any other metric.

### 3.4 CLI

- `experiments score-calibration <store> --experiment E --method M --depth D [--repeats 0,1,...] [--check] [--force]`. With `--check`, it exits non-zero when any selected run is stale and writes nothing. `--repeats` lets the workflow fan out.
- `experiments calibration-report --experiment E --method M --depth D --k K --out PATH`. It writes `PATH.txt`.

### 3.5 Workflow (`cv_charges.sh`)

Study F comes after Study E and before the final held-out evaluation.

```bash
step study-f-scores \
  "'$PYTHON' -m experiments score-calibration '$STORE' --experiment $SIEVE_STUDY_B \
     --method sieve-element-continuation-eb --depth $SIEVE_SELECTED_DEPTH --check" -- \
  dispatch_study_f_repeats      # one process per repeat, xargs -P "$STUDY_F_JOBS" (default 5)

step study-f-report "study_f_report_is_up_to_date" -- run_study_f_report
```

`study_f_report_is_up_to_date` holds when `$FIGURES_DIR/study-f.txt` exists and no `calibration_metrics.json` under `$SIEVE_STUDY_B` is newer. This is the same shape as Study D's guard.

The report is committed by the workflow's existing `commit_figures`.

**Cost.** About 1 minute per sample, one featurisation each. That is about 25 minutes serially, or about 5 minutes with 5 parallel jobs. No refits.

## 4. The report (`study-f.txt`)

The report has four blocks. Numbers carry six decimals for normalised errors and four elsewhere.

1. **Header.** The samples (experiment, method, depth, n) and the pooled σ²_w range across folds. It also says which arms reproduce the old variance: `no_sigma2_w` at α^v = 30, whose numbers are in the within-structure spec.
2. **Means per arm, over all 25 samples.** Columns: `nll`, `ez2`, `cov50`, `cov90`, `cov95`, `cov99`, `rho`, `norm_rmse`, `norm_mae`. Extra rows give `equal weights` (normalised errors only) and `unnormalised` (`rmse` and `mae` from `metrics.json`).
3. **E[z²] by k\*.** One row per arm, one column per k from 0 to the depth, and the share of atoms at each k from `form_b`'s sample.
4. **Paired differences, each ablation minus `form_b`,** on `nll`, `norm_rmse` and `norm_mae`, plus `form_b` minus `equal` on the two normalised errors. Each row gives the mean, the Nadeau–Bengio 95% interval (`stereo_subsets.paired_difference`, k = K), the relative change and the win count. There are two sub-blocks: all 25 samples, then repeats 1–4 alone, because α^v was tuned on repeat 0.

The report is deterministic given the sidecars, so running it twice gives byte-identical output.

## 5. Tests (written before the code)

1. **`variance_terms`:** their weighted sum equals `predictive_variance` bit for bit, across stereo tracks and with and without σ²_w.
2. **`calibration_metrics` on synthetic arrays:**
   - z drawn from N(0, 1) gives `ez2` ≈ 1 and coverage ≈ nominal;
   - doubling σ divides `ez2` by 4;
   - a hand-computed NLL;
   - `ez2_k` keys only for radii that occur;
   - `norm_*` equals `variance_weighted_normalize` applied directly;
   - equal weights equal `equal_weighted_normalize`.
3. **`arm_variances`:** `form_b` equals `predict_detailed(...).predictive_variance`. `no_sigma2_w` is `form_b − σ²_w` on matched atoms and `global_msd` on unmatched ones. `alpha_v_30` equals `predictive_variance(model, alpha_v=30)` indexed per atom.
4. **`sieve_train_models`:** it returns the same models as `run_sieve_cv`'s previous inline assembly, both from the cache and from shards. The existing `run_sieve_cv` tests stay green unchanged.
5. **`score_run` on a small synthetic store** (`experiments/tests` fixtures):
   - the sidecar has every arm's keys;
   - a run whose `predictions.npz` disagrees with the model is refused;
   - `missing_scores`: absent sidecars, stale ones and ones missing an arm are reported, and fresh ones are not.
6. **`aggregate`** merges the sidecar and refuses a clash.
7. **`calibration_report`** on hand-written sidecars: the blocks and the paired-difference numbers against `paired_difference` directly, and byte-identical output across runs.
8. **CLI:** `--check` exits non-zero on a stale run and writes nothing.

## 6. Acceptance on the real store

Run `study-f-scores` and then `study-f-report` on `dash-molecules`. The `form_b` means must reproduce the phase-1 evaluation to four figures:

- NLL −2.8365;
- E[z²] 0.9588;
- cov95 0.9563;
- normalised RMSE 0.018876;
- and, for `form_b` minus `equal`, a normalised RMSE difference of −7.0 × 10⁻⁴ lower in 25/25.

`alpha_v_30` reproduces the evaluation's B30 column: NLL −2.8309, E[z²] 0.9673.

## 7. Not in scope

- DASH's σ, and other methods' normalisation.
- Figures and LaTeX tables. A later spec can read the sidecars.
- Per-class σ²_w, which was rejected (within-structure spec §4.1).
- Conformal intervals by k\*.
- The heavy-tail analysis (Student-t fits, tail composition). The report shows the tails only through the coverage columns.
- Changing any study's reported normalisation, including the final held-out evaluation's.
