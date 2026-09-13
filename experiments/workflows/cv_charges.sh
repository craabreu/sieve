#!/usr/bin/env bash
# The charges series' model-evaluation procedure, end to end, as a
# sequence of individually-guarded steps: prepare the corpus, choose a
# shard count from measured evidence, split, fit each shard once, run the
# two CV studies, compare, and score the untouched held-out test split.
#
# Supersedes dash_charges.sh and sieve_charges.sh, which described the
# same procedure twice (once per predictor) and shared a deliberately
# byte-identical data-preparation stage. Under the CV redesign the two
# series genuinely share one store, one shard partition and one fold
# assignment per repeat -- that sharing is what makes their samples
# pairable in compare.py's repeated-measures design -- so describing the
# procedure once is not just tidier, it is what keeps the pairing true.
#
# Run from the repo root: experiments/workflows/cv_charges.sh
# Every step is idempotent, so re-running resumes rather than redoes.
# `CV_UNTIL=<step>` stops after that step (see the step harness below).
#
# Nothing here is tracked by MLflow, deliberately and at every layer:
# cv.py contains no MLflow code at all, and `run` leaves tracking off
# unless given --track, which no step passes. Corpus preparation, shard
# fits and CV samples are all read back off disk by summarize/sweep/
# compare, so a tracking server would add a second, duplicating copy of
# every artifact and nothing else -- which has caused real disk incidents
# on this box before (see runner.execute's own docstring).
#
# Design: docs/superpowers/specs/2026-09-12-cv-shard-redesign-design.md
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Explicit venv interpreter, not bare `python` off PATH -- from the repo
# root, the literal `experiments/` directory shadows the real installed
# package on an interpreter that isn't this project's own .venv (silent
# `No module named experiments.__main__` otherwise).
PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
  echo "no $PYTHON -- run 'uv sync --extra dev --extra chem --extra charges' first" >&2
  exit 1
fi

# ===========================================================================
# The step harness
# ===========================================================================
#
#   step <name> <guard> -- <command...>
#
# `guard` is a shell expression whose *success means "already done"*. It is
# evaluated twice: before the step (to decide whether to skip) and again
# after it (to confirm the step actually produced what it claimed). That
# second evaluation is the part that earns its keep -- a step which
# silently no-ops fails the workflow here, loudly, instead of leaving a
# store or a run directory that misrepresents itself downstream.
#
# Guards deliberately check the *real artifact* -- a parquet's columns, a
# run's metrics.json, a shard's tree_stats.npz -- never a side marker
# recording that something once ran. The two claims differ: a marker says
# "this step executed at some point", an artifact check says "its output
# is here now". Only the second survives the artifact being rebuilt,
# deleted, or half-written underneath it.
#
# Adding a step as the procedure evolves is therefore three lines and one
# guard, with no new bespoke idempotency logic to get subtly wrong.
#
# `CV_UNTIL=<step>` stops cleanly after that step. Not merely a
# convenience: the procedure has a genuine human-in-the-loop break at
# Study A, whose depth curve someone has to *read* before Study B can be
# told which depth to fix. Running to a named stop point is how that break
# is expressed, rather than by commenting steps out and forgetting to put
# them back.

CV_UNTIL="${CV_UNTIL:-}"

STEPS_RUN=()
STEPS_SKIPPED=()
_until_matched=false

step() {
  local name=$1 guard=$2
  shift 2
  [ "${1:-}" = "--" ] && shift

  local last=false
  if [ -n "$CV_UNTIL" ] && [ "$name" = "$CV_UNTIL" ]; then
    last=true
    _until_matched=true
  fi

  if eval "$guard" > /dev/null 2>&1; then
    echo "== skip  $name"
    STEPS_SKIPPED+=("$name")
  else
    echo "== run   $name"
    "$@"
    if ! eval "$guard" > /dev/null 2>&1; then
      echo "!! $name completed but its guard still reports it undone --" >&2
      echo "!! the step did not produce what it promised. Guard: $guard" >&2
      exit 1
    fi
    echo "== done  $name"
    STEPS_RUN+=("$name")
  fi

  if $last; then
    echo "== stop  (CV_UNTIL=$CV_UNTIL reached)"
    exit 0
  fi
}

summarize_steps() {
  local status=$?
  echo
  echo "=== workflow summary ==="
  echo "ran:     ${#STEPS_RUN[@]} step(s)${STEPS_RUN[*]+: ${STEPS_RUN[*]}}"
  echo "skipped: ${#STEPS_SKIPPED[@]} step(s)${STEPS_SKIPPED[*]+: ${STEPS_SKIPPED[*]}}"
  # A CV_UNTIL naming no step at all would otherwise run the whole
  # workflow silently -- the opposite of what was asked for. Only a clean
  # exit proves that, though: a step that died before CV_UNTIL's step was
  # reached leaves _until_matched false for an entirely different reason,
  # and blaming the name there hides the actual failure.
  if [ "$status" -ne 0 ]; then
    echo "!! workflow exited $status -- a step failed; see the traceback above" >&2
  elif [ -n "$CV_UNTIL" ] && ! $_until_matched; then
    echo "!! CV_UNTIL=$CV_UNTIL matched no step; the whole workflow ran" >&2
  fi
}
trap summarize_steps EXIT

# ===========================================================================
# Guard vocabulary
# ===========================================================================

file_exists() { [ -f "$1" ]; }

store_has_columns() {
  local store=$1
  shift
  "$PYTHON" - "$store" "$@" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

store, *columns = sys.argv[1:]
path = Path("experiments/stores") / store / "molecules.parquet"
if not path.exists():
    sys.exit(1)
present = set(pq.ParquetFile(path).schema.names)
sys.exit(0 if set(columns) <= present else 1)
PY
}

# The store is curated *and* the summary describes this very parquet --
# not merely "curation ran here once". Comparing the recorded
# post-curation conformer count against the parquet's actual row count is
# what tells a curated store apart from one re-parsed underneath a
# surviving summary (prepare_dash.curate_conformers makes the same check
# for the same reason).
store_is_curated() {
  local store=$1
  "$PYTHON" - "$store" <<'PY'
import sys
from pathlib import Path

from experiments.prepare_dash import (
    CURATION_SUMMARY,
    _curated_conformer_count,
    _parquet_row_count,
)

store_dir = Path("experiments/stores") / sys.argv[1]
summary = store_dir / CURATION_SUMMARY
if not summary.exists():
    sys.exit(1)
recorded = _curated_conformer_count(summary.read_text())
actual = _parquet_row_count(store_dir / "molecules.parquet")
sys.exit(0 if recorded is not None and recorded == actual else 1)
PY
}

runs_exist() {
  local experiment=$1 batch=$2
  compgen -G "experiments/runs/$experiment/${batch}__*/metrics.json" > /dev/null
}

runs_count_is() {
  local experiment=$1 expected=$2 found
  found=$(ls -d experiments/runs/"$experiment"/*__*/metrics.json 2>/dev/null | wc -l)
  [ "$found" -eq "$expected" ]
}

shard_fits_count_is() {
  local prefix=$1 expected=$2 found
  found=$(ls -d experiments/runs/cv-shard-fits/"$prefix"*__*/tree_stats.npz 2>/dev/null | wc -l)
  [ "$found" -eq "$expected" ]
}

n_items() { echo "$1" | tr ',' '\n' | grep -c .; }

# ===========================================================================
# Configuration
# ===========================================================================

STORE=dash-molecules

# Chosen from cluster-report's own measured output, not guessed: the train
# split holds 313,964 molecules in 17,995 Butina clusters whose largest is
# 3,733 (1.19% of train). That largest cluster is the binding constraint --
# past N ~= 84 it no longer fits inside one shard's target and pins a shard,
# unbalancing the rest (visible at N=100: max=3733 against a 3,140 target,
# std 59.6, against std 0.4 at N=50). N=50 is the largest round value that
# still balances to a single molecule, divides by K, and keeps ~40% headroom
# on that cluster.
N_SHARDS="${CV_N_SHARDS:-50}"
K="${CV_K:-5}"

STUDY_A_REPEATS=0
STUDY_B_REPEATS=0,1,2,3,4

DASH_MAX_DEPTH=16  # the published tree's own ceiling; deeper cannot lengthen a path
DASH_DEPTHS=2,4,6,8,10,12,14,16

SIEVE_CONFIG_LABEL=element-eb
SIEVE_DEPTHS=0,1,2,3,4,5,6,7,8,9,10
SIEVE_PREDICTOR_PARAMS='{"attributes": ["element"], "edge_attributes": [], "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}'
SIEVE_METHOD="sieve-$SIEVE_CONFIG_LABEL"

CODES_PATH="experiments/stores/$STORE/sieve-codes.json"
CLUSTER_REPORT="experiments/results/cluster-report.txt"

# Study B fixes each method at the depth Study A selected. Override once
# Study A's curve has been read; the defaults are each series' own prior
# best, not a result.
# Per-atom predictions are written for Study B only, and only at each
# method's one selected depth -- that is the set a per-element or
# worst-atom error analysis actually reads. Study A deliberately does not:
# across a depth sweep everything in predictions.npz except
# atom_target_pred is identical between depths of a given (repeat, fold),
# so it would write ~14GB to say the same thing 8-11 times over. Study B's
# own cost is ~7.5GB (50 runs x ~151MB); set CV_SAVE_PREDICTIONS=0 to skip
# it and rely on the final-holdout runs, which write theirs regardless.
SAVE_PREDICTIONS_FLAG=""
if [ "${CV_SAVE_PREDICTIONS:-1}" != "0" ]; then
  SAVE_PREDICTIONS_FLAG="--save-predictions"
fi

DASH_SELECTED_DEPTH="${DASH_SELECTED_DEPTH:-16}"
SIEVE_SELECTED_DEPTH="${SIEVE_SELECTED_DEPTH:-6}"

DASH_STUDY_A=dash-cv-study-a
DASH_STUDY_B=dash-cv-study-b
SIEVE_STUDY_A=sieve-cv-study-a
SIEVE_STUDY_B=sieve-cv-study-b
# Tracked, deliberately: experiments/results is gitignored (friction
# observation 3), and the Tukey plot is the study's headline result, not an
# intermediate. It carries its own provenance header (store, commit, run ids).
TUKEY_PLOT=experiments/docs/figures/tukey-study-b.png
# The same test drawn the other way: one interval per method on the metric's
# own scale rather than one per pair on a difference scale (statsmodels'
# plot_simultaneous layout, as used in Pat Walters' ADME model comparison).
# Two methods make a thin plot; it earns its keep once the open-ended list of
# Sieve configs in the design has more than one entry, because it grows as k
# rather than as k-choose-2.
SIMULTANEOUS_PLOT=experiments/docs/figures/simultaneous-study-b.png

# ===========================================================================
# Steps
# ===========================================================================

# --- corpus ---------------------------------------------------------------
#
# Download (~8.3GB, skipped when a correctly-sized copy is present), parse
# the SDF, and curate conformers by the DASH paper's own 0.4 e sibling
# criterion -- stopping *before* the split, so the next step can choose
# N_SHARDS from real evidence rather than a guess.
step prepare-corpus \
  "store_is_curated $STORE" -- \
  "$PYTHON" -m experiments prepare-store "$STORE" --stop-before-split

# --- choose the shard count ------------------------------------------------
#
# Advisory and read-only: reports the train-split cluster size distribution
# and the shard balance each candidate N would actually achieve. It informs
# N_SHARDS above (which is where the decision is recorded durably -- this
# file, in git, not the report, which lands in gitignored results/).
step cluster-report \
  "file_exists $CLUSTER_REPORT" -- \
  bash -c "set -euo pipefail; mkdir -p \"\$(dirname '$CLUSTER_REPORT')\" && \
           '$PYTHON' -m experiments cluster-report '$STORE' \
             --candidates 10,20,25,50,100,200 | tee '$CLUSTER_REPORT'"

# --- split -----------------------------------------------------------------
#
# 90/10 train/test by whole Butina cluster, plus the cluster ids and the
# N_SHARDS cluster-clean train shards, all from one clustering pass.
step split-store \
  "store_has_columns $STORE split cluster shard" -- \
  "$PYTHON" -m experiments prepare-store "$STORE" --n-shards "$N_SHARDS"

# --- freeze the Sieve vocabulary -------------------------------------------
#
# Blocking for Sieve, not optional: a shard that discovers its own
# attribute_codes from its own ~1/N of train can disagree with its siblings
# on what an integer code means, and attribute_codes feeds schema_version,
# which check_mergeable compares exactly. Freezing one vocabulary over the
# whole train split is what makes the shards mergeable at all.
step sieve-codes \
  "file_exists $CODES_PATH" -- \
  "$PYTHON" -m experiments build-sieve-codes "$STORE" \
    --attributes element --edge-attributes "" --out "$CODES_PATH"

# --- shard fits ------------------------------------------------------------
#
# One fit per shard, predicting nothing (a single shard is only ever used
# merged) -- and for both predictors, one set at the deepest depth serves
# every shallower one. DASH's node stats are depth-invariant and its paths
# prefix-nested; Sieve's levels are bottom-up, so a deep fit's levels 0..d
# are exactly a depth-d fit's, recovered by truncating the *merged* model
# (cv.truncate_model). The one exception is DASH depth 1, which truncation
# cannot reproduce (the H-atom redirect consumes a depth unit before
# max_depth is checked) -- hence DASH_DEPTHS starts at 2, and run_dash_cv
# refuses anything shallower rather than scoring it wrongly.
# Shards are independent and each fit is ~1 minute of real work, so the
# unit of parallelism is one shard per process, dispatched with xargs -P --
# the same shape the pre-CV workflows used for folds, and for the same
# reason: it is what actually uses this box's headroom (64 cores / 503GB).
# Running these sequentially cost 53.8 min for DASH's 50 shards where a
# filled dispatch queue is ~1-2 min.
#
# design.md 5.5's warning about process overhead does not bite here: it
# concerns worker pools spun up per `fit()` call, where startup (260-390ms)
# rivals the fit itself. A shard fit is ~65s -- three orders of magnitude
# above that -- so the per-process DASHTree preload is noise by comparison.
# Nor does friction observation 7 (repeated identical work across
# processes): every process here fits a *different* shard.
#
# Each process is left single-threaded (`n_jobs` unset) because the
# dispatch already fills the box. That is the old workflows' own rule:
# dispatch N single-threaded processes, or run one process with `n_jobs`,
# never both.
#
# The job counts are set from measured peak RSS, not from `nproc` and not
# from a guess -- a guess of 16 already cost one OOM-killed shard here:
#
#   dash  shard fit, max_depth=16 : 35.1 GB peak RSS, 2m07 wall
#   sieve shard fit, max_wl_depth=10: 32.0 GB peak RSS, 1m11 wall
#
# 16 x 32GB = 512GB against this box's 503GB, which is exactly why s10 was
# killed. A ~400GB budget (leaving the OS and page cache room) allows ~11
# concurrent fits of either kind; 8 keeps real headroom for shard-to-shard
# variance while still being 8x a sequential run.
#
# The non-obvious part, worth stating because it defeats the intuition that
# sharding shrinks memory: a Sieve fit's footprint is driven by **depth**,
# not by shard size. 32GB here is for 1/50th of train at depth 10, against
# the old workflow's ~37GB for the *whole* corpus at depth 6 -- the class
# count is what explodes, and WL depth is what explodes it. Fitting more,
# smaller shards therefore does not buy proportionally more concurrency.
DASH_SHARD_JOBS="${DASH_SHARD_JOBS:-8}"
SIEVE_SHARD_JOBS="${SIEVE_SHARD_JOBS:-8}"
SIEVE_MAX_DEPTH="${SIEVE_MAX_DEPTH:-10}"  # the deepest SIEVE_DEPTHS asks for

fit_one_dash_shard() {
  "$PYTHON" -m experiments cv-fit-dash-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$DASH_MAX_DEPTH" --shard "$1"
}

fit_one_sieve_shard() {
  "$PYTHON" -m experiments cv-fit-sieve-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$SIEVE_MAX_DEPTH" --shard "$1" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS"
}
export -f fit_one_dash_shard fit_one_sieve_shard
export PYTHON STORE N_SHARDS DASH_MAX_DEPTH SIEVE_MAX_DEPTH
export CODES_PATH SIEVE_CONFIG_LABEL SIEVE_PREDICTOR_PARAMS

# Each shard's own fit is idempotent, and xargs hands a given shard to
# exactly one process, so an interrupted dispatch resumes cleanly -- and
# under a different job count, as the old workflows also guaranteed.
all_shard_ids() { seq -f "s%02g" 0 $((N_SHARDS - 1)); }

dispatch_dash_shards() {
  all_shard_ids | xargs -P "$DASH_SHARD_JOBS" -n 1 \
    bash -c 'fit_one_dash_shard "$1"' --
}

dispatch_sieve_shards() {
  all_shard_ids | xargs -P "$SIEVE_SHARD_JOBS" -n 1 \
    bash -c 'fit_one_sieve_shard "$1"' --
}

step dash-shard-fits \
  "shard_fits_count_is fit-dash-s $N_SHARDS" -- \
  dispatch_dash_shards

step sieve-shard-fits \
  "shard_fits_count_is fit-sieve-$SIEVE_CONFIG_LABEL-w$SIEVE_MAX_DEPTH-s $N_SHARDS" -- \
  dispatch_sieve_shards

# --- Study A: depth selection ----------------------------------------------
#
# One repeat, K folds, no ANOVA and no Tukey -- just a depth curve per
# method, each fold's training model assembled by merging the other K-1
# groups' shards. Read the normalized curve (the form each method is
# actually deployed in) to pick the depths Study B then fixes:
#   "$PYTHON" -m experiments sweep --experiment dash-cv-study-a \
#     --x config.cv.depth --metric norm/mae --metric mae
step study-a-dash \
  "runs_count_is $DASH_STUDY_A $((K * $(n_items "$DASH_DEPTHS")))" -- \
  "$PYTHON" -m experiments cv-run-dash "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" --max-depth "$DASH_MAX_DEPTH" \
    --depths "$DASH_DEPTHS" --repeats "$STUDY_A_REPEATS" \
    --normalization std_weighted --method dash --experiment "$DASH_STUDY_A"

step study-a-sieve \
  "runs_count_is $SIEVE_STUDY_A $((K * $(n_items "$SIEVE_DEPTHS")))" -- \
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$SIEVE_DEPTHS" --repeats "$STUDY_A_REPEATS" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --fit-depth "$SIEVE_MAX_DEPTH" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS" \
    --normalization equal_weighted --method "$SIEVE_METHOD" \
    --experiment "$SIEVE_STUDY_A"

# --- Study B: model comparison ---------------------------------------------
#
# Five repeats x K folds = 25 samples per method at its selected depth.
# Repeat 0 reuses Study A's own partition, so only four are new work. Both
# methods draw their folds from permute_into_folds(shard_ids, k, seed=repeat),
# which is deterministic in (n, k, seed) and not in the caller -- so a given
# (repeat, fold) holds out the *same* molecules for both, which is what makes
# them pairable subjects in compare's repeated-measures design.
# One process per repeat. A repeat's five folds share its own merge
# assembly so they stay together, but repeats are fully independent -- and
# the per-fold walk is the cost, so this is ~5x. Job count kept low because
# each process holds a DASHTree (~35GB measured) plus its merged stats.
STUDY_B_JOBS="${STUDY_B_JOBS:-3}"

run_dash_repeat() {
  "$PYTHON" -m experiments cv-run-dash "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" --max-depth "$DASH_MAX_DEPTH" \
    --depths "$DASH_SELECTED_DEPTH" --repeats "$1" \
    --normalization std_weighted --method dash --experiment "$DASH_STUDY_B" \
    $SAVE_PREDICTIONS_FLAG
}

run_sieve_repeat() {
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$SIEVE_SELECTED_DEPTH" --repeats "$1" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --fit-depth "$SIEVE_MAX_DEPTH" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS" \
    --normalization equal_weighted --method "$SIEVE_METHOD" \
    --experiment "$SIEVE_STUDY_B" \
    $SAVE_PREDICTIONS_FLAG
}
export -f run_dash_repeat run_sieve_repeat
export DASH_SELECTED_DEPTH SIEVE_SELECTED_DEPTH SAVE_PREDICTIONS_FLAG
export DASH_STUDY_B SIEVE_STUDY_B SIEVE_METHOD K DASH_MAX_DEPTH

each_repeat() { echo "$STUDY_B_REPEATS" | tr ',' '\n'; }

dispatch_dash_repeats() {
  each_repeat | xargs -P "$STUDY_B_JOBS" -n 1 bash -c 'run_dash_repeat "$1"' --
}

dispatch_sieve_repeats() {
  each_repeat | xargs -P "$STUDY_B_JOBS" -n 1 bash -c 'run_sieve_repeat "$1"' --
}

step study-b-dash \
  "runs_count_is $DASH_STUDY_B $((K * $(n_items "$STUDY_B_REPEATS")))" -- \
  dispatch_dash_repeats

step study-b-sieve \
  "runs_count_is $SIEVE_STUDY_B $((K * $(n_items "$STUDY_B_REPEATS")))" -- \
  dispatch_sieve_repeats

# --- compare ---------------------------------------------------------------
#
# Repeated-measures ANOVA + Tukey HSD over Study B's 25 paired samples per
# method (Ash, Wognum, Rodriguez-Perez et al., JCIM 2025). Read the effect
# sizes and interval widths, not only the stars: at ~62.8k held-out
# molecules per fold, a systematic difference of almost any size will reach
# significance.
#
# Compared on the *unnormalized* metric by default. Two reasons. The two
# methods currently use different normalizers, so part of any gap in
# norm/mae would be the normalizer rather than the estimator. And Sieve's
# is equal_weighted (sigma^0), which design.md 13 item 8 measured as worse
# than not normalizing at all on MAE -- the right sigma for Sieve is the
# predictive variance of docs/design-update-v2-predictive-variance, not yet
# ported onto this layout. The raw metric compares estimator to estimator,
# with neither given the molecule's true total charge.
#
# RMSE rather than MAE, because it is the metric the DASH paper itself
# reports throughout -- its GNN at 0.0153 e, and the conformational-
# variation floor of 0.0125 e it calls "a lower bound on the accuracy that
# can be reached by an ML model" -- so these numbers are directly
# comparable to the published ones. Both are recorded in every run, so
# COMPARE_METRIC=mae re-runs the comparison on MAE without re-running
# anything.
COMPARE_METRIC="${COMPARE_METRIC:-rmse}"
step compare \
  "file_exists $TUKEY_PLOT && file_exists $SIMULTANEOUS_PLOT" -- \
  bash -c "set -euo pipefail; mkdir -p \"\$(dirname '$TUKEY_PLOT')\" && \
           '$PYTHON' -m experiments compare \
             --experiment '$DASH_STUDY_B' --experiment '$SIEVE_STUDY_B' \
             --metric "$COMPARE_METRIC" \
             --depth-by-method '{\"dash\": $DASH_SELECTED_DEPTH, \"$SIEVE_METHOD\": $SIEVE_SELECTED_DEPTH}' \
             --out '$TUKEY_PLOT' \
             --out-simultaneous '$SIMULTANEOUS_PLOT'"

# --- final held-out evaluation ---------------------------------------------
#
# Once, at the end, outside both studies: merge every shard into one
# full-train model and score the untouched 10% test split. The headline
# number, and the only thing here that touches `test`.
DASH_MERGED=experiments/results/dash-merged/tree_stats.npz
SIEVE_MERGED="experiments/results/sieve-merged/tree_stats-w${SIEVE_SELECTED_DEPTH}.npz"

# Merge order does not affect the result (both merges are commutative and
# associative); sorted purely for a readable command line.
step merge-dash-shards \
  "file_exists $DASH_MERGED" -- \
  bash -c "set -euo pipefail; '$PYTHON' -m experiments merge-states --predictor dash \
             --out '$DASH_MERGED' \
             \$(ls experiments/runs/cv-shard-fits/fit-dash-s*__*/tree_stats.npz | sort)"

step merge-sieve-shards \
  "file_exists $SIEVE_MERGED" -- \
  bash -c "set -euo pipefail; '$PYTHON' -m experiments merge-states --predictor sieve \
             --out '$SIEVE_MERGED' \
             \$(ls experiments/runs/cv-shard-fits/fit-sieve-$SIEVE_CONFIG_LABEL-w${SIEVE_SELECTED_DEPTH}-s*__*/tree_stats.npz | sort)"

step final-holdout-dash \
  "runs_exist dash-final-holdout final" -- \
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_depth="$DASH_SELECTED_DEPTH" \
    --set tree_stats_load_path="$DASH_MERGED" \
    --set normalization=std_weighted \
    --set run.experiment=dash-final-holdout \
    --set run.batch_id=final

step final-holdout-sieve \
  "runs_exist sieve-final-holdout final" -- \
  "$PYTHON" -m experiments run \
    --config experiments/configs/sieve-charge-sweep.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_wl_depth="$SIEVE_SELECTED_DEPTH" \
    --set tree_stats_load_path="$SIEVE_MERGED" \
    --set normalization=equal_weighted \
    --set run.experiment=sieve-final-holdout \
    --set run.batch_id=final
