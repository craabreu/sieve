#!/usr/bin/env bash
# Reproduces the Sieve-charges experiment series end to end, as the
# structural parallel of experiments/workflows/dash_charges.sh: the same
# prepared store, the same 10 disjoint folds, a WL-depth sweep across
# them, per-fold normalization of the deepest fit, and one full-corpus
# fit. Stage 1 is shared with the DASH workflow verbatim, which is what
# makes the two series comparable fold for fold.
#
# Every step below is a plain call to the `experiments` CLI (see
# experiments/README.md) -- this script only fixes the sequence and the
# arguments, so each stage can also be re-run by hand exactly as shown.
# Run from the repo root: experiments/workflows/sieve_charges.sh
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

CONFIG=experiments/configs/sieve-charge-sweep.yaml

# --- Stage 1: data preparation -------------------------------------------
#
# Byte-identical to dash_charges.sh's own Stage 1, deliberately: both
# series read the *same* partition, so their per-fold numbers are
# comparable point by point rather than merely on average. Both commands
# are idempotent, so on a box that already ran the DASH workflow this
# whole stage is a no-op.
#
# `prepare-store` is idempotent at each of its three stages (download,
# parse, split). It downloads the real ~8.3GB dashMoleculesSDF_v2.sdf
# (ETH Research Collection) and writes experiments/stores/dash-molecules
# (~1M conformers, ~9.6GB parquet).
"$PYTHON" -m experiments prepare-store

# `partition-store` divides dash-molecules' entire molecule set into 10
# disjoint folds -- every conformer of every molecule kept, none used
# twice -- named dash-molecules-10fold-1 .. dash-molecules-10fold-10.
# Idempotent as a whole and deterministic at the default seed (0).
"$PYTHON" -m experiments partition-store dash-molecules-10fold --n-stores 10

# --- Stage 2: Sieve WL-depth sweep ---------------------------------------
#
# One plain `run` per (max_wl_depth, fold) pair. Unlike the DASH depth
# sweep there is no sweep module here, and under this series' own
# `class_estimator: continuation` there could not be one: an independent
# fit per depth is *required*, not merely equivalent. Level k's estimate
# depends on whether k is the model's own deepest level --
# continuation._child_of_level returns -1 for the level with no children
# on the backoff path, so level 3 is an atom-weighted pooled mean in a
# depth-3 fit and the unweighted mean of its children in a depth-6 one --
# and empirical-Bayes alpha_k = tau^2_k / tau^2_{k-1} is estimated per
# level against that same population. Nothing shallower can therefore be
# read out of one deep fit by truncation.
#
# (Under the pooled estimator the shallower depths *are* recoverable
# from a deep fit, which is what
# docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md
# verified on a real corpus when it dropped the per-level in-run design.
# That equivalence is what does not survive continuation. It never
# motivated a sweep module either way: a sieve fit on a real fold is a
# fraction of a DASH tree-matching walk on the same fold, which is the
# other half of why dash_depth_sweep.py exists and this doesn't.)
#
# `run.batch_id=w<depth>-f<fold>` is what makes the grid navigable and
# resumable: `_run_name` puts it *ahead* of the predictor/store/seed part
# of the run directory name, so every point of one depth greps and sorts
# together (`runs/sieve-depth-sweep/w6-f*`). Idempotent per pair via a
# metrics.json existence check against that prefix -- `run` itself always
# creates a fresh timestamped/uuid'd directory and has no such check on
# its own -- so an interrupted sweep resumes by just running this script
# again, including under a different SIEVE_DEPTH_SWEEP_JOBS.
#
# Parallelism is over the 70 (depth, fold) pairs rather than over folds:
# with no shared fit to hoist, the pair *is* the unit of work here.
# Each process is left single-threaded (`n_jobs` unset) since the
# dispatch already fills the box.
#
# No `report_loo`, and not by choice: SievePredictor refuses it at
# construction under either of this series' two estimator settings
# (`sieve.predict_loo` supports neither `class_estimator=continuation`
# nor a `shrinkage_weight` outside (None, "count") -- a deliberate scope
# cut in sieve.predict._search, not a structural one). So the sweep
# records train/* but no train_loo/*, and the train/train_loo
# memorization gap (design.md 10.3) is not available for this series
# until predict_loo grows continuation support. The in-sample train/*
# numbers stay optimistic at minimum_support=1 and should be read as
# such.
#
# The deepest depth additionally saves its fitted model
# (`save_tree_stats`), so Stage 3 is predict-only. Only the deepest:
# `sieve` models are O(60MB) each and only the deepest is what Stage 3
# wants.
#
# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment sieve-depth-sweep \
#     --x predictor.params.max_wl_depth \
#     --metric mae --metric r2 \
#     --split test --split train
EXPERIMENT=sieve-depth-sweep
DEPTHS="0 1 2 3 4 5 6"
DEEPEST=6
N_FOLDS=10
PARALLEL_JOBS="${SIEVE_DEPTH_SWEEP_JOBS:-$N_FOLDS}"

run_pair() {
  local depth=$1
  local fold=$2
  local batch="w${depth}-f${fold}"
  if compgen -G "experiments/runs/$EXPERIMENT/${batch}__*/metrics.json" \
    > /dev/null; then
    echo "skip $batch (already done)"
    return 0
  fi
  local extra=()
  if [ "$depth" -eq "$DEEPEST" ]; then
    extra=(--set save_tree_stats=true)
  fi
  "$PYTHON" -m experiments run \
    --config "$CONFIG" \
    --set data.store=dash-molecules-10fold-"$fold" \
    --set predictor.params.max_wl_depth="$depth" \
    --set run.experiment="$EXPERIMENT" \
    --set run.batch_id="$batch" \
    ${extra[@]+"${extra[@]}"}
}
export -f run_pair
export PYTHON CONFIG EXPERIMENT DEEPEST

for depth in $DEPTHS; do
  for fold in $(seq 1 "$N_FOLDS"); do
    echo "$depth $fold"
  done
done | xargs -P "$PARALLEL_JOBS" -n 2 bash -c 'run_pair "$1" "$2"' --

# --- Stage 3: normalize the deepest fit, per fold ------------------------
#
# The parallel of dash_charges.sh's own Stage 3: post-hoc charge
# conservation applied to each fold's deepest prediction.
# `tree_stats_load_path` loads that fold's own saved model (Stage 2), so
# this is predict-only -- no re-fit -- and `normalization` re-scores
# against the sum constraint. One process per fold, idempotent (skip a
# fold once its own run directory exists).
#
# `equal_weighted`, not the `std_weighted` the DASH workflow uses:
# sieve's own atom_std is NaN wherever a class has support 1 (its
# genuine "no spread observed" case, not a filler), and
# predictors/sieve_predictor.py's docstring defers std_weighted until
# that atom_std has been checked against real data. equal_weighted needs
# no std at all.
#
# LOO does not enter here either -- see Stage 2 on why this series
# cannot report it at all.
NORM_EXPERIMENT=sieve-depth${DEEPEST}-equal-weighted

normalize_fold() {
  local fold=$1
  if compgen -G "experiments/runs/$NORM_EXPERIMENT/f${fold}__*/metrics.json" \
    > /dev/null; then
    echo "skip normalize f${fold} (already done)"
    return 0
  fi
  local shard
  # Guarded, not just globbed: `ls | head -1` hides its own failure even
  # under pipefail (head succeeds), which would hand `run` an empty
  # tree_stats_load_path and quietly re-fit instead of loading.
  shard=$(ls -t \
    experiments/runs/"$EXPERIMENT"/w"${DEEPEST}"-f"${fold}"__*/tree_stats.npz \
    2>/dev/null | head -1)
  if [ -z "$shard" ]; then
    echo "no depth-${DEEPEST} shard for fold ${fold} -- rerun Stage 2" >&2
    return 1
  fi
  "$PYTHON" -m experiments run \
    --config "$CONFIG" \
    --set data.store=dash-molecules-10fold-"$fold" \
    --set predictor.params.max_wl_depth="$DEEPEST" \
    --set tree_stats_load_path="$shard" \
    --set normalization=equal_weighted \
    --set run.experiment="$NORM_EXPERIMENT" \
    --set run.batch_id=f"$fold"
}
export -f normalize_fold
export NORM_EXPERIMENT

seq 1 "$N_FOLDS" | xargs -P "$PARALLEL_JOBS" -n 1 bash -c 'normalize_fold "$1"' --

# --- Stage 4: the full-corpus model, normalized --------------------------
#
# Where dash_charges.sh needs two stages -- merge the 10 per-fold shards
# (`merge-shards`), then predict with the merged tree -- sieve needs one
# and no merge machinery at all: it has no merge_node_stats analogue, and
# does not want one, because a single fit on dash-molecules' own train
# split is direct and exact. That is this stage. It predicts the same
# store's own test split, `equal_weighted` normalized, idempotent on the
# run directory.
#
# This is the one stage of this workflow whose footprint has not been
# measured: it fits ~10x a single fold's atom count in one process, and
# featurization (build_codes/from_rdkit) is ~96% of a sieve fit, so
# `n_jobs` is set here -- the only stage where it is, since this stage is
# a single process rather than a filled dispatch queue. Lower
# SIEVE_FULL_JOBS if the box is tight; raise it if it isn't.
# LOO does not enter here either (Stage 2's note applies).
FULL_EXPERIMENT=sieve-full-equal-weighted
FULL_JOBS="${SIEVE_FULL_JOBS:-8}"

if compgen -G "experiments/runs/$FULL_EXPERIMENT/full__*/metrics.json" \
  > /dev/null; then
  echo "skip full-corpus run (already done)"
else
  "$PYTHON" -m experiments run \
    --config "$CONFIG" \
    --set data.store=dash-molecules \
    --set predictor.params.max_wl_depth="$DEEPEST" \
    --set predictor.params.n_jobs="$FULL_JOBS" \
    --set normalization=equal_weighted \
    --set run.experiment="$FULL_EXPERIMENT" \
    --set run.batch_id=full
fi

# --- Stage 5: full-corpus depth sweep -------------------------------------
#
# Stage 2's curve is a statement about a fold-sized training set (~82k
# conformers), where it flattens past 4 WL rounds. Stage 4 then showed
# the same depth-6 model reaching a far lower error once fit on the
# whole corpus (~824k), which says the fold plateau is a data limit
# rather than a capacity one -- and therefore that the depth question
# has to be re-asked at full scale. This stage asks it.
#
# Plain `run` per depth, as in Stage 2 and for the same reason: under
# continuation, level k's estimate depends on the model's own deepest
# level, so an independent fit per depth is required, not merely
# equivalent (see Stage 2's own note).
#
# Depths run to 10 rather than Stage 2's 6: the ceiling here is unknown,
# which is the point of the stage.
#
# Idempotent per depth via a full-w<depth> batch_id, and dispatched via
# xargs -P (default 4, override with SIEVE_FULL_SWEEP_JOBS). Each
# process featurizes the whole corpus, which measured ~37GB RSS at
# depth 6 with n_jobs=8 -- deeper is more, so the default concurrency is
# deliberately far below the fold sweep's.
FULL_SWEEP_EXPERIMENT=sieve-full-depth-sweep
FULL_SWEEP_DEPTHS="0 1 2 3 4 5 6 7 8 9 10"
FULL_SWEEP_JOBS="${SIEVE_FULL_SWEEP_JOBS:-4}"

run_full_depth() {
  local depth=$1
  local batch="full-w${depth}"
  if compgen -G "experiments/runs/$FULL_SWEEP_EXPERIMENT/${batch}__*/metrics.json" \
    > /dev/null; then
    echo "skip $batch (already done)"
    return 0
  fi
  "$PYTHON" -m experiments run \
    --config "$CONFIG" \
    --set data.store=dash-molecules \
    --set predictor.params.max_wl_depth="$depth" \
    --set predictor.params.n_jobs="$FULL_JOBS" \
    --set run.experiment="$FULL_SWEEP_EXPERIMENT" \
    --set run.batch_id="$batch"
}
export -f run_full_depth
export FULL_SWEEP_EXPERIMENT FULL_JOBS

# shellcheck disable=SC2086
echo $FULL_SWEEP_DEPTHS | tr ' ' '\n' \
  | xargs -P "$FULL_SWEEP_JOBS" -n 1 bash -c 'run_full_depth "$1"' --

# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment sieve-full-depth-sweep \
#     --x predictor.params.max_wl_depth --metric mae --metric r2
