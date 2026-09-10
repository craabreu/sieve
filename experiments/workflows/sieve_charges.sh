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

CONFIG=experiments/configs/sieve-charge-example.yaml

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
# sweep there is no sweep module here, and deliberately so: level k's
# partition, class means/counts and (depth-local) shrinkage are
# byte-identical between a depth-k fit and a depth-N fit for N > k --
# verified directly on a real corpus -- so N independent runs *are* the
# capped-at-k curve, exactly, with nothing to amortize between them. See
# docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md, which
# is where that equivalence was established and where the per-level
# in-run design it replaced was dropped. A sieve fit on a real fold is
# also a fraction of a DASH tree-matching walk on the same fold, which is
# the other half of why dash_depth_sweep.py exists and this doesn't.
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
# `report_loo=true` throughout. The train/train_loo gap is the
# memorization signal (design.md 10.3), it is the one quantity the curve
# cannot be re-derived for afterwards without re-running everything, and
# its cost -- a second featurization of train, ~+38% -- is cheap at these
# run times. Off by default in the predictor, on here on purpose.
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
#     --split test --split train --split train_loo
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
    --set predictor.params.report_loo=true \
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
# `report_loo` is left at the config default (off) here: LOO is a
# statement about the *fit*, which Stage 2 already recorded at every
# depth, and turning it on would buy a second full train featurization
# for a run that is otherwise predict-only.
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
# `report_loo` stays off: a LOO pass over the full training corpus is
# both the expensive case and not what this stage is for.
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
