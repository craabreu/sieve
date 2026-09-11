#!/usr/bin/env bash
# Reproduces the DASH-charges experiment series end to end: download +
# parse + split the real published SDF, partition it into 10 disjoint
# folds, run a predictor against them, and summarize the results.
#
# Every step below is a plain call to the `experiments` CLI (see
# experiments/README.md) -- this script only fixes the sequence and the
# arguments, so each stage can also be re-run by hand exactly as shown.
# Run from the repo root: experiments/workflows/dash_charges.sh
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

# --- Stage 1: data preparation -------------------------------------------
#
# `prepare-store` is idempotent at each of its three stages (download,
# parse, split) -- safe to re-run; it skips whatever it already did. It
# downloads the real ~8.3GB dashMoleculesSDF_v2.sdf (ETH Research
# Collection) and writes experiments/stores/dash-molecules
# (~1M conformers, ~9.6GB parquet).
"$PYTHON" -m experiments prepare-store

# `partition-store` divides dash-molecules' entire molecule set into 10
# disjoint folds -- every conformer of every molecule kept, none used
# twice -- named dash-molecules-10fold-1 .. dash-molecules-10fold-10.
# Idempotent as a whole: skips entirely once every fold already exists,
# and deterministic at the default seed (0) if it does run -- safe to
# re-run, including after this script was interrupted partway through.
"$PYTHON" -m experiments partition-store dash-molecules-10fold --n-stores 10

# --- Stage 2: DASH depth sweep --------------------------------------------
#
# One fit + one tree-matching walk per fold (at the deepest depth
# requested), with every shallower depth's own metrics derived from the
# already-walked paths instead of re-walking from scratch -- ~10 fits
# total instead of ~90 independent ones. See
# experiments/experiments/dash_depth_sweep.py's own module docstring for
# why this is exact, not an approximation (and for the one genuine
# exception: depth 1 is always run for real, never derived, because
# DASH-tree's own match_new_atom redirects a hydrogen atom to its heavy
# neighbor and pre-consumes one depth unit -- confirmed correct against
# the real store and DASH-tree clone, bit-for-bit against an independent
# run, before this default depths list was trusted with it).
#
# Dispatched one process per fold via xargs -P (default: all 10 at once,
# override with DASH_DEPTH_SWEEP_JOBS=N) -- each fold's own fit+walk is
# now a much coarser unit of work than the old per-(depth,fold) runs, so
# fold-level parallelism (not depth-level) is what actually uses this
# box's real headroom (64 cores/500GB, each fold single-threaded).
# `--fold` runs exactly one fold in that process, per-fold idempotent
# (skip once every depth's own run directory already has a metrics.json)
# -- safe to interrupt and resume by running this script again, including
# under a different DASH_DEPTH_SWEEP_JOBS. Untracked by MLflow (the
# default; no --track).
#
# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment dash-depth-sweep \
#     --x predictor.params.max_depth --metric mae --metric r2
EXPERIMENT=dash-depth-sweep
DEPTHS=1,2,4,6,8,10,12,14,16
N_FOLDS=10
PARALLEL_JOBS="${DASH_DEPTH_SWEEP_JOBS:-$N_FOLDS}"

run_fold() {
  local fold=$1
  "$PYTHON" -m experiments dash-depth-sweep \
    --config experiments/configs/dash-charge-example.yaml \
    --store-prefix dash-molecules-10fold \
    --fold "$fold" \
    --depths "$DEPTHS" \
    --experiment "$EXPERIMENT"
}
export -f run_fold
export PYTHON DEPTHS EXPERIMENT

seq 1 "$N_FOLDS" | xargs -P "$PARALLEL_JOBS" -n 1 bash -c 'run_fold "$1"' --

# --- Stage 3: normalize the deepest fit, per fold -----------------------
#
# DASH's own published post-hoc charge conservation (normalize.py's
# std_weighted, the paper's eq. 4) applied to each fold's depth-16
# prediction. `tree_stats_load_path` loads that fold's own saved shard
# (Stage 2), so this is predict-only -- no re-fit -- and `normalization`
# re-scores against the sum constraint. One process per fold, idempotent
# (skip a fold once its own run directory exists).
NORM_EXPERIMENT=dash-depth16-std-weighted

normalize_fold() {
  local fold=$1
  if compgen -G "experiments/runs/$NORM_EXPERIMENT/f${fold}__*/metrics.json" \
    > /dev/null; then
    echo "skip normalize f${fold} (already done)"
    return 0
  fi
  local shard
  shard=$(ls -t experiments/runs/"$EXPERIMENT"/d16-f"${fold}"__*/tree_stats.npz \
    | head -1)
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store=dash-molecules-10fold-"$fold" \
    --set predictor.params.max_depth=16 \
    --set tree_stats_load_path="$shard" \
    --set normalization=std_weighted \
    --set run.experiment="$NORM_EXPERIMENT" \
    --set run.batch_id=f"$fold"
}
export -f normalize_fold
export NORM_EXPERIMENT

seq 1 "$N_FOLDS" | xargs -P "$PARALLEL_JOBS" -n 1 bash -c 'normalize_fold "$1"' --

# --- Stage 4: merge the 10 shards -------------------------------------------
#
# fold_node_stats over the 10 depth-16 shards -- exact, no re-fit. The
# folds partition the corpus by molecule and each shard is train-only, so
# the merged result is one fit on the whole training set. Idempotent
# (skips if the output already exists).
MERGED_SHARD=experiments/results/dash-merged/tree_stats.npz
"$PYTHON" -m experiments merge-shards \
  --from-experiment "$EXPERIMENT" \
  --depth 16 \
  --n-folds "$N_FOLDS" \
  --out "$MERGED_SHARD"

# --- Stage 5: the merged full-corpus model, normalized --------------------
#
# The merged shard predicting the *original* store's own test split
# (~103k conformers) -- genuinely held out, since every fold's shard was
# fit on train molecules only and the folds partition by molecule.
# Predict-only (tree_stats_load_path), std_weighted normalized.
# Idempotent (skip if the run directory exists).
MERGED_EXPERIMENT=dash-merged-std-weighted
if compgen -G "experiments/runs/$MERGED_EXPERIMENT/merged__*/metrics.json" \
  > /dev/null; then
  echo "skip merged run (already done)"
else
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store=dash-molecules \
    --set predictor.params.max_depth=16 \
    --set tree_stats_load_path="$MERGED_SHARD" \
    --set normalization=std_weighted \
    --set run.experiment="$MERGED_EXPERIMENT" \
    --set run.batch_id=merged
fi

# --- Stage 6: full-corpus depth sweep -------------------------------------
#
# The fold sweep (Stage 2) answers "how does depth pay at fold scale?"
# -- ~82k train conformers. This one asks it at full scale, ~824k, on
# dash-molecules' own train/test split. Worth asking separately because
# the fold curve's flat tail is a statement about a fold-sized training
# set, not about the corpus: a model still data-limited at 82k can keep
# paying for depth at 824k.
#
# Same `dash-depth-sweep` command as Stage 2, pointed at one store
# instead of a partition (`--store`), so this is again one fit + one
# tree-matching walk at the deepest depth requested, with every
# shallower depth derived from the already-walked paths -- 1 walk over
# the full corpus rather than 6. Runs are labelled d<depth>-full
# (`--label`), and the stage is idempotent as a whole: it skips once
# every depth already has a metrics.json.
#
# Depths run past Stage 2's 16, to 20. Nothing in the predictor caps
# max_depth -- it is handed straight to DASH-tree's own match_new_atom --
# so 18 and 20 are cheap to ask for, being derived from the same walk as
# every other depth. Expect them to reproduce 16 exactly once every
# path has bottomed out in the published tree: the fold curve is already
# identical to five decimals at 14 and 16 (0.019743, 0.019742). A pair
# of duplicate points is the evidence that the tree, not the sweep, is
# what ends the curve.
#
# One process, single-threaded. This is the heaviest stage in the file
# -- one DASH fit plus one walk at depth 20 over ~43M atoms -- and its
# footprint is unmeasured at this scale (a single fold peaked around
# 8GB).
FULL_EXPERIMENT=dash-full-depth-sweep
FULL_DEPTHS=1,2,4,6,8,10,12,14,16,18,20

"$PYTHON" -m experiments dash-depth-sweep \
  --config experiments/configs/dash-charge-example.yaml \
  --store dash-molecules \
  --label full \
  --depths "$FULL_DEPTHS" \
  --experiment "$FULL_EXPERIMENT"

# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment dash-full-depth-sweep \
#     --x predictor.params.max_depth --metric mae --metric r2
