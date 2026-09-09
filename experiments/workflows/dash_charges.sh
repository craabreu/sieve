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
# One `dash` predictor run per (depth, fold) pair -- untracked by MLflow
# (the default; no --track), all under one run.experiment so `sweep` can
# read them back as a single curve:
#   "$PYTHON" -m experiments sweep --experiment dash-depth-sweep \
#     --x predictor.params.max_depth --metric mae --metric r2
#
# ~7 minutes per run on a full, unlimited fold (measured directly, depth
# 8), single-threaded, ~8GB RSS -- ~90 runs total. Dispatched
# PARALLEL_JOBS at a time via xargs -P (default 8; override with
# DASH_DEPTH_SWEEP_JOBS=N), well inside a 64-core/500GB-RAM box's
# headroom at that width.
#
# `experiments run` itself always creates a fresh, timestamped/uuid'd
# directory, so it is never idempotent on its own; run_one makes the pair
# idempotent instead: run.batch_id is depth+fold-specific
# ("d<depth>-f<fold>"), and a directory already matching that batch_id
# with a metrics.json inside it (a completed run, not a crashed or
# still-running one) is skipped rather than relaunched -- safe to
# interrupt (Ctrl-C, a killed process, a crashed run) and resume by just
# running this script again, including under a different PARALLEL_JOBS.
EXPERIMENT=dash-depth-sweep
DEPTHS="1 2 4 6 8 10 12 14 16"
PARALLEL_JOBS="${DASH_DEPTH_SWEEP_JOBS:-8}"

run_one() {
  local depth=$1 fold=$2
  local batch_id="d${depth}-f${fold}"
  if compgen -G "experiments/runs/$EXPERIMENT/${batch_id}__*/metrics.json" \
    > /dev/null; then
    echo "skip $batch_id (already done)"
    return 0
  fi
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store=dash-molecules-10fold-"$fold" \
    --set predictor.params.max_depth="$depth" \
    --set run.experiment="$EXPERIMENT" \
    --set run.batch_id="$batch_id"
}
export -f run_one
export PYTHON EXPERIMENT

for depth in $DEPTHS; do
  for fold in $(seq 1 10); do
    printf '%s %s\n' "$depth" "$fold"
  done
done | xargs -P "$PARALLEL_JOBS" -n 2 bash -c 'run_one "$1" "$2"' --
