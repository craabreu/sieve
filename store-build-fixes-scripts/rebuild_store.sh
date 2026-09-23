#!/usr/bin/env bash
# Rebuild the DASH store from the SDF into a FRESH store directory, running
# the same commands as cv_charges.sh's store steps (prepare-corpus,
# split-store, annotate-collapse, floor-cache) with the same interpreter and
# the same N_SHARDS. The current store is read (its SDF) and never written.
#
# Then compare it against the staged patch:
#
#   python store-build-fixes-scripts/compare_rebuild.py dash-molecules-rebuild
#
# and, only if that passes (or its differences are understood), swap it in:
#
#   mv experiments/stores/dash-molecules experiments/stores/dash-molecules.pre-rebuild
#   mv experiments/stores/dash-molecules-rebuild experiments/stores/dash-molecules
#   mv experiments/stores/dash-molecules.pre-rebuild/dashMoleculesSDF_v2.sdf \
#      experiments/stores/dash-molecules/
#
# The old directory keeps molecules.parquet.uncurated and every earlier
# backup; nothing is deleted.
#
# Usage, from the repository root:  store-build-fixes-scripts/rebuild_store.sh [NEW_STORE]
set -euo pipefail

NEW="${1:-dash-molecules-rebuild}"
PYTHON=.venv/bin/python
N_SHARDS=50  # cv_charges.sh's N_SHARDS; see the reasoning recorded there
SDF=experiments/stores/dash-molecules/dashMoleculesSDF_v2.sdf

if [ -e "experiments/stores/$NEW" ]; then
  echo "experiments/stores/$NEW exists; refusing to build over it" >&2
  exit 1
fi
if [ -n "$(git status --porcelain -- experiments src)" ]; then
  echo "experiments/ or src/ has uncommitted changes; rebuild from a clean tree" >&2
  exit 1
fi
echo "rebuilding into experiments/stores/$NEW at $(git rev-parse --short HEAD)"

run() { echo "+ $*"; "$@"; }
run "$PYTHON" -m experiments prepare-store "$NEW" --sdf-path "$SDF" \
  --stop-before-split --keep-uncurated
run "$PYTHON" -m experiments prepare-store "$NEW" --sdf-path "$SDF" \
  --n-shards "$N_SHARDS"
run "$PYTHON" -m experiments annotate-collapse "$NEW"
run "$PYTHON" -m experiments build-floor-cache "$NEW" --n-shards "$N_SHARDS"
git rev-parse HEAD > "experiments/stores/$NEW/built-at-commit.txt"
echo "done: experiments/stores/$NEW"
