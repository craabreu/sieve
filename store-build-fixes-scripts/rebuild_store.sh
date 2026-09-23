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
# --recurate redoes everything after the parse, for a store this script
# already built: it resets NEW_STORE to its kept uncurated parse and runs the
# same steps from curation on. A change to curation, collapse or floors then
# costs minutes, not the 40-minute re-parse of the SDF. Only files the steps
# themselves write are removed; anything else there stops it.
#
# Usage, from the repository root:
#   store-build-fixes-scripts/rebuild_store.sh [--recurate] [NEW_STORE]
set -euo pipefail

RECURATE=false
if [ "${1:-}" = "--recurate" ]; then
  RECURATE=true
  shift
fi
NEW="${1:-dash-molecules-rebuild}"
PYTHON=.venv/bin/python
N_SHARDS=50  # cv_charges.sh's N_SHARDS; see the reasoning recorded there
SDF=experiments/stores/dash-molecules/dashMoleculesSDF_v2.sdf
DIR="experiments/stores/$NEW"
DERIVED="molecules.parquet curation_summary.txt split_summary.txt floor-components.json built-at-commit.txt"

if $RECURATE; then
  if [ ! -f "$DIR/molecules.parquet.uncurated" ]; then
    echo "$DIR has no molecules.parquet.uncurated to re-curate from" >&2
    exit 1
  fi
  for f in "$DIR"/*; do
    name=$(basename "$f")
    case " $DERIVED molecules.parquet.uncurated " in
      *" $name "*) ;;
      *) echo "$DIR holds $name, which this script did not write; refusing" >&2; exit 1 ;;
    esac
  done
elif [ -e "$DIR" ]; then
  echo "$DIR exists; refusing to build over it (--recurate reuses its parse)" >&2
  exit 1
fi
if [ -n "$(git status --porcelain -- experiments src)" ]; then
  echo "experiments/ or src/ has uncommitted changes; rebuild from a clean tree" >&2
  exit 1
fi
echo "rebuilding into $DIR at $(git rev-parse --short HEAD)"

run() { echo "+ $*"; "$@"; }
if $RECURATE; then
  for name in $DERIVED; do rm -f "$DIR/$name"; done
  run cp "$DIR/molecules.parquet.uncurated" "$DIR/molecules.parquet"
fi
run "$PYTHON" -m experiments prepare-store "$NEW" --sdf-path "$SDF" \
  --stop-before-split --keep-uncurated
run "$PYTHON" -m experiments prepare-store "$NEW" --sdf-path "$SDF" \
  --n-shards "$N_SHARDS"
run "$PYTHON" -m experiments annotate-collapse "$NEW"
run "$PYTHON" -m experiments build-floor-cache "$NEW" --n-shards "$N_SHARDS"
git rev-parse HEAD > "$DIR/built-at-commit.txt"
echo "done: $DIR"
