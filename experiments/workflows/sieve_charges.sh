#!/usr/bin/env bash
# Reproduces the Sieve-charges experiment series end to end under the CV
# redesign -- the structural parallel of dash_charges.sh: Stage 1 is
# byte-identical (both series read the same store, so their per-sample
# numbers are comparable point for point), then a Sieve-specific shard
# fit (one shard set per depth -- see Stage 2's own note) and the same
# two-study CV shape. See docs/superpowers/specs/2026-09-12-cv-shard-
# redesign-design.md for the full design.
#
# Superseded by this script: the old partition-store/per-(depth,fold)-run
# sweep and the per-fold equal_weighted normalization stage.
#
# Every step below is a plain call to the `experiments` CLI (see
# experiments/README.md) -- this script only fixes the sequence and the
# arguments, so each stage can also be re-run by hand exactly as shown.
# Run from the repo root: experiments/workflows/sieve_charges.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
  echo "no $PYTHON -- run 'uv sync --extra dev --extra chem --extra charges' first" >&2
  exit 1
fi

STORE=dash-molecules
N_SHARDS="${DASH_N_SHARDS:-25}"
K="${SIEVE_CV_K:-5}"
CODES_PATH=experiments/stores/dash-molecules/sieve-codes.json

# --- Stage 1: data preparation --------------------------------------------
#
# Byte-identical to dash_charges.sh's own Stage 1, deliberately: both
# series read the *same* partition, so their per-sample numbers are
# comparable point by point rather than merely on average. Idempotent --
# a box that already ran the DASH workflow finds this a no-op.
"$PYTHON" -m experiments prepare-store "$STORE" --n-shards "$N_SHARDS"

# --- Stage 2: freeze the Sieve attribute vocabulary ------------------------
#
# Blocking, not optional: SievePredictor.fit's own build_codes assigns
# each attribute value a dense rank over whatever that call's own
# training molecules happened to contain -- a ~1/N-of-train shard's own
# vocabulary can therefore disagree with another shard's on what integer
# code means what value. attribute_codes feeds SieveConfig.schema_version,
# which check_mergeable compares exactly, so two shards with a shifted
# element table simply refuse to merge. Freezing the vocabulary once, over
# the *whole* train split, before any shard is fit, is what makes the
# shards mergeable at all -- see SievePredictor's own codes_path docstring.
# Idempotent (skips if CODES_PATH already exists).
#
# The one node attribute + no edge attributes this series settled on
# (element only, continuation + empirical-Bayes shrinkage -- see the
# CONFIG below, which every later stage shares).
if [ ! -f "$CODES_PATH" ]; then
  "$PYTHON" -m experiments build-sieve-codes "$STORE" \
    --attributes element --edge-attributes "" \
    --out "$CODES_PATH"
fi

CONFIG_LABEL=element-eb
PREDICTOR_PARAMS='{"attributes": ["element"], "edge_attributes": [], "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}'

# --- Stage 3: shard fits ---------------------------------------------------
#
# One shard set **per depth**, unlike DASH: under class_estimator=
# continuation, level k's estimate depends on whether k is the model's
# own deepest level (backoff reads differently for a depth-3 fit than for
# a depth-6 one at the same level), so a shallow config is not a
# truncation of a deep one -- the two aren't even mergeable, since
# max_wl_depth feeds schema_version too. `cv-fit-sieve-shards` loops
# every (depth, shard) pair internally; idempotent per pair.
STUDY_A_DEPTHS=0,1,2,3,4,5,6,7,8,9,10

"$PYTHON" -m experiments cv-fit-sieve-shards "$STORE" \
  --n-shards "$N_SHARDS" --depths "$STUDY_A_DEPTHS" \
  --codes-path "$CODES_PATH" --config-label "$CONFIG_LABEL" \
  --predictor-params "$PREDICTOR_PARAMS"

# --- Stage 4: Study A -- depth selection -----------------------------------
#
# One repeat (seed 0), K folds. Each fold's training model is assembled
# per depth (sieve.merge.fold/merge_models, exact); evaluation reuses one
# featurized batch across every depth (SievePredictor.build_predict_batch/
# predict_raw_from_batch), valid because every depth's shards share
# CODES_PATH's one frozen vocabulary. Both raw and equal_weighted-
# normalized metrics land in one metrics.json (norm/* prefix).
STUDY_A_EXPERIMENT=sieve-cv-study-a
METHOD="sieve-$CONFIG_LABEL"

"$PYTHON" -m experiments cv-run-sieve "$STORE" \
  --n-shards "$N_SHARDS" --k "$K" \
  --depths "$STUDY_A_DEPTHS" --repeats 0 \
  --codes-path "$CODES_PATH" --config-label "$CONFIG_LABEL" \
  --predictor-params "$PREDICTOR_PARAMS" \
  --normalization equal_weighted --method "$METHOD" --experiment "$STUDY_A_EXPERIMENT"

# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment sieve-cv-study-a \
#     --x config.cv.depth --metric norm/mae --metric mae

# --- Stage 5: Study B -- model comparison -----------------------------------
#
# Pick SIEVE_SELECTED_DEPTH from Stage 4's own curve before running this
# (defaults to 6, this series' original starting depth -- override once
# Stage 4 has run). Four further repeats (seeds 1-4; Study A's own
# repeat/K-partition is reused as the first of the five), 5x5 = 25
# samples at the one selected depth.
SELECTED_DEPTH="${SIEVE_SELECTED_DEPTH:-6}"
STUDY_B_EXPERIMENT=sieve-cv-study-b

"$PYTHON" -m experiments cv-run-sieve "$STORE" \
  --n-shards "$N_SHARDS" --k "$K" \
  --depths "$SELECTED_DEPTH" --repeats 0,1,2,3,4 \
  --codes-path "$CODES_PATH" --config-label "$CONFIG_LABEL" \
  --predictor-params "$PREDICTOR_PARAMS" \
  --normalization equal_weighted --method "$METHOD" --experiment "$STUDY_B_EXPERIMENT"

# --- Stage 6: final held-out evaluation ------------------------------------
#
# Once, at the end: merge all N of the selected depth's own shards into a
# single full-train model, predict the untouched 10% test split. Sieve's
# own merge (sieve.merge.fold, in merge-states) is already the balanced-
# tree form design.md 5.4 calls for, so this is one command regardless of
# N. Idempotent (skips if the merged shard or the run directory already
# exists).
MERGED_SHARD="experiments/results/sieve-merged/tree_stats-w${SELECTED_DEPTH}.npz"
if [ ! -f "$MERGED_SHARD" ]; then
  mapfile -t SHARD_PATHS < <(
    ls experiments/runs/cv-shard-fits/fit-sieve-"$CONFIG_LABEL"-w"$SELECTED_DEPTH"-s*__*/tree_stats.npz \
      | sort
  )
  "$PYTHON" -m experiments merge-states --predictor sieve --out "$MERGED_SHARD" \
    "${SHARD_PATHS[@]}"
fi

FINAL_EXPERIMENT=sieve-final-holdout
if compgen -G "experiments/runs/$FINAL_EXPERIMENT/final__*/metrics.json" \
  > /dev/null; then
  echo "skip final holdout run (already done)"
else
  "$PYTHON" -m experiments run \
    --config experiments/configs/sieve-charge-sweep.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_wl_depth="$SELECTED_DEPTH" \
    --set tree_stats_load_path="$MERGED_SHARD" \
    --set normalization=equal_weighted \
    --set run.experiment="$FINAL_EXPERIMENT" \
    --set run.batch_id=final
fi
