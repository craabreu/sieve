#!/usr/bin/env bash
# Reproduces the DASH-charges experiment series end to end under the CV
# redesign: download + parse + curate + split the real published SDF into
# 90% train / 10% test with N cluster-clean train shards, fit each shard
# once, then run two CV studies (depth selection, model comparison) by
# assembling every sample's training model from shard merges -- no
# per-sample refit. See docs/superpowers/specs/2026-09-12-cv-shard-
# redesign-design.md for the full design and why shard merging replaces
# the old 10-fold sweep this script used to run.
#
# Superseded by this script: the old partition-store/dash-depth-sweep/
# merge-shards sequence (still available as CLI commands, kept for other
# uses, but no longer this workflow's own path).
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

STORE=dash-molecules
N_SHARDS="${DASH_N_SHARDS:-25}"
# The published tree's own depth ceiling -- deeper cannot lengthen a path
# (see the old script's own Stage 6 note, carried over unchanged).
MAX_DEPTH=16
K="${DASH_CV_K:-5}"

# --- Stage 1: data preparation --------------------------------------------
#
# `prepare-store` is idempotent at each of its stages (download, parse,
# curate, split) -- safe to re-run; it skips whatever it already did. It
# downloads the real ~8.3GB dashMoleculesSDF_v2.sdf (ETH Research
# Collection), curates conformers (the DASH paper's own 0.4 e criterion,
# reproduced at parse time -- see prepare_dash.curate_conformers), and
# writes a 90/10 train/test split plus a `cluster`/`shard` column on
# experiments/stores/dash-molecules.
#
# Before running this for the first time, consider
#   "$PYTHON" -m experiments cluster-report
# against a store that is parsed+curated but not yet split, to pick
# N_SHARDS from the real cluster-size distribution rather than blind --
# see prepare_dash.cluster_size_report's own docstring. The default below
# (25) is what the CV redesign discussion settled on as a starting point.
"$PYTHON" -m experiments prepare-store "$STORE" --n-shards "$N_SHARDS"

# --- Stage 2: shard fits ---------------------------------------------------
#
# One DASH fit per shard, at MAX_DEPTH -- node stats are depth-invariant
# (compute_node_stats accumulates over every node on every atom's path,
# not just the deepest), so one fit per shard serves every shallower
# depth Study A below asks for, exactly as the old per-fold sweep already
# exploited. Predicts nothing: a single ~1/N-of-train shard is only ever
# used merged. Idempotent per shard (skips once its own tree_stats.npz
# exists). One process, not one per shard: `cv-fit-dash-shards` already
# loops every shard s00..s{N-1} internally (run_dash_shard_fits), and
# DASHTree(preload=True) -- the expensive part of constructing a
# predictor -- is amortized across every shard fit that one process does
# in sequence, rather than paid again per shard under N separate
# processes.
"$PYTHON" -m experiments cv-fit-dash-shards "$STORE" \
  --n-shards "$N_SHARDS" --max-depth "$MAX_DEPTH"

# --- Stage 3: Study A -- depth selection -----------------------------------
#
# One repeat (seed 0), K folds -- no ANOVA, no Tukey, just a depth curve.
# Each fold's training model is the merge of the other K-1 groups'
# shards (tree_artifact.merge_node_stats, exact); one tree-matching walk
# per fold, shared across every depth via predict_raw_at_depth. Both raw
# and std_weighted-normalized metrics land in one metrics.json per run
# (norm/* prefix) -- read the normalized curve to pick a depth, since
# that is the form DASH is actually deployed in.
STUDY_A_EXPERIMENT=dash-cv-study-a
STUDY_A_DEPTHS=1,2,4,6,8,10,12,14,16

"$PYTHON" -m experiments cv-run-dash "$STORE" \
  --n-shards "$N_SHARDS" --k "$K" --max-depth "$MAX_DEPTH" \
  --depths "$STUDY_A_DEPTHS" --repeats 0 \
  --normalization std_weighted --method dash --experiment "$STUDY_A_EXPERIMENT"

# Read the resulting curve with:
#   "$PYTHON" -m experiments sweep --experiment dash-cv-study-a \
#     --x config.cv.depth --metric norm/mae --metric mae

# --- Stage 4: Study B -- model comparison -----------------------------------
#
# Pick DASH_SELECTED_DEPTH from Stage 3's own curve before running this
# (defaults to MAX_DEPTH, the published tree's own ceiling, which is
# rarely the actual optimum -- override it once Stage 3 has run). Four
# further repeats (seeds 1-4; Study A's own repeat 0/K-partition is
# reused as the first of the five, per the CV design), 5x5 = 25 samples
# at the one selected depth, feeding compare.py's repeated-measures
# ANOVA + Tukey HSD (Ash/Wognum/Rodriguez-Perez JCIM 2025 protocol).
SELECTED_DEPTH="${DASH_SELECTED_DEPTH:-$MAX_DEPTH}"
STUDY_B_EXPERIMENT=dash-cv-study-b

"$PYTHON" -m experiments cv-run-dash "$STORE" \
  --n-shards "$N_SHARDS" --k "$K" --max-depth "$MAX_DEPTH" \
  --depths "$SELECTED_DEPTH" --repeats 0,1,2,3,4 \
  --normalization std_weighted --method dash --experiment "$STUDY_B_EXPERIMENT"

# --- Stage 5: final held-out evaluation ------------------------------------
#
# Once, at the end, not part of either CV study: merge all N shards
# (built once, in Stage 2) into a single full-train model and predict the
# untouched 10% test split -- the headline number. Idempotent (skips if
# the merged shard or the run directory already exists).
MERGED_SHARD=experiments/results/dash-merged/tree_stats.npz
if [ ! -f "$MERGED_SHARD" ]; then
  # Merge order does not affect the result (merge_node_stats is
  # commutative/associative); sorted here only for a readable command.
  mapfile -t SHARD_PATHS < <(
    ls experiments/runs/cv-shard-fits/fit-dash-s*__*/tree_stats.npz | sort
  )
  "$PYTHON" -m experiments merge-states --predictor dash --out "$MERGED_SHARD" \
    "${SHARD_PATHS[@]}"
fi

FINAL_EXPERIMENT=dash-final-holdout
if compgen -G "experiments/runs/$FINAL_EXPERIMENT/final__*/metrics.json" \
  > /dev/null; then
  echo "skip final holdout run (already done)"
else
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_depth="$SELECTED_DEPTH" \
    --set tree_stats_load_path="$MERGED_SHARD" \
    --set normalization=std_weighted \
    --set run.experiment="$FINAL_EXPERIMENT" \
    --set run.batch_id=final
fi
