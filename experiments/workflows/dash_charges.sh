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

# --- Stage 1: data preparation -------------------------------------------
#
# `prepare-store` is idempotent at each of its three stages (download,
# parse, split) -- safe to re-run; it skips whatever it already did. It
# downloads the real ~8.3GB dashMoleculesSDF_v2.sdf (ETH Research
# Collection) and writes experiments/stores/dash-molecules
# (~1M conformers, ~9.6GB parquet).
python -m experiments prepare-store

# `partition-store` divides dash-molecules' entire molecule set into 10
# disjoint folds -- every conformer of every molecule kept, none used
# twice -- named dash-molecules-10fold-1 .. dash-molecules-10fold-10.
# Not internally idempotent (always rewrites its destination stores), but
# deterministic: the default seed (0) reproduces the same 10 folds byte-
# for-byte on every run.
python -m experiments partition-store dash-molecules-10fold --n-stores 10
