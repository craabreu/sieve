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
