#!/usr/bin/env bash
# The charges series' model-evaluation procedure, end to end, as a
# sequence of individually-guarded steps: prepare the corpus, choose a
# shard count from measured evidence, split, fit each shard once, run the
# two CV studies, compare, and score the untouched held-out test split.
#
# Supersedes dash_charges.sh and sieve_charges.sh, which described the
# same procedure twice (once per predictor) and shared a deliberately
# byte-identical data-preparation stage. Under the CV redesign the two
# series genuinely share one store, one shard partition and one fold
# assignment per repeat -- that sharing is what makes their samples
# pairable in compare.py's repeated-measures design -- so describing the
# procedure once is not just tidier, it is what keeps the pairing true.
#
# Run from the repo root: experiments/workflows/cv_charges.sh
# Every step is idempotent, so re-running resumes rather than redoes.
# `CV_UNTIL=<step>` stops after that step (see the step harness below).
#
# Nothing here is tracked by MLflow, deliberately and at every layer:
# cv.py contains no MLflow code at all, and `run` leaves tracking off
# unless given --track, which no step passes. Corpus preparation, shard
# fits and CV samples are all read back off disk by summarize/sweep/
# compare, so a tracking server would add a second, duplicating copy of
# every artifact and nothing else -- which has caused real disk incidents
# on this box before (see runner.execute's own docstring).
#
# Design: docs/superpowers/specs/2026-09-12-cv-shard-redesign-design.md
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

# ===========================================================================
# The step harness
# ===========================================================================
#
#   step <name> <guard> -- <command...>
#
# `guard` is a shell expression whose *success means "already done"*. It is
# evaluated twice: before the step (to decide whether to skip) and again
# after it (to confirm the step actually produced what it claimed). That
# second evaluation is the part that earns its keep -- a step which
# silently no-ops fails the workflow here, loudly, instead of leaving a
# store or a run directory that misrepresents itself downstream.
#
# Guards deliberately check the *real artifact* -- a parquet's columns, a
# run's metrics.json, a shard's tree_stats.npz -- never a side marker
# recording that something once ran. The two claims differ: a marker says
# "this step executed at some point", an artifact check says "its output
# is here now". Only the second survives the artifact being rebuilt,
# deleted, or half-written underneath it.
#
# Adding a step as the procedure evolves is therefore three lines and one
# guard, with no new bespoke idempotency logic to get subtly wrong.
#
# `CV_UNTIL=<step>` stops cleanly after that step. Not merely a
# convenience: the procedure has a genuine human-in-the-loop break at
# Study A, whose depth curve someone has to *read* before Study B can be
# told which depth to fix. Running to a named stop point is how that break
# is expressed, rather than by commenting steps out and forgetting to put
# them back.

CV_UNTIL="${CV_UNTIL:-}"

STEPS_RUN=()
STEPS_SKIPPED=()
_until_matched=false

step() {
  local name=$1 guard=$2
  shift 2
  [ "${1:-}" = "--" ] && shift

  local last=false
  if [ -n "$CV_UNTIL" ] && [ "$name" = "$CV_UNTIL" ]; then
    last=true
    _until_matched=true
  fi

  if eval "$guard" > /dev/null 2>&1; then
    echo "== skip  $name"
    STEPS_SKIPPED+=("$name")
  else
    echo "== run   $name"
    "$@"
    if ! eval "$guard" > /dev/null 2>&1; then
      echo "!! $name completed but its guard still reports it undone --" >&2
      echo "!! the step did not produce what it promised. Guard: $guard" >&2
      exit 1
    fi
    echo "== done  $name"
    STEPS_RUN+=("$name")
  fi

  if $last; then
    echo "== stop  (CV_UNTIL=$CV_UNTIL reached)"
    exit 0
  fi
}

# Every step name declared in this file, read from the file itself. Whether
# CV_UNTIL names a real step is a *static* fact, so it must not be inferred
# from how far a particular run got: an interrupted or failed run reaches
# fewer steps for reasons that have nothing to do with the name.
declared_steps() { grep -oE '^step [A-Za-z0-9_-]+' "$0" | awk '{print $2}'; }

summarize_steps() {
  local status=$?
  echo
  echo "=== workflow summary ==="
  echo "ran:     ${#STEPS_RUN[@]} step(s)${STEPS_RUN[*]+: ${STEPS_RUN[*]}}"
  echo "skipped: ${#STEPS_SKIPPED[@]} step(s)${STEPS_SKIPPED[*]+: ${STEPS_SKIPPED[*]}}"
  if [ -n "$CV_UNTIL" ] && ! declared_steps | grep -qx "$CV_UNTIL"; then
    echo "!! CV_UNTIL=$CV_UNTIL matches no step in $0; nothing stopped early" >&2
    echo "!! steps are: $(declared_steps | tr '\n' ' ')" >&2
  elif [ -n "$CV_UNTIL" ] && ! $_until_matched; then
    # The name is real, so this run simply never got there.
    echo "!! stopped before reaching CV_UNTIL=$CV_UNTIL" >&2
    echo "!! (exit $status -- a step failed, or the workflow was interrupted)" >&2
  elif [ "$status" -ne 0 ]; then
    echo "!! workflow exited $status -- see the traceback above" >&2
  fi
}
trap summarize_steps EXIT

# ===========================================================================
# Guard vocabulary
# ===========================================================================

file_exists() { [ -f "$1" ]; }

store_has_columns() {
  local store=$1
  shift
  "$PYTHON" - "$store" "$@" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

store, *columns = sys.argv[1:]
path = Path("experiments/stores") / store / "molecules.parquet"
if not path.exists():
    sys.exit(1)
present = set(pq.ParquetFile(path).schema.names)
sys.exit(0 if set(columns) <= present else 1)
PY
}

# The store is curated *and* the summary describes this very parquet --
# not merely "curation ran here once". Comparing the recorded
# post-curation conformer count against the parquet's actual row count is
# what tells a curated store apart from one re-parsed underneath a
# surviving summary (prepare_dash.curate_conformers makes the same check
# for the same reason).
store_is_curated() {
  local store=$1
  "$PYTHON" - "$store" <<'PY'
import sys
from pathlib import Path

from experiments.prepare_dash import (
    CURATION_SUMMARY,
    _curated_conformer_count,
    _parquet_row_count,
)

store_dir = Path("experiments/stores") / sys.argv[1]
summary = store_dir / CURATION_SUMMARY
if not summary.exists():
    sys.exit(1)
recorded = _curated_conformer_count(summary.read_text())
actual = _parquet_row_count(store_dir / "molecules.parquet")
sys.exit(0 if recorded is not None and recorded == actual else 1)
PY
}

runs_exist() {
  local experiment=$1 batch=$2
  compgen -G "experiments/runs/$experiment/${batch}__*/metrics.json" > /dev/null
}

runs_count_is() {
  local experiment=$1 expected=$2 found
  found=$(ls -d experiments/runs/"$experiment"/*__*/metrics.json 2>/dev/null | wc -l)
  [ "$found" -eq "$expected" ]
}

# Guards on the *arm*, not merely on how many runs an experiment holds.
# runs_count_is cannot tell one method's curve from another's: the Sieve
# Study A experiment held exactly 55 runs both when they were all
# continuation-eb and when they were all pooled, so a bare count happily
# skips a study that never ran the arm the depth is being selected on --
# the same blind spot file_is_newer_than_runs was written to close for
# compare, one artifact over.
#
# The method is read from each run's *structured* manifest field, never
# parsed back out of the batch_id string; the design records repeat, fold,
# method and depth structurally for exactly this reason.
method_runs_count_is() {
  local experiment=$1 method=$2 expected=$3
  "$PYTHON" - "$experiment" "$method" "$expected" <<'PY'
import json
import sys
from pathlib import Path

experiment, method, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
found = 0
for manifest in Path("experiments/runs", experiment).glob("*__*/manifest.json"):
    if not (manifest.parent / "metrics.json").exists():
        continue  # a half-written run is not a finished sample
    cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
    found += cv.get("method") == method
sys.exit(0 if found == expected else 1)
PY
}

# Study B's claim is "K x repeats x variants samples AT THE SELECTED DEPTH",
# which runs_count_is cannot express: it counts a whole experiment. Once a
# selected depth changes, the runs from the previous one are still on disk --
# deliberately, they are evidence -- and a bare count then reads 350 where it
# expects 175 and wedges the workflow, exactly as the Sieve Study A guard did
# when a second arm appeared beside the first.
depth_runs_count_is() {
  local experiment=$1 depth=$2 expected=$3
  "$PYTHON" - "$experiment" "$depth" "$expected" <<'PY'
import json
import sys
from pathlib import Path

experiment, depth, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
found = 0
for manifest in Path("experiments/runs", experiment).glob("*__*/manifest.json"):
    if not (manifest.parent / "metrics.json").exists():
        continue  # a half-written run is not a finished sample
    cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
    found += int(cv.get("depth", -1)) == depth
sys.exit(0 if found == expected else 1)
PY
}

# file_exists is the wrong guard for a *derived* artifact: it cannot tell
# that the runs the artifact was derived from have changed. compare's plots
# survived a study growing from 2 arms to 7, and the step skipped. This
# compares mtimes instead: the artifact is up to date only if nothing it
# reads is newer than it.
file_is_newer_than_runs() {
  local artifact=$1
  shift
  [ -f "$artifact" ] || return 1
  local newer
  newer=$(find "$@" -name metrics.json -newer "$artifact" -print -quit 2>/dev/null)
  [ -z "$newer" ]
}

shard_fits_count_is() {
  local prefix=$1 expected=$2 found
  found=$(ls -d experiments/runs/cv-shard-fits/"$prefix"*__*/tree_stats.npz 2>/dev/null | wc -l)
  [ "$found" -eq "$expected" ]
}

n_items() { echo "$1" | tr ',' '\n' | grep -c .; }

# The variant JSON is authored as a multi-line literal for readability; count
# its entries by counting "method" keys rather than by parsing JSON in shell.
n_variants() { echo "$SIEVE_VARIANTS" | grep -c '"method"'; }

# ===========================================================================
# Configuration
# ===========================================================================

STORE=dash-molecules

# Chosen from cluster-report's own measured output, not guessed: the train
# split holds 313,964 molecules in 17,995 Butina clusters whose largest is
# 3,733 (1.19% of train). That largest cluster is the binding constraint --
# past N ~= 84 it no longer fits inside one shard's target and pins a shard,
# unbalancing the rest (visible at N=100: max=3733 against a 3,140 target,
# std 59.6, against std 0.4 at N=50). N=50 is the largest round value that
# still balances to a single molecule, divides by K, and keeps ~40% headroom
# on that cluster.
N_SHARDS="${CV_N_SHARDS:-50}"
K="${CV_K:-5}"

STUDY_A_REPEATS=0
STUDY_B_REPEATS=0,1,2,3,4

DASH_MAX_DEPTH=16  # the published tree's own ceiling; deeper cannot lengthen a path
DASH_DEPTHS=2,4,6,8,10,12,14,16

SIEVE_CONFIG_LABEL=element-eb
SIEVE_DEPTHS=0,1,2,3,4,5,6,7,8,9,10
SIEVE_PREDICTOR_PARAMS='{"attributes": ["element"], "edge_attributes": [], "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}'
# Deliberately NOT derived from SIEVE_CONFIG_LABEL. The label names the
# *fit* -- it is baked into the 50 shard-fit directory names
# (fit-sieve-element-eb-w10-sNN), so changing it invalidates them and forces
# a refit. The method names the *arm*, and arms are named for what they are:
# the fit's own reading is continuation + empirical Bayes.
SIEVE_METHOD="sieve-element-continuation-eb"

# Readings of the SAME shard fits. Each entry is a method name plus anything
# SieveModel.with_params accepts -- class_estimator, shrinkage_weight,
# shrinkage_strength, minimum_support -- which is exactly the set
# schema_version excludes, so this costs no refit, no re-merge and not even a
# re-featurization: the variants share the merged model and the eval batch.
#
# The cutoff arm is the hard-threshold counterpart to shrinkage: a class is
# used only once it has minimum_support observations, otherwise the search
# backs off. The intent has always been "at least four molecules": under the
# old per-conformer fitting that meant 12, three conformers apiece, chosen so
# the threshold was not an arbitrary count of correlated conformers. Under
# --collapse a row IS a distinct structure, so the same intent is 4 -- the
# number changed because the unit did, not because the threshold moved. It is
# given shrinkage_weight null so that it differs from
# sieve-element-continuation in the cutoff alone.
#
# These PATCH the fitted config (SIEVE_PREDICTOR_PARAMS above), so the
# non-shrinking variants must say "shrinkage_weight": null explicitly --
# omitting it would inherit the fit's empirical_bayes and quietly make
# "sieve-element-pooled" mean pooled+eb.
# Three class estimators crossed with shrinkage on/off. "recursive" is
# class_estimator="continuation_recursive": the same one-level-deep
# aggregation applied to the children's own continuation estimates rather
# than to their stored means (src/sieve/continuation.py::class_means). It
# differs from flat continuation only at levels at least two steps above the
# deepest, so at SIEVE_SELECTED_DEPTH it is a real arm, not a duplicate.
SIEVE_VARIANTS='[
  {"method": "sieve-element-pooled",          "class_estimator": "pooled",                 "shrinkage_weight": null},
  {"method": "sieve-element-pooled-eb",       "class_estimator": "pooled",                 "shrinkage_weight": "empirical_bayes"},
  {"method": "sieve-element-continuation",    "class_estimator": "continuation",           "shrinkage_weight": null},
  {"method": "sieve-element-continuation-eb", "class_estimator": "continuation",           "shrinkage_weight": "empirical_bayes"},
  {"method": "sieve-element-recursive",       "class_estimator": "continuation_recursive", "shrinkage_weight": null},
  {"method": "sieve-element-recursive-eb",    "class_estimator": "continuation_recursive", "shrinkage_weight": "empirical_bayes"},
  {"method": "sieve-element-continuation-cutoff", "class_estimator": "continuation",        "shrinkage_weight": null, "minimum_support": 4}
]'
# Every variant is scored at SIEVE_SELECTED_DEPTH; the variants are a
# comparison of estimators at a fixed depth, not seven more depth studies.
#
# Depth selection (Study A) runs exactly ONE of them, named here. It used to
# run the fit's own reading, sieve-element-continuation-eb, passed as plain
# --method; it now selects on the pooled arm, and that necessarily moves
# Study A onto --variants as well. The reason is the patch semantics above:
# the fitted config applies empirical_bayes, so "pooled" as a bare --method
# would still be scored with the fit's shrinkage, and only a variant can say
# "shrinkage_weight": null to turn it back off. Scoring pooled is a different
# *reading* of the same 50 shard fits, so switching arms costs no refit.
#
# Derived from SIEVE_VARIANTS by name rather than written out a second time,
# so the arm Study A selects a depth for is, by construction, the same
# specification Study B then scores at that depth. Two copies could drift,
# and the drift would be silent in the worst way: a depth chosen for one
# estimator and applied to a slightly different one.
# Comma-separated, and the FIRST is the arm the depth-curve figure shows and
# the one SIEVE_SELECTED_DEPTH is read off. The second is here because the
# manuscript calls continuation the estimator and pooled the naive reading it
# corrects, while the depth was in fact selected on pooled -- so the two
# curves have to be seen together before that selection can be defended or
# moved. Scoring both costs no refit: class_estimator is excluded from
# schema_version, so continuation is a different reading of the very same
# shard fits, and adding it to this invocation re-uses each fold's featurized
# batch rather than building a second one.
SIEVE_STUDY_A_METHODS="${SIEVE_STUDY_A_METHODS:-sieve-element-pooled,sieve-element-continuation}"
SIEVE_STUDY_A_METHOD="${SIEVE_STUDY_A_METHODS%%,*}"
SIEVE_STUDY_A_VARIANTS=$(
  echo "$SIEVE_VARIANTS" | "$PYTHON" -c '
import json, sys

wanted = [m for m in sys.argv[1].split(",") if m]
variants = {v["method"]: v for v in json.load(sys.stdin)}
missing = [m for m in wanted if m not in variants]
if missing:
    raise SystemExit(
        f"SIEVE_STUDY_A_METHODS names no entry in SIEVE_VARIANTS: {missing}"
    )
print(json.dumps([variants[m] for m in wanted]))
' "$SIEVE_STUDY_A_METHODS"
)

CODES_PATH="experiments/stores/$STORE/sieve-codes.json"
CLUSTER_REPORT="experiments/results/cluster-report.txt"

# Study B fixes each method at the depth Study A selected. Override once
# Study A's curve has been read; the defaults are each series' own prior
# best, not a result.
# Per-atom predictions are written for Study B only, and only at each
# method's one selected depth -- that is the set a per-element or
# worst-atom error analysis actually reads. Study A deliberately does not:
# across a depth sweep everything in predictions.npz except
# atom_target_pred is identical between depths of a given (repeat, fold),
# so it would write ~14GB to say the same thing 8-11 times over. Study B's
# own cost is ~7.5GB (50 runs x ~151MB); set CV_SAVE_PREDICTIONS=0 to skip
# it and rely on the final-holdout runs, which write theirs regardless.
SAVE_PREDICTIONS_FLAG=""
if [ "${CV_SAVE_PREDICTIONS:-1}" != "0" ]; then
  SAVE_PREDICTIONS_FLAG="--save-predictions"
fi

# Persist the assembled CV training models and reuse them across studies.
# Assembling one repeat's K models costs ~123s and 30GB of peak RSS at N=50
# (load 2.6s, fold into K groups 27.3s, leave-one-group-out 92.9s, measured on
# the real shards), and Study A and Study B share repeat 0's partition, so
# without this that repeat is assembled twice. Set CV_MODEL_CACHE= (empty) to
# turn it off and trade the disk back for the time.
CV_MODEL_CACHE="${CV_MODEL_CACHE-experiments/results/cv-model-cache}"
MODEL_CACHE_FLAG=""
if [ -n "$CV_MODEL_CACHE" ]; then
  MODEL_CACHE_FLAG="--model-cache $CV_MODEL_CACHE"
fi

# DASH's Study A also scores each fold's TRAINING shards with --score-train,
# so its depth curve can show the train-vs-validation gap -- where DASH
# starts fitting its own training molecules rather than the chemistry. It is
# DASH's only route to that curve (docs/superpowers/specs/2026-09-17-
# analytic-training-metrics-design.md section 2: its node stats describe the
# atoms passing *through* a node, not the ones a shallower depth would answer
# at their own terminal node, so no analytic shortcut exists for it) and it
# is expensive: measured ~1.1 h -> ~5.3 h, since the per-fold tree walk
# dominates. Set CV_SCORE_TRAIN=0 to skip it.
#
# Sieve and HOSE need no such flag any more: run_sieve_cv/run_hose_cv record
# train/rmse, train/r2, train/eta2, train/matched_fraction and the support
# distribution on every run already, computed analytically from the fitted
# model's own stored statistics -- no extra fit, no extra walk, no molecules
# loaded. That is what made Sieve's own former --score-train pass (~15 min ->
# ~1.2 h) worth dropping outright, here and in Study C's stage-1 curve below.
#
# Study B does not carry even the DASH flag: it compares methods at one fixed
# depth, a question the training error does not enter, and it would pay
# DASH's cost 5 times over (five repeats) for a number no Tukey interval reads.
SCORE_TRAIN_FLAG=""
if [ "${CV_SCORE_TRAIN:-1}" != "0" ]; then
  SCORE_TRAIN_FLAG="--score-train"
fi

# Fit one row per collapse_key rather than one per conformer: a molecule's
# conformers, an exact duplicate and an enantiomer are indistinguishable to
# every arm here, so counting them separately reweights class means for no
# informational reason (docs/superpowers/specs/2026-09-17-fit-time-collapse-
# design.md). The held-out side is never collapsed, so the metric still
# measures per-conformer error and can still detect the premise failing.
#
# Applies to the TRAINING side of every shard fit, which is why it must reach
# all four fit_one_* functions below -- and be exported, since xargs runs them
# in a nested shell that inherits nothing unexported.
#
# --weight-by-collapse is deliberately NOT set here. It replaces a group by k
# copies of its mean, which fabricates a class variance that never existed;
# it exists only for the one-off identity check that the grouping and
# representative selection are right (see experiments/README.md), never for
# the science.
COLLAPSE_FLAG=""
if [ "${CV_COLLAPSE:-1}" != "0" ]; then
  COLLAPSE_FLAG="--collapse"
fi

DASH_SELECTED_DEPTH="${DASH_SELECTED_DEPTH:-16}"
# 5, not the 6 that minimizes the curve. Study A's own 95% intervals put
# depths 4 through 10 in a tie with 6 (RMSE 0.02045 +/- 0.00061 e at the
# minimum), so the minimum is not distinguishable from its neighbours and
# picking it would be reading noise. 5 is the shallowest depth inside that
# interval that still sits on the plateau rather than on the descent, and a
# shallower tree is cheaper to fit, to merge and to walk.
#
# DASH stays at 16: 10 through 16 are likewise tied, but 16 is the published
# tree's own ceiling and the depth its authors optimized, so the comparison
# is made against the incumbent as its authors deployed it.
SIEVE_SELECTED_DEPTH="${SIEVE_SELECTED_DEPTH:-5}"

# --- the HOSE arm ----------------------------------------------------------
#
# The published lookup tradition (Bremser's register, NMRShiftDB), scored on
# this corpus as a third method beside DASH and Sieve. See
# docs/superpowers/specs/2026-09-16-hose-baseline-design.md.
#
# ONE FIT PER RADIUS, unlike the other two arms. DASH truncates walked paths
# and Sieve truncates merged levels, so each fits once at its deepest setting
# and reads every shallower one out of that. A HOSE code is a linearization
# whose sphere ordering consults what lies beyond it, so generating deeper
# re-renders shallower spheres -- 8.5% of k-sphere prefixes differ by which
# radius they were cut from (spec section 7). The radius-k point must be
# generated at k, so the sweep is one shard set per radius. That is what keeps
# this arm's x axis meaning the same thing as the other two arms' do.
#
# Radii 1-6, not 1-8: code generation cost grows steeply with radius (measured
# on this box, 88 us/atom at r=1 against 1223 at r=8 -- radius 8 alone costs
# more than 1 through 5 together), and the spec's own deployed setting is 5.
HOSE_RADII="${HOSE_RADII:-1,2,3,4,5,6}"
HOSE_STUDY_A=hose-cv-study-a
HOSE_STUDY_B=hose-cv-study-b
HOSE_SHARD_JOBS="${HOSE_SHARD_JOBS:-16}"
# One process per (radius, fold) for Study A and per (repeat, fold) for Study
# B. Fold-level, not radius- or repeat-level: this arm's cost is code
# generation over each fold's held-out set, and nothing shares it across folds
# the way the other arms share one assembled model across depths. At
# radius-level parallelism alone a single process would carry a whole radius's
# 5 folds, which measured out at ~18 h for Study A.
HOSE_CV_JOBS="${HOSE_CV_JOBS:-12}"
# Pinned rather than read off Study A's curve, so Studies A and B can run at
# the same time instead of one waiting on the other. 5 is the spec's deployed
# setting and the radius its one-shard probe used; Study A's curve, drawn from
# the same runs, is what checks it afterwards.
HOSE_SELECTED_RADIUS="${HOSE_SELECTED_RADIUS:-5}"
HOSE_SELECT_SCRIPT=experiments/workflows/hose_select_radius.py

# Resolved from Study A's own curve when it exists, by the same rule that
# chose Sieve's depth 5: the shallowest radius whose mean RMSE lies inside the
# minimum\'s own Nadeau-Bengio interval, so a plateau is not read as a peak.
# Falls back to 5 -- the spec's deployed setting and the radius its one-shard
# probe used -- when Study A has not run yet. Override to pin it.
hose_radius() {
  if [ -n "${HOSE_SELECTED_RADIUS:-}" ]; then
    echo "$HOSE_SELECTED_RADIUS"
    return
  fi
  "$PYTHON" "$HOSE_SELECT_SCRIPT" "$HOSE_STUDY_A" 2>/dev/null || echo 5
}

DASH_STUDY_A=dash-cv-study-a
DASH_STUDY_B=dash-cv-study-b
SIEVE_STUDY_A=sieve-cv-study-a
SIEVE_STUDY_B=sieve-cv-study-b
# Tracked, deliberately: experiments/results is gitignored (friction
# observation 3), and the Tukey plot is the study's headline result, not an
# intermediate. It carries its own provenance header (store, commit, run ids).
FIGURES_DIR=experiments/docs/figures
# The same test drawn the other way: one interval per method on the metric's
# own scale rather than one per pair on a difference scale (statsmodels'
# plot_simultaneous layout, as used in Pat Walters' ADME model comparison).
# Two methods make a thin plot; it earns its keep once the open-ended list of
# Sieve configs in the design has more than one entry, because it grows as k
# rather than as k-choose-2.

# One pair of plots per metric. Every CV run already records all of these from
# a single prediction, so extra metrics cost a re-analysis, never a re-run.
# sum_constraint/* score the per-molecule total the normalizers enforce, which
# is a different question from per-atom accuracy and worth its own panel.
COMPARE_METRICS="${COMPARE_METRICS:-rmse,mae,r2,sum_constraint/rmse}"
# Metrics where larger is better, so --reference best picks the right arm.
COMPARE_HIGHER_IS_BETTER="r2,sum_constraint/r2"

metric_slug()   { echo "$1" | tr '/' '-'; }
tukey_plot()    { echo "$FIGURES_DIR/tukey-study-b-$(metric_slug "$1").png"; }
simult_plot()   { echo "$FIGURES_DIR/simultaneous-study-b-$(metric_slug "$1").png"; }
each_metric()   { echo "$COMPARE_METRICS" | tr ',' '\n' | grep -v '^$'; }

# ===========================================================================
# Steps
# ===========================================================================

# --- corpus ---------------------------------------------------------------
#
# Download (~8.3GB, skipped when a correctly-sized copy is present), parse
# the SDF, and curate conformers by the DASH paper's own 0.4 e sibling
# criterion -- stopping *before* the split, so the next step can choose
# N_SHARDS from real evidence rather than a guess.
step prepare-corpus \
  "store_is_curated $STORE" -- \
  "$PYTHON" -m experiments prepare-store "$STORE" --stop-before-split

# --- choose the shard count ------------------------------------------------
#
# Advisory and read-only: reports the train-split cluster size distribution
# and the shard balance each candidate N would actually achieve. It informs
# N_SHARDS above (which is where the decision is recorded durably -- this
# file, in git, not the report, which lands in gitignored results/).
step cluster-report \
  "file_exists $CLUSTER_REPORT" -- \
  bash -c "set -euo pipefail; mkdir -p \"\$(dirname '$CLUSTER_REPORT')\" && \
           '$PYTHON' -m experiments cluster-report '$STORE' \
             --candidates 10,20,25,50,100,200 | tee '$CLUSTER_REPORT'"

# --- split -----------------------------------------------------------------
#
# 90/10 train/test by whole Butina cluster, plus the cluster ids and the
# N_SHARDS cluster-clean train shards, all from one clustering pass.
step split-store \
  "store_has_columns $STORE split cluster shard" -- \
  "$PYTHON" -m experiments prepare-store "$STORE" --n-shards "$N_SHARDS"

# --- collapse annotation ----------------------------------------------------
#
# Adds collapse_key plus its three counts, in place. Must run AFTER the split:
# it refuses a key group that straddles a split, cluster or shard, which is a
# check on the partition as much as on the key. Measured on the real corpus:
# 0 groups straddle any of the three, because identical molecules share a
# fingerprint so Butina cannot separate them, and both the split and the
# sharding are by whole cluster.
#
# Note that a dash_id is NOT a structure key -- 2.14% of them hold conformers
# that are diastereomers or E/Z isomers of one another, which collapse_key
# correctly keeps apart. The unit here is the structure, not the dash_id.
step annotate-collapse \
  "store_has_columns $STORE collapse_key n_collapsed n_molecules n_enantiomer_forms" -- \
  "$PYTHON" -m experiments annotate-collapse "$STORE"

# --- irreducible-floor components ------------------------------------------
#
# A floor depends only on which molecules are held out -- never on the model,
# the depth, the variant or which arm is scoring -- so computing it per run
# repeats identical work across every depth, every variant, all three arms
# and all three studies (~5 h over the campaign). The components are additive
# over shards, which is valid because no collapse group spans one: the
# annotation refuses a straddling collapse_key group, and the coarser
# stereo-blind grouping was measured at 0 straddling on the real corpus.
#
# So: 50 shard computations, once, and every fold's floors are a sum.
step floor-cache \
  "file_exists experiments/stores/$STORE/floor-components.json" -- \
  "$PYTHON" -m experiments build-floor-cache "$STORE" --n-shards "$N_SHARDS"

# --- freeze the Sieve vocabulary -------------------------------------------
#
# Blocking for Sieve, not optional: a shard that discovers its own
# attribute_codes from its own ~1/N of train can disagree with its siblings
# on what an integer code means, and attribute_codes feeds schema_version,
# which check_mergeable compares exactly. Freezing one vocabulary over the
# whole train split is what makes the shards mergeable at all.
step sieve-codes \
  "file_exists $CODES_PATH" -- \
  "$PYTHON" -m experiments build-sieve-codes "$STORE" \
    --attributes element --edge-attributes "" --out "$CODES_PATH"

# --- shard fits ------------------------------------------------------------
#
# One fit per shard, predicting nothing (a single shard is only ever used
# merged) -- and for both predictors, one set at the deepest depth serves
# every shallower one. DASH's node stats are depth-invariant and its paths
# prefix-nested; Sieve's levels are bottom-up, so a deep fit's levels 0..d
# are exactly a depth-d fit's, recovered by truncating the *merged* model
# (cv.truncate_model). The one exception is DASH depth 1, which truncation
# cannot reproduce (the H-atom redirect consumes a depth unit before
# max_depth is checked) -- hence DASH_DEPTHS starts at 2, and run_dash_cv
# refuses anything shallower rather than scoring it wrongly.
# Shards are independent and each fit is ~1 minute of real work, so the
# unit of parallelism is one shard per process, dispatched with xargs -P --
# the same shape the pre-CV workflows used for folds, and for the same
# reason: it is what actually uses this box's headroom (64 cores / 503GB).
# Running these sequentially cost 53.8 min for DASH's 50 shards where a
# filled dispatch queue is ~1-2 min.
#
# design.md 5.5's warning about process overhead does not bite here: it
# concerns worker pools spun up per `fit()` call, where startup (260-390ms)
# rivals the fit itself. A shard fit is ~65s -- three orders of magnitude
# above that -- so the per-process DASHTree preload is noise by comparison.
# Nor does friction observation 7 (repeated identical work across
# processes): every process here fits a *different* shard.
#
# Each process is left single-threaded (`n_jobs` unset) because the
# dispatch already fills the box. That is the old workflows' own rule:
# dispatch N single-threaded processes, or run one process with `n_jobs`,
# never both.
#
# The job counts are set from measured peak RSS, not from `nproc` and not
# from a guess -- a guess of 16 already cost one OOM-killed shard here:
#
#   dash  shard fit, max_depth=16 : 35.1 GB peak RSS, 2m07 wall
#   sieve shard fit, max_wl_depth=10: 32.0 GB peak RSS, 1m11 wall
#
# 16 x 32GB = 512GB against this box's 503GB, which is exactly why s10 was
# killed. A ~400GB budget (leaving the OS and page cache room) allows ~11
# concurrent fits of either kind; 8 keeps real headroom for shard-to-shard
# variance while still being 8x a sequential run.
#
# The non-obvious part, worth stating because it defeats the intuition that
# sharding shrinks memory: a Sieve fit's footprint is driven by **depth**,
# not by shard size. 32GB here is for 1/50th of train at depth 10, against
# the old workflow's ~37GB for the *whole* corpus at depth 6 -- the class
# count is what explodes, and WL depth is what explodes it. Fitting more,
# smaller shards therefore does not buy proportionally more concurrency.
DASH_SHARD_JOBS="${DASH_SHARD_JOBS:-8}"
SIEVE_SHARD_JOBS="${SIEVE_SHARD_JOBS:-8}"
SIEVE_MAX_DEPTH="${SIEVE_MAX_DEPTH:-10}"  # the deepest SIEVE_DEPTHS asks for

fit_one_dash_shard() {
  "$PYTHON" -m experiments cv-fit-dash-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$DASH_MAX_DEPTH" --shard "$1" \
    $COLLAPSE_FLAG
}

fit_one_sieve_shard() {
  "$PYTHON" -m experiments cv-fit-sieve-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$SIEVE_MAX_DEPTH" --shard "$1" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS" \
    $COLLAPSE_FLAG
}
export -f fit_one_dash_shard fit_one_sieve_shard
export PYTHON STORE N_SHARDS DASH_MAX_DEPTH SIEVE_MAX_DEPTH
export CODES_PATH SIEVE_CONFIG_LABEL SIEVE_PREDICTOR_PARAMS COLLAPSE_FLAG

# Each shard's own fit is idempotent, and xargs hands a given shard to
# exactly one process, so an interrupted dispatch resumes cleanly -- and
# under a different job count, as the old workflows also guaranteed.
all_shard_ids() { seq -f "s%02g" 0 $((N_SHARDS - 1)); }

dispatch_dash_shards() {
  all_shard_ids | xargs -P "$DASH_SHARD_JOBS" -n 1 \
    bash -c 'fit_one_dash_shard "$1"' --
}

dispatch_sieve_shards() {
  all_shard_ids | xargs -P "$SIEVE_SHARD_JOBS" -n 1 \
    bash -c 'fit_one_sieve_shard "$1"' --
}

step dash-shard-fits \
  "shard_fits_count_is fit-dash-s $N_SHARDS" -- \
  dispatch_dash_shards

step sieve-shard-fits \
  "shard_fits_count_is fit-sieve-$SIEVE_CONFIG_LABEL-w$SIEVE_MAX_DEPTH-s $N_SHARDS" -- \
  dispatch_sieve_shards

# --- HOSE shard fits, one set per radius ------------------------------------
#
# The loop over radii is the whole difference from the two dispatches above:
# they fit once at their deepest setting and truncate, this one cannot (spec
# section 7), so every radius the studies score needs its own 50 fits.
#
# Cheap per fit relative to the other arms -- no tree to load and no WL
# refinement, just code generation and a dict -- so the dispatch is 8 wide
# like the others and bounded by the generator, not by memory.
each_hose_radius() {
  { echo "$HOSE_RADII" | tr ',' '\n'; hose_radius; } | grep -v '^$' | sort -n -u
}

fit_one_hose_shard() {
  "$PYTHON" -m experiments cv-fit-hose-shards "$STORE" \
    --n-shards "$N_SHARDS" --radius "$HOSE_RADIUS" --shard "$1" \
    $COLLAPSE_FLAG
}
export -f fit_one_hose_shard

hose_shard_fits_done() {
  local r
  for r in $(each_hose_radius); do
    shard_fits_count_is "fit-hose-r$r-s" "$N_SHARDS" || return 1
  done
}

dispatch_hose_shards() {
  local r
  for r in $(each_hose_radius); do
    echo "--- hose shard fits: radius $r ---"
    export HOSE_RADIUS="$r"
    all_shard_ids | xargs -P "$HOSE_SHARD_JOBS" -n 1 \
      bash -c 'fit_one_hose_shard "$1"' --
  done
}

step hose-shard-fits "hose_shard_fits_done" -- dispatch_hose_shards

# --- Study A: depth selection ----------------------------------------------
#
# One repeat, K folds, no ANOVA and no Tukey -- just a depth curve per
# method, each fold's training model assembled by merging the other K-1
# groups' shards. Read the normalized curve (the form each method is
# actually deployed in) to pick the depths Study B then fixes:
#   "$PYTHON" -m experiments sweep --experiment dash-cv-study-a \
#     --x cv.depth --metric norm/mae --metric mae
step study-a-dash \
  "runs_count_is $DASH_STUDY_A $((K * $(n_items "$DASH_DEPTHS")))" -- \
  "$PYTHON" -m experiments cv-run-dash "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" --max-depth "$DASH_MAX_DEPTH" \
    --depths "$DASH_DEPTHS" --repeats "$STUDY_A_REPEATS" \
    $MODEL_CACHE_FLAG $SCORE_TRAIN_FLAG $COLLAPSE_FLAG \
    --normalization std_weighted --method dash --experiment "$DASH_STUDY_A"

# Guarded arm by arm, never on the experiment's run count: that count is now
# the sum over however many arms are listed, and was already briefly satisfied
# by the wrong set of 55 runs once (see method_runs_count_is).
each_method_runs_count_is() {
  local experiment=$1 methods=$2 expected=$3 method
  for method in ${methods//,/ }; do
    method_runs_count_is "$experiment" "$method" "$expected" || return 1
  done
}

step study-a-sieve \
  "each_method_runs_count_is $SIEVE_STUDY_A $SIEVE_STUDY_A_METHODS \
     $((K * $(n_items "$SIEVE_DEPTHS")))" -- \
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$SIEVE_DEPTHS" --repeats "$STUDY_A_REPEATS" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --fit-depth "$SIEVE_MAX_DEPTH" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS" \
    --variants "$SIEVE_STUDY_A_VARIANTS" \
    $MODEL_CACHE_FLAG $COLLAPSE_FLAG \
    --normalization equal_weighted --method "$SIEVE_METHOD" \
    --experiment "$SIEVE_STUDY_A"

# --- Study A: the HOSE arm --------------------------------------------------
#
# One repeat, K folds, every radius -- the same shape as the other two arms'
# Study A, and deliberately without --score-train: the train curve is a
# diagnostic the other arms already carry, and here it would double the
# generator cost, which is this arm's entire budget.
run_one_hose_cv() {
  # "<radius> <fold>" on one line, from xargs
  local radius=${1%% *} fold=${1##* }
  "$PYTHON" -m experiments cv-run-hose "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --radii "$radius" --folds "$fold" --repeats "$HOSE_CV_REPEATS" \
    --normalization equal_weighted --method hose $COLLAPSE_FLAG \
    --experiment "$HOSE_CV_EXPERIMENT" $HOSE_CV_EXTRA
}
export -f run_one_hose_cv
export HOSE_STUDY_A HOSE_STUDY_B STUDY_A_REPEATS STUDY_B_REPEATS K

dispatch_study_a_hose() {
  local r f
  export HOSE_CV_REPEATS="$STUDY_A_REPEATS"
  export HOSE_CV_EXPERIMENT="$HOSE_STUDY_A"
  export HOSE_CV_EXTRA=""
  for r in $(echo "$HOSE_RADII" | tr ',' ' '); do
    for f in $(seq 0 $((K - 1))); do echo "$r $f"; done
  done | xargs -P "$HOSE_CV_JOBS" -I{} bash -c 'run_one_hose_cv "$1"' -- {}
}

step study-a-hose \
  "method_runs_count_is $HOSE_STUDY_A hose $((K * $(n_items "$HOSE_RADII")))" -- \
  dispatch_study_a_hose

# --- Study A's figure -------------------------------------------------------
#
# The depth curve as a manuscript figure rather than as sweep's diagnostic
# PNG: one panel per method, each on its own depth axis (Sieve counts WL
# iterations, DASH counts path length -- different quantities, so they do not
# share an x axis), sharing the metric axis.
#
# The error bars are Nadeau-Bengio corrected (Mach. Learn. 52:239, 2003).
# Study A is ONE repeat, so its k folds are not independent replicates: any
# two share k-2 of their k-1 training groups, and the naive s/sqrt(k) is
# optimistic by sqrt(1 + k*n_test/n_train) -- exactly 1.5x at this study's
# geometry (k=5, 10 of 50 shards held out). Drawing the naive interval would
# make neighbouring depths look more separated than the data supports, which
# is the one thing a depth-selection figure must not do.
#
# Drawn on the RAW metrics only. Same reason the compare step gives: the two
# methods use different normalizers, so anything normalized would report the
# normalizer as much as the estimator.
#
# One row of panels per metric, in this order, sharing a y axis across the
# row; columns share the depth axis down the column. Each panel also carries
# a dashed line at the best value the OTHER method reached for that metric --
# the two count depth in different units so they cannot share an x axis, and
# that line is what still lets one be read against the other.
DEPTH_CURVE_METRIC="${DEPTH_CURVE_METRIC:-rmse,r2}"
DEPTH_CURVE_STEM="$FIGURES_DIR/depth-curve-study-a"
DEPTH_CURVE_STAMP="$FIGURES_DIR/.depth-curve-inputs"
STUDY_A_RUNS="experiments/runs/$DASH_STUDY_A experiments/runs/$SIEVE_STUDY_A experiments/runs/$HOSE_STUDY_A"

# Sieve's WL depth 0 is kept off the figure: with no refinement at all the
# model is element-wise pooled means, whose R^2 of 0.46 compresses the 0.99
# band where every difference between the real depths lives. It is scored
# and recorded like any other depth, and the caption names it as dropped.
# DASH needs no floor, its own sweep starting at 2.
#
# Built from the same variables the studies ran under, so the figure cannot
# name an arm or an experiment nobody produced.
DEPTH_CURVE_ARMS=$(
  printf '[{"experiment": "%s", "method": "dash", "label": "DASH",' \
    "$DASH_STUDY_A"
  printf ' "x_label": "Maximum Path Depth"},'
  # Every Sieve arm shares one panel, so the estimators are read against
  # each other in place rather than across the figure; DASH keeps its own,
  # since its depth axis counts something else entirely.
  # DASH's entry above already ends in a comma, so the separator goes
  # before every Sieve arm after the first.
  sep=""
  for method in ${SIEVE_STUDY_A_METHODS//,/ }; do
    printf '%s {"experiment": "%s", "method": "%s", "label": "Sieve, %s",' \
      "$sep" "$SIEVE_STUDY_A" "$method" "${method#sieve-element-}"
    printf ' "panel": "Sieve", "x_label": "Maximum Refinement Depth",'
    printf ' "min_depth": 1}'
    sep=","
  done
  # HOSE gets its own panel for the same reason DASH does: a sphere count is
  # not a WL depth, so they cannot share an x axis.
  printf ', {"experiment": "%s", "method": "hose", "label": "HOSE lookup",' \
    "$HOSE_STUDY_A"
  printf ' "panel": "HOSE", "x_label": "Number of Spheres"}'
  printf ']' 
)

# Same two-part guard as compare, for the same reason: mtimes catch new runs,
# but neither a changed metric nor a changed set of arms moves any run's
# mtime, and both change the figure.
depth_curve_is_up_to_date() {
  [ -f "$DEPTH_CURVE_STAMP" ] || return 1
  [ "$(cat "$DEPTH_CURVE_STAMP")" = "$DEPTH_CURVE_METRIC $DEPTH_CURVE_ARMS" ] || return 1
  file_is_newer_than_runs "$DEPTH_CURVE_STEM.pdf" $STUDY_A_RUNS || return 1
  file_is_newer_than_runs "$DEPTH_CURVE_STEM.png" $STUDY_A_RUNS || return 1
  # The caption carries the run counts and the arms, so it goes stale for
  # exactly the reasons the figure does.
  file_is_newer_than_runs "$DEPTH_CURVE_STEM.txt" $STUDY_A_RUNS
}

run_depth_curve() {
  mkdir -p "$FIGURES_DIR"
  "$PYTHON" -m experiments depth-curve \
    --arms "$DEPTH_CURVE_ARMS" \
    --metric "$DEPTH_CURVE_METRIC" \
    --store "$STORE" \
    --out "$DEPTH_CURVE_STEM"
  # Written only after the figure succeeded, so a failed run reads as a miss.
  printf '%s %s' "$DEPTH_CURVE_METRIC" "$DEPTH_CURVE_ARMS" > "$DEPTH_CURVE_STAMP"
}

step depth-curve "depth_curve_is_up_to_date" -- run_depth_curve

# --- Study B: model comparison ---------------------------------------------
#
# Five repeats x K folds = 25 samples per method at its selected depth.
# Repeat 0 reuses Study A's own partition, so only four are new work. Both
# methods draw their folds from permute_into_folds(shard_ids, k, seed=repeat),
# which is deterministic in (n, k, seed) and not in the caller -- so a given
# (repeat, fold) holds out the *same* molecules for both, which is what makes
# them pairable subjects in compare's repeated-measures design.
# One process per repeat. A repeat's five folds share its own merge
# assembly so they stay together, but repeats are fully independent -- and
# the per-fold walk is the cost, so this is ~5x. Job count kept low because
# each process holds a DASHTree (~35GB measured) plus its merged stats.
STUDY_B_JOBS="${STUDY_B_JOBS:-3}"

run_dash_repeat() {
  "$PYTHON" -m experiments cv-run-dash "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" --max-depth "$DASH_MAX_DEPTH" \
    --depths "$DASH_SELECTED_DEPTH" --repeats "$1" \
    $MODEL_CACHE_FLAG \
    --normalization std_weighted --method dash --experiment "$DASH_STUDY_B" \
    $COLLAPSE_FLAG \
    $SAVE_PREDICTIONS_FLAG
}

# One invocation per repeat, carrying every variant: assembling a repeat's
# five training models costs ~123s (load 2.6s, fold 27.3s, leave-one-group-out
# 92.9s, 30GB peak, measured on the 50 real shards), and the variants differ
# only in how that model is *read*. Splitting them across processes would pay
# that again per process for nothing.
run_sieve_repeat() {
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$SIEVE_SELECTED_DEPTH" --repeats "$1" \
    --codes-path "$CODES_PATH" --config-label "$SIEVE_CONFIG_LABEL" \
    --fit-depth "$SIEVE_MAX_DEPTH" \
    --predictor-params "$SIEVE_PREDICTOR_PARAMS" \
    --variants "$SIEVE_VARIANTS" \
    $MODEL_CACHE_FLAG \
    --normalization equal_weighted --method "$SIEVE_METHOD" $COLLAPSE_FLAG \
    --experiment "$SIEVE_STUDY_B" \
    $SAVE_PREDICTIONS_FLAG
}
export -f run_dash_repeat run_sieve_repeat
export DASH_SELECTED_DEPTH SIEVE_SELECTED_DEPTH SAVE_PREDICTIONS_FLAG
export DASH_STUDY_B SIEVE_STUDY_B SIEVE_METHOD K DASH_MAX_DEPTH
export SIEVE_VARIANTS SIEVE_SELECTED_DEPTH SIEVE_MAX_DEPTH MODEL_CACHE_FLAG

each_repeat() { echo "$STUDY_B_REPEATS" | tr ',' '\n'; }

dispatch_dash_repeats() {
  each_repeat | xargs -P "$STUDY_B_JOBS" -n 1 bash -c 'run_dash_repeat "$1"' --
}

dispatch_sieve_repeats() {
  each_repeat | xargs -P "$STUDY_B_JOBS" -n 1 bash -c 'run_sieve_repeat "$1"' --
}

step study-b-dash \
  "depth_runs_count_is $DASH_STUDY_B $DASH_SELECTED_DEPTH \
     $((K * $(n_items "$STUDY_B_REPEATS")))" -- \
  dispatch_dash_repeats

step study-b-sieve \
  "depth_runs_count_is $SIEVE_STUDY_B $SIEVE_SELECTED_DEPTH \
     $((K * $(n_items "$STUDY_B_REPEATS") * $(n_variants)))" -- \
  dispatch_sieve_repeats

# --- Study B: the HOSE arm --------------------------------------------------
#
# Five repeats x K folds at the radius Study A selected, with per-atom
# predictions, so the arm enters `compare` as 25 samples paired with the other
# two methods' -- same (repeat, fold), same held-out molecules, since
# permute_into_folds is deterministic in (n, k, seed=repeat).
# Run through a function, not `bash -c`: a nested shell does not inherit
# hose_radius, so $(hose_radius) would expand to nothing there and the arm
# would be scored at whatever radius cv-run-hose defaulted to.
# One process per (repeat, fold), at the one selected radius.
run_one_hose_study_b() {
  local repeat=${1%% *} fold=${1##* }
  "$PYTHON" -m experiments cv-run-hose "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --radii "$HOSE_B_RADIUS" --folds "$fold" --repeats "$repeat" \
    --normalization equal_weighted --method hose $COLLAPSE_FLAG \
    --experiment "$HOSE_STUDY_B" $HOSE_B_EXTRA
}
export -f run_one_hose_study_b

run_study_b_hose() {
  local rep f
  HOSE_B_RADIUS="$(hose_radius)"
  export HOSE_B_RADIUS
  export HOSE_B_EXTRA="$SAVE_PREDICTIONS_FLAG"
  export HOSE_STUDY_B
  echo "--- study B hose: radius $HOSE_B_RADIUS ---"
  for rep in $(echo "$STUDY_B_REPEATS" | tr ',' ' '); do
    for f in $(seq 0 $((K - 1))); do echo "$rep $f"; done
  done | xargs -P "$HOSE_CV_JOBS" -I{} bash -c 'run_one_hose_study_b "$1"' -- {}
}

step study-b-hose \
  "method_depth_runs_count_is $HOSE_STUDY_B hose \$(hose_radius) \
     $((K * $(n_items "$STUDY_B_REPEATS")))" -- \
  run_study_b_hose

# --- compare ---------------------------------------------------------------
#
# Repeated-measures ANOVA + Tukey HSD over Study B's 25 paired samples per
# method (Ash, Wognum, Rodriguez-Perez et al., JCIM 2025). Read the effect
# sizes and interval widths, not only the stars: at ~62.8k held-out
# molecules per fold, a systematic difference of almost any size will reach
# significance.
#
# Compared on the *unnormalized* metric by default. Two reasons. The two
# methods currently use different normalizers, so part of any gap in
# norm/mae would be the normalizer rather than the estimator. And Sieve's
# is equal_weighted (sigma^0), which design.md 13 item 8 measured as worse
# than not normalizing at all on MAE -- the right sigma for Sieve is the
# predictive variance of docs/design-update-v2-predictive-variance, not yet
# ported onto this layout. The raw metric compares estimator to estimator,
# with neither given the molecule's true total charge.
#
# RMSE leads, because it is the metric the DASH paper itself reports
# throughout -- its GNN at 0.0153 e, and the conformational-variation floor of
# 0.0125 e it calls "a lower bound on the accuracy that can be reached by an
# ML model" -- so those numbers are directly comparable to the published ones.
# The rest are drawn alongside it (COMPARE_METRICS above), since every run
# records them all from one prediction and a second panel costs no re-run.

# {method: depth} for every arm: DASH at its depth, every Sieve variant at
# SIEVE_SELECTED_DEPTH. Derived from the same SIEVE_VARIANTS that Study B ran,
# so the comparison cannot read a method or depth nobody produced.
# Arms that are RUN but kept out of the figure. The runs stay -- they are
# evidence, and the design spec's addendum cites them -- but a Tukey plot is
# for the comparison being reported, and two arms that are statistically
# indistinguishable from two others add rows without adding information. The
# continuation_recursive pair sits within 1e-6 of the flat continuation pair
# (Tukey p = 1); that similarity is a finding for the text, not a row on a
# chart. Excluding them also narrows the ANOVA to the reported arms, which is
# what the reported F and df should describe.
COMPARE_EXCLUDE="${COMPARE_EXCLUDE-sieve-element-recursive,sieve-element-recursive-eb}"

DEPTH_BY_METHOD=$(
  echo "$SIEVE_VARIANTS" | "$PYTHON" -c '
import json, sys
drop = {m for m in sys.argv[4].split(",") if m}
variants = json.load(sys.stdin)
known = {v["method"] for v in variants} | {"dash", "hose"}
unknown = drop - known
if unknown:
    raise SystemExit(f"COMPARE_EXCLUDE names no such arm: {sorted(unknown)}")
out = {"dash": int(sys.argv[1]), "hose": int(sys.argv[3])}
out = {m: d for m, d in out.items() if m not in drop}
out.update({v["method"]: int(sys.argv[2]) for v in variants if v["method"] not in drop})
print(json.dumps(out))
' "$DASH_SELECTED_DEPTH" "$SIEVE_SELECTED_DEPTH" "$(hose_radius)" "$COMPARE_EXCLUDE"
)

# A stamp of what the figures were drawn FROM, beside them. mtimes catch new
# runs; they cannot catch a changed metric list or a changed set of arms,
# which change the figures just as much. Guard on both.
COMPARE_STAMP="$FIGURES_DIR/.compare-inputs"
STUDY_B_RUNS="experiments/runs/$DASH_STUDY_B experiments/runs/$SIEVE_STUDY_B experiments/runs/$HOSE_STUDY_B"

compare_is_up_to_date() {
  [ -f "$COMPARE_STAMP" ] || return 1
  [ "$(cat "$COMPARE_STAMP")" = "$COMPARE_METRICS $DEPTH_BY_METHOD" ] || return 1
  local m
  for m in $(each_metric); do
    file_is_newer_than_runs "$(tukey_plot "$m")" $STUDY_B_RUNS || return 1
    file_is_newer_than_runs "$(simult_plot "$m")" $STUDY_B_RUNS || return 1
  done
}

run_compare() {
  mkdir -p "$FIGURES_DIR"
  local m flag
  for m in $(each_metric); do
    flag=""
    case ",$COMPARE_HIGHER_IS_BETTER," in *",$m,"*) flag="--higher-is-better" ;; esac
    echo "--- $m ---"
    "$PYTHON" -m experiments compare \
      --experiment "$DASH_STUDY_B" --experiment "$SIEVE_STUDY_B" \
      --experiment "$HOSE_STUDY_B" \
      --metric "$m" \
      --depth-by-method "$DEPTH_BY_METHOD" \
      --out "$(tukey_plot "$m")" \
      --out-simultaneous "$(simult_plot "$m")" \
      $flag
  done
  # Written only after every metric succeeded, so a partial run reads as a
  # miss rather than as a finished set of figures.
  printf '%s %s' "$COMPARE_METRICS" "$DEPTH_BY_METHOD" > "$COMPARE_STAMP"
}

step compare "compare_is_up_to_date" -- run_compare

# ===========================================================================
# Study C: which featurization?
# ===========================================================================
#
# Studies A and B both hold the featurization fixed at the incumbent
# `node:element; edge:none` and vary the estimator. Study C varies the
# featurization and holds the estimator fixed, against that same incumbent.
#
# Unlike Study B's variants, a featurization is NOT a re-reading of an
# existing fit: `attributes`/`edge_attributes` feed schema_version, so each
# arm needs its own frozen vocabulary, its own N_SHARDS fits and its own CV
# runs. That is the whole cost of this study; everything else is reused.
#
# It runs in two stages, mirroring A then B, because the effects at stake are
# small. The attribute notebook (docs/sieve-attribute-experiments-
# observations.md) measures [group,element,hybridization] at r^2 0.9895
# against plain element's 0.9907, with a fold-to-fold std of ~0.0004 -- so a
# one-repeat depth curve, whose Nadeau-Bengio intervals are 1.5x the naive
# ones at this geometry, cannot separate these arms. It is here to select a
# depth per arm, not to report a winner; the comparison stage does that, with
# 25 paired samples per arm and the same ANOVA+Tukey protocol Study B uses.
#
# Selecting a depth per arm rather than borrowing the incumbent's matters for
# this study specifically: the notebook's mechanism for why richer attribute
# sets lose at depth is class fragmentation, which predicts their optimum
# sits SHALLOWER than element's. Fixing every arm at 5 would pre-commit the
# new arms to their competitor's optimum.

# Each entry varies `attributes` and nothing else. Both arms keep
# `edge_attributes: []`, which is load-bearing rather than incidental: the
# notebook measured `bond_type` edges as determining atom-level `aromatic`
# EXACTLY (48 round-1 views, 0 ambiguous, 0 atoms affected), so an arm
# carrying both would be testing a redundancy, not a featurization.
#
# el-hyb-arom is the atom-local side of the notebook's own unrun "clean
# test"; el-ringmem carries the one thing on offer that 1-WL provably cannot
# compute, cycle membership, and so is the arm its selection principle
# ("prefer attributes that neither WL nor an enabled edge attribute can
# derive") points at most directly.
# Each entry names the arm and the flat attribute list its vocabulary is
# frozen over. `edge_attributes` defaults to none. `params` carries anything
# else SievePredictor takes that changes the SHAPE of the refinement rather
# than the attribute set -- attribute_levels, neighbor_depth -- and
# `shard_jobs` overrides the dispatch width for an arm whose fit is bigger
# than the rest.
#
# el-hyb-arom is the atom-local side of the attribute notebook's own unrun
# "clean test". el-ringmem carries the one thing on offer that 1-WL provably
# cannot compute, cycle membership.
#
# el-edgering asks the same ring question from the other side: the same count,
# moved onto the BONDS. It leaves the node alphabet at element's 11 and lets
# ring information enter through WL's (neighbor label, edge) pair encoding
# instead, so it separates "ring membership does not help" from "ring
# membership as a node attribute fragments the seed partition" -- two readings
# el-ringmem alone cannot tell apart, since it realizes 33 level-0 classes of
# which ~20 hold under 0.05% of atoms.
#
# el-hyb-arom-wlel is design.md 3.6's own hypothesis, and the direct response
# to el-hyb-arom's measured crossover: keep the full triple on the CENTRE,
# where it earned r^2 0.953 against element's 0.941 at depth 1, but seed the
# WL neighbour chain from element alone, so the neighbour alphabet -- which is
# what fragments at depth -- stays exactly the incumbent's. The notebook
# measured this shape as removing the fragmentation cost of extra centre
# attributes; it has never been measured against this corpus under CV.
#
# It was given shard_jobs 4 on the reasoning that neighbor_depth doubles the
# level count (2 + 2x10 = 22 against the others' 11) and so should roughly
# double the footprint. MEASURED: 32.7 GB peak against the flat arms' 32.0, a
# 2% difference. The level COUNT doubles; the level SIZES do not, because the
# coarse chain is seeded from element alone and stays near the incumbent's
# class counts while the expensive main chain is unchanged. It dispatches at
# the default width like every other arm.
SIEVE_FEATURIZATIONS='[
  {"label": "el-hyb-arom", "figure_label": "+ hybridization + aromatic",
   "attributes": ["element", "hybridization", "aromatic"]},
  {"label": "el-ringmem",  "figure_label": "+ num_ring_memberships (atoms)",
   "attributes": ["element", "num_ring_memberships"]},
  {"label": "el-edgering", "figure_label": "+ num_ring_memberships (bonds)",
   "attributes": ["element"],
   "edge_attributes": ["bond_num_ring_memberships"]},
  {"label": "el-hyb-arom-wlel", "figure_label": "+ hyb + arom, WL on element",
   "attributes": ["element", "hybridization", "aromatic"],
   "params": {"attribute_levels": [["element"], ["hybridization", "aromatic"]],
              "neighbor_depth": 1}}
]'

# The incumbent arm is not listed above and is never re-run: Study A already
# scored `sieve-element-continuation` across SIEVE_DEPTHS at repeat 0, and
# Study B already scored it at SIEVE_SELECTED_DEPTH across every repeat, both
# on this same store, shard partition and K. Reusing those runs is not an
# approximation -- permute_into_folds is deterministic in (n, k, seed=repeat)
# and not in the caller, so a given (repeat, fold) holds out the same
# molecules here as it did there, which is what makes the arms pairable
# subjects in the comparison stage's repeated-measures design rather than
# merely comparable numbers.
STUDY_C_INCUMBENT_METHOD=sieve-element-continuation

# `continuation`, no shrinkage: the estimator the manuscript calls the method
# proper, and one of the two arms Study A and Study B both already carry, so
# the incumbent stays free in both stages. As in Study A, the arm is reached
# by PATCHING the fitted config -- the fits below apply empirical_bayes, so
# only an explicit "shrinkage_weight": null turns it back off.
STUDY_C_ESTIMATOR='"class_estimator": "continuation", "shrinkage_weight": null'

STUDY_C_REPEATS="$STUDY_B_REPEATS"
SIEVE_STUDY_C=sieve-cv-study-c      # stage 1, the depth curve
SIEVE_STUDY_C_B=sieve-cv-study-c-b  # stage 2, the fixed-depth comparison

# Read off stage 1's curve, by a human, exactly as SIEVE_SELECTED_DEPTH is.
# These defaults are a prior -- the incumbent's own selected depth -- not a
# result: stage 2 must not be run until the curve has been looked at, which
# is what `CV_UNTIL=study-c-depth-curve` is for.
STUDY_C_SELECTED_DEPTHS="${STUDY_C_SELECTED_DEPTHS:-$(
  echo "$SIEVE_FEATURIZATIONS" | "$PYTHON" -c '
import json, sys
print(json.dumps({f["label"]: int(sys.argv[1]) for f in json.load(sys.stdin)}))
' "$SIEVE_SELECTED_DEPTH"
)}"

# label, comma-joined attributes, comma-joined edge attributes: one line per
# arm. Emitted by one python pass rather than parsed in shell, so
# SIEVE_FEATURIZATIONS stays the single place an arm is described.
each_featurization() {
  echo "$SIEVE_FEATURIZATIONS" | "$PYTHON" -c '
import json, sys
for f in json.load(sys.stdin):
    print(f["label"], ",".join(f["attributes"]),
          ",".join(f.get("edge_attributes", [])), sep="\t")
'
}

# One arm entry, by label, for the helpers that need more than the three TSV
# fields.
featurization_entry() {
  echo "$SIEVE_FEATURIZATIONS" | "$PYTHON" -c '
import json, sys
label = sys.argv[1]
for f in json.load(sys.stdin):
    if f["label"] == label:
        print(json.dumps(f)); break
else:
    raise SystemExit(f"SIEVE_FEATURIZATIONS has no arm {label!r}")
' "$1"
}

# DERIVED from the incumbent's own fit params with `attributes` swapped,
# never written out a second time. That is what makes "featurization is the
# only axis that moves" a property of the script rather than a claim in a
# comment: a change to SIEVE_PREDICTOR_PARAMS reaches every Study C arm too.
featurization_params() {
  featurization_entry "$1" | "$PYTHON" -c '
import json, sys
params = json.loads(sys.argv[1])
arm = json.load(sys.stdin)
params["attributes"] = arm["attributes"]
params["edge_attributes"] = arm.get("edge_attributes", [])
params.update(arm.get("params", {}))
print(json.dumps(params))
' "$SIEVE_PREDICTOR_PARAMS"
}

# Per-arm dispatch width, for an arm whose fit is bigger than the rest.
featurization_jobs() {
  featurization_entry "$1" | "$PYTHON" -c '
import json, sys
print(int(json.load(sys.stdin).get("shard_jobs", sys.argv[1])))
' "$STUDY_C_SHARD_JOBS"
}

# What the legend shows. Explicit per arm, because the attribute join alone
# cannot tell an arm that moved information onto the EDGES, or into the
# neighbour chain, from one that did neither -- both would read "element".
featurization_figure_label() {
  featurization_entry "$1" | "$PYTHON" -c '
import json, sys
arm = json.load(sys.stdin)
print(arm.get("figure_label") or " + ".join(arm["attributes"]))
'
}

featurization_codes()  { echo "experiments/stores/$STORE/sieve-codes-$1.json"; }
featurization_label()  { echo "$1-eb"; }   # matches SIEVE_CONFIG_LABEL's convention: the FIT applies empirical_bayes
featurization_method() { echo "sieve-$1-continuation"; }
featurization_variant() {
  printf '[{"method": "%s", %s}]' "$(featurization_method "$1")" "$STUDY_C_ESTIMATOR"
}
featurization_depth() {
  echo "$STUDY_C_SELECTED_DEPTHS" | "$PYTHON" -c '
import json, sys
depths = json.load(sys.stdin)
label = sys.argv[1]
if label not in depths:
    raise SystemExit(f"STUDY_C_SELECTED_DEPTHS names no depth for {label!r}")
print(int(depths[label]))
' "$1"
}

# Neither method_runs_count_is nor depth_runs_count_is can express stage 2's
# claim on its own. Its arms sit at DIFFERENT depths inside one experiment, so
# counting an experiment's runs at one depth spans arms, and counting one
# arm's runs across all depths keeps counting the runs from a previously
# selected depth -- which stay on disk deliberately, as evidence, and would
# wedge the guard the moment a depth is revised. Both fields are read
# structurally from the manifest, as the two guards above read theirs.
method_depth_runs_count_is() {
  local experiment=$1 method=$2 depth=$3 expected=$4
  "$PYTHON" - "$experiment" "$method" "$depth" "$expected" <<'PY'
import json
import sys
from pathlib import Path

experiment, method, depth, expected = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
)
found = 0
for manifest in Path("experiments/runs", experiment).glob("*__*/manifest.json"):
    if not (manifest.parent / "metrics.json").exists():
        continue  # a half-written run is not a finished sample
    cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
    found += cv.get("method") == method and int(cv.get("depth", -1)) == depth
sys.exit(0 if found == expected else 1)
PY
}

# --- Study C: frozen vocabularies ------------------------------------------
#
# One per arm, for the same reason the incumbent has one: shards fit on
# disjoint molecule sets must agree on what an integer code means, or
# check_mergeable refuses them.
study_c_codes_exist() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    file_exists "$(featurization_codes "$label")" || return 1
  done < <(each_featurization)
}

# Per-arm, not all-or-nothing: study_c_codes_exist fails as soon as ONE arm
# is missing a vocabulary, so adding a fifth arm would otherwise rebuild the
# four that already have one. That rebuild is deterministic -- measured, it
# reproduced the exact schema_version the fits on disk carry, so nothing broke
# -- but it is minutes of work per arm to write a file back identical, and it
# rewrites the very file 50 shard fits are pinned to. Skipping is both cheaper
# and the safer of the two.
build_study_c_codes() {
  local label attrs edges out
  while IFS=$'\t' read -r label attrs edges; do
    out="$(featurization_codes "$label")"
    if file_exists "$out"; then
      echo "--- codes: $label already frozen; skipping ---"
      continue
    fi
    echo "--- codes: $label (node $attrs; edge ${edges:-none}) ---"
    "$PYTHON" -m experiments build-sieve-codes "$STORE" \
      --attributes "$attrs" --edge-attributes "$edges" \
      --out "$out"
  done < <(each_featurization)
}

step study-c-codes "study_c_codes_exist" -- build_study_c_codes

# --- Study C: shard fits ---------------------------------------------------
#
# One dispatch per arm, each the same shape as sieve-shard-fits: N_SHARDS
# single-threaded processes, one shard each, at SIEVE_MAX_DEPTH so every
# shallower depth comes from truncating the merged model.
#
# Six concurrent fits, set before either arm had been measured, on the
# reasoning that a Sieve fit's footprint is driven by its CLASS COUNT and that
# jointly refining three attributes fragments classes faster than one does.
# MEASURED, both arms, depth 10: 32.0 GB peak RSS, the same as the incumbent's.
# Class fragmentation is real at the seed (27 and 33 level-0 classes against
# element's 11) but does not inflate peak RSS at depth, where the class count
# is near saturation either way. Eight is therefore as safe here as it is for
# the incumbent; six is left as the default only because nothing has needed
# the extra two, and 8 x 32GB sits at 256GB of this box's 503GB.
STUDY_C_SHARD_JOBS="${STUDY_C_SHARD_JOBS:-6}"

fit_one_study_c_shard() {
  "$PYTHON" -m experiments cv-fit-sieve-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$SIEVE_MAX_DEPTH" --shard "$1" \
    --codes-path "$FEAT_CODES" --config-label "$FEAT_CONFIG_LABEL" \
    --predictor-params "$FEAT_PARAMS" \
    $COLLAPSE_FLAG
}
export -f fit_one_study_c_shard

study_c_shard_fits_done() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    shard_fits_count_is \
      "fit-sieve-$(featurization_label "$label")-w$SIEVE_MAX_DEPTH-s" \
      "$N_SHARDS" || return 1
  done < <(each_featurization)
}

# The per-arm settings are exported into the environment rather than wrapped
# in a nested `bash -c`: a nested shell does not inherit the caller's shell
# FUNCTIONS unless they are exported, so all_shard_ids vanished there and
# xargs got an empty -P. Exporting and then dispatching in place is also
# exactly dispatch_sieve_shards' own shape, one arm at a time.
dispatch_study_c_shards() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    echo "--- shard fits: $label ---"
    export FEAT_CODES="$(featurization_codes "$label")"
    export FEAT_CONFIG_LABEL="$(featurization_label "$label")"
    export FEAT_PARAMS="$(featurization_params "$label")"
    all_shard_ids | xargs -P "$(featurization_jobs "$label")" -n 1 \
      bash -c 'fit_one_study_c_shard "$1"' --
  done < <(each_featurization)
}

step study-c-shard-fits "study_c_shard_fits_done" -- dispatch_study_c_shards

# --- Study C stage 1: the depth curve --------------------------------------
#
# Study A's shape -- one repeat, K folds, every depth -- and, like Study A,
# no --score-train flag needed: run_sieve_cv already records train/rmse and
# friends analytically, from the fitted model's own stored statistics, for
# every arm here at no extra cost. Five arms that would once have skipped
# the train curve outright to afford stage 2's repeats now get it for free.
study_c_curve_done() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    method_runs_count_is "$SIEVE_STUDY_C" "$(featurization_method "$label")" \
      "$((K * $(n_items "$SIEVE_DEPTHS")))" || return 1
  done < <(each_featurization)
}

run_study_c_curve() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    echo "--- study C curve: $label ---"
    "$PYTHON" -m experiments cv-run-sieve "$STORE" \
      --n-shards "$N_SHARDS" --k "$K" \
      --depths "$SIEVE_DEPTHS" --repeats "$STUDY_A_REPEATS" \
      --codes-path "$(featurization_codes "$label")" \
      --config-label "$(featurization_label "$label")" \
      --fit-depth "$SIEVE_MAX_DEPTH" \
      --predictor-params "$(featurization_params "$label")" \
      --variants "$(featurization_variant "$label")" \
      $MODEL_CACHE_FLAG $COLLAPSE_FLAG \
      --normalization equal_weighted \
      --method "$(featurization_method "$label")" \
      --experiment "$SIEVE_STUDY_C"
  done < <(each_featurization)
}

step study-c-curve "study_c_curve_done" -- run_study_c_curve

# --- Study C stage 1's figure ----------------------------------------------
#
# One panel per metric, not one per arm: every arm here counts depth in the
# same unit (WL refinement rounds), so unlike Study A -- where DASH's path
# length and Sieve's refinement depth are different quantities and cannot
# share an x axis -- these are read against each other in place.
#
# Depth 0 is kept off for the incumbent's own reason: with no refinement the
# model is attribute-wise pooled means, whose r^2 near 0.46 compresses the
# 0.99 band where every difference between the real depths lives.
STUDY_C_CURVE_STEM="$FIGURES_DIR/depth-curve-study-c"
STUDY_C_CURVE_STAMP="$FIGURES_DIR/.depth-curve-study-c-inputs"
STUDY_C_RUNS="experiments/runs/$SIEVE_STUDY_A experiments/runs/$SIEVE_STUDY_C"

STUDY_C_CURVE_ARMS=$(
  printf '[{"experiment": "%s", "method": "%s", "label": "element (incumbent)",' \
    "$SIEVE_STUDY_A" "$STUDY_C_INCUMBENT_METHOD"
  printf ' "panel": "Sieve", "x_label": "Maximum Refinement Depth", "min_depth": 1}'
  while IFS=$'\t' read -r label attrs edges; do
    printf ', {"experiment": "%s", "method": "%s", "label": "%s",' \
      "$SIEVE_STUDY_C" "$(featurization_method "$label")" \
      "$(featurization_figure_label "$label")"
    printf ' "panel": "Sieve", "x_label": "Maximum Refinement Depth", "min_depth": 1}'
  done < <(each_featurization)
  printf ']'
)

study_c_curve_is_up_to_date() {
  [ -f "$STUDY_C_CURVE_STAMP" ] || return 1
  [ "$(cat "$STUDY_C_CURVE_STAMP")" = "$DEPTH_CURVE_METRIC $STUDY_C_CURVE_ARMS" ] || return 1
  file_is_newer_than_runs "$STUDY_C_CURVE_STEM.pdf" $STUDY_C_RUNS || return 1
  file_is_newer_than_runs "$STUDY_C_CURVE_STEM.png" $STUDY_C_RUNS || return 1
  file_is_newer_than_runs "$STUDY_C_CURVE_STEM.txt" $STUDY_C_RUNS
}

run_study_c_curve_figure() {
  mkdir -p "$FIGURES_DIR"
  "$PYTHON" -m experiments depth-curve \
    --arms "$STUDY_C_CURVE_ARMS" \
    --metric "$DEPTH_CURVE_METRIC" \
    --store "$STORE" \
    --out "$STUDY_C_CURVE_STEM"
  printf '%s %s' "$DEPTH_CURVE_METRIC" "$STUDY_C_CURVE_ARMS" > "$STUDY_C_CURVE_STAMP"
}

step study-c-depth-curve "study_c_curve_is_up_to_date" -- run_study_c_curve_figure

# --- Study C stage 2: the comparison ---------------------------------------
#
# Study B's shape: five repeats x K folds = 25 paired samples per arm, each
# at the depth stage 1 selected for it, with per-atom predictions written.
#
# STOP HERE unless STUDY_C_SELECTED_DEPTHS has actually been set from stage
# 1's curve -- its defaults are the incumbent's depth, which is a prior and
# not a result.
#
# One process per (arm, repeat). Unlike Study B these hold no DASHTree, but a
# repeat's merge assembly still peaks near 30GB, so the job count stays low.
STUDY_C_JOBS="${STUDY_C_JOBS:-3}"

run_study_c_repeat() {
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$FEAT_DEPTH" --repeats "$1" \
    --codes-path "$FEAT_CODES" --config-label "$FEAT_CONFIG_LABEL" \
    --fit-depth "$SIEVE_MAX_DEPTH" \
    --predictor-params "$FEAT_PARAMS" \
    --variants "$FEAT_VARIANT" \
    $MODEL_CACHE_FLAG \
    --normalization equal_weighted --method "$FEAT_METHOD" $COLLAPSE_FLAG \
    --experiment "$SIEVE_STUDY_C_B" \
    $SAVE_PREDICTIONS_FLAG
}
export -f run_study_c_repeat
export SIEVE_STUDY_C_B

study_c_comparison_done() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    method_depth_runs_count_is "$SIEVE_STUDY_C_B" \
      "$(featurization_method "$label")" "$(featurization_depth "$label")" \
      "$((K * $(n_items "$STUDY_C_REPEATS")))" || return 1
  done < <(each_featurization)
}

dispatch_study_c_repeats() {
  local label attrs
  while IFS=$'\t' read -r label attrs edges; do
    echo "--- study C comparison: $label at depth $(featurization_depth "$label") ---"
    export FEAT_CODES="$(featurization_codes "$label")"
    export FEAT_CONFIG_LABEL="$(featurization_label "$label")"
    export FEAT_PARAMS="$(featurization_params "$label")"
    export FEAT_VARIANT="$(featurization_variant "$label")"
    export FEAT_METHOD="$(featurization_method "$label")"
    export FEAT_DEPTH="$(featurization_depth "$label")"
    echo "$STUDY_C_REPEATS" | tr ',' '\n' \
      | xargs -P "$STUDY_C_JOBS" -n 1 \
          bash -c 'run_study_c_repeat "$1"' --
  done < <(each_featurization)
}

step study-c-comparison "study_c_comparison_done" -- dispatch_study_c_repeats

# --- Study C stage 2's figures ---------------------------------------------
#
# The same ANOVA + Tukey protocol and the same metric set as `compare`, over
# Study B's incumbent arm plus Study C's -- pooled across two experiments
# because the incumbent's 25 samples are already there and share this study's
# subjects exactly. Read the effect sizes and interval widths, not the stars:
# at ~62.8k held-out molecules per fold almost any systematic difference
# reaches significance, and the differences this study is looking for are
# around 0.001 in r^2.
tukey_plot_c()  { echo "$FIGURES_DIR/tukey-study-c-$(metric_slug "$1").png"; }
simult_plot_c() { echo "$FIGURES_DIR/simultaneous-study-c-$(metric_slug "$1").png"; }

STUDY_C_DEPTH_BY_METHOD=$(
  {
    printf '%s\t%s\n' "$STUDY_C_INCUMBENT_METHOD" "$SIEVE_SELECTED_DEPTH"
    while IFS=$'\t' read -r label attrs edges; do
      printf '%s\t%s\n' "$(featurization_method "$label")" "$(featurization_depth "$label")"
    done < <(each_featurization)
  } | "$PYTHON" -c '
import json, sys
print(json.dumps({
    m: int(d) for m, d in (line.split("\t") for line in sys.stdin.read().splitlines() if line)
}))
'
)

STUDY_C_COMPARE_STAMP="$FIGURES_DIR/.compare-study-c-inputs"
STUDY_C_B_RUNS="experiments/runs/$SIEVE_STUDY_B experiments/runs/$SIEVE_STUDY_C_B"

study_c_compare_is_up_to_date() {
  [ -f "$STUDY_C_COMPARE_STAMP" ] || return 1
  [ "$(cat "$STUDY_C_COMPARE_STAMP")" = "$COMPARE_METRICS $STUDY_C_DEPTH_BY_METHOD" ] || return 1
  local m
  for m in $(each_metric); do
    file_is_newer_than_runs "$(tukey_plot_c "$m")" $STUDY_C_B_RUNS || return 1
    file_is_newer_than_runs "$(simult_plot_c "$m")" $STUDY_C_B_RUNS || return 1
  done
}

run_study_c_compare() {
  mkdir -p "$FIGURES_DIR"
  local m flag
  for m in $(each_metric); do
    flag=""
    case ",$COMPARE_HIGHER_IS_BETTER," in *",$m,"*) flag="--higher-is-better" ;; esac
    echo "--- $m ---"
    "$PYTHON" -m experiments compare \
      --experiment "$SIEVE_STUDY_B" --experiment "$SIEVE_STUDY_C_B" \
      --metric "$m" \
      --depth-by-method "$STUDY_C_DEPTH_BY_METHOD" \
      --out "$(tukey_plot_c "$m")" \
      --out-simultaneous "$(simult_plot_c "$m")" \
      $flag
  done
  printf '%s %s' "$COMPARE_METRICS" "$STUDY_C_DEPTH_BY_METHOD" > "$STUDY_C_COMPARE_STAMP"
}

step study-c-compare "study_c_compare_is_up_to_date" -- run_study_c_compare

# --- final held-out evaluation ---------------------------------------------
#
# Once, at the end, outside both studies: merge every shard into one
# full-train model and score the untouched 10% test split. The headline
# number, and the only thing here that touches `test`.
DASH_MERGED=experiments/results/dash-merged/tree_stats.npz
# Named for the depth the shards were FIT at, not a selected one: there is a
# single merged artifact, and a shallower model is a truncation of it. Using
# the selected depth here globbed shard fits that never existed.
SIEVE_MERGED="experiments/results/sieve-merged/tree_stats-w${SIEVE_MAX_DEPTH}.npz"

# Merge order does not affect the result (both merges are commutative and
# associative); sorted purely for a readable command line.
step merge-dash-shards \
  "file_exists $DASH_MERGED" -- \
  bash -c "set -euo pipefail; '$PYTHON' -m experiments merge-states --predictor dash \
             --out '$DASH_MERGED' \
             \$(ls experiments/runs/cv-shard-fits/fit-dash-s*__*/tree_stats.npz | sort)"

step merge-sieve-shards \
  "file_exists $SIEVE_MERGED" -- \
  bash -c "set -euo pipefail; '$PYTHON' -m experiments merge-states --predictor sieve \
             --out '$SIEVE_MERGED' \
             \$(ls experiments/runs/cv-shard-fits/fit-sieve-$SIEVE_CONFIG_LABEL-w${SIEVE_MAX_DEPTH}-s*__*/tree_stats.npz | sort)"

step final-holdout-dash \
  "runs_exist dash-final-holdout final" -- \
  "$PYTHON" -m experiments run \
    --config experiments/configs/dash-charge-example.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_depth="$DASH_SELECTED_DEPTH" \
    --set tree_stats_load_path="$DASH_MERGED" \
    --set normalization=std_weighted \
    --set run.experiment=dash-final-holdout \
    --set run.batch_id=final

# NOTE: unrun so far. $SIEVE_MERGED is a depth-$SIEVE_MAX_DEPTH model, so this
# step needs the truncation seam that run_sieve_cv gets from truncate_model --
# setting predictor.params.max_wl_depth alone has not been verified to
# truncate a loaded model. Check that before trusting its number.
step final-holdout-sieve \
  "runs_exist sieve-final-holdout final" -- \
  "$PYTHON" -m experiments run \
    --config experiments/configs/sieve-charge-sweep.yaml \
    --set data.store="$STORE" \
    --set data.split_column=split \
    --set predictor.params.max_wl_depth="$SIEVE_SELECTED_DEPTH" \
    --set tree_stats_load_path="$SIEVE_MERGED" \
    --set normalization=equal_weighted \
    --set run.experiment=sieve-final-holdout \
    --set run.batch_id=final
