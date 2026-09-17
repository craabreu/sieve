# experiments

A node-level regression harness for molecular stores: a run names an atomic
property, fits a predictor on the train split, and scores per-atom
predictions against it. Nothing in the harness is specific to one dataset --
a dataset arrives as a data-preparation script that writes the store format
below, plus a config naming its property.

DASH's MBIS atomic partial charges (`MBIScharge`, from
`dashMoleculesSDF_v2.sdf`) are the series this harness was built for, and
`prepare_dash.py` is its prep script; `docs/dash_molecules_sdf.md` records
what running it against the real published SDF turned up.

Fully independent of `cosmo_experiments` (sigma-profile prediction) at the
harness level -- no shared package, only the core `sieve` dependency in
common. See
`docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`
and `docs/superpowers/specs/2026-09-09-experiments-generalization-design.md`
for the full design; `docs/dash_molecules_sdf.md` covers findings from
actually running `prepare-store` against the real published SDF (a download
gotcha, a property-serialization bug, and the discovery that the file holds
two distinct record schemas).

`docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md`
describes an earlier nested-run orchestration (a parent run doing
`fit()`+raw-predict, one MLflow child run per normalization scheme) that has
since been replaced by the flat run's own `normalization` config key --
superseded, kept for history.

## Adding a dataset

1. Write `experiments/<dataset>_prep.py` producing
   `stores/<name>/molecules.parquet` with:
   - `mol` -- `data.mol_to_blob(mol)` bytes, one conformer per row, its
     atoms carrying your target property (`atom.SetDoubleProp`);
   - `split` -- `"train"` / `"val"` / `"test"` per row;
   - optionally a per-molecule column your atoms' values should sum to;
   - any further columns, carried through as per-conformer identifiers.
2. Name it in a config:
   ```yaml
   target:
     atom_property: my_property
     molecule_property: my_total   # omit if there is no sum constraint
     label: my property (units)    # optional, plot axes
   ```
3. Everything else already works: `store_ops.py`'s subsample / partition /
   united-atom commands, every predictor, the metrics, the plots.

Omitting `molecule_property` disables the `sum_constraint/*` metrics, the
residual panel, and the `normalization` key -- a config that sets
`normalization` without it is rejected at load.

## Usage

### Data prep

`prepare-store` (no arguments needed) is the only command that runs on its
own -- it downloads, parses, and splits the real ~1M-conformer store. It
creates exactly one store, `dash-molecules`. Everything else below is a
separate, opt-in follow-up step; none of them chain automatically.

    uv run python -m experiments prepare-store

For quick, scientifically sound iteration against a much smaller store,
subsample it -- `subsample-store` preserves the source's own real
train/val/test fractions (measured directly, not assumed 80/10/10), so a
50k-molecule subsample is a representative stand-in for the full store, not
a biased slice like `run --limit`'s literal row prefix:

    uv run python -m experiments subsample-store dash-molecules-50k

For several such stores at once -- independent replicates, or folds whose
molecules must not overlap -- `--n-stores` draws them *without replacement
across all of them*: each split is shuffled once and handed out in
contiguous blocks, so no molecule lands in two stores and every store still
carries the source's own split fractions. They are named `DEST-1` ...
`DEST-N` (with the default `--n-stores 1` the store keeps the bare `DEST`
name):

    uv run python -m experiments subsample-store dash-molecules-10k --n-stores 5 --n-molecules 10000

Because disjoint stores can't be clamped independently, a request the
source can't fill raises before any store is written, naming the split that
came up short -- unlike the single-store case, which clamps and warns.

`subsample-store` samples; it always leaves most of a large source unused.
To instead divide a store's *entire* molecule set into N disjoint stores --
nothing left over, every conformer of every molecule kept by default --
use `partition-store`. Each split is shuffled once and cut into N near-equal
contiguous blocks (sizes differ by at most one), so no molecule is used
twice and none is skipped:

    uv run python -m experiments partition-store dash-molecules-part --n-stores 10

`--conformers-per-molecule N` caps conformers the same way `subsample-store`
does, if a run needs a smaller per-molecule footprint; the default is
unlimited.

To also get a united-atom (heavy-atom-only) version of a store -- every
conformer's hydrogens removed via rdkit's own `Chem.RemoveHs`, each removed
H's own atom property (`--atom-property`, default `MBIScharge`) folded onto
the heavy atom it was bonded to, any H rdkit itself declines to remove left
untouched:

    uv run python -m experiments to-united-atom dash-molecules-50k-ua --source dash-molecules-50k

`subsample-store` and `to-united-atom` commute (selection and per-row
content transform act on disjoint columns), but subsample first is the
efficient order -- it runs `to-united-atom`'s own rdkit chemistry on the
smaller store instead of the full one.

### Running an experiment

A single-predictor run:

    uv run python -m experiments run --config configs/dash-charge-example.yaml

By default a run's own metrics come straight from `predictor.predict()`. To
instead apply one of DASH's own post-hoc charge-conservation schemes to a
normalizable predictor's raw walk output (`predict_raw`, no re-fit needed),
set the top-level `normalization` key to `std_weighted` or `equal_weighted`:

    uv run python -m experiments run --config configs/dash-charge-example.yaml --set normalization=std_weighted

Set `save_tree_stats: true` to write a fitted predictor's state to that
run's own `tree_stats.npz`, for reuse later. It is opt-in on purpose: it
only pays when `fit()` is genuinely expensive (as for `dash`), and a
sweep of cheap-to-fit `sieve` runs will otherwise write tens of GB of
model state to avoid refits measured in seconds. A later run can then
skip `fit()` entirely via `tree_stats_load_path` (which never re-saves a
copy of its own -- the loaded path is already the provenance record):

    uv run python -m experiments run --config configs/dash-charge-example.yaml --set tree_stats_load_path=experiments/runs/dash-charges/<earlier-run-dir>/tree_stats.npz

Point a run at a different store with `--set data.store=dash-molecules-50k`,
or use `--limit N` for a quick sanity check against whatever store is
configured.

Runs are **untracked by MLflow by default** -- pass `--track` to log one.
`summarize`/`sweep` always read `manifest.json`/`metrics.json` straight off
disk regardless, and MLflow's own artifact duplication
(`mlflow.log_artifacts` copies every file a run writes into
`mlflow_artifacts/` too) has been the direct cause of more than one
disk-usage incident on a shared machine. `promote-run` gives an
already-untracked run an MLflow record after the fact, if you decide you
want one.

Set `run.batch_id` to tie several independently-launched runs together --
e.g. one predictor run per `partition-store` fold -- into one shared,
sortable/greppable run-directory prefix (`<batch_id>__<predictor>-<store>-
s<seed>__<timestamp>__<uuid>`), and a matching MLflow tag. `experiment`
stays a broad, reused category (`dash-charges`); `batch_id` identifies one
specific sweep instance, so re-running the same batch next month is
visibly distinct from today's:

    for i in 1 2 3 4 5 6 7 8 9 10; do
      uv run python -m experiments run --config configs/dash-charge-example.yaml \
        --set data.store=dash-molecules-10fold-$i --set run.batch_id=dash-10fold-2026-09-03
    done

### Collapsing equivalent molecules

A molecule's conformers, an exactly duplicated structure and an enantiomer are
indistinguishable to every predictor here, so counting them separately
reweights class means for no informational reason. Annotate the store once:

    uv run python -m experiments annotate-collapse dash-molecules

then pass `--collapse` to any `cv-fit-*-shards` command. The collapse applies
to the **training** side only; held-out rows stay one per conformer, so the
metric still measures per-conformer error and still detects the equivalence
premise failing. Every CV run whose store carries the key also records
`floor/rmse`, the within-key scatter on its own held-out set -- the error no
graph-based model can avoid.

Diastereomers and E/Z isomers are deliberately NOT merged: they are
measurably different molecules.

**Before re-running any published result**, check the migration: fit one arm
with `--collapse --weight-by-collapse` and confirm it reproduces that arm's
existing number exactly. That validates the annotation, the grouping and the
representative selection at once. `--collapse` alone changes the fit, which is
the point; `minimum_support` then counts distinct structures, so the present
value of 12 (chosen as "at least four molecules") becomes 4.

**Use a non-empirical-Bayes arm for that check.** `--weight-by-collapse`
replaces a group's *k* members with *k* copies of their mean, which preserves
each class's count and sum -- and therefore its mean -- exactly, but *not* its
variance: the within-group scatter is gone. Measured on a synthetic corpus
with known duplicate groups, the largest discrepancy in a class mean is 0 and
in a class variance 8.2e-02, and predictions come back bit-for-bit identical
under `pooled`, under `continuation`, and under count-rule shrinkage (3e-18,
i.e. rounding). Under `shrinkage_weight="empirical_bayes"` they differ by
1.2e-01, because EB estimates its own alpha from exactly the variance
components that mean-replication destroys. That is correct behaviour, not a
broken annotation -- but run the check on `sieve-element-continuation-eb` and
it looks identical to one, so don't.

See `docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md`.

### Collecting results

Gather every run's `metrics.json` under `runs/` into one CSV:

    uv run python -m experiments summarize

### Analytic training metrics

A fitted Sieve or HOSE model already stores, per class, the count, mean and
mean-squared deviation of the training atoms it holds -- enough to reproduce
the training error, R², η², the support distribution and the leave-one-out
error exactly, with no molecules loaded and no walk run. Every `cv-run-sieve`
and `cv-run-hose` run records these as `train/rmse`, `train/r2`, `train/eta2`,
`train/matched_fraction` and `train/frac_support_lt_*` automatically, at no
extra cost -- `--score-train`'s own predict-based pass is no longer needed for
either arm.

To read the same curve from an already-saved state directly, at every depth
or radius it carries:

    uv run python -m experiments analytic-curve --predictor sieve tree_stats.npz
    uv run python -m experiments analytic-curve --predictor hose --loo hose.npz --out curve.csv

**DASH has no analytic curve.** Its per-node statistics describe the atoms
passing *through* a node, not the ones that stop there at a shallower depth,
and reconstructing that split would need a refit; DASH keeps `--score-train`
as its only route to a training curve. HOSE's curve is exact only at the
baseline `n_min=1`, and only for a state saved after this feature landed --
an older state predating the `sumsq` column still loads and predicts exactly
as before, but its analytic entry point raises rather than fabricate a
number. See `docs/superpowers/specs/2026-09-17-analytic-training-metrics-
design.md` for the full argument and both measurements.

### Cross-validation (model comparison)

`cv.py` supersedes the fold-sweep workflow above for the charges series --
see `docs/superpowers/specs/2026-09-12-cv-shard-redesign-design.md` for the
full design. `prepare-store` now writes a 90/10 train/test split plus
`cluster`/`shard` columns (`s00`..`s{N-1}` on train rows); choose `N`
(`--n-shards`) from real evidence first:

    uv run python -m experiments cluster-report

Freeze the Sieve attribute vocabulary once, over the whole train split
(skip this for `dash`, which has no vocabulary to discover):

    uv run python -m experiments build-sieve-codes --attributes element --edge-attributes "" --out codes.json

Fit each shard once, at the deepest depth needed -- shallower depths are
recovered by truncating the merged model, so this is N fits, not N per
depth (predicts nothing; a shard is only ever used merged). Add `--shard
sNN` to fit one shard per process under an `xargs -P` dispatch:

    uv run python -m experiments cv-fit-dash-shards --n-shards 25 --max-depth 16
    uv run python -m experiments cv-fit-sieve-shards --n-shards 25 --max-depth 10 \
      --codes-path codes.json --config-label element-eb \
      --predictor-params '{"attributes": ["element"], "edge_attributes": [], "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}'

Then run a CV study -- Study A (one repeat, a depth curve) or Study B (5
repeats at one selected depth, feeding `compare`):

    uv run python -m experiments cv-run-dash --n-shards 25 --max-depth 16 --depths 1,2,4,6,8,10,12,14,16 --repeats 0
    uv run python -m experiments cv-run-sieve --n-shards 25 --depths 6 --repeats 0,1,2,3,4 \
      --codes-path codes.json --config-label element-eb --predictor-params '...'

`compare` reads any set of CV experiments and runs repeated-measures ANOVA +
Tukey HSD (Ash, Wognum, Rodríguez-Pérez et al., *JCIM* 2025,
doi:10.1021/acs.jcim.5c01609):

    uv run python -m experiments compare --experiment dash-cv-study-b --experiment sieve-cv-study-b \
      --depth-by-method '{"dash": 16, "sieve-element-continuation-eb": 6}' --out tukey.png

`merge-states` generalizes the old `merge-shards` to either predictor, for a
final held-out evaluation once all shards are fit:

    uv run python -m experiments merge-states --predictor dash --out merged.npz runs/cv-shard-fits/fit-dash-s*/tree_stats.npz

See `experiments/workflows/cv_charges.sh` for the full sequence, end to
end -- one guarded, idempotent step per stage, with `CV_UNTIL=<step>` to
stop after a given one.
