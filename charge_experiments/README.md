# charge_experiments

A second, independent experiment series: predicting DASH's MBIS atomic
partial charges (the `MBIScharge` SDF property) on DASH's own published
training data (`dashMoleculesSDF_v2.sdf`, ETH Research Collection).

Fully independent of `cosmo_experiments` (sigma-profile prediction) at the
harness level -- no shared package, only the core `sieve` dependency in
common. See
`docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`
for the full design and `docs/dash_molecules_sdf.md` for findings from
actually running `prepare-store` against the real published SDF (a download
gotcha, a property-serialization bug, and the discovery that the file holds
two distinct record schemas).

`docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md`
describes an earlier nested-run orchestration (a parent run doing
`fit()`+raw-predict, one MLflow child run per normalization scheme) that has
since been replaced by the flat run's own `normalization` config key --
superseded, kept for history.

## Usage

### Data prep

`prepare-store` (no arguments needed) is the only command that runs on its
own -- it downloads, parses, and splits the real ~1M-conformer store. It
creates exactly one store, `dash-molecules`. Everything else below is a
separate, opt-in follow-up step; none of them chain automatically.

    uv run python -m charge_experiments prepare-store

For quick, scientifically sound iteration against a much smaller store,
subsample it -- `subsample-store` preserves the source's own real
train/val/test fractions (measured directly, not assumed 80/10/10), so a
50k-molecule subsample is a representative stand-in for the full store, not
a biased slice like `run --limit`'s literal row prefix:

    uv run python -m charge_experiments subsample-store dash-molecules-50k

For several such stores at once -- independent replicates, or folds whose
molecules must not overlap -- `--n-stores` draws them *without replacement
across all of them*: each split is shuffled once and handed out in
contiguous blocks, so no molecule lands in two stores and every store still
carries the source's own split fractions. They are named `DEST-1` ...
`DEST-N` (with the default `--n-stores 1` the store keeps the bare `DEST`
name):

    uv run python -m charge_experiments subsample-store dash-molecules-10k --n-stores 5 --n-molecules 10000

Because disjoint stores can't be clamped independently, a request the
source can't fill raises before any store is written, naming the split that
came up short -- unlike the single-store case, which clamps and warns.

`subsample-store` samples; it always leaves most of a large source unused.
To instead divide a store's *entire* molecule set into N disjoint stores --
nothing left over, every conformer of every molecule kept by default --
use `partition-store`. Each split is shuffled once and cut into N near-equal
contiguous blocks (sizes differ by at most one), so no molecule is used
twice and none is skipped:

    uv run python -m charge_experiments partition-store dash-molecules-part --n-stores 10

`--conformers-per-molecule N` caps conformers the same way `subsample-store`
does, if a run needs a smaller per-molecule footprint; the default is
unlimited.

To also get a united-atom (heavy-atom-only) version of a store -- every
conformer's hydrogens removed via rdkit's own `Chem.RemoveHs`, each removed
H's charge folded onto the heavy atom it was bonded to, any H rdkit itself
declines to remove left untouched:

    uv run python -m charge_experiments to-united-atom dash-molecules-50k-ua --source dash-molecules-50k

`subsample-store` and `to-united-atom` commute (selection and per-row
content transform act on disjoint columns), but subsample first is the
efficient order -- it runs `to-united-atom`'s own rdkit chemistry on the
smaller store instead of the full one.

### Running an experiment

A single-predictor run:

    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml

By default a run's own metrics come straight from `predictor.predict()`. To
instead apply one of DASH's own post-hoc charge-conservation schemes to a
normalizable predictor's raw walk output (`predict_raw`, no re-fit needed),
set the top-level `normalization` key to `std_weighted` or `equal_weighted`:

    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml --set normalization=std_weighted

Set `save_tree_stats: true` to write a fitted predictor's state to that
run's own `tree_stats.npz`, for reuse later. It is opt-in on purpose: it
only pays when `fit()` is genuinely expensive (as for `dash`), and a
sweep of cheap-to-fit `sieve` runs will otherwise write tens of GB of
model state to avoid refits measured in seconds. A later run can then
skip `fit()` entirely via `tree_stats_load_path` (which never re-saves a
copy of its own -- the loaded path is already the provenance record):

    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml --set tree_stats_load_path=charge_experiments/runs/dash-charges/<earlier-run-dir>/tree_stats.npz

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
      uv run python -m charge_experiments run --config configs/dash-charge-example.yaml \
        --set data.store=dash-molecules-10fold-$i --set run.batch_id=dash-10fold-2026-09-03
    done

### Collecting results

Gather every run's `metrics.json` under `runs/` into one CSV:

    uv run python -m charge_experiments summarize
