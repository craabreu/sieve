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

A predictor whose `fit()` is expensive to redo (has `save_model_state`,
e.g. `dash`, `sieve`) has its fitted state written to that run's own
`tree_stats.npz` automatically -- no config flag needed. A later run can
skip `fit()` entirely and load it back via `tree_stats_load_path` (no
re-save of its own -- the loaded path is already the provenance record):

    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml --set tree_stats_load_path=charge_experiments/runs/dash-charges/<earlier-run-dir>/tree_stats.npz

Point a run at a different store with `--set data.store=dash-molecules-50k`,
or use `--limit N` for a quick sanity check against whatever store is
configured.

### Collecting results

Gather every run's `metrics.json` under `runs/` into one CSV:

    uv run python -m charge_experiments summarize
