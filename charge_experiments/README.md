# charge_experiments

A second, independent experiment series: predicting DASH's MBIS atomic
partial charges (the `MBIScharge` SDF property) on DASH's own published
training data (`dashMoleculesSDF_v2.sdf`, ETH Research Collection).

Fully independent of `cosmo_experiments` (sigma-profile prediction) at the
harness level -- no shared package, only the core `sieve` dependency in
common. See
`docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`
for the full design, `docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md`
for the nested-run (raw + per-normalization-scheme) orchestration, and
`docs/dash_molecules_sdf.md` for findings from actually running
`prepare-store` against the real published SDF (a download gotcha, a
property-serialization bug, and the discovery that the file holds two
distinct record schemas).

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

A flat, single-predictor run:

    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml

A nested run -- one predictor's `fit()`+save+raw-predict as an MLflow
parent, plus one child run per normalization scheme (`std_weighted`/
`equal_weighted`), reusing the same raw predictions with no re-fit:

    uv run python -m charge_experiments run-nested --config configs/dash-nested-charge-example.yaml

Point either at a different store with `--set data.store=dash-molecules-50k`,
or use `--limit N` for a quick sanity check against whatever store is
configured.

### Collecting results

Gather every run's `metrics.json` under `runs/` into one CSV:

    uv run python -m charge_experiments summarize
