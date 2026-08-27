# charge_experiments

A second, independent experiment series: predicting DASH's MBIS atomic
partial charges (the `MBIScharge` SDF property) on DASH's own published
training data (`dashMoleculesSDF_v2.sdf`, ETH Research Collection).

Fully independent of `cosmo_experiments` (sigma-profile prediction) at the
harness level -- no shared package, only the core `sieve` dependency in
common. See
`docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`
for the full design, and `docs/dash_molecules_sdf.md` for findings from
actually running `prepare-store` against the real published SDF (a
download gotcha, a property-serialization bug, and the discovery that the
file holds two distinct record schemas).

## Usage

    uv run python -m charge_experiments prepare-store
    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml
    uv run python -m charge_experiments summarize
