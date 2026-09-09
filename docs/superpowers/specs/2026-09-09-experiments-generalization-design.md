# Design: generalize `charge_experiments` into `experiments`

**Status:** approved, ready for implementation planning
**Date:** 2026-09-09
**Scope:** the experiment harness only. No changes to the `sieve` core
(`src/sieve/**`), and no store rewrites.

## Background

`charge_experiments` was built as one dataset's harness: DASH's published
`dashMoleculesSDF_v2.sdf`, whose per-atom target is the SDF property
`MBIScharge`. That name is a string literal in `data.py` (`MoleculeSet.atom_charge`),
and the concept leaks outward through `Prediction.atom_charge`,
`metrics.charge_conservation_metrics`, `normalize.py`, the parity panels, and
the metric keys written to `metrics.json`.

Nothing about the machinery is actually charge-specific. A run is: read a store
of serialized RDKit `Mol`s carrying a per-atom scalar; fit a node-level
regressor; score per-atom predictions; optionally check that each molecule's
atom values reproduce a per-molecule total. Any per-atom scalar property
fits that shape.

The goal: `experiments` is a dataset-agnostic harness whose config names the
atomic property to train and predict, and **the data-preparation script is the
only DASH-specific component**. Other datasets arrive by writing another
prep script that emits the same store format.

## Decisions

Settled during brainstorming, recorded here so the plan does not relitigate them:

1. **Scalar target only.** The config names exactly one atom property.
   `sieve.io.rdkit_adapter.from_rdkit` takes a single `y_from_atom_prop`
   (y shape `(n, 1)`); a vector target would be a change to the sieve core and
   is out of scope.
2. **Inputs need no generalization.** Model inputs stay sieve's own computed
   attribute vocabulary (`element`, `degree`, `aromatic`, …), already named per
   run in `predictor.params.attribute_levels`. No new attribute source, no
   reading of stored atom properties as attributes.
3. **The molecule-level quantity is optional and config-named.** Today's
   `net_charge` becomes a named parquet column. Configured: sum-constraint
   metrics, the normalizers, and the second parity panel are available.
   Unset: they are absent.
4. **Normalizers generalize; DASH predictors do not.** `std_weighted` /
   `equal_weighted` become dataset-agnostic sum-constraint schemes.
   `dash` / `dash_pretrained` / `tree_artifact.py` stay DASH charge baselines.
5. **Code moves; data does not.** The ~25 GB of `stores/`, `runs/`,
   `mlflow_artifacts/`, `cache/`, `external/` stay physically under
   `charge_experiments/`, reached through symlinks from `experiments/`.
6. **Metric keys are renamed outright:** `charge_conservation/*` →
   `sum_constraint/*`, with no legacy alias in the aggregator.
7. **Two phases.** A mechanical rename, proven by the existing suite passing
   unchanged; then the semantic generalization, developed test-first.

## Layout

```
experiments/
  experiments/         # the package
    config.py  data.py  runner.py  cli.py  metrics.py  normalize.py
    plots.py  aggregate.py  tree_artifact.py
    prepare_dash.py    # DASH-only: download, parse, cluster, split
    store_ops.py       # dataset-agnostic: subsample / partition / united-atom
    predictors/  _chalcedon/
  configs/  tests/  docs/  README.md  pins.toml
  stores        -> ../charge_experiments/stores
  runs          -> ../charge_experiments/runs
  cache         -> ../charge_experiments/cache
  external      -> ../charge_experiments/external
  mlflow_artifacts -> ../charge_experiments/mlflow_artifacts
  results       -> ../charge_experiments/results
  mlflow_runs.db -> ../charge_experiments/mlflow_runs.db
```

Every tracked `.py` / `.yaml` / `.md` under `charge_experiments/` moves via
`git mv`; the directory survives on disk purely as the data holder. Path
constants repoint to `experiments/…` and resolve through the symlinks:
`data.DEFAULT_STORES_ROOT`, `data.DEFAULT_CACHE_DIR`,
`runner.DEFAULT_RUNS_ROOT`, `runner.DEFAULT_ARTIFACT_ROOT`,
`runner._MLFLOW_RUNS_DB`, and `predictors/dash.py`'s `_DASH_TREE_ROOT`.

`.gitignore` gains the `experiments/` symlink names alongside the existing
`charge_experiments/` entries (both paths exist; both must stay ignored).
`pyproject.toml` updates `[tool.setuptools.packages.find] where`, the sdist
exclude, `testpaths`, the coverage `include`, and the per-file ruff ignores
for `_chalcedon`. The CLI entry point becomes `python -m experiments <cmd>`.

Tests move as-is and drop their `charge_` infix (`test_charge_config.py` →
`test_config.py`), except the DASH-specific ones, which keep `dash` in the
name (`test_predictor_dash.py`, `test_predictor_dash_pretrained_optional.py`,
`test_prepare_dash.py`).

### Why `prepare_store.py` splits

The current 884-line `prepare_store.py` mixes two concerns: the DASH-only
pipeline (download `dashMoleculesSDF_v2.sdf`, parse its two record schemas,
Butina-cluster, assign splits, write the parquet) and store operations that
work on any store already in the format (`subsample_store`, `partition_store`,
`to_united_atom_store` — all of which operate on the parquet's own columns).
Splitting them is what makes "the prep script is the only dataset-specific
piece" literally true: a new dataset writes a `prepare_<dataset>.py` and
inherits every store operation.

## The target in config and code

```yaml
run:
  experiment: dash-charges
  seed: 0
data:
  store: dash-molecules
  split_column: split
target:
  atom_property: MBIScharge      # required
  molecule_property: net_charge  # optional; enables the sum constraint
  label: charge (e)              # optional; plot axis text
predictor:
  name: sieve
  params:
    max_wl_depth: 3
```

`TargetCfg` is a frozen dataclass beside `RunCfg` / `DataCfg` / `PredictorCfg`,
added to `_TOP_KEYS`, key-checked like the others, and threaded through
`to_dict` (so it reaches `config.resolved.yaml`) and `to_flat_params` (so it
reaches MLflow). `label` defaults to `atom_property` when omitted.

`target` is required: a config without it raises at load, naming the key. Every
existing config under `experiments/configs/` gains the three lines.

**No store is rewritten.** `molecule_property` names a column the parquet
already has: `dash-molecules` carries `net_charge`, written by the DASH prep
script, and the harness simply reads whichever column the config names.

### Renames

| Today | After |
|---|---|
| `MoleculeSet.atom_charge` | `MoleculeSet.atom_target` |
| `MoleculeSet.net_charge` | `MoleculeSet.molecule_value` (`None` when unconfigured) |
| `Prediction.atom_charge`, `RawPrediction.atom_charge` | `.atom_value` |
| `MoleculeSet.chembl_id` / `.conf_id` / `.dash_id` | `MoleculeSet.ids` (see below) |
| `metrics.charge_conservation_metrics` | `metrics.sum_constraint_metrics` |
| `charge_conservation/{mae,rmse,r2}` | `sum_constraint/{mae,rmse,r2}` |
| `predictions.npz` keys `net_charge`, `atom_charge_true`, `atom_charge_pred` | `molecule_value`, `atom_target_true`, `atom_target_pred` |

`MoleculeSet` gains `atom_property` and `molecule_property` fields, set by the
loader from the config; `atom_target` reads the named property via
`GetDoubleProp`, and `select()` carries both names through. An atom missing the
named property raises with the property name and the conformer id, rather than
surfacing as a `KeyError` from RDKit.

`SievePredictor` passes the configured property to `from_rdkit`'s
`y_from_atom_prop` in place of the literal `"MBIScharge"`.

## Store format and identifiers

The store's own required shape is small, and stays a stable contract for every
future prep script:

- a `mol` column of `mol_to_blob` bytes — one serialized conformer per row,
  its atoms carrying the target property;
- the split column named by `data.split_column` (`split`);
- optionally, the molecule-level column named by `target.molecule_property`;
- any number of further columns, treated as per-conformer identifiers.

`MoleculeSet`'s three hard-wired DASH identifier fields (`chembl_id`,
`conf_id`, `dash_id`) are exactly as dataset-specific as `MBIScharge` and go
the same way: they become one `ids: Mapping[str, list[str | None]]` field
holding every remaining column, populated by the loader and subset by
`select()`. Nothing in the harness groups or clusters by them — they are
provenance carried to the output — so a mapping serves every consumer:
`runner._savez_run` writes one array per key, whatever a dataset's keys are.
DASH stores keep all three, unchanged on disk and unchanged in
`predictions.npz`.

## Sum constraint

`normalize.py`'s two schemes keep their names and their formulas — they only
ever needed a per-atom value, a per-atom std, and a per-molecule total — with
charge vocabulary replaced in signatures and docstrings. The provenance
comments (DASH eq. 4, `get_molecules_partial_charges`'s actual behavior, and
the sign discrepancy against the published formula) stay verbatim: they are
findings about a real upstream implementation, and are still the reason the
code reads as it does.

`metrics.sum_constraint_metrics` compares each molecule's summed predicted atom
values against its `molecule_value`, writing `sum_constraint/{mae,rmse,r2}`.

When `molecule_property` is unset:

- `sum_constraint/*` metrics are not computed,
- the second parity panel (the residual histogram) is not emitted,
- `normalization` is unavailable — a config setting `normalization` without
  `molecule_property` **raises at load**, not mid-run.

`cli.py`'s summary column list moves to `sum_constraint/*`. Existing run
directories on disk hold the old `charge_conservation/*` keys and will show
blanks in those columns; per decision 6 there is no alias.

## DASH predictors and plots

`dash`, `dash_pretrained` and `tree_artifact.py` remain charge-named
internally — they walk DASH's published tree and read its charge columns, and
cannot predict anything else. They change only where they construct a
`Prediction`/`RawPrediction` (field rename) and where they read
`MoleculeSet.molecule_value`. They stay lazily registered in
`predictors/__init__.py`, so a non-charge dataset never imports them.

`plots.py` is already generic (it takes `quantity` / `xlabel` / `title` per
panel). `runner._panel_specs` stops hardcoding `"charge (e)"` and
`"atom charge"`, using `cfg.target.label` instead.

## Testing

**Phase 1** is proven by the existing suite: every test passes unchanged after
the move, import paths aside. Any behavioral difference is a bug in the rename.

**Phase 2** is test-first:

- `test_config.py`: `target` missing entirely; `atom_property` missing;
  unknown key inside `target`; `label` defaulting to `atom_property`;
  `normalization` set without `molecule_property` (raises); round-trip through
  `to_dict` / `to_flat_params`.
- `test_data.py`: a synthetic `MoleculeSet` whose atoms carry a non-charge
  property (e.g. `alpha`) reads correctly through `atom_target`; a missing
  property raises a message naming the property; `select()` preserves both
  property names; `molecule_value` is `None` when unconfigured.
- `test_metrics.py`: `sum_constraint_metrics` against hand-computed numbers.
- `test_smoke.py`: a run with `molecule_property` unset produces no
  `sum_constraint/*` key and a single-panel parity figure; the existing
  charge-configured smoke run still produces both.
- The existing DASH predictor tests are the regression net: charge results must
  be numerically identical to today, differing only in metric key names.

## Documentation

`experiments/README.md` is rewritten around the harness/dataset split, with an
"adding a dataset" section: write a prep script that emits the store parquet
(`mol` blob, split column, optional id columns, optional molecule-level
column), then name the property in `target:`. The DASH specifics move under
that heading rather than framing the whole document.

`experiments/docs/dash_molecules_sdf.md` and the two prior specs
(`2026-08-26-dash-charges-experiment-series-design.md`,
`2026-08-27-dash-charges-nested-runs-design.md`) stay as history, unedited.

## Out of scope

- Vector / multi-property targets (needs a `src/sieve` adapter change).
- Stored atom properties as sieve attributes.
- Any second dataset's prep script. This design makes room for one; writing
  one is separate work.
- Migrating existing run directories' `metrics.json` to the new metric keys.
- `cosmo_experiments`, which is a separate harness on a separate branch.
