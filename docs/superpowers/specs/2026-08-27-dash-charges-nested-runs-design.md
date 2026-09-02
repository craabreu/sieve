> **Superseded (2026-09-02).** The nested-run machinery this spec describes
> (`nested_runner.py`/`nested_config.py`, the `run-nested` CLI command) has
> been deleted in favor of a single top-level `normalization` config key on
> the flat `run` command -- the only thing the nesting bought (comparing
> normalization schemes without re-fitting) needs no second run level, since
> a flat run's own `predict_raw` already computes the raw walk once. The
> tree-stats persistence problem this spec also solved
> (`save_model_state`/`load_model_state`, `tree_artifact.py`) was dropped
> along with it and not replaced -- see `charge_experiments/README.md`'s
> Usage section for the current mechanism. Kept below for history.

# Nested MLflow runs: persisted tree stats + raw/normalized DASH predictions

## Context

Two full-store runs already exist as flat, single-shot MLflow runs:
`dash` (our own trained predictor: `fit()` matches every train atom against
DASH-tree's published topology and populates a fresh per-node mean charge,
`predict()` walks it, unnormalized) and `dash_pretrained` (zero-training:
walks the tree's own published `"result"`/`"std"` columns, then applies
DASH's own `std_weighted` eq-4 charge-conservation renormalization).

Two problems surfaced running these: (1) `dash`'s `fit()` is the expensive
step (~42 min on the full 823k-conformer train split, dominated by
`match_new_atom` tree-matching, not the numpy aggregation after it) and its
result is never persisted -- every future `predict()` call requires
re-running `fit()` from scratch; (2) `dash`'s `predict()` never conserves
molecule charge (no normalization step at all), while `dash_pretrained`'s
always does (hard-baked into `predict()`), so the two predictors aren't
comparable at the same normalization tier.

The user wants: one run that calls `fit()`, saves the trained per-node
stats, and predicts *raw* (no normalization); further run(s) that load the
saved stats (skipping `fit()` entirely) and predict *normalized*, by one or
more normalization schemes. `dash_pretrained` gets the same raw/normalized
split (it already computes `raw_charge`/`raw_std` internally before
normalizing, no `fit()` involved). MLflow's nested-run feature
(`mlflow.start_run(nested=True)`) is confirmed sufficient for grouping
these: a parent run per predictor (`fit()` + save + raw predict) with one
child run per normalization scheme, computed by re-normalizing the same
already-computed raw predictions -- no re-matching, no re-`fit()`.

## Decisions (settled by conversation)

| Question | Decision |
|---|---|
| What gets persisted | Not the whole `DASHTree` (that's DASH's own ~unchanged published data, already on disk in the pinned clone). Only what `fit()` actually derives: one row per populated `(branch_idx, node_id)` with `mean`, `std`, `count` of the train atoms matched there. A few hundred KB `.npz`, not the multi-GB tree. |
| Where the expense is | `fit()`'s cost is `_atom_paths`' own `match_new_atom` walk over every train atom -- unavoidable once, but never needs repeating just to try a different normalization. The numpy aggregation into per-node stats is comparatively instant. |
| Raw vs. normalized split | Both predictors implementing a new `predict_raw(test) -> RawPrediction(atom_charge, atom_std)` -- the walk only, no normalization. Existing `predict(test) -> Prediction` (the `Predictor` protocol method, used by today's flat single-run configs) stays behavior-identical, implemented as a thin wrapper: `dash.predict` returns `predict_raw(test).atom_charge` unnormalized (matches today's real behavior exactly); `dash_pretrained.predict` returns `std_weighted_normalize(*predict_raw(test), ...)` (matches today's real behavior exactly). No existing flat-config run's output changes. |
| Normalization schemes | `charge_experiments/normalize.py` (new): `std_weighted_normalize` (moved unchanged from `dash_pretrained.py`, same eq-4 sign-corrected implementation) plus a new `equal_weighted_normalize` (residual split equally across a conformer's atoms, ignoring std entirely) -- a second, simpler baseline scheme, useful precisely because it needs no `std` at all. `NORMALIZERS: dict[str, Callable]` registry, both functions sharing one signature `(raw_charge, raw_std, net_charge, mol_id, n_conformers) -> atom_charge` (`equal_weighted_normalize` ignores `raw_std`) so calling code (the nested runner) is normalization-agnostic. |
| `dash`'s missing `std` | `dash`'s own `fit()` only ever computed a per-node **mean**; std-weighted normalization needs a per-node **std** too. `populate_tree_with_charge_property`'s aggregation is extended (two-pass sum/sum-of-squares) to also produce it. This is *our own* predictor's own derived statistic, not standing in for a missing published value, so the "no invented fallback" principle that governs `dash_pretrained` does not apply here -- unlike that predictor, `dash` was never claiming to faithfully reproduce DASH's own inference. |
| Persistence format | `tree_artifact.py` (new): `TreeNodeStats` (parallel numpy arrays: `branch_idx`, `node_id`, `mean`, `std`, `count`), `compute_node_stats(paths, atom_charge) -> TreeNodeStats` (pure numpy, no tree needed -- testable without the real DASH-tree clone), `save_node_stats`/`load_node_stats` (`np.savez`/`np.load` round trip), `apply_node_stats(tree, stats, *, mean_column, std_column)` (writes both columns onto the tree's own `data_storage`, same mechanism `populate_tree_with_charge_property` already uses, returning the `LiteralTreeChargeProperties` pair `predict_via_data_storage_walk` needs). `DASHChargePredictor` gains `save_tree_stats(path)`/`load_tree_stats(path)`, the latter loading the tree (fast -- reads the pinned clone's own `.h5` files, no matching) and applying the saved stats, skipping `fit()`'s expensive matching pass entirely. |
| Orchestration | A new, separate config/CLI/runner path -- `nested_config.py`, `nested_runner.py`, a `run-nested` CLI subcommand -- rather than complicating the existing flat `config.py`/`runner.py`/`run` path, which stays untouched and keeps producing today's exact single-run behavior for `dash`/`dash_pretrained`/`sieve`/`global_mean`. |
| Nested config shape | `run`/`data`/`predictor` sections reuse `config.py`'s existing `RunCfg`/`DataCfg`/`PredictorCfg` dataclasses verbatim (no duplicated schema). New sections: `tree_stats: {save_path: str \| None, load_path: str \| None}` (in practice only ever set for `predictor.name == "dash"` -- `dash_pretrained` has nothing to save, `fit()` is a no-op for it -- but `nested_config.py` doesn't hardcode that predictor name: it's a duck-typed runtime concern, not a parse-time one, so a config naming a predictor without `save_tree_stats`/`load_tree_stats` methods gets a clear error from `execute_nested` itself, not from config loading), `children: list[str]` (normalization scheme names, each validated against `normalize.NORMALIZERS`). |
| Where raw predictions get reused | `execute_nested` calls `predictor.predict_raw(...)` **once per split** (train/val/test) at the parent level, scores+logs those as the parent's own raw metrics, and passes the same three `RawPrediction` objects to every child -- no re-matching per child, only a normalize-and-score pass (numpy, fast). |
| Run-dir/MLflow shape | Every logged run (parent and each child) gets an ordinary `runs/<experiment>/<run_name>__<stamp>__<id>/` directory with the same artifact set flat runs already produce (`metrics.json`, `manifest.json`, `predictions.npz`, `plots/parity_panel.png`, `config.resolved.yaml`) -- no new directory-nesting convention. Nesting is expressed only in MLflow itself: `with mlflow.start_run(run_name=f"{predictor}-raw") as parent:` wraps `with mlflow.start_run(run_name=f"{predictor}-{child}", nested=True):` for each child. A child's `manifest.json` additionally records `"normalization": <name>` and `"tree_stats_source": "fit" \| "loaded"`. |
| Result shape at the two tiers | Concretely: `dash` parent (`fit()`+save+raw) with children `std_weighted`/`equal_weighted`; `dash_pretrained` parent (load published tree, no `fit()`, raw) with the same two children -- six logged runs total, directly comparable pairwise at every tier (raw-vs-raw, std_weighted-vs-std_weighted, equal_weighted-vs-equal_weighted). The nested config's own `children` list is what drives this, not a hardcoded pair, so a third scheme is a one-line config change plus one `normalize.py` function. |

## Architecture

### File layout

```
charge_experiments/charge_experiments/
  normalize.py                 # new: std_weighted_normalize (moved), equal_weighted_normalize, NORMALIZERS
  tree_artifact.py              # new: TreeNodeStats, compute/save/load/apply_node_stats
  nested_config.py               # new: NestedExperimentCfg, TreeStatsCfg, load_nested_config
  nested_runner.py                # new: execute_nested, run_nested, NestedRunResult
  predictors/
    base.py                        # + RawPrediction, NormalizableChargePredictor protocol
    dash.py                        # populate_tree_with_charge_property now std-aware;
                                    # + DASHChargePredictor.predict_raw/save_tree_stats/load_tree_stats
    dash_pretrained.py             # predict_raw extracted; predict() delegates to it +
                                    # normalize.std_weighted_normalize (moved from here)
  cli.py                          # + `run-nested` subcommand
charge_experiments/tests/
  test_charge_normalize.py         # new (renamed from test_charge_predictor_dash_pretrained.py,
                                    # + equal_weighted_normalize cases)
  test_charge_tree_artifact.py      # new
  test_charge_nested_config.py       # new
  test_charge_nested_runner.py        # new (synthetic-data smoke, mirrors test_charge_smoke.py)
  test_charge_predictor_dash.py        # + predict_raw/save+load-round-trip cases
  test_charge_predictor_dash_pretrained.py  # deleted (moved to test_charge_normalize.py);
                                              # + a new predict_raw case lands in
                                              # test_charge_predictor_dash_pretrained_optional.py
```

### `normalize.py`

```python
NORMALIZERS: dict[str, Callable[..., NDArray[np.float64]]]
```
keyed `"std_weighted"` / `"equal_weighted"`, both `(raw_charge, raw_std,
net_charge, mol_id, n_conformers) -> atom_charge`. `equal_weighted_normalize`
spreads each conformer's residual (`net_charge - molecule_sum(raw_charge,
...)`) evenly across its own atom count, ignoring `raw_std` (accepted but
unused, for signature parity); a NaN `raw_charge` on any atom propagates to
the whole conformer via `molecule_sum`, same as `std_weighted_normalize`
already does.

### `tree_artifact.py`

`TreeNodeStats` is the derived-statistics table `dash.py`'s `fit()` needs to
save/reload; `compute_node_stats` replaces `populate_tree_with_charge_property`'s
inline dict-based grouping with a reusable, tree-independent function;
`apply_node_stats` is the write-onto-`tree.data_storage` step, shared by both
"just fit, write in-memory" and "load a saved artifact, write in-memory."

### `nested_config.py` / `nested_runner.py` / `cli.py run-nested`

```yaml
run:
  experiment: dash-charges-nested
  seed: 0
data:
  store: dash-molecules
  split_column: split
predictor:
  name: dash          # or dash_pretrained
  params: {}
tree_stats:            # only for predictor.name == dash; must be absent for dash_pretrained
  save_path: charge_experiments/artifacts/dash-tree-stats.npz
  load_path: null      # if set, fit() is skipped entirely
children:
  - std_weighted
  - equal_weighted
```

`execute_nested` mirrors `runner.execute`'s shape (git-dirty check, run-dir
naming, manifest/metrics/predictions/plots writing) but reuses `runner`'s own
private helpers directly (`runner._score`, `runner._write_plots`,
`runner._savez_run`, `runner._git_info`, `runner._package_versions`,
`runner._score_extra_split`-equivalent logic) rather than duplicating them --
this codebase already has precedent for one module importing another's
underscore-prefixed helpers (`dash_pretrained.py` imports `dash.py`'s
`_atom_paths`), so this is a continuation of an established pattern, not a
new one. `runner._log_mlflow`'s body (tag/param/metric/artifact logging) is
extracted into a reusable `runner._log_mlflow_run(cfg, run_metrics, run_dir,
extra_tags)` that assumes an MLflow run is already open -- `runner._log_mlflow`
becomes a thin wrapper opening that run itself (flat-config path, unchanged
behavior), while `nested_runner.py` opens its own parent/`nested=True`
contexts and calls `_log_mlflow_run` inside each.

## Out of scope (this spec)

- A third normalization scheme beyond `std_weighted`/`equal_weighted` (the
  registry is built to make adding one a small, isolated change, but none is
  specified here).
- Persisting/reusing a *Sieve* predictor's trained state (this spec is
  scoped to the two DASH-tree predictors; `sieve_predictor.py` is untouched).
- Any change to the existing flat `run`/`config.py`/`runner.py` path's
  observable behavior -- every existing config keeps producing bit-identical
  output.
- A smaller, laptop-friendly pre-built store (a separate, previously
  discussed idea, not part of this feature).
