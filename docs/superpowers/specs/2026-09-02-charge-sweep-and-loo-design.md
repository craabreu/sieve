# Design: LOO training metric, and a `sweep` command for parameter curves

**Status:** approved, ready for implementation planning
**Date:** 2026-09-02
**Scope:** two independent features in `charge_experiments`, shipped together because
they serve one workflow — sweeping a run parameter and reading the resulting curve.
No changes to the `sieve` core.

## Background

The question that started this was "can we store metrics after every sieve level, for
training and prediction?" — i.e. what would accuracy be if the model were only allowed to
refine down to level *k*.

A design was drafted for computing that inside a single run: a `max_level` cap on
`sieve.predict`'s backoff search, a per-level loop in `SievePredictor`, a
`metrics_by_level.json` artifact, and per-level re-normalization for nested children.

**That design was dropped.** Running the sweep as *N separate experiments* and gathering
the results afterwards is exactly equivalent, and needs no code at all.

### Why the sweep is equivalent — verified, not assumed

Level *k*'s partition and fitted statistics do not depend on how many levels the model
was fit with:

- `refine()` computes level *k* from level *k−1* and the graph. `max_wl_depth` never
  enters.
- Consequently `sieve.fit`'s per-level class signatures, means and counts at levels
  `0..k` are byte-identical between a depth-*k* model and a depth-*N* model, `N > k`.
- Shrinkage is **depth-local**: `shrunk_means(model)[k]` blends level *k* toward level
  *k−1* only, so it too matches across depths.

All three were checked directly on a real corpus, with `shrinkage_strength` both unset
and set. A run at `max_wl_depth=k` therefore *is* the capped-at-*k* prediction.

### Why the sweep is also cheap enough

A real run from `runs/dash-charges/` — 40,000 train conformers, 1.69 M train atoms —
records `time/fit_s` 24.4 s, `time/train_predict_s` 16.9 s, `time/predict_s` 1.6 s:
about 45 s end to end. A four-point depth sweep is ~3 minutes.

The sweep does re-featurize at every point (`build_codes` and `from_rdkit` never read
`max_wl_depth`, so the batch is byte-identical each time), and avoiding that redundancy
was the *only* thing the dropped design bought. It is not worth a new public API in the
core, a per-level artifact format, and per-level normalization in the nested runner.

The sweep also gives something back: each point is a complete run — its own
`metrics.json`, `predictions.npz`, parity plots, manifest, MLflow run — rather than a row
in a JSON array.

### What the sweep genuinely cannot give

**Leave-one-out.** The runner computes no LOO metric at any depth, so no amount of
sweeping produces one. That is orthogonal to the per-level question, valuable at every
depth on its own, and much smaller than the dropped design. It is Part A below.

## Part A — LOO training metric

### Motivation

`runner._score_extra_split` scores the train split with plain `predictor.predict`, so
every atom's class mean still contains that atom's own target. On the run inspected
above that shows as `train/mae` 0.0089 against test 0.0171 — optimistic by ~2×.

`sieve.predict_loo` subtracts a node's own contribution before the support check, and
treats a class with one member as unsupported rather than dividing by zero. At
`minimum_support: 1` — which both example configs set — a singleton class gives
`eff_n = 0`, which fails the support check, so LOO backs off instead of recalling the
node. **The gap between `train` and `train_loo` is the memorization signal**, and it is
the reason to report both rather than replacing one with the other.

### Core `sieve`

No change. `sieve.predict_loo(model, batch)` already exists.

### `SievePredictor`

One new method:

```python
def predict_loo_raw(self, train: MoleculeSet) -> RawPrediction
```

Featurizes with `with_target=True` (LOO reads `batch.y`) and calls `sieve.predict_loo`,
returning `atom_charge` and `atom_std` exactly as `predict_raw` does.

**The train-only constraint cannot be enforced structurally.** LOO computes
`(cnt·mean − y_node) / (cnt − 1)`. For a val or test node, that subtracts a value which
was never in the class mean, so the result is corrupt rather than merely uninformative.
But every `MoleculeSet` in this series carries `MBIScharge` on its `Mol`s, so a val set
would satisfy `predict_loo`'s only guard (`batch.y is not None`) and return quietly wrong
numbers. Mitigation is naming and documentation: the parameter is named `train`, the
docstring states the constraint, and the runner calls it for the train split only.

### `runner`

Scores the result as `train_loo/*` alongside the existing `train/*`, detected via
`hasattr(predictor, "predict_loo_raw")` — the same pattern `match_stats` already uses, so
non-sieve predictors are unaffected without an `isinstance` check.

### Gate

`predictor.params.report_loo`, default `false`.

It costs a second featurization of the train split (~17 s of that 45 s run, ≈ +38%), and
a default that silently changes the shape of `metrics.json` mid-series would break
comparability with runs already recorded. Riding in `predictor.params` also means
`to_flat_params` logs it to MLflow as provenance for free.

Sharing one `with_target=True` train batch between the in-sample and LOO passes would
remove that cost — deliberately not done now, as it means caching a full train batch.

## Part B — the `sweep` command

### Relationship to `summarize`

`cli.py` already has `summarize`: it walks `runs/*/*/manifest.json`, reads each
`metrics.json`, and writes a fixed-column CSV to `results/summary.csv`. It takes no
arguments, has a hardcoded `SUMMARY_COLUMNS`, reads only fixed manifest fields, and does
no plotting.

`summarize` is **left exactly as it is** — it is an inventory tool. `sweep` is a
different job: one curve you asked for. They share an extracted run-reading helper.

Usefully, `manifest["config"]` is the full resolved nested config, so an arbitrary dotted
x-axis path is available from the run directory without opening
`config.resolved.yaml`.

### New module: `charge_experiments/aggregate.py`

```python
@dataclass(frozen=True)
class RunRow:
    """One run, with both sources normalized to the same shape."""
    run_dir: str
    params: dict[str, str]     # flat dotted, e.g. "predictor.params.max_wl_depth"
    metrics: dict[str, float]

def read_runs_from_dirs(runs_root, experiment=None) -> list[RunRow]
def read_runs_from_mlflow(tracking_uri, experiment) -> list[RunRow]
def build_curve(rows, *, x, metrics, group_by=None) -> CurveTable
```

`CurveTable` carries both what is plotted and what is written: the aggregated points
keyed by `(series, metric)` — each a sorted list of `(x, mean, lo, hi, n_runs)` — and the
flat list of raw per-run rows the CSV is written from.

### Three normalizations, or the two sources disagree

1. **MLflow metric names.** `_log_mlflow_run` prefixes *everything* with `test/`, so
   run-dir `mae` is MLflow `test/mae` and run-dir `train/mae` is MLflow
   `test/train/mae`. The MLflow reader strips one leading `test/` to recover the run-dir
   spelling. Parameters need no fixing — `to_flat_params` uses the same dotted spelling
   MLflow stores.
2. **Split → key prefix.** Test metrics are *unprefixed* (`mae`, `rmse`); the others are
   (`train/mae`, `val/mae`, `train_loo/mae`). Series selection maps a split name to a key
   prefix, with `test` mapping to the empty prefix.
3. **Missing metric.** A run lacking the requested metric is dropped from that point.
   Never coerced to `0`, which would read as a real and excellent value.

### Aggregation

Points sharing an x value (a sweep repeated over seeds, say) are aggregated to a **mean
with a min–max band**. The raw per-run rows go to the CSV, so the band is never the only
record of what was measured.

Numeric x values are plotted numerically. A non-numeric x falls back to sorted-unique
categorical positions.

### `plots.py`

Gains `curve_panel` beside `parity_panel`: one subplot per metric, one line per series,
mean line plus min–max band. Same lazy-`matplotlib` import and `ImportError` tolerance
`parity_panel` already has, so a run without matplotlib still gets the CSV.

### CLI

```
charge-exp sweep --x predictor.params.max_wl_depth
                 [--experiment NAME] [--source runs|mlflow]
                 [--metric mae --metric rmse --metric r2]   # default: these three
                 [--split test --split train --split val]   # default series
                 [--group-by DOTTED_PATH]                   # optional second dimension
                 [--out results/NAME]
```

Writes `results/<name>/curve.png` and `results/<name>/curve.csv`. When `--out` is
omitted, `<name>` is derived from the x-axis path's last segment (so
`predictor.params.max_wl_depth` writes to `results/max_wl_depth/`), which keeps repeated
sweeps over the same parameter overwriting one place rather than accumulating
timestamped directories.

### The workflow this enables

```bash
for d in 0 1 2 3; do
  charge-exp run --config configs/sieve-charge-example.yaml \
                 --set predictor.params.max_wl_depth=$d \
                 --set predictor.params.report_loo=true
done
charge-exp sweep --x predictor.params.max_wl_depth \
                 --split test --split train --split train_loo
```

which is the original request — accuracy as a function of refinement depth, for training
and prediction — with the train/`train_loo` gap showing where refinement stops
generalizing and starts memorizing.

## Testing

**Part A**
- `train_loo/*` keys present iff `report_loo` is on; `metrics.json` byte-identical when off.
- `train_loo/mae > train/mae` on a corpus with singleton classes — the memorization gap is
  measured, not asserted by construction.
- A predictor without `predict_loo_raw` (e.g. the DASH baselines) is unaffected.

**Part B**
- Run-dir reader against a synthetic `tmp_path` tree of fake `manifest.json`/`metrics.json`
  — never touches the real store.
- MLflow reader's `test/` stripping, in a `*_optional.py` test, per this repo's existing
  convention for optional dependencies.
- **Both sources yield identical `RunRow`s for the same run.** This is the test that keeps
  the normalizations above honest; without it the two `--source` values drift apart
  silently.
- Aggregation math over repeated x (mean, min, max).
- CSV contains raw per-run rows, not aggregates.
- A run missing a requested metric is dropped, not zero-filled.
- `curve_panel` guarded like `parity_panel` (no matplotlib ⇒ CSV still written).

## Out of scope

- Per-level metrics computed inside a single run, and the `max_level` search cap that
  would require — superseded by the sweep, as argued above.
- Making LOO the headline `train/*` metric in `metrics.json`.
- Any change to `summarize`.
- Fixing the `test/` MLflow prefix quirk. It is worked around in the reader and left
  alone at the source, since changing it would break comparison with every run already
  logged.
- Sharing one train batch between the in-sample and LOO passes.
