# LOO Metric and `sweep` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in leave-one-out training metric to the sieve charge predictor, and a `sweep` CLI command that gathers many runs and plots any metric against any config parameter.

**Architecture:** Two independent features in `charge_experiments`, no change to the `sieve` core. Part A wires the existing `sieve.predict_loo` through `SievePredictor` and scores it as `train_loo/*`. Part B adds `aggregate.py` (two run readers normalized to one `RunRow` shape, plus curve aggregation), a `curve_panel` plot, and a `sweep` subcommand.

**Tech Stack:** Python 3.11+, NumPy, RDKit, matplotlib (lazy/optional), MLflow (optional), pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md`

## Global Constraints

- **All four CI checks must pass before every commit:** `ruff check src tests charge_experiments`, `ruff format --check src tests charge_experiments`, `ty check src tests charge_experiments`, and `pytest`. `ty` is easy to forget and has failed CI on this repo before.
- Prefix commands with `export PATH="/home/craabreu/miniforge3/bin:$PATH"` — `uv` is not on the default PATH here. Run everything through `uv run`.
- Baselines at the start of this plan: **full suite 327 passed, 1 skipped**; `charge_experiments/tests` alone **118 passed** (~110 s).
- Line length 88 (ruff default). `from __future__ import annotations` at the top of every module touched.
- **Never touch `src/sieve/`.** This plan needs no core change; `sieve.predict_loo` already exists. If you find yourself editing the core, stop and re-read the spec.
- **`_cmd_summarize` keeps its contract.** Task 2 has it call the extracted reader; its output path, columns, formatting, sort order, zero-argument interface and printed line are unchanged. The one intended difference — a non-finite metric rendering as `""` instead of the literal `nan` — is documented and tested in Task 2, Steps 5–6. Any *other* difference is a bug.
- matplotlib and mlflow are **optional** dependencies. Guard imports the way `plots.parity_panel` and `runner._log_mlflow` already do (lazy import, catch `ImportError`, log a warning, keep going).
- Tests must never touch the real store. Use `charge_experiments/tests/helpers.synthetic_molecule_set` or a synthetic `tmp_path` run tree.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `charge_experiments/charge_experiments/predictors/sieve_predictor.py` | `predict_loo_raw`, `report_loo` flag | 1 |
| `charge_experiments/charge_experiments/runner.py` | score `train_loo/*` | 1 |
| `charge_experiments/charge_experiments/aggregate.py` | **new** — `RunRow`, both readers, `build_curve`, `CurveTable` | 2, 3, 4 |
| `charge_experiments/charge_experiments/plots.py` | `curve_panel` | 5 |
| `charge_experiments/charge_experiments/cli.py` | `sweep` subcommand; `_cmd_summarize` calls the extracted reader | 2, 6 |
| `charge_experiments/tests/test_charge_predictor_sieve.py` | LOO tests | 1 |
| `charge_experiments/tests/test_charge_aggregate.py` | **new** — reader/aggregation tests | 2, 3, 4 |
| `charge_experiments/tests/test_charge_aggregate_optional.py` | **new** — mlflow-dependent tests | 3 |
| `charge_experiments/tests/test_charge_plots.py` | `curve_panel` test | 5 |
| `charge_experiments/tests/test_charge_cli.py` | `sweep` end-to-end | 6 |

---

## Task 1: LOO training metric

**Files:**
- Modify: `charge_experiments/charge_experiments/predictors/sieve_predictor.py`
- Modify: `charge_experiments/charge_experiments/runner.py:305-345` (`_execute_inner`)
- Test: `charge_experiments/tests/test_charge_predictor_sieve.py`

**Interfaces:**
- Consumes: `sieve.predict_loo(model, batch)` — already exists, requires `batch.y`.
- Produces:
  - `SievePredictor.report_loo: bool` (constructor kwarg, default `False`)
  - `SievePredictor.predict_loo_raw(self, train: MoleculeSet) -> RawPrediction`
  - Metric keys `train_loo/mae`, `train_loo/rmse`, `train_loo/r2`, `train_loo/n_nan`, `train_loo/n_test_atoms`, `train_loo/n_test_conformers`, `train_loo/charge_conservation/*` — the full `_score` key set, prefixed.

- [ ] **Step 1: Write the failing tests**

Add to `charge_experiments/tests/test_charge_predictor_sieve.py`:

```python
def test_predict_loo_raw_backs_off_instead_of_recalling_the_node():
    """Leave-one-out removes a node's own contribution from its class mean
    before the support check, so at minimum_support=1 a singleton class has
    eff_n == 0 and fails it -- the node backs off to its parent instead of
    recalling itself. In-sample prediction has no such guard, so its error
    is optimistically low. The gap is the memorization signal."""
    import numpy as np

    from charge_experiments.predictors.sieve_predictor import SievePredictor
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=0)
    p = SievePredictor(max_wl_depth=3, minimum_support=1, report_loo=True)
    p.fit(mset, mset, rng=np.random.default_rng(0))

    in_sample = p.predict_raw(mset).atom_charge
    loo = p.predict_loo_raw(mset).atom_charge

    in_sample_mae = float(np.nanmean(np.abs(in_sample - mset.atom_charge)))
    loo_mae = float(np.nanmean(np.abs(loo - mset.atom_charge)))
    assert loo_mae > in_sample_mae


def test_predict_loo_raw_requires_a_fitted_model():
    import numpy as np
    import pytest

    from charge_experiments.predictors.sieve_predictor import SievePredictor
    from charge_experiments.tests.helpers import synthetic_molecule_set

    del np
    p = SievePredictor()
    with pytest.raises(RuntimeError, match="fit"):
        p.predict_loo_raw(synthetic_molecule_set(n_mol=2))


def test_report_loo_defaults_off_and_is_recorded_on_the_predictor():
    from charge_experiments.predictors.sieve_predictor import SievePredictor

    assert SievePredictor().report_loo is False
    assert SievePredictor(report_loo=True).report_loo is True
```

- [ ] **Step 2: Run them to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_predictor_sieve.py -k "loo" -v
```

Expected: FAIL — `SievePredictor.__init__() got an unexpected keyword argument 'report_loo'`.

- [ ] **Step 3: Add the flag and the method**

In `SievePredictor.__init__`, add the keyword after `n_jobs` and store it:

```python
        report_loo: bool = False,
```
```python
        self.report_loo = report_loo
```

Add the method after `predict_raw`:

```python
    def predict_loo_raw(self, train: MoleculeSet) -> RawPrediction:
        """Leave-one-out prediction for the *training* split (design.md 10.3).

        A training node contributes its own target to its class mean, so any
        in-sample score is optimistic -- at minimum_support=1 and large depth
        it approaches perfect recall. ``sieve.predict_loo`` subtracts that
        contribution before the support check and treats a class with one
        member as unsupported, so the node backs off to its parent instead of
        recalling itself.

        **Train-only, and not structurally enforceable.** LOO computes
        ``(cnt*mean - y_node) / (cnt - 1)``; for a val or test node that
        subtracts a value which was never in the class mean, so the result is
        corrupt rather than merely uninformative. Every MoleculeSet in this
        series carries MBIScharge on its Mols, so a val set would satisfy
        ``predict_loo``'s only guard (``batch.y is not None``) and return
        quietly wrong numbers. The parameter is named ``train`` and
        ``runner`` calls this for the train split only.
        """
        if self._model is None or self._config is None:
            raise RuntimeError(
                "fit (or load_model_state) must be called before predict_loo_raw"
            )
        import time

        import sieve

        t0 = time.perf_counter()
        batch = _batch_for(
            train.mols, self._config, with_target=True, n_jobs=self.n_jobs
        )
        self.last_featurize_s += time.perf_counter() - t0
        detailed = sieve.predict_loo(self._model, batch)
        atom_charge = np.asarray(detailed.value, dtype=np.float64)[:, 0]
        atom_std = np.sqrt(np.asarray(detailed.variance, dtype=np.float64)[:, 0])
        return RawPrediction(atom_charge=atom_charge, atom_std=atom_std)
```

Note `with_target=True` — unlike `predict_raw`, LOO reads `batch.y`.

Add to the class docstring, after the `n_jobs` paragraph:

```
    ``report_loo`` opts into a leave-one-out pass over the training split
    (``predict_loo_raw``), which ``runner`` scores as ``train_loo/*``. Off by
    default: it costs a second featurization of train (~38% on a real run),
    and a default that changed the shape of metrics.json mid-series would
    break comparability with runs already recorded.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_predictor_sieve.py -k "loo" -v
```

Expected: PASS.

- [ ] **Step 5: Write the failing runner test**

Add to `charge_experiments/tests/test_charge_smoke.py`, following
`test_smoke_reports_featurize_time_for_sieve_predictor`'s own shape (`_tiny_cfg`,
`_synthetic_masks`, `synthetic_molecule_set`, `execute(..., tracking=None)` — all
already imported in that module):

```python
def _sieve_cfg(**params):
    cfg = _tiny_cfg()
    return cfg.__class__(
        run=cfg.run,
        data=cfg.data,
        predictor=PredictorCfg(name="sieve", params={"max_wl_depth": 2, **params}),
    )


def test_report_loo_adds_train_loo_metrics(tmp_path):
    """train_loo/* appears only when the predictor opts in. Off is the
    default precisely so an existing run's key set does not shift under
    anyone, so the negative case is asserted too."""
    import pytest

    pytest.importorskip("rdkit")

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)

    on = execute(
        _sieve_cfg(report_loo=True), mset, masks,
        runs_root=tmp_path / "on", allow_dirty=True, tracking=None,
    )
    assert "train_loo/mae" in on.metrics
    assert "train_loo/r2" in on.metrics
    assert "time/train_loo_predict_s" in on.metrics

    off = execute(
        _sieve_cfg(), mset, masks,
        runs_root=tmp_path / "off", allow_dirty=True, tracking=None,
    )
    assert not any(k.startswith("train_loo/") for k in off.metrics)
    assert "time/train_loo_predict_s" not in off.metrics


def test_report_loo_is_ignored_by_predictors_without_it(tmp_path):
    """runner reads report_loo/predict_loo_raw via getattr/hasattr, so a
    predictor that has neither is unaffected rather than erroring."""
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    result = execute(
        _tiny_cfg(), mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert not any(k.startswith("train_loo/") for k in result.metrics)
```

- [ ] **Step 6: Run it to verify it fails**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/ -k "report_loo" -v
```

Expected: FAIL — no `train_loo/` keys.

- [ ] **Step 7: Score it in the runner**

In `runner.py`, add beside `_score_extra_split`:

```python
def _score_loo(predictor: Any, train: MoleculeSet) -> dict[str, float]:
    """Leave-one-out scoring of the *training* split, when the predictor
    opts in via ``report_loo``. Train-only by construction: LOO subtracts a
    node's own contribution from its class mean, which for a val or test
    node was never there (see SievePredictor.predict_loo_raw)."""
    if not getattr(predictor, "report_loo", False):
        return {}
    if not hasattr(predictor, "predict_loo_raw"):
        return {}
    if train.n_conformers == 0:
        return {}
    raw = predictor.predict_loo_raw(train)
    score = _score(train, Prediction(atom_charge=raw.atom_charge))
    return {f"train_loo/{k}": v for k, v in score.items()}
```

In `_execute_inner`, after the `val_metrics` block and before `pred = predictor.predict(test)`:

```python
    t0 = time.perf_counter()
    loo_metrics = _score_loo(predictor, train)
    loo_predict_s = time.perf_counter() - t0
```

and after `run_metrics.update(val_metrics)`:

```python
    run_metrics.update(loo_metrics)
    if loo_metrics:
        run_metrics["time/train_loo_predict_s"] = loo_predict_s
```

- [ ] **Step 8: Run the runner test, then the whole charge suite**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/ -k "report_loo" -v
uv run pytest charge_experiments/tests -q
```

Expected: the new tests PASS; charge suite **123 passed** (118 + 5).

- [ ] **Step 9: Lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(charge): opt-in leave-one-out training metric

SievePredictor.predict_loo_raw wires sieve.predict_loo (which already
exists) through the predictor; runner scores it as train_loo/*. At
minimum_support=1 a singleton class has eff_n == 0 under LOO and fails the
support check, so the node backs off instead of recalling itself -- the gap
between train/ and train_loo/ is the memorization signal.

Off by default (predictor.params.report_loo): it costs a second
featurization of train, and a default that changed metrics.json's key set
mid-series would break comparability with runs already recorded.

Train-only, and not structurally enforceable -- every MoleculeSet carries
MBIScharge, so a val set would pass predict_loo's only guard and return
quietly wrong numbers. Documented at the method, and the runner calls it
for train alone."
```

---

## Task 2: `aggregate.py` — `RunRow` and the run-directory reader

**Files:**
- Create: `charge_experiments/charge_experiments/aggregate.py`
- Modify: `charge_experiments/charge_experiments/cli.py:112-153` (`_cmd_summarize` calls the extracted reader)
- Test: `charge_experiments/tests/test_charge_aggregate.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `RunRow` frozen dataclass — `run_dir: str`, `params: dict[str, str]`, `metrics: dict[str, float]`, `meta: dict[str, str]`
  - `flatten_params(config: Mapping) -> dict[str, str]` — dotted keys matching `config.to_flat_params`' spelling
  - `read_runs_from_dirs(runs_root: Path, experiment: str | None = None) -> list[RunRow]`

`meta` carries the manifest fields that live *outside* `config` (`run_name`, `split_column`, `seed`, `git_commit`) — `_cmd_summarize` needs them, and putting them on `RunRow` is what lets it drop its own file-reading loop in Step 5 rather than reading each manifest twice.

- [ ] **Step 1: Write the failing test**

Create `charge_experiments/tests/test_charge_aggregate.py`:

```python
"""Reader/aggregation tests for the sweep command. Never touches the real
store or the real runs tree -- every fixture is a synthetic tmp_path run."""

from __future__ import annotations

import json
from pathlib import Path


def write_run(root: Path, experiment: str, name: str, *, params, metrics) -> Path:
    """A minimal run directory: manifest.json (carrying the resolved config
    that params are read from) plus metrics.json."""
    run_dir = root / experiment / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": params}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def test_read_runs_from_dirs_flattens_config_to_dotted_params(tmp_path):
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(
        tmp_path,
        "exp",
        "r1",
        params={"predictor": {"name": "sieve", "params": {"max_wl_depth": 3}}},
        metrics={"mae": 0.017, "train/mae": 0.009},
    )
    rows = read_runs_from_dirs(tmp_path)
    assert len(rows) == 1
    assert rows[0].params["predictor.params.max_wl_depth"] == "3"
    assert rows[0].params["predictor.name"] == "sieve"
    assert rows[0].metrics["mae"] == 0.017
    assert rows[0].metrics["train/mae"] == 0.009


def test_read_runs_from_dirs_filters_by_experiment(tmp_path):
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(tmp_path, "a", "r1", params={}, metrics={"mae": 1.0})
    write_run(tmp_path, "b", "r1", params={}, metrics={"mae": 2.0})
    assert len(read_runs_from_dirs(tmp_path)) == 2
    assert [r.metrics["mae"] for r in read_runs_from_dirs(tmp_path, "a")] == [1.0]


def test_read_runs_from_dirs_skips_a_run_with_no_metrics(tmp_path):
    """A run that crashed before writing metrics.json is skipped, not read as
    an empty row that would later be plotted as a gap at zero."""
    from charge_experiments.aggregate import read_runs_from_dirs

    run_dir = tmp_path / "exp" / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {}}))
    assert read_runs_from_dirs(tmp_path) == []


def test_nan_metrics_are_dropped_not_read_as_values(tmp_path):
    """metrics.json stores NaN for an undefined r2 (json.dumps writes bare
    NaN, which json.loads accepts). A NaN must not survive into a plotted
    point."""
    from charge_experiments.aggregate import read_runs_from_dirs

    write_run(tmp_path, "exp", "r1", params={}, metrics={"mae": 0.1, "r2": float("nan")})
    row = read_runs_from_dirs(tmp_path)[0]
    assert row.metrics["mae"] == 0.1
    assert "r2" not in row.metrics
```

- [ ] **Step 2: Run to verify it fails**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'charge_experiments.aggregate'`.

- [ ] **Step 3: Write the module**

Create `charge_experiments/charge_experiments/aggregate.py`:

```python
"""Gather many runs into one table, for the ``sweep`` command.

Two sources -- run directories and MLflow -- normalized to the same
``RunRow`` shape, so ``--source`` never changes what a curve means. See
docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRow:
    """One run. ``params`` are flat dotted keys spelled exactly as
    ``config.to_flat_params`` spells them (and therefore as MLflow stores
    them); ``metrics`` are spelled as ``metrics.json`` spells them, which is
    what the MLflow reader normalizes back to."""

    run_dir: str
    params: dict[str, str]
    metrics: dict[str, float]
    meta: dict[str, str]
    """Manifest fields outside ``config`` -- run_name, split_column, seed,
    git_commit. ``_cmd_summarize`` needs them, and carrying them here is what
    lets it share this reader instead of re-opening every manifest."""


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested resolved config -> dotted string params, matching
    ``config.to_flat_params``' own spelling."""
    out: dict[str, str] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_params(value, path))
        else:
            out[path] = str(value)
    return out


def _finite_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    """Only real numbers survive. metrics.json stores NaN for an undefined
    r2 and json.loads accepts it; a NaN reaching a plot becomes a silent gap
    or an axis blow-up, so drop it here rather than downstream."""
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            out[key] = float(value)
    return out


def read_runs_from_dirs(
    runs_root: Path, experiment: str | None = None
) -> list[RunRow]:
    """Walk ``runs_root/<experiment>/<run>/`` for manifest+metrics pairs.

    A run with no ``metrics.json`` (crashed before writing one) is skipped
    rather than returned as an empty row, which would later plot as a point
    at zero.
    """
    pattern = f"{experiment}/*/manifest.json" if experiment else "*/*/manifest.json"
    rows: list[RunRow] = []
    for manifest_path in sorted(Path(runs_root).glob(pattern)):
        run_dir = manifest_path.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        rows.append(
            RunRow(
                run_dir=str(run_dir),
                params=flatten_params(manifest.get("config", {})),
                metrics=_finite_metrics(json.loads(metrics_path.read_text())),
                meta={
                    "run_name": str(manifest.get("run_name", "")),
                    "split_column": str(
                        manifest.get("data", {}).get("split_column", "")
                    ),
                    "seed": str(manifest.get("seed", "")),
                    "git_commit": str(manifest.get("git", {}).get("commit", "")),
                },
            )
        )
    return rows
```

The tests in Step 1 construct `RunRow`s by keyword, so they are unaffected by
`meta`; the `read_runs_from_mlflow` reader in Task 3 fills it with `""` values
except `run_name`, which MLflow carries as `tags.mlflow.runName`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Point `_cmd_summarize` at the shared reader**

`_cmd_summarize` currently inlines its own walk. Replace the loop body's manifest/metrics reading with `read_runs_from_dirs`, keeping **every output detail identical**: the same `SUMMARY_COLUMNS`, the same `%.6g` float formatting, the same `""` for a missing value, the same sort key, the same `results/summary.csv` path, the same printed line.

`_cmd_summarize`'s non-metric columns come from `RunRow.meta` (already populated in
Step 3), except `predictor`, which is `params["predictor.name"]`, and `run_dir`, which
is `row.run_dir`. The loop body becomes:

```python
    rows = []
    for run in read_runs_from_dirs(runs_root):
        row = {
            "run_name": run.meta["run_name"],
            "predictor": run.params.get("predictor.name", ""),
            "split_column": run.meta["split_column"],
            "seed": run.meta["seed"],
            "git_commit": run.meta["git_commit"],
            "run_dir": run.run_dir,
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                value = run.metrics.get(key, "")
                row[key] = f"{value:.6g}" if isinstance(value, float) else value
        rows.append(row)
```

⚠️ **One behavior difference to be deliberate about.** The old loop read raw
`metrics.json` values, so a NaN `r2` reached `f"{value:.6g}"` and was written as `nan`.
`read_runs_from_dirs` drops non-finite metrics, so that cell now comes out `""`. Both
are defensible; `""` matches how a genuinely missing metric was already rendered. Accept
it, and expect the Step 6 diff to show exactly that on `charge_conservation/r2` — which
is `NaN` in the run I inspected. If the diff shows anything *else*, the refactor changed
something it shouldn't have.

- [ ] **Step 6: Diff `summarize`'s output against pre-refactor**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run python -m charge_experiments summarize
cp charge_experiments/results/summary.csv /tmp/after.csv
git stash && uv run python -m charge_experiments summarize \
  && cp charge_experiments/results/summary.csv /tmp/before.csv && git stash pop
diff /tmp/before.csv /tmp/after.csv
```

Expected: **the only differences are `nan` cells becoming empty**, per the note above.
Same row count, same row order, same column order, every finite number identical. Any
other difference means the refactor changed something it shouldn't have — fix it rather
than accepting the diff.

Add a regression test for the one intended change so it is recorded rather than folklore:

```python
def test_summarize_renders_a_non_finite_metric_as_empty(tmp_path, monkeypatch):
    """metrics.json stores NaN for an undefined r2. The shared reader drops
    non-finite values, so the cell is empty rather than the literal "nan"
    the pre-2026-09 inline loop wrote."""
    import csv
    import json

    from charge_experiments import cli

    run_dir = tmp_path / "runs" / "exp" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {}}))
    (run_dir / "metrics.json").write_text(
        json.dumps({"mae": 0.1, "charge_conservation/r2": float("nan")})
    )
    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", tmp_path / "runs")
    assert cli.main(["summarize"]) == 0

    row = next(iter(csv.DictReader((tmp_path / "results" / "summary.csv").open())))
    assert row["mae"] == "0.1"
    assert row["charge_conservation/r2"] == ""
```

- [ ] **Step 7: Full charge suite, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: **128 passed** (123 + 5).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(charge): aggregate.RunRow and the run-directory reader

One normalized shape for a gathered run: dotted params spelled as
to_flat_params spells them, metrics spelled as metrics.json spells them.
NaN metrics (an undefined r2, which json.loads happily returns) are dropped
at read time rather than reaching a plot as a gap or an axis blow-up, and a
run that crashed before writing metrics.json is skipped rather than
returned as an empty row.

_cmd_summarize now calls the same reader; its columns, formatting, sort
order, output path and printed line are unchanged (verified by diffing the
generated summary.csv against the pre-refactor one)."
```

---

## Task 3: the MLflow reader

**Files:**
- Modify: `charge_experiments/charge_experiments/aggregate.py`
- Test: `charge_experiments/tests/test_charge_aggregate.py`, `charge_experiments/tests/test_charge_aggregate_optional.py` (create)

**Interfaces:**
- Consumes: `RunRow` (Task 2).
- Produces:
  - `normalize_mlflow_metric_name(name: str) -> str` — strips one leading `test/`
  - `read_runs_from_mlflow(tracking_uri: str, experiment: str) -> list[RunRow]`

- [ ] **Step 1: Write the failing tests**

The name normalization is pure and testable without mlflow. Add to `test_charge_aggregate.py`:

```python
def test_mlflow_metric_names_normalize_to_the_run_dir_spelling():
    """_log_mlflow_run prefixes EVERY metric with "test/", so run-dir `mae`
    is MLflow `test/mae` and run-dir `train/mae` is MLflow `test/train/mae`.
    Exactly one leading `test/` is stripped -- stripping greedily would turn
    a hypothetical `test/test/x` into `x`, and stripping none would make the
    two sources disagree about what a curve means."""
    from charge_experiments.aggregate import normalize_mlflow_metric_name as norm

    assert norm("test/mae") == "mae"
    assert norm("test/train/mae") == "train/mae"
    assert norm("test/train_loo/r2") == "train_loo/r2"
    assert norm("test/charge_conservation/mae") == "charge_conservation/mae"
    assert norm("mae") == "mae"  # already normalized: left alone
```

Create `charge_experiments/tests/test_charge_aggregate_optional.py`:

```python
"""MLflow-dependent aggregate tests. Skipped when mlflow is absent --
matches this series' own *_optional.py convention for optional deps."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")


def test_both_sources_yield_identical_run_rows(tmp_path):
    """The test that keeps the two readers honest. A run logged to MLflow
    the way _log_mlflow_run logs it, and the same run read off disk, must
    produce equal params and equal metrics -- otherwise `--source` silently
    changes what a curve means."""
    import json

    import mlflow

    from charge_experiments.aggregate import (
        read_runs_from_dirs,
        read_runs_from_mlflow,
    )

    config = {"predictor": {"name": "sieve", "params": {"max_wl_depth": 3}}}
    metrics = {"mae": 0.017, "train/mae": 0.009, "r2": 0.98}

    run_dir = tmp_path / "runs" / "exp" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    mlflow.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("exp")
    with mlflow.start_run(run_name="r1"):
        mlflow.log_params({"predictor.name": "sieve",
                           "predictor.params.max_wl_depth": "3"})
        # exactly what _log_mlflow_run does: every key prefixed with "test/"
        mlflow.log_metrics({f"test/{k}": v for k, v in metrics.items()})

    from_dirs = read_runs_from_dirs(tmp_path / "runs")[0]
    from_mlflow = read_runs_from_mlflow(tracking, "exp")[0]

    assert from_mlflow.params == from_dirs.params
    assert from_mlflow.metrics == from_dirs.metrics
```

- [ ] **Step 2: Run to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py \
              charge_experiments/tests/test_charge_aggregate_optional.py -v
```

Expected: FAIL — `cannot import name 'normalize_mlflow_metric_name'`.

- [ ] **Step 3: Implement**

Add to `aggregate.py`:

```python
_MLFLOW_METRIC_PREFIX = "test/"


def normalize_mlflow_metric_name(name: str) -> str:
    """MLflow's spelling -> metrics.json's spelling.

    ``runner._log_mlflow_run`` logs every metric as ``f"test/{key}"``,
    including the ones already prefixed with their own split -- so run-dir
    ``mae`` is MLflow ``test/mae`` and run-dir ``train/mae`` is MLflow
    ``test/train/mae``. Exactly one prefix is removed. That quirk is worked
    around here rather than fixed at the source, since changing it would
    break comparison with every run already logged.
    """
    return name[len(_MLFLOW_METRIC_PREFIX):] if name.startswith(
        _MLFLOW_METRIC_PREFIX
    ) else name


def read_runs_from_mlflow(tracking_uri: str, experiment: str) -> list[RunRow]:
    """Read runs from an MLflow tracking backend, normalized to the same
    ``RunRow`` shape ``read_runs_from_dirs`` returns."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    frame = mlflow.search_runs(experiment_names=[experiment])
    rows: list[RunRow] = []
    for record in frame.to_dict("records"):
        params = {
            key[len("params."):]: str(value)
            for key, value in record.items()
            if key.startswith("params.") and value is not None
        }
        raw_metrics = {
            normalize_mlflow_metric_name(key[len("metrics."):]): value
            for key, value in record.items()
            if key.startswith("metrics.") and value is not None
        }
        rows.append(
            RunRow(
                run_dir=str(record.get("tags.run_dir", "")),
                params=params,
                metrics=_finite_metrics(raw_metrics),
            )
        )
    return rows
```

Note the tag `run_dir` — `runner._log_mlflow`'s `tags` dict already sets it, so an MLflow-sourced row still points back at its directory.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py \
              charge_experiments/tests/test_charge_aggregate_optional.py -v
```

Expected: PASS (the optional file skips if mlflow is absent — confirm it does not *error*).

- [ ] **Step 5: Suite, lint, format, type-check; then commit**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
git add -A
git commit -m "feat(charge): MLflow reader, normalized to the run-dir spelling

_log_mlflow_run prefixes every metric with \"test/\", so run-dir \`mae\` is
MLflow \`test/mae\` and run-dir \`train/mae\` is MLflow \`test/train/mae\`.
The reader strips exactly one prefix, so --source never changes what a
curve means; the quirk is left alone at the source, where fixing it would
break comparison with every run already logged.

Pinned by a test that logs a run the way _log_mlflow_run does, reads it
both ways, and asserts the two RunRows are equal."
```

Expected: **130 passed** (128 + 2, one of which skips without mlflow).

---

## Task 4: `build_curve`

**Files:**
- Modify: `charge_experiments/charge_experiments/aggregate.py`
- Test: `charge_experiments/tests/test_charge_aggregate.py`

**Interfaces:**
- Consumes: `RunRow` (Task 2).
- Produces:
  - `CurvePoint` — `x_label: str`, `x_pos: float`, `mean: float`, `lo: float`, `hi: float`, `n_runs: int`
  - `CurveTable` — `x: str`, `series: dict[tuple[str, str], list[CurvePoint]]` keyed by `(series_name, metric)`, `raw_rows: list[dict[str, str]]`
  - `SPLIT_PREFIX: dict[str, str]` — `{"test": "", "train": "train/", "val": "val/", "train_loo": "train_loo/"}`
  - `build_curve(rows, *, x, metrics, splits, group_by=None) -> CurveTable`

- [ ] **Step 1: Write the failing tests**

```python
def _rows(triples):
    """(max_wl_depth, mae, train_mae) -> RunRows."""
    from charge_experiments.aggregate import RunRow

    return [
        RunRow(
            run_dir=f"r{i}",
            params={"predictor.params.max_wl_depth": str(d)},
            metrics={"mae": m, "train/mae": t},
        )
        for i, (d, m, t) in enumerate(triples)
    ]


def test_split_names_map_to_metric_key_prefixes():
    """Test metrics are UNPREFIXED in metrics.json (`mae`); every other
    split is prefixed (`train/mae`). Series selection has to know that."""
    from charge_experiments.aggregate import SPLIT_PREFIX

    assert SPLIT_PREFIX["test"] == ""
    assert SPLIT_PREFIX["train"] == "train/"
    assert SPLIT_PREFIX["train_loo"] == "train_loo/"


def test_build_curve_aggregates_repeated_x_to_mean_and_min_max():
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(1, 0.10, 0.05), (1, 0.20, 0.05), (2, 0.05, 0.01)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    points = table.series[("test", "mae")]
    assert [p.x_label for p in points] == ["1", "2"]
    assert points[0].mean == pytest.approx(0.15)
    assert (points[0].lo, points[0].hi) == pytest.approx((0.10, 0.20))
    assert points[0].n_runs == 2
    assert points[1].n_runs == 1


def test_build_curve_sorts_numeric_x_numerically_not_lexically():
    """Depths 2 and 10 must not sort as "10" < "2"."""
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(10, 0.01, 0.0), (2, 0.05, 0.0)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    assert [p.x_label for p in table.series[("test", "mae")]] == ["2", "10"]


def test_build_curve_drops_a_run_missing_the_metric():
    """A run without the requested key contributes no point -- it is never
    coerced to 0, which would plot as a real and excellent value."""
    from charge_experiments.aggregate import RunRow, build_curve

    rows = [
        RunRow("r0", {"d": "1"}, {"mae": 0.1}),
        RunRow("r1", {"d": "1"}, {}),  # crashed before scoring
    ]
    table = build_curve(rows, x="d", metrics=["mae"], splits=["test"])
    assert table.series[("test", "mae")][0].n_runs == 1


def test_build_curve_groups_by_a_second_parameter():
    from charge_experiments.aggregate import RunRow, build_curve

    rows = [
        RunRow("r0", {"d": "1", "n": "a"}, {"mae": 0.1}),
        RunRow("r1", {"d": "1", "n": "b"}, {"mae": 0.2}),
    ]
    table = build_curve(
        rows, x="d", metrics=["mae"], splits=["test"], group_by="n"
    )
    assert set(table.series) == {("test n=a", "mae"), ("test n=b", "mae")}


def test_raw_rows_hold_every_run_not_the_aggregate():
    """The CSV must record what was measured; the band is never the only
    record."""
    from charge_experiments.aggregate import build_curve

    table = build_curve(
        _rows([(1, 0.10, 0.05), (1, 0.20, 0.05)]),
        x="predictor.params.max_wl_depth",
        metrics=["mae"],
        splits=["test"],
    )
    assert len(table.raw_rows) == 2
    assert {r["mae"] for r in table.raw_rows} == {"0.1", "0.2"}
```

Add `import pytest` at the top of the test module.

- [ ] **Step 2: Run to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py -k "curve or split_names or raw_rows" -v
```

Expected: FAIL — `cannot import name 'build_curve'`.

- [ ] **Step 3: Implement**

```python
# metrics.json spells test metrics with NO prefix ("mae"); every other split
# carries its own ("train/mae"). Series selection has to know that.
SPLIT_PREFIX = {
    "test": "",
    "train": "train/",
    "val": "val/",
    "train_loo": "train_loo/",
}


@dataclass(frozen=True)
class CurvePoint:
    x_label: str
    x_pos: float
    mean: float
    lo: float
    hi: float
    n_runs: int


@dataclass(frozen=True)
class CurveTable:
    x: str
    series: dict[tuple[str, str], list[CurvePoint]]
    raw_rows: list[dict[str, str]]


def _x_sort_key(labels: list[str]) -> dict[str, float]:
    """Numeric x sorts numerically (2 before 10, not "10" before "2"); a
    non-numeric x falls back to sorted-unique categorical positions."""
    try:
        return {label: float(label) for label in labels}
    except ValueError:
        return {label: float(i) for i, label in enumerate(sorted(labels))}


def build_curve(
    rows: list[RunRow],
    *,
    x: str,
    metrics: list[str],
    splits: list[str],
    group_by: str | None = None,
) -> CurveTable:
    """Aggregate runs into one point per (series, metric, x value).

    Runs sharing an x value -- a sweep repeated over seeds, say -- become a
    mean with a min-max band. ``raw_rows`` keeps every contributing run, so
    the band is never the only record of what was measured.
    """
    from collections import defaultdict

    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    raw_rows: list[dict[str, str]] = []
    labels: set[str] = set()

    for row in rows:
        if x not in row.params:
            continue
        x_label = row.params[x]
        labels.add(x_label)
        raw: dict[str, str] = {"run_dir": row.run_dir, x: x_label}
        if group_by is not None:
            raw[group_by] = row.params.get(group_by, "")
        for split in splits:
            prefix = SPLIT_PREFIX.get(split, f"{split}/")
            name = split if group_by is None else (
                f"{split} {group_by.rsplit('.', 1)[-1]}={row.params.get(group_by, '')}"
            )
            for metric in metrics:
                value = row.metrics.get(f"{prefix}{metric}")
                if value is None:
                    continue
                buckets[(name, metric, x_label)].append(value)
                raw[f"{prefix}{metric}" if prefix else metric] = f"{value:.6g}"
        raw_rows.append(raw)

    positions = _x_sort_key(sorted(labels))
    series: dict[tuple[str, str], list[CurvePoint]] = defaultdict(list)
    for (name, metric, x_label), values in buckets.items():
        series[(name, metric)].append(
            CurvePoint(
                x_label=x_label,
                x_pos=positions[x_label],
                mean=sum(values) / len(values),
                lo=min(values),
                hi=max(values),
                n_runs=len(values),
            )
        )
    for points in series.values():
        points.sort(key=lambda p: p.x_pos)
    return CurveTable(x=x, series=dict(series), raw_rows=raw_rows)
```

Hoist `from collections import defaultdict` to the module's import block.

- [ ] **Step 4: Run the tests, then suite/lint/format/ty, then commit**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_aggregate.py -v
uv run pytest charge_experiments/tests -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
git add -A
git commit -m "feat(charge): build_curve aggregation

One point per (series, metric, x): mean with a min-max band over runs
sharing an x value, and raw_rows keeping every contributing run so the band
is never the only record.

Three things the tests pin: test metrics are UNPREFIXED in metrics.json
while every other split carries its own, so SPLIT_PREFIX maps test -> \"\";
numeric x sorts numerically, so depth 2 comes before depth 10; and a run
missing the requested metric contributes no point rather than being coerced
to 0, which would plot as a real and excellent value."
```

Expected: **136 passed** (130 + 6).

---

## Task 5: `plots.curve_panel`

**Files:**
- Modify: `charge_experiments/charge_experiments/plots.py`
- Test: `charge_experiments/tests/test_charge_plots.py`

**Interfaces:**
- Consumes: `CurveTable`, `CurvePoint` (Task 4).
- Produces: `plots.curve_panel(table, out_path, *, suptitle, n_cols=3) -> None`

- [ ] **Step 1: Write the failing test**

Look at how `test_charge_plots.py` guards matplotlib and follow it. Add:

```python
def test_curve_panel_writes_one_subplot_per_metric(tmp_path):
    """One subplot per metric, one line per series, with a min-max band."""
    pytest.importorskip("matplotlib")

    from charge_experiments.aggregate import CurvePoint, CurveTable
    from charge_experiments.plots import curve_panel

    table = CurveTable(
        x="predictor.params.max_wl_depth",
        series={
            ("test", "mae"): [
                CurvePoint("1", 1.0, 0.20, 0.18, 0.22, 2),
                CurvePoint("2", 2.0, 0.10, 0.09, 0.11, 2),
            ],
            ("train", "mae"): [
                CurvePoint("1", 1.0, 0.15, 0.15, 0.15, 1),
                CurvePoint("2", 2.0, 0.05, 0.05, 0.05, 1),
            ],
            ("test", "r2"): [CurvePoint("1", 1.0, 0.9, 0.9, 0.9, 1)],
        },
        raw_rows=[],
    )
    out = tmp_path / "curve.png"
    curve_panel(table, out, suptitle="sieve")
    assert out.exists() and out.stat().st_size > 0


def test_curve_panel_on_an_empty_table_writes_nothing(tmp_path):
    pytest.importorskip("matplotlib")

    from charge_experiments.aggregate import CurveTable
    from charge_experiments.plots import curve_panel

    out = tmp_path / "curve.png"
    curve_panel(CurveTable(x="d", series={}, raw_rows=[]), out, suptitle="none")
    assert not out.exists()
```

- [ ] **Step 2: Run to verify it fails**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_plots.py -k curve -v
```

Expected: FAIL — `cannot import name 'curve_panel'`.

- [ ] **Step 3: Implement**

Add to `plots.py`, after `parity_panel`:

```python
def curve_panel(
    table: Any,
    out_path: str | Path,
    *,
    suptitle: str,
    n_cols: int = 3,
) -> None:
    """One subplot per metric, one line per series, mean with a min-max band.

    ``table`` is an ``aggregate.CurveTable``. Typed as ``Any`` to keep this
    module free of a runtime import from ``aggregate`` -- ``plots`` is
    imported by ``runner`` on every run, and this is the only function that
    needs it.

    matplotlib is imported lazily, as in ``parity_panel``: a caller without
    it still gets the CSV.
    """
    import matplotlib.pyplot as plt

    metrics = sorted({metric for _, metric in table.series})
    if not metrics:
        return
    n_cols = min(n_cols, len(metrics))
    n_rows = -(-len(metrics) // n_cols)

    # Tick labels come from the union of every series' x positions, not from
    # one arbitrary series: series can cover different subsets of x (a run
    # missing one metric contributes no point for it), and taking the first
    # would silently drop ticks the other lines actually occupy.
    ticks = {
        point.x_pos: point.x_label
        for points in table.series.values()
        for point in points
    }
    tick_pos = sorted(ticks)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, metrics, strict=False):
        for (name, series_metric), points in sorted(table.series.items()):
            if series_metric != metric or not points:
                continue
            xs = [p.x_pos for p in points]
            ax.plot(xs, [p.mean for p in points], marker="o", ms=4, label=name)
            if any(p.n_runs > 1 for p in points):
                ax.fill_between(
                    xs, [p.lo for p in points], [p.hi for p in points], alpha=0.2
                )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([ticks[p] for p in tick_pos])
        ax.set_xlabel(table.x, fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run the tests, then suite/lint/format/ty, then commit**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_plots.py -v
uv run pytest charge_experiments/tests -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
git add -A
git commit -m "feat(charge): curve_panel, metric-vs-parameter plots

One subplot per metric, one line per series, mean with a min-max band drawn
only where a point actually aggregates more than one run. matplotlib stays
lazily imported as in parity_panel, so a caller without it still gets the
CSV."
```

Expected: **138 passed** (136 + 2).

---

## Task 6: the `sweep` subcommand

**Files:**
- Modify: `charge_experiments/charge_experiments/cli.py`
- Test: `charge_experiments/tests/test_charge_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: `charge-exp sweep` writing `results/<name>/curve.png` and `results/<name>/curve.csv`.

- [ ] **Step 1: Write the failing test**

```python
def _sweep_tree(root):
    """Three synthetic runs at depths 1/2/3 -- no store, no MLflow, no real
    experiment."""
    import json

    for i, depth in enumerate([1, 2, 3]):
        run_dir = root / "runs" / "exp" / f"r{i}"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"config": {"predictor": {"params": {"max_wl_depth": depth}}}})
        )
        (run_dir / "metrics.json").write_text(
            json.dumps({"mae": 0.3 / depth, "train/mae": 0.2 / depth})
        )
    return root / "runs"


def test_sweep_writes_a_curve_csv_from_run_dirs(tmp_path, monkeypatch):
    """The CSV holds one raw row per run, and is written whether or not
    matplotlib is installed."""
    import csv

    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    rc = cli.main(
        [
            "sweep",
            "--x", "predictor.params.max_wl_depth",
            "--metric", "mae",
            "--split", "test",
            "--split", "train",
        ]
    )
    assert rc == 0

    out = tmp_path / "results" / "max_wl_depth" / "curve.csv"
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    assert {r["predictor.params.max_wl_depth"] for r in rows} == {"1", "2", "3"}
    # both requested splits present, under their metrics.json spellings
    assert rows[0]["mae"] and rows[0]["train/mae"]


def test_sweep_out_name_defaults_to_the_x_paths_last_segment(tmp_path, monkeypatch):
    """Repeated sweeps over one parameter overwrite one directory rather
    than accumulating timestamped ones."""
    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    assert cli.main(["sweep", "--x", "predictor.params.max_wl_depth"]) == 0
    assert (tmp_path / "results" / "max_wl_depth" / "curve.csv").exists()

    assert cli.main(["sweep", "--x", "predictor.params.max_wl_depth"]) == 0
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["max_wl_depth"]


def test_sweep_reports_no_match_instead_of_writing_an_empty_curve(
    tmp_path, monkeypatch, capsys
):
    """An x path no run carries is a user error, not an empty plot."""
    from charge_experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", _sweep_tree(tmp_path))
    assert cli.main(["sweep", "--x", "predictor.params.nonexistent"]) == 1
    assert "no runs matched" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
```

Note the default-metrics path in the second test: `--metric` is omitted, so `mae`/`rmse`/`r2`
are requested and only `mae` exists in the fixture. That exercises Task 4's "a run missing
the requested metric contributes no point" behavior end to end.

- [ ] **Step 2: Run to verify it fails**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_cli.py -k sweep -v
```

Expected: FAIL — `invalid choice: 'sweep'`.

- [ ] **Step 3: Implement the command**

Add to `cli.py`:

```python
def _cmd_sweep(args: argparse.Namespace) -> int:
    from charge_experiments.aggregate import (
        build_curve,
        read_runs_from_dirs,
        read_runs_from_mlflow,
    )

    if args.source == "mlflow":
        if not args.experiment:
            raise SystemExit("--source mlflow requires --experiment")
        rows = read_runs_from_mlflow(args.tracking, args.experiment)
    else:
        rows = read_runs_from_dirs(DEFAULT_RUNS_ROOT, args.experiment)

    table = build_curve(
        rows,
        x=args.x,
        metrics=args.metric,
        splits=args.split,
        group_by=args.group_by,
    )
    if not table.raw_rows:
        print(f"no runs matched {args.x!r}; nothing written")
        return 1

    name = args.out or args.x.rsplit(".", 1)[-1]
    out_dir = DEFAULT_RUNS_ROOT.parent / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in table.raw_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = out_dir / "curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(table.raw_rows)
    print(f"wrote {len(table.raw_rows)} row(s) to {csv_path}")

    try:
        from charge_experiments.plots import curve_panel

        curve_panel(table, out_dir / "curve.png", suptitle=args.x)
        print(f"wrote {out_dir / 'curve.png'}")
    except ImportError:
        logging.getLogger("charge_experiments").warning(
            "matplotlib not installed; wrote curve.csv only"
        )
    return 0
```

And in `build_parser`, after `p_summary`:

```python
    p_sweep = sub.add_parser(
        "sweep",
        help="plot metrics against a run parameter, gathered across many runs",
    )
    p_sweep.add_argument(
        "--x", required=True,
        help="dotted config path for the x axis, e.g. predictor.params.max_wl_depth",
    )
    p_sweep.add_argument("--experiment", default=None)
    p_sweep.add_argument("--source", choices=("runs", "mlflow"), default="runs")
    p_sweep.add_argument("--tracking", default=DEFAULT_TRACKING_URI)
    p_sweep.add_argument(
        "--metric", action="append", default=None,
        help="repeatable; default: mae, rmse, r2",
    )
    p_sweep.add_argument(
        "--split", action="append", default=None,
        help="repeatable; default: test, train",
    )
    p_sweep.add_argument("--group-by", dest="group_by", default=None)
    p_sweep.add_argument("--out", default=None, help="results/<NAME>/")
    p_sweep.set_defaults(func=_cmd_sweep)
```

`action="append"` with a non-empty `default` would *extend* it rather than replace, so both default to `None` and are filled in `_cmd_sweep`:

```python
    args.metric = args.metric or ["mae", "rmse", "r2"]
    args.split = args.split or ["test", "train"]
```

Put those two lines at the top of `_cmd_sweep`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest charge_experiments/tests/test_charge_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Exercise the real workflow by hand**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run python -m charge_experiments sweep --x predictor.params.max_wl_depth --metric mae
```

Against the repo's existing `runs/` tree this should either write a curve or print "no runs matched" — both are correct outcomes; a traceback is not.

- [ ] **Step 6: Full suite, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: charge suite **141 passed** (138 + 3); full suite **350 passed, 1 skipped** (327 + 23 new; one of the 23 skips without mlflow).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(charge): the sweep subcommand

Gathers runs from run directories or MLflow and plots any metric against
any dotted config path, writing results/<name>/curve.{png,csv}. The CSV
holds raw per-run rows, and is written even when matplotlib is absent.

--out defaults to the x path's last segment, so repeated sweeps over one
parameter overwrite one directory rather than accumulating timestamped
ones. --metric/--split use action=append against a None default, since an
append action against a non-empty default extends it instead of replacing
it.

Together with report_loo this answers the original question -- accuracy as
a function of refinement depth, for training and prediction -- by sweeping
max_wl_depth rather than instrumenting a single run:

  for d in 0 1 2 3; do
    charge-exp run --config configs/sieve-charge-example.yaml \\
      --set predictor.params.max_wl_depth=\$d \\
      --set predictor.params.report_loo=true
  done
  charge-exp sweep --x predictor.params.max_wl_depth \\
    --split test --split train --split train_loo"
```

---

## Verification Checklist

- [ ] `uv run pytest -q` → 350 passed, 1 skipped (327 baseline + 23). Counts are a sanity check, not a contract — reconcile against the tests each task adds before assuming breakage.
- [ ] `uv run ruff check src tests charge_experiments` → clean
- [ ] `uv run ruff format --check src tests charge_experiments` → clean
- [ ] `uv run ty check src tests charge_experiments` → clean
- [ ] `git diff main --stat` reviewed — **`src/sieve/` is not in it**
- [ ] `summarize`'s CSV is byte-identical to pre-refactor (Task 2 Step 6)
- [ ] `charge_experiments/tests/test_charge_aggregate_optional.py` *skips* rather than errors when mlflow is absent
- [ ] Every spec section maps to a task: Part A → 1; Part B readers → 2, 3; aggregation → 4; plots → 5; CLI → 6
