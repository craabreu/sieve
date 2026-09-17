# Analytic Training Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute a fit's training error, R², η², support distribution and leave-one-out error exactly from the statistics it already stores, with no molecules loaded and no walk run.

**Architecture:** One backoff walk in a new `experiments/analytic.py`, parameterized by `(loo, weighted)`, fed per-class `(n, Σy, Σy²)` triples by a thin adapter per predictor. The walk mirrors `sieve.predict._search` bottom-up: a class either has the support the config demands and answers every atom still carrying it, or its moments fold into its parent and try again one level up.

**Tech Stack:** Python 3.11+, numpy, scipy (already present). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-analytic-training-metrics-design.md`

## Global Constraints

- **Never assert against a stored constant.** Every analytic number is tested by comparing it to brute-force predicting the training set with the real predictor. The closed form is an identity on the stored statistics, so a test against anything else would pass even if the identity were wrong.
- **DASH is out of scope entirely.** Do not touch `tree_artifact.py`, `predictors/dash.py`, or any DASH workflow step except to *keep* `--score-train` where it already is. Spec section 2 records the measurement that excludes it.
- **HOSE's predictions must not change.** The only edit to `predictors/hose.py` is accumulating a second moment during `fit`. `predict`, `n_min` handling and the backoff loop stay byte-identical.
- **`minimum_support` semantics:** a class answers when `count >= minimum_support`; under LOO when `count >= minimum_support + 1`.
- **Existing artifacts stay loadable.** A `HoseState` saved before this change must still load, merge and score. Only the analytic entry point may refuse it.
- Run `uv run ruff check`, `uv run ruff format --check` and `uv run ty check` before every commit. Local `ty` is newer than CI's, so a clean local run is a lower bound, not a guarantee.
- Tests that write a parquet store must guard `pandas`/`pyarrow` with `pytest.importorskip` — CI installs `[dev,chem]`, which excludes them.

---

### Task 1: The backoff walk and the training identity

**Files:**
- Create: `experiments/experiments/analytic.py`
- Test: `experiments/tests/test_analytic.py`

**Interfaces:**
- Consumes: `sieve.shrinkage.shrunk_means`, `model.config.{backoff_path,level_parents,minimum_support,max_wl_depth}`, `model.levels[k].{count,mean,msd,parent}`, `model.{global_mean,global_msd,global_count}`
- Produces: `TrainStats` dataclass with fields `depth, n_classes, n_atoms, sse, rmse, r_squared, eta_squared, matched_fraction, support_fractions` and method `as_row() -> dict`; `sieve_train_stats(model, *, loo=False, thresholds=SUPPORT_THRESHOLDS) -> TrainStats`; module constant `SUPPORT_THRESHOLDS = (2, 4, 12, 50)`

> **Note for the implementer:** a draft of this module and its tests already exists uncommitted on the branch, written before the spec was approved. Treat it as a starting point, not as done — it has never been run. Steps 2 and 4 below are what establish whether it is correct.

- [ ] **Step 1: Write the failing test**

Create `experiments/tests/test_analytic.py`:

```python
"""The analytic curve must agree with actually predicting the training set."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _fit(n_mol=12, seed=0, depth=3, minimum_support=1, shrinkage_strength=None, **params):
    """A fitted model plus the training batch it was fitted on."""
    import sieve
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _batch_for,
        _build_config,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=n_mol, seed=seed)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=depth,
        minimum_support=minimum_support,
        shrinkage_strength=shrinkage_strength,
    )
    if params:
        import dataclasses

        config = dataclasses.replace(config, **params)
    batch = _batch_for(mset, config)
    return sieve.fit(batch, config), batch


def _brute_force_sse(model, batch):
    import sieve

    pred = sieve.predict(model, batch)
    return float(((batch.y - pred) ** 2).sum())


def test_analytic_sse_matches_predicting_the_training_set():
    from experiments.analytic import sieve_train_stats

    model, batch = _fit()
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest experiments/tests/test_analytic.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.analytic'`

- [ ] **Step 3: Write the module**

Create `experiments/experiments/analytic.py`. The walk:

```python
def sieve_train_stats(model, *, loo=False, thresholds=SUPPORT_THRESHOLDS):
    from sieve.shrinkage import shrunk_means

    cfg = model.config
    if loo:
        _refuse_loo_unsupported(cfg)

    path = cfg.backoff_path
    parents = cfg.level_parents
    floor = cfg.minimum_support + 1 if loo else cfg.minimum_support
    values = shrunk_means(model)

    deepest = model.levels[path[-1]]
    n_in, s_in, q_in = _moments(deepest.count, deepest.mean, deepest.msd)
    n_total = float(n_in.sum())
    if n_total == 0:
        raise ValueError("model holds no training atoms; nothing to score")

    sse = 0.0
    n_matched = 0.0
    for k in reversed(path):
        level = model.levels[k]
        carried = n_in > 0
        hit = carried & (level.count >= floor)
        if hit.any():
            scale = _loo_scale(level.count[hit]) if loo else 1.0
            centre = level.mean[hit] if loo else values[k][hit]
            sse += scale * _sse_against(n_in[hit], s_in[hit], q_in[hit], centre)
            n_matched += float(n_in[hit].sum())

        rest = carried & ~(level.count >= floor)
        parent = parents[k]
        if parent < 0:
            if rest.any():
                sse += _global_sse(model, n_in[rest], s_in[rest], q_in[rest], loo=loo)
            break
        if not rest.any():
            break
        n_in, s_in, q_in = _fold_into_parent(
            model.levels[parent], level.parent, rest, n_in, s_in, q_in
        )

    return _finish(model, deepest, sse, n_total, n_matched, thresholds,
                   depth=cfg.max_wl_depth)
```

with the helpers:

```python
def _moments(count, mean, msd):
    """Per class: (n, sum y, sum y^2). msd is the population variance, so
    sum y^2 = n * (msd + mean^2)."""
    n = count.astype(np.float64)
    return n, n[:, None] * mean, n[:, None] * (msd + mean * mean)


def _sse_against(n, s, q, value):
    """sum_i (y_i - v)^2 for a group summarized by (n, sum y, sum y^2).

    Expanded rather than centered: under shrinkage, backoff or LOO the
    prediction is not the group's own mean, and only the expansion takes an
    arbitrary v."""
    return float((q - 2.0 * value * s + n[:, None] * value * value).sum())


def _loo_scale(count):
    """(N/(N-1))^2 -- the factor a leave-one-out residual picks up, against
    the class's own full count."""
    n = count.astype(np.float64)
    return (n / (n - 1.0))[:, None] ** 2


def _global_sse(model, n, s, q, *, loo):
    value = np.broadcast_to(model.global_mean, s.shape)
    if not loo:
        return _sse_against(n, s, q, value)
    total = float(model.global_count)
    return (total / (total - 1.0)) ** 2 * _sse_against(n, s, q, value)


def _fold_into_parent(parent_level, parent_of, rest, n_in, s_in, q_in):
    """Aggregate unanswered classes' moments onto their parent classes.

    np.add.at rather than bincount: several child classes share one parent
    and the accumulation is over an (n_classes, d) array."""
    n_parent = parent_level.count.shape[0]
    d = parent_level.mean.shape[1]
    n_out = np.zeros(n_parent)
    s_out = np.zeros((n_parent, d))
    q_out = np.zeros((n_parent, d))
    target = parent_of[rest]
    np.add.at(n_out, target, n_in[rest])
    np.add.at(s_out, target, s_in[rest])
    np.add.at(q_out, target, q_in[rest])
    return n_out, s_out, q_out


def _finish(model, deepest, sse, n_total, n_matched, thresholds, *, depth):
    d = model.global_mean.shape[0]
    tss = float(n_total * np.sum(model.global_msd))
    within = float((deepest.count[:, None] * deepest.msd).sum())
    counts = deepest.count.astype(np.float64)
    populated = counts > 0
    return TrainStats(
        depth=depth,
        n_classes=int(populated.sum()),
        n_atoms=int(n_total),
        sse=sse,
        rmse=float(np.sqrt(sse / (n_total * d))),
        r_squared=float(1.0 - sse / tss) if tss > 0 else float("nan"),
        eta_squared=float(1.0 - within / tss) if tss > 0 else float("nan"),
        matched_fraction=float(n_matched / n_total),
        support_fractions={
            t: float(counts[populated & (counts < t)].sum() / n_total)
            for t in thresholds
        },
    )
```

`_refuse_loo_unsupported` is Task 2; for now define it as a no-op stub that Task 2 fills in.

`values = shrunk_means(model)` is correct for every reading of the class tables: with no shrinkage it returns the class's own estimate (pooled or continuation) for populated classes, and under every shrinkage rule `predict` provably lands on the same class-indexed value. Do not substitute `class_means`.

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest experiments/tests/test_analytic.py -x -q`
Expected: PASS

- [ ] **Step 5: Add the cases that make it the general form**

Append to the test file:

```python
@pytest.mark.parametrize("minimum_support", [1, 2, 3, 8])
def test_analytic_sse_holds_when_training_atoms_back_off(minimum_support):
    """The case spec section 3 generalizes. With minimum_support > 1 a
    training atom's own deepest class may be too small to answer it."""
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=minimum_support, depth=4)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


@pytest.mark.parametrize("class_estimator", ["pooled", "continuation"])
def test_analytic_sse_holds_under_each_class_estimator(class_estimator):
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(class_estimator=class_estimator)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


@pytest.mark.parametrize("shrinkage_weight", ["count", "diversity", "empirical_bayes"])
def test_analytic_sse_holds_under_each_shrinkage_rule(shrinkage_weight):
    """Shrinkage moves the prediction off the class mean, so the centered
    form sum n_c var_c is simply wrong here."""
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(
        shrinkage_strength=0.5,
        shrinkage_weight=shrinkage_weight,
        class_estimator="continuation",
    )
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


def test_analytic_sse_holds_under_neighbor_depth():
    """The coarse chain is scaffolding, never a backoff target -- the walk
    must skip exactly the levels predict skips."""
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(depth=3, neighbor_depth=1)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


def test_reduces_to_the_note_s_own_formula_in_the_simple_case():
    """SSE = sum_c n_c * msd_c -- the note's section 1, which the general
    walk must reproduce when nothing backs off."""
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(minimum_support=1)
    deepest = model.levels[-1]
    simple = float((deepest.count[:, None] * deepest.msd).sum())
    assert sieve_train_stats(model).sse == pytest.approx(simple, rel=1e-12)


def test_an_empty_model_is_refused_rather_than_dividing_by_zero():
    import sieve
    from experiments.analytic import sieve_train_stats

    model, _ = _fit()
    with pytest.raises(ValueError, match="no training atoms"):
        sieve_train_stats(sieve.SieveModel.empty(model.config))
```

- [ ] **Step 6: Run them**

Run: `uv run pytest experiments/tests/test_analytic.py -q`
Expected: all PASS. If `neighbor_depth` or a shrinkage rule fails, the bug is in the walk, not the test — `predict` is the reference.

- [ ] **Step 7: Add the free diagnostics tests**

```python
def test_r_squared_and_eta_squared_agree_when_nothing_shrinks():
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(_fit(minimum_support=1)[0])
    assert stats.r_squared == pytest.approx(stats.eta_squared, rel=1e-12)


def test_shrinkage_separates_r_squared_from_eta_squared():
    """eta^2 describes the partition, R^2 the estimator."""
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(_fit(shrinkage_strength=5.0)[0])
    assert stats.r_squared < stats.eta_squared


def test_support_fractions_count_atoms_not_classes():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(depth=4)
    counts = model.levels[-1].count.astype(float)
    expected = counts[counts < 12].sum() / counts.sum()
    assert sieve_train_stats(model).support_fractions[12] == pytest.approx(expected)


def test_everything_matches_at_minimum_support_one():
    from experiments.analytic import sieve_train_stats

    assert sieve_train_stats(_fit(minimum_support=1)[0]).matched_fraction == 1.0
```

- [ ] **Step 8: Run, lint, commit**

```bash
uv run pytest experiments/tests/test_analytic.py -q
uv run ruff check && uv run ruff format --check && uv run ty check
git add experiments/experiments/analytic.py experiments/tests/test_analytic.py
git commit -m "feat(experiments): training error from the stored statistics"
```

---

### Task 2: Leave-one-out

**Files:**
- Modify: `experiments/experiments/analytic.py`
- Test: `experiments/tests/test_analytic.py`

**Interfaces:**
- Consumes: Task 1's walk (the `loo` parameter is already threaded through it)
- Produces: `_refuse_loo_unsupported(cfg)` raising `NotImplementedError`; `sieve_train_stats(model, loo=True)` returning LOO statistics

- [ ] **Step 1: Write the failing tests**

```python
def _brute_force_loo_sse(model, batch):
    from sieve.predict import predict_loo

    pred = predict_loo(model, batch).value
    return float(((batch.y - pred) ** 2).sum())


def test_analytic_loo_matches_predict_loo():
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=1, depth=3)
    stats = sieve_train_stats(model, loo=True)
    assert stats.sse == pytest.approx(_brute_force_loo_sse(model, batch), rel=1e-10)


@pytest.mark.parametrize("minimum_support", [1, 2, 4])
def test_analytic_loo_matches_predict_loo_with_backoff(minimum_support):
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=minimum_support, depth=4)
    stats = sieve_train_stats(model, loo=True)
    assert stats.sse == pytest.approx(_brute_force_loo_sse(model, batch), rel=1e-10)


def test_analytic_loo_is_never_better_than_the_training_error():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(depth=4)
    assert sieve_train_stats(model, loo=True).sse >= sieve_train_stats(model).sse


def test_analytic_loo_refuses_continuation():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(class_estimator="continuation")
    with pytest.raises(NotImplementedError, match="class_estimator"):
        sieve_train_stats(model, loo=True)


def test_analytic_loo_refuses_shrinkage():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(shrinkage_strength=0.5)
    with pytest.raises(NotImplementedError, match="shrinkage"):
        sieve_train_stats(model, loo=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest experiments/tests/test_analytic.py -q -k loo`
Expected: the two refusal tests FAIL (stub does not raise); the numeric ones may already pass if Task 1's `loo` branch is right.

- [ ] **Step 3: Implement the refusals**

Replace the stub:

```python
def _refuse_loo_unsupported(cfg):
    from sieve.config import CLASS_ESTIMATOR_POOLED

    if cfg.class_estimator != CLASS_ESTIMATOR_POOLED:
        raise NotImplementedError(
            f"analytic LOO does not support class_estimator="
            f"{cfg.class_estimator!r}: a non-deepest class estimates its "
            f"children's mean, so removing one atom needs the child identity "
            f"this walk does not carry -- the same limit sieve.predict_loo has"
        )
    if cfg.applies_shrinkage:
        raise NotImplementedError(
            "analytic LOO does not support shrinkage: the shrunk value "
            "depends on the class count the held-out atom contributes to, so "
            "the correction does not factor out of the class"
        )
```

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest experiments/tests/test_analytic.py -q
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "feat(experiments): closed-form leave-one-out from the same walk"
```

---

### Task 3: The depth curve and its CLI

**Files:**
- Modify: `experiments/experiments/analytic.py`, `experiments/experiments/cli.py`
- Test: `experiments/tests/test_analytic.py`, `experiments/tests/test_cli.py`

**Interfaces:**
- Consumes: `experiments.cv.truncate_model`, Task 1's `sieve_train_stats`
- Produces: `sieve_curve(model, depths, *, loo=False, thresholds=...) -> list[TrainStats]`; CLI subcommand `analytic-curve`

- [ ] **Step 1: Write the failing tests**

```python
def test_curve_matches_fitting_each_depth_natively():
    from experiments.analytic import sieve_curve, sieve_train_stats

    model, _ = _fit(depth=4)
    curve = sieve_curve(model, [1, 2, 3, 4])
    assert [row.depth for row in curve] == [1, 2, 3, 4]
    for row in curve:
        native, _ = _fit(depth=row.depth)
        assert row.sse == pytest.approx(sieve_train_stats(native).sse, rel=1e-12)


def test_training_error_falls_monotonically_with_depth():
    """It can only fall: refining a partition cannot raise within-class
    variance. A violation means the walk reads the wrong level."""
    from experiments.analytic import sieve_curve

    model, _ = _fit(depth=5, n_mol=16)
    rmse = [row.rmse for row in sieve_curve(model, [1, 2, 3, 4, 5])]
    assert rmse == sorted(rmse, reverse=True)


def test_as_row_is_flat_and_carries_the_support_columns():
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(_fit()[0])
    row = stats.as_row()
    assert row["frac_support_lt_12"] == pytest.approx(stats.support_fractions[12])
    assert set(row) >= {"depth", "n_classes", "n_atoms", "sse", "rmse", "r2", "eta2"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest experiments/tests/test_analytic.py -q -k "curve or as_row"`
Expected: FAIL with `ImportError: cannot import name 'sieve_curve'`

- [ ] **Step 3: Implement `sieve_curve` and `as_row`**

```python
def sieve_curve(model, depths, *, loo=False, thresholds=SUPPORT_THRESHOLDS):
    """sieve_train_stats at each depth, by truncating the fitted model.

    Truncation is exact -- WL refinement never looks ahead, so a depth-D
    fit's levels 0..d are bit-for-bit what a native depth-d fit stored
    (cv.truncate_model). One merged model gives the whole curve."""
    from experiments.cv import truncate_model

    return [
        sieve_train_stats(truncate_model(model, d), loo=loo, thresholds=thresholds)
        for d in sorted(depths)
    ]
```

`as_row` flattens `support_fractions` into `frac_support_lt_{t}` keys; see the `TrainStats` definition in Task 1.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest experiments/tests/test_analytic.py -q`
Expected: PASS

- [ ] **Step 5: Add the CLI command**

In `cli.py`, next to `_cmd_merge_states`:

```python
def _cmd_analytic_curve(args: argparse.Namespace) -> int:
    import csv
    import sys

    from experiments.analytic import sieve_curve

    if args.predictor == "sieve":
        import sieve

        model = sieve.SieveModel.load(args.state)
        depths = args.depths or list(range(1, model.config.max_wl_depth + 1))
        rows = [s.as_row() for s in sieve_curve(model, depths, loo=args.loo)]
    else:
        from experiments.analytic import hose_curve
        from experiments.hose_artifact import load_hose_state

        state = load_hose_state(args.state)
        radii = args.depths or list(range(1, state.max_radius + 1))
        rows = [s.as_row() for s in hose_curve(state, radii, loo=args.loo)]

    handle = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            handle.close()
    if args.out:
        print(f"wrote {len(rows)} row(s) -> {args.out}")
    return 0
```

and the parser, after `p_merge_states`:

```python
p_analytic = sub.add_parser(
    "analytic-curve",
    help="training error, R^2, eta^2, support distribution and LOO for a "
    "saved model state, computed from its own stored statistics -- no "
    "store, no molecules, no walk",
)
p_analytic.add_argument("state", type=Path)
p_analytic.add_argument("--predictor", required=True, choices=("sieve", "hose"))
p_analytic.add_argument(
    "--depths",
    type=lambda s: [int(x) for x in s.split(",")],
    help="comma-separated; defaults to every depth/radius the state carries",
)
p_analytic.add_argument("--loo", action="store_true")
p_analytic.add_argument("--out", type=Path, help="CSV path; stdout if omitted")
p_analytic.set_defaults(func=_cmd_analytic_curve)
```

The `hose_curve` branch will not import until Task 5. That is fine — it is inside the function.

- [ ] **Step 6: Test the CLI end to end**

In `experiments/tests/test_cli.py`:

```python
def test_analytic_curve_writes_a_csv(tmp_path):
    import csv

    from experiments.__main__ import main
    from experiments.tests.test_analytic import _fit

    model, _ = _fit(depth=3)
    state = tmp_path / "model.npz"
    model.save(state)
    out = tmp_path / "curve.csv"
    assert main(["analytic-curve", str(state), "--predictor", "sieve",
                 "--depths", "1,2,3", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert [r["depth"] for r in rows] == ["1", "2", "3"]
    assert float(rows[0]["rmse"]) > float(rows[-1]["rmse"])
```

Check the existing `test_cli.py` for how it invokes `main` and match it.

- [ ] **Step 7: Run, lint, commit**

```bash
uv run pytest experiments/tests/test_analytic.py experiments/tests/test_cli.py -q
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "feat(experiments): analytic-curve command over a saved state"
```

---

### Task 4: HOSE stores a second moment

**Files:**
- Modify: `experiments/experiments/hose_artifact.py`, `experiments/experiments/predictors/hose.py`
- Test: `experiments/tests/test_hose_artifact.py`, `experiments/tests/test_predictor_hose.py`

**Interfaces:**
- Produces: `HoseState.tables[k]` maps key -> `(sum, sumsq, count)`; `HoseState.has_second_moment -> bool`
- Consumes: nothing new

**This task must not change a single HOSE prediction.** `predict`, `n_min` and the backoff loop stay as they are.

- [ ] **Step 1: Write the failing tests**

In `test_hose_artifact.py`:

```python
def test_state_carries_a_second_moment():
    from experiments.hose_artifact import HoseState

    state = HoseState(
        max_radius=1,
        tables=({}, {"C": (3.0, 5.0, 2)}),
        global_sum=3.0,
        global_count=2,
        global_sumsq=5.0,
    )
    assert state.has_second_moment
    assert state.tables[1]["C"] == (3.0, 5.0, 2)


def test_merge_sums_all_three_columns():
    from experiments.hose_artifact import HoseState, merge_hose_states

    a = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    b = HoseState(
        1, ({}, {"C": (1.0, 1.0, 1), "N": (2.0, 4.0, 1)}), 3.0, 2, global_sumsq=5.0
    )
    merged = merge_hose_states(a, b)
    assert merged.tables[1]["C"] == (4.0, 6.0, 3)
    assert merged.tables[1]["N"] == (2.0, 4.0, 1)


def test_a_legacy_state_still_loads_and_merges(tmp_path):
    """Pre-existing artifacts must keep working; only the analytic call may
    refuse them."""
    import numpy as np

    from experiments.hose_artifact import load_hose_state, merge_hose_states

    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        max_radius=np.asarray(1),
        global_sum=np.asarray(3.0),
        global_count=np.asarray(2),
        r1_keys=np.asarray(["C"], dtype=np.str_),
        r1_sum=np.asarray([3.0]),
        r1_count=np.asarray([2]),
    )
    state = load_hose_state(path)
    assert not state.has_second_moment
    assert state.tables[1]["C"][0] == 3.0
    assert state.tables[1]["C"][2] == 2
    merged = merge_hose_states(state, state)
    assert not merged.has_second_moment
    assert merged.tables[1]["C"][2] == 4
```

In `test_predictor_hose.py`:

```python
def test_fit_records_the_second_moment_and_leaves_predictions_alone():
    import numpy as np

    from experiments.predictors.hose import HoseLookupPredictor
    from experiments.tests.helpers import synthetic_molecule_set

    pytest.importorskip("hosegen")
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    p = HoseLookupPredictor(max_radius=2)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    before = p.predict(mset).atom_value
    state = p.model_state()
    assert state.has_second_moment
    for key, (s, qq, c) in state.tables[1].items():
        assert qq >= s * s / c - 1e-12   # sum of squares >= n * mean^2
    p.set_model_state(state)
    np.testing.assert_allclose(p.predict(mset).atom_value, before)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest experiments/tests/test_hose_artifact.py experiments/tests/test_predictor_hose.py -q`
Expected: FAIL — `HoseState` takes no `global_sumsq`.

- [ ] **Step 3: Widen `HoseState`**

In `hose_artifact.py`:

- `tables: tuple[dict[str, tuple[float, float, int]], ...]`, entries `(sum, sumsq, count)`.
- Add field `global_sumsq: float | None = None` **last**, after `global_count` — a defaulted field cannot precede a non-defaulted one. Construct it by keyword everywhere.
- Add `has_second_moment` property: `self.global_sumsq is not None`.
- `merge_hose_states`: sum all three columns; `global_sumsq` is `None` if either side's is `None`, else the sum. **Do not substitute zeros for a missing column** — zero reads as "no spread" and would silently corrupt any later curve.
- `save_hose_state`: write `r{k}_sumsq` and `global_sumsq` only when `has_second_moment`.
- `load_hose_state`: read them when present; when absent set every entry's sumsq to `float("nan")` and `global_sumsq=None`.

Legacy entries therefore carry `nan` in the middle slot, which propagates rather than silently reading as zero.

- [ ] **Step 4: Accumulate it in `fit`**

In `predictors/hose.py`'s `fit`, add a `sumsqs` accumulator beside `sums`/`counts`:

```python
sumsqs: list[defaultdict[str, float]] = [
    defaultdict(float) for _ in range(self.max_radius + 1)
]
...
for value, code in zip(target, self._codes(train.mols), strict=True):
    v = float(value)
    for k in range(1, self.max_radius + 1):
        key = sphere_prefix(code, k)
        sums[k][key] += v
        sumsqs[k][key] += v * v
        counts[k][key] += 1
```

Keep `self._tables` exactly as it is — `{key: (mean, count)}` — so `predict` is untouched. Store the second moment separately as `self._sumsq: list[dict[str, float]] | None`, and have `model_state()` emit the triple and `set_model_state()` unpack it.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest experiments/tests/test_hose_artifact.py experiments/tests/test_predictor_hose.py experiments/tests/test_predictor_hose_optional.py -q`
Expected: PASS, including every pre-existing HOSE test.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "feat(experiments): HOSE records a second moment per key"
```

---

### Task 5: The HOSE analytic curve

**Files:**
- Modify: `experiments/experiments/analytic.py`
- Test: `experiments/tests/test_analytic.py`

**Interfaces:**
- Consumes: `HoseState` from Task 4
- Produces: `hose_train_stats(state, radius, *, n_min=1, loo=False, thresholds=...) -> TrainStats`; `hose_curve(state, radii, *, ...) -> list[TrainStats]`

At `n_min = 1` the deepest key always answers a training atom, so there is no backoff and the walk collapses to one level:

```
SSE  = sum_keys ( Q_key - S_key^2 / N_key )
LOO  = sum_keys N^3 var / (N-1)^2       (keys with N >= 2)
```

- [ ] **Step 1: Write the failing test**

```python
def test_hose_analytic_matches_predicting_its_own_training_set():
    import numpy as np

    from experiments.analytic import hose_train_stats
    from experiments.predictors.hose import HoseLookupPredictor
    from experiments.tests.helpers import synthetic_molecule_set

    pytest.importorskip("hosegen")
    mset = synthetic_molecule_set(n_mol=8, seed=0)
    p = HoseLookupPredictor(max_radius=3, n_min=1)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    brute = float(((mset.atom_target - p.predict(mset).atom_value) ** 2).sum())
    stats = hose_train_stats(p.model_state(), radius=3)
    assert stats.sse == pytest.approx(brute, rel=1e-10)


def test_hose_analytic_refuses_a_state_without_the_second_moment(tmp_path):
    from experiments.analytic import hose_train_stats
    from experiments.hose_artifact import HoseState

    legacy = HoseState(1, ({}, {"C": (3.0, float("nan"), 2)}), 3.0, 2, global_sumsq=None)
    with pytest.raises(ValueError, match="refit"):
        hose_train_stats(legacy, radius=1)


def test_hose_analytic_refuses_a_non_baseline_n_min():
    from experiments.analytic import hose_train_stats
    from experiments.hose_artifact import HoseState

    state = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    with pytest.raises(NotImplementedError, match="n_min"):
        hose_train_stats(state, radius=1, n_min=3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest experiments/tests/test_analytic.py -q -k hose`
Expected: FAIL with `ImportError: cannot import name 'hose_train_stats'`

- [ ] **Step 3: Implement**

```python
def hose_train_stats(state, radius, *, n_min=1, loo=False,
                     thresholds=SUPPORT_THRESHOLDS):
    """Training statistics for one HOSE radius, from its own key table.

    Restricted to the baseline n_min=1, where the deepest key always answers
    a training atom -- it contributed to that key -- so there is no backoff
    and the walk collapses to a single level. Generalizing would mean
    implementing prefix backoff over the key tables, i.e. analysing a method
    nobody runs (spec section 2)."""
    if n_min != 1:
        raise NotImplementedError(
            f"analytic HOSE statistics assume the baseline n_min=1, got "
            f"{n_min}: at a higher threshold a training atom backs off to "
            f"its own sphere prefix, which this does not implement"
        )
    if not state.has_second_moment:
        raise ValueError(
            "this HOSE state predates the sumsq column and carries no second "
            "moment, so its training error is not recoverable -- refit to use "
            "the analytic curve"
        )
    table = state.tables[radius]
    n = np.array([c for _, _, c in table.values()], dtype=np.float64)
    s = np.array([v for v, _, _ in table.values()], dtype=np.float64)[:, None]
    q = np.array([qq for _, qq, _ in table.values()], dtype=np.float64)[:, None]
    ...
```

Reuse `_sse_against` with `value = s / n[:, None]`, and `_loo_scale(n)` under `loo` (skipping keys with `n < 2`, which back off to nothing and are answered by the global mean). Build a `TrainStats` with `depth=radius`, `eta_squared` from the same within/total split, and the global moments from `state.global_sum`/`global_sumsq`/`global_count`.

`hose_curve(state, radii, ...)` is the list comprehension over radii; no truncation is involved, because a state carries every radius up to its own.

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest experiments/tests/test_analytic.py -q
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "feat(experiments): analytic training curve for the HOSE baseline"
```

---

### Task 6: CV records it, and the cross-check against `--score-train`

**Files:**
- Modify: `experiments/experiments/cv.py`
- Test: `experiments/tests/test_cv.py`

**Interfaces:**
- Consumes: `sieve_train_stats`, `hose_train_stats`
- Produces: `_write_cv_run(..., analytic_model=None)` adding `train/{rmse,r2,eta2,matched_fraction}` and `train/frac_support_lt_*` to `run_metrics`

- [ ] **Step 1: Write the failing test**

The decisive one — the analytic numbers must equal what `--score-train` records:

```python
def test_analytic_train_metrics_equal_what_score_train_records():
    """The cross-check the diagnostics note argues for. The analytic curve is
    an identity on the fitted statistics, so if predict ever disagreed with
    what the fit stored, the analytic curve would agree with the bug."""
    from experiments.analytic import sieve_train_stats
    from experiments.tests.test_analytic import _brute_force_sse, _fit

    model, batch = _fit(depth=3)
    stats = sieve_train_stats(model)
    recorded_rmse = np.sqrt(_brute_force_sse(model, batch) / batch.n_nodes)
    assert stats.rmse == pytest.approx(recorded_rmse, rel=1e-12)
```

and one that the metrics actually reach `metrics.json` — follow `test_cv.py`'s existing pattern for driving a small CV run and reading the written run directory.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest experiments/tests/test_cv.py -q -k analytic`

- [ ] **Step 3: Thread the model into `_write_cv_run`**

Add a keyword-only `analytic_model: Any = None` parameter, and after the `floor/rmse` block (~line 910):

```python
if analytic_model is not None:
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(analytic_model)
    run_metrics.update(
        {
            "train/rmse": stats.rmse,
            "train/r2": stats.r_squared,
            "train/eta2": stats.eta_squared,
            "train/matched_fraction": stats.matched_fraction,
            **{
                f"train/frac_support_lt_{t}": v
                for t, v in stats.support_fractions.items()
            },
        }
    )
```

Order matters: this runs **before** `_score_train`'s own update, so that when `--score-train` is also passed its measured value wins and any disagreement is visible in the run rather than hidden. Add a comment saying so.

In `run_sieve_cv`, pass `analytic_model=predictor._model` (the truncated / respecified model already set on the predictor) at the `_write_cv_run` call site. Do the same in `run_hose_cv` with `hose_train_stats`.

- [ ] **Step 4: Run the CV suite**

Run: `uv run pytest experiments/tests/test_cv.py experiments/tests/test_cv_optional.py -q`
Expected: PASS. Existing runs that did not pass `--score-train` now carry `train/*` keys; if a test asserts an exact metric-key set, update it deliberately and note it in the commit.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "feat(experiments): CV runs record the analytic training metrics"
```

---

### Task 7: Per-conformer reconstruction under `--collapse`

**Files:**
- Modify: `experiments/experiments/collapse.py`, `experiments/experiments/cv.py`, `experiments/experiments/analytic.py`
- Test: `experiments/tests/test_analytic.py`

**Interfaces:**
- Consumes: `collapse_molecule_set`, Task 1's walk
- Produces: `collapse_molecule_set(..., return_within_ss=False)` optionally also returning `float`; `per_conformer_sse(model, weighted_model, within_ss) -> float`

**This task is separable.** Tasks 1–6 stand on their own; if it proves heavier than it is worth, stop after Task 6 and record why.

The identity (spec section 5):

```
SSE_atoms = sum W_mp  +  sum k_mp * (ubar_mp - v)^2
```

The second term needs k-weighted moments scored against the **unweighted** fit's values. The partition is purely structural, so a `weight_by_collapse=True` fit has identical class *signatures* — but **not** identical class *indices*, because Sieve discovers its vocabulary per fit and remaps on merge. The two must be aligned by signature, with `sieve.merge._lookup_rows`, not by position. Assert the alignment is total (every signature found) rather than assuming it.

- [ ] **Step 1: Write the failing test**

```python
def test_per_conformer_sse_reconstructs_the_uncollapsed_training_error():
    """The reconstruction must reproduce, exactly, the training SSE of a fit
    on the uncollapsed set scored against the collapsed fit's predictions."""
    ...
```

Build a `MoleculeSet` with a known duplicate group, fit uncollapsed and collapsed, and assert the reconstruction equals brute-force predicting the uncollapsed training set with the *collapsed* model. Use `experiments/tests/test_collapse.py`'s own fixtures for the duplicate construction.

- [ ] **Step 2: Return the within-key sum of squares from the collapse**

`collapse_molecule_set` already stacks each key's members to take their mean; `((stacked - stacked.mean(axis=0)) ** 2).sum()` accumulated over keys is `sum W_mp` at no extra cost. Return it behind `return_within_ss=True` so no existing caller changes.

- [ ] **Step 3: Implement `per_conformer_sse`**

Align the two models' classes by signature, then run the Task 1 walk with `v` from the unweighted model and `(n, S, Q)` from the weighted one, and add `within_ss`.

- [ ] **Step 4: Wire it into the shard fits and the run**

`fit_sieve_shard` under `--collapse` saves a companion `tree_stats_weighted.npz`; `run_sieve_cv` merges and truncates it alongside the main one and passes both to `_write_cv_run`, which records `train/rmse_per_conformer` and `train/n_units`.

**Check the cost before committing to this.** It doubles shard-fit storage and merge time. If that is not acceptable, record the measurement and stop — `train/rmse` in collapsed units is still correct and still recorded.

- [ ] **Step 5: Run, lint, commit**

---

### Task 8: Workflow, docs, and the note

**Files:**
- Modify: `experiments/workflows/cv_charges.sh`, `experiments/README.md`, `experiments/docs/analytic-diagnostics-and-conformer-collapse.md`

- [ ] **Step 1: Drop `--score-train` from the Sieve and HOSE study steps**

In `cv_charges.sh` around line 429, `SCORE_TRAIN_FLAG="--score-train"` becomes predictor-conditional. **The DASH steps keep it** — spec section 2 makes it DASH's only route to a training curve. Add a comment saying exactly that, so the asymmetry is not later "tidied up".

- [ ] **Step 2: Document the command in `experiments/README.md`**

A short section after "Collecting results", mirroring the "Collapsing equivalent molecules" section's shape: what it computes, the one-line invocation, that it reads a state and nothing else, and that DASH is excluded with a pointer to the spec.

- [ ] **Step 3: Update the diagnostics note**

In `analytic-diagnostics-and-conformer-collapse.md`, section 7 lists these as things the finding "enables". Change the tense: they are now implemented, name the command, and record the DASH measurement from spec section 2 as the reason DASH is not among them. Move the leave-one-cluster-out item to section 8 (Open) if it is not already there.

- [ ] **Step 4: Full suite, lint, commit**

```bash
uv run pytest -q
uv run ruff check && uv run ruff format --check && uv run ty check
git commit -am "docs(experiments): analytic curves replace --score-train except for DASH"
```

- [ ] **Step 5: Finish the branch**

**REQUIRED SUB-SKILL:** Use superpowers:finishing-a-development-branch.
