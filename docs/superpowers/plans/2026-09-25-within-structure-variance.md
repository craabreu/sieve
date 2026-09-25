# Within-Structure Variance, Phase 1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pooled within-structure variance σ²_w to `uncertainty.predictive_variance` ("form B"), carried from conformer collapse through the adapter, the batch and the model, and change `ALPHA_V` from 30 to 10.

**Architecture:** `collapse_molecule_set` writes each representative atom's summed squared deviation from its orbit mean and its member count as two companion atom properties. `from_rdkit(..., within_from_atom_prop=...)` reads them into two optional `NodeBatch` fields. `fit` sums them into two additive `SieveModel` statistics, `merge_models` adds them, and `save`/`load` persist them under an optional `within` key. `predictive_variance` and the unmatched-node fallback in `predict` add σ²_w = `within_sse / within_n`. Cached CV fold models, fitted without the statistics, receive them from the floor cache of their training shards through `SieveModel.with_within_structure`.

**Tech Stack:** Python 3.11+, NumPy, RDKit, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-25-within-structure-variance-design.md` (phase 1 = §3; phase 2, §4, is not in this plan).

## Global Constraints

- One PR: this branch, `within-structure-variance-spec` (PR #47), carries the spec, this plan and the code.
- Form B: σ²ᵢ = within-class(shrunk, α^v) + a·selection(α^t) + mean-estimation + σ²_w. The three existing terms and their estimators are unchanged.
- `ALPHA_V = 10.0`; `ALPHA_T = 1.0` and `SELECTION_WEIGHT = 0.5` unchanged.
- σ²_w = `within_sse / within_n` per target dimension, and 0 when `within_n == 0`.
- Companion atom properties: `<atom_property>__within_sse` and `<atom_property>__within_n`. The suffixes are the constants `WITHIN_SSE_SUFFIX = "__within_sse"` and `WITHIN_N_SUFFIX = "__within_n"` in `sieve.io.rdkit_adapter`; `experiments.collapse` imports them.
- Under `weight_by_collapse`, each of the `n` copies carries `sse / n` and `1`, so the sums are invariant.
- `NodeBatch.within_sse` is `(n_nodes, d)` float64 and `within_n` is `(n_nodes,)` float64; both set or both `None`; finite, non-negative, and `within_n ≥ 1` wherever any SSE entry is positive.
- `SieveModel.within_sse: np.ndarray | None = None` (`(d,)`) and `within_n: float = 0.0`. They are statistics, not configuration: `schema_version` does not change, and a model with them merges with one without.
- A saved file carries a `within` array `[within_n, *within_sse]` only when `within_sse is not None`; files without it load with `within_sse=None`, `within_n=0.0` and save back without it. No `FORMAT_VERSION` change.
- A model with no within-structure statistics behaves exactly as today apart from the α^v default.
- Gate: `.venv/bin/ruff check src tests experiments`, `.venv/bin/ruff format --check src tests experiments`, `.venv/bin/ty check src tests experiments` (two known `assert_array_equal` diagnostics), `.venv/bin/python -m pytest -q`. Tests that need pandas use `pytest.importorskip("pandas")`.
- The test split is never read. The reproduction check (Task 6) uses repeat 0, folds 0 and 1 of the train-split CV only.

## Review Focus

- **A model that crossed an older code path loses the statistics silently.** `truncate_model` rebuilds `SieveModel` positionally; every such constructor must carry `within_sse`/`within_n`. Pinned in Task 5 (`test_truncate_model_carries_the_within_structure_statistics`).
- **Chunked fitting (`config.chunk_size`) slices the batch and folds.** The sums must equal an unchunked fit's. Pinned in Task 3 (`test_chunked_fit_sums_the_within_statistics`).
- **Parallel featurisation (`n_jobs > 1`) ships molecules as blobs.** The companion properties must survive serialisation, and the result must equal the sequential path. Pinned in Task 3 (`test_within_properties_survive_parallel_featurisation`).
- **A training set with the companion properties on some molecules but not others.** It must fail loudly, not fit a silently partial σ²_w. Pinned in Task 3 (`test_a_missing_within_property_is_refused`).
- **A cached fold model whose training shards are missing from the floor cache.** It must keep σ²_w = 0 and log a warning, not guess from a partial sum. Pinned in Task 5 (`test_training_floor_needs_every_training_shard`).

---

### File map

| File | Change |
|---|---|
| `src/sieve/model.py` | `within_sse`, `within_n`, `within_variance`, `with_within_structure`; `fit` sums; save/load |
| `src/sieve/merge.py` | `merge_models` adds the sums |
| `src/sieve/uncertainty.py` | `ALPHA_V = 10.0`; σ²_w term; docstring |
| `src/sieve/predict.py` | unmatched fallback `global_msd + within_variance` |
| `src/sieve/batch.py` | `within_sse`/`within_n` fields: validation, slicing, concat |
| `src/sieve/io/rdkit_adapter.py` | suffix constants; `within_from_atom_prop` |
| `experiments/experiments/collapse.py` | attach the companion properties |
| `experiments/experiments/predictors/sieve_predictor.py` | `_batch_for` reads them when present |
| `experiments/experiments/cv.py` | `truncate_model` carries them; `_with_training_floor` for fold models |
| `tests/test_within_structure.py` (new) | model, merge, persistence, variance, batch, adapter, fit |
| `experiments/tests/test_collapse.py`, `experiments/tests/test_cv.py`, `experiments/tests/test_predictor_sieve.py` | collapse sums, CV attachment, predictor wiring |

---

### Task 1: The model's within-structure statistics

**Files:**
- Modify: `src/sieve/model.py` (dataclass fields, new property and method, `save`, `load`)
- Modify: `src/sieve/merge.py:305-313` (`merge_models` return statements)
- Test: `tests/test_within_structure.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `SieveModel.within_sse: np.ndarray | None` (`(d,)`, default `None`), `SieveModel.within_n: float` (default `0.0`), as the last two dataclass fields, after `global_msd`.
  - `SieveModel.within_variance -> np.ndarray` (property, `(d,)`, zeros when `within_n == 0` or `within_sse is None`).
  - `SieveModel.with_within_structure(sse, n) -> SieveModel`; `sse` is a scalar or `(d,)` array-like, `n` a non-negative number.
  - `merge_models(a, b).within_sse == a + b` (a `None` side counts as zeros; both `None` stays `None`), `.within_n == a.within_n + b.within_n`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_within_structure.py`:

```python
"""Within-structure variance (within-structure-variance spec, phase 1).

Collapse removes each structure's conformer-to-conformer scatter from the fit;
these two additive statistics carry it back, so the predictive variance can
add it to every class.
"""

from __future__ import annotations

import numpy as np
import pytest

import sieve
from sieve.model import SieveModel
from tests.helpers import chain_batch, simple_config


def _fitted(**kw):
    return sieve.fit(chain_batch(10, graphs=3), simple_config(**kw))


# ----------------------------------------------------------------- model --


def test_a_model_without_statistics_has_zero_within_variance():
    m = _fitted()
    assert m.within_sse is None
    assert m.within_n == 0.0
    np.testing.assert_array_equal(m.within_variance, np.zeros(1))


def test_with_within_structure_sets_the_sums():
    m = _fitted().with_within_structure(3.0e-3, 30)
    np.testing.assert_array_equal(m.within_sse, [3.0e-3])
    assert m.within_n == 30.0
    np.testing.assert_allclose(m.within_variance, [1.0e-4])


@pytest.mark.parametrize(
    ("sse", "n"), [(-1.0, 3), (np.nan, 3), (1.0, -2), (1.0, np.inf), (1.0, 0)]
)
def test_with_within_structure_refuses_impossible_sums(sse, n):
    with pytest.raises(ValueError):
        _fitted().with_within_structure(sse, n)


def test_with_within_structure_changes_neither_predictions_nor_the_digest():
    cfg = simple_config(max_wl_depth=2)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(1.0, 10)
    np.testing.assert_array_equal(sieve.predict(w, batch), sieve.predict(m, batch))
    assert w.config.schema_version == m.config.schema_version


def test_merge_adds_the_sums():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted().with_within_structure(1.0, 5)
    ab = a.merge(b)
    np.testing.assert_allclose(ab.within_sse, [3.0])
    assert ab.within_n == 15.0


def test_merge_with_a_model_without_statistics_keeps_the_other_side():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted()
    for merged in (a.merge(b), b.merge(a)):
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


def test_merge_of_two_models_without_statistics_has_none():
    ab = _fitted().merge(_fitted())
    assert ab.within_sse is None
    assert ab.within_n == 0.0


def test_the_empty_model_is_still_the_merge_identity():
    a = _fitted().with_within_structure(2.0, 10)
    e = SieveModel.empty(a.config)
    for merged in (a.merge(e), e.merge(a)):
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


# ----------------------------------------------------------- persistence --


def test_save_load_round_trips_the_sums(tmp_path):
    m = _fitted().with_within_structure(2.5e-3, 25)
    path = tmp_path / "m.npz"
    m.save(path)
    back = SieveModel.load(path)
    np.testing.assert_array_equal(back.within_sse, m.within_sse)
    assert back.within_n == m.within_n


def test_a_file_without_statistics_loads_and_saves_back_without_them(tmp_path):
    m = _fitted()
    path = tmp_path / "m.npz"
    m.save(path)
    assert "within" not in np.load(path).files
    back = SieveModel.load(path)
    assert back.within_sse is None
    assert back.within_n == 0.0
    again = tmp_path / "again.npz"
    back.save(again)
    assert sorted(np.load(again).files) == sorted(np.load(path).files)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py -q`
Expected: FAIL with `AttributeError: 'SieveModel' object has no attribute 'within_sse'`.

- [ ] **Step 3: Implement the fields, the property and the method**

In `src/sieve/model.py`, after `global_msd: np.ndarray` in the dataclass:

```python
    # Within-structure statistics (within-structure-variance spec 3.3): the
    # summed squared deviation of each training atom from its structure's
    # orbit mean over conformers, and the number of atoms it was summed over.
    # Collapse removes this scatter from the fit; these carry it back so the
    # predictive variance can add it. Statistics, not configuration: they are
    # outside schema_version, and they add under merge.
    within_sse: np.ndarray | None = None  # (d,)
    within_n: float = 0.0

    @property
    def within_variance(self) -> np.ndarray:
        """σ²_w per target dimension, and 0 when no statistics were carried."""
        d = self.config.target_dim
        if self.within_sse is None or self.within_n == 0:
            return np.zeros(d)
        return self.within_sse / self.within_n

    def with_within_structure(self, sse, n) -> SieveModel:
        """A copy carrying the given within-structure sums.

        For models fitted without them -- the cached CV fold models -- whose
        training shards' sums are known from elsewhere (spec 3.4). Predictions
        and ``schema_version`` are unchanged; only the predictive variance
        reads them.
        """
        d = self.config.target_dim
        sse = np.broadcast_to(np.asarray(sse, np.float64), (d,)).copy()
        n = float(n)
        if not (np.isfinite(sse).all() and np.isfinite(n)):
            raise ValueError("within-structure sums must be finite")
        if (sse < 0).any() or n < 0:
            raise ValueError("within-structure sums must be non-negative")
        if n == 0 and (sse > 0).any():
            raise ValueError("a positive within-structure SSE needs a positive count")
        return replace(self, within_sse=sse, within_n=n)
```

- [ ] **Step 4: Implement persistence**

In `SieveModel.save`, after the `arrays = {...}` literal and before the level loop:

```python
        if self.within_sse is not None:
            # Only when present, so a file without the statistics keeps exactly
            # the keys it always had.
            arrays["within"] = np.concatenate([[self.within_n], self.within_sse])
```

In `SieveModel.load`, replace the final `return` with:

```python
        within_sse, within_n = None, 0.0
        if "within" in data.files:
            w = data["within"]
            within_sse, within_n = w[1 : 1 + d], float(w[0])
        return cls(
            cfg,
            levels,
            int(g[0]),
            g[1 : 1 + d],
            g[1 + d : 1 + 2 * d],
            within_sse,
            within_n,
        )
```

- [ ] **Step 5: Implement the merge**

In `src/sieve/merge.py`, add above `merge_models`:

```python
def _merge_within(a, b) -> tuple[np.ndarray | None, float]:
    """The within-structure sums add; a side without them counts as zero."""
    if a.within_sse is None and b.within_sse is None:
        return None, 0.0
    d = a.config.target_dim
    sa = np.zeros(d) if a.within_sse is None else a.within_sse
    sb = np.zeros(d) if b.within_sse is None else b.within_sse
    return sa + sb, float(a.within_n) + float(b.within_n)
```

and replace the two `return SieveModel(...)` lines at the end of `merge_models` with:

```python
    within_sse, within_n = _merge_within(a, b)
    nA, nB = float(a.global_count), float(b.global_count)
    n = nA + nB
    if n == 0:
        return SieveModel(
            cfg, tuple(levels), 0, a.global_mean, a.global_msd, within_sse, within_n
        )
    wA, wB = nA / n, nB / n
    delta = b.global_mean - a.global_mean
    g_msd = wA * a.global_msd + wB * b.global_msd + wA * wB * delta * delta
    g_mean = wA * a.global_mean + wB * b.global_mean
    return SieveModel(
        cfg, tuple(levels), int(n), g_mean, g_msd, within_sse, within_n
    )
```

(The existing `nA, nB = ...` line moves below the new `_merge_within` call; nothing else in the function changes.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py tests/test_merge.py tests/test_io.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/sieve/model.py src/sieve/merge.py tests/test_within_structure.py
git commit -m "feat(model): within-structure sums, merged and persisted"
```

---

### Task 2: Form B in the predictive variance, and α^v = 10

**Files:**
- Modify: `src/sieve/uncertainty.py` (module docstring, `ALPHA_V` and its comment, the `out.append` line)
- Modify: `src/sieve/predict.py:231-235` (the unmatched fallback)
- Test: `tests/test_within_structure.py` (append), `tests/test_uncertainty.py:190` needs no edit (it reads `ALPHA_V` by name)

**Interfaces:**
- Consumes: `SieveModel.within_variance` and `with_within_structure` (Task 1).
- Produces: `predictive_variance(model, ...)` returns, per level, today's three terms plus `model.within_variance` broadcast over classes; `predict_detailed(...).predictive_variance` for an unmatched node is `global_msd + within_variance`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_within_structure.py`:

```python
# ------------------------------------------------------ predictive variance --


def test_alpha_v_is_ten():
    from sieve.uncertainty import ALPHA_T, ALPHA_V, SELECTION_WEIGHT

    assert (ALPHA_V, ALPHA_T, SELECTION_WEIGHT) == (10.0, 1.0, 0.5)


def test_predictive_variance_adds_the_within_variance_exactly():
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    w = m.with_within_structure(4.0e-2, 100)
    for base, form_b in zip(predictive_variance(m), predictive_variance(w), strict=True):
        np.testing.assert_array_equal(form_b, base + 4.0e-4)


def test_without_statistics_the_variance_is_the_three_terms():
    """σ²_w = 0 adds nothing: bit-identical to an explicit zero."""
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    z = m.with_within_structure(0.0, 0)
    for a, b in zip(predictive_variance(m), predictive_variance(z), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unmatched_nodes_fall_back_to_global_msd_plus_the_within_variance():
    from sieve.batch import NodeBatch

    cfg = simple_config(max_wl_depth=1, predictive_variance=True)
    m = sieve.fit(chain_batch(10, graphs=2), cfg).with_within_structure(1.0, 10)
    oov = NodeBatch(
        node_attrs=np.array([[7]], np.int64),  # an element code never fitted
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attrs=np.zeros((0, 1), np.int64),
        graph_id=np.zeros(1, np.int64),
    )
    out = sieve.predict_detailed(m, oov)
    assert out.matched_level[0] == -1
    assert out.predictive_variance is not None
    np.testing.assert_allclose(out.predictive_variance[0], m.global_msd + 0.1)


def test_matched_nodes_read_form_b():
    cfg = simple_config(max_wl_depth=2, predictive_variance=True)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(2.0, 10)
    base = sieve.predict_detailed(m, batch).predictive_variance
    form_b = sieve.predict_detailed(w, batch).predictive_variance
    assert base is not None and form_b is not None
    np.testing.assert_allclose(form_b, base + 0.2, rtol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py -q -k "alpha_v or predictive or unmatched or matched or three_terms"`
Expected: FAIL: `test_alpha_v_is_ten` (30.0 ≠ 10.0), `adds_the_within_variance_exactly` and both fallback tests (the variance is unchanged).

- [ ] **Step 3: Implement**

In `src/sieve/uncertainty.py`:

1. Replace `ALPHA_V = 30.0  # within-class shrinkage toward the level-pooled variance` and the comment block above the three constants with:

```python
# ALPHA_T and SELECTION_WEIGHT were selected on the val split of
# dash-molecules-10fold-1 by Gaussian NLL over a 200-point grid. ALPHA_V was
# re-selected on 2026-09-25, with the within-structure term in place, by
# rotating over the five folds of the Study B incumbent's repeat 0 (chosen on
# four, scored on the fifth): 10 in every fold, by both NLL and normalised
# RMSE, against the earlier 30 (within-structure-variance spec 1).
#
# Deliberately module constants rather than SieveConfig fields. The objective
# is flat -- the gain of 10 over 30 is 0.006 in NLL -- so exposing them as
# knobs would invite tuning that the measurement says cannot pay. They are
# named, not magic, and a caller who genuinely needs to sweep can pass them to
# `predictive_variance`.
ALPHA_V = 10.0  # within-class shrinkage toward the level-pooled variance
```

2. In the module docstring, replace "Three terms, each with a distinct reason to exist:" and the display with:

```
Four terms, each with a distinct reason to exist:

.. math::

    \sigma^2_{\mathrm{pred}} =
      \underbrace{\frac{(N_c-1)s^2_c + \alpha^v \bar\sigma^2_k}
                       {(N_c-1) + \alpha^v}}_{\text{within-class, shrunk}}
    + a \underbrace{\frac{(C_c-1)\tau^2_c + \alpha^t \hat\tau^2_k}
                         {(C_c-1) + \alpha^t}}_{\text{selection}}
    + \underbrace{\hat\tau^2_{\mathrm{pa}(k)}(1 - w_c)}_{\text{mean estimation}}
    + \underbrace{\sigma^2_w}_{\text{within structure}}
```

and add, after the "mean estimation" paragraph:

```
**within structure** is the scatter collapse removed from the fit: the
expected squared deviation of one conformer's charge from its structure's
mean over conformers and symmetry-equivalent atoms
(``SieveModel.within_variance``). Training classes hold one row per structure,
so none of the three terms above can see it, while every held-out atom is an
individual conformer and carries it. Without it the variance was five times
overconfident at the deepest radius (within-structure-variance spec 1). Zero
for a model that carries no within-structure statistics.
```

3. Before the `for k, lvl in enumerate(model.levels):` loop add `sigma2_w = model.within_variance`, and replace the loop's last line with:

```python
        out.append(within + selection_weight * selection + estimation + sigma2_w)
```

In `src/sieve/predict.py`, replace the fallback comment and line:

```python
        # global_msd for a node that matched nothing: there is no class row to
        # index, exactly as `value` falls back to global_mean. The
        # within-structure variance is added here as it is to every class
        # (uncertainty.py), since an unmatched conformer carries it too.
        table = predictive_variance(model)
        fallback = model.global_msd + model.within_variance
        pred_var = np.broadcast_to(fallback, (n, d)).astype(np.float64).copy()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py tests/test_uncertainty.py tests/test_predict.py -q`
Expected: PASS. If a test in `tests/test_uncertainty.py` hard-codes a number computed at α^v = 30, pass `alpha_v=30.0` to that test's call rather than changing the expected number, and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/sieve/uncertainty.py src/sieve/predict.py tests/test_within_structure.py
git commit -m "feat(uncertainty): add the within-structure variance; alpha_v 30 -> 10"
```

---

### Task 3: The batch, the adapter and the fit carry the statistics

**Files:**
- Modify: `src/sieve/batch.py` (fields, `_check_shapes`, new `_check_within`, `__post_init__`, `__getitem__`, `concat_batches`)
- Modify: `src/sieve/io/rdkit_adapter.py` (suffix constants; `within_from_atom_prop` through `from_rdkit`, `_from_rdkit_worker`, `_from_rdkit_sequential`)
- Modify: `src/sieve/model.py` (`fit` sums the batch arrays)
- Test: `tests/test_within_structure.py` (append)

**Interfaces:**
- Consumes: `SieveModel(..., within_sse, within_n)` positional order (Task 1).
- Produces:
  - `NodeBatch.within_sse: np.ndarray | None` (`(n_nodes, d)`), `NodeBatch.within_n: np.ndarray | None` (`(n_nodes,)`), the last two dataclass fields.
  - `sieve.io.rdkit_adapter.WITHIN_SSE_SUFFIX = "__within_sse"`, `WITHIN_N_SUFFIX = "__within_n"`.
  - `from_rdkit(mols, y=None, *, config, node_order=None, y_from_atom_prop=None, within_from_atom_prop: str | None = None, n_jobs=None)`.
  - `fit(batch, config).within_sse == batch.within_sse.sum(axis=0)`, `.within_n == batch.within_n.sum()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_within_structure.py`:

```python
# ----------------------------------------------------------------- batch --


def _with_within(batch, sse, n):
    from sieve.batch import NodeBatch

    return NodeBatch(**{**batch.__dict__, "within_sse": sse, "within_n": n})


def _within_batch(n=10, graphs=3, seed=0):
    b = chain_batch(n, graphs=graphs, seed=seed)
    rng = np.random.default_rng(seed + 100)
    counts = rng.integers(1, 5, size=b.n_nodes).astype(np.float64)
    sse = rng.uniform(0.0, 1e-3, size=(b.n_nodes, 1)) * counts[:, None]
    return _with_within(b, sse, counts)


def test_a_batch_carries_the_within_arrays_through_slicing():
    b = _within_batch()
    mask = b.graph_id != 1
    sub = b[mask]
    assert sub.within_sse is not None and sub.within_n is not None
    assert b.within_sse is not None and b.within_n is not None
    np.testing.assert_array_equal(sub.within_sse, b.within_sse[mask])
    np.testing.assert_array_equal(sub.within_n, b.within_n[mask])


def test_concat_joins_the_within_arrays():
    from sieve.batch import concat_batches

    a, b = _within_batch(seed=0), _within_batch(seed=1)
    ab = concat_batches([a, b])
    assert ab.within_sse is not None and ab.within_n is not None
    np.testing.assert_array_equal(
        ab.within_sse, np.concatenate([a.within_sse, b.within_sse])
    )
    np.testing.assert_array_equal(
        ab.within_n, np.concatenate([a.within_n, b.within_n])
    )


def test_concat_refuses_within_arrays_on_some_parts_only():
    from sieve.batch import concat_batches

    with pytest.raises(ValueError, match="within"):
        concat_batches([_within_batch(), chain_batch(10, graphs=3)])


@pytest.mark.parametrize(
    "bad",
    [
        {"sse": -1e-4},  # negative SSE
        {"sse": np.nan},  # not finite
        {"n": -1.0},  # negative count
        {"n": 0.0},  # positive SSE with no members
    ],
)
def test_impossible_within_arrays_are_refused(bad):
    b = chain_batch(4)
    sse = np.full((b.n_nodes, 1), bad.get("sse", 1e-4))
    n = np.full(b.n_nodes, bad.get("n", 2.0))
    with pytest.raises(ValueError, match="within"):
        _with_within(b, sse, n)


def test_within_arrays_must_come_together_and_match_the_node_count():
    b = chain_batch(4)
    with pytest.raises(ValueError, match="within"):
        _with_within(b, np.zeros((b.n_nodes, 1)), None)
    with pytest.raises(ValueError, match="within"):
        _with_within(b, np.zeros((b.n_nodes - 1, 1)), np.ones(b.n_nodes - 1))


def test_a_zero_count_is_allowed_where_the_sse_is_zero():
    b = chain_batch(4)
    _with_within(b, np.zeros((b.n_nodes, 1)), np.zeros(b.n_nodes))


# ------------------------------------------------------------------- fit --


def test_fit_sums_the_batch_within_arrays():
    b = _within_batch()
    m = sieve.fit(b, simple_config(max_wl_depth=2))
    assert m.within_sse is not None and b.within_sse is not None
    assert b.within_n is not None
    np.testing.assert_allclose(m.within_sse, b.within_sse.sum(axis=0), rtol=1e-12)
    assert m.within_n == pytest.approx(float(b.within_n.sum()), rel=1e-12)
    # statistics, not configuration: the digest and every class are unchanged
    plain = sieve.fit(chain_batch(10, graphs=3), simple_config(max_wl_depth=2))
    assert m.config.schema_version == plain.config.schema_version
    np.testing.assert_array_equal(sieve.predict(m, b), sieve.predict(plain, b))


def test_merge_of_two_fits_equals_the_fit_of_their_union():
    """The merge monoid (design.md 5.4), for the new statistics."""
    from tests.helpers import split_batch

    b = _within_batch(graphs=4)
    cfg = simple_config(max_wl_depth=2)
    mask = b.graph_id < 2
    merged = sieve.fit(split_batch(b, mask), cfg).merge(
        sieve.fit(split_batch(b, ~mask), cfg)
    )
    whole = sieve.fit(b, cfg)
    np.testing.assert_allclose(merged.within_sse, whole.within_sse, rtol=1e-12)
    assert merged.within_n == pytest.approx(whole.within_n, rel=1e-12)


def test_chunked_fit_sums_the_within_statistics():
    b = _within_batch(graphs=6)
    whole = sieve.fit(b, simple_config(max_wl_depth=2))
    chunked = sieve.fit(b, simple_config(max_wl_depth=2, chunk_size=12))
    np.testing.assert_allclose(chunked.within_sse, whole.within_sse, rtol=1e-12)
    assert chunked.within_n == pytest.approx(whole.within_n, rel=1e-12)


def test_a_batch_without_within_arrays_fits_as_before():
    m = sieve.fit(chain_batch(10, graphs=3), simple_config())
    assert m.within_sse is None and m.within_n == 0.0


# --------------------------------------------------------------- adapter --


def _labelled_mols(smiles_list, *, within=True):
    from rdkit import Chem

    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    mols = []
    for j, smi in enumerate(smiles_list):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        for a in m.GetAtoms():
            a.SetDoubleProp("q", 0.01 * a.GetIdx())
            if within:
                a.SetDoubleProp("q" + WITHIN_SSE_SUFFIX, 1e-4 * (a.GetIdx() + j))
                a.SetDoubleProp("q" + WITHIN_N_SUFFIX, float(1 + j))
        mols.append(m)
    return mols


def _adapter_config(mols):
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes

    codes, edges = build_codes(mols, ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=1,
    )


def test_the_adapter_reads_the_companion_properties():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO", "CCN"])
    cfg = _adapter_config(mols)
    b = from_rdkit(mols, config=cfg, y_from_atom_prop="q", within_from_atom_prop="q")
    assert b.within_sse is not None and b.within_n is not None
    n0 = mols[0].GetNumAtoms()
    np.testing.assert_allclose(b.within_sse[:n0, 0], 1e-4 * np.arange(n0))
    np.testing.assert_array_equal(b.within_n[:n0], np.ones(n0))
    np.testing.assert_array_equal(b.within_n[n0:], np.full(b.n_nodes - n0, 2.0))


def test_without_within_from_atom_prop_the_batch_has_none():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO"])
    b = from_rdkit(mols, config=_adapter_config(mols), y_from_atom_prop="q")
    assert b.within_sse is None and b.within_n is None


def test_a_missing_within_property_is_refused():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO"]) + _labelled_mols(["CCN"], within=False)
    with pytest.raises(KeyError):
        from_rdkit(
            mols,
            config=_adapter_config(mols),
            y_from_atom_prop="q",
            within_from_atom_prop="q",
        )


def test_within_properties_survive_parallel_featurisation():
    pytest.importorskip("rdkit")
    from sieve.io.rdkit_adapter import from_rdkit

    mols = _labelled_mols(["CCO", "CCN", "CCC", "OCO", "NCN", "CC=O"])
    cfg = _adapter_config(mols)
    kw = {"config": cfg, "y_from_atom_prop": "q", "within_from_atom_prop": "q"}
    seq = from_rdkit(mols, **kw)
    par = from_rdkit(mols, n_jobs=2, **kw)
    np.testing.assert_array_equal(par.within_sse, seq.within_sse)
    np.testing.assert_array_equal(par.within_n, seq.within_n)
```

`_adapter_config` follows `cfg_for` in `tests/test_rdkit_adapter.py:18`, the idiom the suite already uses.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py -q -k "batch or concat or refused or within_arrays or zero_count or fit or adapter or companion or parallel or missing"`
Expected: FAIL with `TypeError: NodeBatch.__init__() got an unexpected keyword argument 'within_sse'` and `ImportError: cannot import name 'WITHIN_N_SUFFIX'`.

- [ ] **Step 3: Implement the batch fields**

In `src/sieve/batch.py`, after `stereo_centres: ...` in the dataclass:

```python
    # Within-structure statistics (within-structure-variance spec 3.2): each
    # node's summed squared deviation from its structure's orbit mean, and the
    # number of conformers summed over. Set by collapse, read by fit.
    within_sse: np.ndarray | None = None  # (n_nodes, d) float64
    within_n: np.ndarray | None = None  # (n_nodes,) float64
```

Add `self._check_within()` as the last call in `__post_init__`. At the end of `_check_shapes` (which also runs on the trusted path):

```python
        if (self.within_sse is None) != (self.within_n is None):
            raise ValueError("within_sse and within_n must be set together")
        if self.within_sse is not None and self.within_n is not None:
            if self.within_sse.ndim != 2 or self.within_sse.shape[0] != n:
                raise ValueError(
                    f"within_sse must have shape ({n}, d), got {self.within_sse.shape}"
                )
            if self.y is not None and self.within_sse.shape[1] != self.y.shape[1]:
                raise ValueError("within_sse must have as many columns as y")
            if self.within_n.shape != (n,):
                raise ValueError(
                    f"within_n must have shape ({n},), got {self.within_n.shape}"
                )
```

and a new method beside the other checks:

```python
    def _check_within(self) -> None:
        """The within-structure sums are finite, non-negative, and a positive
        SSE has at least one member behind it."""
        if self.within_sse is None or self.within_n is None:
            return
        if not (np.isfinite(self.within_sse).all() and np.isfinite(self.within_n).all()):
            raise ValueError("within_sse and within_n must be finite")
        if (self.within_sse < 0).any() or (self.within_n < 0).any():
            raise ValueError("within_sse and within_n must be non-negative")
        if ((self.within_sse > 0).any(axis=1) & (self.within_n < 1)).any():
            raise ValueError("a positive within_sse needs within_n >= 1")
```

In `__getitem__`, add to the `_with_trusted_edges(...)` call:

```python
            within_sse=None if self.within_sse is None else self.within_sse[sel],
            within_n=None if self.within_n is None else self.within_n[sel],
```

In `concat_batches`, after the `has_stereo_centres` check:

```python
    has_within = {p.within_sse is not None for p in parts}
    if len(has_within) > 1:
        raise ValueError("within_sse is set on some but not all parts")
```

and add to its `_with_trusted_edges(...)` call:

```python
        within_sse=(
            np.concatenate([p.within_sse for p in parts], axis=0)
            if has_within == {True}
            else None
        ),
        within_n=(
            np.concatenate([p.within_n for p in parts]) if has_within == {True} else None
        ),
```

(`ty` may not narrow `p.within_sse` inside the comprehension; if it complains, add `# ty: ignore[...]` with the exact code it prints, as the file already does elsewhere, or build the lists with an explicit `assert p.within_sse is not None` loop.)

- [ ] **Step 4: Implement the adapter**

In `src/sieve/io/rdkit_adapter.py`, near the top-level constants:

```python
# Companion atom properties written by conformer collapse beside the target
# (within-structure-variance spec 3.1): `<target>__within_sse` and
# `<target>__within_n`.
WITHIN_SSE_SUFFIX = "__within_sse"
WITHIN_N_SUFFIX = "__within_n"
```

Add `within_from_atom_prop: str | None = None` as a keyword to `_from_rdkit_sequential` (after `y_from_atom_prop`), as a positional parameter to `_from_rdkit_worker` (after `y_from_atom_prop`, passed through), and as a keyword to `from_rdkit` (after `y_from_atom_prop`), passed to both the sequential call and the `delayed(_from_rdkit_worker)(blobs, y_chunk, config, order_chunk, y_from_atom_prop, within_from_atom_prop)` call. Document it in `from_rdkit`'s docstring:

```
    ``within_from_atom_prop`` names the target property whose collapse
    companions (``WITHIN_SSE_SUFFIX``, ``WITHIN_N_SUFFIX``) are read into
    ``within_sse``/``within_n``; every atom must carry both, or RDKit's
    ``GetDoubleProp`` raises ``KeyError``.
```

In `_from_rdkit_sequential`, beside `y_out`:

```python
    w_sse = w_n = None
    if within_from_atom_prop is not None:
        w_sse = np.zeros((n, 1), np.float64)
        w_n = np.zeros(n, np.float64)
        sse_prop = within_from_atom_prop + WITHIN_SSE_SUFFIX
        n_prop = within_from_atom_prop + WITHIN_N_SUFFIX
```

in the per-atom loop, after the `y_out` line:

```python
            if w_sse is not None and w_n is not None:
                w_sse[g, 0] = a.GetDoubleProp(sse_prop)
                w_n[g] = a.GetDoubleProp(n_prop)
```

and in the returned `NodeBatch(...)`, add `within_sse=w_sse, within_n=w_n,`.

- [ ] **Step 5: Implement the fit**

In `src/sieve/model.py`, replace the last line of `fit`:

```python
    n, mean, msd = global_stats(batch.y)
    within_sse, within_n = None, 0.0
    if batch.within_sse is not None and batch.within_n is not None:
        within_sse = batch.within_sse.sum(axis=0)
        within_n = float(batch.within_n.sum())
    return SieveModel(config, levels, n, mean, msd, within_sse, within_n)
```

(The chunked branch needs nothing: `batch[mask]` slices the arrays, and `fold` merges the sums.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_structure.py tests/test_batch.py tests/test_rdkit_adapter.py tests/test_fit.py tests/test_sklearn.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/sieve/batch.py src/sieve/io/rdkit_adapter.py src/sieve/model.py tests/test_within_structure.py
git commit -m "feat(batch): carry within-structure sums from the adapter to the fit"
```

---

### Task 4: Collapse attaches the companion properties; the predictor reads them

**Files:**
- Modify: `experiments/experiments/collapse.py:319-339` (`collapse_molecule_set`'s loop)
- Modify: `experiments/experiments/predictors/sieve_predictor.py:168-186` (`_batch_for`)
- Test: `experiments/tests/test_collapse.py` (append), `experiments/tests/test_predictor_sieve.py` (append)

**Interfaces:**
- Consumes: `WITHIN_SSE_SUFFIX`, `WITHIN_N_SUFFIX`, `from_rdkit(..., within_from_atom_prop=...)` (Task 3); `SieveModel.within_sse`/`within_n` (Task 1).
- Produces: every atom of `collapse_molecule_set(mset)`'s molecules carries `<atom_property>__within_sse` and `<atom_property>__within_n`; `SievePredictor.fit` on a collapsed set yields a model whose `within_sse`/`within_n` equal `floor_components(original)["sse"]`/`["n_atoms"]`.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/tests/test_collapse.py`:

```python
# --------------------------------------------------------------------------
# Within-structure statistics carried by the collapsed molecules
# --------------------------------------------------------------------------


def _within_sums(mset):
    import numpy as np
    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    sse = n = 0.0
    for m in mset.mols:
        for a in m.GetAtoms():
            sse += a.GetDoubleProp(mset.atom_property + WITHIN_SSE_SUFFIX)
            n += a.GetDoubleProp(mset.atom_property + WITHIN_N_SUFFIX)
    return np.float64(sse), np.float64(n)


def _multi_conformer_set():
    """Two conformers of ethanol that disagree, a lone propane whose symmetric
    methyl carbons disagree (orbit scatter in a group of one), and a lone
    methanol with no scatter at all."""
    return _floor_set_of(
        [
            _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
            _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
            _charged("CCC", {0: 1.0, 1: 0.5, 2: 2.0}),
            _charged("CO", {0: 0.3, 1: -0.3}),
        ]
    )


def _keyed_like(mset):
    """_floor_set_of numbers every row as its own dash_id; the two ethanols
    must share a collapse key for the test to mean anything."""
    keys = mset.ids["collapse_key"]
    assert keys[0] == keys[1]
    return mset


def test_collapse_attaches_sums_equal_to_the_floor_components():
    from experiments.collapse import collapse_molecule_set, floor_components

    mset = _keyed_like(_multi_conformer_set())
    sse, n = _within_sums(collapse_molecule_set(mset))
    parts = floor_components(mset)
    assert parts["sse"] > 0.0  # or the test is vacuous
    assert sse == pytest.approx(parts["sse"], rel=1e-12)
    assert n == pytest.approx(parts["n_atoms"], rel=1e-12)


def test_weight_by_collapse_leaves_the_sums_unchanged():
    from experiments.collapse import collapse_molecule_set

    mset = _keyed_like(_multi_conformer_set())
    plain = _within_sums(collapse_molecule_set(mset))
    weighted = _within_sums(collapse_molecule_set(mset, weight_by_collapse=True))
    assert weighted[0] == pytest.approx(plain[0], rel=1e-12)
    assert weighted[1] == pytest.approx(plain[1], rel=1e-12)


def test_a_collapsed_atom_carries_its_own_deviations():
    """Ethanol's C0 is 1 and 3 in the two conformers: mean 2, SSE 2, two members."""
    from experiments.collapse import collapse_molecule_set
    from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX

    mset = _keyed_like(_multi_conformer_set())
    out = collapse_molecule_set(mset)
    ethanol = next(m for m in out.mols if m.GetNumAtoms() == 9)
    heavy = [ethanol.GetAtomWithIdx(i) for i in range(3)]
    sse = sorted(a.GetDoubleProp("MBIScharge" + WITHIN_SSE_SUFFIX) for a in heavy)
    assert sse == pytest.approx([2.0, 2.0, 2.0])
    assert all(a.GetDoubleProp("MBIScharge" + WITHIN_N_SUFFIX) == 2.0 for a in heavy)
```

Append to `experiments/tests/test_predictor_sieve.py` (read the file's imports first and reuse its fixtures if they build a `MoleculeSet` with `collapse_key`; otherwise use this one):

```python
def test_a_collapsed_training_set_fits_the_within_structure_sums():
    import numpy as np
    from experiments.collapse import collapse_molecule_set, floor_components
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.test_collapse import _keyed_like, _multi_conformer_set

    mset = _keyed_like(_multi_conformer_set())
    p = SievePredictor(attributes=("element",), edge_attributes=(), max_wl_depth=1)
    p.fit(collapse_molecule_set(mset), mset, rng=np.random.default_rng(0))
    parts = floor_components(mset)
    assert p._model.within_sse is not None
    assert float(p._model.within_sse[0]) == pytest.approx(parts["sse"], rel=1e-12)
    assert p._model.within_n == pytest.approx(parts["n_atoms"], rel=1e-12)


def test_an_uncollapsed_training_set_fits_without_them():
    import numpy as np
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.test_collapse import _multi_conformer_set

    mset = _multi_conformer_set()
    p = SievePredictor(attributes=("element",), edge_attributes=(), max_wl_depth=1)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    assert p._model.within_sse is None
    assert p._model.within_n == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_collapse.py experiments/tests/test_predictor_sieve.py -q -k "within or sums or deviations or collapsed_training or uncollapsed"`
Expected: FAIL with `KeyError` on `MBIScharge__within_sse` (collapse tests) and `within_sse is None` (predictor test).

- [ ] **Step 3: Implement the collapse**

In `experiments/experiments/collapse.py`, import beside the other `sieve` imports (or at the top of `collapse_molecule_set` if the module imports `sieve` lazily):

```python
from sieve.io.rdkit_adapter import WITHIN_N_SUFFIX, WITHIN_SSE_SUFFIX
```

Replace the block from `target = orbit_means(aligned, orbit)` through `repeats = ...` with:

```python
        target = orbit_means(aligned, orbit)
        # What the averaging removes, kept beside the target so the fit can
        # add it back to the predictive variance (within-structure-variance
        # spec 3.1): each atom's squared deviations from that mean, summed over
        # the members. Summed over atoms and keys this is floor_components'
        # `sse`, and the member counts sum to its `n_atoms`.
        sse = ((aligned - target) ** 2).sum(axis=0)
        # A weighted collapse repeats the representative, so each copy carries
        # its share and the sums stay the same.
        repeats = len(members) if weight_by_collapse else 1
        n_share = len(members) / repeats

        rep_i = members[0]
        rep = Chem.Mol(mset.mols[rep_i])
        for atom_idx, value in enumerate(target):
            atom = rep.GetAtomWithIdx(atom_idx)
            atom.SetDoubleProp(mset.atom_property, float(value))
            atom.SetDoubleProp(
                mset.atom_property + WITHIN_SSE_SUFFIX, float(sse[atom_idx]) / repeats
            )
            atom.SetDoubleProp(mset.atom_property + WITHIN_N_SUFFIX, n_share)
```

(keep the `for _ in range(repeats):` loop that follows unchanged). Add one sentence to the docstring: "Each atom also carries ``<atom_property>__within_sse`` and ``__within_n``, the scatter the averaging removed (within-structure-variance spec 3.1)."

Check `aligned`'s shape before relying on the broadcast: `aligned_values` returns `(n_members, n_atoms)` and `orbit_means` `(n_atoms,)`, as `_orbit_sse` already assumes (`values - orbit_means(values, orbit)`).

- [ ] **Step 4: Implement the predictor wiring**

In `experiments/experiments/predictors/sieve_predictor.py`, replace `_batch_for`:

```python
def _carries_within(mols: list[Any], atom_property: str) -> bool:
    """Whether ``mols`` came out of collapse, which attaches the
    within-structure companions to every atom (collapse_molecule_set)."""
    from sieve.io.rdkit_adapter import WITHIN_SSE_SUFFIX

    return bool(mols) and any(
        a.HasProp(atom_property + WITHIN_SSE_SUFFIX) for a in mols[0].GetAtoms()
    )


def _batch_for(
    mols: list[Any],
    config: Any,
    *,
    atom_property: str,
    with_target: bool,
    n_jobs: int | None = None,
) -> Any:
    """Build a ``NodeBatch`` for ``mols`` under an already-fitted
    ``config``. ``node_order`` is left ``None``: each ``Mol``'s own atom
    order is already this series' canonical order.

    A collapsed training set also hands over its within-structure sums; the
    first molecule decides, and a set that carries them on some molecules
    only is refused by the adapter rather than fitted partially."""
    from sieve.io.rdkit_adapter import from_rdkit

    within = with_target and _carries_within(mols, atom_property)
    return from_rdkit(
        mols,
        config=config,
        y_from_atom_prop=atom_property if with_target else None,
        within_from_atom_prop=atom_property if within else None,
        n_jobs=n_jobs,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_collapse.py experiments/tests/test_predictor_sieve.py experiments/tests/test_cv.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments/collapse.py experiments/experiments/predictors/sieve_predictor.py experiments/tests/test_collapse.py experiments/tests/test_predictor_sieve.py
git commit -m "feat(collapse): attach within-structure sums; the predictor fits them"
```

---

### Task 5: CV fold models carry σ²_w

**Files:**
- Modify: `experiments/experiments/cv.py:889-895` (`truncate_model`'s constructor)
- Modify: `experiments/experiments/cv.py` (new `_with_training_floor` beside `_floors_for`; call it in the fold loop after `train_models` is settled, before `for fold, group in enumerate(plan.groups):`)
- Test: `experiments/tests/test_cv.py` (append)

**Interfaces:**
- Consumes: `SieveModel.with_within_structure`, `within_n` (Task 1); `floor_cache_path(store, *, stores_root)`; `_other_groups(plan, fold)`.
- Produces: `_with_training_floor(model, store, train_shards, *, stores_root=None) -> SieveModel`; `truncate_model` preserves `within_sse`/`within_n`.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/tests/test_cv.py`:

```python
def test_truncate_model_carries_the_within_structure_statistics():
    from experiments.cv import truncate_model
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=12, seed=0)
    p = SievePredictor(attributes=("element",), edge_attributes=(), max_wl_depth=3)
    p.fit(train, train, rng=np.random.default_rng(0))
    deep = p._model.with_within_structure(2.0e-3, 20)
    shallow = truncate_model(deep, 1)
    np.testing.assert_array_equal(shallow.within_sse, deep.within_sse)
    assert shallow.within_n == deep.within_n


def _floor_cache(tmp_path, entries):
    import json

    from experiments.cv import floor_cache_path

    path = floor_cache_path("st", stores_root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def _small_model():
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=8, seed=0)
    p = SievePredictor(attributes=("element",), edge_attributes=(), max_wl_depth=1)
    p.fit(train, train, rng=np.random.default_rng(0))
    return p._model


def test_training_floor_is_the_pooled_sum_over_the_training_shards(tmp_path):
    from experiments.cv import _with_training_floor

    _floor_cache(
        tmp_path,
        {
            "s00": {"sse": 1.0, "sse_stereo_blind": 1.5, "n_atoms": 10.0},
            "s01": {"sse": 3.0, "sse_stereo_blind": 3.5, "n_atoms": 30.0},
            "s02": {"sse": 99.0, "sse_stereo_blind": 99.0, "n_atoms": 1.0},
        },
    )
    m = _with_training_floor(_small_model(), "st", ["s00", "s01"], stores_root=tmp_path)
    np.testing.assert_allclose(m.within_sse, [4.0])
    assert m.within_n == 40.0


def test_training_floor_needs_every_training_shard(tmp_path, caplog):
    from experiments.cv import _with_training_floor

    _floor_cache(
        tmp_path, {"s00": {"sse": 1.0, "sse_stereo_blind": 1.0, "n_atoms": 10.0}}
    )
    with caplog.at_level("WARNING"):
        m = _with_training_floor(
            _small_model(), "st", ["s00", "s01"], stores_root=tmp_path
        )
    assert m.within_n == 0.0
    assert "s01" in caplog.text


def test_training_floor_leaves_a_model_that_already_has_sums(tmp_path):
    from experiments.cv import _with_training_floor

    _floor_cache(
        tmp_path, {"s00": {"sse": 1.0, "sse_stereo_blind": 1.0, "n_atoms": 10.0}}
    )
    own = _small_model().with_within_structure(7.0, 70)
    m = _with_training_floor(own, "st", ["s00"], stores_root=tmp_path)
    np.testing.assert_allclose(m.within_sse, [7.0])
    assert m.within_n == 70.0
```

`floor_cache_path(store, stores_root=root)` is `root / store / "floor-components.json"` (`cv.py:953`), which is what `_floor_cache` writes.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_cv.py -q -k "within_structure or training_floor"`
Expected: FAIL: `truncate_model` returns `within_sse=None`; `ImportError: cannot import name '_with_training_floor'`.

- [ ] **Step 3: Implement**

In `truncate_model`, replace the final constructor with:

```python
    return sieve.SieveModel(
        cfg,
        tuple(levels),
        model.global_count,
        model.global_mean,
        model.global_msd,
        model.within_sse,
        model.within_n,
    )
```

After `_floors_for`, add:

```python
def _with_training_floor(
    model: Any,
    store: str,
    train_shards: Sequence[str],
    *,
    stores_root: Path | None = None,
) -> Any:
    """The fold model with its training shards' pooled within-structure sums.

    Cached fold models were fitted before collapse carried these sums
    (within-structure-variance spec 3.4); the floor cache holds exactly the
    same sums per shard -- the same deviations from the same orbit means -- so
    no refit is needed. A model that already carries sums keeps its own. When
    the cache is absent or misses a training shard, the model is returned
    unchanged (σ²_w = 0) with a warning, rather than with a partial sum.
    """
    if model.within_n > 0:
        return model
    path = floor_cache_path(store, stores_root=stores_root)
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [s for s in train_shards if s not in cache]
    if missing:
        logger.warning(
            "no floor-cache entry for training shard(s) %s of %s; the fold "
            "model's predictive variance omits the within-structure term",
            missing,
            store,
        )
        return model
    sse = sum(float(cache[s]["sse"]) for s in train_shards)
    n = sum(float(cache[s]["n_atoms"]) for s in train_shards)
    return model.with_within_structure(sse, n)
```

In the sieve CV assembly loop (the one around line 1720 that builds `train_models` from the cache or by `leave_one_group_out`), immediately after the `if train_models is None:` block and before `for fold, group in enumerate(plan.groups):`:

```python
        # After the cache is written, so cached files stay as they were.
        train_models = [
            _with_training_floor(
                m,
                store,
                [s for g in _other_groups(plan, f) for s in g],
                stores_root=stores_root,
            )
            for f, m in enumerate(train_models)
        ]
```

(`logger` is the module's `logging.getLogger("experiments")`, `cv.py:67`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_cv.py experiments/tests/test_cv_optional.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments/cv.py experiments/tests/test_cv.py
git commit -m "feat(cv): fold models take sigma2_w from their training shards"
```

---

### Task 6: Reproduction on real folds, the gate, and the PR

**Files:**
- Create (scratchpad, not committed): `$SCRATCH/uvar_reproduce.py`, where `$SCRATCH=/tmp/claude-1020/-data3-craabreu-github-repos-sieve/96a32280-bfb5-4416-a7ba-f4f3b00d40f9/scratchpad` (the analysis library `uvar_lib.py` lives there)
- Modify: `docs/superpowers/specs/2026-09-25-within-structure-variance-design.md` (status line only)

**Interfaces:**
- Consumes: everything above; `uvar_lib.sample(r, f)` returns `(df, sigma2_w)` with per-atom columns including the reconstructed three-term variance inputs; `uvar_lib.variance(df, *, alpha_v, alpha_t, a, sigma2_w)`; `uvar_lib.score(df, sig2, label)`.

- [ ] **Step 1: Write the reproduction script**

`uvar_lib.sample(r, f)` loads the cached fold model `results/cv-model-cache/sieve/element-eb-w10-n50-k5/r{r}-f{f}.npz`, truncates it to depth 5, sets continuation + EB and `predictive_variance=True`, predicts the fold's held-out conformers (train split only; it asserts the predictions equal Study B's `predictions.npz` to 1e-12) and returns per-atom class terms plus the training shards' pooled σ²_w. `uvar_lib.variance(df, ..., sigma2_w=...)` is the analysis' form B, including σ²_w on the global fallback. Write `$SCRATCH/uvar_reproduce.py`:

```python
"""Spec test 6: the library's form B reproduces the analysis on r0 folds 0-1."""

import glob
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import uvar_lib as U
from experiments.cv import _with_training_floor, truncate_model
from experiments.data import blob_to_mol

import sieve
from sieve.io.rdkit_adapter import from_rdkit
from sieve.uncertainty import predictive_variance

RECORDED = {0: {"E[z2]": 0.9732}, 1: {"E[z2]": 0.98, "NLL": -2.8148}}

for f in (0, 1):
    df, sigma2_w = U.sample(0, f)

    # The library path: the same cached model, given its training floor.
    (path,) = glob.glob(
        f"{U.ROOT}/runs/sieve-cv-study-b/r0-f{f}-sieve-element-continuation-eb-w5__*/predictions.npz"
    )
    z = np.load(path, allow_pickle=True)
    manifest = json.load(open(path.replace("predictions.npz", "manifest.json")))
    held = set(manifest["config"]["cv"]["held_out_shards"].split(","))
    train = [s for s in U.FLOORS if s not in held]
    m = truncate_model(
        sieve.SieveModel.load(
            f"{U.ROOT}/results/cv-model-cache/sieve/element-eb-w10-n50-k5/r0-f{f}.npz"
        ),
        5,
    )
    m = m.with_params(class_estimator="continuation", shrinkage_weight="empirical_bayes")
    m = replace(m, config=replace(m.config, predictive_variance=True))
    m = _with_training_floor(m, "dash-molecules", train, stores_root=Path(U.ROOT) / "stores")
    assert np.isclose(m.within_variance[0], sigma2_w, rtol=1e-12), (m.within_variance, sigma2_w)

    blob = U._blobs()
    mols = [blob_to_mol(blob[(str(a), str(b))]) for a, b in zip(z["dash_id"], z["conf_id"])]
    p = sieve.predict_detailed(m, from_rdkit(mols, config=m.config, n_jobs=16))
    k, cid = p.matched_level, p.class_id

    # alpha_v = 10: the library default, read straight off predict_detailed.
    ref10 = U.variance(df, alpha_v=10.0, alpha_t=1.0, a=0.5, sigma2_w=sigma2_w).values
    np.testing.assert_allclose(p.predictive_variance[:, 0], ref10, rtol=1e-10)

    # alpha_v = 30: the table indexed as predict does, with the same fallback.
    table = predictive_variance(m, alpha_v=30.0)
    lib30 = np.full(len(k), float(m.global_msd[0] + m.within_variance[0]))
    for lv in np.unique(k[k >= 0]):
        sel = k == lv
        lib30[sel] = table[lv][cid[sel], 0]
    ref30 = U.variance(df, alpha_v=30.0, alpha_t=1.0, a=0.5, sigma2_w=sigma2_w).values
    np.testing.assert_allclose(lib30, ref30, rtol=1e-10)

    print(f"fold {f}: sigma2_w = {sigma2_w:.4e}")
    print(U.score(df, ref30, f"r0-f{f} form B, alpha_v=30"), "recorded:", RECORDED[f])
    print(U.score(df, ref10, f"r0-f{f} form B, alpha_v=10"))
```

`U.ROOT` is `"experiments"`, relative to the repository root, so the script runs from there (Step 2); `floor_cache_path("dash-molecules", stores_root=Path(U.ROOT) / "stores")` is the file `uvar_lib.FLOORS` reads.

The check has two parts:

1. Per atom, the library's variance equals `uvar_lib.variance(...)` to `rtol=1e-10`, at both α^v values, and the model's σ²_w equals the analysis' pooled value (9.870e-05 on fold 0).
2. The printed scores at α^v = 30 match the recorded analysis to four figures: fold 0 E[z²] = 0.9732 (`b503ioa1d`); fold 1 E[z²] = 0.98 and NLL −2.8148 (`bve9x06af`).

- [ ] **Step 2: Run it**

Run (from the repository root): `PYTHONPATH=$SCRATCH .venv/bin/python $SCRATCH/uvar_reproduce.py`
Expected: no assertion error; the printed scores match the numbers above. If a number is off in the fourth figure, stop and report it rather than adjusting the tolerance.

- [ ] **Step 3: Run the full gate**

```bash
.venv/bin/ruff check src tests experiments
.venv/bin/ruff format --check src tests experiments
.venv/bin/ty check src tests experiments
.venv/bin/python -m pytest -q
```

Expected: ruff clean; format clean; ty reports only the two known `assert_array_equal` diagnostics; pytest all pass. Fix anything else before continuing.

- [ ] **Step 4: Update the spec status and commit**

In the spec, change `**Status:** design, awaiting review; not implemented` to `**Status:** phase 1 implemented (plan: docs/superpowers/plans/2026-09-25-within-structure-variance.md); phase 2 not started`.

```bash
git add docs/superpowers/specs/2026-09-25-within-structure-variance-design.md
git commit -m "docs(spec): within-structure variance phase 1 implemented"
git push
```

- [ ] **Step 5: Update the PR description**

Retitle PR #47 to "Within-structure variance in the predictive variance (phase 1)" and state in the body: the change (form B, α^v 30 → 10), the evidence table from spec §1, that no store rebuild or shard refit is needed, the reproduction result from Step 2, and the gate result. End the body with the attribution line. If `gh pr edit` fails on the GraphQL deprecation, use `gh api -X PATCH repos/craabreu/sieve/pulls/47 -f title=... -f body=...`.
