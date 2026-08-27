# DASH charges nested runs: persisted tree stats + raw/normalized predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one MLflow parent run (`fit()` + save trained tree stats + raw, unnormalized predict) spawn nested child runs (each loading the same raw predictions and applying a different charge-conservation normalization scheme), for both the `dash` (trained) and `dash_pretrained` (zero-training) predictors, without ever re-running the expensive tree-matching walk more than once per split.

**Architecture:** A new `normalize.py` (both normalization schemes, sharing one signature), a new `tree_artifact.py` (the small, saveable per-node mean/std/count table `dash.py`'s `fit()` derives), a `predict_raw`/`RawPrediction` split added to both DASH predictors (existing `predict()` stays behavior-identical, implemented on top of `predict_raw`), and a new, separate `nested_config.py`/`nested_runner.py`/`cli.py run-nested` path that reuses the existing flat `runner.py`'s own private helpers rather than duplicating them -- the existing flat `run`/`config.py`/`runner.py` path is untouched and keeps producing bit-identical output.

**Tech Stack:** Python 3.12, numpy, pandas, rdkit, mlflow (optional import, already the pattern in `runner.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md`

## Global Constraints

- No existing flat-config run's observable behavior may change: `dash`'s `predict()` stays unnormalized, `dash_pretrained`'s `predict()` stays std-weighted-normalized, both bit-identical to today.
- `tree_artifact.py`'s `compute_node_stats` must be pure numpy/python (no tree object, no rdkit) so it's testable without the real DASH-tree clone.
- `nested_config.py` must not hardcode which predictor names support `tree_stats` -- that's a duck-typed runtime concern for `execute_nested`, checked via `hasattr`.
- Every logged run (parent and each child) gets the same artifact set flat runs already produce (`metrics.json`, `manifest.json`, `predictions.npz`, `plots/parity_panel.png`, `config.resolved.yaml` is flat-only -- nested runs' resolved config lives inline in `manifest.json["config"]` instead, since there's no single YAML file per child).
- Reuse `charge_experiments/tests/helpers.py`'s `synthetic_molecule_set` for every fast test; never touch the real 8GB store or the real DASH-tree clone outside `_optional`-suffixed, skip-if-absent test files (existing convention).
- Run `uv run pytest charge_experiments/tests/ -x -q` (prefix with the miniforge3 PATH export if `uv` isn't found) and `uv run ruff check charge_experiments/` and `uv run ty check charge_experiments/` after every task; all three must be clean before committing.

---

### Task 1: `normalize.py` -- move `std_weighted_normalize`, add `equal_weighted_normalize`

**Files:**
- Create: `charge_experiments/charge_experiments/normalize.py`
- Modify: `charge_experiments/charge_experiments/predictors/dash_pretrained.py`
- Create: `charge_experiments/tests/test_charge_normalize.py`
- Delete: `charge_experiments/tests/test_charge_predictor_dash_pretrained.py` (its content moves into the new file)

**Interfaces:**
- Produces: `normalize.std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, n_conformers) -> NDArray[np.float64]`, `normalize.equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, n_conformers) -> NDArray[np.float64]`, `normalize.NORMALIZERS: dict[str, Callable[[NDArray, NDArray, NDArray, NDArray, int], NDArray[np.float64]]]` keyed `"std_weighted"`/`"equal_weighted"`.

- [ ] **Step 1: Write the failing test file**

Create `charge_experiments/tests/test_charge_normalize.py`:

```python
"""Fast-suite tests for normalize.py's pure-numpy normalization schemes --
no real DASH-tree clone needed. Moved here from
test_charge_predictor_dash_pretrained.py (std_weighted_normalize's own
implementation moved from predictors/dash_pretrained.py to normalize.py --
see docs/superpowers/specs/2026-08-27-dash-charges-nested-runs-design.md)."""

from __future__ import annotations

import numpy as np
import pytest


def test_std_weighted_normalize_conserves_charge():
    """Verified against get_molecules_partial_charges' own real code (not
    the paper's printed eq 2/3, which has a sign error -- see
    predictors/dash_pretrained.py's own module docstring for both proofs):
    the renormalized atom charges of one conformer must sum exactly to that
    conformer's own net_charge."""
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2])  # sums to 0.0
    raw_std = np.array([0.2, 0.1, 0.3])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert out.sum() == pytest.approx(1.0)


def test_std_weighted_normalize_matches_hand_computed_example():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.5, -0.5])  # sums to 0.0
    raw_std = np.array([0.4, 0.1])  # sums to 0.5
    net_charge = np.array([0.5])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    residual = 0.5 - 0.0  # Q_formal - sum(Q)
    expected = np.array([0.5 + residual * 0.4 / 0.5, -0.5 + residual * 0.1 / 0.5])
    np.testing.assert_allclose(out, expected)


def test_std_weighted_normalize_floors_nonpositive_std_to_default():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([0.0, -1.0])  # both non-positive -> both floored to 0.1
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_floors_nan_std_to_default():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.0, 0.0])
    raw_std = np.array([np.nan, np.nan])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    np.testing.assert_allclose(out, [0.5, 0.5])


def test_std_weighted_normalize_propagates_nan_charge_to_whole_conformer():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, np.nan, -0.1])
    raw_std = np.array([0.2, 0.2, 0.2])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert np.all(np.isnan(out))


def test_std_weighted_normalize_multiple_conformers_are_independent():
    from charge_experiments.normalize import std_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2, np.nan, 0.0])
    raw_std = np.array([0.2, 0.1, 0.3, 0.2, 0.2])
    net_charge = np.array([1.0, 0.0])
    mol_id = np.array([0, 0, 0, 1, 1])

    out = std_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 2)

    assert out[:3].sum() == pytest.approx(1.0)
    assert np.all(np.isnan(out[3:]))


def test_equal_weighted_normalize_conserves_charge():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.3, -0.1, -0.2])  # sums to 0.0
    raw_std = np.array([999.0, 0.0, -5.0])  # deliberately ignored
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert out.sum() == pytest.approx(1.0)


def test_equal_weighted_normalize_splits_residual_evenly():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.5, -0.5])  # sums to 0.0
    raw_std = np.array([0.0, 0.0])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    # residual 1.0 split evenly across 2 atoms -> +0.5 each
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_equal_weighted_normalize_ignores_raw_std():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.5, -0.5])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0])

    out_a = equal_weighted_normalize(
        raw_charge, np.array([0.1, 0.9]), net_charge, mol_id, 1
    )
    out_b = equal_weighted_normalize(
        raw_charge, np.array([50.0, 0.001]), net_charge, mol_id, 1
    )

    np.testing.assert_array_equal(out_a, out_b)


def test_equal_weighted_normalize_propagates_nan_charge_to_whole_conformer():
    from charge_experiments.normalize import equal_weighted_normalize

    raw_charge = np.array([0.3, np.nan, -0.1])
    raw_std = np.array([0.2, 0.2, 0.2])
    net_charge = np.array([1.0])
    mol_id = np.array([0, 0, 0])

    out = equal_weighted_normalize(raw_charge, raw_std, net_charge, mol_id, 1)

    assert np.all(np.isnan(out))


def test_normalizers_registry_has_both_schemes():
    from charge_experiments.normalize import (
        NORMALIZERS,
        equal_weighted_normalize,
        std_weighted_normalize,
    )

    assert NORMALIZERS["std_weighted"] is std_weighted_normalize
    assert NORMALIZERS["equal_weighted"] is equal_weighted_normalize
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_normalize.py -v`
Expected: FAIL/ERROR -- `ModuleNotFoundError: No module named 'charge_experiments.normalize'`

- [ ] **Step 3: Create `normalize.py`**

```python
"""Normalization schemes applied to a DASH-tree predictor's raw, unnormalized
per-atom charge walk -- kept independent of any predictor so a nested run can
apply several to the same already-computed raw predictions without
re-matching or re-fitting anything (see nested_runner.py).

Every entry in ``NORMALIZERS`` shares one signature, ``(raw_charge, raw_std,
net_charge, mol_id, n_conformers) -> atom_charge``, even though
``equal_weighted_normalize`` ignores ``raw_std`` entirely -- this lets
calling code stay normalization-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import molecule_sum

# get_molecules_partial_charges' own hardcoded default -- see
# std_weighted_normalize's own docstring, and predictors/dash_pretrained.py's
# module docstring, for why using it here is faithful, not invented.
_DEFAULT_STD_VALUE = 0.1


def std_weighted_normalize(
    raw_charge: NDArray[np.floating],
    raw_std: NDArray[np.floating],
    net_charge: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    n_conformers: int,
) -> NDArray[np.float64]:
    """DASH's own eq 4 (std-weighted normalization), pure numpy -- no tree,
    no rdkit. Verified against ``get_molecules_partial_charges``'s actual
    ``symmetric`` branch, not the paper's printed eq 2/3 (which has a sign
    error -- see predictors/dash_pretrained.py's own module docstring for
    both proofs).

    A non-positive (including NaN) entry in ``raw_std`` is floored to
    ``get_molecules_partial_charges``'s own ``default_std_value`` (0.1) --
    the authors' own published fallback for *that* quantity. A NaN entry in
    ``raw_charge`` is not floored or substituted: it propagates through
    ``molecule_sum`` into that whole conformer's residual, so every atom in
    a conformer with even one unmatched raw charge ends up NaN.
    """
    raw_charge = np.asarray(raw_charge, dtype=np.float64)
    raw_std = np.asarray(raw_std, dtype=np.float64)
    effective_std = np.where(raw_std > 0, raw_std, _DEFAULT_STD_VALUE)
    tot_charge_tree = molecule_sum(raw_charge, mol_id, n_conformers)
    tot_std_tree = molecule_sum(effective_std, mol_id, n_conformers)
    residual = np.asarray(net_charge, dtype=np.float64) - tot_charge_tree
    return raw_charge + (residual[mol_id] * effective_std / tot_std_tree[mol_id])


def equal_weighted_normalize(
    raw_charge: NDArray[np.floating],
    raw_std: NDArray[np.floating],
    net_charge: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    n_conformers: int,
) -> NDArray[np.float64]:
    """A simpler charge-conservation scheme: spread each conformer's
    residual equally across its own atoms, ignoring ``raw_std`` entirely
    (accepted only for signature parity with ``std_weighted_normalize`` --
    see module docstring). A NaN ``raw_charge`` on any atom propagates to
    the whole conformer the same way ``std_weighted_normalize``'s does, via
    ``molecule_sum``.
    """
    del raw_std
    raw_charge = np.asarray(raw_charge, dtype=np.float64)
    tot_charge_tree = molecule_sum(raw_charge, mol_id, n_conformers)
    residual = np.asarray(net_charge, dtype=np.float64) - tot_charge_tree
    n_atoms_per_mol = molecule_sum(np.ones_like(raw_charge), mol_id, n_conformers)
    return raw_charge + residual[mol_id] / n_atoms_per_mol[mol_id]


NORMALIZERS: dict[
    str,
    Callable[
        [
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.int64],
            int,
        ],
        NDArray[np.float64],
    ],
] = {
    "std_weighted": std_weighted_normalize,
    "equal_weighted": equal_weighted_normalize,
}
```

- [ ] **Step 4: Update `predictors/dash_pretrained.py` to import from `normalize.py`**

Remove the module-level `_DEFAULT_STD_VALUE = 0.1` and the entire
`def std_weighted_normalize(...)` function body from
`charge_experiments/charge_experiments/predictors/dash_pretrained.py`. Add
near the top, alongside the other `charge_experiments` imports:

```python
from charge_experiments.normalize import std_weighted_normalize
```

Every call site (`predict()`'s own `std_weighted_normalize(...)` call) is
unchanged -- only the definition moved.

- [ ] **Step 5: Delete the old test file**

```bash
git rm charge_experiments/tests/test_charge_predictor_dash_pretrained.py
```

- [ ] **Step 6: Run the full fast suite**

Run: `uv run pytest charge_experiments/tests/ -x -q`
Expected: PASS (the new `test_charge_normalize.py` file, plus every existing
test -- `test_charge_predictor_dash_pretrained_optional.py`'s own tests
import `std_weighted_normalize` only indirectly via the predictor's
`predict()`, so they're unaffected).

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check charge_experiments/` and `uv run ty check charge_experiments/`
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add charge_experiments/charge_experiments/normalize.py \
        charge_experiments/charge_experiments/predictors/dash_pretrained.py \
        charge_experiments/tests/test_charge_normalize.py \
        charge_experiments/tests/test_charge_predictor_dash_pretrained.py
git commit -m "refactor(charges): move std_weighted_normalize to normalize.py, add equal_weighted_normalize"
```

---

### Task 2: `predictors/base.py` -- `RawPrediction` + `NormalizableChargePredictor`

**Files:**
- Modify: `charge_experiments/charge_experiments/predictors/base.py`
- Create: `charge_experiments/tests/test_charge_predictor_base.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `base.RawPrediction(atom_charge: NDArray[np.float64], atom_std: NDArray[np.float64])`, `base.NormalizableChargePredictor` (a `@runtime_checkable` `Protocol` extending `Predictor` with `predict_raw(self, test: MoleculeSet) -> RawPrediction`).

- [ ] **Step 1: Write the failing test**

Create `charge_experiments/tests/test_charge_predictor_base.py`:

```python
"""Tests for base.py's RawPrediction/NormalizableChargePredictor -- the
predict_raw split both DASH predictors implement (Tasks 4/5)."""

from __future__ import annotations

import numpy as np


def test_raw_prediction_holds_atom_charge_and_atom_std():
    from charge_experiments.predictors.base import RawPrediction

    raw = RawPrediction(
        atom_charge=np.array([0.1, 0.2]), atom_std=np.array([0.05, 0.06])
    )
    np.testing.assert_array_equal(raw.atom_charge, [0.1, 0.2])
    np.testing.assert_array_equal(raw.atom_std, [0.05, 0.06])


def test_normalizable_predictor_protocol_matches_a_conforming_class():
    from charge_experiments.predictors.base import (
        NormalizableChargePredictor,
        Prediction,
        RawPrediction,
    )

    class _Conforming:
        name = "fake"

        def fit(self, train, val, *, rng):
            pass

        def predict(self, test):
            return Prediction(atom_charge=np.zeros(0))

        def predict_raw(self, test):
            return RawPrediction(atom_charge=np.zeros(0), atom_std=np.zeros(0))

    class _NonConforming:
        name = "fake2"

        def fit(self, train, val, *, rng):
            pass

        def predict(self, test):
            return Prediction(atom_charge=np.zeros(0))

    assert isinstance(_Conforming(), NormalizableChargePredictor)
    assert not isinstance(_NonConforming(), NormalizableChargePredictor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_base.py -v`
Expected: FAIL -- `ImportError: cannot import name 'RawPrediction'`

- [ ] **Step 3: Implement in `predictors/base.py`**

Add `runtime_checkable` to the `typing` import, and append after the
existing `Predictor` protocol:

```python
from typing import ClassVar, Protocol, runtime_checkable
```

```python
@dataclass(frozen=True)
class RawPrediction:
    """What a normalizable predictor's own ``predict_raw`` returns: the
    unnormalized per-atom walk output, plus its per-atom std (needed by
    ``normalize.std_weighted_normalize``) -- both required before any
    function in ``normalize.NORMALIZERS`` can be applied."""

    atom_charge: NDArray[np.float64]
    atom_std: NDArray[np.float64]


@runtime_checkable
class NormalizableChargePredictor(Predictor, Protocol):
    """A ``Predictor`` that can also return its raw, unnormalized walk
    output separately -- ``predictors/dash.py``'s ``DASHChargePredictor`` and
    ``predictors/dash_pretrained.py``'s ``DASHPretrainedChargePredictor``
    both implement this; ``nested_runner.py`` requires it."""

    def predict_raw(self, test: MoleculeSet) -> RawPrediction: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_base.py -v`
Expected: PASS

- [ ] **Step 5: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add charge_experiments/charge_experiments/predictors/base.py \
        charge_experiments/tests/test_charge_predictor_base.py
git commit -m "feat(charges): add RawPrediction and NormalizableChargePredictor protocol"
```

---

### Task 3: `tree_artifact.py` -- the saveable per-node stats table

**Files:**
- Create: `charge_experiments/charge_experiments/tree_artifact.py`
- Create: `charge_experiments/tests/test_charge_tree_artifact.py`

**Interfaces:**
- Consumes: `charge_experiments.data.molecule_sum` is not needed here (no per-atom aggregation across molecules, only per-node).
- Produces: `tree_artifact.LiteralTreeChargeProperties(charge_column: str, fallback_charge: float)` (moved here from `predictors/dash.py`, which will re-import it in Task 4), `tree_artifact.TreeNodeStats(branch_idx, node_id, mean, std, count: NDArray, fallback_mean: float, fallback_std: float)`, `tree_artifact.compute_node_stats(paths: list[list[tuple[int,int]]], atom_charge: NDArray[np.floating]) -> TreeNodeStats`, `tree_artifact.save_node_stats(stats: TreeNodeStats, path: str | Path) -> None`, `tree_artifact.load_node_stats(path: str | Path) -> TreeNodeStats`, `tree_artifact.apply_node_stats(tree: Any, stats: TreeNodeStats, *, mean_column: str = "dash_charge_mean", std_column: str = "dash_charge_std") -> tuple[LiteralTreeChargeProperties, LiteralTreeChargeProperties]`.

- [ ] **Step 1: Write the failing test file**

Create `charge_experiments/tests/test_charge_tree_artifact.py`:

```python
"""Pure-numpy/pandas tests for tree_artifact.py -- a fake tree-like object
stands in for a real DASHTree (same pattern as
test_charge_predictor_dash.py's own _FakeTree), so these need no rdkit and
no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

import pandas as pd


class _FakeTree:
    def __init__(self, branch_sizes: dict[int, int]):
        self.data_storage = {
            branch: pd.DataFrame(index=range(size))
            for branch, size in branch_sizes.items()
        }


def test_compute_node_stats_computes_mean_std_count():
    from charge_experiments.tree_artifact import compute_node_stats

    # Node (0, 1): charges 0.2, 0.4, 0.6 -> mean 0.4, population std sqrt(2/150)... compute directly.
    paths = [[(0, 1)], [(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4, 0.6])

    stats = compute_node_stats(paths, atom_charge)

    assert stats.branch_idx.tolist() == [0]
    assert stats.node_id.tolist() == [1]
    assert stats.count.tolist() == [3]
    np.testing.assert_allclose(stats.mean, [0.4])
    np.testing.assert_allclose(stats.std, [np.std([0.2, 0.4, 0.6])])


def test_compute_node_stats_std_zero_for_singleton_node():
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)]]
    atom_charge = np.array([0.7])

    stats = compute_node_stats(paths, atom_charge)

    np.testing.assert_allclose(stats.std, [0.0])


def test_compute_node_stats_fallback_matches_global_mean_and_std():
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)], [(0, 2)], [(1, 1)]]
    atom_charge = np.array([0.1, 0.5, -0.3])

    stats = compute_node_stats(paths, atom_charge)

    assert stats.fallback_mean == pytest.approx(atom_charge.mean())
    assert stats.fallback_std == pytest.approx(atom_charge.std())


def test_compute_node_stats_aggregates_a_node_visited_via_multiple_paths():
    """A node reached along more than one atom's path pools all of them,
    same as populate_tree_with_charge_property's own prior behavior."""
    from charge_experiments.tree_artifact import compute_node_stats

    paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_charge = np.array([0.2, 0.4])

    stats = compute_node_stats(paths, atom_charge)

    idx = stats.node_id.tolist().index(1)
    assert stats.count[idx] == 2
    assert stats.mean[idx] == pytest.approx(0.3)


def test_apply_node_stats_writes_mean_and_std_columns():
    from charge_experiments.tree_artifact import apply_node_stats, compute_node_stats

    tree = _FakeTree({0: 3})
    paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4])
    stats = compute_node_stats(paths, atom_charge)

    mean_props, std_props = apply_node_stats(tree, stats)

    df = tree.data_storage[0]
    assert df.loc[1, mean_props.charge_column] == pytest.approx(0.3)
    assert pd.isna(df.loc[0, mean_props.charge_column])
    assert df.loc[1, std_props.charge_column] == pytest.approx(0.1)
    assert mean_props.fallback_charge == pytest.approx(0.3)
    assert std_props.fallback_charge == pytest.approx(np.std([0.2, 0.4]))


def test_save_and_load_node_stats_round_trips(tmp_path):
    from charge_experiments.tree_artifact import (
        compute_node_stats,
        load_node_stats,
        save_node_stats,
    )

    paths = [[(0, 1)], [(0, 1), (1, 3)]]
    atom_charge = np.array([0.2, 0.4])
    stats = compute_node_stats(paths, atom_charge)

    path = tmp_path / "stats.npz"
    save_node_stats(stats, path)
    loaded = load_node_stats(path)

    np.testing.assert_array_equal(loaded.branch_idx, stats.branch_idx)
    np.testing.assert_array_equal(loaded.node_id, stats.node_id)
    np.testing.assert_allclose(loaded.mean, stats.mean)
    np.testing.assert_allclose(loaded.std, stats.std)
    np.testing.assert_array_equal(loaded.count, stats.count)
    assert loaded.fallback_mean == pytest.approx(stats.fallback_mean)
    assert loaded.fallback_std == pytest.approx(stats.fallback_std)


def test_apply_node_stats_from_loaded_stats_matches_direct_apply(tmp_path):
    from charge_experiments.tree_artifact import (
        apply_node_stats,
        compute_node_stats,
        load_node_stats,
        save_node_stats,
    )

    paths = [[(0, 1)], [(0, 1), (0, 2)]]
    atom_charge = np.array([0.2, 0.5])
    stats = compute_node_stats(paths, atom_charge)

    tree_direct = _FakeTree({0: 3})
    direct_mean_props, _ = apply_node_stats(tree_direct, stats)

    path = tmp_path / "stats.npz"
    save_node_stats(stats, path)
    loaded = load_node_stats(path)
    tree_loaded = _FakeTree({0: 3})
    loaded_mean_props, _ = apply_node_stats(tree_loaded, loaded)

    pd.testing.assert_frame_equal(
        tree_direct.data_storage[0], tree_loaded.data_storage[0]
    )
    assert direct_mean_props == loaded_mean_props
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_tree_artifact.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'charge_experiments.tree_artifact'`

- [ ] **Step 3: Implement `tree_artifact.py`**

```python
"""Persisted, per-tree-node charge statistics -- what
``predictors/dash.py``'s ``DASHChargePredictor.fit()`` actually derives from
a train split (a mean and std of ``MBIScharge`` at every ``(branch_idx,
node_id)`` its own atoms matched), separated out so it can be saved once and
reloaded without re-running the expensive ``match_new_atom`` walk that
produces it -- see nested_runner.py, where a "raw predict" parent run saves
this and later children reload it. See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PathKey = tuple[int, int]
NodePath = list[PathKey]


@dataclass(frozen=True)
class LiteralTreeChargeProperties:
    """What ``apply_node_stats`` writes onto a ``DASHTree``'s own
    ``data_storage``, and what ``predictors.dash.predict_via_data_storage_walk``
    needs to read it back."""

    charge_column: str
    fallback_charge: float


@dataclass(frozen=True)
class TreeNodeStats:
    """One row per populated ``(branch_idx, node_id)``: ``mean``/``std``/
    ``count`` of the train atoms matched there, plus the two global
    fallbacks (mean/std over every train atom, regardless of node) a
    predictor uses when an atom's own path is entirely unpopulated. Parallel
    numpy arrays, not a dict -- this whole object is what gets saved to /
    loaded from disk."""

    branch_idx: NDArray[np.int64]
    node_id: NDArray[np.int64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]
    fallback_mean: float
    fallback_std: float


def compute_node_stats(
    paths: list[NodePath], atom_charge: NDArray[np.floating]
) -> TreeNodeStats:
    """Two-pass (sum, sum-of-squares, count) aggregation of ``atom_charge``
    over every node on every atom's own matched path -- pure numpy/python,
    no tree object needed, so this is testable without a real ``DASHTree``.
    A node with only one matching atom gets ``std=0.0`` (population std,
    ``ddof=0`` -- consistent with a single observation having no spread).
    """
    if len(paths) != len(atom_charge):
        raise ValueError("paths and atom_charge must have the same length")
    atom_charge = np.asarray(atom_charge, dtype=np.float64)

    charge_sum: dict[PathKey, float] = {}
    charge_sumsq: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}
    for path, charge in zip(paths, atom_charge, strict=True):
        for key in path:
            charge_sum[key] = charge_sum.get(key, 0.0) + float(charge)
            charge_sumsq[key] = charge_sumsq.get(key, 0.0) + float(charge) ** 2
            count[key] = count.get(key, 0) + 1

    keys = list(count)
    branch_idx = np.array([k[0] for k in keys], dtype=np.int64)
    node_id = np.array([k[1] for k in keys], dtype=np.int64)
    count_arr = np.array([count[k] for k in keys], dtype=np.int64)
    sum_arr = np.array([charge_sum[k] for k in keys], dtype=np.float64)
    sumsq_arr = np.array([charge_sumsq[k] for k in keys], dtype=np.float64)
    mean_arr = sum_arr / count_arr
    variance = np.clip(sumsq_arr / count_arr - mean_arr**2, 0.0, None)
    std_arr = np.sqrt(variance)

    return TreeNodeStats(
        branch_idx=branch_idx,
        node_id=node_id,
        mean=mean_arr,
        std=std_arr,
        count=count_arr,
        fallback_mean=float(atom_charge.mean()),
        fallback_std=float(atom_charge.std()),
    )


def save_node_stats(stats: TreeNodeStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        branch_idx=stats.branch_idx,
        node_id=stats.node_id,
        mean=stats.mean,
        std=stats.std,
        count=stats.count,
        fallback_mean=np.float64(stats.fallback_mean),
        fallback_std=np.float64(stats.fallback_std),
    )


def load_node_stats(path: str | Path) -> TreeNodeStats:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")
    with np.load(path) as data:
        return TreeNodeStats(
            branch_idx=data["branch_idx"],
            node_id=data["node_id"],
            mean=data["mean"],
            std=data["std"],
            count=data["count"],
            fallback_mean=float(data["fallback_mean"]),
            fallback_std=float(data["fallback_std"]),
        )


def apply_node_stats(
    tree: Any,
    stats: TreeNodeStats,
    *,
    mean_column: str = "dash_charge_mean",
    std_column: str = "dash_charge_std",
) -> tuple[LiteralTreeChargeProperties, LiteralTreeChargeProperties]:
    """Write ``stats``'s mean/std onto ``tree.data_storage`` (one column
    each, indexed by ``node_id`` -- a node with no entry stays ``NaN``,
    exactly ``DASHTree.get_property_noNAN``'s own missing-value semantics),
    grouped by branch. Returns the ``(mean_props, std_props)``
    ``predict_via_data_storage_walk`` needs."""
    by_branch: dict[int, list[int]] = {}
    for i, branch_idx in enumerate(stats.branch_idx):
        by_branch.setdefault(int(branch_idx), []).append(i)

    for branch_idx, indices in by_branch.items():
        df = tree.data_storage[branch_idx]
        n_rows = len(df)
        node_ids = stats.node_id[indices]
        means = np.full(n_rows, np.nan)
        stds = np.full(n_rows, np.nan)
        means[node_ids] = stats.mean[indices]
        stds[node_ids] = stats.std[indices]
        df[mean_column] = means
        df[std_column] = stds

    return (
        LiteralTreeChargeProperties(
            charge_column=mean_column, fallback_charge=stats.fallback_mean
        ),
        LiteralTreeChargeProperties(
            charge_column=std_column, fallback_charge=stats.fallback_std
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_tree_artifact.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add charge_experiments/charge_experiments/tree_artifact.py \
        charge_experiments/tests/test_charge_tree_artifact.py
git commit -m "feat(charges): add tree_artifact.py -- saveable per-node mean/std/count stats"
```

---

### Task 4: `predictors/dash.py` -- `predict_raw`, `save_tree_stats`, `load_tree_stats`

**Files:**
- Modify: `charge_experiments/charge_experiments/predictors/dash.py`
- Modify: `charge_experiments/tests/test_charge_predictor_dash.py` (existing 3 tests must keep passing unmodified)
- Modify: `charge_experiments/tests/test_charge_predictor_dash_optional.py`

**Interfaces:**
- Consumes: `tree_artifact.{LiteralTreeChargeProperties, TreeNodeStats, compute_node_stats, apply_node_stats, save_node_stats, load_node_stats}` (Task 3), `predictors.base.{Prediction, RawPrediction}` (Task 2).
- Produces: `DASHChargePredictor.predict_raw(test: MoleculeSet) -> RawPrediction`, `DASHChargePredictor.save_tree_stats(path: str | Path) -> None`, `DASHChargePredictor.load_tree_stats(path: str | Path) -> None`. `populate_tree_with_charge_property`'s public signature/return type is unchanged (still `(tree, paths, atom_charge) -> LiteralTreeChargeProperties`, now `LiteralTreeChargeProperties` imported from `tree_artifact`, not defined locally).

- [ ] **Step 1: Confirm the existing tests still describe the target behavior**

`charge_experiments/tests/test_charge_predictor_dash.py`'s three existing
tests (`test_populate_tree_with_charge_property_writes_node_means`,
`test_predict_via_data_storage_walk_prefers_deepest_populated_node`,
`test_predict_via_data_storage_walk_backs_off_to_shallower_node`) already
pin `populate_tree_with_charge_property`'s and
`predict_via_data_storage_walk`'s exact contract. No changes needed to that
file for this step -- they double as the regression check for Step 3 below.

- [ ] **Step 2: Add the new failing tests to `test_charge_predictor_dash_optional.py`**

Append to `charge_experiments/tests/test_charge_predictor_dash_optional.py`
(after the existing `test_dash_charge_predictor_fits_and_predicts_on_synthetic_molecules`):

```python
def test_dash_charge_predictor_predict_equals_predict_raw_atom_charge():
    """predict() must stay behavior-identical to today: unnormalized, i.e.
    exactly predict_raw(...).atom_charge."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHChargePredictor()
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    np.testing.assert_array_equal(pred.atom_charge, raw.atom_charge)
    assert raw.atom_std.shape == raw.atom_charge.shape


def test_dash_charge_predictor_save_and_load_tree_stats_round_trips(tmp_path):
    """A predictor that loads a saved artifact (no fit() call at all)
    predicts identically to a freshly-fit one on the same train data."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=6, seed=0)
    test = synthetic_molecule_set(n_mol=4, seed=1)
    rng = np.random.default_rng(0)

    fitted = DASHChargePredictor()
    fitted.fit(train, train, rng=rng)
    fitted_pred = fitted.predict(test)

    stats_path = tmp_path / "dash-tree-stats.npz"
    fitted.save_tree_stats(stats_path)

    loaded = DASHChargePredictor()
    loaded.load_tree_stats(stats_path)
    loaded_pred = loaded.predict(test)

    np.testing.assert_array_equal(loaded_pred.atom_charge, fitted_pred.atom_charge)


def test_dash_charge_predictor_load_tree_stats_does_not_call_fit(tmp_path, monkeypatch):
    """Proves load_tree_stats never touches match_new_atom for train atoms
    -- the whole point of persisting stats."""
    from charge_experiments.predictors.dash import DASHChargePredictor

    from charge_experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=6, seed=0)
    test = synthetic_molecule_set(n_mol=4, seed=1)
    rng = np.random.default_rng(0)

    fitted = DASHChargePredictor()
    fitted.fit(train, train, rng=rng)
    stats_path = tmp_path / "dash-tree-stats.npz"
    fitted.save_tree_stats(stats_path)

    loaded = DASHChargePredictor()

    def _boom(*args, **kwargs):
        raise AssertionError("fit()/_atom_paths must not be called")

    monkeypatch.setattr(loaded, "fit", _boom)
    loaded.load_tree_stats(stats_path)  # must not raise
    pred = loaded.predict(test)
    assert pred.atom_charge.shape == (test.n_atoms,)
```

- [ ] **Step 3: Run to verify these fail**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_dash_optional.py -v`
Expected: FAIL -- `AttributeError: 'DASHChargePredictor' object has no attribute 'predict_raw'`
(skipped entirely if the DASH-tree clone isn't present locally -- run this
step only where it is; the plan's overall test command in later tasks still
covers it.)

- [ ] **Step 4: Rewrite `predictors/dash.py`**

Remove the local `@dataclass class LiteralTreeChargeProperties` block
entirely. Change the imports block to:

```python
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction, RawPrediction
from charge_experiments.tree_artifact import (
    LiteralTreeChargeProperties,
    TreeNodeStats,
    apply_node_stats,
    compute_node_stats,
    load_node_stats,
    save_node_stats,
)
```

Replace `populate_tree_with_charge_property`'s body:

```python
def populate_tree_with_charge_property(
    tree: Any, paths: list[NodePath], atom_charge: NDArray[np.floating]
) -> LiteralTreeChargeProperties:
    """Populate an already-loaded ``DASHTree``'s own storage with our own
    per-node mean/std ``MBIScharge`` over every node on every atom's path --
    a thin wrapper over tree_artifact's own compute_node_stats/
    apply_node_stats, kept for its existing callers/tests. Returns only the
    mean props (the std props are also written onto ``tree.data_storage``,
    just not returned here -- DASHChargePredictor.fit() below calls
    compute_node_stats/apply_node_stats directly instead, to keep both)."""
    stats = compute_node_stats(paths, atom_charge)
    mean_props, _std_props = apply_node_stats(tree, stats)
    return mean_props
```

`predict_via_data_storage_walk` is unchanged (still generic over any
`LiteralTreeChargeProperties`, now imported rather than locally defined --
no code change needed there beyond the import).

In `DASHChargePredictor.__init__`, replace `self._props:
LiteralTreeChargeProperties | None = None` with:

```python
        self._stats: TreeNodeStats | None = None
        self._mean_props: LiteralTreeChargeProperties | None = None
        self._std_props: LiteralTreeChargeProperties | None = None
```

Replace `fit`:

```python
    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        paths = self._paths_for(train, split="train")
        self._stats = compute_node_stats(paths, train.atom_charge)
        self._mean_props, self._std_props = apply_node_stats(self._tree, self._stats)
```

Replace `predict`, adding `predict_raw` and the save/load methods:

```python
    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        if self._mean_props is None or self._std_props is None:
            raise RuntimeError(
                "fit (or load_tree_stats) must be called before predict_raw"
            )
        paths = self._paths_for(test, split="test")
        atom_charge = predict_via_data_storage_walk(self._tree, paths, self._mean_props)
        atom_std = predict_via_data_storage_walk(self._tree, paths, self._std_props)
        return RawPrediction(atom_charge=atom_charge, atom_std=atom_std)

    def predict(self, test: MoleculeSet) -> Prediction:
        return Prediction(atom_charge=self.predict_raw(test).atom_charge)

    def save_tree_stats(self, path: str | Path) -> None:
        if self._stats is None:
            raise RuntimeError("fit must be called before save_tree_stats")
        save_node_stats(self._stats, path)

    def load_tree_stats(self, path: str | Path) -> None:
        """Loads the tree (fast -- reads the pinned clone's own data files,
        no atom matching) and applies a previously-saved stats artifact,
        skipping fit()'s own expensive match_new_atom walk over train
        entirely."""
        self._load_tree()
        self._stats = load_node_stats(path)
        self._mean_props, self._std_props = apply_node_stats(self._tree, self._stats)
```

- [ ] **Step 5: Run the existing pure-logic tests to confirm no regression**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_dash.py -v`
Expected: PASS, all 3 pre-existing tests unchanged.

- [ ] **Step 6: Run the new optional tests (only where the DASH-tree clone is present)**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_dash_optional.py -v`
Expected: PASS (5 tests total: 1 pre-existing + 3 new), or all skipped if the
clone is absent.

- [ ] **Step 7: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add charge_experiments/charge_experiments/predictors/dash.py \
        charge_experiments/tests/test_charge_predictor_dash_optional.py
git commit -m "feat(charges): DASHChargePredictor gains predict_raw/save_tree_stats/load_tree_stats"
```

---

### Task 5: `predictors/dash_pretrained.py` -- extract `predict_raw`

**Files:**
- Modify: `charge_experiments/charge_experiments/predictors/dash_pretrained.py`
- Modify: `charge_experiments/tests/test_charge_predictor_dash_pretrained_optional.py`

**Interfaces:**
- Consumes: `predictors.base.RawPrediction` (Task 2), `normalize.std_weighted_normalize` (Task 1), `tree_artifact.LiteralTreeChargeProperties` (Task 3).
- Produces: `DASHPretrainedChargePredictor.predict_raw(test: MoleculeSet) -> RawPrediction`. `predict()`'s external behavior (including all `match_stats` bookkeeping and warning logs) is unchanged.

- [ ] **Step 1: Add the failing test**

Append to `charge_experiments/tests/test_charge_predictor_dash_pretrained_optional.py`
(after the existing `test_dash_pretrained_predictor_runs_end_to_end_via_run`):

```python
def test_dash_pretrained_predict_raw_matches_predict_before_normalization():
    from charge_experiments.normalize import std_weighted_normalize
    from charge_experiments.predictors.dash_pretrained import (
        DASHPretrainedChargePredictor,
    )

    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHPretrainedChargePredictor()
    predictor.fit(mset, mset, rng=rng)

    raw = predictor.predict_raw(mset)
    pred = predictor.predict(mset)

    expected = std_weighted_normalize(
        raw.atom_charge, raw.atom_std, mset.net_charge, mset.atom_mol_id,
        mset.n_conformers,
    )
    np.testing.assert_array_equal(pred.atom_charge, expected)
    assert predictor.match_stats["n_final_nan_atoms"] == int(
        np.isnan(pred.atom_charge).sum()
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_dash_pretrained_optional.py -v -k predict_raw`
Expected: FAIL -- `AttributeError: 'DASHPretrainedChargePredictor' object has no attribute 'predict_raw'`

- [ ] **Step 3: Rewrite `predict` into `predict_raw` + `predict`**

Change the imports at the top of `dash_pretrained.py`:

```python
from charge_experiments.data import MoleculeSet
from charge_experiments.normalize import std_weighted_normalize
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction, RawPrediction
from charge_experiments.predictors.dash import _atom_paths, predict_via_data_storage_walk
from charge_experiments.tree_artifact import LiteralTreeChargeProperties
```

(drops the old `from charge_experiments.predictors.dash import
(LiteralTreeChargeProperties, _atom_paths, predict_via_data_storage_walk)`
line and the local `_DEFAULT_STD_VALUE`/`std_weighted_normalize` def, both
already removed in Task 1's Step 4.)

Replace the `predict` method:

```python
    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        tree = self._load_tree()
        paths, stats = _atom_paths(
            test,
            tree,
            max_depth=self.max_depth,
            attention_threshold=self.attention_threshold,
        )
        self.match_stats = stats
        if stats["n_unmatched_atoms"]:
            logger.warning(
                "DASH (pretrained) could not match %d/%d atoms (%d/%d molecules "
                "rejected outright); these are reported as NaN, not guessed at",
                stats["n_unmatched_atoms"],
                stats["n_atoms"],
                stats["n_unmatched_molecules"],
                stats["n_conformers"],
            )
        value_props = LiteralTreeChargeProperties(
            charge_column=tree.default_value_column, fallback_charge=float("nan")
        )
        std_props = LiteralTreeChargeProperties(
            charge_column=tree.default_std_column, fallback_charge=float("nan")
        )
        raw_charge = predict_via_data_storage_walk(tree, paths, value_props)
        raw_std = predict_via_data_storage_walk(tree, paths, std_props)
        return RawPrediction(atom_charge=raw_charge, atom_std=raw_std)

    def predict(self, test: MoleculeSet) -> Prediction:
        raw = self.predict_raw(test)
        atom_charge = std_weighted_normalize(
            raw.atom_charge, raw.atom_std, test.net_charge, test.atom_mol_id,
            test.n_conformers,
        )

        # match_stats' own n_unmatched_atoms (set in predict_raw) only
        # counts atoms whose path-matching itself failed -- a strict
        # undercount of the real NaN rate: raw_charge is also NaN for a
        # matched-but-nothing-populated path, and atom_charge is NaN for
        # every atom in a conformer where even one other atom's raw_charge
        # was NaN (see std_weighted_normalize's own docstring). Tracked here
        # so match_stats -- the one place a run's manifest.json reports
        # coverage -- reflects the true, final NaN rate, not just its
        # path-matching-failure subset.
        n_walk_nan_atoms = int(np.isnan(raw.atom_charge).sum())
        n_final_nan_atoms = int(np.isnan(atom_charge).sum())
        self.match_stats["n_walk_nan_atoms"] = n_walk_nan_atoms
        self.match_stats["n_final_nan_atoms"] = n_final_nan_atoms
        if n_final_nan_atoms:
            logger.warning(
                "DASH (pretrained) predicted NaN for %d/%d atoms after "
                "normalization (%d unmatched by path, %d unpopulated after "
                "a successful match); these are reported as NaN, not "
                "guessed at",
                n_final_nan_atoms,
                self.match_stats["n_atoms"],
                self.match_stats["n_unmatched_atoms"],
                n_walk_nan_atoms - self.match_stats["n_unmatched_atoms"],
            )
        return Prediction(atom_charge=atom_charge)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_predictor_dash_pretrained_optional.py -v`
Expected: PASS (all tests in the file, including the pre-existing three and
the new one), or all skipped if the DASH-tree clone is absent.

- [ ] **Step 5: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add charge_experiments/charge_experiments/predictors/dash_pretrained.py \
        charge_experiments/tests/test_charge_predictor_dash_pretrained_optional.py
git commit -m "refactor(charges): extract DASHPretrainedChargePredictor.predict_raw"
```

---

### Task 6: `nested_config.py`

**Files:**
- Create: `charge_experiments/charge_experiments/nested_config.py`
- Create: `charge_experiments/tests/test_charge_nested_config.py`

**Interfaces:**
- Consumes: `config.{RunCfg, DataCfg, PredictorCfg, _check_keys, apply_overrides}` (existing, `charge_experiments/charge_experiments/config.py`), `normalize.NORMALIZERS` (Task 1).
- Produces: `nested_config.TreeStatsCfg(save_path: str | None, load_path: str | None)`, `nested_config.NestedExperimentCfg(run: RunCfg, data: DataCfg, predictor: PredictorCfg, tree_stats: TreeStatsCfg, children: tuple[str, ...])`, `nested_config.load_nested_config(path, overrides=()) -> NestedExperimentCfg`.

- [ ] **Step 1: Write the failing test file**

Create `charge_experiments/tests/test_charge_nested_config.py`:

```python
from __future__ import annotations

import pytest
import yaml


def _write_yaml(tmp_path, data):
    path = tmp_path / "nested_config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base_raw():
    return {
        "run": {"experiment": "charge-nested-smoke", "seed": 0},
        "data": {"store": "dash-molecules", "split_column": "split"},
        "predictor": {"name": "dash", "params": {}},
        "tree_stats": {"save_path": "artifacts/stats.npz"},
        "children": ["std_weighted", "equal_weighted"],
    }


def test_load_nested_config_round_trips(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_nested_config(path)

    assert cfg.run.experiment == "charge-nested-smoke"
    assert cfg.predictor.name == "dash"
    assert cfg.tree_stats.save_path == "artifacts/stats.npz"
    assert cfg.tree_stats.load_path is None
    assert cfg.children == ("std_weighted", "equal_weighted")


def test_load_nested_config_rejects_unknown_normalization_scheme(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["children"] = ["not_a_real_scheme"]
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="unknown normalization scheme"):
        load_nested_config(path)


def test_load_nested_config_requires_at_least_one_child(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["children"] = []
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="at least one"):
        load_nested_config(path)


def test_load_nested_config_allows_a_predictor_with_no_tree_stats(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["predictor"] = {"name": "dash_pretrained", "params": {}}
    del raw["tree_stats"]
    path = _write_yaml(tmp_path, raw)

    cfg = load_nested_config(path)
    assert cfg.predictor.name == "dash_pretrained"
    assert cfg.tree_stats.save_path is None
    assert cfg.tree_stats.load_path is None


def test_load_nested_config_rejects_unknown_top_level_key(tmp_path):
    from charge_experiments.nested_config import load_nested_config

    raw = _base_raw()
    raw["bogus"] = 1
    path = _write_yaml(tmp_path, raw)

    with pytest.raises(ValueError, match="unknown key"):
        load_nested_config(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_nested_config.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'charge_experiments.nested_config'`

- [ ] **Step 3: Implement `nested_config.py`**

```python
"""Config for a nested run: one predictor's fit()+save+raw-predict (the
"parent"), plus one child run per normalization scheme applied to the same
already-computed raw predictions. See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.

``tree_stats`` doesn't hardcode which predictor names support it -- that's a
duck-typed runtime concern for nested_runner.execute_nested (checked via
hasattr), not a parse-time one here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from charge_experiments.config import (
    DataCfg,
    PredictorCfg,
    RunCfg,
    _check_keys,
    apply_overrides,
)
from charge_experiments.normalize import NORMALIZERS

_TOP_KEYS = {"run", "data", "predictor", "tree_stats", "children"}
_TREE_STATS_KEYS = {"save_path", "load_path"}


@dataclass(frozen=True)
class TreeStatsCfg:
    save_path: str | None = None
    load_path: str | None = None


@dataclass(frozen=True)
class NestedExperimentCfg:
    run: RunCfg
    data: DataCfg
    predictor: PredictorCfg
    tree_stats: TreeStatsCfg
    children: tuple[str, ...]


def _build(raw: Mapping[str, Any]) -> NestedExperimentCfg:
    _check_keys(raw, _TOP_KEYS, "config")
    for section in ("run", "data", "predictor", "children"):
        if section not in raw:
            raise ValueError(f"config is missing required section {section!r}")

    run_raw = raw["run"]
    run = RunCfg(
        experiment=run_raw["experiment"],
        seed=run_raw["seed"],
        tags=dict(run_raw.get("tags", {})),
    )

    data_raw = raw["data"]
    data = DataCfg(
        store=data_raw["store"],
        split_column=data_raw["split_column"],
        **{
            k: data_raw[k]
            for k in ("train_split", "val_split", "eval_split")
            if k in data_raw
        },
    )

    predictor_raw = raw["predictor"]
    predictor = PredictorCfg(
        name=predictor_raw["name"], params=dict(predictor_raw.get("params", {}))
    )

    tree_stats_raw = raw.get("tree_stats")
    if tree_stats_raw is not None:
        _check_keys(tree_stats_raw, _TREE_STATS_KEYS, "tree_stats")
        tree_stats = TreeStatsCfg(
            save_path=tree_stats_raw.get("save_path"),
            load_path=tree_stats_raw.get("load_path"),
        )
    else:
        tree_stats = TreeStatsCfg()

    children = tuple(raw["children"])
    if not children:
        raise ValueError("children must list at least one normalization scheme")
    unknown = [c for c in children if c not in NORMALIZERS]
    if unknown:
        raise ValueError(
            f"unknown normalization scheme(s) in children: {unknown}; "
            f"known: {sorted(NORMALIZERS)}"
        )

    return NestedExperimentCfg(
        run=run, data=data, predictor=predictor, tree_stats=tree_stats,
        children=children,
    )


def load_nested_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> NestedExperimentCfg:
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_nested_config.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add charge_experiments/charge_experiments/nested_config.py \
        charge_experiments/tests/test_charge_nested_config.py
git commit -m "feat(charges): add nested_config.py for parent+children run configs"
```

---

### Task 7: `nested_runner.py`

**Files:**
- Modify: `charge_experiments/charge_experiments/runner.py` (extract `_log_mlflow_run`)
- Create: `charge_experiments/charge_experiments/nested_runner.py`
- Create: `charge_experiments/tests/test_charge_nested_runner.py`

**Interfaces:**
- Consumes: `runner.{RunResult, _score, _write_plots (via _build_parity_panels), _savez_run, _git_info, _package_versions, DEFAULT_RUNS_ROOT, DEFAULT_TRACKING_URI}` (existing), `runner._log_mlflow_run(tags: dict[str,str], params: dict[str,str], run_metrics: dict[str,float], run_dir: Path) -> None` (new, this task), `nested_config.NestedExperimentCfg` (Task 6), `normalize.NORMALIZERS` (Task 1), `predictors.build` (existing), `predictors.base.{Prediction, RawPrediction}` (Task 2).
- Produces: `nested_runner.NestedRunResult(parent: RunResult, children: dict[str, RunResult])`, `nested_runner.execute_nested(cfg, mset, masks, *, runs_root=DEFAULT_RUNS_ROOT, allow_dirty=False, tracking=DEFAULT_TRACKING_URI, data_seconds=0.0) -> NestedRunResult`, `nested_runner.run_nested(cfg, *, runs_root=DEFAULT_RUNS_ROOT, allow_dirty=False, tracking=DEFAULT_TRACKING_URI, limit=None) -> NestedRunResult`.

- [ ] **Step 1: Refactor `runner.py`'s `_log_mlflow` into a reusable `_log_mlflow_run`**

In `charge_experiments/charge_experiments/runner.py`, replace the existing
`_log_mlflow` function with:

```python
def _log_mlflow_run(
    tags: dict[str, str],
    params: dict[str, str],
    run_metrics: dict[str, float],
    run_dir: Path,
) -> None:
    """Log tags/params/metrics/artifacts onto whatever MLflow run is
    currently open (inside ``mlflow.start_run``'s own context) -- shared by
    the flat run path (``_log_mlflow``, below) and nested_runner.py's own
    parent/child ``mlflow.start_run(nested=True)`` contexts."""
    import mlflow

    mlflow.set_tags(tags)
    mlflow.log_params(params)
    clean_metrics = {
        k: v
        for k, v in run_metrics.items()
        if isinstance(v, float) and not np.isnan(v)
    }
    mlflow.log_metrics({f"test/{k}": v for k, v in clean_metrics.items()})
    mlflow.log_artifacts(str(run_dir))


def _log_mlflow(
    cfg: ExperimentCfg, run_metrics: dict[str, float], run_dir: Path, tracking: str
) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping tracking for this run")
        return

    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(cfg.run.experiment)
    tags = {
        "predictor": cfg.predictor.name,
        "split_column": cfg.data.split_column,
        "store": cfg.data.store,
        "seed": str(cfg.run.seed),
        "run_dir": str(run_dir),
        **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
    }
    with mlflow.start_run(run_name=_run_name(cfg)):
        _log_mlflow_run(tags, to_flat_params(cfg), run_metrics, run_dir)
```

This is a pure refactor -- no existing test's expected behavior changes.

- [ ] **Step 2: Run the full fast suite to confirm no regression from the refactor**

Run: `uv run pytest charge_experiments/tests/ -x -q`
Expected: PASS, unchanged.

- [ ] **Step 3: Commit the refactor separately**

```bash
git add charge_experiments/charge_experiments/runner.py
git commit -m "refactor(charges): extract runner._log_mlflow_run for reuse by nested_runner"
```

- [ ] **Step 4: Write the failing test file for `nested_runner.py`**

Create `charge_experiments/tests/test_charge_nested_runner.py`:

```python
"""Smoke tests for nested_runner.py's orchestration -- a fake, in-process
NormalizableChargePredictor stands in for dash/dash_pretrained (both need
the real DASH-tree clone; this file tests execute_nested's own logic, not
DASH-tree matching), mirroring test_charge_smoke.py's synthetic-data
pattern. tracking=None throughout: no real MLflow server/tracking dir
needed here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction, RawPrediction

from charge_experiments.tests.helpers import synthetic_molecule_set


class _FakeNormalizablePredictor:
    """fit() sets one scalar (train's own mean atom charge); predict_raw()
    returns that scalar for every atom, plus a constant std. Good enough to
    exercise execute_nested's own orchestration (parent/children run-dirs,
    save/load-skips-fit, no re-matching per child) without rdkit or a real
    DASH-tree clone."""

    name = "fake_normalizable"

    def __init__(self) -> None:
        self.fit_calls = 0
        self.predict_raw_calls = 0
        self._value: float | None = None

    def fit(self, train, val, *, rng) -> None:
        del val, rng
        self.fit_calls += 1
        self._value = float(train.atom_charge.mean())

    def predict(self, test) -> Prediction:
        return Prediction(atom_charge=self.predict_raw(test).atom_charge)

    def predict_raw(self, test) -> RawPrediction:
        self.predict_raw_calls += 1
        assert self._value is not None
        n = test.n_atoms
        return RawPrediction(
            atom_charge=np.full(n, self._value), atom_std=np.full(n, 0.1)
        )

    def save_tree_stats(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"value": self._value}))

    def load_tree_stats(self, path) -> None:
        self._value = json.loads(Path(path).read_text())["value"]


register("fake_normalizable", lambda params: _FakeNormalizablePredictor())


def _nested_cfg(tmp_path, *, load_path=None, save_path=None):
    from charge_experiments.nested_config import (
        NestedExperimentCfg,
        TreeStatsCfg,
    )
    from charge_experiments.config import DataCfg, PredictorCfg, RunCfg

    return NestedExperimentCfg(
        run=RunCfg(experiment="charge-nested-smoke", seed=0),
        data=DataCfg(store="synthetic", split_column="split"),
        predictor=PredictorCfg(name="fake_normalizable", params={}),
        tree_stats=TreeStatsCfg(save_path=save_path, load_path=load_path),
        children=("std_weighted", "equal_weighted"),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=n_mol, p=[0.6, 0.2, 0.2])
    return {name: labels == name for name in ("train", "val", "test")}


def test_execute_nested_writes_a_parent_and_two_child_runs(tmp_path):
    from charge_experiments.nested_runner import execute_nested

    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _nested_cfg(tmp_path)

    result = execute_nested(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert result.parent.run_dir.is_dir()
    assert (result.parent.run_dir / "metrics.json").exists()
    assert (result.parent.run_dir / "manifest.json").exists()
    assert set(result.children) == {"std_weighted", "equal_weighted"}
    for name, child in result.children.items():
        assert child.run_dir.is_dir()
        assert (child.run_dir / "metrics.json").exists()
        manifest = json.loads((child.run_dir / "manifest.json").read_text())
        assert manifest["normalization"] == name


def test_execute_nested_children_reuse_the_same_raw_predictions(tmp_path):
    """predict_raw must be called exactly once per split, not once per
    child -- children only re-normalize."""
    from charge_experiments.nested_runner import execute_nested
    from charge_experiments.predictors import build

    mset = synthetic_molecule_set(n_mol=12, seed=0)
    masks = _synthetic_masks(12, seed=1)
    cfg = _nested_cfg(tmp_path)

    captured: list = []
    real_build = build

    def _spying_build(name, params):
        predictor = real_build(name, params)
        captured.append(predictor)
        return predictor

    import charge_experiments.nested_runner as nested_runner_mod

    original = nested_runner_mod.build
    nested_runner_mod.build = _spying_build
    try:
        execute_nested(
            cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
        )
    finally:
        nested_runner_mod.build = original

    assert len(captured) == 1
    # one predict_raw call per non-empty split (train/val/test), not per child
    assert captured[0].predict_raw_calls <= 3


def test_execute_nested_load_tree_stats_skips_fit(tmp_path):
    from charge_experiments.nested_runner import execute_nested

    mset = synthetic_molecule_set(n_mol=12, seed=0)
    masks = _synthetic_masks(12, seed=1)

    stats_path = tmp_path / "stats.json"
    save_cfg = _nested_cfg(tmp_path, save_path=str(stats_path))
    execute_nested(
        save_cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert stats_path.exists()

    load_cfg = _nested_cfg(tmp_path, load_path=str(stats_path))
    import charge_experiments.nested_runner as nested_runner_mod
    from charge_experiments.predictors import build

    captured: list = []
    real_build = build

    def _spying_build(name, params):
        predictor = real_build(name, params)
        captured.append(predictor)
        return predictor

    original = nested_runner_mod.build
    nested_runner_mod.build = _spying_build
    try:
        result = execute_nested(
            load_cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
        )
    finally:
        nested_runner_mod.build = original

    assert captured[0].fit_calls == 0
    manifest = json.loads((result.parent.run_dir / "manifest.json").read_text())
    assert manifest["tree_stats_source"] == "loaded"


def test_execute_nested_rejects_a_dirty_tree_by_default(tmp_path, monkeypatch):
    import pytest

    import charge_experiments.nested_runner as nested_runner_mod

    monkeypatch.setattr(
        nested_runner_mod._runner,
        "_git_info",
        lambda repo_root: {
            "commit": "deadbeef", "branch": "main", "dirty": True,
            "describe": "deadbeef-dirty",
        },
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _nested_cfg(tmp_path)

    with pytest.raises(RuntimeError, match="dirty"):
        execute_nested(
            cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None
        )
```

- [ ] **Step 5: Run test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_nested_runner.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'charge_experiments.nested_runner'`

- [ ] **Step 6: Implement `nested_runner.py`**

```python
"""Nested-run orchestration: one predictor's fit()+save+raw-predict as an
MLflow parent run, one child run per normalization scheme in
NestedExperimentCfg.children, each re-normalizing the same already-computed
raw predictions (no re-matching, no re-fit). See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.

Reuses charge_experiments.runner's own private helpers directly (``_score``,
``_build_parity_panels``, ``_savez_run``, ``_git_info``,
``_package_versions``, ``_log_mlflow_run``) -- this codebase already has
precedent for one module importing another's underscore-prefixed helpers
(predictors/dash_pretrained.py imports predictors/dash.py's own
``_atom_paths``), so this continues an established pattern rather than
starting a new one.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charge_experiments import runner as _runner
from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.nested_config import NestedExperimentCfg
from charge_experiments.normalize import NORMALIZERS
from charge_experiments.predictors import build
from charge_experiments.predictors.base import Prediction, RawPrediction

logger = logging.getLogger("charge_experiments")

DEFAULT_RUNS_ROOT = _runner.DEFAULT_RUNS_ROOT
DEFAULT_TRACKING_URI = _runner.DEFAULT_TRACKING_URI


@dataclass(frozen=True)
class NestedRunResult:
    parent: "_runner.RunResult"
    children: dict[str, "_runner.RunResult"]


def _tags_for(
    cfg: NestedExperimentCfg, *, normalization: str, tree_stats_source: str,
    run_dir: Path,
) -> dict[str, str]:
    return {
        "predictor": cfg.predictor.name,
        "split_column": cfg.data.split_column,
        "store": cfg.data.store,
        "seed": str(cfg.run.seed),
        "normalization": normalization,
        "tree_stats_source": tree_stats_source,
        "run_dir": str(run_dir),
        **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
    }


def _params_for(cfg: NestedExperimentCfg) -> dict[str, str]:
    return {
        "run.experiment": cfg.run.experiment,
        "run.seed": str(cfg.run.seed),
        "data.store": cfg.data.store,
        "data.split_column": cfg.data.split_column,
        "predictor.name": cfg.predictor.name,
        **{f"predictor.params.{k}": str(v) for k, v in cfg.predictor.params.items()},
    }


def _write_one_run(
    *,
    run_name: str,
    cfg: NestedExperimentCfg,
    test: MoleculeSet,
    pred: Prediction,
    run_metrics: dict[str, float],
    runs_root: Path,
    started: datetime,
    git_info: dict[str, Any],
    extra_manifest: dict[str, Any],
) -> "_runner.RunResult":
    """Write one run directory's full artifact set (metrics/manifest/
    predictions/plots), mirroring runner._execute_inner's own file writing
    -- but for a run whose fit()/predict() timing doesn't fit that
    function's flat shape (a child never calls either)."""
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / cfg.run.experiment / f"{run_name}__{stamp}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "git": git_info,
        "seed": cfg.run.seed,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _runner._package_versions(),
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "n_test_conformers": test.n_conformers,
            "n_test_atoms": test.n_atoms,
        },
        "config": {
            "run": {
                "experiment": cfg.run.experiment, "seed": cfg.run.seed,
                "tags": dict(cfg.run.tags),
            },
            "data": {
                "store": cfg.data.store, "split_column": cfg.data.split_column,
                "train_split": cfg.data.train_split, "val_split": cfg.data.val_split,
                "eval_split": cfg.data.eval_split,
            },
            "predictor": {
                "name": cfg.predictor.name, "params": dict(cfg.predictor.params),
            },
        },
        **extra_manifest,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    _runner._savez_run(run_dir / "predictions.npz", test, pred)
    if test.n_conformers:
        try:
            from charge_experiments import plots

            panels = _runner._build_parity_panels(test, pred, run_metrics)
            plots.parity_panel(
                panels, run_dir / "plots" / "parity_panel.png", suptitle=run_name
            )
        except ImportError:
            logger.warning("matplotlib not installed; skipping plots for %s", run_name)

    return _runner.RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


def execute_nested(
    cfg: NestedExperimentCfg,
    mset: MoleculeSet,
    masks: dict[str, NDArray[np.bool_]],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    data_seconds: float = 0.0,
) -> NestedRunResult:
    git_info = _runner._git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
        )

    random.seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)

    splits = {
        "train": mset.select(masks[cfg.data.train_split]),
        "val": mset.select(masks[cfg.data.val_split]),
        "test": mset.select(masks[cfg.data.eval_split]),
    }

    predictor = build(cfg.predictor.name, cfg.predictor.params)

    t0 = time.perf_counter()
    if cfg.tree_stats.load_path is not None:
        if not hasattr(predictor, "load_tree_stats"):
            raise AttributeError(
                f"predictor {cfg.predictor.name!r} has no load_tree_stats "
                "method; tree_stats.load_path is not usable with it"
            )
        predictor.load_tree_stats(cfg.tree_stats.load_path)
        tree_stats_source = "loaded"
    else:
        predictor.fit(splits["train"], splits["val"], rng=rng)
        tree_stats_source = "fit"
        if cfg.tree_stats.save_path is not None:
            if not hasattr(predictor, "save_tree_stats"):
                raise AttributeError(
                    f"predictor {cfg.predictor.name!r} has no save_tree_stats "
                    "method; tree_stats.save_path is not usable with it"
                )
            predictor.save_tree_stats(cfg.tree_stats.save_path)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_splits: dict[str, RawPrediction] = {}
    for name, split_mset in splits.items():
        if split_mset.n_conformers == 0:
            raw_splits[name] = RawPrediction(atom_charge=np.zeros(0), atom_std=np.zeros(0))
        else:
            raw_splits[name] = predictor.predict_raw(split_mset)
    predict_s = time.perf_counter() - t0

    match_stats = getattr(predictor, "match_stats", None)

    def _run_metrics_for(atom_charge_by_split: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in ("test", "train", "val"):
            split_mset = splits[name]
            if split_mset.n_conformers == 0:
                continue
            score = _runner._score(
                split_mset, Prediction(atom_charge=atom_charge_by_split[name])
            )
            out.update(score if name == "test" else {f"{name}/{k}": v for k, v in score.items()})
        out["time/fit_s"] = fit_s
        out["time/predict_s"] = predict_s
        out["time/data_s"] = data_seconds
        return out

    tracking_ok = tracking is not None
    mlflow: Any = None
    if tracking_ok:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_tracking_uri(tracking)
            mlflow.set_experiment(cfg.run.experiment)
        except ImportError:
            logger.warning("mlflow not installed; skipping tracking for this run")
            tracking_ok = False

    parent_charges = {name: raw.atom_charge for name, raw in raw_splits.items()}
    parent_metrics = _run_metrics_for(parent_charges)
    parent_extra_manifest: dict[str, Any] = {
        "elapsed_s": {"fit": fit_s, "predict": predict_s, "data": data_seconds},
        "tree_stats_source": tree_stats_source,
    }
    if match_stats:
        parent_extra_manifest["match_stats"] = match_stats

    parent_run_name = f"{cfg.predictor.name}-raw"
    started = datetime.now(UTC)
    children_results: dict[str, _runner.RunResult] = {}

    parent_ctx = mlflow.start_run(run_name=parent_run_name) if tracking_ok else nullcontext()
    with parent_ctx:
        parent_result = _write_one_run(
            run_name=parent_run_name, cfg=cfg, test=splits["test"],
            pred=Prediction(atom_charge=parent_charges["test"]),
            run_metrics=parent_metrics, runs_root=runs_root, started=started,
            git_info=git_info, extra_manifest=parent_extra_manifest,
        )
        if tracking_ok:
            _runner._log_mlflow_run(
                _tags_for(
                    cfg, normalization="raw", tree_stats_source=tree_stats_source,
                    run_dir=parent_result.run_dir,
                ),
                _params_for(cfg), parent_metrics, parent_result.run_dir,
            )

        for name in cfg.children:
            normalize_fn = NORMALIZERS[name]
            child_charges = {
                split_name: (
                    normalize_fn(
                        raw.atom_charge, raw.atom_std, splits[split_name].net_charge,
                        splits[split_name].atom_mol_id, splits[split_name].n_conformers,
                    )
                    if splits[split_name].n_conformers
                    else np.zeros(0)
                )
                for split_name, raw in raw_splits.items()
            }
            child_metrics = _run_metrics_for(child_charges)
            child_run_name = f"{cfg.predictor.name}-{name}"
            child_extra_manifest: dict[str, Any] = {
                "elapsed_s": {"fit": 0.0, "predict": 0.0, "data": 0.0},
                "normalization": name,
                "tree_stats_source": tree_stats_source,
            }
            if match_stats:
                child_extra_manifest["match_stats"] = match_stats

            child_ctx = (
                mlflow.start_run(run_name=child_run_name, nested=True)
                if tracking_ok
                else nullcontext()
            )
            with child_ctx:
                child_result = _write_one_run(
                    run_name=child_run_name, cfg=cfg, test=splits["test"],
                    pred=Prediction(atom_charge=child_charges["test"]),
                    run_metrics=child_metrics, runs_root=runs_root,
                    started=datetime.now(UTC), git_info=git_info,
                    extra_manifest=child_extra_manifest,
                )
                if tracking_ok:
                    _runner._log_mlflow_run(
                        _tags_for(
                            cfg, normalization=name, tree_stats_source=tree_stats_source,
                            run_dir=child_result.run_dir,
                        ),
                        _params_for(cfg), child_metrics, child_result.run_dir,
                    )
                children_results[name] = child_result

    return NestedRunResult(parent=parent_result, children=children_results)


def run_nested(
    cfg: NestedExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    limit: int | None = None,
) -> NestedRunResult:
    """Load the real store, then run the nested pipeline (see
    execute_nested). Mirrors runner.run's own shape."""
    t0 = time.perf_counter()
    mset, masks = _runner.load_molecule_set(
        cfg.data.store,
        split_column=cfg.data.split_column,
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.eval_split),
        limit=limit,
    )
    data_seconds = time.perf_counter() - t0

    return execute_nested(
        cfg, mset, masks, runs_root=runs_root, allow_dirty=allow_dirty,
        tracking=tracking, data_seconds=data_seconds,
    )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_nested_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 9: Commit**

```bash
git add charge_experiments/charge_experiments/nested_runner.py \
        charge_experiments/tests/test_charge_nested_runner.py
git commit -m "feat(charges): add nested_runner.py -- parent+children MLflow orchestration"
```

---

### Task 8: `cli.py run-nested` subcommand

**Files:**
- Modify: `charge_experiments/charge_experiments/cli.py`
- Create: `charge_experiments/tests/test_charge_cli.py`
- Create: `charge_experiments/configs/dash-nested-charge-example.yaml`
- Create: `charge_experiments/configs/dash-pretrained-nested-charge-example.yaml`

**Interfaces:**
- Consumes: `nested_config.load_nested_config` (Task 6), `nested_runner.{DEFAULT_RUNS_ROOT, DEFAULT_TRACKING_URI, run_nested}` (Task 7).
- Produces: `python -m charge_experiments run-nested --config <path> [--set k=v] [--limit N] [--allow-dirty] [--no-tracking]`.

- [ ] **Step 1: Write the failing test**

Create `charge_experiments/tests/test_charge_cli.py`:

```python
"""Argparse-level tests for cli.py -- verifies flags parse and route to the
right handler function, not end-to-end execution (that's
test_charge_nested_runner.py's/test_charge_smoke.py's job)."""

from __future__ import annotations


def test_build_parser_accepts_run_nested_with_all_flags():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run-nested",
            "--config",
            "charge_experiments/configs/dash-nested-charge-example.yaml",
            "--set",
            "predictor.params.max_depth=8",
            "--limit",
            "100",
            "--allow-dirty",
            "--no-tracking",
        ]
    )
    assert args.command == "run-nested"
    assert str(args.config) == "charge_experiments/configs/dash-nested-charge-example.yaml"
    assert args.set == ["predictor.params.max_depth=8"]
    assert args.limit == 100
    assert args.allow_dirty is True
    assert args.no_tracking is True


def test_build_parser_run_nested_defaults():
    from charge_experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["run-nested", "--config", "some-config.yaml"]
    )
    assert args.set == []
    assert args.limit is None
    assert args.allow_dirty is False
    assert args.no_tracking is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest charge_experiments/tests/test_charge_cli.py -v`
Expected: FAIL -- `SystemExit` / argparse error (`run-nested` isn't a known subcommand)

- [ ] **Step 3: Add the subcommand to `cli.py`**

Add near the top, alongside the existing imports:

```python
from charge_experiments.nested_config import load_nested_config
```

Add a new handler function (near `_cmd_run`):

```python
def _cmd_run_nested(args: argparse.Namespace) -> int:
    from charge_experiments.nested_runner import (
        DEFAULT_RUNS_ROOT as NESTED_RUNS_ROOT,
        DEFAULT_TRACKING_URI as NESTED_TRACKING_URI,
        run_nested,
    )

    cfg = load_nested_config(args.config, overrides=args.set)
    tracking = None if args.no_tracking else NESTED_TRACKING_URI
    result = run_nested(
        cfg,
        runs_root=NESTED_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
        tracking=tracking,
        limit=args.limit,
    )
    print(f"parent run written to {result.parent.run_dir}")
    for name, child in result.children.items():
        print(f"child run ({name}) written to {child.run_dir}")
    return 0
```

In `build_parser()`, after the existing `p_run` block (right before
`p_prepare = sub.add_parser(...)`):

```python
    p_run_nested = sub.add_parser(
        "run-nested",
        help="run one predictor's fit()+save+raw-predict, plus a nested "
        "child run per normalization scheme",
    )
    p_run_nested.add_argument("--config", required=True, type=Path)
    p_run_nested.add_argument(
        "--set", action="append", default=[], metavar="key.path=value",
        help="override a config value; may be passed multiple times",
    )
    p_run_nested.add_argument(
        "--limit", type=int, default=None, help="use only the first N conformers"
    )
    p_run_nested.add_argument(
        "--allow-dirty", action="store_true", help="run with an uncommitted git tree"
    )
    p_run_nested.add_argument(
        "--no-tracking", action="store_true", help="skip MLflow logging for this run"
    )
    p_run_nested.set_defaults(func=_cmd_run_nested)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest charge_experiments/tests/test_charge_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the two example configs**

Create `charge_experiments/configs/dash-nested-charge-example.yaml`:

```yaml
run:
  experiment: dash-charges-nested
  seed: 0
data:
  store: dash-molecules
  split_column: split
predictor:
  name: dash
  params: {}
tree_stats:
  save_path: charge_experiments/artifacts/dash-tree-stats.npz
children:
  - std_weighted
  - equal_weighted
```

Create `charge_experiments/configs/dash-pretrained-nested-charge-example.yaml`:

```yaml
run:
  experiment: dash-charges-nested
  seed: 0
data:
  store: dash-molecules
  split_column: split
predictor:
  name: dash_pretrained
  params: {}
children:
  - std_weighted
  - equal_weighted
```

- [ ] **Step 6: Run the full fast suite, lint, type-check**

Run: `uv run pytest charge_experiments/tests/ -x -q && uv run ruff check charge_experiments/ && uv run ty check charge_experiments/`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add charge_experiments/charge_experiments/cli.py \
        charge_experiments/tests/test_charge_cli.py \
        charge_experiments/configs/dash-nested-charge-example.yaml \
        charge_experiments/configs/dash-pretrained-nested-charge-example.yaml
git commit -m "feat(charges): add run-nested CLI subcommand"
```

---

## Self-review notes

- **Spec coverage:** every decision row in the spec maps to a task -- artifact persistence/format (Task 3), `predict_raw` split for both predictors (Tasks 4-5), normalization module (Task 1), base protocol (Task 2), nested config/runner/CLI (Tasks 6-8). The spec's "six logged runs total" scenario is exercised end-to-end once Tasks 4, 5, 7, 8 are all done (two real `run-nested` invocations, one per predictor, each config's `children` list producing the two child runs) -- not itself a separate task, since it needs the real DASH-tree clone and the real store, matching this series' existing `_optional`/manual-run convention rather than a fast unit test.
- **No placeholders:** every step above has literal code, not a description of code.
- **Type/name consistency check:** `RawPrediction` (Task 2) is used with the same two fields (`atom_charge`, `atom_std`) everywhere it appears (Tasks 4, 5, 7). `LiteralTreeChargeProperties` is defined once (Task 3, `tree_artifact.py`) and only ever imported elsewhere (Task 4's `dash.py`, Task 5's `dash_pretrained.py`) -- no second definition. `NORMALIZERS`' two keys (`"std_weighted"`, `"equal_weighted"`) are the same strings used in `nested_config.py`'s children validation (Task 6) and `nested_runner.py`'s lookup (Task 7). `predict_via_data_storage_walk` and `_atom_paths` (both pre-existing, `dash.py`) are referenced with their existing signatures throughout, unchanged.
