# Study F: Calibration of Sieve's Predictive Variance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A workflow step that scores Sieve's predictive variance (form B and four ablations) for calibration and variance-weighted normalisation on the 25 Study B incumbent runs, and writes one text report.

**Architecture:**
- **Library.** `sieve.uncertainty.variance_terms` exposes the four per-level terms, and `predictive_variance` becomes their weighted sum.
- **New module.** `experiments/experiments/calibration.py` reloads each run's fold model and checks that it reproduces the run's saved predictions. It computes every arm's per-atom σ² from the terms, scores them, and writes a `calibration_metrics.json` sidecar, following the pattern of Study D's `stereo_subsets.py`.
- **Report.** A report function reads the sidecars and writes `study-f.txt`.
- **Refactor.** Fold-model assembly is factored out of `run_sieve_cv` into `experiments.cv.SieveTrainModels`, so both callers share it.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.stats.norm`), RDKit, pytest, ruff, ty; bash workflow.

**Spec:** `docs/superpowers/specs/2026-09-25-study-f-calibration-design.md`

## Global Constraints

- One branch, `study-f-calibration-spec`, and one PR carrying the spec, this plan, the code and the workflow steps.
- Samples: experiment `$SIEVE_STUDY_B`, method `$SIEVE_METHOD` (= `sieve-element-continuation-eb`), depth `$SIEVE_SELECTED_DEPTH`; 5 repeats × K = 5 folds; train split only. The test split is never read.
- Arms (spec §2.2):

  | arm | α^v | a | estimation | σ²_w |
  |---|---:|---:|---|---|
  | `form_b` | 10 | 0.5 | yes | yes |
  | `no_sigma2_w` | 10 | 0.5 | yes | no |
  | `alpha_v_30` | 30 | 0.5 | yes | yes |
  | `no_selection` | 10 | 0 | yes | yes |
  | `no_estimation` | 10 | 0.5 | no | yes |

  α^t = 1 in every arm. Unmatched atoms take `global_msd + σ²_w` where the arm has σ²_w, and `global_msd` otherwise.
- `form_b` must equal `predict_detailed(...).predictive_variance` **bit for bit**, and `predictive_variance` must stay bit-identical after the refactor.
- The fold model must reproduce the run's `predictions.npz` to within 1e-12, with atoms aligned by `(dash_id, conf_id)`. Otherwise the run is refused.
- Sidecar keys:
  - `calibration/<arm>/<metric>` for each arm;
  - `calibration/equal/norm_rmse` and `calibration/equal/norm_mae`;
  - `calibration/n_atoms`, `calibration/sigma2_w` and `calibration/share_k{k}`.

  The metrics are `nll`, `ez2`, `cov50`, `cov90`, `cov95`, `cov99`, `ez2_k{k}` (only for radii that occur), `rho`, `norm_rmse` and `norm_mae`.
- Normalisation uses `experiments.normalize.variance_weighted_normalize` (the arm's σ) and `equal_weighted_normalize`, with each conformer's total constrained to its `molecule_value`.
- The report prints six decimals for normalised errors, four for everything else and `.2e` for differences. Paired differences use `stereo_subsets.paired_difference(a, b, k=K)`, which returns b − a. The report is byte-identical across reruns.
- `run_sieve_cv`'s behaviour and outputs are unchanged by the refactor.
- No figures and no LaTeX.
- Gate: `.venv/bin/ruff check src tests experiments`, `.venv/bin/ruff format --check src tests experiments`, `.venv/bin/ty check src tests experiments` (two known `assert_array_equal` diagnostics), `.venv/bin/python -m pytest -q`, **and** `.venv/bin/python -m pytest experiments/tests -q` on its own. The last one catches `tests.helpers` import-order bugs; new tests under `experiments/tests` must not import the root `tests.helpers`.

## Review Focus

- **Runs whose fold models are not in the cache.** The step must fall back to merging the shard fits, exactly as `run_sieve_cv` does. Pinned in Task 3 (`test_train_models_from_shards_equal_the_cached_ones`).
- **A stereo-track or multi-level config, where the backoff position is not the raw level index.** `arm_variances` must index the variance tables by raw level. Pinned in Task 2 (`test_form_b_equals_the_library_under_a_stereo_track`).
- **A run whose held-out order differs from the store's shard order.** The step must refuse to score it rather than mis-pair atoms. Pinned in Task 4 (`test_a_run_whose_ids_disagree_is_refused`).
- **Running the scorer twice, or on only some repeats.** A second run writes nothing, and `--repeats` restricts both `--check` and scoring. Pinned in Task 4 (`test_missing_scores_reports_absent_stale_and_incomplete`) and Task 5 (`test_score_calibration_check_is_scoped_by_repeats`).
- **A sidecar key that clashes with `metrics.json` or with the other sidecar.** `aggregate` must refuse it. Pinned in Task 4 (`test_aggregate_merges_the_calibration_sidecar_and_refuses_a_clash`).

---

### File map

| File | Change |
|---|---|
| `src/sieve/uncertainty.py` | `variance_terms`; `predictive_variance` sums it |
| `experiments/experiments/calibration.py` (new) | arms, metrics, arm variances, scoring, staleness, report |
| `experiments/experiments/cv.py` | `SieveTrainModels`; `run_sieve_cv` uses it |
| `experiments/experiments/aggregate.py` | merge `calibration_metrics.json` too |
| `experiments/experiments/cli.py` | `score-calibration`, `calibration-report` |
| `experiments/workflows/cv_charges.sh` | `study-f-scores`, `study-f-report` |
| `tests/test_uncertainty.py` | `variance_terms` tests |
| `experiments/tests/test_calibration.py` (new) | everything in `calibration.py` |
| `experiments/tests/test_cv.py` | `SieveTrainModels` tests |

---

### Task 1: `variance_terms`

**Files:**
- Modify: `src/sieve/uncertainty.py`
- Test: `tests/test_uncertainty.py` (append)

**Interfaces:**
- Produces: `variance_terms(model, *, alpha_v: float = ALPHA_V, alpha_t: float = ALPHA_T) -> list[dict[str, np.ndarray]]`, one dict per level (raw level index), with keys `"within"` `(nc, d)`, `"selection"` `(nc, d)`, `"estimation"` `(nc, d)`, `"within_structure"` `(d,)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_uncertainty.py`:

```python
# ------------------------------------------------------------ the terms --


def _terms_models():
    """A plain model, and the same with sigma2_w."""
    m = sieve.fit(_structured_batch(graphs=5, n=10), simple_config(max_wl_depth=3))
    return [m, m.with_within_structure(3.0e-3, 30)]


@pytest.mark.parametrize("which", [0, 1])
def test_predictive_variance_is_the_weighted_sum_of_its_terms(which):
    from sieve.uncertainty import variance_terms

    m = _terms_models()[which]
    for alpha_v, a in ((ALPHA_V, SELECTION_WEIGHT), (30.0, 0.0), (3.0, 1.0)):
        terms = variance_terms(m, alpha_v=alpha_v)
        total = predictive_variance(m, alpha_v=alpha_v, selection_weight=a)
        assert len(terms) == len(total) == len(m.levels)
        for t, p in zip(terms, total, strict=True):
            assert set(t) == {"within", "selection", "estimation", "within_structure"}
            np.testing.assert_array_equal(
                t["within"] + a * t["selection"] + t["estimation"] + t["within_structure"],
                p,
            )


def test_within_structure_term_is_the_pooled_sigma2_w():
    from sieve.uncertainty import variance_terms

    plain, with_w = _terms_models()
    for t in variance_terms(plain):
        np.testing.assert_array_equal(t["within_structure"], np.zeros(1))
    for t in variance_terms(with_w):
        np.testing.assert_allclose(t["within_structure"], [1.0e-4])


def test_variance_terms_under_a_stereo_track():
    pytest.importorskip("rdkit")
    import dataclasses

    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit
    from sieve.uncertainty import variance_terms

    smiles = ["C/C=C/C", r"C/C=C\C", "C/C=C/CC", r"C/C=C\CC", "CCCC"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    cfg = simple_config(stereo=("cis_trans",), max_wl_depth=3)
    b = from_rdkit(mols, config=cfg)
    b = dataclasses.replace(b, y=np.random.default_rng(0).normal(size=(b.n_nodes, 1)))
    m = sieve.fit(b, cfg)
    for t, p in zip(variance_terms(m), predictive_variance(m), strict=True):
        np.testing.assert_array_equal(
            t["within"] + SELECTION_WEIGHT * t["selection"] + t["estimation"]
            + t["within_structure"],
            p,
        )
```

`simple_config`'s codes cover C and H only, and these SMILES have no explicit hydrogens, so every element code is known.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_uncertainty.py -q -k "terms or within_structure_term"`
Expected: FAIL with `ImportError: cannot import name 'variance_terms'`.

- [ ] **Step 3: Implement**

In `src/sieve/uncertainty.py`, rename the body of `predictive_variance` into `variance_terms`. Keep the prologue (`cfg`, `parents`, `av`, `tau_pooled`, `tau_class`, `counts`, `weights`, `root`, `tau_aware`, `sigma2_w`) and the per-level computation of `within`, `selection` and `estimation` unchanged. The loop's last line becomes:

```python
        out.append(
            {
                "within": within,
                "selection": selection,
                "estimation": estimation,
                "within_structure": sigma2_w,
            }
        )
```

Give it the signature `def variance_terms(model, *, alpha_v: float = ALPHA_V, alpha_t: float = ALPHA_T) -> list[dict[str, np.ndarray]]:` and a docstring:

```python
    """The four terms of the predictive variance, per level (raw level index).

    ``predictive_variance`` is ``within + selection_weight * selection +
    estimation + within_structure`` of these; exposed so an ablation is a sum
    of terms rather than a flag per term (Study F spec, section 3.1).
    ``within_structure`` is the pooled sigma2_w, ``(d,)``, broadcast over
    classes.
    """
```

Then `predictive_variance` keeps its signature and docstring, and its body becomes:

```python
    return [
        t["within"]
        + selection_weight * t["selection"]
        + t["estimation"]
        + t["within_structure"]
        for t in variance_terms(model, alpha_v=alpha_v, alpha_t=alpha_t)
    ]
```

The sum is in the same order as before (`within + selection_weight * selection + estimation + sigma2_w`), which is what keeps it bit-identical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_uncertainty.py tests/test_within_structure.py tests/test_predict.py tests/test_stereo_refines_blind.py -q`
Expected: PASS. The existing hand-computed tests in `test_uncertainty.py` pin the values, so they must pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/sieve/uncertainty.py tests/test_uncertainty.py
git commit -m "refactor(uncertainty): expose the predictive variance's four terms"
```

---

### Task 2: Arms, per-atom variances and metrics

**Files:**
- Create: `experiments/experiments/calibration.py`
- Test: `experiments/tests/test_calibration.py` (create)

**Interfaces:**
- Consumes: `variance_terms` (Task 1).
- Produces:
  - `METRICS_FILE = "calibration_metrics.json"`, `PREFIX = "calibration"` and `COVERAGE = (0.5, 0.9, 0.95, 0.99)`.
  - `Arm(name, alpha_v, selection_weight, estimation, within_structure)` (frozen dataclass) and `ARMS: tuple[Arm, ...]`, in the table's order.
  - `arm_variances(model, prediction) -> dict[str, np.ndarray]`: per-atom σ², shape `(n,)`, keyed by arm name.
  - `calibration_metrics(*, raw, target, sigma2, k_star, conf, molecule_value, depth) -> dict[str, float]`.
  - `normalised_errors(normalised, target) -> tuple[float, float]`, which returns (rmse, mae).
  - `group_percentile_rank(values, groups) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `experiments/tests/test_calibration.py`:

```python
"""Study F: calibration of Sieve's predictive variance (spec 2026-09-25)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from experiments.calibration import (
    ARMS,
    arm_variances,
    calibration_metrics,
    group_percentile_rank,
)


def _chain(n=24, graphs=4, seed=0):
    """Disjoint alternating C/H paths with random targets -- built here, not
    imported: `tests.helpers` resolves to this directory's own module or the
    root suite's depending on import order."""
    from sieve.batch import NodeBatch

    per = n // graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for i in range(per - 1):
            src += [off + i, off + i + 1]
            dst += [off + i + 1, off + i]
        gid += [g] * per
    return NodeBatch(
        node_attrs=(np.arange(n) % 2).reshape(-1, 1).astype(np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attrs=np.ones((len(src), 1), np.int64),
        graph_id=np.array(gid, np.int64),
        y=np.random.default_rng(seed).normal(size=(n, 1)),
    )


def _config(**kw):
    from sieve.config import SieveConfig

    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes={"element": {"C": 0, "H": 1}},
        edge_codes={"bond_type": {"SINGLE": 0}},
        max_wl_depth=2,
        predictive_variance=True,
        **kw,
    )


def _model_and_prediction(**kw):
    import sieve

    b = _chain()
    m = sieve.fit(b, _config(**kw)).with_within_structure(2.0e-3, 20)
    # one unseen element code, so the unmatched fallback is exercised
    oov = dataclasses.replace(
        b, node_attrs=np.where(np.arange(b.n_nodes)[:, None] == 0, 7, b.node_attrs)
    )
    return m, sieve.predict_detailed(m, oov)


# ------------------------------------------------------------------ arms --


def test_the_arm_table_is_the_spec_s():
    assert [a.name for a in ARMS] == [
        "form_b",
        "no_sigma2_w",
        "alpha_v_30",
        "no_selection",
        "no_estimation",
    ]
    by = {a.name: a for a in ARMS}
    assert (by["form_b"].alpha_v, by["form_b"].selection_weight) == (10.0, 0.5)
    assert by["alpha_v_30"].alpha_v == 30.0
    assert by["no_selection"].selection_weight == 0.0
    assert not by["no_estimation"].estimation
    assert not by["no_sigma2_w"].within_structure


def test_form_b_is_the_library_s_predictive_variance_bit_for_bit():
    m, p = _model_and_prediction()
    assert p.matched_level[0] == -1  # the unmatched atom is there
    assert p.predictive_variance is not None
    np.testing.assert_array_equal(arm_variances(m, p)["form_b"], p.predictive_variance[:, 0])


def test_no_sigma2_w_drops_it_everywhere():
    m, p = _model_and_prediction()
    v = arm_variances(m, p)
    s2w = float(m.within_variance[0])
    matched = p.matched_level >= 0
    np.testing.assert_allclose(v["no_sigma2_w"][matched], v["form_b"][matched] - s2w, rtol=1e-12)
    np.testing.assert_allclose(v["no_sigma2_w"][~matched], m.global_msd[0])


def test_alpha_v_30_is_the_library_at_alpha_v_30():
    from sieve.uncertainty import predictive_variance

    m, p = _model_and_prediction()
    table = predictive_variance(m, alpha_v=30.0)
    got = arm_variances(m, p)["alpha_v_30"]
    backoff = np.asarray(m.config.backoff_path)
    for i in np.flatnonzero(p.matched_level >= 0):
        assert got[i] == table[backoff[p.matched_level[i]]][p.class_id[i], 0]


def test_form_b_equals_the_library_under_a_stereo_track():
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes, from_rdkit

    import sieve

    smiles = ["C/C=C/C", r"C/C=C\C", "C/C=C/CC", r"C/C=C\CC", "CCCC", "CC(C)C"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    codes, edges = build_codes(mols, ["element"])
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=3,
        stereo=("cis_trans",),
        predictive_variance=True,
    )
    b = from_rdkit(mols, config=cfg)
    b = dataclasses.replace(b, y=np.random.default_rng(1).normal(size=(b.n_nodes, 1)))
    m = sieve.fit(b, cfg).with_within_structure(1.0e-3, 10)
    p = sieve.predict_detailed(m, b)
    assert p.predictive_variance is not None
    np.testing.assert_array_equal(arm_variances(m, p)["form_b"], p.predictive_variance[:, 0])


# --------------------------------------------------------------- metrics --


def _synthetic(n_conf=400, per=10, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    n = n_conf * per
    conf = np.repeat(np.arange(n_conf), per)
    sigma2 = rng.uniform(0.5, 2.0, size=n) * 1e-3
    target = rng.normal(size=n) * 0.1
    raw = target + rng.normal(size=n) * np.sqrt(sigma2) * scale
    k_star = rng.integers(0, 3, size=n)
    k_star[:5] = -1
    return {
        "raw": raw,
        "target": target,
        "sigma2": sigma2,
        "k_star": k_star,
        "conf": conf,
        "molecule_value": np.zeros(n_conf),
        "depth": 3,
    }


def test_calibrated_errors_score_as_calibrated():
    m = calibration_metrics(**_synthetic())
    assert m["ez2"] == pytest.approx(1.0, abs=0.05)
    for c in (50, 90, 95, 99):
        assert m[f"cov{c}"] == pytest.approx(c / 100, abs=0.02)


def test_doubling_sigma_quarters_ez2():
    s = _synthetic()
    a = calibration_metrics(**s)
    b = calibration_metrics(**{**s, "sigma2": 4 * s["sigma2"]})
    assert b["ez2"] == pytest.approx(a["ez2"] / 4, rel=1e-12)


def test_nll_is_the_gaussian_mean():
    s = _synthetic(n_conf=3, per=2)
    e = s["raw"] - s["target"]
    want = np.mean(0.5 * (np.log(2 * np.pi * s["sigma2"]) + e**2 / s["sigma2"]))
    assert calibration_metrics(**s)["nll"] == pytest.approx(want, rel=1e-12)


def test_ez2_by_radius_only_for_radii_that_occur():
    s = _synthetic()
    m = calibration_metrics(**s)
    assert {k for k in m if k.startswith("ez2_k")} == {"ez2_k0", "ez2_k1", "ez2_k2"}
    e = s["raw"] - s["target"]
    sel = s["k_star"] == 1
    assert m["ez2_k1"] == pytest.approx(np.mean(e[sel] ** 2 / s["sigma2"][sel]), rel=1e-12)


def test_normalised_errors_use_the_variance_weighted_normaliser():
    from experiments.normalize import variance_weighted_normalize

    s = _synthetic(n_conf=20)
    x = variance_weighted_normalize(
        s["raw"], np.sqrt(s["sigma2"]), s["molecule_value"], s["conf"], 20
    )
    m = calibration_metrics(**s)
    assert m["norm_rmse"] == pytest.approx(np.sqrt(np.mean((x - s["target"]) ** 2)), rel=1e-12)
    assert m["norm_mae"] == pytest.approx(np.mean(np.abs(x - s["target"])), rel=1e-12)


def test_group_percentile_rank_matches_a_brute_force_average_rank():
    from scipy.stats import rankdata

    rng = np.random.default_rng(4)
    groups = rng.integers(0, 6, size=60)
    values = rng.integers(0, 5, size=60).astype(float)  # ties on purpose
    got = group_percentile_rank(values, groups)
    for g in np.unique(groups):
        sel = groups == g
        np.testing.assert_allclose(got[sel], rankdata(values[sel]) / sel.sum())


def test_rho_is_one_when_sigma_orders_the_errors_exactly():
    s = _synthetic()
    e = np.abs(s["raw"] - s["target"])
    m = calibration_metrics(**{**s, "sigma2": e**2 + 1e-9})
    assert m["rho"] == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'experiments.calibration'`.

- [ ] **Step 3: Implement**

Create `experiments/experiments/calibration.py`:

```python
"""Study F: calibration of Sieve's predictive variance.

Spec: docs/superpowers/specs/2026-09-25-study-f-calibration-design.md.

The shipped predictive variance (form B, within-structure-variance spec) and
four ablations, each dropping one ingredient, are scored on the Study B
incumbent's runs: calibration (NLL, E[z^2], coverage, E[z^2] by matched
radius, within-conformer ranking) and the variance-weighted normalisation they
drive. Every arm is a per-atom variance from one prediction, so all five are
scored on the same atoms. Scores go to a sidecar beside each run's own
metrics, like Study D's subset scores (``stereo_subsets``), so a run's record
is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

METRICS_FILE = "calibration_metrics.json"
PREFIX = "calibration"
COVERAGE = (0.5, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class Arm:
    """One predictive variance: which terms, at which constants."""

    name: str
    alpha_v: float
    selection_weight: float
    estimation: bool
    within_structure: bool


ARMS = (
    Arm("form_b", 10.0, 0.5, True, True),
    Arm("no_sigma2_w", 10.0, 0.5, True, False),
    Arm("alpha_v_30", 30.0, 0.5, True, True),
    Arm("no_selection", 10.0, 0.0, True, True),
    Arm("no_estimation", 10.0, 0.5, False, True),
)


def arm_variances(model: Any, prediction: Any) -> dict[str, np.ndarray]:
    """Per-atom sigma^2 of every arm, ``(n,)`` each, for a ``d == 1`` model.

    ``prediction.matched_level`` is a backoff-path position, while the
    variance tables are indexed by raw level, so positions are mapped through
    ``config.backoff_path`` first. Unmatched atoms take ``global_msd``, plus
    sigma2_w in the arms that carry it -- ``predict``'s own fallback.
    """
    from sieve.uncertainty import variance_terms

    backoff = np.asarray(model.config.backoff_path, dtype=np.int64)
    pos = np.asarray(prediction.matched_level)
    level = np.where(pos >= 0, backoff[np.maximum(pos, 0)], -1)
    cid = np.asarray(prediction.class_id)
    s2w = float(model.within_variance[0])
    gmsd = float(model.global_msd[0])
    terms: dict[float, list[dict[str, np.ndarray]]] = {}
    out: dict[str, np.ndarray] = {}
    for arm in ARMS:
        if arm.alpha_v not in terms:
            terms[arm.alpha_v] = variance_terms(model, alpha_v=arm.alpha_v)
        s2 = np.full(pos.shape[0], gmsd + (s2w if arm.within_structure else 0.0))
        for k in np.unique(level[level >= 0]):
            t = terms[arm.alpha_v][int(k)]
            # the same order as predictive_variance's sum, so form_b is
            # bit-identical to it
            table = t["within"] + arm.selection_weight * t["selection"]
            if arm.estimation:
                table = table + t["estimation"]
            if arm.within_structure:
                table = table + t["within_structure"]
            sel = level == k
            s2[sel] = table[cid[sel], 0]
        out[arm.name] = s2
    return out


def group_percentile_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each value's average rank within its group, divided by the group's size
    (pandas' ``groupby().rank(pct=True)``, without pandas)."""
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    n = values.shape[0]
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    size = np.diff(np.r_[start, n])
    pos = np.arange(n) - np.repeat(start, size)
    new_run = np.r_[True, (np.diff(g) != 0) | (np.diff(v) != 0)]
    run_id = np.cumsum(new_run) - 1
    first = np.flatnonzero(new_run)
    run_len = np.diff(np.r_[first, n])
    avg_rank = pos[first] + (run_len - 1) / 2.0 + 1.0
    out = np.empty(n)
    out[order] = avg_rank[run_id] / np.repeat(size, size)
    return out


def normalised_errors(normalised: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    d = np.asarray(normalised) - np.asarray(target)
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def calibration_metrics(
    *,
    raw: np.ndarray,
    target: np.ndarray,
    sigma2: np.ndarray,
    k_star: np.ndarray,
    conf: np.ndarray,
    molecule_value: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """One arm's calibration and normalisation scores (spec section 2.3).

    ``conf`` is each atom's conformer index, ``0..n_conformers-1``, and
    ``molecule_value`` each conformer's constrained total.
    """
    from scipy.stats import norm

    from experiments.normalize import variance_weighted_normalize

    e = np.asarray(raw) - np.asarray(target)
    z2 = e * e / sigma2
    out = {
        "nll": float(np.mean(0.5 * (np.log(2 * np.pi * sigma2) + z2))),
        "ez2": float(np.mean(z2)),
    }
    for c in COVERAGE:
        out[f"cov{round(100 * c)}"] = float(np.mean(z2 < norm.ppf(0.5 + c / 2) ** 2))
    for k in range(depth + 1):
        sel = k_star == k
        if sel.any():
            out[f"ez2_k{k}"] = float(np.mean(z2[sel]))
    ra = group_percentile_rank(np.abs(e), conf)
    rs = group_percentile_rank(sigma2, conf)
    out["rho"] = float(np.corrcoef(ra, rs)[0, 1])
    n_conf = int(np.asarray(molecule_value).shape[0])
    x = variance_weighted_normalize(raw, np.sqrt(sigma2), molecule_value, conf, n_conf)
    out["norm_rmse"], out["norm_mae"] = normalised_errors(x, target)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments/calibration.py experiments/tests/test_calibration.py
git commit -m "feat(calibration): Study F's arms, per-atom variances and metrics"
```

---

### Task 3: `SieveTrainModels`, the fold-model assembly shared by `run_sieve_cv` and Study F

**Files:**
- Modify: `experiments/experiments/cv.py`. In `run_sieve_cv`, the block from `paths_or_none = {` down to the end of the `if model_cache is not None:` sidecar block is replaced. So is the `train_models = None ... _save_train_models(...)` block inside the repeat loop.
- Test: `experiments/tests/test_cv.py` (append)

**Interfaces:**
- Produces: `class SieveTrainModels`, constructed with `SieveTrainModels(*, store: str, n_shards: int, k: int, config_label: str, fit_depth: int, model_cache: str | Path | None = None, runs_root: Path = DEFAULT_RUNS_ROOT, stores_root: Path | None = None)`, and called as `__call__(self, repeat: int, plan: CVPlan) -> list[Any]`. The call returns the K training models of `repeat`, untruncated, without training floors attached.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/tests/test_cv.py`:

```python
def test_train_models_from_shards_equal_the_cached_ones(tmp_path):
    """SieveTrainModels, cache miss then cache hit, gives the same models --
    and the same ones run_sieve_cv's own runs were scored with."""
    from experiments.cv import SieveTrainModels, build_cv_plan, shard_ids

    kw = _sieve_cv_kwargs(tmp_path)
    ids = shard_ids(kw["n_shards"])
    plan = build_cv_plan(ids, k=kw["k"], repeat=0)
    common = {
        "store": kw["store"],
        "n_shards": kw["n_shards"],
        "k": kw["k"],
        "config_label": kw["config_label"],
        "fit_depth": 1,
        "runs_root": kw["runs_root"],
        "stores_root": kw["stores_root"],
    }
    uncached = SieveTrainModels(**common)(0, plan)
    cache = tmp_path / "cache"
    miss = SieveTrainModels(model_cache=cache, **common)(0, plan)
    hit = SieveTrainModels(model_cache=cache, **common)(0, plan)
    assert len(uncached) == len(miss) == len(hit) == kw["k"]
    for a, b, c in zip(uncached, miss, hit, strict=True):
        assert a.global_count == b.global_count == c.global_count
        for la, lb, lc in zip(a.levels, b.levels, c.levels, strict=True):
            np.testing.assert_array_equal(la.count, lb.count)
            np.testing.assert_array_equal(lb.count, lc.count)
            np.testing.assert_array_equal(lb.mean, lc.mean)


def test_train_models_refuse_a_missing_shard_fit(tmp_path):
    from experiments.cv import SieveTrainModels

    kw = _sieve_cv_kwargs(tmp_path)
    with pytest.raises(FileNotFoundError, match="no shard fit"):
        SieveTrainModels(
            store=kw["store"],
            n_shards=kw["n_shards"],
            k=kw["k"],
            config_label="absent",
            fit_depth=1,
            runs_root=kw["runs_root"],
            stores_root=kw["stores_root"],
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_cv.py -q -k "train_models"`
Expected: FAIL with `ImportError: cannot import name 'SieveTrainModels'`.

- [ ] **Step 3: Implement**

Add above `run_sieve_cv` in `experiments/experiments/cv.py`. The body is `run_sieve_cv`'s current code, moved:

```python
class SieveTrainModels:
    """The K training models of a repeat: from the model cache when it holds
    the whole repeat, otherwise merged from the shard fits (``sieve_fold`` per
    group, then ``leave_one_group_out``) and written back to the cache.

    Untruncated, at ``fit_depth``, and without training floors: those are the
    caller's (``truncate_model``, ``_attach_training_floors``). Shared by
    ``run_sieve_cv`` and Study F (``experiments.calibration``), so both read the
    very models the runs were scored with.
    """

    def __init__(
        self,
        *,
        store: str,
        n_shards: int,
        k: int,
        config_label: str,
        fit_depth: int,
        model_cache: str | Path | None = None,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        stores_root: Path | None = None,
    ) -> None:
        import sieve

        ids = shard_ids(n_shards)
        paths_or_none = {
            s: _shard_fit_done(runs_root, sieve_shard_batch_id(config_label, fit_depth, s))
            for s in ids
        }
        missing = [s for s, p in paths_or_none.items() if p is None]
        if missing:
            raise FileNotFoundError(
                f"no shard fit for {config_label!r} at depth {fit_depth}, shard(s) "
                f"{missing}; run run_sieve_shard_fits(max_depth={fit_depth}) first"
            )
        self._shard_paths = {s: p for s, p in paths_or_none.items() if p is not None}
        # Loaded lazily: a fully cached repeat never needs the shards at all, and
        # opening 50 of them is the first 2.6 s of the 123 s this cache exists to
        # avoid. One is still read eagerly below, to pin schema_version.
        self._models_by_shard: dict[str, Any] = {}
        self._cache_dir: Path | None = None
        self._sidecar: dict[str, Any] = {}
        if model_cache is not None:
            reference = sieve.SieveModel.load(self._shard_paths[ids[0]])
            self._cache_dir = cv_model_cache_dir(
                model_cache, "sieve", f"{config_label}-w{fit_depth}-n{n_shards}-k{k}"
            )
            # schema_version pins the vocabulary and depth the fits were built
            # with; a cached model that disagrees is not the same model.
            self._sidecar = {
                "schema_version": reference.config.schema_version,
                **store_identity(store, stores_root=stores_root),
            }

    def _shards(self) -> dict[str, Any]:
        import sieve

        if not self._models_by_shard:
            self._models_by_shard.update(
                {s: sieve.SieveModel.load(p) for s, p in self._shard_paths.items()}
            )
        return self._models_by_shard

    def __call__(self, repeat: int, plan: CVPlan) -> list[Any]:
        import sieve
        from sieve.merge import fold as sieve_fold
        from sieve.merge import merge_models

        if self._cache_dir is not None:
            cached = _load_cached_train_models(
                self._cache_dir,
                repeat=repeat,
                plan=plan,
                load_one=sieve.SieveModel.load,
                sidecar_extra=self._sidecar,
            )
            if cached is not None:
                return cached
        # Merged once, at fit_depth; every requested depth is a truncation of
        # these, not a separate merge of a separate shard set.
        shards = self._shards()
        group_models = [
            sieve_fold([shards[s] for s in g], shards[g[0]].config) for g in plan.groups
        ]
        models = leave_one_group_out(group_models, merge=merge_models)
        if self._cache_dir is not None:
            _save_train_models(
                self._cache_dir,
                repeat=repeat,
                plan=plan,
                models=models,
                save_one=lambda m, path: m.save(path),
                sidecar_extra=self._sidecar,
            )
        return models
```

In `run_sieve_cv`, replace the moved block (`paths_or_none = {` down to and including the sidecar dict) with:

```python
    train_models_for = SieveTrainModels(
        store=store,
        n_shards=n_shards,
        k=k,
        config_label=config_label,
        fit_depth=fit_depth,
        model_cache=model_cache,
        runs_root=runs_root,
        stores_root=stores_root,
    )
```

Inside the repeat loop, replace everything from `train_models = None` through the `_save_train_models(...)` call with `train_models = train_models_for(repeat, plan)`. Keep the `_attach_training_floors` call that follows. Drop the now-unused `sieve_fold`/`merge_models` imports from `run_sieve_cv` if ruff flags them. Keep `ids = shard_ids(n_shards)`, since the function uses it below.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_cv.py -q`
Expected: PASS. Every existing `run_sieve_cv` test must pass unchanged, including the idempotence, cache-hit and assembly-matches-direct-fit tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments/cv.py experiments/tests/test_cv.py
git commit -m "refactor(cv): SieveTrainModels, the fold-model assembly as its own unit"
```

---

### Task 4: Scoring runs, staleness, and the aggregate sidecar

**Files:**
- Modify: `experiments/experiments/calibration.py` (append)
- Modify: `experiments/experiments/aggregate.py:137-146` (the sidecar merge)
- Test: `experiments/tests/test_calibration.py` (append)

**Interfaces:**
- Consumes:
  - from Tasks 2–3: `arm_variances`, `calibration_metrics`, `normalised_errors` and `SieveTrainModels`;
  - existing: `experiments.cv.build_cv_plan`, `shard_ids`, `load_shards`, `concat_molecule_sets`, `truncate_model` and `_attach_training_floors`; `sieve.io.rdkit_adapter.from_rdkit`; `experiments.normalize.equal_weighted_normalize`.
- Produces:
  - `selected_runs(runs_root, *, experiment, method, depth, repeats=None) -> list[Path]`, sorted by (repeat, fold);
  - `missing_scores(runs_root, *, experiment, method, depth, repeats=None) -> list[Path]`;
  - `score_run(run: Path, model: Any, held_out: MoleculeSet, *, n_jobs: int | None = None) -> dict[str, float]`;
  - `score_runs(runs_root, *, store, experiment, method, depth, n_shards, k, config_label, fit_depth, collapse, model_cache=None, stores_root=None, repeats=None, force=False, n_jobs=None) -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/tests/test_calibration.py`:

```python
# --------------------------------------------------------- scoring runs --


@pytest.fixture
def scored_cv(tmp_path):
    """A tiny Study-B-like CV experiment with saved predictions."""
    pytest.importorskip("pandas")
    from experiments.cv import run_sieve_cv

    from experiments.tests.test_cv import _sieve_cv_kwargs

    kw = _sieve_cv_kwargs(tmp_path)
    run_sieve_cv(save_predictions=True, experiment="sieve-cv", **kw)
    return kw


def _score(kw, **extra):
    from experiments.calibration import score_runs

    return score_runs(
        kw["runs_root"],
        store=kw["store"],
        experiment="sieve-cv",
        method=kw["method"],
        depth=1,
        n_shards=kw["n_shards"],
        k=kw["k"],
        config_label=kw["config_label"],
        fit_depth=1,
        collapse=False,
        stores_root=kw["stores_root"],
        **extra,
    )


def test_every_run_gets_every_arm_s_keys(scored_cv):
    import json

    from experiments.calibration import ARMS, METRICS_FILE

    written = _score(scored_cv)
    assert len(written) == scored_cv["k"]
    for run in written:
        side = json.loads((run / METRICS_FILE).read_text())
        for arm in ARMS:
            for m in ("nll", "ez2", "cov95", "rho", "norm_rmse", "norm_mae"):
                assert f"calibration/{arm.name}/{m}" in side
        assert "calibration/equal/norm_rmse" in side
        assert side["calibration/n_atoms"] > 0
        assert sum(v for k, v in side.items() if "/share_k" in k) == pytest.approx(1.0)


def test_missing_scores_reports_absent_stale_and_incomplete(scored_cv):
    import json
    import os

    from experiments.calibration import METRICS_FILE, missing_scores

    sel = {"experiment": "sieve-cv", "method": scored_cv["method"], "depth": 1}
    root = scored_cv["runs_root"]
    assert len(missing_scores(root, **sel)) == scored_cv["k"]  # absent
    runs = _score(scored_cv)
    assert missing_scores(root, **sel) == []
    assert _score(scored_cv) == []  # nothing left to do
    # stale: predictions newer than the sidecar
    pred = runs[0] / "predictions.npz"
    side = runs[0] / METRICS_FILE
    os.utime(pred, (side.stat().st_mtime + 10, side.stat().st_mtime + 10))
    # incomplete: an arm's keys missing
    d = json.loads((runs[1] / METRICS_FILE).read_text())
    (runs[1] / METRICS_FILE).write_text(
        json.dumps({k: v for k, v in d.items() if "/no_estimation/" not in k})
    )
    assert set(missing_scores(root, **sel)) == {runs[0], runs[1]}


def test_repeats_restrict_the_selection(scored_cv):
    from experiments.calibration import selected_runs

    sel = {"experiment": "sieve-cv", "method": scored_cv["method"], "depth": 1}
    assert len(selected_runs(scored_cv["runs_root"], repeats=[0], **sel)) == scored_cv["k"]
    assert selected_runs(scored_cv["runs_root"], repeats=[3], **sel) == []


def test_a_run_whose_predictions_disagree_is_refused(scored_cv):
    from experiments.calibration import selected_runs

    run = selected_runs(
        scored_cv["runs_root"], experiment="sieve-cv", method=scored_cv["method"], depth=1
    )[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    z["atom_target_pred"] = z["atom_target_pred"] + 1e-6
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="does not reproduce"):
        _score(scored_cv)


def test_a_run_whose_ids_disagree_is_refused(scored_cv):
    from experiments.calibration import selected_runs

    run = selected_runs(
        scored_cv["runs_root"], experiment="sieve-cv", method=scored_cv["method"], depth=1
    )[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    z["dash_id"] = np.roll(z["dash_id"], 1)  # dash ids are distinct per molecule
    assert len(set(z["dash_id"].tolist())) > 1
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="held-out order"):
        _score(scored_cv)


def test_aggregate_merges_the_calibration_sidecar_and_refuses_a_clash(scored_cv):
    import json

    from experiments.aggregate import read_runs_from_dirs
    from experiments.calibration import METRICS_FILE

    runs = _score(scored_cv)
    rows = read_runs_from_dirs(scored_cv["runs_root"], "sieve-cv")
    assert all("calibration/form_b/nll" in r.metrics for r in rows)
    d = json.loads((runs[0] / METRICS_FILE).read_text())
    d["rmse"] = 0.0  # clashes with metrics.json
    (runs[0] / METRICS_FILE).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="repeats"):
        read_runs_from_dirs(scored_cv["runs_root"], "sieve-cv")
```

Check `aggregate.read_runs_from_dirs`'s real signature and its row's metrics attribute name first (`grep -n "def read_runs_from_dirs" -A6 experiments/experiments/aggregate.py`; `grep -n "class RunRow" -A10`), and adapt the last test's call and `.metrics` to them. Also check that `run_sieve_cv` accepts `experiment=`. If `_sieve_cv_kwargs`'s runs already default to an experiment name, use that name in place of `"sieve-cv"` throughout.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py -q -k "scor or missing or repeats or refused or aggregate"`
Expected: FAIL with `ImportError` on `score_runs`, `missing_scores` and `selected_runs`.

- [ ] **Step 3: Implement scoring**

Append to `experiments/experiments/calibration.py`:

```python
# ---------------------------------------------------------------- runs ---


def _cv(run) -> dict[str, Any]:
    import json

    return json.loads((run / "manifest.json").read_text()).get("config", {}).get("cv", {})


def selected_runs(
    runs_root, *, experiment: str, method: str, depth: int, repeats=None
) -> list:
    """Run directories of ``method`` at ``depth`` under ``experiment``, by
    (repeat, fold); only ``repeats`` when given."""
    from pathlib import Path

    found = []
    for manifest in sorted(Path(runs_root).glob(f"{experiment}/*/manifest.json")):
        cv = _cv(manifest.parent)
        if cv.get("method") != method or int(cv.get("depth", -1)) != depth:
            continue
        if repeats is not None and int(cv["repeat"]) not in set(repeats):
            continue
        found.append((int(cv["repeat"]), int(cv["fold"]), manifest.parent))
    return [run for *_, run in sorted(found)]


def _complete(side: dict[str, Any]) -> bool:
    return all(f"{PREFIX}/{arm.name}/nll" in side for arm in ARMS) and (
        f"{PREFIX}/equal/norm_rmse" in side
    )


def missing_scores(runs_root, *, experiment, method, depth, repeats=None) -> list:
    """Selected runs whose sidecar is absent, older than their predictions, or
    missing an arm. A run with no predictions is reported too: it can never
    be scored, and passing over it would silently drop a paired sample."""
    import json

    stale = []
    for run in selected_runs(
        runs_root, experiment=experiment, method=method, depth=depth, repeats=repeats
    ):
        pred, side = run / "predictions.npz", run / METRICS_FILE
        if not pred.exists() or not side.exists():
            stale.append(run)
        elif side.stat().st_mtime < pred.stat().st_mtime:
            stale.append(run)
        elif not _complete(json.loads(side.read_text())):
            stale.append(run)
    return stale


def score_run(run, model: Any, held_out: Any, *, n_jobs: int | None = None) -> dict[str, float]:
    """One run's sidecar: ``model`` is the run's untruncated training model
    with its training floor attached, ``held_out`` the run's held-out set in
    the run's own order."""
    from dataclasses import replace

    import sieve
    from sieve.io.rdkit_adapter import from_rdkit

    from experiments.cv import truncate_model
    from experiments.normalize import equal_weighted_normalize

    cv = _cv(run)
    depth = int(cv["depth"])
    z = np.load(run / "predictions.npz", allow_pickle=True)
    same = np.array_equal(
        np.asarray(held_out.ids["dash_id"], dtype=str), z["dash_id"].astype(str)
    ) and np.array_equal(
        np.asarray(held_out.ids["conf_id"], dtype=str), z["conf_id"].astype(str)
    )
    if not same:
        raise ValueError(f"{run}: the held-out order differs from predictions.npz")

    m = truncate_model(model, depth)
    params = {k.split("/", 1)[1]: v for k, v in cv.items() if k.startswith("param/")}
    if params:
        m = m.with_params(**params)
    m = replace(m, config=replace(m.config, predictive_variance=True))
    p = sieve.predict_detailed(m, from_rdkit(held_out.mols, config=m.config, n_jobs=n_jobs))

    raw = p.value[:, 0]
    if np.max(np.abs(raw - z["atom_target_pred"].ravel())) > 1e-12:
        raise ValueError(f"{run}: the fold model does not reproduce its predictions")

    target = z["atom_target_true"].ravel()
    num_atoms = np.asarray(z["num_atoms"], dtype=np.int64)
    conf = np.repeat(np.arange(num_atoms.size), num_atoms)
    molecule_value = np.asarray(z["molecule_value"], dtype=np.float64)
    k_star = np.asarray(p.matched_level)

    out: dict[str, float] = {
        f"{PREFIX}/n_atoms": float(raw.size),
        f"{PREFIX}/sigma2_w": float(m.within_variance[0]),
    }
    for k in range(-1, depth + 1):
        share = float(np.mean(k_star == k))
        if share:
            out[f"{PREFIX}/share_k{k}"] = share
    for name, s2 in arm_variances(m, p).items():
        scores = calibration_metrics(
            raw=raw,
            target=target,
            sigma2=s2,
            k_star=k_star,
            conf=conf,
            molecule_value=molecule_value,
            depth=depth,
        )
        out.update({f"{PREFIX}/{name}/{key}": v for key, v in scores.items()})
    x = equal_weighted_normalize(raw, np.ones_like(raw), molecule_value, conf, num_atoms.size)
    out[f"{PREFIX}/equal/norm_rmse"], out[f"{PREFIX}/equal/norm_mae"] = normalised_errors(
        x, target
    )
    return out


def score_runs(
    runs_root,
    *,
    store: str,
    experiment: str,
    method: str,
    depth: int,
    n_shards: int,
    k: int,
    config_label: str,
    fit_depth: int,
    collapse: bool,
    model_cache=None,
    stores_root=None,
    repeats=None,
    force: bool = False,
    n_jobs: int | None = None,
) -> list:
    """Write the sidecar of every selected run that needs one, and return
    those runs. Fold models come from ``SieveTrainModels``, the same assembly
    ``run_sieve_cv`` scored the runs with; training floors are attached as
    ``run_sieve_cv`` does (``collapse`` must be the CV's own)."""
    import json

    from experiments.cv import (
        SieveTrainModels,
        _attach_training_floors,
        build_cv_plan,
        concat_molecule_sets,
        load_shards,
        shard_ids,
    )

    sel = {"experiment": experiment, "method": method, "depth": depth, "repeats": repeats}
    todo = selected_runs(runs_root, **sel) if force else missing_scores(runs_root, **sel)
    if not todo:
        return []
    for run in todo:
        if not (run / "predictions.npz").exists():
            raise FileNotFoundError(f"{run} has no predictions.npz; rerun it with --save-predictions")
    ids = shard_ids(n_shards)
    train_models_for = SieveTrainModels(
        store=store,
        n_shards=n_shards,
        k=k,
        config_label=config_label,
        fit_depth=fit_depth,
        model_cache=model_cache,
        runs_root=runs_root,
        stores_root=stores_root,
    )
    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    written = []
    by_repeat: dict[int, list] = {}
    for run in todo:
        by_repeat.setdefault(int(_cv(run)["repeat"]), []).append(run)
    for repeat, runs in sorted(by_repeat.items()):
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        models = _attach_training_floors(
            train_models_for(repeat, plan), plan, store, collapse=collapse, stores_root=stores_root
        )
        for run in runs:
            cv = _cv(run)
            fold = int(cv["fold"])
            group = plan.groups[fold]
            if cv.get("held_out_shards") != ",".join(group):
                raise ValueError(f"{run}: held_out_shards disagrees with the CV plan")
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])
            scores = score_run(run, models[fold], held_out, n_jobs=n_jobs)
            (run / METRICS_FILE).write_text(json.dumps(scores, indent=1, sort_keys=True))
            written.append(run)
    return written
```

Check the real key name of the held-out shard list in a run manifest. It is `held_out_shards`, a comma-joined string in `plan.groups[fold]` order, as the Study B manifest shows. If `run_sieve_cv` writes it differently in tests, match that.

- [ ] **Step 4: Implement the aggregate merge**

In `experiments/experiments/aggregate.py`, replace the single-sidecar block with a loop over both sidecars:

```python
        # Metrics scored after the run from its saved predictions (Study D's
        # stereo subsets, experiments.stereo_subsets; Study F's calibration,
        # experiments.calibration) live in sidecars, so the run's own record
        # is never rewritten. A key in two places would make the run say two
        # things, so it is refused rather than overwritten.
        for name in ("subset_metrics.json", "calibration_metrics.json"):
            sidecar = run_dir / name
            if sidecar.exists():
                extra = json.loads(sidecar.read_text())
                clash = sorted(set(extra) & set(raw_metrics))
                if clash:
                    raise ValueError(f"{sidecar} repeats metrics already read: {clash}")
                raw_metrics = {**raw_metrics, **extra}
```

The message changes from "repeats metrics.json's" to "repeats metrics already read". If an existing test matches the old wording (`grep -rn "repeats metrics.json" experiments/tests`), update that test's `match=` to `"repeats"`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py experiments/tests/test_aggregate.py experiments/tests/test_stereo_subsets.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments/calibration.py experiments/experiments/aggregate.py experiments/tests/test_calibration.py
git commit -m "feat(calibration): score Study B runs into a calibration sidecar"
```

---

### Task 5: The report and the CLI

**Files:**
- Modify: `experiments/experiments/calibration.py` (append `calibration_report`)
- Modify: `experiments/experiments/cli.py` (two handlers and two subparsers, beside `score-stereo-subsets`/`stereo-report`)
- Test: `experiments/tests/test_calibration.py` (append)

**Interfaces:**
- Consumes: `selected_runs`, `ARMS`, `METRICS_FILE`, `PREFIX` (Tasks 2 and 4); `stereo_subsets.paired_difference`.
- Produces:
  - `calibration_report(runs_root, *, experiment, method, depth, k) -> str`;
  - CLI `score-calibration` and `calibration-report`.

- [ ] **Step 1: Write the failing tests**

Append to `experiments/tests/test_calibration.py`:

```python
# ---------------------------------------------------------------- report --


def _fake_runs(tmp_path, n_repeats=2, k=5, seed=0):
    """Hand-written runs: a manifest, metrics.json and a calibration sidecar."""
    import json

    from experiments.calibration import ARMS, METRICS_FILE

    rng = np.random.default_rng(seed)
    root = tmp_path / "runs"
    for r in range(n_repeats):
        for f in range(k):
            run = root / "exp" / f"r{r}-f{f}"
            run.mkdir(parents=True)
            cv = {"method": "m", "depth": 2, "repeat": r, "fold": f}
            (run / "manifest.json").write_text(json.dumps({"config": {"cv": cv}}))
            (run / "metrics.json").write_text(json.dumps({"rmse": 0.02, "mae": 0.01}))
            side = {
                "calibration/n_atoms": 100.0,
                "calibration/sigma2_w": 1e-4,
                "calibration/share_k1": 0.4,
                "calibration/share_k2": 0.6,
                "calibration/equal/norm_rmse": 0.0196 + 1e-5 * rng.normal(),
                "calibration/equal/norm_mae": 0.0109,
            }
            for i, arm in enumerate(ARMS):
                for key in ("nll", "ez2", "cov50", "cov90", "cov95", "cov99", "rho",
                            "norm_rmse", "norm_mae", "ez2_k1", "ez2_k2"):
                    side[f"calibration/{arm.name}/{key}"] = 1.0 + 0.01 * i + 1e-3 * rng.normal()
            (run / METRICS_FILE).write_text(json.dumps(side))
    return root


def test_the_report_has_its_blocks_and_is_deterministic(tmp_path):
    from experiments.calibration import calibration_report

    root = _fake_runs(tmp_path)
    kw = {"experiment": "exp", "method": "m", "depth": 2, "k": 5}
    a = calibration_report(root, **kw)
    assert a == calibration_report(root, **kw)
    for heading in ("means over all 10 samples", "E[z^2] by matched radius",
                    "paired differences, all 10 samples", "paired differences, repeats 1-"):
        assert heading in a
    for arm in ("form_b", "no_sigma2_w", "alpha_v_30", "no_selection", "no_estimation"):
        assert arm in a
    assert "no_sigma2_w at alpha_v = 30" in a


def test_the_report_s_differences_are_paired_difference_s(tmp_path):
    import json

    from experiments.calibration import METRICS_FILE, calibration_report, selected_runs
    from experiments.stereo_subsets import paired_difference

    root = _fake_runs(tmp_path)
    runs = selected_runs(root, experiment="exp", method="m", depth=2)
    sides = [json.loads((r / METRICS_FILE).read_text()) for r in runs]
    a = np.array([s["calibration/form_b/nll"] for s in sides])
    b = np.array([s["calibration/alpha_v_30/nll"] for s in sides])
    d = paired_difference(a, b, k=5)
    text = calibration_report(root, experiment="exp", method="m", depth=2, k=5)
    assert f"{d.mean:+.2e}" in text and f"[{d.lo:+.2e}, {d.hi:+.2e}]" in text


def test_score_calibration_check_is_scoped_by_repeats(scored_cv, monkeypatch):
    from experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", scored_cv["runs_root"])
    monkeypatch.setattr(cli, "DEFAULT_STORES_ROOT", scored_cv["stores_root"])
    base = [
        "score-calibration", scored_cv["store"], "--experiment", "sieve-cv",
        "--method", scored_cv["method"], "--depth", "1", "--k", str(scored_cv["k"]),
        "--n-shards", str(scored_cv["n_shards"]), "--config-label", scored_cv["config_label"],
        "--fit-depth", "1",
    ]
    assert cli.main([*base, "--check"]) == 1  # unscored, and nothing written
    assert cli.main([*base, "--check", "--repeats", "3"]) == 0  # no runs in repeat 3
    assert cli.main(base) == 0
    assert cli.main([*base, "--check"]) == 0
```

Check how `cli.main` is invoked in `experiments/tests/test_cli.py` (its entry point name and whether it returns an int or raises `SystemExit`), and whether the stores root is read from `DEFAULT_STORES_ROOT`. Adapt the last test's calls to that.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py -q -k "report or check_is_scoped"`
Expected: FAIL with `ImportError` on `calibration_report`, and argparse rejecting `score-calibration`.

- [ ] **Step 3: Implement the report**

Append to `experiments/experiments/calibration.py`:

```python
# -------------------------------------------------------------- report ---

_MEAN_COLS = ("nll", "ez2", "cov50", "cov90", "cov95", "cov99", "rho", "norm_rmse", "norm_mae")
_DIFF_METRICS = ("nll", "norm_rmse", "norm_mae")


def _fmt(metric: str, v: float) -> str:
    return f"{v:10.6f}" if metric.startswith("norm_") else f"{v:10.4f}"


def calibration_report(runs_root, *, experiment: str, method: str, depth: int, k: int) -> str:
    """Study F's report (spec section 4), from the selected runs' sidecars."""
    import json

    from experiments.stereo_subsets import paired_difference

    runs = selected_runs(runs_root, experiment=experiment, method=method, depth=depth)
    missing = [r for r in runs if not (r / METRICS_FILE).exists()]
    if not runs or missing:
        raise FileNotFoundError(f"unscored runs: {missing or 'none selected'}")
    sides = [json.loads((r / METRICS_FILE).read_text()) for r in runs]
    own = [json.loads((r / "metrics.json").read_text()) for r in runs]
    repeat = np.array([int(_cv(r)["repeat"]) for r in runs])
    n = len(runs)

    def col(key: str, rows=None) -> np.ndarray:
        rows = range(n) if rows is None else rows
        return np.array([sides[i][key] for i in rows], np.float64)

    s2w = col(f"{PREFIX}/sigma2_w")
    lines = [
        "Study F: calibration of Sieve's predictive variance",
        f"samples: {experiment} / {method} at depth {depth}, n = {n} "
        f"(repeats {repeat.min()}-{repeat.max()} x {k} folds); "
        f"pooled sigma2_w {s2w.min():.4e}..{s2w.max():.4e}",
        "the old three-term variance is no_sigma2_w at alpha_v = 30 "
        "(within-structure-variance spec, section 1)",
        "",
        f"== means over all {n} samples ==",
        f"{'arm':<16}" + "".join(f"{c:>10}" for c in _MEAN_COLS),
    ]
    for arm in ARMS:
        lines.append(
            f"{arm.name:<16}"
            + "".join(_fmt(c, col(f"{PREFIX}/{arm.name}/{c}").mean()) for c in _MEAN_COLS)
        )
    pad = " " * 10 * (len(_MEAN_COLS) - 2)
    lines.append(
        f"{'equal weights':<16}{pad}"
        + _fmt("norm_rmse", col(f"{PREFIX}/equal/norm_rmse").mean())
        + _fmt("norm_mae", col(f"{PREFIX}/equal/norm_mae").mean())
    )
    lines.append(
        f"{'unnormalised':<16}{pad}"
        + _fmt("norm_rmse", float(np.mean([o["rmse"] for o in own])))
        + _fmt("norm_mae", float(np.mean([o["mae"] for o in own])))
    )

    radii = [r for r in range(-1, depth + 1) if all(f"{PREFIX}/share_k{r}" in s for s in sides)]
    lines += ["", "== E[z^2] by matched radius k* (k* = -1: unmatched) ==",
              f"{'':<16}" + "".join(f"{'k' + str(r):>10}" for r in radii)]
    lines.append(
        f"{'atom share':<16}" + "".join(_fmt("x", col(f"{PREFIX}/share_k{r}").mean()) for r in radii)
    )
    for arm in ARMS:
        cells = []
        for r in radii:
            key = f"{PREFIX}/{arm.name}/ez2_k{r}"
            cells.append(_fmt("x", col(key).mean()) if all(key in s for s in sides) else f"{'-':>10}")
        lines.append(f"{arm.name:<16}" + "".join(cells))

    def diff_block(title: str, rows) -> list[str]:
        out = ["", f"== {title} ==",
               f"{'comparison':<28} {'metric':<10} {'change':>11} "
               f"{'95% CI (NB-corrected)':>26} {'rel.':>7} {'lower':>7}"]
        base = f"{PREFIX}/form_b/"
        for arm in ARMS[1:]:
            for m in _DIFF_METRICS:
                a, b = col(base + m, rows), col(f"{PREFIX}/{arm.name}/{m}", rows)
                out.append(_diff_line(f"{arm.name} - form_b", m, a, b, k))
        for m in ("norm_rmse", "norm_mae"):
            a, b = col(f"{PREFIX}/equal/{m}", rows), col(base + m, rows)
            out.append(_diff_line("form_b - equal", m, a, b, k))
        return out

    def _diff_line(label, metric, a, b, k):
        d = paired_difference(a, b, k=k)
        ci = f"[{d.lo:+.2e}, {d.hi:+.2e}]"
        rel = 100 * d.mean / abs(float(a.mean()))
        return (f"{label:<28} {metric:<10} {d.mean:>+11.2e} {ci:>26} "
                f"{rel:>+6.2f}% {d.n_better:>3}/{d.n:<3}")

    lines += diff_block(f"paired differences, all {n} samples", None)
    later = [i for i in range(n) if repeat[i] > 0]
    if len(later) > 1:
        lines += diff_block(
            f"paired differences, repeats 1-{repeat.max()} ({len(later)} samples; "
            "alpha_v was tuned on repeat 0)",
            later,
        )
    return "\n".join(lines) + "\n"
```

Move `_diff_line` above `diff_block` if ty flags the forward reference, since it is used only at call time. "lower" counts samples where the second value is lower: the ablation below `form_b`, or `form_b` below `equal`.

- [ ] **Step 4: Implement the CLI**

In `experiments/experiments/cli.py`, beside `_cmd_score_stereo_subsets`:

```python
def _cmd_score_calibration(args: argparse.Namespace) -> int:
    from experiments.calibration import missing_scores, score_runs

    repeats = None if args.repeats is None else [int(r) for r in args.repeats.split(",")]
    sel = {"experiment": args.experiment, "method": args.method, "depth": args.depth,
           "repeats": repeats}
    if args.check:
        missing = missing_scores(DEFAULT_RUNS_ROOT, **sel)
        for run in missing:
            print(f"unscored: {run}")
        return 1 if missing else 0
    written = score_runs(
        DEFAULT_RUNS_ROOT,
        store=args.store,
        n_shards=args.n_shards,
        k=args.k,
        config_label=args.config_label,
        fit_depth=args.fit_depth,
        collapse=args.collapse,
        model_cache=args.model_cache,
        stores_root=DEFAULT_STORES_ROOT,
        force=args.force,
        n_jobs=args.n_jobs,
        **sel,
    )
    print(f"scored {len(written)} run(s)")
    return 0


def _cmd_calibration_report(args: argparse.Namespace) -> int:
    from experiments.calibration import calibration_report

    text = calibration_report(
        DEFAULT_RUNS_ROOT, experiment=args.experiment, method=args.method,
        depth=args.depth, k=args.k,
    )
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".txt").write_text(text)
        print(f"wrote {args.out.with_suffix('.txt')}")
    return 0
```

and, beside `p_score_sub`/`p_sreport`:

```python
    p_score_cal = sub.add_parser(
        "score-calibration",
        help="Study F: score a CV experiment's runs for calibration of the "
        "predictive variance (form B and its ablations) into a sidecar",
    )
    p_score_cal.add_argument("store", nargs="?", default="dash-molecules")
    p_score_cal.add_argument("--experiment", required=True)
    p_score_cal.add_argument("--method", required=True)
    p_score_cal.add_argument("--depth", type=int, required=True)
    p_score_cal.add_argument("--k", type=int, default=5)
    p_score_cal.add_argument("--n-shards", type=int, default=50)
    p_score_cal.add_argument("--config-label", default="element-eb")
    p_score_cal.add_argument("--fit-depth", type=int, default=10)
    p_score_cal.add_argument("--collapse", action="store_true",
                             help="the CV's own --collapse: attach training floors")
    p_score_cal.add_argument("--model-cache", type=Path, default=None)
    p_score_cal.add_argument("--repeats", default=None, help="comma-separated; default all")
    p_score_cal.add_argument("--n-jobs", type=int, default=None)
    p_score_cal.add_argument("--check", action="store_true",
                             help="exit 1 if any selected run is unscored or stale; write nothing")
    p_score_cal.add_argument("--force", action="store_true")
    p_score_cal.set_defaults(func=_cmd_score_calibration)

    p_cal_report = sub.add_parser(
        "calibration-report", help="Study F: the calibration report from the sidecars"
    )
    p_cal_report.add_argument("--experiment", required=True)
    p_cal_report.add_argument("--method", required=True)
    p_cal_report.add_argument("--depth", type=int, required=True)
    p_cal_report.add_argument("--k", type=int, default=5)
    p_cal_report.add_argument("--out", type=Path, default=None)
    p_cal_report.set_defaults(func=_cmd_calibration_report)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest experiments/tests/test_calibration.py experiments/tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments/calibration.py experiments/experiments/cli.py experiments/tests/test_calibration.py
git commit -m "feat(calibration): Study F report and CLI"
```

---

### Task 6: The workflow steps, the real run, the gate and the PR

**Files:**
- Modify: `experiments/workflows/cv_charges.sh` (after `step study-e-report`, before `# --- final held-out evaluation`)

**Interfaces:**
- Consumes: the CLI from Task 5; workflow variables `PYTHON`, `STORE`, `N_SHARDS`, `K`, `SIEVE_STUDY_B`, `SIEVE_METHOD`, `SIEVE_SELECTED_DEPTH`, `SIEVE_CONFIG_LABEL`, `SIEVE_MAX_DEPTH`, `COLLAPSE_FLAG`, `MODEL_CACHE_FLAG`, `STUDY_B_REPEATS`, `FIGURES_DIR`.

- [ ] **Step 1: Add the steps**

```bash
# ===========================================================================
# Study F: calibration of Sieve's predictive variance
# ===========================================================================
#
# docs/superpowers/specs/2026-09-25-study-f-calibration-design.md. Scores the
# Study B incumbent's own runs -- no refits, no new runs: each run's fold model
# is reloaded (SieveTrainModels, the assembly run_sieve_cv used), must
# reproduce the run's saved predictions, and the shipped form B and four
# ablations are scored from one prediction into a sidecar beside the run.
STUDY_F_JOBS="${STUDY_F_JOBS:-5}"          # repeats scored in parallel
STUDY_F_NJOBS="${STUDY_F_NJOBS:-4}"        # featurisation workers per repeat
STUDY_F_REPORT="$FIGURES_DIR/study-f"
STUDY_F_ARGS="--experiment $SIEVE_STUDY_B --method $SIEVE_METHOD \
  --depth $SIEVE_SELECTED_DEPTH --k $K --n-shards $N_SHARDS \
  --config-label $SIEVE_CONFIG_LABEL --fit-depth $SIEVE_MAX_DEPTH \
  --n-jobs $STUDY_F_NJOBS $COLLAPSE_FLAG $MODEL_CACHE_FLAG"
export STUDY_F_ARGS

run_one_study_f_repeat() {
  "$PYTHON" -m experiments score-calibration "$STORE" $STUDY_F_ARGS --repeats "$1"
}
export -f run_one_study_f_repeat

dispatch_study_f_repeats() {
  echo "$STUDY_B_REPEATS" | tr ',' '\n' \
    | xargs -P "$STUDY_F_JOBS" -I{} bash -c 'run_one_study_f_repeat "$1"' -- {}
}

step study-f-scores \
  "'$PYTHON' -m experiments score-calibration '$STORE' $STUDY_F_ARGS --check" -- \
  dispatch_study_f_repeats

study_f_report_is_up_to_date() {
  [ -f "$STUDY_F_REPORT.txt" ] || return 1
  [ -z "$(find "experiments/runs/$SIEVE_STUDY_B" -name calibration_metrics.json \
            -newer "$STUDY_F_REPORT.txt" -print -quit)" ]
}

run_study_f_report() {
  "$PYTHON" -m experiments calibration-report --experiment "$SIEVE_STUDY_B" \
    --method "$SIEVE_METHOD" --depth "$SIEVE_SELECTED_DEPTH" --k "$K" \
    --out "$STUDY_F_REPORT"
}

step study-f-report "study_f_report_is_up_to_date" -- run_study_f_report
```

First confirm that `SIEVE_METHOD`, `COLLAPSE_FLAG` and `MODEL_CACHE_FLAG` are defined before this point, and that Study B was run with `$COLLAPSE_FLAG` (grep `run_one_sieve_study_b`/`dispatch_sieve_repeats`). Then run `bash -n experiments/workflows/cv_charges.sh`. Expected: no output.

- [ ] **Step 2: Run the two steps on the real store**

Commit first, since the workflow may refuse a dirty tree. Then run:
`CV_UNTIL=study-f-report bash experiments/workflows/cv_charges.sh > $SCRATCH/study-f.log 2>&1`

If the workflow's earlier steps are guarded as done (they are on this machine), only `study-f-scores` and `study-f-report` run. Expected:
- `== done  study-f-scores` and `== done  study-f-report`;
- 25 `calibration_metrics.json` files;
- `experiments/docs/figures/study-f.txt` written and committed by `commit_figures`.

- [ ] **Step 3: Check acceptance (spec section 6)**

Read `study-f.txt`. The `form_b` means must be:
- NLL −2.8365;
- E[z²] 0.9588;
- cov95 0.9563;
- norm_rmse 0.018876.

`alpha_v_30` must be NLL −2.8309 and E[z²] 0.9673. `form_b − equal` on norm_rmse must be about −7.0e−4, lower in 25/25.

If a number is off in the fourth figure, stop and report it with the diff; do not adjust anything to match.

- [ ] **Step 4: The full gate**

Run the five gate commands from Global Constraints. Expected: ruff and format clean, ty with the 2 known diagnostics, and both pytest runs passing.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin study-f-calibration-spec
gh pr create --base main --head study-f-calibration-spec \
  --title "Study F: calibration of Sieve's predictive variance" \
  --body "<what; the report's headline numbers; acceptance against the phase-1 evaluation; the gate; end with the attribution line>"
```
