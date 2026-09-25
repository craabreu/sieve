# Within-Structure Variance, Phase 2 (Per-Class σ²_w): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the within-structure sums per class, use a shrunk per-class σ²_w,c in the predictive variance, and decide out of fold whether it beats the pooled σ²_w of phase 1.

**Architecture:** `FrozenLevel` gains two optional per-class arrays, `within_sse` and `within_n`. `fit_level` accrues them through exactly the same memberships as `y`: blind labels for every atom, and the aware (and mirror) pass for aware-only classes. `merge_level` adds them, and `save`/`load` persist them per level. `predictive_variance` replaces the pooled σ²_w in each class with σ²_w,c = (sse_c + β·σ²_w) / (n_c + β). β = ∞ reproduces phase 1 exactly, and it is the default until the experiment decides. The experiment refits the incumbent's 50 shards under a new config label, assembles the 25 fold models, picks β by rotation over repeat 0, confirms on repeats 1–4, and applies the spec's adoption rule.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse, RDKit, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-25-within-structure-variance-design.md` §4 (phase 1, §3, is merged).

## Global Constraints

- One branch, `within-structure-per-class`, from `main`; one PR, opened as a **draft**. Spec §4: "Adopted only if it beats pooled σ²_w out of fold on both NLL and normalised RMSE. Otherwise it is not merged."
- σ²_w,c = (within_sse_c + β·σ²_w) / (within_n_c + β), shrunk toward the model's pooled σ²_w (`SieveModel.within_variance`), per target dimension.
- `uncertainty.WITHIN_SHRINKAGE = np.inf` (β) while the experiment runs. At β = ∞ the predictive variance is **bit-identical** to phase 1. A class with `within_n_c == 0` gets the pooled σ²_w at any β. The unmatched-node fallback keeps the pooled σ²_w.
- Per-class membership is exactly `y`'s. Blind classes accrue every atom (`level.blind`). Under a stereo track, aware-only classes take the aware pass, and under the tetrahedral track also the mirror pass, just as `count`/`mean`/`msd` do.
- `FrozenLevel.within_sse` is `(nc, d)` float64 and `within_n` is `(nc,)` float64, both `None` when the batch carried no within arrays. Saved as `level_{k}_within_sse` and `level_{k}_within_n` only when present. `schema_version` and `FORMAT_VERSION` are unchanged.
- The model-level pooled `within_sse`/`within_n` of phase 1 stay as they are. The per-class sums over blind classes at any level equal them.
- `ALPHA_V = 10`, `ALPHA_T = 1` and `SELECTION_WEIGHT = 0.5` are unchanged. Only β is tuned.
- The experiment uses the train split only; the test split is never read. The refit goes under config label `element-eb-wc`, and the existing `element-eb` fits and the model cache are not touched.
- β grid: {0, 1, 3, 10, 30, 100, 300, 1000, ∞}.
- **Adoption rule** (made precise here; spec §4 states it in words). Both conditions are required:
  - (a) on repeat 0, rotating β (chosen on four folds by NLL, scored on the fifth), the out-of-fold means of NLL **and** normalised RMSE are both lower than pooled (β = ∞);
  - (b) at that β fixed, the mean paired differences against pooled over the 20 samples of repeats 1–4 are both negative. They are reported with Nadeau–Bengio 95% intervals, variance factor (1/J + 1/4).
- Gate: `.venv/bin/ruff check src tests experiments`, `.venv/bin/ruff format --check src tests experiments`, `.venv/bin/ty check src tests experiments` (two known `assert_array_equal` diagnostics), `.venv/bin/python -m pytest -q`.
- Scratchpad for experiment scripts and outputs: `$SCRATCH=/tmp/claude-1020/-data3-craabreu-github-repos-sieve/96a32280-bfb5-4416-a7ba-f4f3b00d40f9/scratchpad`. The scripts are not committed; their numbers go into the spec.

## Review Focus

- **Per-class sums that drift from `y`'s membership under a stereo track.** An aware-only class must hold exactly its own atoms' sums, including the mirror pass. Pinned in Task 1 (`test_within_sums_follow_y_under_every_track`, parametrised over no stereo, cis/trans and both tracks).
- **Merging a model that has per-class sums with one that has none** (for example the empty model, or a shard fitted from an uncollapsed set). The side without them must read as zeros, never be dropped. Pinned in Task 2 (`test_merge_with_a_level_without_sums_keeps_the_other_side`).
- **β = ∞ must reproduce phase 1 exactly, not approximately.** A `(sse + inf*s)/(n + inf)` evaluation gives NaN. Pinned in Task 3 (`test_infinite_shrinkage_is_phase_one_bit_for_bit`).
- **Truncation and `with_params` keep the per-class arrays.** `truncate_model` slices `levels`, and a sliced tuple must carry them. Pinned in Task 2 (`test_truncation_keeps_per_class_sums`).
- **The refit is the same model.** The `element-eb-wc` fits must give statistics bit-identical to the `element-eb` fits and the same predictions as Study B. Otherwise the comparison is against a different model. Pinned in Task 4 (Step 3's check).

---

### File map

| File | Change |
|---|---|
| `src/sieve/level.py` | `FrozenLevel.within_sse`, `within_n`; `_sums`; `fit_level(level, y, within=None)` |
| `src/sieve/model.py` | `fit` passes the batch's within arrays to `fit_level`; `empty`, `save`, `load` |
| `src/sieve/merge.py` | `merge_level` adds per-class sums |
| `src/sieve/uncertainty.py` | `WITHIN_SHRINKAGE`; per-class σ²_w,c |
| `tests/test_within_per_class.py` (new) | accrual, merge, persistence, truncation, variance |
| `docs/superpowers/specs/2026-09-25-within-structure-variance-design.md` | §4 results and decision |

---

### Task 1: Per-class sums accrue like `y`

**Files:**
- Modify: `src/sieve/level.py` (`FrozenLevel` fields, new `_sums`, `fit_level`)
- Modify: `src/sieve/model.py` (`fit`'s `fit_level` call; `empty`'s `FrozenLevel` needs nothing, since the defaults are `None`)
- Test: `tests/test_within_per_class.py` (create)

**Interfaces:**
- Consumes: `NodeBatch.within_sse` `(n, d)`, `NodeBatch.within_n` `(n,)` (phase 1).
- Produces: `FrozenLevel.within_sse: np.ndarray | None` `(nc, d)` and `FrozenLevel.within_n: np.ndarray | None` `(nc,)`, as the last two dataclass fields after `mirror_of`; `fit_level(level, y, within: tuple[np.ndarray, np.ndarray] | None = None) -> FrozenLevel`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_within_per_class.py`:

```python
"""Per-class within-structure variance (within-structure-variance spec, section 4).

The per-class sums must follow exactly the membership `y` follows. The
cleanest check is to feed a copy of `y` through them: with within_sse = y and
within_n = 1 on every atom, each class's within_n must equal its count and its
within_sse / within_n its mean, under every stereo track.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import sieve
from tests.helpers import chain_batch, simple_config, split_batch

CIS_TRANS = [
    "C/C=C/C",
    r"C/C=C\C",
    "C/C=C/CC",
    r"C/C=C\CC",
    "C/C(F)=C(Cl)/C",
    r"C/C(F)=C(Cl)\C",
    "CCCC",
    "C/C=C/Br",
]
CHIRAL = [
    "N[C@@H](C)C(=O)O",
    "N[C@H](C)C(=O)O",
    "C[C@H](O)[C@H](N)C",
    "C[C@H](O)[C@@H](N)C",
    "OC[C@@H](O)[C@H](O)C=O",
    "CC(C)C",
]


def _echo(batch, seed=0):
    """y positive, and the within arrays a copy of it with one member each."""
    y = np.random.default_rng(seed).uniform(0.1, 1.0, size=(batch.n_nodes, 1))
    return dataclasses.replace(
        batch, y=y, within_sse=y.copy(), within_n=np.ones(batch.n_nodes)
    )


def _stereo_batch(smiles, stereo):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes, from_rdkit

    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    codes, edges = build_codes(mols, ["element"])
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=4,
        stereo=stereo,
    )
    return _echo(from_rdkit(mols, config=cfg)), cfg


def _assert_echoes_y(model):
    for lvl in model.levels:
        assert lvl.within_sse is not None and lvl.within_n is not None
        np.testing.assert_array_equal(lvl.within_n, lvl.count.astype(np.float64))
        filled = lvl.count > 0
        np.testing.assert_allclose(
            lvl.within_sse[filled] / lvl.within_n[filled, None],
            lvl.mean[filled],
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    ("smiles", "stereo"),
    [
        (CIS_TRANS, ()),
        (CIS_TRANS, ("cis_trans",)),
        (CHIRAL, ("cis_trans", "tetrahedral")),
    ],
    ids=["blind", "cis_trans", "both_tracks"],
)
def test_within_sums_follow_y_under_every_track(smiles, stereo):
    batch, cfg = _stereo_batch(smiles, stereo)
    _assert_echoes_y(sieve.fit(batch, cfg))


def test_within_sums_follow_y_on_a_plain_chain():
    _assert_echoes_y(sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config()))


def test_a_batch_without_within_arrays_has_no_per_class_sums():
    m = sieve.fit(chain_batch(12, graphs=4), simple_config())
    assert all(lvl.within_sse is None and lvl.within_n is None for lvl in m.levels)


def test_per_class_sums_leave_the_class_statistics_unchanged():
    b = chain_batch(12, graphs=4)
    plain = sieve.fit(b, simple_config())
    rng = np.random.default_rng(3)
    with_sums = sieve.fit(
        dataclasses.replace(
            b, within_sse=rng.uniform(size=(b.n_nodes, 1)), within_n=np.ones(b.n_nodes)
        ),
        simple_config(),
    )
    for p, w in zip(plain.levels, with_sums.levels, strict=True):
        np.testing.assert_array_equal(p.count, w.count)
        np.testing.assert_array_equal(p.mean, w.mean)
        np.testing.assert_array_equal(p.msd, w.msd)


def test_blind_class_sums_add_up_to_the_pooled_sums():
    from sieve.config import KIND_BLIND
    from sieve.level import class_kinds

    batch, cfg = _stereo_batch(CIS_TRANS, ("cis_trans",))
    rng = np.random.default_rng(5)
    batch = dataclasses.replace(
        batch,
        within_sse=rng.uniform(size=(batch.n_nodes, 1)),
        within_n=rng.integers(1, 4, size=batch.n_nodes).astype(np.float64),
    )
    m = sieve.fit(batch, cfg)
    assert m.within_sse is not None
    for lvl in m.levels:
        assert lvl.within_sse is not None and lvl.within_n is not None
        blind = (class_kinds(lvl) & KIND_BLIND) > 0
        np.testing.assert_allclose(lvl.within_sse[blind].sum(axis=0), m.within_sse)
        assert lvl.within_n[blind].sum() == pytest.approx(m.within_n)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py -q`
Expected: FAIL with `AttributeError: 'FrozenLevel' object has no attribute 'within_sse'`. The exception is `test_per_class_sums_leave_the_class_statistics_unchanged`, which passes already.

- [ ] **Step 3: Implement**

In `src/sieve/level.py`, after `mirror_of` in `FrozenLevel`:

```python
    # Within-structure sums per class (within-structure-variance spec 4): the
    # summed squared deviations of the class's training atoms from their
    # structures' orbit means, and the number of atoms summed over. Accrued
    # through exactly the membership `y` accrues through. None when the fit
    # carried no within-structure statistics.
    within_sse: np.ndarray | None = None  # (nc, d) float64
    within_n: np.ndarray | None = None  # (nc,) float64
```

Beside `_reduce`:

```python
def _sums(labels: np.ndarray, values: np.ndarray, nc: int) -> np.ndarray:
    """Per-class sums of ``values`` rows ((n,) or (n, d)), by the same sparse
    membership operator ``_reduce`` builds."""
    n = labels.shape[0]
    P = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(nc, n))
    return np.asarray(P @ values)
```

Replace `fit_level` with the same body plus the within accrual. Every membership line is `y`'s, applied to `(sse, cnt)`:

```python
def fit_level(
    level: LevelLabels,
    y: np.ndarray,
    within: tuple[np.ndarray, np.ndarray] | None = None,
) -> FrozenLevel:
    """(docstring unchanged, plus:)

    ``within``, when given, is the batch's ``(within_sse, within_n)``; they
    are summed per class through exactly the memberships above
    (within-structure-variance spec 4).
    """
    nc = level.n_classes
    count, mean, msd = _reduce(level.blind, y, nc)
    w_sse = w_n = None
    if within is not None:
        sse, cnt = within
        w_sse, w_n = _sums(level.blind, sse, nc), _sums(level.blind, cnt, nc)
    if level.kind is not None:
        differs = level.labels != level.blind
        extra_labels, extra_y = [level.labels[differs]], [y[differs]]
        extra_rows = [np.flatnonzero(differs)]
        if level.mirror_labels is not None:
            moved = level.mirror_labels != level.labels
            extra_labels.append(level.mirror_labels[moved])
            extra_y.append(y[moved])
            extra_rows.append(np.flatnonzero(moved))
        lab = np.concatenate(extra_labels)
        if lab.size:
            yy = np.concatenate(extra_y)
            c2, m2, s2 = _reduce(lab, yy, nc)
            only = level.kind == KIND_AWARE
            count[only], mean[only], msd[only] = c2[only], m2[only], s2[only]
            if within is not None and w_sse is not None and w_n is not None:
                rows = np.concatenate(extra_rows)
                w_sse[only] = _sums(lab, sse[rows], nc)[only]
                w_n[only] = _sums(lab, cnt[rows], nc)[only]
    return FrozenLevel(
        level.signatures,
        count,
        mean,
        msd,
        level.parent,
        level.kind,
        level.blind_of,
        level.mirror_of,
        w_sse,
        w_n,
    )
```

In `src/sieve/model.py` `fit`, replace `levels = tuple(fit_level(lv, batch.y) for lv in levels_lbl)` with:

```python
    within = None
    if batch.within_sse is not None and batch.within_n is not None:
        within = (batch.within_sse, batch.within_n)
    levels = tuple(fit_level(lv, batch.y, within) for lv in levels_lbl)
```

(the pooled sums further down stay as they are).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py tests/test_level.py tests/test_fit.py tests/test_stereo_refines_blind.py tests/test_tetrahedral.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sieve/level.py src/sieve/model.py tests/test_within_per_class.py
git commit -m "feat(level): per-class within-structure sums, accrued like y"
```

---

### Task 2: Merge, persistence and truncation carry the per-class sums

**Files:**
- Modify: `src/sieve/merge.py` (`merge_level`)
- Modify: `src/sieve/model.py` (`save`, `load`)
- Test: `tests/test_within_per_class.py` (append)

**Interfaces:**
- Consumes: `FrozenLevel.within_sse`/`within_n` (Task 1).
- Produces: `merge_level` returns levels whose sums are `a`'s plus `b`'s, remapped (a side with `None` reads as zeros; both `None` stays `None`); saved keys `level_{k}_within_sse`, `level_{k}_within_n`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_within_per_class.py`:

```python
# --------------------------------------------------------- merge and I/O --


def test_merge_of_two_fits_echoes_y_like_the_fit_of_their_union():
    b = _echo(chain_batch(12, graphs=6))
    cfg = simple_config()
    mask = b.graph_id < 3
    merged = sieve.fit(split_batch(b, mask), cfg).merge(
        sieve.fit(split_batch(b, ~mask), cfg)
    )
    _assert_echoes_y(merged)


def test_stereo_merge_echoes_y():
    batch, cfg = _stereo_batch(CHIRAL, ("cis_trans", "tetrahedral"))
    mask = batch.graph_id < 3
    merged = sieve.fit(split_batch(batch, mask), cfg).merge(
        sieve.fit(split_batch(batch, ~mask), cfg)
    )
    _assert_echoes_y(merged)


def test_merge_with_a_level_without_sums_keeps_the_other_side():
    b = chain_batch(12, graphs=6)
    cfg = simple_config()
    mask = b.graph_id < 3
    with_sums = sieve.fit(_echo(split_batch(b, mask)), cfg)
    without = sieve.fit(split_batch(b, ~mask), cfg)
    for merged in (with_sums.merge(without), without.merge(with_sums)):
        for lvl in merged.levels:
            assert lvl.within_n is not None and lvl.within_sse is not None
        total = sum(float(lvl.within_n.sum()) for lvl in merged.levels[:1])
        assert total == pytest.approx(float(mask.sum()))


def test_the_empty_model_keeps_per_class_sums_under_merge():
    from sieve.model import SieveModel

    m = sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config())
    e = SieveModel.empty(m.config)
    _assert_echoes_y(m.merge(e))
    _assert_echoes_y(e.merge(m))


def test_save_load_round_trips_per_class_sums(tmp_path):
    from sieve.model import SieveModel

    m = sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config())
    path = tmp_path / "m.npz"
    m.save(path)
    back = SieveModel.load(path)
    for a, b in zip(m.levels, back.levels, strict=True):
        assert a.within_sse is not None and b.within_sse is not None
        assert a.within_n is not None and b.within_n is not None
        np.testing.assert_array_equal(a.within_sse, b.within_sse)
        np.testing.assert_array_equal(a.within_n, b.within_n)


def test_a_file_without_per_class_sums_keeps_its_keys(tmp_path):
    from sieve.model import SieveModel

    m = sieve.fit(chain_batch(12, graphs=4), simple_config())
    path = tmp_path / "m.npz"
    m.save(path)
    assert not any("within" in f for f in np.load(path).files)
    assert all(lvl.within_sse is None for lvl in SieveModel.load(path).levels)


def test_truncation_keeps_per_class_sums():
    pytest.importorskip("experiments")
    from experiments.cv import truncate_model

    m = sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config(max_wl_depth=3))
    _assert_echoes_y(truncate_model(m, 1))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py -q -k "merge or save or file_without or truncation or empty"`
Expected:
- The merge tests FAIL with `AssertionError` (`within_sse is None` after the merge).
- The save/load round trip FAILS the same way.
- `test_a_file_without_per_class_sums_keeps_its_keys` and `test_truncation_keeps_per_class_sums` PASS already. Truncation slices the level tuple, so it keeps the arrays by construction, and the test stays as a guard.

- [ ] **Step 3: Implement the merge**

In `merge_level`, after the `mirror_of` block and before `return`:

```python
    within_sse = within_n = None
    if a.within_sse is not None or b.within_sse is not None:
        # Sums, so they add; a side without them (the empty model, or a fit
        # of an uncollapsed set) contributes zero (within-structure-variance
        # spec 4). `i` is a bijection, so the scatter-add has no collisions.
        within_sse = np.zeros((n_new, d), np.float64)
        within_n = np.zeros(n_new, np.float64)
        if a.within_sse is not None and a.within_n is not None:
            within_sse[:m] = a.within_sse
            within_n[:m] = a.within_n
        if b.within_sse is not None and b.within_n is not None:
            within_sse[i] += b.within_sse
            within_n[i] += b.within_n
```

and change the return to:

```python
    return (
        FrozenLevel(
            uniq,
            count,
            mean,
            msd,
            parent,
            class_kind,
            blind_of,
            mirror_of,
            within_sse,
            within_n,
        ),
        remap,
    )
```

- [ ] **Step 4: Implement persistence**

In `SieveModel.save`, inside the level loop after the `mirror_of` block:

```python
            if lvl.within_sse is not None and lvl.within_n is not None:
                # Only when present, so a file without per-class sums keeps
                # exactly the keys it always had.
                arrays[f"level_{k}_within_sse"] = lvl.within_sse
                arrays[f"level_{k}_within_n"] = lvl.within_n
```

In `SieveModel.load`, add two arguments to the `FrozenLevel(...)` construction after the `mirror_of` one:

```python
                data[f"level_{k}_within_sse"]
                if f"level_{k}_within_sse" in data.files
                else None,
                data[f"level_{k}_within_n"]
                if f"level_{k}_within_n" in data.files
                else None,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py tests/test_merge.py tests/test_io.py tests/test_within_structure.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sieve/merge.py src/sieve/model.py tests/test_within_per_class.py
git commit -m "feat(merge): per-class within-structure sums merge and persist"
```

---

### Task 3: The predictive variance reads σ²_w,c

**Files:**
- Modify: `src/sieve/uncertainty.py` (constant, signature, the σ²_w term, docstring)
- Test: `tests/test_within_per_class.py` (append)

**Interfaces:**
- Consumes: per-class sums (Tasks 1–2); `SieveModel.within_variance` (phase 1).
- Produces: `uncertainty.WITHIN_SHRINKAGE: float = np.inf`; `predictive_variance(model, *, alpha_v=ALPHA_V, alpha_t=ALPHA_T, selection_weight=SELECTION_WEIGHT, within_shrinkage=WITHIN_SHRINKAGE)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_within_per_class.py`:

```python
# ----------------------------------------------------- predictive variance --


def _model_with_class_sums(seed=0):
    b = chain_batch(12, graphs=4, seed=seed)
    rng = np.random.default_rng(seed + 10)
    n = rng.integers(1, 5, size=b.n_nodes).astype(np.float64)
    sse = rng.uniform(0.0, 2e-4, size=(b.n_nodes, 1)) * n[:, None]
    return sieve.fit(
        dataclasses.replace(b, within_sse=sse, within_n=n),
        simple_config(max_wl_depth=2),
    )


def test_infinite_shrinkage_is_phase_one_bit_for_bit():
    from sieve.uncertainty import WITHIN_SHRINKAGE, predictive_variance

    assert WITHIN_SHRINKAGE == np.inf
    m = _model_with_class_sums()
    # phase 1: the same model with its per-class sums stripped
    pooled_only = dataclasses.replace(
        m,
        levels=tuple(
            dataclasses.replace(lvl, within_sse=None, within_n=None) for lvl in m.levels
        ),
    )
    for a, b in zip(predictive_variance(m), predictive_variance(pooled_only), strict=True):
        np.testing.assert_array_equal(a, b)


def test_finite_shrinkage_is_the_hand_computed_blend():
    from sieve.uncertainty import predictive_variance

    m = _model_with_class_sums()
    beta = 3.0
    pooled = m.within_variance
    base = predictive_variance(m)  # pooled sigma2_w in every class
    blended = predictive_variance(m, within_shrinkage=beta)
    for lvl, b0, b1 in zip(m.levels, base, blended, strict=True):
        assert lvl.within_sse is not None and lvl.within_n is not None
        s2c = (lvl.within_sse + beta * pooled) / (lvl.within_n[:, None] + beta)
        np.testing.assert_allclose(b1, b0 - pooled + s2c, rtol=1e-12)


def test_zero_shrinkage_uses_each_class_s_own_ratio_and_pooled_where_empty():
    from sieve.uncertainty import predictive_variance

    m = _model_with_class_sums()
    base = predictive_variance(m)
    own = predictive_variance(m, within_shrinkage=0.0)
    pooled = m.within_variance
    for lvl, b0, b1 in zip(m.levels, base, own, strict=True):
        assert lvl.within_sse is not None and lvl.within_n is not None
        n = lvl.within_n[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(n > 0, lvl.within_sse / n, pooled)
        np.testing.assert_allclose(b1, b0 - pooled + ratio, rtol=1e-12)
        assert np.isfinite(b1).all()


def test_a_model_without_per_class_sums_ignores_the_shrinkage():
    from sieve.uncertainty import predictive_variance

    m = sieve.fit(chain_batch(12, graphs=4), simple_config()).with_within_structure(
        1e-3, 10
    )
    for a, b in zip(
        predictive_variance(m), predictive_variance(m, within_shrinkage=1.0), strict=True
    ):
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py -q -k "shrinkage or blend"`
Expected: FAIL with `ImportError: cannot import name 'WITHIN_SHRINKAGE'`.

- [ ] **Step 3: Implement**

In `src/sieve/uncertainty.py`, after `SELECTION_WEIGHT`:

```python
# beta: per-class within-structure variance shrinkage toward the pooled one
# (within-structure-variance spec 4). Infinite -- the pooled sigma2_w in every
# class, phase 1 exactly -- until the out-of-fold experiment decides.
WITHIN_SHRINKAGE = np.inf
```

Add `within_shrinkage: float = WITHIN_SHRINKAGE,` to `predictive_variance`'s keyword arguments, and replace the last line of the level loop with:

```python
        out.append(
            within
            + selection_weight * selection
            + estimation
            + _within_structure(lvl, sigma2_w, within_shrinkage)
        )
```

with, at module level:

```python
def _within_structure(lvl, pooled: np.ndarray, beta: float) -> np.ndarray:
    """sigma2_w per class: (sse_c + beta*pooled) / (n_c + beta), and the pooled
    value itself where the class carries no sums or beta is infinite -- the
    limit, taken explicitly because inf/inf is nan."""
    if lvl.within_sse is None or lvl.within_n is None or np.isinf(beta):
        return pooled
    n = lvl.within_n[:, None]
    den = n + beta
    with np.errstate(invalid="ignore", divide="ignore"):
        blend = (lvl.within_sse + beta * pooled) / den
    return np.where(den > 0, blend, pooled)
```

Add a sentence to the module docstring's **within structure** paragraph: "Per class, when the fit carried per-class sums, σ²_w,c = (sse_c + β σ²_w)/(n_c + β), shrunk toward the pooled value (``WITHIN_SHRINKAGE``); β = ∞ is the pooled value in every class."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_within_per_class.py tests/test_within_structure.py tests/test_uncertainty.py tests/test_predict.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full gate, commit, open the draft PR**

Run the four gate commands from Global Constraints. Expected: all clean, ty reports only the 2 known diagnostics, and pytest passes (about 12 minutes).

```bash
git add src/sieve/uncertainty.py tests/test_within_per_class.py
git commit -m "feat(uncertainty): per-class within-structure variance, shrunk by beta"
git push -u origin within-structure-per-class
gh pr create --draft --title "Per-class within-structure variance (phase 2, experiment)" --body "<what, the adoption rule verbatim from the plan, 'draft until the experiment decides'; end with the attribution line>"
```

---

### Task 4: Refit the incumbent's shards under `element-eb-wc` and verify the refit

**Files:**
- Create (scratchpad): `$SCRATCH/phase2_refit_check.py`
- No repository changes.

**Interfaces:**
- Consumes: Tasks 1–3 on the branch; the CLI `python -m experiments cv-fit-sieve-shards`.
- Produces: 50 shard fits `experiments/runs/cv-shard-fits/fit-sieve-element-eb-wc-w10-s{00..49}__*/tree_stats.npz` carrying per-class sums.

- [ ] **Step 1: Refit**

The incumbent's parameters (`experiments/workflows/cv_charges.sh:355-357`), with collapse:

```bash
PARAMS='{"attributes": ["element"], "edge_attributes": [], "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}'
seq -f "s%02g" 0 49 | xargs -P 8 -n 1 -I{} .venv/bin/python -m experiments cv-fit-sieve-shards dash-molecules \
  --n-shards 50 --max-depth 10 --shard {} \
  --codes-path experiments/stores/dash-molecules/sieve-codes.json \
  --config-label element-eb-wc --predictor-params "$PARAMS" --collapse \
  > $SCRATCH/phase2_refit.log 2>&1
ls -d experiments/runs/cv-shard-fits/fit-sieve-element-eb-wc-w10-s* | wc -l
```

Expected: 50. Each fit takes a few seconds, since the `element-eb` manifests record about 3 s of featurisation plus fit per shard. If the CLI refuses a dirty tree, commit first; the branch is clean after Task 3.

- [ ] **Step 2: Write the refit check**

`$SCRATCH/phase2_refit_check.py`:

```python
"""The element-eb-wc fits are the element-eb fits plus per-class sums."""

import glob
import json

import numpy as np

import sieve

FLOORS = json.load(open("experiments/stores/dash-molecules/floor-components.json"))


def one(label, shard):
    (p,) = sorted(
        glob.glob(f"experiments/runs/cv-shard-fits/fit-sieve-{label}-w10-{shard}__*/tree_stats.npz")
    )[-1:]
    return sieve.SieveModel.load(p)


for s in [f"s{i:02d}" for i in range(50)]:
    old, new = one("element-eb", s), one("element-eb-wc", s)
    assert old.config.schema_version == new.config.schema_version
    for a, b in zip(old.levels, new.levels, strict=True):
        np.testing.assert_array_equal(a.signatures, b.signatures)
        np.testing.assert_array_equal(a.count, b.count)
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.msd, b.msd)
        assert b.within_sse is not None and b.within_n is not None
        # no stereo track: every class is blind, so each level sums to the pool
        assert np.isclose(b.within_n.sum(), new.within_n, rtol=1e-12)
    assert np.isclose(new.within_sse[0], FLOORS[s]["sse"], rtol=1e-10), s
    assert np.isclose(new.within_n, FLOORS[s]["n_atoms"], rtol=1e-12), s
print("50 shards: element-eb-wc == element-eb plus per-class sums; sums == floor cache")
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python $SCRATCH/phase2_refit_check.py`
Expected: the final line prints. If a statistic differs, stop and report: the comparison would be against a different model.

---

### Task 5: The experiment: rotate β on repeat 0, confirm on repeats 1–4

**Files:**
- Create (scratchpad): `$SCRATCH/phase2_eval.py`, `$SCRATCH/phase2_decide.py`; outputs in `$SCRATCH/phase2_eval/`.

**Interfaces:**
- Consumes: the refit (Task 4); `predictive_variance(..., within_shrinkage=β)` (Task 3); `uvar_lib.score(df, sig2, label)` and `uvar_lib.FLOORS` from `$SCRATCH/uvar_lib.py`; `experiments.cv.build_cv_plan`, `leave_one_group_out`, `truncate_model`; `sieve.merge.fold`, `merge_models`.
- Produces: `$SCRATCH/phase2_eval/r{r}-f{f}.json`, each holding `{beta_label: score_dict}` for every β in the grid.

- [ ] **Step 1: Write the evaluation script**

`$SCRATCH/phase2_eval.py`:

```python
"""Phase 2: score every beta on all 25 samples, one featurisation each.

Train split only. Fold models are assembled from the element-eb-wc shard
fits exactly as run_sieve_cv does: sieve_fold per group, leave_one_group_out,
truncate to depth 5, continuation + EB, predictive_variance on.
"""

import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import uvar_lib as U
from experiments.cv import build_cv_plan, leave_one_group_out, truncate_model
from experiments.data import blob_to_mol

import sieve
from sieve.io.rdkit_adapter import from_rdkit
from sieve.merge import fold as sieve_fold
from sieve.merge import merge_models
from sieve.uncertainty import predictive_variance

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
BETAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, np.inf]
IDS = [f"s{i:02d}" for i in range(50)]


def shard(s):
    (p,) = sorted(
        glob.glob(f"experiments/runs/cv-shard-fits/fit-sieve-element-eb-wc-w10-{s}__*/tree_stats.npz")
    )[-1:]
    return sieve.SieveModel.load(p)


shards = {s: shard(s) for s in IDS}
for r in range(5):
    plan = build_cv_plan(IDS, k=5, repeat=r)
    groups = [sieve_fold([shards[s] for s in g], shards[g[0]].config) for g in plan.groups]
    fold_models = leave_one_group_out(groups, merge=merge_models)
    for f, group in enumerate(plan.groups):
        out = OUT / f"r{r}-f{f}.json"
        if out.exists():
            continue
        (path,) = glob.glob(
            f"experiments/runs/sieve-cv-study-b/r{r}-f{f}-sieve-element-continuation-eb-w5__*/predictions.npz"
        )
        z = np.load(path, allow_pickle=True)
        manifest = json.load(open(path.replace("predictions.npz", "manifest.json")))
        assert set(manifest["config"]["cv"]["held_out_shards"].split(",")) == set(group)

        m = truncate_model(fold_models[f], 5)
        m = m.with_params(class_estimator="continuation", shrinkage_weight="empirical_bayes")
        m = replace(m, config=replace(m.config, predictive_variance=True))
        assert m.within_n > 0

        blob = U._blobs()
        mols = [blob_to_mol(blob[(str(a), str(b))]) for a, b in zip(z["dash_id"], z["conf_id"], strict=True)]
        p = sieve.predict_detailed(m, from_rdkit(mols, config=m.config, n_jobs=16))
        # the refit is the Study B model: same predictions
        assert np.abs(p.value[:, 0] - z["atom_target_pred"].ravel()).max() < 1e-12
        k, cid = p.matched_level, p.class_id
        df = pd.DataFrame(
            {
                "mu": p.value[:, 0],
                "y": z["atom_target_true"].ravel(),
                "k": k,
                "conf": np.repeat(np.arange(len(mols)), z["num_atoms"]),
            }
        )
        df["e"] = df.mu - df.y
        fallback = float(m.global_msd[0] + m.within_variance[0])
        res = {"r": r, "f": f, "sigma2_w": float(m.within_variance[0])}
        for beta in BETAS:
            table = predictive_variance(m, within_shrinkage=beta)
            s2 = np.full(len(k), fallback)
            for lv in np.unique(k[k >= 0]):
                sel = k == lv
                s2[sel] = table[lv][cid[sel], 0]
            if np.isinf(beta):
                np.testing.assert_array_equal(s2, p.predictive_variance[:, 0])
            score = U.score(df, s2, str(beta))
            res[str(beta)] = {kk: float(v) for kk, v in score.items() if kk != "label"}
        out.write_text(json.dumps(res, indent=1))
        print(f"r{r}-f{f} done", flush=True)
```

- [ ] **Step 2: Run it**

Run (from the repository root, in the background, about 25 × 50 s): `PYTHONPATH=$SCRATCH .venv/bin/python $SCRATCH/phase2_eval.py $SCRATCH/phase2_eval > $SCRATCH/phase2_eval.log 2>&1`
Expected: 25 `done` lines, and none of the three assertions fires. The `β = inf` column must equal the shipped phase-1 numbers in `$SCRATCH/phase1_eval/` to the last digit, which is a free cross-check.

- [ ] **Step 3: Write the decision script**

`$SCRATCH/phase2_decide.py`:

```python
"""Apply the adoption rule (plan, Global Constraints)."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

rows = {(r["r"], r["f"]): r for r in (json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("r*-f*.json"))}
assert len(rows) == 25, len(rows)
BETAS = [k for k in rows[(0, 0)] if k not in ("r", "f", "sigma2_w")]
POOLED = "inf"

# (a) rotation on repeat 0
oof = {"NLL": [], "norm_rmse": []}
pooled = {"NLL": [], "norm_rmse": []}
picks = []
for f in range(5):
    others = [rows[(0, g)] for g in range(5) if g != f]
    best = min(BETAS, key=lambda b: np.mean([o[b]["NLL"] for o in others]))
    picks.append(best)
    for c in oof:
        oof[c].append(rows[(0, f)][best][c])
        pooled[c].append(rows[(0, f)][POOLED][c])
print("rotation picks:", picks)
a_ok = all(np.mean(oof[c]) < np.mean(pooled[c]) for c in oof)
for c in oof:
    print(f"(a) {c}: rotated {np.mean(oof[c]):.6f} vs pooled {np.mean(pooled[c]):.6f}")

# (b) the modal pick fixed, repeats 1-4
beta = max(set(picks), key=picks.count)
rest = [rows[(r, f)] for r in range(1, 5) for f in range(5)]
b_ok = True
for c in ("NLL", "norm_rmse"):
    d = np.array([x[beta][c] - x[POOLED][c] for x in rest])
    se = np.sqrt((1 / len(d) + 1 / 4) * d.var(ddof=1))
    h = student_t.ppf(0.975, len(d) - 1) * se
    print(f"(b) beta={beta} {c}: {d.mean():+.3e} [{d.mean() - h:+.3e}, {d.mean() + h:+.3e}], lower in {(d < 0).sum()}/{len(d)}")
    b_ok &= d.mean() < 0

# calibration at the chosen beta against pooled, all 25
for b in (beta, POOLED):
    print(b, {c: round(np.mean([x[b][c] for x in rows.values()]), 4) for c in ("E[z2]", "cov95", "k1", "k3", "k5")})
print("ADOPT" if (a_ok and b_ok and beta != POOLED) else "REJECT", "beta =", beta)
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python $SCRATCH/phase2_decide.py $SCRATCH/phase2_eval`
Expected: a final `ADOPT beta = …` or `REJECT beta = …`. If the rotation itself picks `inf`, the verdict is REJECT: pooled wins.

---

### Task 6: Record the result and act on the verdict

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-within-structure-variance-design.md` (§4 results, status line)
- If ADOPT: modify `src/sieve/uncertainty.py` (`WITHIN_SHRINKAGE` and its comment), `tests/test_within_per_class.py` (the constant's assertion)

**Interfaces:**
- Consumes: Task 5's printed output.

- [ ] **Step 1: Write the results into the spec**

Append to §4 a "Result (2026-09-2x)" paragraph, written in the spec's register. It gives:
- the rotation picks;
- (a)'s two out-of-fold comparisons;
- (b)'s two mean differences with their Nadeau–Bengio intervals and win counts;
- the calibration line at the chosen β against pooled;
- the verdict.

Update the status line to say phase 2 was run and whether it was adopted.

- [ ] **Step 2a (ADOPT): Set β and pin it**

In `uncertainty.py`, set `WITHIN_SHRINKAGE` to the chosen value and replace its comment with one sentence citing the rotation and the repeats 1–4 confirmation. In `test_infinite_shrinkage_is_phase_one_bit_for_bit`, replace `assert WITHIN_SHRINKAGE == np.inf` with a call at `within_shrinkage=np.inf`, so the test still pins the limit, and add `assert WITHIN_SHRINKAGE == <chosen>`. Run the full gate. Commit, push, and mark the PR ready (`gh pr ready`). Merging needs the user's go-ahead.

- [ ] **Step 2b (REJECT): Keep main unchanged**

Commit the spec update on the branch and push. Leave the PR in draft with a closing comment that summarises the result, and ask the user whether to close it. Then open a docs-only PR that carries just the spec's §4 result to `main` (cherry-pick that commit onto a branch `within-structure-phase2-result`), so the negative result is recorded where the spec lives.

- [ ] **Step 3: Remember the result**

Write a memory file `within-structure-phase2.md` (type `project`) recording the verdict, the chosen β or "pooled wins", and the headline numbers, with a pointer line in `MEMORY.md`.
