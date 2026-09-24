# Stereo Refines the Stereo-Blind Class: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fused cis/trans construction with design D: one vocabulary per level
holding stereo-blind and stereo-aware classes, the blind part identical to a stereo-blind
fit, and prediction answering from the aware class only when it exists at the incumbent's
matched radius.

**Architecture:** `refine` builds two label arrays per WL level (aware, and recursively
blind), deduplicated into one vocabulary with a per-class `kind` bitmask and a `blind_of`
map. Statistics, continuation and τ² are computed so that blind classes reproduce the
stereo-blind fit; aware-only classes are pooled means shrunk toward their blind
counterpart. `predict` walks the blind labels to *k*\*, then looks up the aware class at
*k*\*.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse, RDKit (tests), pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-23-stereo-refines-the-blind-class-design.md`

## Global Constraints

- Stereo-blind models (`stereo=()`) are unchanged: same classes, same statistics, same
  `schema_version` digest, same saved-file keys. `kind`/`blind_of` are `None` on them.
- `None` for `kind` means "every class is both blind and aware"; `None` for `blind_of`
  means the identity. Every reader goes through `level.class_kinds` / `level.blind_targets`.
- Kind bits: `KIND_BLIND = 1`, `KIND_AWARE = 2`, `KIND_BOTH = 3` (`sieve.config`).
- "Identical to the stereo-blind fit" means: class partition and `count`/`mean`/`msd`
  bit-identical; derived estimates (continuation, τ², shrunk means, predictions) equal to
  `rtol=1e-12`, since `bincount` sums follow class numbering, which `dense_rows` does not fix.
- `schema_version` gains `"stereo_construction": "refines_blind"` only when `stereo` is
  non-empty. No `FORMAT_VERSION` bump: the new arrays are optional keys.
- `predict_loo` and `experiments.analytic.sieve_train_stats` refuse stereo models
  (`NotImplementedError`): an atom now contributes to two classes at a level.
- Aware lookups respect `minimum_support` like any other class.
- CI gate: `ruff check`, `ruff format --check`, `ty check` over `src tests experiments`
  (two known `assert_array_equal` diagnostics), then `pytest`. Use `.venv/bin/python`.
- Everything below lands as **one PR** on branch `stereo-refines-blind`.

---

### File map

| File | Change |
|---|---|
| `src/sieve/config.py` | kind constants; schema marker |
| `src/sieve/refine.py` | `LevelLabels` gains `blind_labels`, `kind`, `blind_of`, `blind`; dual WL rounds |
| `src/sieve/level.py` | `FrozenLevel` gains `kind`, `blind_of`; `class_kinds`, `blind_targets`; split reduction |
| `src/sieve/continuation.py` | blind-children filter; `atom_variance` over blind classes; `aware_variance` |
| `src/sieve/shrinkage.py` | aware-only classes shrunk toward `blind_of`; EB weight for them |
| `src/sieve/uncertainty.py` | estimation term for aware-only classes |
| `src/sieve/predict.py` | blind walk, aware lookup at *k*\*, `stereo_refined`; LOO refusal |
| `src/sieve/merge.py` | `kind` by OR, `blind_of` remapped and checked |
| `src/sieve/model.py` | save/load the two arrays |
| `experiments/experiments/analytic.py` | refuse stereo models |
| `experiments/experiments/cv.py` | skip the analytic training statistics where `supports_train_stats` is false (found during execution: the CV path called them unconditionally) |
| `tests/test_stereo_refines_blind.py` | new: §8 tests |

---

### Task 1: Level data and the dual refinement

**Files:** Modify `src/sieve/config.py`, `src/sieve/refine.py`, `src/sieve/level.py` (dataclass
fields and helpers only). Test: `tests/test_stereo_refines_blind.py`.

**Interfaces produced:**
- `sieve.config.KIND_BLIND, KIND_AWARE, KIND_BOTH: int`
- `LevelLabels(labels, signatures, parent, blind_labels=None, kind=None, blind_of=None)`;
  property `blind -> np.ndarray` (`blind_labels` or `labels`).
- `FrozenLevel(signatures, count, mean, msd, parent, kind=None, blind_of=None)`.
- `sieve.level.class_kinds(level) -> np.ndarray[uint8]`, `sieve.level.blind_targets(level) -> np.ndarray[int64]`.

- [ ] **Step 1: failing tests** (new file, module header `pytest.importorskip("rdkit")`):

```python
import dataclasses

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import sieve
from sieve.config import KIND_AWARE, KIND_BLIND, KIND_BOTH, SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit
from sieve.level import blind_targets, class_kinds
from sieve.refine import refine

CORPUS = [
    "C/C=C/C", r"C/C=C\C", "C/C=C/CC", r"C/C=C\CC", "C/C(F)=C(Cl)/C",
    r"C/C(F)=C(Cl)\C", "CCCC", "CC(C)C", "C/C=C/Br", r"OC/C=C\CN",
    "CCN1/C(=C2/OC(=S)N(C)C2=O)Sc2ccccc21",
]
PHOSPHORUS = "COP12(OC)NC(=O)O[C@]1(C(F)(F)F)c1ccccc1O2"


def _config(smiles, *, stereo, depth=4, **kw):
    codes, edges = build_codes([Chem.MolFromSmiles(s) for s in smiles], ["element"])
    return SieveConfig(
        target_dim=1, attribute_levels=(("element",),), attribute_codes=codes,
        edge_codes=edges, max_wl_depth=depth,
        stereo=("cis_trans",) if stereo else (), **kw,
    )


def _batch(smiles, cfg, seed=0):
    b = from_rdkit([Chem.MolFromSmiles(s) for s in smiles], y=None, config=cfg)
    return dataclasses.replace(b, y=np.random.default_rng(seed).normal(size=(b.n_nodes, 1)))


def _class_map(src, dst):
    """src id -> dst id for two labelings of the same atoms; asserts a bijection."""
    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    assert len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1]))
    out = np.full(int(src.max()) + 1, -1, np.int64)
    out[pairs[:, 0]] = pairs[:, 1]
    return out


def test_blind_labels_partition_atoms_as_a_stereo_blind_refine_does():
    """Recursive blinding (spec §8.4): the blind label is the incumbent's class."""
    s_cfg, b_cfg = _config(CORPUS, stereo=True), _config(CORPUS, stereo=False)
    s = refine(_batch(CORPUS, s_cfg), s_cfg)
    b = refine(_batch(CORPUS, b_cfg), b_cfg)
    for ls, lb in zip(s, b, strict=True):
        _class_map(ls.blind, lb.labels)


def test_methyls_of_the_two_butenes_share_a_blind_class_at_radius_three():
    """Fails if only the current round's trit is dropped: the methyl's neighbour
    already differs by geometry at radius 2."""
    cfg = _config(CORPUS, stereo=True, depth=3)
    lv = refine(_batch(["C/C=C/C", r"C/C=C\C"], cfg), cfg)[-1]
    assert lv.blind[0] == lv.blind[4]
    assert lv.labels[0] != lv.labels[4]


def test_kinds_and_blind_counterparts_are_consistent():
    cfg = _config(CORPUS, stereo=True)
    for lv in refine(_batch(CORPUS, cfg), cfg):
        kind, target = class_kinds(lv), blind_targets(lv)
        assert np.all(kind[lv.blind] & KIND_BLIND)
        assert np.all(kind[lv.labels] & KIND_AWARE)
        np.testing.assert_array_equal(target[lv.labels], lv.blind)
        np.testing.assert_array_equal(target[lv.blind], lv.blind)
        differs = lv.labels != lv.blind
        assert np.all(kind[lv.labels[differs]] == KIND_AWARE)


def test_no_stereo_bond_means_no_aware_only_class():
    cfg = _config(CORPUS, stereo=True)
    for lv in refine(_batch(["CCCC", "CC(C)C"], cfg), cfg):
        assert np.all(class_kinds(lv) == KIND_BOTH)
        np.testing.assert_array_equal(lv.labels, lv.blind)
```

- [ ] **Step 2:** `.venv/bin/python -m pytest -q tests/test_stereo_refines_blind.py` → ImportError (`KIND_AWARE`).

- [ ] **Step 3: implement.**

`config.py`, next to `STEREO_TRACKS`:

```python
# Class kinds under a stereo track (stereo-refines-blind spec §3): a bitmask,
# so a class that is both the blind and the aware class of its atoms is 3.
KIND_BLIND = 1
KIND_AWARE = 2
KIND_BOTH = KIND_BLIND | KIND_AWARE
```

`refine.py`: `LevelLabels` gains three trailing fields defaulting to `None` and

```python
    @property
    def blind(self) -> np.ndarray:
        """Each atom's stereo-blind class: its own class when no track is on."""
        return self.labels if self.blind_labels is None else self.blind_labels
```

Factor the WL row builder out of the loop and add the union step:

```python
def _wl_rows(base, csr, full, n, n_edge_types):
    pair = base[csr.dst] * n_edge_types + full
    pad = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
    pad[csr.src, csr.slot] = pair
    pad.sort(axis=1)
    return np.concatenate([base[:, None], pad], axis=1)


def _union_level(sig_aware, sig_blind):
    """One vocabulary for both rows of every atom (spec §3)."""
    n = sig_blind.shape[0]
    differs = np.flatnonzero((sig_aware != sig_blind).any(axis=1))
    labels, uniq = dense_rows(np.concatenate([sig_blind, sig_aware[differs]]))
    blind = labels[:n]
    aware = blind.copy()
    aware[differs] = labels[n:]
    kind = np.zeros(uniq.shape[0], np.uint8)
    kind[blind] |= KIND_BLIND
    kind[aware] |= KIND_AWARE
    blind_of = np.arange(uniq.shape[0], dtype=np.int64)
    blind_of[aware] = blind
    return LevelLabels(aware, uniq, uniq[:, 0].astype(np.int32), blind, kind, blind_of)
```

In the WL branch, when `stereo_bonds is not None`: build `sig_aware = _wl_rows(base, csr,
edge_code * stereo_radix + stereo_code, ...)` and `sig_blind = _wl_rows(levels[parents[offset]].blind,
csr, edge_code * stereo_radix, ...)`, append `_union_level(sig_aware, sig_blind)` and
`continue`; otherwise the unchanged path via `_wl_rows(base, csr, edge_code, ...)`.
Update the stereo comments to say the trit refines the blind class rather than replacing it.

`level.py`: `FrozenLevel` gains `kind: np.ndarray | None = None` and
`blind_of: np.ndarray | None = None`, and

```python
def class_kinds(level) -> np.ndarray:
    """Per-class kind bits; ``None`` stored means every class is both."""
    if level.kind is None:
        return np.full(level.n_classes, KIND_BOTH, np.uint8)
    return level.kind


def blind_targets(level) -> np.ndarray:
    """Per-class blind counterpart; ``None`` stored means the identity."""
    if level.blind_of is None:
        return np.arange(level.n_classes, dtype=np.int64)
    return level.blind_of
```

- [ ] **Step 4:** the new tests pass; `tests/test_refine.py tests/test_merge.py tests/test_stereo.py` still pass.

---

### Task 2: Statistics accrue to both classes

**Files:** `src/sieve/level.py` (`fit_level`). Test: same file.

- [ ] **Step 1: failing test**

```python
def _pair(**kw):
    s_cfg = _config(CORPUS, stereo=True, **kw)
    b_cfg = _config(CORPUS, stereo=False, **kw)
    s_batch, b_batch = _batch(CORPUS, s_cfg), _batch(CORPUS, b_cfg)
    maps = [
        _class_map(ls.blind, lb.labels)
        for ls, lb in zip(refine(s_batch, s_cfg), refine(b_batch, b_cfg), strict=True)
    ]
    return sieve.fit(s_batch, s_cfg), sieve.fit(b_batch, b_cfg), maps, s_batch, b_batch


def test_blind_classes_carry_exactly_the_stereo_blind_statistics():
    s, b, maps, *_ = _pair()
    for k, (ls, lb, m) in enumerate(zip(s.levels, b.levels, maps, strict=True)):
        ids = np.flatnonzero(class_kinds(ls) & KIND_BLIND)
        assert ids.size == lb.n_classes
        np.testing.assert_array_equal(ls.count[ids], lb.count[m[ids]])
        np.testing.assert_array_equal(ls.mean[ids], lb.mean[m[ids]])
        np.testing.assert_array_equal(ls.msd[ids], lb.msd[m[ids]])
        if k:
            np.testing.assert_array_equal(maps[k - 1][ls.parent[ids]], lb.parent[m[ids]])


def test_an_aware_only_class_holds_its_own_atoms():
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    model, labels = sieve.fit(batch, cfg), refine(batch, cfg)
    for lv, fl in zip(labels, model.levels, strict=True):
        for c in np.flatnonzero(class_kinds(fl) == KIND_AWARE):
            members = lv.labels == c
            assert fl.count[c] == members.sum()
            np.testing.assert_allclose(fl.mean[c], batch.y[members].mean(axis=0))
```

- [ ] **Step 2:** run → the stereo model's blind counts include no aware-only atoms yet
  (currently `FrozenLevel` counts only aware labels), so the first test fails.

- [ ] **Step 3: implement.** Move the body of `fit_level` into
  `_reduce(labels, y, nc) -> (count, mean, msd)` unchanged, then

```python
def fit_level(level: LevelLabels, y: np.ndarray) -> FrozenLevel:
    nc = level.n_classes
    # Blind classes reduce over every atom in atom order, exactly as a
    # stereo-blind fit does, so their statistics are bit-identical to it.
    count, mean, msd = _reduce(level.blind, y, nc)
    if level.kind is not None:
        differs = level.labels != level.blind
        if differs.any():
            # An atom whose aware class differs from its blind one sits in an
            # aware-only class (spec §3), which holds exactly those atoms.
            c2, m2, s2 = _reduce(level.labels[differs], y[differs], nc)
            only = level.kind == KIND_AWARE
            count[only], mean[only], msd[only] = c2[only], m2[only], s2[only]
    return FrozenLevel(
        level.signatures, count, mean, msd, level.parent, level.kind, level.blind_of
    )
```

- [ ] **Step 4:** both tests pass; `tests/test_level.py` passes.

---

### Task 3: Continuation and τ² over blind children

**Files:** `src/sieve/continuation.py`. Test: same file.

**Interfaces produced:** `continuation.aware_variance(model) -> list[float]` (per level; `nan`
when no blind class has two aware refinements).

- [ ] **Step 1: failing tests**

```python
EB = dict(class_estimator="continuation", shrinkage_weight="empirical_bayes")


def test_continuation_and_tau_squared_match_the_stereo_blind_fit():
    from sieve.continuation import (
        atom_variance, child_counts, class_means, class_sibling_variance,
        sibling_variance,
    )

    s, b, maps, *_ = _pair(**EB)
    np.testing.assert_allclose(sibling_variance(s), sibling_variance(b), rtol=1e-12)
    np.testing.assert_allclose(atom_variance(s), atom_variance(b), rtol=1e-12)
    for k, m in enumerate(maps):
        ids = np.flatnonzero(class_kinds(s.levels[k]) & KIND_BLIND)
        for f in (class_means, child_counts, class_sibling_variance):
            np.testing.assert_allclose(f(s)[k][ids], f(b)[k][m[ids]], rtol=1e-12)


def test_an_aware_only_class_has_no_children_and_keeps_its_pooled_mean():
    from sieve.continuation import child_counts, class_means

    cfg = _config(CORPUS, stereo=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    for k, lv in enumerate(model.levels):
        only = class_kinds(lv) == KIND_AWARE
        assert np.all(child_counts(model)[k][only] == 0)
        np.testing.assert_array_equal(class_means(model)[k][only], lv.mean[only])


def test_aware_variance_is_the_debiased_spread_about_the_blind_counterpart():
    from sieve.continuation import aware_variance

    cfg = _config(CORPUS, stereo=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    tau = aware_variance(model)
    assert len(tau) == model.config.n_levels
    assert all(t != t or t >= 0.0 for t in tau)
    assert tau[0] != tau[0]  # attribute level: no aware-only class
```

- [ ] **Step 2:** run → `aware_variance` missing; the first test fails on levels ≥ 2
  (children counted over aware-only classes too).

- [ ] **Step 3: implement.** Add

```python
def _blind_children(model, c):
    """The child level and a mask of its blind classes (spec §5): an atom
    sits in two children where the chains diverge, so only one may count."""
    child = model.levels[c]
    return child, (class_kinds(child) & KIND_BLIND) != 0
```

and apply the mask in `child_counts` (`child.parent[mask]`), `class_means`
(`parent_of_child = child.parent[mask]`, `source = (...)[mask]`), `sibling_variance` and
`class_sibling_variance` (`par`, `child.mean[mask, j]`, `child.msd[mask, j]`,
`child.count[mask]`). `atom_variance` weights by `count * blind_mask`. New:

```python
def aware_variance(model) -> list[float]:
    r"""Per-level $\hat\tau^2_{\mathrm{aware},k}$: the spread of aware-class
    means about the other refinements of the same blind class, debiased for
    sampling noise like ``sibling_variance`` (spec §5). Groups are the
    aware-flagged classes sharing a ``blind_of``; only groups of two or more
    enter. ``nan`` where no group qualifies."""
    out: list[float] = []
    for lvl in model.levels:
        aware = np.flatnonzero(class_kinds(lvl) & KIND_AWARE)
        group = blind_targets(lvl)[aware]
        size = np.bincount(group, minlength=lvl.n_classes).astype(np.float64)
        keep = size >= 2
        dof = (size[keep] - 1).sum()
        if lvl.kind is None or dof <= 0:
            out.append(float("nan"))
            continue
        members = keep[group]
        inv_n = 1.0 / np.maximum(lvl.count[aware], 1)
        total = 0.0
        for j in range(lvl.mean.shape[1]):
            cm = lvl.mean[aware, j]
            s1 = np.bincount(group, weights=cm, minlength=lvl.n_classes)
            s2 = np.bincount(group, weights=cm * cm, minlength=lvl.n_classes)
            with np.errstate(invalid="ignore", divide="ignore"):
                ss = s2 - s1 * s1 / np.maximum(size, 1)
            msw = ss[keep].sum() / dof
            noise = float(np.mean(lvl.msd[aware, j][members] * inv_n[members]))
            total += max(0.0, msw - noise)
        out.append(float(total))
    return out
```

- [ ] **Step 4:** tests pass; `tests/test_continuation.py tests/test_shrinkage.py` pass.

---

### Task 4: Shrinkage toward the blind counterpart

**Files:** `src/sieve/shrinkage.py`, `src/sieve/uncertainty.py`. Test: same file.

- [ ] **Step 1: failing tests**

```python
@pytest.mark.parametrize("rule", [
    dict(shrinkage_weight="count", shrinkage_strength=2.0), EB,
])
def test_blind_shrunk_means_match_and_aware_only_shrink_toward_blind(rule):
    from sieve.shrinkage import empirical_bayes_weights, shrunk_means

    s, b, maps, *_ = _pair(**rule)
    ss, sb = shrunk_means(s), shrunk_means(b)
    for k, m in enumerate(maps):
        lv = s.levels[k]
        ids = np.flatnonzero(class_kinds(lv) & KIND_BLIND)
        np.testing.assert_allclose(ss[k][ids], sb[k][m[ids]], rtol=1e-12)
        only = np.flatnonzero(class_kinds(lv) == KIND_AWARE)
        if only.size:
            n = lv.count[only].astype(float)[:, None]
            w = (empirical_bayes_weights(s)[k][only][:, None]
                 if "empirical_bayes" in rule.values() else n / (n + 2.0))
            target = ss[k][blind_targets(lv)[only]]
            np.testing.assert_allclose(ss[k][only], w * lv.mean[only] + (1 - w) * target)


def test_predictive_variance_is_total_on_aware_only_classes():
    from sieve.uncertainty import predictive_variance

    cfg = _config(CORPUS, stereo=True, predictive_variance=True, **EB)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    for table in predictive_variance(model):
        assert np.all(np.isfinite(table)) and np.all(table > 0)
```

(`predictive_variance` is a `SieveConfig` field; if its name differs, use the field
`Predictions.predictive_variance` is gated on.)

- [ ] **Step 2:** run → aware-only classes are still shrunk toward their aware parent.

- [ ] **Step 3: implement.** In `empirical_bayes_weights`, compute each level's `w` as now,
  then for `only = class_kinds(lvl) == KIND_AWARE`: `t = aware_variance(model)[k]`;
  `w[only] = 1.0` if `t` is `nan`, `0.0` if `t <= 0`, else
  `n[only] / (n[only] + sigma[k] / t)`. Restructure the early `continue`s into assignments
  so the override always runs. In `shrunk_means`, after building level *k*'s array and before
  appending:

```python
        only = class_kinds(lvl) == KIND_AWARE
        if only.any() and (eb or (applies and not diversity)):
            # spec §5: an aware-only class refines its blind counterpart at the
            # same radius, so that is what it is shrunk toward, not its parent.
            w = (eb_w[k][only][:, None] if eb
                 else n[only] / (n[only] + shrinkage_strength))
            level_out[only] = w * raw[only] + (1.0 - w) * level_out[blind_targets(lvl)[only]]
```

  (Diversity keeps its literal rule: an aware-only class has no children, so λ = 0.)
  In `predictive_variance`, for aware-only rows replace the estimation term's
  `tau_parent` with `aware_variance(model)[k]` (0 where `nan`).

- [ ] **Step 4:** tests pass; `tests/test_shrinkage.py tests/test_uncertainty.py` pass.

---

### Task 5: Prediction

**Files:** `src/sieve/predict.py`. Test: same file.

**Interfaces produced:** `Predictions.stereo_refined: np.ndarray | None` (bool, set iff
`cfg.stereo`). `class_id`, `support`, `variance`, `raw_value`, `shrinkage_weight` describe
the answering class.

- [ ] **Step 1: failing tests**

```python
TRANS = ["C/C=C/C", "C/C=C/CC", "CC/C=C/CC", "C/C=C/CO"]
CIS = [r"C/C=C\C", r"C/C=C\CC", r"CC/C=C\CC", r"C/C=C\CO"]


@pytest.mark.parametrize("rule", [{}, dict(shrinkage_weight="count", shrinkage_strength=2.0), EB])
def test_unrefined_atoms_get_exactly_the_stereo_blind_prediction(rule):
    s, b, _, s_batch, b_batch = _pair(**rule)
    ps, pb = sieve.predict_detailed(s, s_batch), sieve.predict_detailed(b, b_batch)
    assert ps.stereo_refined is not None and ps.stereo_refined.any()
    keep = ~ps.stereo_refined
    np.testing.assert_allclose(ps.value[keep], pb.value[keep], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(ps.matched_level, pb.matched_level)
    assert pb.stereo_refined is None


def test_an_isomer_absent_from_training_falls_back_to_the_blind_answer():
    smiles = TRANS + CIS
    s_cfg, b_cfg = _config(smiles, stereo=True, **EB), _config(smiles, stereo=False, **EB)
    s = sieve.fit(_batch(TRANS, s_cfg), s_cfg)
    b = sieve.fit(_batch(TRANS, b_cfg), b_cfg)
    ps = sieve.predict_detailed(s, from_rdkit([Chem.MolFromSmiles(x) for x in CIS], config=s_cfg))
    pb = sieve.predict(b, from_rdkit([Chem.MolFromSmiles(x) for x in CIS], config=b_cfg))
    assert not ps.stereo_refined.any()
    np.testing.assert_allclose(ps.value, pb, rtol=1e-12, atol=1e-15)


def test_both_butenes_are_answered_by_their_own_aware_classes():
    cfg = _config(["C/C=C/C"], stereo=True, depth=3)
    pair = ["C/C=C/C", r"C/C=C\C"]
    batch = from_rdkit([Chem.MolFromSmiles(x) for x in pair], config=cfg)
    y = np.array([[1.0], [2.0], [2.0], [1.0], [3.0], [4.0], [4.0], [3.0]])
    model = sieve.fit(dataclasses.replace(batch, y=y), cfg)
    p = sieve.predict_detailed(model, batch)
    assert p.stereo_refined.all()
    np.testing.assert_array_equal(p.value, y)


def test_predictions_do_not_depend_on_a_pentavalent_neighbour():
    cfg = _config([*CORPUS, PHOSPHORUS], stereo=True, **EB)
    model = sieve.fit(_batch([*CORPUS, PHOSPHORUS], cfg), cfg)
    mols = [Chem.MolFromSmiles(x) for x in CORPUS]
    alone = sieve.predict(model, from_rdkit(mols, config=cfg))
    beside = sieve.predict(model, from_rdkit([*mols, Chem.MolFromSmiles(PHOSPHORUS)], config=cfg))
    np.testing.assert_array_equal(alone, beside[: alone.shape[0]])


def test_predict_loo_refuses_a_stereo_model():
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    with pytest.raises(NotImplementedError, match="stereo"):
        sieve.predict_loo(sieve.fit(batch, cfg), batch)
```

- [ ] **Step 2:** run → `stereo_refined` missing.

- [ ] **Step 3: implement** in `_search`:
  - refuse `loo_y is not None and cfg.stereo` with `NotImplementedError` naming the reason
    (an atom contributes to two classes at a level);
  - per backoff level: `cid = found[q.blind]` (the walk) and `aware_cid[hit] = found[q.labels][hit]`
    with `aware_cid = np.full(n, -1, np.int64)` before the loop — `hit` shrinks level by level,
    so the last write is at *k*\*;
  - after the loop, when `cfg.stereo`:

```python
    refined = np.zeros(n, bool)
    if cfg.stereo:
        # spec §6: one aware lookup at the incumbent's k*. aware != blind
        # there means an aware-only class, which exists only if every class
        # its signature names does, so a miss anywhere below is already -1.
        for k in backoff_path:
            idx = np.flatnonzero((matched == k) & (aware_cid >= 0) & (aware_cid != class_id))
            if idx.size == 0:
                continue
            a = aware_cid[idx]
            idx, a = idx[model.levels[k].count[a] >= cfg.minimum_support], a[model.levels[k].count[a] >= cfg.minimum_support]
            refined[idx] = True
            class_id[idx] = a
            support[idx] = model.levels[k].count[a]
            variance[idx] = model.levels[k].variance[a]
            value[idx] = means[k][a]
```

  - in the shrinkage block, after the per-level loop: `value[refined] =
    shrunk[k][class_id]` for each level's refined atoms, and `weight` = the EB weight,
    `n/(n+strength)` under count, `1.0` under diversity;
  - pass `stereo_refined=refined if cfg.stereo else None` to both `Predictions` returns.
    The predictive-variance loop already reads `class_id`.

- [ ] **Step 4:** tests pass; `tests/test_predict.py` passes.

---

### Task 6: Merge

**Files:** `src/sieve/merge.py`. Test: same file.

- [ ] **Step 1: failing tests**

```python
def _canonical(level):
    order = np.lexsort(level.signatures.T[::-1])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return level.signatures[order], class_kinds(level)[order], rank[blind_targets(level)][order]


@pytest.mark.parametrize("cuts", [(6,), (3, 7)])
def test_merge_reproduces_kinds_and_blind_counterparts(cuts):
    cfg = _config(CORPUS, stereo=True)
    batch = _batch(CORPUS, cfg)
    edges = [0, *cuts, len(CORPUS)]
    shards = [batch[(batch.graph_id >= lo) & (batch.graph_id < hi)]
              for lo, hi in zip(edges, edges[1:])]
    merged = sieve.fit(shards[0], cfg)
    for shard in shards[1:]:
        merged = merged.merge(sieve.fit(shard, cfg))
    whole = sieve.fit(batch, cfg)
    for lm, lw in zip(merged.levels, whole.levels, strict=True):
        for x, y in zip(_canonical(lm), _canonical(lw), strict=True):
            np.testing.assert_array_equal(x, y)


def test_a_shard_holding_a_pentavalent_atom_mints_no_class_for_a_shared_molecule():
    cfg = _config([*CORPUS, PHOSPHORUS], stereo=True)
    with_p = sieve.fit(_batch([*CORPUS, PHOSPHORUS], cfg), cfg)
    without = sieve.fit(_batch(CORPUS, cfg), cfg)
    merged = with_p.merge(without)
    assert [lv.n_classes for lv in merged.levels] == [lv.n_classes for lv in with_p.levels]


def test_merging_into_the_empty_model_keeps_the_kinds():
    cfg = _config(CORPUS, stereo=True)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    merged = sieve.SieveModel.empty(cfg).merge(model)
    for lm, lw in zip(merged.levels, model.levels, strict=True):
        np.testing.assert_array_equal(class_kinds(lm), class_kinds(lw))
```

(`_canonical` compares signatures and `kind`; since signatures are in merged ids, sort both
sides by the per-level canonical rank before comparing if raw signatures differ — reuse
`tests/test_merge.py::_assert_same_statistics`'s ordering if it is simpler.)

- [ ] **Step 2:** run → merged levels have `kind is None`.

- [ ] **Step 3: implement** in `merge_level`, after `count[i]` is updated:

```python
    kind = blind_of = None
    if a.kind is not None or b.kind is not None:
        # spec §7: kinds combine by OR, blind_of is remapped like parent. A
        # stereo-blind or empty side reads as all-both / identity.
        kind = np.zeros(n_new, np.uint8)
        kind[:m] = class_kinds(a)
        kind[i] |= class_kinds(b)
        blind_of = np.arange(n_new, dtype=np.int64)
        blind_of[:m] = blind_targets(a)
        b_blind_of = remap[blind_targets(b)].astype(np.int64)
        if np.any(both) and not np.array_equal(blind_of[i][both], b_blind_of[both]):
            raise AssertionError("blind_of disagreement: the blind counterpart moved")
        blind_of[i] = np.where(nA > 0, blind_of[i], b_blind_of)
    return FrozenLevel(uniq, count, mean, msd, parent, kind, blind_of), remap
```

- [ ] **Step 4:** tests pass; `tests/test_merge.py` passes.

---

### Task 7: Schema marker, save/load, analytic refusal

**Files:** `src/sieve/config.py`, `src/sieve/model.py`, `experiments/experiments/analytic.py`.
Tests: same file, plus `experiments/tests/` for the refusal.

- [ ] **Step 1: failing tests**

```python
def test_the_schema_marks_the_construction_only_when_stereo_is_on():
    import hashlib, json

    on, off = _config(CORPUS, stereo=True), _config(CORPUS, stereo=False)
    assert on._schema_payload()["stereo_construction"] == "refines_blind"
    assert "stereo_construction" not in off._schema_payload()
    fused = {k: v for k, v in on._schema_payload().items() if k != "stereo_construction"}
    blob = json.dumps(fused, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() != on.schema_version


def test_save_and_load_keep_kinds_and_blind_counterparts(tmp_path):
    cfg = _config(CORPUS, stereo=True)
    model = sieve.fit(_batch(CORPUS, cfg), cfg)
    model.save(tmp_path / "m.npz")
    loaded = sieve.SieveModel.load(tmp_path / "m.npz")
    for a, b in zip(model.levels, loaded.levels, strict=True):
        assert (a.kind is None) == (b.kind is None)
        np.testing.assert_array_equal(class_kinds(a), class_kinds(b))
        np.testing.assert_array_equal(blind_targets(a), blind_targets(b))
```

  and in `experiments/tests/test_analytic.py` (or wherever `sieve_train_stats` is tested):
  `sieve_train_stats` on a stereo model raises `NotImplementedError` matching `"stereo"`.

- [ ] **Step 2:** run → `_schema_payload` missing.

- [ ] **Step 3: implement.** Split `schema_version` into `_schema_payload() -> dict` and the
  hash; add `payload["stereo_construction"] = "refines_blind"` inside the existing
  `if self.stereo:`. `save` writes `level_{k}_kind` / `level_{k}_blind_of` when
  `lvl.kind is not None`; `load` reads them when the key is in `data.files`, else `None`.
  `sieve_train_stats` raises at entry when `model.config.stereo`: the walk it reproduces
  assumes each atom sits in one class per level.

- [ ] **Step 4:** tests pass; `tests/test_io.py tests/test_config.py` pass (the existing stereo
  save/load test must survive the digest change because it saves and loads under the new code).

---

### Task 8: Whole-suite gate, docs, PR

- [ ] Update `docs/superpowers/specs/2026-09-22-cis-trans-featurisation-design.md`'s status
  line to point at the new spec for §5.4, and the spec's status to "implemented".
- [ ] Update docstrings that describe the fused construction (`refine.py` stereo block,
  `predict.Predictions`).
- [ ] `.venv/bin/ruff check src tests experiments && .venv/bin/ruff format --check src tests experiments && .venv/bin/ty check src tests experiments`
- [ ] `.venv/bin/python -m pytest -q` (≈12 min).
- [ ] Commit, push `stereo-refines-blind`, open one PR, merge when CI is green.
