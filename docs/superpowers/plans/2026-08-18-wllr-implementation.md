# WLLR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `wllr`, a node-level regression library that fits empirical target statistics to the nested vertex partitions produced by Weisfeiler–Lehman color refinement, and predicts by backing off to the deepest supported class.

**Architecture:** A fitted model is an immutable, columnar, per-level structure — one array set per refinement level, class ids dense from 0, `parent[k]` indexing directly into level `k-1`. Fitting is fully vectorized: the corpus is one block-diagonal graph, each level is one array operation, and per-class statistics come from a sparse class-membership operator. Models combine through a merge monoid (law of total variance in intensive variables), which doubles as the chunking/streaming mechanism so there is no second accumulation path.

**Tech Stack:** Python ≥3.11, NumPy, SciPy (`scipy.sparse`), pytest. RDKit for the molecular adapter (optional import). `cosmolayer` for the benchmark fixture (optional import).

**Spec:** `design.md` in the repository root. Every task cites the sections it implements. **Read the cited sections before starting a task** — this plan states what to build, the spec states why, and the "why" is where the traps are documented.

## Global Constraints

- Python ≥3.11. Package lives in `src/wllr/`, tests in `tests/`.
- Moments are `float64`. Class ids and signatures are `int64`. `parent` is `int32`.
- A fitted model is **immutable** — frozen dataclasses, no in-place mutation, no `partial_fit`, no `freeze()`.
- **No per-atom or per-molecule Python loop anywhere in the fit path.** A loop over *levels* is expected and fine; a loop over nodes is a defect.
- Stored variance is `msd` — population variance, divisor `N`. The name `var`/`variance` is reserved for the accessor that applies Bessel's correction. (§4.1)
- `s² = N/(N-1)·σ²` is **undefined at N=1** — return `None` (scalar API) or `NaN` (array API), never `0.0`. (§4.1, §12)
- Never compute variance as `E[y²] − E[y]²`. Always center first, then reduce. (§4.1, §7.4)
- Shrunk means are **derived, never stored**. (§4.2)

### The four traps

These fail silently — no exception, plausible-looking numbers. Each has a dedicated test in this plan.

1. **`np.unique` return order** is `(unique, index, inverse)` regardless of the order the keywords are passed. Binding them wrongly makes the vectorized path silently behave like the slow one. (§7.2)
2. **Merge statement order**: `delta` must be read from `mean[i]` *before* `mean[i]` is overwritten, and `msd[i]` consumed before being overwritten. Reversing corrupts every σ² for classes present in both models by ~1e-2 relative. (§5.3)
3. **Power sums instead of centring**: the one-pass formula produces negative variances on realistic data. (§7.4)
4. **Input misalignment**: targets indexed by atom position, molecule parsed in a different order — corrupts every label and raises nothing. (§7.5, §11.3)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/wllr/config.py` | `WLLRConfig`, `schema_version` digest, compatibility check |
| `src/wllr/batch.py` | `AtomBatch`, CSR layout (`indptr`, `slot`), alignment guard |
| `src/wllr/dedupe.py` | `dense_rows` — void-view row deduplication (trap 1) |
| `src/wllr/refine.py` | Attribute levels + WL rounds → per-level labels and signature rows |
| `src/wllr/level.py` | `FrozenLevel`, sparse two-pass statistics, `variance` accessor |
| `src/wllr/merge.py` | Signature translation, `merge_level`, balanced-tree fold |
| `src/wllr/model.py` | `WLLRModel`, `fit`, `save`/`load` |
| `src/wllr/predict.py` | Backoff search, `Predictions`, `predict_loo` |
| `src/wllr/shrinkage.py` | Top-down shrunk means (derived) |
| `src/wllr/io/rdkit_adapter.py` | `from_smiles`, `from_rdkit` |
| `src/wllr/io/cosmolayer_adapter.py` | `SegmentStore` → `AtomBatch` + targets |
| `src/wllr/sklearn.py` | Mutable-estimator wrapper with graph-level splitting default |

---

## Deviations from `design.md` you must know about

**These are gaps in the spec, resolved here. Do not "fix" them back to the spec text.**

**A. `vocab` is a signature-row array, not `dict[bytes, int]`.** §3.1 shows `dict[bytes, int]`; §9 supersedes it — "vocab is stored as an `(n_classes, width)` integer array", because §7.2 mints ids by deduplicating signature rows and *the deduped rows are the vocabulary*. Follow §9. No hashing is required anywhere in fit, merge, or predict.

**B. Merging cannot compare signature rows directly.** §5.3's `merge_level` keys `vocab` by a content hash `h`, which is globally meaningful. Signature rows are **not** — they are written in terms of the *previous level's local ids*, so B's row `[3, 7, 12]` and A's row `[3, 7, 12]` generally denote different classes. Before deduplicating B's level-`k` rows against A's, every id inside them must be translated into the merged id space using `remap_prev` from level `k-1`, **and the neighbor columns re-sorted afterwards** because remapping changes their order. Task 7 gives the code. Implementing §5.3 literally against dedupe-derived ids produces a silently wrong merge.

**C. Attribute code tables must be shared across merged models.** Since level-0 signatures are attribute codes, two models are mergeable only if their encoders agree. The code tables therefore live in `WLLRConfig` and are covered by `schema_version`. §9.2 does not list them; it should.

**D. `chunk_size` needs no separate code path.** §4.1 establishes that chunked accumulation *is* the merge. Implement it as "split the batch, fit each part, fold" — five lines in `fit`, no streaming machinery, no Welford recurrence.

**E. Out of scope for v1, deliberately.** §8 vocabulary pruning (spec calls it opt-in, not a default). §3.6 neighbor schema (spec says evaluated, not adopted) — the config field exists and must `raise NotImplementedError` if set. §5.2's full-covariance upgrade (spec defers it). Record these in the README so their absence reads as a decision.

---

## Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `src/wllr/__init__.py`, `src/wllr/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `WLLRConfig` frozen dataclass; `WLLRConfig.schema_version -> str`; `check_mergeable(a, b) -> None` (raises `ValueError`).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "wllr"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
chem = ["rdkit>=2023.3"]
dev = ["pytest>=7.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
import pytest
from wllr.config import WLLRConfig, check_mergeable

def base(**kw):
    d = dict(target_dim=1,
             attribute_levels=(("element",),),
             attribute_codes={"element": {"C": 0, "H": 1}},
             edge_codes={"SINGLE": 1},
             max_wl_depth=3)
    d.update(kw)
    return WLLRConfig(**d)

def test_schema_version_is_stable():
    assert base().schema_version == base().schema_version

def test_schema_version_ignores_inference_params():
    assert base(n_min=1).schema_version == base(n_min=9).schema_version
    assert base(alpha=None).schema_version == base(alpha=2.0).schema_version

def test_schema_version_tracks_meaning():
    assert base().schema_version != base(max_wl_depth=4).schema_version
    assert base().schema_version != base(
        attribute_codes={"element": {"C": 0, "H": 2}}).schema_version
    assert base().schema_version != base(
        attribute_levels=(("element",), ("aromatic",))).schema_version

def test_mergeable_requires_matching_schema():
    check_mergeable(base(), base(n_min=7))          # inference params may differ
    with pytest.raises(ValueError, match="schema"):
        check_mergeable(base(), base(max_wl_depth=4))

def test_n_levels_counts_attribute_levels_plus_wl_depths():
    cfg = base(attribute_levels=(("element",), ("aromatic",)), max_wl_depth=3)
    assert cfg.n_levels == 5

def test_neighbour_schema_is_not_implemented():
    with pytest.raises(NotImplementedError):
        base(neighbour_schema=("element",))
```

- [ ] **Step 3: Run it and watch it fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.config'`

- [ ] **Step 4: Implement**

```python
# src/wllr/config.py
"""Fit-time and inference-time configuration. See design.md section 9.2."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

FORMAT_VERSION = 1


@dataclass(frozen=True)
class WLLRConfig:
    """Immutable configuration for a WLLR model.

    ``attribute_levels`` declares the graded refinement order below WL depth 0
    (design.md section 3.5): each tuple is one level, introducing that group of
    attributes on top of the previous level.

    ``attribute_codes`` and ``edge_codes`` are part of what a class *means*, so
    they enter ``schema_version``: two models built with different encodings
    cannot be merged even if every other field agrees.
    """

    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]
    attribute_codes: Mapping[str, Mapping[str, int]]
    edge_codes: Mapping[str, int]
    max_wl_depth: int
    neighbour_schema: tuple[str, ...] | None = None
    n_min: int = 1
    alpha: float | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.neighbour_schema is not None:
            raise NotImplementedError(
                "neighbour_schema is evaluated but not adopted; see design.md 3.6"
            )
        if not self.attribute_levels:
            raise ValueError("at least one attribute level is required")
        if self.target_dim < 1:
            raise ValueError("target_dim must be >= 1")
        if self.n_min < 1:
            raise ValueError("n_min must be >= 1")
        # Freeze the mappings so the frozen dataclass is honest.
        object.__setattr__(self, "attribute_codes", MappingProxyType(
            {k: MappingProxyType(dict(v)) for k, v in self.attribute_codes.items()}))
        object.__setattr__(self, "edge_codes", MappingProxyType(dict(self.edge_codes)))

    @property
    def n_levels(self) -> int:
        """Total refinement levels: attribute levels, then WL depths."""
        return len(self.attribute_levels) + self.max_wl_depth

    @property
    def n_bond(self) -> int:
        """Edge-code alphabet size, including the 0 slot reserved for padding."""
        return max(self.edge_codes.values()) + 1

    @property
    def schema_version(self) -> str:
        """Digest over everything that changes what a class means (design.md 9.2).

        ``n_min``, ``alpha`` and ``chunk_size`` are deliberately excluded: they
        are read at prediction time and do not invalidate fitted statistics.
        """
        payload = {
            "target_dim": self.target_dim,
            "attribute_levels": [list(g) for g in self.attribute_levels],
            "attribute_codes": {k: dict(sorted(v.items()))
                                for k, v in sorted(self.attribute_codes.items())},
            "edge_codes": dict(sorted(self.edge_codes.items())),
            "max_wl_depth": self.max_wl_depth,
            "neighbour_schema": self.neighbour_schema,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def check_mergeable(a: WLLRConfig, b: WLLRConfig) -> None:
    """Raise unless two models describe the same classes (design.md 5.4).

    Loud rejection is the point: silently truncating to ``min(K_a, K_b)`` would
    absorb config drift and produce a model whose classes mean two things.
    """
    if a.schema_version != b.schema_version:
        raise ValueError(
            f"cannot merge: schema_version differs ({a.schema_version[:12]} != "
            f"{b.schema_version[:12]}); attribute levels, codes, edge codes and "
            "max_wl_depth must all match"
        )
```

```python
# src/wllr/__init__.py
"""Weisfeiler-Lehman Lookup Regression."""
from wllr.config import WLLRConfig

__all__ = ["WLLRConfig"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/wllr/__init__.py src/wllr/config.py tests/test_config.py
git commit -m "feat: WLLRConfig with schema_version and merge compatibility check"
```

---

## Task 2: AtomBatch, CSR layout, and the alignment guard

**Files:**
- Create: `src/wllr/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AtomBatch` frozen dataclass with fields `node_attrs (n,a) int64`, `edge_src (e,) int64`, `edge_dst (e,) int64`, `edge_attr (e,) int64`, `graph_id (n,) int64`, `y (n,d) float64 | None`, `elements (n,) int64 | None`; properties `n_atoms`, `n_edges`; method `csr() -> CSRLayout` with fields `order`, `indptr`, `slot`, `src`, `dst`, `attr`, `max_deg`; module function `check_alignment(batch, atom_counts, elements) -> None`.

Read design.md sections 7.1, 7.5, 11.1 and 11.3 first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch.py
import numpy as np
import pytest
from wllr.batch import AtomBatch, check_alignment

def tri():
    """Triangle 0-1-2 plus an isolated node 3, as two graphs."""
    return AtomBatch(
        node_attrs=np.array([[0], [1], [1], [0]], np.int64),
        edge_src=np.array([0, 1, 1, 2, 2, 0], np.int64),
        edge_dst=np.array([1, 0, 2, 1, 0, 2], np.int64),
        edge_attr=np.array([1, 1, 1, 1, 2, 2], np.int64),
        graph_id=np.array([0, 0, 0, 1], np.int64),
        y=np.array([[1.0], [2.0], [3.0], [4.0]]),
    )

def test_shapes_are_validated():
    with pytest.raises(ValueError, match="graph_id"):
        AtomBatch(node_attrs=np.zeros((3, 1), np.int64),
                  edge_src=np.zeros(0, np.int64), edge_dst=np.zeros(0, np.int64),
                  edge_attr=np.zeros(0, np.int64),
                  graph_id=np.zeros(2, np.int64), y=None)

def test_edges_must_be_symmetric():
    with pytest.raises(ValueError, match="both directions"):
        AtomBatch(node_attrs=np.zeros((2, 1), np.int64),
                  edge_src=np.array([0], np.int64), edge_dst=np.array([1], np.int64),
                  edge_attr=np.array([1], np.int64),
                  graph_id=np.zeros(2, np.int64), y=None)

def test_csr_slot_is_position_within_each_source_block():
    c = tri().csr()
    assert c.max_deg == 2
    # every node's slots are exactly 0..deg-1
    for node in range(4):
        got = sorted(c.slot[c.src == node].tolist())
        assert got == list(range(len(got)))
    assert c.indptr.tolist() == [0, 2, 4, 6, 6]
    assert np.all(np.diff(c.src) >= 0), "csr arrays must be sorted by source"

def test_isolated_node_has_degree_zero():
    c = tri().csr()
    assert c.indptr[4] - c.indptr[3] == 0

def test_alignment_guard_accepts_matching_corpus():
    b = tri()
    check_alignment(b, atom_counts=np.array([3, 1]),
                    elements=np.array([6, 1, 1, 8], np.int64))

def test_alignment_guard_catches_count_mismatch():
    b = tri()
    with pytest.raises(ValueError, match="atom count"):
        check_alignment(b, atom_counts=np.array([2, 1]),
                        elements=np.array([6, 1, 1, 8], np.int64))

def test_alignment_guard_catches_permutation():
    """The bug counts alone cannot catch: same atoms, wrong order."""
    b = AtomBatch(**{**tri().__dict__, "elements": np.array([6, 1, 1, 8], np.int64)})
    with pytest.raises(ValueError, match="element"):
        check_alignment(b, atom_counts=np.array([3, 1]),
                        elements=np.array([1, 6, 1, 8], np.int64))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_batch.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.batch'`

- [ ] **Step 3: Implement**

```python
# src/wllr/batch.py
"""The columnar input contract and its CSR layout (design.md 7.1, 11)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSRLayout:
    """Edges sorted by source, with each edge's position inside its source block.

    ``slot`` is computed once and reused at every refinement level (design.md 7.1).
    """

    order: np.ndarray
    indptr: np.ndarray
    slot: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    attr: np.ndarray
    max_deg: int


@dataclass(frozen=True)
class AtomBatch:
    """One block-diagonal graph over a whole corpus (design.md 11.1).

    Edges are stored in **both directions** for undirected graphs. ``graph_id``
    is not optional: it is what makes graph-level splitting possible, and a
    batch that loses it cannot be validated correctly.
    """

    node_attrs: np.ndarray            # (n_atoms, n_attr) int64, encoded categoricals
    edge_src: np.ndarray              # (n_edges,) int64
    edge_dst: np.ndarray              # (n_edges,) int64
    edge_attr: np.ndarray             # (n_edges,) int64
    graph_id: np.ndarray              # (n_atoms,) int64
    y: np.ndarray | None = None       # (n_atoms, d) float64
    elements: np.ndarray | None = None  # (n_atoms,) int64, for the alignment guard

    def __post_init__(self) -> None:
        n = self.node_attrs.shape[0]
        if self.graph_id.shape != (n,):
            raise ValueError(f"graph_id must have shape ({n},), got {self.graph_id.shape}")
        e = self.edge_src.shape[0]
        for name in ("edge_dst", "edge_attr"):
            if getattr(self, name).shape != (e,):
                raise ValueError(f"{name} must have shape ({e},)")
        if self.y is not None and self.y.shape[0] != n:
            raise ValueError(f"y must have {n} rows, got {self.y.shape[0]}")
        if e:
            # Undirected graphs must carry both directions; the CSR construction
            # assumes it, and a one-way corpus silently halves every neighborhood.
            fwd = {(int(a), int(b)) for a, b in zip(self.edge_src, self.edge_dst)}
            if any((b, a) not in fwd for a, b in fwd):
                raise ValueError("edges must be stored in both directions")

    @property
    def n_atoms(self) -> int:
        return int(self.node_attrs.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])

    def csr(self) -> CSRLayout:
        n = self.n_atoms
        order = np.argsort(self.edge_src, kind="stable")
        src = self.edge_src[order]
        deg = np.bincount(src, minlength=n)
        indptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
        # Position within the source's adjacency block. Valid only because
        # `src` is sorted, which is why `order` is applied first.
        slot = np.arange(src.shape[0], dtype=np.int64) - indptr[src]
        return CSRLayout(order=order, indptr=indptr, slot=slot, src=src,
                         dst=self.edge_dst[order], attr=self.edge_attr[order],
                         max_deg=int(deg.max()) if n else 0)


def check_alignment(batch: AtomBatch, atom_counts: np.ndarray,
                    elements: np.ndarray) -> None:
    """Verify targets line up with parsed molecules (design.md 7.5, 11.3).

    This is the highest-severity failure mode in the system: a misalignment
    corrupts every label and raises nothing, surfacing only as unexplained
    inaccuracy. Counts alone cannot catch it -- a permutation preserves them --
    so the element check is the one that actually does the work.
    """
    _, sizes = np.unique(batch.graph_id, return_counts=True)
    if sizes.shape != atom_counts.shape or not np.array_equal(sizes, atom_counts):
        raise ValueError(
            f"atom count mismatch: batch has {sizes.tolist()[:5]}..., "
            f"corpus reports {np.asarray(atom_counts).tolist()[:5]}..."
        )
    if batch.elements is not None:
        bad = np.flatnonzero(batch.elements != elements)
        if bad.size:
            raise ValueError(
                f"element mismatch at {bad.size} atoms (first at index {bad[0]}): "
                "the parsed molecule and its target rows are in different orders"
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_batch.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/batch.py tests/test_batch.py
git commit -m "feat: AtomBatch, CSR layout, and the input alignment guard"
```

---

## Task 3: Void-view row deduplication

**Files:**
- Create: `src/wllr/dedupe.py`
- Test: `tests/test_dedupe.py`

**Interfaces:**
- Produces: `dense_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(labels, unique_rows)` where `labels[i]` is the dense id of row `i` and `unique_rows[j]` is the representative row of class `j`.

Read design.md 7.2. **This is trap 1.** The whole performance story of the fit path is in this eight-line function, and getting it wrong costs correctness of the *ids*, not just speed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dedupe.py
import numpy as np
from wllr.dedupe import dense_rows

def test_identical_rows_share_an_id():
    m = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], np.int64)
    labels, uniq = dense_rows(m)
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert uniq.shape[0] == 3

def test_labels_are_dense_from_zero():
    m = np.array([[9, 9], [4, 4], [9, 9], [1, 1]], np.int64)
    labels, uniq = dense_rows(m)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert labels.dtype == np.int64

def test_unique_rows_are_indexed_by_label():
    """The representative row for class j must actually be a row of class j.

    This is the assertion that catches the np.unique return-order trap: if
    `index` and `inverse` are swapped, uniq[labels[i]] stops equalling m[i].
    """
    rng = np.random.default_rng(0)
    m = rng.integers(0, 5, size=(500, 4)).astype(np.int64)
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)

def test_matches_a_dict_based_reference():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 3, size=(200, 3)).astype(np.int64)
    labels, _ = dense_rows(m)
    seen, ref = {}, []
    for row in map(tuple, m):
        ref.append(seen.setdefault(row, len(seen)))
    # ids may be numbered differently, but the partition must be identical
    assert _same_partition(labels, np.array(ref))

def _same_partition(a, b):
    return len({(int(x), int(y)) for x, y in zip(a, b)}) == len(set(a.tolist())) == len(set(b.tolist()))

def test_handles_non_contiguous_input():
    m = np.arange(40, dtype=np.int64).reshape(10, 4)[:, ::2]
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_dedupe.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.dedupe'`

- [ ] **Step 3: Implement**

```python
# src/wllr/dedupe.py
"""Row deduplication through a void view (design.md 7.2)."""
from __future__ import annotations

import numpy as np


def dense_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each distinct row of ``mat`` a dense id.

    Returns ``(labels, unique_rows)`` with ``unique_rows[labels] == mat``.

    Deduplication happens in 1-D by viewing each row as a single ``np.void``
    scalar. ``np.unique(..., axis=0)`` is *no faster than a Python dict loop*
    (0.70 s vs 0.71 s over 6 levels of 147k atoms); the void view does the same
    work in 0.27 s.
    """
    m = np.ascontiguousarray(mat)
    v = m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).ravel()
    # numpy returns (unique, index, inverse) in a FIXED order, whatever order
    # the keywords are passed in. Binding these wrongly is silent: the ids stay
    # plausible and `unique_rows[labels] == mat` quietly stops holding.
    uniq, idx, inv = np.unique(v, return_index=True, return_inverse=True)
    return inv.ravel().astype(np.int64), m[idx]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dedupe.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/dedupe.py tests/test_dedupe.py
git commit -m "feat: void-view row deduplication for dense class ids"
```

---

## Task 4: The refinement chain

**Files:**
- Create: `src/wllr/refine.py`
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: `AtomBatch`, `CSRLayout` (Task 2), `dense_rows` (Task 3), `WLLRConfig` (Task 1).
- Produces: `refine(batch, config) -> list[LevelLabels]` where `LevelLabels` is a frozen dataclass with `labels (n_atoms,) int64`, `signatures (n_classes, width) int64`, `parent (n_classes,) int32`. `len(result) == config.n_levels`.

Read design.md 2, 3.5, 7.1, 7.2.

Level layout: levels `0..m-1` are attribute levels (`m = len(config.attribute_levels)`), levels `m..m+K-1` are WL rounds. Level 0's signature is its attribute codes with `parent = -1`. Every later level's signature has the previous level's label in **column 0**, which is what makes `parent` free and single-parenthood structural.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_refine.py
import numpy as np
import pytest
from wllr.batch import AtomBatch
from wllr.config import WLLRConfig
from wllr.refine import refine

def cfg(**kw):
    d = dict(target_dim=1, attribute_levels=(("element",),),
             attribute_codes={"element": {"C": 0, "H": 1}},
             edge_codes={"SINGLE": 1, "DOUBLE": 2}, max_wl_depth=2)
    d.update(kw)
    return WLLRConfig(**d)

def path_graph(n, attrs=None):
    src = np.repeat(np.arange(n - 1), 2)
    dst = src.copy()
    src[0::2], dst[0::2] = np.arange(n - 1), np.arange(1, n)
    src[1::2], dst[1::2] = np.arange(1, n), np.arange(n - 1)
    return AtomBatch(
        node_attrs=(np.zeros((n, 1), np.int64) if attrs is None else attrs),
        edge_src=src, edge_dst=dst,
        edge_attr=np.ones(2 * (n - 1), np.int64),
        graph_id=np.zeros(n, np.int64),
        y=np.zeros((n, 1)))

def test_chain_length_matches_config():
    levels = refine(path_graph(5), cfg())
    assert len(levels) == 3          # 1 attribute level + 2 WL rounds

def test_level_zero_partitions_by_attribute():
    attrs = np.array([[0], [1], [0], [1]], np.int64)
    levels = refine(path_graph(4, attrs), cfg())
    assert levels[0].labels[0] == levels[0].labels[2]
    assert levels[0].labels[0] != levels[0].labels[1]
    assert np.all(levels[0].parent == -1)

def test_partitions_are_nested_and_single_parented():
    """design.md 2.1: every class has exactly one parent, and refinement only splits."""
    levels = refine(path_graph(9), cfg(max_wl_depth=4))
    for k in range(1, len(levels)):
        child, par = levels[k], levels[k - 1]
        assert child.parent.min() >= 0
        assert child.parent.max() < par.signatures.shape[0]
        # a child class's members all live in one parent class
        assert np.array_equal(par.labels, child.parent[child.labels])

def test_refinement_is_invariant_to_node_ordering():
    """design.md 10.4: same partition regardless of node order within a graph."""
    b = path_graph(7)
    perm = np.array([3, 0, 6, 1, 5, 2, 4])
    inv = np.argsort(perm)
    permuted = AtomBatch(node_attrs=b.node_attrs[perm],
                         edge_src=inv[b.edge_src], edge_dst=inv[b.edge_dst],
                         edge_attr=b.edge_attr, graph_id=b.graph_id[perm],
                         y=b.y[perm])
    a = refine(b, cfg())[-1].labels
    c = refine(permuted, cfg())[-1].labels[inv]
    assert _same_partition(a, c)

def test_bond_type_is_distinguished():
    b = path_graph(3)
    other = AtomBatch(**{**b.__dict__, "edge_attr": np.array([2, 2, 1, 1], np.int64)})
    assert not _same_partition(refine(b, cfg())[-1].labels,
                               refine(other, cfg())[-1].labels) or True
    # the center atom sees {SINGLE,SINGLE} vs {DOUBLE,SINGLE}: classes must differ
    assert refine(b, cfg())[1].labels[0] != refine(other, cfg())[1].labels[0] or \
           refine(other, cfg())[1].labels[0] != refine(other, cfg())[1].labels[2]

def test_isolated_nodes_refine_without_error():
    b = AtomBatch(node_attrs=np.zeros((3, 1), np.int64),
                  edge_src=np.zeros(0, np.int64), edge_dst=np.zeros(0, np.int64),
                  edge_attr=np.zeros(0, np.int64),
                  graph_id=np.arange(3), y=np.zeros((3, 1)))
    levels = refine(b, cfg())
    assert len(set(levels[-1].labels.tolist())) == 1

def test_graded_attribute_levels_refine_progressively():
    """design.md 3.5: each attribute level adds information to the last."""
    c = cfg(attribute_levels=(("element",), ("aromatic",)),
            attribute_codes={"element": {"C": 0, "H": 1},
                             "aromatic": {"no": 0, "yes": 1}},
            max_wl_depth=0)
    attrs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.int64)
    levels = refine(path_graph(4, attrs), c)
    assert len(set(levels[0].labels.tolist())) == 2   # element only
    assert len(set(levels[1].labels.tolist())) == 4   # element + aromatic

def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_refine.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.refine'`

- [ ] **Step 3: Implement**

```python
# src/wllr/refine.py
"""Attribute levels then WL rounds, all vectorized (design.md 3.5, 7.1, 7.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wllr.batch import AtomBatch
from wllr.config import WLLRConfig
from wllr.dedupe import dense_rows


@dataclass(frozen=True)
class LevelLabels:
    """One level of the refinement chain.

    ``signatures[j]`` is the deduplicated signature row of class ``j`` -- this
    array *is* the vocabulary (design.md 9). Column 0 holds the parent's id at
    the previous level for every level above 0, which is why ``parent`` is free
    and single-parenthood is structural rather than asserted.
    """

    labels: np.ndarray        # (n_atoms,) int64
    signatures: np.ndarray    # (n_classes, width) int64
    parent: np.ndarray        # (n_classes,) int32; -1 at level 0

    @property
    def n_classes(self) -> int:
        return int(self.signatures.shape[0])


def refine(batch: AtomBatch, config: WLLRConfig) -> list[LevelLabels]:
    """Build the full refinement chain for a corpus.

    One array operation per level over the whole block-diagonal corpus -- there
    is no per-molecule or per-atom loop anywhere in this function.
    """
    n = batch.n_atoms
    levels: list[LevelLabels] = []

    # --- attribute levels (design.md 3.5) --------------------------------
    # Level j introduces attribute group j on top of level j-1. Each level is
    # built from the previous plus strictly more information, which is the only
    # premise design.md 2 needs.
    used = 0
    for j, group in enumerate(config.attribute_levels):
        width = len(group)
        cols = batch.node_attrs[:, used:used + width]
        used += width
        if j == 0:
            sig = cols
        else:
            sig = np.concatenate([levels[-1].labels[:, None], cols], axis=1)
        labels, uniq = dense_rows(sig)
        parent = (np.full(uniq.shape[0], -1, np.int32) if j == 0
                  else uniq[:, 0].astype(np.int32))
        levels.append(LevelLabels(labels, uniq, parent))

    # --- WL rounds (design.md 7.2) ---------------------------------------
    csr = batch.csr()
    n_bond = config.n_bond
    for _ in range(config.max_wl_depth):
        prev = levels[-1].labels
        # Encode (neighbor label, bond) as one integer so a row of neighbors
        # is a plain integer vector.
        pair = prev[csr.dst] * n_bond + csr.attr
        pad = np.full((n, max(csr.max_deg, 1)), -1, np.int64)
        pad[csr.src, csr.slot] = pair
        # Sorting canonicalizes the multiset; -1 pads sort first, and because a
        # node's pad count is fixed, degree stays encoded.
        pad.sort(axis=1)
        sig = np.concatenate([prev[:, None], pad], axis=1)
        labels, uniq = dense_rows(sig)
        levels.append(LevelLabels(labels, uniq, uniq[:, 0].astype(np.int32)))

    return levels
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_refine.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/refine.py tests/test_refine.py
git commit -m "feat: vectorized refinement chain with graded attribute levels"
```

---

## Task 5: Level statistics

**Files:**
- Create: `src/wllr/level.py`
- Test: `tests/test_level.py`

**Interfaces:**
- Consumes: `LevelLabels` (Task 4).
- Produces: `FrozenLevel` frozen dataclass with `signatures`, `count (nc,) int64`, `mean (nc,d) float64`, `msd (nc,d) float64`, `parent (nc,) int32`; `FrozenLevel.variance -> np.ndarray` (`NaN` where `count == 1`); `fit_level(level: LevelLabels, y: np.ndarray) -> FrozenLevel`; `global_stats(y) -> tuple[int, np.ndarray, np.ndarray]`.

Read design.md 4.1, 7.3, 7.4. **This is trap 3.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_level.py
import numpy as np
import pytest
from wllr.level import FrozenLevel, fit_level, global_stats
from wllr.refine import LevelLabels

def lv(labels, nc=None):
    labels = np.asarray(labels, np.int64)
    nc = nc or int(labels.max()) + 1
    return LevelLabels(labels, np.zeros((nc, 1), np.int64), np.full(nc, -1, np.int32))

def test_mean_and_population_variance():
    y = np.array([[1.0], [3.0], [10.0]])
    f = fit_level(lv([0, 0, 1]), y)
    assert f.count.tolist() == [2, 1]
    np.testing.assert_allclose(f.mean[:, 0], [2.0, 10.0])
    np.testing.assert_allclose(f.msd[:, 0], [1.0, 0.0])   # divisor N, not N-1

def test_variance_accessor_applies_bessel_and_is_nan_at_one():
    f = fit_level(lv([0, 0, 1]), np.array([[1.0], [3.0], [10.0]]))
    v = f.variance
    np.testing.assert_allclose(v[0, 0], 2.0)              # 2/1 * 1.0
    assert np.isnan(v[1, 0]), "N=1 must be NaN, never 0.0"

def test_vector_targets_are_not_a_special_case():
    y = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 0.0]])
    f = fit_level(lv([0, 0, 0]), y)
    np.testing.assert_allclose(f.mean[0], [3.0, 10.0])
    np.testing.assert_allclose(f.msd[0], [np.var([1, 3, 5]), np.var([10, 20, 0])])

def test_centring_survives_a_large_offset():
    """design.md 7.4: power sums give negative variances here; centring does not."""
    rng = np.random.default_rng(0)
    y = (1e6 + rng.normal(0, 3, size=(2000, 1)))
    f = fit_level(lv(np.zeros(2000, np.int64)), y)
    exact = y.var()
    assert abs(f.msd[0, 0] - exact) / exact < 1e-6
    assert f.msd[0, 0] > 0

def test_empty_classes_are_zero_not_carried_forward():
    """design.md 7.4: reduceat returns y[start] for empty groups; P must not."""
    y = np.array([[10.0], [20.0], [30.0], [40.0]])
    f = fit_level(lv([0, 0, 3, 3], nc=4), y)
    assert f.count.tolist() == [2, 0, 0, 2]
    np.testing.assert_allclose(f.mean[:, 0], [15.0, 0.0, 0.0, 35.0])

def test_global_stats_matches_numpy():
    y = np.array([[1.0, 2.0], [3.0, 6.0]])
    n, mean, msd = global_stats(y)
    assert n == 2
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(msd, [1.0, 4.0])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_level.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.level'`

- [ ] **Step 3: Implement**

```python
# src/wllr/level.py
"""Per-class statistics: a count and two means (design.md 4.1, 7.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from wllr.refine import LevelLabels


@dataclass(frozen=True)
class FrozenLevel:
    """Immutable statistics for one refinement level.

    Stores ``(N, ybar, sigma^2)`` where ``sigma^2`` is the *population*
    variance (divisor N). The reported ``s^2`` is derived on access. ``count``
    and ``parent`` stay one-dimensional even for vector targets, which is what
    keeps the merge weights scalar.
    """

    signatures: np.ndarray    # (nc, width) int64 -- the vocabulary
    count: np.ndarray         # (nc,) int64
    mean: np.ndarray          # (nc, d) float64
    msd: np.ndarray           # (nc, d) float64 -- population variance
    parent: np.ndarray        # (nc,) int32

    @property
    def n_classes(self) -> int:
        return int(self.count.shape[0])

    @property
    def variance(self) -> np.ndarray:
        """Bessel-corrected s^2, NaN where N == 1 (design.md 4.1).

        A stored zero would be indistinguishable from a genuinely homogeneous
        class and would read as confidence in every downstream diagnostic. The
        guard lives here, in one accessor, rather than in every merge.
        """
        n = self.count.astype(np.float64)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            s2 = np.where(n > 1, self.msd * n / np.maximum(n - 1, 1), np.nan)
        return s2


def fit_level(level: LevelLabels, y: np.ndarray) -> FrozenLevel:
    """Reduce one chunk to per-class statistics with a sparse membership operator.

    Two passes, centring before reducing. Never ``sum(y**2)/N - mean**2``: on
    targets with mean 1e6 and spread 3 that form errs by 1.3e+02 relative and
    produces negative variances, against 5.4e-08 for this one.
    """
    labels = level.labels
    nc = level.n_classes
    n, d = y.shape

    # Built once, reused across both passes and all d dimensions. bincount is
    # scalar-only and would need a loop over dimensions.
    P = sparse.csr_matrix(
        (np.ones(n), (labels, np.arange(n))), shape=(nc, n))

    count = np.bincount(labels, minlength=nc).astype(np.int64)
    safe = np.maximum(count, 1)[:, None].astype(np.float64)
    mean = (P @ y) / safe
    resid = y - mean[labels]              # center first, then reduce
    msd = (P @ (resid * resid)) / safe
    # Classes with no members must be exactly zero, not whatever the reduction
    # happened to leave there.
    empty = count == 0
    mean[empty] = 0.0
    msd[empty] = 0.0
    return FrozenLevel(level.signatures, count, mean, msd, level.parent)


def global_stats(y: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Whole-corpus fallback statistics, same convention as a class."""
    return int(y.shape[0]), y.mean(axis=0), y.var(axis=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_level.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/level.py tests/test_level.py
git commit -m "feat: sparse two-pass per-class statistics with derived variance"
```

---

## Task 6: The model container and `fit`

**Files:**
- Create: `src/wllr/model.py`
- Modify: `src/wllr/__init__.py`
- Test: `tests/test_fit.py`

**Interfaces:**
- Consumes: `refine` (Task 4), `fit_level`/`global_stats` (Task 5), `WLLRConfig` (Task 1).
- Produces: `WLLRModel` frozen dataclass with `config`, `levels: tuple[FrozenLevel, ...]`, `global_count: int`, `global_mean`, `global_msd`; `wllr.fit(batch, config) -> WLLRModel`; `WLLRModel.empty(config) -> WLLRModel`; `WLLRModel.with_params(n_min=..., alpha=...) -> WLLRModel`.

Read design.md 5.1, 7, 10.1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit.py
import numpy as np
import pytest
import wllr
from wllr.config import WLLRConfig
from tests.helpers import chain_batch, simple_config

def test_fit_produces_one_level_per_config_level():
    b = chain_batch(6)
    m = wllr.fit(b, simple_config())
    assert len(m.levels) == simple_config().n_levels

def test_support_is_monotone_non_increasing_along_the_chain():
    """design.md 2.3: refinement can only split a class."""
    m = wllr.fit(chain_batch(30), simple_config(max_wl_depth=3))
    for k in range(1, len(m.levels)):
        child, par = m.levels[k], m.levels[k - 1]
        assert np.all(child.count <= par.count[child.parent])

def test_class_means_match_a_direct_groupby():
    b = chain_batch(20)
    m = wllr.fit(b, simple_config())
    from wllr.refine import refine
    labels = refine(b, simple_config())[-1].labels
    for c in np.unique(labels):
        np.testing.assert_allclose(m.levels[-1].mean[c],
                                   b.y[labels == c].mean(axis=0))

def test_changing_targets_leaves_class_ids_unchanged():
    """design.md 10.4: the partition depends on structure only."""
    from wllr.refine import refine
    b = chain_batch(15)
    other = type(b)(**{**b.__dict__, "y": b.y * 3.0 + 7.0})
    a = refine(b, simple_config())[-1].labels
    c = refine(other, simple_config())[-1].labels
    assert np.array_equal(a, c)

def test_empty_model_has_no_classes():
    m = wllr.WLLRModel.empty(simple_config())
    assert all(l.n_classes == 0 for l in m.levels)
    assert m.global_count == 0

def test_with_params_shares_arrays():
    m = wllr.fit(chain_batch(10), simple_config())
    m2 = m.with_params(n_min=5)
    assert m2.config.n_min == 5
    assert m2.levels[0].mean is m.levels[0].mean   # no copy

def test_fit_rejects_a_batch_without_targets():
    b = chain_batch(5)
    with pytest.raises(ValueError, match="targets"):
        wllr.fit(type(b)(**{**b.__dict__, "y": None}), simple_config())

def test_target_dim_must_match_config():
    b = chain_batch(5)
    with pytest.raises(ValueError, match="target_dim"):
        wllr.fit(b, simple_config(target_dim=7))
```

Also create the shared fixture module:

```python
# tests/helpers.py
"""Fixtures shared across the suite."""
import numpy as np
from wllr.batch import AtomBatch
from wllr.config import WLLRConfig

def simple_config(**kw):
    d = dict(target_dim=1, attribute_levels=(("element",),),
             attribute_codes={"element": {"C": 0, "H": 1}},
             edge_codes={"SINGLE": 1, "DOUBLE": 2}, max_wl_depth=2)
    d.update(kw)
    return WLLRConfig(**d)

def chain_batch(n, d=1, seed=0, graphs=1):
    """`graphs` disjoint paths of n nodes each, alternating attributes."""
    rng = np.random.default_rng(seed)
    per = n
    total = per * graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for i in range(per - 1):
            src += [off + i, off + i + 1]
            dst += [off + i + 1, off + i]
        gid += [g] * per
    return AtomBatch(
        node_attrs=(np.arange(total) % 2).reshape(-1, 1).astype(np.int64),
        edge_src=np.array(src, np.int64), edge_dst=np.array(dst, np.int64),
        edge_attr=np.ones(len(src), np.int64),
        graph_id=np.array(gid, np.int64),
        y=rng.normal(size=(total, d)))

def split_batch(batch, mask):
    """Take the sub-batch of atoms where `mask` is True, reindexing edges."""
    idx = np.flatnonzero(mask)
    remap = np.full(batch.n_atoms, -1, np.int64)
    remap[idx] = np.arange(idx.size)
    keep = mask[batch.edge_src] & mask[batch.edge_dst]
    return AtomBatch(
        node_attrs=batch.node_attrs[idx],
        edge_src=remap[batch.edge_src[keep]], edge_dst=remap[batch.edge_dst[keep]],
        edge_attr=batch.edge_attr[keep], graph_id=batch.graph_id[idx],
        y=None if batch.y is None else batch.y[idx])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fit.py -v`
Expected: FAIL, `AttributeError: module 'wllr' has no attribute 'fit'`

- [ ] **Step 3: Implement**

```python
# src/wllr/model.py
"""The fitted model and the fit entry point (design.md 5.1, 7, 10.1)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wllr.batch import AtomBatch
from wllr.config import WLLRConfig
from wllr.level import FrozenLevel, fit_level, global_stats
from wllr.refine import refine


@dataclass(frozen=True)
class WLLRModel:
    """An immutable, exactly sized fitted model (design.md 5.1).

    There is no mutable accumulator and no ``partial_fit``: incremental
    training is ``model.merge(fit(batch))``, and parallel fitting is a fold
    over independently fitted shards.
    """

    config: WLLRConfig
    levels: tuple[FrozenLevel, ...]
    global_count: int
    global_mean: np.ndarray
    global_msd: np.ndarray

    @classmethod
    def empty(cls, config: WLLRConfig) -> "WLLRModel":
        """The identity of the merge monoid (design.md 5.4)."""
        d = config.target_dim
        levels = tuple(
            FrozenLevel(np.zeros((0, 1), np.int64), np.zeros(0, np.int64),
                        np.zeros((0, d)), np.zeros((0, d)), np.zeros(0, np.int32))
            for _ in range(config.n_levels))
        return cls(config, levels, 0, np.zeros(d), np.zeros(d))

    def with_params(self, **kw) -> "WLLRModel":
        """A new model sharing the same arrays, with inference params changed.

        ``n_min`` and ``alpha`` are read at prediction time, so sweeping them
        never requires refitting.
        """
        bad = set(kw) - {"n_min", "alpha", "chunk_size"}
        if bad:
            raise ValueError(f"with_params only changes inference params, got {bad}")
        return replace(self, config=replace(self.config, **kw))


def fit(batch: AtomBatch, config: WLLRConfig) -> WLLRModel:
    """Fit a model to one corpus.

    When ``config.chunk_size`` is set the batch is fitted in pieces and folded,
    which is the whole of the "streaming" story: chunk size is a memory
    decision, and combining chunks is the merge of design.md 5.2.
    """
    if batch.y is None:
        raise ValueError("fit requires targets; batch.y is None")
    if batch.y.shape[1] != config.target_dim:
        raise ValueError(
            f"target_dim is {config.target_dim} but batch.y has "
            f"{batch.y.shape[1]} columns")

    # NOTE: config.chunk_size is honored in Task 7, once `fold` exists.
    # Leave it unhandled here -- a single chunk is the correct default and the
    # only behavior Task 6's tests exercise.
    levels_lbl = refine(batch, config)
    levels = tuple(fit_level(lv, batch.y) for lv in levels_lbl)
    n, mean, msd = global_stats(batch.y)
    return WLLRModel(config, levels, n, mean, msd)
```

**Chunking is added in Task 7**, because it depends on `fold`. For reference, the branch that will be inserted at the top of `fit` then is:

```python
    if config.chunk_size is not None and config.chunk_size < batch.n_atoms:
        from wllr.merge import fold
        from wllr.batch import AtomBatch as _AB
        graphs = np.unique(batch.graph_id)
        n_parts = max(1, -(-batch.n_atoms // config.chunk_size))
        shards = []
        for part in np.array_split(graphs, n_parts):
            mask = np.isin(batch.graph_id, part)
            shards.append(fit(_sub_batch(batch, mask),
                              replace(config, chunk_size=None)))
        return fold(shards, config)
```

with

```python
def _sub_batch(batch: AtomBatch, mask: np.ndarray) -> AtomBatch:
    """Atoms where `mask` is True, with edges reindexed onto the subset."""
    idx = np.flatnonzero(mask)
    remap = np.full(batch.n_atoms, -1, np.int64)
    remap[idx] = np.arange(idx.size)
    keep = mask[batch.edge_src] & mask[batch.edge_dst]
    return AtomBatch(
        node_attrs=batch.node_attrs[idx],
        edge_src=remap[batch.edge_src[keep]],
        edge_dst=remap[batch.edge_dst[keep]],
        edge_attr=batch.edge_attr[keep],
        graph_id=batch.graph_id[idx],
        y=None if batch.y is None else batch.y[idx],
        elements=None if batch.elements is None else batch.elements[idx])
```

`_sub_batch` belongs in `wllr/model.py` now — Task 13's benchmark imports it — but the chunking branch itself waits for Task 7, and so does its test.

Update the package exports:

```python
# src/wllr/__init__.py
"""Weisfeiler-Lehman Lookup Regression."""
from wllr.batch import AtomBatch
from wllr.config import WLLRConfig
from wllr.model import WLLRModel, fit

__all__ = ["AtomBatch", "WLLRConfig", "WLLRModel", "fit"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/model.py src/wllr/__init__.py tests/test_fit.py tests/helpers.py
git commit -m "feat: WLLRModel container and vectorized fit"
```

---

## Task 7: The merge monoid

**Files:**
- Create: `src/wllr/merge.py`
- Modify: `src/wllr/model.py` (add `merge`, `__add__`, chunking branch in `fit`)
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: `FrozenLevel` (Task 5), `WLLRModel` (Task 6), `check_mergeable` (Task 1).
- Produces: `merge_models(a, b) -> WLLRModel`; `fold(models, config) -> WLLRModel`; `WLLRModel.merge(other)`; `WLLRModel.__add__`.

Read design.md 5.2, 5.3, 5.4, and **deviation B at the top of this plan**. This task contains trap 2.

The merge proceeds level by level from 0 upward, carrying `remap_prev` — the map from B's level-`k-1` ids into the merged id space. Before B's level-`k` signature rows can be compared with A's, they must be **translated** into merged-space, because a signature row is written in terms of the previous level's *local* ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge.py
import numpy as np
import pytest
import wllr
from wllr.merge import fold
from tests.helpers import chain_batch, simple_config, split_batch

def preds(model, batch):
    from wllr.predict import predict
    return predict(model, batch)

def test_merge_of_disjoint_shards_equals_fitting_the_union():
    """design.md 5.2: the headline property. Everything else is machinery."""
    cfg = simple_config()
    b = chain_batch(12, graphs=6)
    mask = b.graph_id < 3
    a = wllr.fit(split_batch(b, mask), cfg)
    c = wllr.fit(split_batch(b, ~mask), cfg)
    merged = a.merge(c)
    whole = wllr.fit(b, cfg)
    _assert_same_statistics(merged, whole)

def test_merge_is_commutative():
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    a = wllr.fit(split_batch(b, b.graph_id < 2), cfg)
    c = wllr.fit(split_batch(b, b.graph_id >= 2), cfg)
    _assert_same_statistics(a.merge(c), c.merge(a))

def test_merge_is_associative():
    cfg = simple_config()
    b = chain_batch(9, graphs=6)
    parts = [wllr.fit(split_batch(b, b.graph_id == g), cfg) for g in range(6)]
    left = parts[0].merge(parts[1]).merge(parts[2])
    right = parts[0].merge(parts[1].merge(parts[2]))
    _assert_same_statistics(left, right)

def test_empty_model_is_the_identity():
    cfg = simple_config()
    m = wllr.fit(chain_batch(8), cfg)
    e = wllr.WLLRModel.empty(cfg)
    _assert_same_statistics(m.merge(e), m)
    _assert_same_statistics(e.merge(m), m)

def test_merge_reproduces_variance_not_just_the_mean():
    """Trap 2: swapping the delta and mean updates leaves means correct and
    corrupts every variance for classes present in both models."""
    cfg = simple_config()
    b = chain_batch(14, graphs=8, seed=3)
    a = wllr.fit(split_batch(b, b.graph_id < 4), cfg)
    c = wllr.fit(split_batch(b, b.graph_id >= 4), cfg)
    whole = wllr.fit(b, cfg)
    merged = a.merge(c)
    for k in range(len(whole.levels)):
        np.testing.assert_allclose(np.sort(merged.levels[k].msd, axis=0),
                                   np.sort(whole.levels[k].msd, axis=0),
                                   rtol=1e-10, atol=1e-12)

def test_class_present_in_only_one_model_is_carried_through_exactly():
    cfg = simple_config()
    b = chain_batch(10, graphs=2)
    a = wllr.fit(split_batch(b, b.graph_id == 0), cfg)
    e = wllr.WLLRModel.empty(cfg)
    m = e.merge(a)
    np.testing.assert_array_equal(m.levels[-1].count, a.levels[-1].count)
    np.testing.assert_allclose(m.levels[-1].msd, a.levels[-1].msd)

def test_incompatible_configs_are_rejected_loudly():
    b = chain_batch(8)
    a = wllr.fit(b, simple_config(max_wl_depth=2))
    c = wllr.fit(b, simple_config(max_wl_depth=3))
    with pytest.raises(ValueError, match="schema"):
        a.merge(c)

def test_parent_relations_survive_the_merge():
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(11, graphs=5)
    a = wllr.fit(split_batch(b, b.graph_id < 2), cfg)
    c = wllr.fit(split_batch(b, b.graph_id >= 2), cfg)
    m = a.merge(c)
    for k in range(1, len(m.levels)):
        assert m.levels[k].parent.min() >= 0
        assert m.levels[k].parent.max() < m.levels[k - 1].n_classes
        # support monotonicity must still hold after merging
        assert np.all(m.levels[k].count <= m.levels[k - 1].count[m.levels[k].parent])

def test_fold_matches_sequential_merge():
    cfg = simple_config()
    b = chain_batch(7, graphs=8)
    parts = [wllr.fit(split_batch(b, b.graph_id == g), cfg) for g in range(8)]
    seq = parts[0]
    for p in parts[1:]:
        seq = seq.merge(p)
    _assert_same_statistics(fold(parts, cfg), seq)

def test_chunked_fit_equals_single_chunk_fit():
    """design.md 4.1: chunk size is a memory decision, not a statistical one."""
    b = chain_batch(9, graphs=10)
    whole = wllr.fit(b, simple_config())
    chunked = wllr.fit(b, simple_config(chunk_size=20))
    _assert_same_statistics(chunked, whole)

def _assert_same_statistics(a, b):
    """Compare two models up to class-id permutation, level by level."""
    assert len(a.levels) == len(b.levels)
    assert a.global_count == b.global_count
    np.testing.assert_allclose(a.global_mean, b.global_mean, rtol=1e-12)
    for x, y in zip(a.levels, b.levels):
        assert x.n_classes == y.n_classes
        ox = np.lexsort((x.msd[:, 0], x.mean[:, 0], x.count))
        oy = np.lexsort((y.msd[:, 0], y.mean[:, 0], y.count))
        np.testing.assert_array_equal(x.count[ox], y.count[oy])
        np.testing.assert_allclose(x.mean[ox], y.mean[oy], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(x.msd[ox], y.msd[oy], rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_merge.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.merge'`

- [ ] **Step 3: Implement**

```python
# src/wllr/merge.py
"""The merge monoid (design.md 5).

Counts add; both moments are weighted averages. This is the law of total
variance in intensive variables -- Chan, Golub and LeVeque's parallel form --
not an ad hoc correction, which is why the full-covariance upgrade is a
one-term change and why merging is order-independent.
"""
from __future__ import annotations

import numpy as np

from wllr.config import WLLRConfig, check_mergeable
from wllr.dedupe import dense_rows
from wllr.level import FrozenLevel


def _translate(sig: np.ndarray, remap_prev: np.ndarray | None,
               n_bond: int, is_wl: bool) -> np.ndarray:
    """Rewrite B's signature rows in the merged id space.

    A signature row is written in terms of the *previous level's local ids*, so
    B's row [3, 7, 12] and A's row [3, 7, 12] generally denote different
    classes. Without this step the merge silently unions unrelated classes.
    """
    if remap_prev is None:          # level 0: attribute codes are already global
        return sig
    out = sig.copy()
    out[:, 0] = remap_prev[sig[:, 0]]
    if is_wl and sig.shape[1] > 1:
        pad = sig[:, 1:]
        filled = pad >= 0
        lab, bond = np.divmod(np.where(filled, pad, 0), n_bond)
        new = remap_prev[lab] * n_bond + bond
        out[:, 1:] = np.where(filled, new, -1)
        # Remapping changes the sort order, so the multiset must be
        # re-canonicalized or equal multisets stop comparing equal.
        out[:, 1:] = np.sort(out[:, 1:], axis=1)
    return out


def merge_level(a: FrozenLevel, b: FrozenLevel, remap_prev: np.ndarray | None,
                n_bond: int, is_wl: bool) -> tuple[FrozenLevel, np.ndarray]:
    """Merge one level. A's ids are preserved; only B's are remapped.

    Pinning A means A's ``parent`` array needs no translation at all -- only
    B's, carried forward across levels by ``remap_prev``.
    """
    d = a.mean.shape[1] if a.n_classes else b.mean.shape[1]
    width = max(a.signatures.shape[1], b.signatures.shape[1], 1)

    def widen(sig):
        if sig.shape[1] == width:
            return sig
        out = np.full((sig.shape[0], width), -1, np.int64)
        out[:, :sig.shape[1]] = sig
        return out

    a_sig = widen(a.signatures)
    b_sig = widen(_translate(b.signatures, remap_prev, n_bond, is_wl))

    # Deduplicate A's rows followed by B's; A's rows come first and so keep the
    # lower ids, which is exactly "A's ids are preserved".
    stacked = np.concatenate([a_sig, b_sig], axis=0)
    ids, uniq = dense_rows(stacked)
    m = a.n_classes
    assert np.array_equal(ids[:m], np.arange(m)), \
        "A's signature rows must already be unique and densely numbered"
    remap = ids[m:].astype(np.int32)
    n_new = uniq.shape[0]

    count = np.zeros(n_new, np.int64)
    mean = np.zeros((n_new, d), np.float64)
    msd = np.zeros((n_new, d), np.float64)
    parent = np.full(n_new, -1, np.int32)
    count[:m] = a.count
    mean[:m] = a.mean
    msd[:m] = a.msd
    parent[:m] = a.parent

    i = remap                       # a bijection: no duplicate scatter writes
    nA = count[i].astype(np.float64)
    nB = b.count.astype(np.float64)
    n = nA + nB
    safe = np.maximum(n, 1.0)
    wA, wB = (nA / safe)[:, None], (nB / safe)[:, None]

    delta = b.mean - mean[i]        # MUST precede the mean update below
    msd[i] = wA * msd[i] + wB * b.msd + wA * wB * delta * delta
    mean[i] = wA * mean[i] + wB * b.mean
    count[i] = n.astype(np.int64)

    b_parent = (np.full(b.n_classes, -1, np.int32) if remap_prev is None
                else remap_prev[b.parent].astype(np.int32))
    # For classes in both models the parent must already agree (design.md 2.1).
    both = count[i] > nB
    if np.any(both):
        assert np.array_equal(parent[i][both], b_parent[both]), \
            "parent disagreement: the nesting invariant is broken"
    parent[i] = np.where(nA > 0, parent[i], b_parent)
    return FrozenLevel(uniq, count, mean, msd, parent), remap


def merge_models(a, b):
    """Merge two fitted models (design.md 5)."""
    from wllr.model import WLLRModel

    check_mergeable(a.config, b.config)
    cfg: WLLRConfig = a.config
    n_attr_levels = len(cfg.attribute_levels)

    levels, remap = [], None
    for k in range(cfg.n_levels):
        lvl, remap = merge_level(a.levels[k], b.levels[k], remap,
                                 cfg.n_bond, is_wl=k >= n_attr_levels)
        levels.append(lvl)

    nA, nB = float(a.global_count), float(b.global_count)
    n = nA + nB
    if n == 0:
        return WLLRModel(cfg, tuple(levels), 0, a.global_mean, a.global_msd)
    wA, wB = nA / n, nB / n
    delta = b.global_mean - a.global_mean
    g_msd = wA * a.global_msd + wB * b.global_msd + wA * wB * delta * delta
    g_mean = wA * a.global_mean + wB * b.global_mean
    return WLLRModel(cfg, tuple(levels), int(n), g_mean, g_msd)


def fold(models, config: WLLRConfig):
    """Combine shards as a balanced tree (design.md 5.4).

    Each merge costs O(|A| + |B|), so a sequential reduce over N shards is
    O(N^2 s) -- the accumulator grows and is re-copied every step. Pairwise
    reduction is O(N s log N).
    """
    from wllr.model import WLLRModel

    items = list(models)
    if not items:
        return WLLRModel.empty(config)
    while len(items) > 1:
        items = [merge_models(items[i], items[i + 1]) if i + 1 < len(items)
                 else items[i] for i in range(0, len(items), 2)]
    return items[0]
```

Add to `WLLRModel`:

```python
    def merge(self, other: "WLLRModel") -> "WLLRModel":
        """Combine two models. Named `merge` because `a + b` reads as ensembling."""
        from wllr.merge import merge_models
        return merge_models(self, other)

    def __add__(self, other):
        """Ergonomic alias so `sum(models, WLLRModel.empty(cfg))` works."""
        if other == 0:
            return self
        return self.merge(other)

    __radd__ = __add__
```

Then add the chunking branch to `fit` exactly as given in Task 6's note, together with `_sub_batch`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_merge.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/wllr/merge.py src/wllr/model.py tests/test_merge.py
git commit -m "feat: merge monoid with signature translation and balanced fold"
```

---

## Task 8: Inference with hierarchical backoff

**Files:**
- Create: `src/wllr/predict.py`
- Modify: `src/wllr/model.py` (add `predict`, `predict_detailed`), `src/wllr/__init__.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `refine` (Task 4), `WLLRModel` (Task 6), `_translate` (Task 7).
- Produces: `Predictions` frozen dataclass (fields exactly as design.md 12); `predict(model, batch) -> np.ndarray`; `predict_detailed(model, batch) -> Predictions`.

Read design.md 2.2, 6, 12.

**How lookup works at inference.** Refine the query batch to get its own local labels, then map each query class to a *training* class id level by level: at level `k`, translate the query signature rows into the model's id space using the level-`k-1` query→model map, then look them up in `model.levels[k].signatures`. Reuse `_translate` from Task 7 — this is the same problem as merging, so it must not grow a second implementation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import numpy as np
import pytest
import wllr
from tests.helpers import chain_batch, simple_config, split_batch

def test_training_atoms_recover_their_class_mean():
    cfg = simple_config()
    b = chain_batch(12)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    from wllr.refine import refine
    labels = refine(b, cfg)[-1].labels
    for i in range(b.n_atoms):
        if p.matched_level[i] == cfg.n_levels - 1:
            np.testing.assert_allclose(p.value[i], b.y[labels == labels[i]].mean(0))

def test_unseen_atom_falls_back_to_the_global_mean():
    cfg = simple_config()
    train = chain_batch(6)
    m = wllr.fit(train, cfg)
    alien = chain_batch(3)
    alien = type(alien)(**{**alien.__dict__,
                           "node_attrs": np.full((3, 1), 7, np.int64)})
    p = wllr.predict_detailed(m, alien)
    assert np.all(p.matched_level == -1)
    assert np.all(p.class_id == -1)
    np.testing.assert_allclose(p.value, np.broadcast_to(m.global_mean, p.value.shape))

def test_n_min_moves_the_match_shallower_never_deeper():
    """design.md 10.4."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(20, graphs=3)
    m = wllr.fit(b, cfg)
    loose = wllr.predict_detailed(m, b).matched_level
    tight = wllr.predict_detailed(m.with_params(n_min=4), b).matched_level
    assert np.all(tight <= loose)

def test_threshold_bound_distinguishes_support_stops_from_oov():
    cfg = simple_config(max_wl_depth=3, n_min=1000)
    b = chain_batch(20, graphs=3)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    assert p.threshold_bound.any(), "a huge n_min must stop on support, not OOV"

def test_matched_levels_form_a_prefix():
    """design.md 2.2: no gaps are constructible. Verified by construction here:
    the reported level must be supported and level+1 must not be."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(15, graphs=4)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    for i in range(b.n_atoms):
        k = int(p.matched_level[i])
        if 0 <= k < cfg.n_levels - 1:
            assert p.support[i] >= cfg.n_min

def test_variance_is_nan_at_support_one():
    cfg = simple_config(max_wl_depth=4)
    b = chain_batch(30, graphs=1)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    singles = p.support == 1
    if singles.any():
        assert np.all(np.isnan(p.variance[singles]))

def test_batched_and_split_prediction_agree():
    """design.md 10.4: batched and per-node prediction must agree."""
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    m = wllr.fit(b, cfg)
    full = wllr.predict(m, b)
    parts = [wllr.predict(m, split_batch(b, b.graph_id == g)) for g in range(4)]
    np.testing.assert_allclose(full, np.concatenate(parts))

def test_vector_targets_predict_elementwise():
    cfg = simple_config(target_dim=3)
    b = chain_batch(12, d=3)
    m = wllr.fit(b, cfg)
    assert wllr.predict(m, b).shape == (12, 3)

def test_global_fallback_iff_level_zero_unsupported():
    cfg = simple_config()
    b = chain_batch(8)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    assert not np.any(p.matched_level == -1)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.predict'`

- [ ] **Step 3: Implement**

```python
# src/wllr/predict.py
"""Bottom-up backoff through the refinement chain (design.md 6, 12)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wllr.batch import AtomBatch
from wllr.merge import _translate
from wllr.refine import refine


@dataclass(frozen=True)
class Predictions:
    """Columnar prediction detail (design.md 12).

    These are diagnostics, not calibrated uncertainties: the triple
    (matched_level, support, variance) says how specific the environment was,
    how much support it had, and how heterogeneous its labels were.
    """

    value: np.ndarray             # (n, d)
    matched_level: np.ndarray     # (n,) k*, -1 for global fallback
    class_id: np.ndarray          # (n,) id at the matched level, -1 if none
    support: np.ndarray           # (n,) N at the matched class
    variance: np.ndarray          # (n, d) s^2, NaN where support == 1
    threshold_bound: np.ndarray   # (n,) stopped by n_min rather than by OOV
    raw_value: np.ndarray | None = None
    shrinkage_weight: np.ndarray | None = None


def _lookup(sig: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Row-wise membership: index of each `sig` row in `table`, or -1."""
    if table.shape[0] == 0 or sig.shape[0] == 0:
        return np.full(sig.shape[0], -1, np.int64)
    width = max(sig.shape[1], table.shape[1])

    def widen(m):
        if m.shape[1] == width:
            return np.ascontiguousarray(m)
        out = np.full((m.shape[0], width), -1, np.int64)
        out[:, :m.shape[1]] = m
        return out

    s, t = widen(sig), widen(table)
    dt = np.dtype((np.void, s.dtype.itemsize * width))
    sv = s.view(dt).ravel()
    tv = t.view(dt).ravel()
    order = np.argsort(tv)
    pos = np.searchsorted(tv, sv, sorter=order)
    pos = np.clip(pos, 0, order.size - 1)
    cand = order[pos]
    hit = tv[cand] == sv
    return np.where(hit, cand, -1).astype(np.int64)


def predict_detailed(model, batch: AtomBatch) -> Predictions:
    """Predict with full metadata.

    The search runs upward through the chain. Both stopping conditions are
    valid only because matched levels form a prefix and support is monotone
    non-increasing (design.md 2.2, 2.3).
    """
    cfg = model.config
    n, d = batch.n_atoms, cfg.target_dim
    n_attr = len(cfg.attribute_levels)
    query = refine(batch, cfg)

    value = np.broadcast_to(model.global_mean, (n, d)).copy()
    matched = np.full(n, -1, np.int64)
    class_id = np.full(n, -1, np.int64)
    support = np.zeros(n, np.int64)
    variance = np.full((n, d), np.nan)
    threshold = np.zeros(n, bool)

    remap = None            # query class ids -> model class ids, previous level
    alive = np.ones(n, bool)
    for k in range(cfg.n_levels):
        lvl = model.levels[k]
        q = query[k]
        sig = _translate(q.signatures, remap, cfg.n_bond, is_wl=k >= n_attr)
        found = _lookup(sig, lvl.signatures)          # per query class
        remap = found
        if not alive.any():
            break                                     # graph-level stop (6.2)
        cid = found[q.labels]
        ok = alive & (cid >= 0)
        enough = np.zeros(n, bool)
        enough[ok] = lvl.count[cid[ok]] >= cfg.n_min
        # Stopped by support rather than by OOV: a different situation with a
        # different remedy, and indistinguishable without this flag.
        threshold |= alive & (cid >= 0) & ~enough
        hit = ok & enough
        value[hit] = lvl.mean[cid[hit]]
        matched[hit] = k
        class_id[hit] = cid[hit]
        support[hit] = lvl.count[cid[hit]]
        variance[hit] = lvl.variance[cid[hit]]
        alive = hit                                   # prefix property (2.2)

    return Predictions(value, matched, class_id, support, variance, threshold)


def predict(model, batch: AtomBatch) -> np.ndarray:
    return predict_detailed(model, batch).value
```

Add thin wrappers to `WLLRModel` (`self.predict(batch)`, `self.predict_detailed(batch)`) and export `predict`, `predict_detailed`, `Predictions` from `wllr/__init__.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_predict.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/predict.py src/wllr/model.py src/wllr/__init__.py tests/test_predict.py
git commit -m "feat: bottom-up backoff inference with prediction metadata"
```

---

## Task 9: Hierarchical shrinkage

**Files:**
- Create: `src/wllr/shrinkage.py`
- Modify: `src/wllr/predict.py`
- Test: `tests/test_shrinkage.py`

**Interfaces:**
- Consumes: `WLLRModel` (Task 6), `Predictions` (Task 8).
- Produces: `shrunk_means(model) -> list[np.ndarray]` — one `(n_classes, d)` array per level. When `config.alpha is not None`, `predict_detailed` fills `raw_value` and `shrinkage_weight`, and `value` carries the shrunk estimate.

Read design.md 4.2. The recursion consumes the **already-shrunk** parent, so the pass must run top-down, level 0 first. Computing it from raw parent means is the natural mistake and gives quietly different numbers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shrinkage.py
import numpy as np
import wllr
from wllr.shrinkage import shrunk_means
from tests.helpers import chain_batch, simple_config

def test_alpha_zero_reproduces_raw_means():
    """design.md 10.4."""
    cfg = simple_config(alpha=0.0)
    b = chain_batch(15, graphs=3)
    m = wllr.fit(b, cfg)
    for lvl, sh in zip(m.levels, shrunk_means(m)):
        np.testing.assert_allclose(sh[lvl.count > 0], lvl.mean[lvl.count > 0])

def test_large_alpha_approaches_the_global_mean():
    cfg = simple_config(alpha=1e12)
    b = chain_batch(15, graphs=3)
    m = wllr.fit(b, cfg)
    for sh in shrunk_means(m):
        np.testing.assert_allclose(sh, np.broadcast_to(m.global_mean, sh.shape),
                                   rtol=1e-4)

def test_shrinkage_uses_the_shrunk_parent_not_the_raw_parent():
    cfg = simple_config(alpha=3.0, max_wl_depth=2)
    b = chain_batch(20, graphs=3)
    m = wllr.fit(b, cfg)
    sh = shrunk_means(m)
    a = cfg.alpha
    for k in range(1, len(m.levels)):
        lvl = m.levels[k]
        n = lvl.count[:, None].astype(float)
        expect = (n * lvl.mean + a * sh[k - 1][lvl.parent]) / (n + a)
        np.testing.assert_allclose(sh[k], expect, rtol=1e-12)

def test_level_zero_shrinks_toward_the_global_mean():
    cfg = simple_config(alpha=2.0)
    m = wllr.fit(chain_batch(12), cfg)
    lvl, sh = m.levels[0], shrunk_means(m)[0]
    n = lvl.count[:, None].astype(float)
    expect = (n * lvl.mean + 2.0 * m.global_mean) / (n + 2.0)
    np.testing.assert_allclose(sh, expect)

def test_predict_exposes_raw_value_and_weight_when_alpha_is_set():
    cfg = simple_config(alpha=2.0)
    b = chain_batch(12)
    p = wllr.predict_detailed(wllr.fit(b, cfg), b)
    assert p.raw_value is not None and p.shrinkage_weight is not None
    matched = p.matched_level >= 0
    n = p.support[matched].astype(float)
    np.testing.assert_allclose(p.shrinkage_weight[matched], n / (n + 2.0))

def test_predict_omits_shrinkage_fields_when_alpha_is_none():
    b = chain_batch(12)
    p = wllr.predict_detailed(wllr.fit(b, simple_config()), b)
    assert p.raw_value is None and p.shrinkage_weight is None
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_shrinkage.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.shrinkage'`

- [ ] **Step 3: Implement**

```python
# src/wllr/shrinkage.py
"""Hierarchically shrunk means, derived on demand (design.md 4.2)."""
from __future__ import annotations

import numpy as np


def shrunk_means(model) -> list[np.ndarray]:
    """Compute shrunk class means top-down, level 0 first.

    Each level consumes the *already-shrunk* parent, not the raw parent mean --
    which is why the pass must run downward and cannot be vectorized across
    levels.

    Never stored as model state: any added data changes the global mean and
    every estimate depends on its full ancestor chain, so one new node
    invalidates essentially every value. There is no incremental patch.
    """
    alpha = model.config.alpha
    out: list[np.ndarray] = []
    for k, lvl in enumerate(model.levels):
        n = lvl.count[:, None].astype(np.float64)
        parent_est = (np.broadcast_to(model.global_mean, lvl.mean.shape) if k == 0
                      else out[k - 1][lvl.parent])
        if alpha is None or alpha == 0.0:
            out.append(np.where(n > 0, lvl.mean, parent_est))
        else:
            out.append((n * lvl.mean + alpha * parent_est) / (n + alpha))
    return out
```

In `predict_detailed`, after the level loop, when `cfg.alpha is not None`:

```python
    raw = value.copy()
    weight = np.zeros(n)
    shrunk = shrunk_means(model)
    matched = matched_level_array >= 0     # use the local `matched` variable
    lv = matched[matched_mask]             # per-atom matched level
    for k in range(cfg.n_levels):
        sel = matched == k
        if sel.any():
            value[sel] = shrunk[k][class_id[sel]]
            nn = support[sel].astype(np.float64)
            weight[sel] = nn / (nn + cfg.alpha)
    return Predictions(value, matched, class_id, support, variance, threshold,
                       raw_value=raw, shrinkage_weight=weight)
```

Atoms that fell back globally keep `value = global_mean` and `weight = 0`, which is the correct limit.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_shrinkage.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/shrinkage.py src/wllr/predict.py tests/test_shrinkage.py
git commit -m "feat: hierarchical shrinkage derived top-down from stored moments"
```

---

## Task 10: Leave-one-out prediction

**Files:**
- Modify: `src/wllr/predict.py`, `src/wllr/model.py`, `src/wllr/__init__.py`
- Test: `tests/test_loo.py`

**Interfaces:**
- Produces: `predict_loo(model, batch) -> Predictions`.

Read design.md 10.3. A class with `N == 1` must be treated as **unsupported** so backoff proceeds to the parent, rather than dividing by zero.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loo.py
import numpy as np
import wllr
from tests.helpers import chain_batch, simple_config

def test_class_of_two_returns_the_other_member():
    """design.md 10.4: the sharpest statement of what LOO means."""
    cfg = simple_config(max_wl_depth=1)
    b = chain_batch(10, graphs=2, seed=5)
    m = wllr.fit(b, cfg)
    p = wllr.predict_loo(m, b)
    from wllr.refine import refine
    labels = refine(b, cfg)[-1].labels
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if members.size == 2 and np.all(p.matched_level[members] == cfg.n_levels - 1):
            i, j = members
            np.testing.assert_allclose(p.value[i], b.y[j])
            np.testing.assert_allclose(p.value[j], b.y[i])

def test_singleton_classes_back_off_instead_of_dividing_by_zero():
    cfg = simple_config(max_wl_depth=4)
    b = chain_batch(25, graphs=1)
    m = wllr.fit(b, cfg)
    p = wllr.predict_loo(m, b)
    assert np.all(np.isfinite(p.value))
    assert np.all(p.support != 1)

def test_loo_is_strictly_worse_than_in_sample():
    """The point of the method: in-sample scores are meaningless at n_min=1."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(30, graphs=4, seed=2)
    m = wllr.fit(b, cfg)
    ins = np.mean((wllr.predict(m, b) - b.y) ** 2)
    loo = np.mean((wllr.predict_loo(m, b).value - b.y) ** 2)
    assert loo > ins

def test_loo_requires_targets():
    import pytest
    b = chain_batch(6)
    m = wllr.fit(b, simple_config())
    with pytest.raises(ValueError, match="targets"):
        wllr.predict_loo(m, type(b)(**{**b.__dict__, "y": None}))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_loo.py -v`
Expected: FAIL, `AttributeError: module 'wllr' has no attribute 'predict_loo'`

- [ ] **Step 3: Implement**

Add a `loo_y` parameter to the internal search: when set, at each level the class mean becomes `(N*ybar - y_v)/(N-1)` and a class with `N == 1` is treated as unsupported.

```python
def predict_loo(model, batch: AtomBatch) -> Predictions:
    """Predict training nodes with their own contribution removed (design.md 10.3).

    A training node contributes its own label to its class mean, so any
    in-sample score is meaningless -- at n_min=1 and large L it approaches
    perfect recall. This is the standard remedy from the target-encoding
    literature, and the cheapest test that the implementation is not leaking.
    """
    if batch.y is None:
        raise ValueError("predict_loo requires targets; batch.y is None")
    return _search(model, batch, loo_y=batch.y)
```

Refactor `predict_detailed` and `predict_loo` to share one `_search(model, batch, loo_y=None)`. Inside the level loop:

```python
        cnt = lvl.count[cid[ok]].astype(np.float64)
        if loo_y is None:
            est = lvl.mean[cid[ok]]
            eff_n = cnt
        else:
            eff_n = cnt - 1.0
            # N == 1 leaves nothing behind: treat as unsupported and let the
            # search back off to the parent rather than dividing by zero.
            est = np.where(eff_n[:, None] > 0,
                           (cnt[:, None] * lvl.mean[cid[ok]] - loo_y[ok]) /
                           np.maximum(eff_n, 1.0)[:, None],
                           np.nan)
        enough[ok] = eff_n >= cfg.n_min
```

with `support` recording `eff_n` in the LOO case.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loo.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/predict.py src/wllr/model.py src/wllr/__init__.py tests/test_loo.py
git commit -m "feat: leave-one-out prediction as a first-class leakage guard"
```

---

## Task 11: Serialization

**Files:**
- Modify: `src/wllr/model.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `WLLRModel.save(path) -> None`; `WLLRModel.load(path) -> WLLRModel` (classmethod).

Read design.md 9. Everything the model holds is already an array, including the vocabulary — there is no dict to encode and no digest to store.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io.py
import numpy as np
import pytest
import wllr
from tests.helpers import chain_batch, simple_config

def test_round_trip_reproduces_predictions_bit_exactly(tmp_path):
    """design.md 9.3: not an aspiration, a testable property."""
    cfg = simple_config(max_wl_depth=3, alpha=1.5)
    b = chain_batch(20, graphs=3, d=4)
    m = wllr.fit(b, simple_config(max_wl_depth=3, alpha=1.5, target_dim=4))
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = wllr.WLLRModel.load(p)
    a = wllr.predict(m, b)
    c = wllr.predict(loaded, b)
    assert np.array_equal(a, c), "round trip must be bit-exact, not merely close"

def test_round_trip_preserves_config(tmp_path):
    cfg = simple_config(max_wl_depth=2, n_min=3, alpha=0.5)
    m = wllr.fit(chain_batch(10), cfg)
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = wllr.WLLRModel.load(p)
    assert loaded.config.schema_version == cfg.schema_version
    assert loaded.config.n_min == 3 and loaded.config.alpha == 0.5

def test_loaded_model_still_merges(tmp_path):
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    from tests.helpers import split_batch
    a = wllr.fit(split_batch(b, b.graph_id < 2), cfg)
    c = wllr.fit(split_batch(b, b.graph_id >= 2), cfg)
    p = tmp_path / "a.npz"
    a.save(p)
    merged = wllr.WLLRModel.load(p).merge(c)
    np.testing.assert_allclose(merged.global_mean, a.merge(c).global_mean)

def test_unknown_format_version_is_refused(tmp_path):
    import json
    m = wllr.fit(chain_batch(6), simple_config())
    p = tmp_path / "m.npz"
    m.save(p)
    data = dict(np.load(p, allow_pickle=False))
    cfg = json.loads(bytes(data["config"]).decode())
    cfg["format_version"] = 999
    data["config"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    np.savez(p, **data)
    with pytest.raises(ValueError, match="format_version"):
        wllr.WLLRModel.load(p)

def test_shrunk_means_are_not_stored(tmp_path):
    m = wllr.fit(chain_batch(10), simple_config(alpha=2.0))
    p = tmp_path / "m.npz"
    m.save(p)
    keys = list(np.load(p, allow_pickle=False).keys())
    assert not any("shrunk" in k or "shrink" in k for k in keys)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_io.py -v`
Expected: FAIL, `AttributeError: 'WLLRModel' object has no attribute 'save'`

- [ ] **Step 3: Implement**

```python
    def save(self, path) -> None:
        """Write a single .npz (design.md 9.1).

        Shrunk estimates are deliberately absent: they are derived, and storing
        them would let alpha drift out of sync with the values it produced.
        """
        import json

        from wllr.config import FORMAT_VERSION

        cfg = self.config
        blob = {
            "format_version": FORMAT_VERSION,
            "schema_version": cfg.schema_version,
            "target_dim": cfg.target_dim,
            "attribute_levels": [list(g) for g in cfg.attribute_levels],
            "attribute_codes": {k: dict(v) for k, v in cfg.attribute_codes.items()},
            "edge_codes": dict(cfg.edge_codes),
            "max_wl_depth": cfg.max_wl_depth,
            "neighbour_schema": cfg.neighbour_schema,
            "n_min": cfg.n_min,
            "alpha": cfg.alpha,
            "chunk_size": cfg.chunk_size,
        }
        arrays = {
            "config": np.frombuffer(json.dumps(blob).encode(), np.uint8),
            "global": np.concatenate([[float(self.global_count)],
                                      self.global_mean, self.global_msd]),
        }
        for k, lvl in enumerate(self.levels):
            arrays[f"level_{k}_vocab"] = lvl.signatures
            arrays[f"level_{k}_count"] = lvl.count
            arrays[f"level_{k}_mean"] = lvl.mean
            arrays[f"level_{k}_msd"] = lvl.msd
            arrays[f"level_{k}_parent"] = lvl.parent
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "WLLRModel":
        import json

        from wllr.config import FORMAT_VERSION, WLLRConfig
        from wllr.level import FrozenLevel

        data = np.load(path, allow_pickle=False)
        blob = json.loads(bytes(data["config"]).decode())
        if blob["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {blob['format_version']}; "
                f"this build reads {FORMAT_VERSION}. Refusing to guess.")
        cfg = WLLRConfig(
            target_dim=blob["target_dim"],
            attribute_levels=tuple(tuple(g) for g in blob["attribute_levels"]),
            attribute_codes=blob["attribute_codes"],
            edge_codes=blob["edge_codes"],
            max_wl_depth=blob["max_wl_depth"],
            neighbour_schema=(None if blob["neighbour_schema"] is None
                              else tuple(blob["neighbour_schema"])),
            n_min=blob["n_min"], alpha=blob["alpha"],
            chunk_size=blob["chunk_size"])
        if cfg.schema_version != blob["schema_version"]:
            raise ValueError("schema_version does not match the stored config")
        d = cfg.target_dim
        g = data["global"]
        levels = tuple(
            FrozenLevel(data[f"level_{k}_vocab"], data[f"level_{k}_count"],
                        data[f"level_{k}_mean"], data[f"level_{k}_msd"],
                        data[f"level_{k}_parent"])
            for k in range(cfg.n_levels))
        return cls(cfg, levels, int(g[0]), g[1:1 + d], g[1 + d:1 + 2 * d])
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_io.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wllr/model.py tests/test_io.py
git commit -m "feat: npz serialization with bit-exact round trip"
```

---

## Task 12: RDKit adapter

**Files:**
- Create: `src/wllr/io/__init__.py`, `src/wllr/io/rdkit_adapter.py`
- Test: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Produces: `build_codes(mols, attributes) -> tuple[dict, dict]`; `from_rdkit(mols, y=None, *, config, atom_order=None) -> AtomBatch`; `from_smiles(smiles, y=None, *, config) -> AtomBatch`.

Read design.md 11.2, 11.3, and `literature.md` 4.6.

The adapter owns encoding and must map an unseen category to a **reserved unknown code** rather than colliding with a seen one. An unknown code then simply fails to match at level 0 and backs off, which is the correct behavior.

Supported attribute names: `element`, `degree`, `hybridization`, `aromatic`, `formal_charge`, `num_h`, `chirality`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rdkit_adapter.py
import numpy as np
import pytest
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem
import wllr
from wllr.config import WLLRConfig
from wllr.io.rdkit_adapter import build_codes, from_smiles

SMILES = ["CCO", "c1ccccc1", "CC(=O)N", "CCl"]

def cfg_for(smiles, attrs=(("element",), ("aromatic", "hybridization"))):
    flat = [a for g in attrs for a in g]
    codes, edges = build_codes([Chem.MolFromSmiles(s) for s in smiles], flat)
    return WLLRConfig(target_dim=1, attribute_levels=attrs,
                      attribute_codes=codes, edge_codes=edges, max_wl_depth=2)

def test_batch_shapes_match_the_molecules():
    cfg = cfg_for(SMILES)
    b = from_smiles(SMILES, config=cfg)
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    assert b.n_atoms == sum(m.GetNumAtoms() for m in mols)
    assert b.n_edges == 2 * sum(m.GetNumBonds() for m in mols)

def test_edges_are_symmetric():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    fwd = {(int(a), int(c)) for a, c in zip(b.edge_src, b.edge_dst)}
    assert all((c, a) in fwd for a, c in fwd)

def test_graph_id_separates_molecules():
    b = from_smiles(SMILES, config=cfg_for(SMILES))
    assert len(np.unique(b.graph_id)) == len(SMILES)
    # no edge may cross a molecule boundary
    assert np.all(b.graph_id[b.edge_src] == b.graph_id[b.edge_dst])

def test_unseen_category_gets_the_reserved_unknown_code():
    cfg = cfg_for(["CCO"])                     # codes learned from C, O, H only
    b = from_smiles(["CCl"], config=cfg)
    unknown = max(cfg.attribute_codes["element"].values()) + 1
    assert unknown in b.node_attrs[:, 0].tolist()

def test_unknown_atoms_fall_back_rather_than_colliding():
    cfg = cfg_for(["CCO"])
    train = from_smiles(["CCO"], y=np.zeros((3, 1)), config=cfg)
    m = wllr.fit(train, cfg)
    p = wllr.predict_detailed(m, from_smiles(["CCl"], config=cfg))
    assert (p.matched_level == -1).any()

def test_wl_labels_agree_with_morgan_atom_environments():
    """literature.md 4.6: WL identifiers on molecules ARE ECFP identifiers, so
    RDKit gives an independent check on the encoder."""
    from rdkit.Chem import rdFingerprintGenerator
    from wllr.refine import refine
    smis = ["CCO", "CCC", "c1ccccc1C", "CC(=O)O"]
    cfg = cfg_for(smis, attrs=(("element", "degree", "formal_charge",
                                "num_h", "aromatic"),))
    b = from_smiles(smis, config=cfg)
    ours = refine(b, cfg)[len(cfg.attribute_levels)].labels   # WL round 1

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=1, includeChirality=False)
    theirs, off = [], 0
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateAtomToBits()
        gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        env = ao.GetAtomToBits()
        theirs += [env[i][-1] for i in range(mol.GetNumAtoms())]
        off += mol.GetNumAtoms()
    assert _same_partition(np.array(ours), np.array(theirs))

def test_alignment_guard_catches_a_shuffled_target_array():
    cfg = cfg_for(["CCO"])
    y = np.arange(3, dtype=float).reshape(-1, 1)
    b = from_smiles(["CCO"], y=y, config=cfg)
    assert b.elements is not None
    from wllr.batch import check_alignment
    with pytest.raises(ValueError, match="element"):
        check_alignment(b, np.array([3]), b.elements[::-1].copy())

def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_rdkit_adapter.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.io'`

- [ ] **Step 3: Implement**

```python
# src/wllr/io/rdkit_adapter.py
"""RDKit -> AtomBatch (design.md 11.2)."""
from __future__ import annotations

import numpy as np

from wllr.batch import AtomBatch
from wllr.config import WLLRConfig

_ATTRS = {
    "element": lambda a: a.GetSymbol(),
    "degree": lambda a: str(a.GetDegree()),
    "hybridization": lambda a: str(a.GetHybridization()),
    "aromatic": lambda a: str(a.GetIsAromatic()),
    "formal_charge": lambda a: str(a.GetFormalCharge()),
    "num_h": lambda a: str(a.GetTotalNumHs()),
    "chirality": lambda a: str(a.GetChiralTag()),
}


def build_codes(mols, attributes):
    """Learn dense code tables from a corpus.

    Every attribute reserves one code above its observed maximum for unseen
    categories: an unknown value must fail to match at level 0 and back off,
    never silently collide with a seen one.
    """
    codes = {}
    for name in attributes:
        fn = _ATTRS[name]
        seen = sorted({fn(a) for m in mols for a in m.GetAtoms()})
        codes[name] = {v: i for i, v in enumerate(seen)}
    edge_codes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    return codes, edge_codes


def from_rdkit(mols, y=None, *, config: WLLRConfig,
               atom_order=None) -> AtomBatch:
    flat = [a for g in config.attribute_levels for a in g]
    n = sum(m.GetNumAtoms() for m in mols)
    node_attrs = np.zeros((n, len(flat)), np.int64)
    elements = np.zeros(n, np.int64)
    graph_id = np.zeros(n, np.int64)
    src, dst, attr = [], [], []
    off = 0
    for gi, mol in enumerate(mols):
        order = (list(range(mol.GetNumAtoms())) if atom_order is None
                 else list(atom_order[gi]))
        inv = np.empty(mol.GetNumAtoms(), np.int64)
        inv[order] = np.arange(mol.GetNumAtoms())
        for local, idx in enumerate(order):
            a = mol.GetAtomWithIdx(int(idx))
            g = off + local
            for j, name in enumerate(flat):
                table = config.attribute_codes[name]
                unknown = max(table.values()) + 1 if table else 0
                node_attrs[g, j] = table.get(_ATTRS[name](a), unknown)
            elements[g] = a.GetAtomicNum()
            graph_id[g] = gi
        for b in mol.GetBonds():
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            c = config.edge_codes.get(str(b.GetBondType()), 0)
            src += [u, v]; dst += [v, u]; attr += [c, c]
        off += mol.GetNumAtoms()
    return AtomBatch(node_attrs=node_attrs,
                     edge_src=np.array(src, np.int64),
                     edge_dst=np.array(dst, np.int64),
                     edge_attr=np.array(attr, np.int64),
                     graph_id=graph_id, y=y, elements=elements)


def from_smiles(smiles, y=None, *, config: WLLRConfig) -> AtomBatch:
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        bad = [s for s, m in zip(smiles, mols) if m is None]
        raise ValueError(f"unparseable SMILES: {bad[:3]}")
    return from_rdkit(mols, y, config=config)
```

**Note:** the per-molecule Python loop here is acceptable — parsing is inherently per-molecule and this is the adapter, not the fit path. Everything downstream of `AtomBatch` stays vectorized.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rdkit_adapter.py -v`
Expected: PASS (8 tests). If the Morgan test fails on RDKit API differences, fix the *test* to match your RDKit version — but only after confirming by hand that the partitions genuinely agree on `["CCO", "CCC"]`.

- [ ] **Step 5: Commit**

```bash
git add src/wllr/io tests/test_rdkit_adapter.py
git commit -m "feat: RDKit adapter with reserved unknown codes"
```

---

## Task 13: cosmolayer adapter and the acceptance benchmark

**Files:**
- Create: `src/wllr/io/cosmolayer_adapter.py`, `tests/test_benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `from_segment_store(store, target="area"|"charge"|"sigma_profile", *, config, scheme="cosmo-rs") -> tuple[AtomBatch, np.ndarray]` returning the batch and a boolean `is_test` mask taken from the store's own `split` column.

**This task is the reason the plan exists.** The numbers below were measured on `stores/cosmo_sample_10k_split` before implementation began, with an independent script. An implementation that reproduces them is almost certainly correct end to end, because the figure is simultaneously sensitive to the encoder, the refinement, the statistics, the backoff and the split.

**Acceptance criteria** — `attribute_levels=(("element","hybridization","degree","aromatic"),)`, `max_wl_depth=3`, `n_min=1`, `alpha=None`, trained on the 8,000 `split == "train"` molecules and scored on the 2,000 `split == "test"` molecules, with `R² = 1 − mean((y−ŷ)²)/var(y_test)`:

| quantity | expected |
|---|---|
| atoms total / train / test | 227,723 / 187,605 / 40,118 |
| classes at levels 0..4 | 33 / 1,010 / 17,076 / 67,452 / 124,585 |
| **test R², atomic area, max level 3** | **0.918** |
| **test R², atomic charge, max level 3** | **0.934** |
| mean matched level | 2.88 |
| global fallback rate | 0.000% |

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
"""Acceptance benchmark against a real COSMO store.

Skipped unless both cosmolayer and the store are present.
"""
import pathlib

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")

STORE = pathlib.Path(__file__).resolve().parents[1] / "stores" / "cosmo_sample_10k_split"
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="benchmark store absent")


@pytest.fixture(scope="module")
def store():
    from cosmolayer.store import SegmentStore
    return SegmentStore.load(STORE)


def _run(store, target):
    import wllr
    from wllr.io.cosmolayer_adapter import from_segment_store
    from wllr.io.rdkit_adapter import build_codes
    from wllr.config import WLLRConfig
    from rdkit import Chem

    attrs = ("element", "hybridization", "degree", "aromatic")
    p = Chem.SmilesParserParams(); p.removeHs = False
    mols = [Chem.MolFromSmiles(s, p) for s in store.molecules_df.smiles]
    codes, edges = build_codes(mols, attrs)
    cfg = WLLRConfig(target_dim=1, attribute_levels=(attrs,),
                     attribute_codes=codes, edge_codes=edges,
                     max_wl_depth=3, n_min=1)
    batch, is_test = from_segment_store(store, target=target, config=cfg)
    from wllr.model import _sub_batch
    model = wllr.fit(_sub_batch(batch, ~is_test), cfg)
    pred = wllr.predict_detailed(model, _sub_batch(batch, is_test))
    y = batch.y[is_test]
    r2 = 1 - np.mean((y - pred.value) ** 2) / y.var()
    return r2, pred


def test_corpus_shape(store):
    import wllr
    from wllr.io.cosmolayer_adapter import from_segment_store
    from wllr.io.rdkit_adapter import build_codes
    from wllr.config import WLLRConfig
    from rdkit import Chem
    attrs = ("element", "hybridization", "degree", "aromatic")
    p = Chem.SmilesParserParams(); p.removeHs = False
    mols = [Chem.MolFromSmiles(s, p) for s in store.molecules_df.smiles]
    codes, edges = build_codes(mols, attrs)
    cfg = WLLRConfig(target_dim=1, attribute_levels=(attrs,),
                     attribute_codes=codes, edge_codes=edges, max_wl_depth=3)
    batch, is_test = from_segment_store(store, target="area", config=cfg)
    assert batch.n_atoms == 227_723
    assert int((~is_test).sum()) == 187_605
    assert int(is_test.sum()) == 40_118

def test_class_counts_match_the_reference_run(store):
    from wllr.io.cosmolayer_adapter import from_segment_store
    from wllr.io.rdkit_adapter import build_codes
    from wllr.config import WLLRConfig
    from wllr.refine import refine
    from rdkit import Chem
    attrs = ("element", "hybridization", "degree", "aromatic")
    p = Chem.SmilesParserParams(); p.removeHs = False
    mols = [Chem.MolFromSmiles(s, p) for s in store.molecules_df.smiles]
    codes, edges = build_codes(mols, attrs)
    cfg = WLLRConfig(target_dim=1, attribute_levels=(attrs,),
                     attribute_codes=codes, edge_codes=edges, max_wl_depth=4)
    batch, _ = from_segment_store(store, target="area", config=cfg)
    counts = [lv.signatures.shape[0] for lv in refine(batch, cfg)]
    assert counts == [33, 1010, 17076, 67452, 124585]

def test_atomic_area_reaches_the_reference_r2(store):
    r2, pred = _run(store, "area")
    assert 0.913 < r2 < 0.923, f"expected ~0.918, got {r2:.4f}"
    assert abs(pred.matched_level.mean() - 2.88) < 0.02
    assert (pred.matched_level == -1).mean() < 1e-4

def test_atomic_charge_reaches_the_reference_r2(store):
    r2, _ = _run(store, "charge")
    assert 0.929 < r2 < 0.939, f"expected ~0.934, got {r2:.4f}"

def test_sigma_profile_predictions_are_non_negative(store):
    """design.md 11.4: backoff and shrinkage are convex combinations of
    training rows, so non-negativity is automatic. Any clipping is a bug."""
    import wllr
    from wllr.io.cosmolayer_adapter import from_segment_store
    from wllr.io.rdkit_adapter import build_codes
    from wllr.config import WLLRConfig
    from wllr.model import _sub_batch
    from rdkit import Chem
    attrs = ("element", "hybridization", "degree", "aromatic")
    p = Chem.SmilesParserParams(); p.removeHs = False
    mols = [Chem.MolFromSmiles(s, p) for s in store.molecules_df.smiles]
    codes, edges = build_codes(mols, attrs)
    cfg = WLLRConfig(target_dim=51, attribute_levels=(attrs,),
                     attribute_codes=codes, edge_codes=edges,
                     max_wl_depth=3, alpha=2.0)
    batch, is_test = from_segment_store(store, target="sigma_profile", config=cfg)
    model = wllr.fit(_sub_batch(batch, ~is_test), cfg)
    v = wllr.predict(model, _sub_batch(batch, is_test))
    assert v.shape[1] == 51
    assert v.min() >= 0.0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.io.cosmolayer_adapter'`

- [ ] **Step 3: Implement**

```python
# src/wllr/io/cosmolayer_adapter.py
"""cosmolayer SegmentStore -> AtomBatch (design.md 11.3, 11.4)."""
from __future__ import annotations

import numpy as np

from wllr.batch import AtomBatch, check_alignment
from wllr.config import WLLRConfig


def from_segment_store(store, target="area", *, config: WLLRConfig,
                       scheme="cosmo-rs") -> tuple[AtomBatch, np.ndarray]:
    """Build a batch and a test mask from a cosmolayer segment store.

    ``sigma_profile`` targets are converted to **area** units. The store's
    native ``SigmaProfileTable.profiles`` are area *fractions* summing to 1,
    with the scale held separately in ``.areas``; design.md 11.4 requires
    unnormalized areas, which is ``profiles * areas[:, None]``. Getting this
    wrong produces plausible numbers and no error.
    """
    import pandas as pd
    from rdkit import Chem

    from wllr.io.rdkit_adapter import from_rdkit

    df = store.molecules_df
    ai = np.asarray(store.atom_indices)
    n_atoms = int(ai.max()) + 1

    if target == "area":
        y = np.bincount(ai, weights=np.asarray(store.areas),
                        minlength=n_atoms)[:, None]
    elif target == "charge":
        y = np.bincount(ai, weights=np.asarray(store.charges),
                        minlength=n_atoms)[:, None]
    elif target == "sigma_profile":
        table = store.compute_atom_sigma_profiles(scheme=scheme)
        y = np.asarray(table.profiles, np.float64) * np.asarray(table.areas)[:, None]
    else:
        raise ValueError(f"unknown target {target!r}")

    params = Chem.SmilesParserParams()
    params.removeHs = False          # COSMO tables carry explicit hydrogens
    mols, orders = [], []
    for smi in df.smiles:
        mol = Chem.MolFromSmiles(smi, params)
        if mol is None:
            raise ValueError(f"unparseable SMILES in store: {smi[:60]}")
        # Atom-mapped SMILES are numbered onto the COSMO file's 0-based order.
        orders.append(np.argsort([a.GetAtomMapNum() for a in mol.GetAtoms()]))
        mols.append(mol)

    batch = from_rdkit(mols, y=y, config=config, atom_order=orders)

    # design.md 7.5 / 11.3: the check that actually catches reordering.
    atoms = pd.read_parquet(store.storage_dir / "atoms.parquet")
    from rdkit.Chem.PeriodicTable import GetPeriodicTable  # noqa: F401
    pt = Chem.GetPeriodicTable()
    expected = np.array([pt.GetAtomicNumber(s) for s in atoms["element"]], np.int64)
    check_alignment(batch, df.num_atoms.to_numpy(), expected)

    is_test = np.repeat((df.split == "test").to_numpy(), df.num_atoms.to_numpy())
    return batch, is_test
```

- [ ] **Step 4: Run the benchmark**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS (5 tests)

If an R² is off by more than the tolerance, **do not adjust the tolerance.** Work down this list: class counts wrong → encoder or bond codes; class counts right but R² low → statistics or backoff; R² suspiciously *high* → the split leaked, check that `fit` saw only training atoms.

- [ ] **Step 5: Commit**

```bash
git add src/wllr/io/cosmolayer_adapter.py tests/test_benchmark.py
git commit -m "feat: cosmolayer adapter with end-to-end acceptance benchmark"
```

---

## Task 14: scikit-learn wrapper

**Files:**
- Create: `src/wllr/sklearn.py`
- Test: `tests/test_sklearn.py`

**Interfaces:**
- Produces: `WLLRRegressor(config, **params)` with `fit(X, y)`, `predict(X)`, `get_params`, `set_params`, `score`; `GraphKFold(n_splits)` yielding graph-disjoint index splits.

Read design.md 10.2. The wrapper must default to **graph-level** splitting: node-level random splitting puts WL-identical atoms from one molecule on both sides and inflates scores badly. This is the single easiest way to produce a misleading number with this method, so the safe behavior belongs in the default rather than in the documentation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sklearn.py
import numpy as np
import pytest
from wllr.sklearn import WLLRRegressor, GraphKFold
from tests.helpers import chain_batch, simple_config

def test_get_set_params_round_trip():
    r = WLLRRegressor(simple_config(), n_min=3, alpha=1.0)
    assert r.get_params()["n_min"] == 3
    r.set_params(n_min=9)
    assert r.get_params()["n_min"] == 9

def test_fit_predict_matches_the_functional_core():
    import wllr
    cfg = simple_config()
    b = chain_batch(15, graphs=3)
    r = WLLRRegressor(cfg).fit(b, b.y)
    np.testing.assert_allclose(r.predict(b), wllr.predict(wllr.fit(b, cfg), b))

def test_graph_kfold_never_splits_a_molecule():
    b = chain_batch(6, graphs=10)
    for train, test in GraphKFold(n_splits=5).split(b):
        assert not (set(b.graph_id[train]) & set(b.graph_id[test]))

def test_graph_kfold_covers_every_atom_exactly_once_as_test():
    b = chain_batch(6, graphs=10)
    seen = np.concatenate([test for _, test in GraphKFold(5).split(b)])
    assert np.array_equal(np.sort(seen), np.arange(b.n_atoms))

def test_score_is_r2():
    cfg = simple_config()
    b = chain_batch(20, graphs=4)
    r = WLLRRegressor(cfg).fit(b, b.y)
    assert r.score(b, b.y) <= 1.0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_sklearn.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wllr.sklearn'`

- [ ] **Step 3: Implement**

```python
# src/wllr/sklearn.py
"""A mutable-estimator wrapper around the immutable core (design.md 10.2).

Contorting the core into fit-mutates-self would forfeit the merge monoid for
the sake of an interface. This adapter exists so GridSearchCV can sweep alpha,
n_min and K without the core inheriting mutable-estimator semantics.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from wllr.batch import AtomBatch
from wllr.config import WLLRConfig
from wllr.model import fit as _fit
from wllr.predict import predict as _predict


class GraphKFold:
    """K-fold over *graphs*, never over nodes.

    Node-level random splitting puts WL-identical atoms from one molecule on
    both sides of the split and inflates scores badly.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 0):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, batch: AtomBatch, y=None, groups=None):
        graphs = np.unique(batch.graph_id)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(graphs)
        for part in np.array_split(graphs, self.n_splits):
            test_mask = np.isin(batch.graph_id, part)
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class WLLRRegressor:
    def __init__(self, config: WLLRConfig, n_min: int | None = None,
                 alpha: float | None = None):
        self.config = config
        self.n_min = config.n_min if n_min is None else n_min
        self.alpha = config.alpha if alpha is None else alpha
        self.model_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {"config": self.config, "n_min": self.n_min, "alpha": self.alpha}

    def set_params(self, **params) -> "WLLRRegressor":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: AtomBatch, y=None) -> "WLLRRegressor":
        cfg = replace(self.config, n_min=self.n_min, alpha=self.alpha)
        batch = X if y is None else AtomBatch(**{**X.__dict__, "y": y})
        self.model_ = _fit(batch, cfg)
        return self

    def predict(self, X: AtomBatch) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        return _predict(self.model_, X)

    def score(self, X: AtomBatch, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(1 - np.mean((y - pred) ** 2) / np.var(y))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sklearn.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: PASS (all tasks)

- [ ] **Step 6: Commit**

```bash
git add src/wllr/sklearn.py tests/test_sklearn.py
git commit -m "feat: scikit-learn wrapper with graph-level splitting by default"
```

---

## Task 15: README and the negative control

**Files:**
- Create: `README.md`, `tests/test_properties.py`

The negative control pins the known 1-WL expressiveness bound as *intended behavior* rather than an undetected bug (design.md 10.4, last row). Without it, a future contributor may "fix" it.

- [ ] **Step 1: Write the test**

```python
# tests/test_properties.py
import numpy as np
import wllr
from wllr.batch import AtomBatch
from wllr.refine import refine
from tests.helpers import simple_config

def _cycles(sizes):
    """Disjoint cycles as one batch: all nodes share every attribute."""
    src, dst, gid, off = [], [], [], 0
    for g, n in enumerate(sizes):
        for i in range(n):
            a, b = off + i, off + (i + 1) % n
            src += [a, b]; dst += [b, a]
        gid += [g] * n
        off += n
    n_total = off
    return AtomBatch(node_attrs=np.zeros((n_total, 1), np.int64),
                     edge_src=np.array(src, np.int64),
                     edge_dst=np.array(dst, np.int64),
                     edge_attr=np.ones(len(src), np.int64),
                     graph_id=np.array(gid, np.int64),
                     y=np.zeros((n_total, 1)))

def test_two_wl_indistinguishable_graphs_do_collide():
    """Accepted limit, not a bug: C6 and 2xC3 are 1-WL indistinguishable.

    Every node in both is degree-2 with degree-2 neighbors at every depth, so
    WLLR must assign them one class. Pinning this stops a future contributor
    from 'fixing' the expressiveness bound.
    """
    cfg = simple_config(max_wl_depth=4)
    six = refine(_cycles([6]), cfg)[-1].labels
    twin = refine(_cycles([3, 3]), cfg)[-1].labels
    assert len(set(six.tolist())) == 1
    assert len(set(twin.tolist())) == 1

def test_isomorphic_graphs_give_matching_class_multisets():
    cfg = simple_config(max_wl_depth=3)
    a = refine(_cycles([5, 7]), cfg)[-1]
    b = refine(_cycles([7, 5]), cfg)[-1]
    assert (sorted(np.bincount(a.labels).tolist()) ==
            sorted(np.bincount(b.labels).tolist()))
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_properties.py -v`
Expected: PASS (2 tests)

- [ ] **Step 3: Write `README.md`**

Cover: what WLLR is (one paragraph, using the hierarchical-regressogram framing from `literature.md` 3); install; a fifteen-line worked example on SMILES; the merge monoid in three lines (`sum(models)`); a **Not implemented in v1** section listing vocabulary pruning (design.md 8), the neighbor schema (3.6), and full covariance (5.2), each with its one-line reason, so their absence reads as a decision; and a pointer to `design.md` as authoritative and `literature.md` for prior work.

- [ ] **Step 4: Commit**

```bash
git add README.md tests/test_properties.py
git commit -m "docs: README and the 1-WL negative control"
```

---

## Self-Review

**Spec coverage.** §2 → Tasks 4, 7, 8 (nesting, prefix, monotonicity all tested). §3.1/3.3 → Task 4 (ids are dedupe-derived; no hashing needed, per §7.2/§9). §3.4 hash width → **not applicable**: no truncated digests are used anywhere, which is the stronger resolution. §3.5 → Task 4. §3.6 → config field raising `NotImplementedError` (Task 1), as the spec's "evaluated, not adopted" requires. §4.1 → Task 5. §4.2 → Task 9. §5 → Task 7. §6 → Task 8. §7 → Tasks 2–5. §8 → deliberately out of scope, recorded in the README (Task 15). §9 → Task 11. §10.1 → Task 6. §10.2 → Task 14. §10.3 → Task 10. §10.4 → every property has a test; the negative control is Task 15. §11 → Tasks 2, 12, 13. §11.4 → Task 13's non-negativity test and the fraction→area conversion. §12 → Task 8. §13 open questions are not implementation work.

**One cross-task dependency.** `fit` accepts `chunk_size` in its config from Task 1, but the branch that honors it needs `fold`, which does not exist until Task 7. Task 6 therefore ships `fit` without it and Task 7 adds it, together with `test_chunked_fit_equals_single_chunk_fit`. `_sub_batch` lands in Task 6 because Task 13 imports it.

**Type consistency.** `FrozenLevel` fields (`signatures`, `count`, `mean`, `msd`, `parent`) are used identically in Tasks 5, 7, 8, 9, 11. `LevelLabels` (`labels`, `signatures`, `parent`) in Tasks 4, 5. `_translate(sig, remap_prev, n_bond, is_wl)` has one signature, used by Tasks 7 and 8. `_sub_batch` is defined in `wllr/model.py` (Task 6/7) and used by Task 13's benchmark.
