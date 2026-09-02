# Edge Attribute Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make edge attributes configurable the way atom attributes already are — a named registry of bond attribute providers, corpus-discovered code tables, and a configurable `edge_attributes` tuple that may be empty.

**Architecture:** `NodeBatch.edge_attr` (1-D) becomes `edge_attrs` (2-D, one column per configured edge attribute). `SieveConfig` gains `edge_attributes: tuple[str, ...]` and nests `edge_codes` into `{name: {value: code}}`, mirroring `attribute_codes`. `refine` collapses the columns into the single integer its hot loop needs using a **mixed radix derived from the config**, never from the batch. The adapter grows `_BOND_ATTRS` / `_MOL_BOND_ATTRS` registries paralleling `_ATTRS` / `_MOL_ATTRS`.

**Tech Stack:** Python 3.11+, NumPy, RDKit, joblib, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-02-edge-attribute-schema-design.md`

## Global Constraints

- **All three CI checks must pass before every commit:** `ruff check src tests charge_experiments`, `ruff format --check src tests charge_experiments`, `ty check src tests charge_experiments`, and `pytest`. `ty` in particular is easy to forget and has failed CI on this repo before.
- Run commands with `uv run`, and prefix with `export PATH="/home/craabreu/miniforge3/bin:$PATH"` — `uv` is not on the default PATH here.
- The full suite takes ~2 minutes. Baseline at the start of this plan: **302 passed, 1 skipped**.
- Line length is 88 (ruff default). `from __future__ import annotations` is already present in every module touched.
- The reserved-unknown convention is **codes `0..k-1` for the `k` sorted observed values, `k` reserved for unseen** — same as `attribute_codes`. Note this *differs* from the old hardcoded edge table, which started at 1 and used 0 as the unknown.
- `edge_attributes` may be empty (`()`). The zero-width guard at `config.py:45` applies to `attribute_levels` only and must **not** be extended to edges.
- Never use `assert` for a runtime invariant — this codebase raises explicitly, because `assert` is compiled away under `python -O` (see the comment at `refine.py:117`).

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/sieve/batch.py` | `NodeBatch.edge_attrs` 2-D storage; `CSRLayout.attr` 2-D; `concat_batches` | 1 |
| `src/sieve/config.py` | `edge_attributes`, nested `edge_codes`, `edge_radices`, `n_edge_types`, validation, `schema_version`, `FORMAT_VERSION` | 2 |
| `src/sieve/model.py` | persist/restore `edge_attributes` + nested `edge_codes` | 2 |
| `src/sieve/refine.py` | config-driven mixed-radix collapse; per-column range check | 1 (interim), 3 (final) |
| `src/sieve/io/rdkit_adapter.py` | `_BOND_ATTRS` / `_MOL_BOND_ATTRS` registries; edge vocabulary discovery; 2-D edge fill | 2 (interim), 4, 6 |
| `design.md` | reconcile §9.2 / §10.1 / §11.1 with the implementation | 7 |
| `tests/helpers.py` | shared `_BASE_CONFIG` and `chain_batch` fixtures | 1, 2 |
| `tests/test_batch.py`, `test_refine.py`, `test_predict.py`, `test_neighbor_depth.py`, `test_merge.py`, `test_properties.py`, `test_config.py`, `test_benchmark.py`, `test_rdkit_adapter.py` | fixture updates + new behavior tests | 1–6 |
| `charge_experiments/charge_experiments/predictors/sieve_predictor.py` | downstream `build_codes` caller | 4 |

**Not touched:** `src/sieve/predict.py` and `src/sieve/merge.py`. Both consume `cfg.n_edge_types` purely as an integer modulus (`np.divmod(pad, n_edge_types)` at `merge.py:55`), and `n_edge_types` keeps its meaning — "size of the collapsed edge alphabet". The collapsed code is opaque to them by design. **If you find yourself editing either file, stop and re-read spec §3 — you have probably made the collapse batch-dependent.**

---

## Task 1: `NodeBatch.edge_attrs` becomes 2-D

Pure shape change. `SieveConfig` is untouched; exactly one edge attribute column exists throughout, so `refine` reads `csr.attr[:, 0]` provisionally (Task 3 generalizes it).

**Files:**
- Modify: `src/sieve/batch.py:38` (field), `:53-56` (`_check_shapes`), `:173` (`__getitem__`), `:194` (`csr`), `:247-271` (`concat_batches`)
- Modify: `src/sieve/refine.py:105`
- Modify: `src/sieve/io/rdkit_adapter.py:447`
- Test: `tests/test_batch.py`, `tests/helpers.py`, `tests/test_refine.py`, `tests/test_predict.py`, `tests/test_neighbor_depth.py`, `tests/test_merge.py`, `tests/test_properties.py`, `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `NodeBatch.edge_attrs: np.ndarray` of shape `(n_edges, n_edge_attr)`, `int64`. `CSRLayout.attr` is likewise 2-D. The old name `edge_attr` no longer exists anywhere.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_batch.py`:

```python
def test_edge_attrs_must_be_two_dimensional():
    """edge_attrs is (n_edges, n_edge_attr) per design.md 11.1. A 1-D array is
    the pre-2026-09 shape and must raise rather than broadcast into a single
    column, which would silently accept a batch built against the old API."""
    import pytest

    with pytest.raises(ValueError, match="edge_attrs"):
        NodeBatch(
            node_attrs=np.zeros((2, 1), np.int64),
            edge_src=np.array([0, 1], np.int64),
            edge_dst=np.array([1, 0], np.int64),
            edge_attrs=np.ones(2, np.int64),  # 1-D: wrong
            graph_id=np.zeros(2, np.int64),
        )


def test_edge_attrs_supports_a_zero_width_schema():
    """An empty edge schema is legal (spec: edge_attributes == ()), so an
    (n_edges, 0) array must construct and slice cleanly rather than tripping
    a shape check written for the one-column case."""
    b = NodeBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([0, 1], np.int64),
        edge_dst=np.array([1, 0], np.int64),
        edge_attrs=np.zeros((2, 0), np.int64),
        graph_id=np.zeros(2, np.int64),
    )
    assert b.edge_attrs.shape == (2, 0)
    assert b[np.array([0, 1])].edge_attrs.shape == (2, 0)
    assert b.csr().attr.shape == (2, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_batch.py::test_edge_attrs_must_be_two_dimensional \
              tests/test_batch.py::test_edge_attrs_supports_a_zero_width_schema -v
```

Expected: both FAIL with `TypeError: NodeBatch.__init__() got an unexpected keyword argument 'edge_attrs'`.

- [ ] **Step 3: Rename the field and tighten the shape check**

In `src/sieve/batch.py`, change the field at line 38:

```python
    edge_attrs: np.ndarray  # (n_edges, n_edge_attr) int64, encoded categoricals
```

Replace the loop in `_check_shapes` (lines 53-56) with:

```python
        e = self.edge_src.shape[0]
        if self.edge_dst.shape != (e,):
            raise ValueError(f"edge_dst must have shape ({e},)")
        if self.edge_attrs.ndim != 2 or self.edge_attrs.shape[0] != e:
            raise ValueError(
                f"edge_attrs must have shape ({e}, n_edge_attr), got "
                f"{self.edge_attrs.shape}"
            )
```

- [ ] **Step 4: Update the three remaining `batch.py` sites**

`__getitem__` (line 173): `edge_attr=self.edge_attr[keep],` → `edge_attrs=self.edge_attrs[keep],`

`csr()` (line 194): `attr=self.edge_attr[order],` → `attr=self.edge_attrs[order],`

`concat_batches` (lines 247, 253, 270): rename the accumulator `edge_attr` → `edge_attrs`, append `p.edge_attrs`, and concatenate on axis 0:

```python
    edge_src, edge_dst, edge_attrs, graph_id = [], [], [], []
```
```python
        edge_attrs.append(p.edge_attrs)
```
```python
        edge_attrs=np.concatenate(edge_attrs, axis=0),
```

Also update `CSRLayout`'s docstring note if it mentions a 1-D `attr` (it does not today, but check).

- [ ] **Step 5: Update `refine.py` provisionally**

In `src/sieve/refine.py`, replace line 105:

```python
            # Encode (neighbor label, bond) as one integer so a row of
            # neighbors is a plain integer vector. Column 0 only for now;
            # the config-driven mixed-radix collapse over every column
            # lands in the next change.
            pair = base[csr.dst] * n_edge_types + csr.attr[:, 0]
```

And in the range check at line 93, subscript the column too:

```python
        bad = (csr.attr[:, 0] < 0) | (csr.attr[:, 0] >= n_edge_types)
```
```python
                f"edge_attrs contains code {int(csr.attr[:, 0][bad][0])}, outside "
```

Guard at line 88 becomes `if kinds and csr.attr.shape[0]:` (`.size` is 0 for a zero-width schema, which would wrongly skip the check when there are edges).

- [ ] **Step 6: Update the adapter's batch construction**

In `src/sieve/io/rdkit_adapter.py`, at the `NodeBatch(...)` return (line ~447):

```python
        edge_attrs=np.array(attr, np.int64).reshape(-1, 1),
```

- [ ] **Step 7: Update every test fixture**

Rename `edge_attr=` → `edge_attrs=` and make each array 2-D. The mechanical patterns:

- `np.ones(K, np.int64)` → `np.ones((K, 1), np.int64)`
- `np.zeros(0, np.int64)` → `np.zeros((0, 1), np.int64)`
- `np.array([1, 1, 2], np.int64)` → `np.array([[1], [1], [2]], np.int64)`
- `np.full(4, 7, np.int64)` → `np.full((4, 1), 7, np.int64)`
- `edge_attr=b.edge_attr` → `edge_attrs=b.edge_attrs`
- `NodeBatch(**{**b.__dict__, "edge_attr": X})` → `NodeBatch(**{**b.__dict__, "edge_attrs": X})`

Files and counts (from `grep -rn edge_attr --include=*.py .`): `tests/test_batch.py` (14), `tests/test_refine.py` (7), `tests/test_predict.py` (3), `tests/test_neighbor_depth.py` (3), `tests/helpers.py` (2), `tests/test_rdkit_adapter.py` (1), `tests/test_properties.py` (1), `tests/test_merge.py` (1).

Find every remaining one with:

```bash
grep -rn "edge_attr\b" --include=*.py . | grep -v "\.venv"
```

That must return nothing when you are done (note `\b` — it excludes `edge_attrs`).

- [ ] **Step 8: Run the new tests, then the full suite**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_batch.py -v
uv run pytest -q
```

Expected: the two new tests PASS; suite is **304 passed, 1 skipped** (302 + 2).

- [ ] **Step 9: Lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: all three clean.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: NodeBatch.edge_attr becomes 2-D edge_attrs

Adopts the (n_edges, n_edge_attr) shape and plural name design.md 11.1
already specifies, so an edge can carry more than one attribute. Pure
shape change: exactly one column exists everywhere, and refine reads
column 0 provisionally until the config-driven collapse lands.

A 1-D edge_attrs now raises rather than broadcasting, and an
(n_edges, 0) zero-width schema constructs, slices and CSR-orders cleanly."
```

---

## Task 2: `SieveConfig.edge_attributes` and nested `edge_codes`

The atomic interface change. `edge_codes` goes from `{value: code}` to `{name: {value: code}}`, `edge_attributes` names the columns and fixes their order, and `n_edge_types` becomes a product over per-attribute radices. `FORMAT_VERSION` 2 → 3, clean break.

The adapter still hardcodes its (now nested) bond-type table here — Task 4 replaces that with real discovery. That is deliberate scaffolding to keep the suite green between tasks.

**Files:**
- Modify: `src/sieve/config.py:11` (`FORMAT_VERSION`), `:32-40` (fields), `:42-71` (`__post_init__`), `:73-86` (`_freeze_mappings`), `:109-112` (`__getstate__`), `:190-193` (`n_edge_types`), `:203-213` (`schema_version`)
- Modify: `src/sieve/model.py:104` (save), `:143` (load)
- Modify: `src/sieve/io/rdkit_adapter.py:375` (hardcoded table), `:411` + `:438` (edge fill)
- Test: `tests/test_config.py`, `tests/helpers.py`, `tests/test_refine.py`, `tests/test_predict.py`, `tests/test_neighbor_depth.py`, `tests/test_benchmark.py`, `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: `NodeBatch.edge_attrs` (Task 1).
- Produces:
  - `SieveConfig.edge_attributes: tuple[str, ...] = ("bond_type",)` — declared after `max_wl_depth`, so the first five positional fields are unchanged.
  - `SieveConfig.edge_codes: Mapping[str, Mapping[str, int]]`
  - `SieveConfig.edge_radices -> tuple[int, ...]` — `len(edge_codes[n]) + 1` per attribute, in `edge_attributes` order.
  - `SieveConfig.n_edge_types -> int` — `math.prod(edge_radices)`; `1` when `edge_attributes == ()`.
  - `FORMAT_VERSION == 3`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_edge_radices_and_n_edge_types_are_a_product_over_attributes():
    """n_edge_types is the size of the collapsed edge alphabet. Each attribute
    contributes its vocabulary plus one reserved unknown code, and the
    collapse is mixed-radix, so the alphabet is the product."""
    cfg = simple_config(
        edge_attributes=("bond_type", "conjugated"),
        edge_codes={
            "bond_type": {"SINGLE": 0, "DOUBLE": 1},
            "conjugated": {"False": 0, "True": 1},
        },
    )
    assert cfg.edge_radices == (3, 3)
    assert cfg.n_edge_types == 9


def test_empty_edge_schema_gives_a_single_edge_type():
    """edge_attributes == () is a supported control arm: every edge becomes
    indistinguishable and refinement is pure topology. The empty product is 1,
    so `pair = base[dst] * 1 + 0` degenerates to `base[dst]` with no
    special-casing anywhere."""
    cfg = simple_config(edge_attributes=(), edge_codes={})
    assert cfg.edge_radices == ()
    assert cfg.n_edge_types == 1


def test_edge_attributes_and_edge_codes_must_agree():
    """A named attribute with no code table, or a table for an unnamed
    attribute, is config drift -- it would silently change column count or
    ordering. Both directions raise."""
    import pytest

    with pytest.raises(ValueError, match="edge_attributes"):
        simple_config(
            edge_attributes=("bond_type", "conjugated"),
            edge_codes={"bond_type": {"SINGLE": 0}},
        )
    with pytest.raises(ValueError, match="edge_attributes"):
        simple_config(
            edge_attributes=("bond_type",),
            edge_codes={"bond_type": {"SINGLE": 0}, "conjugated": {"True": 0}},
        )


def test_edge_attributes_enter_schema_version():
    """Two models whose edge columns mean different things must not merge,
    even when every code table is identical (design.md 9.2)."""
    a = simple_config(
        edge_attributes=("bond_type",), edge_codes={"bond_type": {"SINGLE": 0}}
    )
    b = simple_config(
        edge_attributes=("conjugated",), edge_codes={"conjugated": {"SINGLE": 0}}
    )
    assert a.schema_version != b.schema_version
```

- [ ] **Step 2: Run them to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_config.py -k "edge" -v
```

Expected: FAIL — `TypeError: ... unexpected keyword argument 'edge_attributes'`.

- [ ] **Step 3: Change the config fields and version**

In `src/sieve/config.py`, add `import math` at the top (after `import json`), and set:

```python
FORMAT_VERSION = 3
```

Change the field block (lines 32-40) to:

```python
    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]
    attribute_codes: Mapping[str, Mapping[str, int]]
    edge_codes: Mapping[str, Mapping[str, int]]
    max_wl_depth: int
    edge_attributes: tuple[str, ...] = ("bond_type",)
    neighbor_depth: int | None = None
    minimum_support: int = 1
    shrinkage_strength: float | None = None
    chunk_size: int | None = None
```

Update the class docstring's second paragraph to:

```
    ``attribute_codes`` and ``edge_codes`` are part of what a class *means*, so
    they enter ``schema_version``, and so does ``edge_attributes`` -- it fixes
    which edge column is which: two models built with different encodings or a
    different column order cannot be merged even if every other field agrees.
```

- [ ] **Step 4: Validate in `__post_init__`**

Insert immediately before `self._freeze_mappings()` (line 71):

```python
        # An empty edge schema is legal -- it is the pure-topology control arm
        # -- so the zero-width rule above deliberately does not extend here.
        object.__setattr__(self, "edge_attributes", tuple(self.edge_attributes))
        if len(set(self.edge_attributes)) != len(self.edge_attributes):
            raise ValueError(
                f"edge_attributes has a duplicate: {self.edge_attributes}"
            )
        if set(self.edge_attributes) != set(self.edge_codes):
            raise ValueError(
                "edge_attributes and edge_codes must name the same attributes: "
                f"{sorted(self.edge_attributes)} != {sorted(self.edge_codes)}"
            )
```

- [ ] **Step 5: Nest the freezing and pickling**

In `_freeze_mappings`, replace the `edge_codes` line (line 86) with:

```python
        object.__setattr__(
            self,
            "edge_codes",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in self.edge_codes.items()}
            ),
        )
```

In `__getstate__`, replace line 111 with:

```python
        state["edge_codes"] = {k: dict(v) for k, v in self.edge_codes.items()}
```

- [ ] **Step 6: Replace `n_edge_types` and extend `schema_version`**

Replace the `n_edge_types` property (lines 190-193) with:

```python
    @property
    def edge_radices(self) -> tuple[int, ...]:
        """Alphabet size per edge attribute, in ``edge_attributes`` order,
        including the code reserved for an unseen value."""
        return tuple(len(self.edge_codes[n]) + 1 for n in self.edge_attributes)

    @property
    def n_edge_types(self) -> int:
        """Size of the collapsed edge alphabet -- the modulus ``refine``,
        ``merge`` and ``predict`` encode a (neighbor label, edge) pair with.

        A product over ``edge_radices`` because the columns collapse mixed
        radix. The empty product is 1, which is exactly right for an empty
        edge schema: every edge collapses to code 0 and the pair encoding
        degenerates to the neighbor label alone.
        """
        return math.prod(self.edge_radices)
```

In `schema_version`'s payload (line 210), replace the `edge_codes` entry with:

```python
            "edge_attributes": list(self.edge_attributes),
            "edge_codes": {
                k: dict(sorted(v.items())) for k, v in sorted(self.edge_codes.items())
            },
```

Update `check_mergeable`'s error message to mention edge attributes:

```python
            f"cannot merge: schema_version differs ({a.schema_version[:12]} != "
            f"{b.schema_version[:12]}); attribute levels, codes, edge attributes, "
            "edge codes and max_wl_depth must all match"
```

- [ ] **Step 7: Update model serialization**

In `src/sieve/model.py`, in `save`'s blob (line 104):

```python
            "edge_attributes": list(cfg.edge_attributes),
            "edge_codes": {k: dict(v) for k, v in cfg.edge_codes.items()},
```

In `load`'s `SieveConfig(...)` call (line 143):

```python
            edge_codes=blob["edge_codes"],
            edge_attributes=tuple(blob["edge_attributes"]),
```

No change is needed to the `format_version` guard — it already refuses anything that is not `FORMAT_VERSION`.

- [ ] **Step 8: Update the adapter to the nested shape (provisional)**

In `src/sieve/io/rdkit_adapter.py`, replace line 375:

```python
    # Provisional: Task 4 replaces this with corpus discovery through
    # _BOND_ATTRS. Codes follow the node convention -- 0..k-1 over the sorted
    # observed values, k reserved for unseen.
    edge_codes = {
        "bond_type": {
            v: i for i, v in enumerate(["AROMATIC", "DOUBLE", "SINGLE", "TRIPLE"])
        }
    }
```

Replace the hoisted `edge_codes = dict(config.edge_codes)` (line 411) with:

```python
    edge_tables = [dict(config.edge_codes[name]) for name in config.edge_attributes]
    edge_unknowns = [len(t) for t in edge_tables]
    edge_getters = [_BOND_ATTRS_PROVISIONAL[name] for name in config.edge_attributes]
```

and add, next to `_ATTRS`:

```python
# Provisional single-entry table; Task 4 promotes this to the real
# _BOND_ATTRS registry alongside _MOL_BOND_ATTRS.
_BOND_ATTRS_PROVISIONAL = {"bond_type": lambda b: str(b.GetBondType())}
```

Replace the bond loop body (lines 435-441):

```python
        for b in mol.GetBonds():
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            row = [
                t.get(f(b), unk)
                for t, unk, f in zip(
                    edge_tables, edge_unknowns, edge_getters, strict=True
                )
            ]
            src += [u, v]
            dst += [v, u]
            attr += [row, row]
```

and the batch construction:

```python
        edge_attrs=np.array(attr, np.int64).reshape(len(attr), -1),
```

- [ ] **Step 9: Update every config fixture**

Rewrite each `edge_codes=` literal into the nested form, keeping `n_edge_types` unchanged where a test depends on it.

- `tests/helpers.py:14` — `edge_codes={"SINGLE": 1, "DOUBLE": 2}` → `edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}}` (radix 3, `n_edge_types` 3 — same as before).
- `tests/test_refine.py:13` — identical change.
- `tests/test_config.py:12` — `edge_codes={"SINGLE": 1}` → `edge_codes={"bond_type": {"SINGLE": 0}}` (radix 2, `n_edge_types` 2 — same as before).
- `tests/test_neighbor_depth.py`, `tests/test_benchmark.py`, `tests/test_rdkit_adapter.py` — same mechanical change; `test_benchmark.py` takes `edges` from `build_codes`, which now returns the nested shape, so it needs no literal edit.

⚠️ **`tests/test_predict.py:146` needs care, not a mechanical rewrite.** `test_oov_neighbor_does_not_falsely_match_a_lower_degree_class` deliberately arranges `bond == n_edge_types - 1`, because that is the only value for which the OOV encoding `-1 * n_edge_types + bond` lands on the pad sentinel `-1`. Under the old table `{"SINGLE": 1}` gave `n_edge_types == 2` and `bond == 1`, satisfying it. Under the new convention `{"bond_type": {"SINGLE": 0}}` still gives `n_edge_types == 2`, but `SINGLE` now encodes as `0`, so the hazard is no longer exercised and the test silently becomes vacuous. Keep the intent by using the **reserved unknown code** in the query batch, which is `n_edge_types - 1 == 1`:

```python
        edge_codes={"bond_type": {"SINGLE": 0}},
```
```python
    # Bond code 1 is the reserved unknown -- and n_edge_types - 1, the one
    # value for which an OOV neighbor's encoded pair collides with the -1 pad
    # sentinel. That collision is the whole point of this test.
    query = NodeBatch(
        node_attrs=np.array([[0], [1]], np.int64),
        edge_src=np.array([0, 1], np.int64),
        edge_dst=np.array([1, 0], np.int64),
        edge_attrs=np.ones((2, 1), np.int64),
        graph_id=np.zeros(2, np.int64),
    )
```

Also extend that test's docstring to record why the bond code must be `n_edge_types - 1`.

- [ ] **Step 10: Add the format-version break test**

Add to `tests/test_model_io.py` (or `tests/test_model.py` — use whichever already holds `SieveModel.load` tests; find it with `grep -rln "SieveModel.load" tests/`):

```python
def test_loading_a_format_version_2_model_refuses_rather_than_guessing():
    """format_version 2 stored edge_codes as a flat {value: code} table. There
    is no migration (spec: clean break), and design.md 9.2 requires a reader
    that does not recognize the layout to refuse, not guess."""
    import json

    import numpy as np
    import pytest

    import sieve

    blob = {"format_version": 2, "schema_version": "deadbeef"}
    path = tmp_path / "old.npz"
    # `global` is a Python keyword, so it cannot be a keyword argument -- the
    # array name has to go through a dict splat.
    np.savez(
        path,
        **{
            "config": np.frombuffer(json.dumps(blob).encode(), np.uint8),
            "global": np.zeros(3),
        },
    )
    with pytest.raises(ValueError, match="unsupported format_version 2"):
        SieveModel.load(path)
```

Declare the test as `def test_loading_a_format_version_2_model_refuses_rather_than_guessing(tmp_path):` and import `SieveModel` with `from sieve.model import SieveModel` (do not rely on a top-level `sieve.SieveModel` re-export unless `grep -n "SieveModel" src/sieve/__init__.py` shows one).

- [ ] **Step 11: Run the suite**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
```

Expected: **309 passed, 1 skipped** (304 + 4 config + 1 format-version).

- [ ] **Step 12: Lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat!: SieveConfig.edge_attributes and nested edge_codes

edge_codes goes from a flat {value: code} table to {name: {value: code}},
mirroring attribute_codes, and edge_attributes names the columns and fixes
their order. n_edge_types becomes a product over edge_radices (vocabulary
plus one reserved unknown per attribute), so an empty edge schema yields
the empty product 1 -- the pure-topology arm, with no special case.

edge_attributes enters schema_version: column order is part of what a class
means. FORMAT_VERSION 2 -> 3, clean break; a v2 model refuses to load.

The adapter's bond table is still hardcoded here, now in the nested shape;
corpus discovery through _BOND_ATTRS follows."
```

---

## Task 3: Config-driven mixed-radix collapse in `refine`

The load-bearing change. Read spec §3 before starting.

**Files:**
- Modify: `src/sieve/refine.py:84-105`
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: `SieveConfig.edge_radices`, `SieveConfig.n_edge_types`, `SieveConfig.edge_attributes` (Task 2); `CSRLayout.attr` 2-D (Task 1).
- Produces: `refine` supports any number of edge attribute columns, including zero. No new public names.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_refine.py`:

```python
def two_bond_config(**kw):
    return replace(
        _BASE_CONFIG,
        edge_attributes=("bond_type", "conjugated"),
        edge_codes={
            "bond_type": {"SINGLE": 0, "DOUBLE": 1},
            "conjugated": {"False": 0, "True": 1},
        },
        max_wl_depth=1,
        **kw,
    )


def two_paths(conj_a, conj_b):
    """Two disjoint 3-node paths. Both bonds are SINGLE (code 0) in each; the
    first path's bonds carry conjugated=`conj_a`, the second's `conj_b`."""
    src = np.array([0, 1, 1, 2, 3, 4, 4, 5], np.int64)
    dst = np.array([1, 0, 2, 1, 4, 3, 5, 4], np.int64)
    cols = np.array([conj_a] * 4 + [conj_b] * 4, np.int64)
    return NodeBatch(
        node_attrs=np.zeros((6, 1), np.int64),
        edge_src=src,
        edge_dst=dst,
        edge_attrs=np.stack([np.zeros(8, np.int64), cols], axis=1),
        graph_id=np.array([0, 0, 0, 1, 1, 1], np.int64),
        y=np.zeros((6, 1)),
    )


def test_a_second_edge_attribute_refines_the_partition():
    """Two path centers identical in every way except their bonds'
    `conjugated` value must land in different classes -- which only happens if
    the second column actually reaches the pair encoding."""
    b = two_paths(conj_a=0, conj_b=1)
    labels = refine(b, two_bond_config())[-1].labels
    assert labels[1] != labels[4]  # the two path centers


def test_a_column_the_config_does_not_declare_is_ignored_consistently():
    """With only bond_type declared, the conjugated column is not part of the
    schema and the same two centers must collapse into one class."""
    b = two_paths(conj_a=0, conj_b=1)
    one_col = replace(
        _BASE_CONFIG,
        edge_attributes=("bond_type",),
        edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}},
        max_wl_depth=1,
    )
    trimmed = NodeBatch(**{**b.__dict__, "edge_attrs": b.edge_attrs[:, :1]})
    labels = refine(trimmed, one_col)[-1].labels
    assert labels[1] == labels[4]


def test_column_count_must_match_the_declared_schema():
    """A batch whose column count disagrees with edge_attributes means the
    batch and config were built against different schemas. Raise -- silently
    reading a prefix would conflate distinct classes."""
    import pytest

    b = two_paths(conj_a=0, conj_b=0)
    with pytest.raises(ValueError, match="edge attribute column"):
        refine(b, cfg())  # cfg() declares one attribute; b has two columns


def test_empty_edge_schema_refines_on_topology_alone():
    """edge_attributes == () collapses every edge to code 0 and n_edge_types
    to 1, so `pair = base[dst] * 1 + 0` is just the neighbor label. The two
    path centers, differing only in bond attributes, must merge."""
    b = two_paths(conj_a=0, conj_b=1)
    empty = replace(
        _BASE_CONFIG, edge_attributes=(), edge_codes={}, max_wl_depth=1
    )
    stripped = NodeBatch(**{**b.__dict__, "edge_attrs": b.edge_attrs[:, :0]})
    labels = refine(stripped, empty)[-1].labels
    assert labels[1] == labels[4]
    assert labels[0] == labels[3]  # endpoints too


def test_edge_alphabet_is_config_determined_not_batch_determined():
    """The collapse must be a pure function of the config. Deriving it from
    the values present in the batch (dense_rows over edge_attrs, say) would
    renumber the alphabet whenever a batch is missing a bond type, and every
    class id would silently change meaning between fit and predict.

    Fit on a corpus containing both bond types, then predict on the subset
    that contains only one. Under a batch-determined collapse the subset's
    codes shift and the nodes stop matching at the deepest level.
    """
    import sieve

    cfg2 = two_bond_config()
    # Graph 0: SINGLE bonds. Graph 1: DOUBLE bonds. Same topology.
    b = two_paths(conj_a=0, conj_b=0)
    attrs = b.edge_attrs.copy()
    attrs[4:, 0] = 1  # graph 1's bonds become DOUBLE
    full = NodeBatch(
        **{**b.__dict__, "edge_attrs": attrs, "y": np.arange(6.0).reshape(-1, 1)}
    )
    model = sieve.fit(full, cfg2)

    only_double = full[np.array([3, 4, 5])]
    p = sieve.predict_detailed(model, only_double)
    assert np.all(p.matched_level == cfg2.n_levels - 1)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_refine.py -k "edge or column or second or empty" -v
```

Expected: `test_a_second_edge_attribute_refines_the_partition` FAILS with `labels[1] == labels[4]` (column 1 is ignored today), and `test_column_count_must_match_the_declared_schema` FAILS because no such check exists.

- [ ] **Step 3: Implement the collapse**

In `src/sieve/refine.py`, replace lines 84-98 with:

```python
    n_edge_types = config.n_edge_types
    radices = config.edge_radices
    if csr.attr.shape[1] != len(radices):
        raise ValueError(
            f"batch has {csr.attr.shape[1]} edge attribute columns, but config "
            f"declares {len(radices)}: {list(config.edge_attributes)}"
        )
    kinds = config.level_kinds[len(levels) :]
    parents = config.level_parents[len(levels) :]
    neighbor_src = config.neighbor_source[len(levels) :]
    if kinds and csr.attr.shape[0]:
        # Each column must stay inside its own radix. A code outside it makes
        # the mixed-radix fold below collide with a *different* combination
        # instead of raising, silently conflating two distinct classes.
        for j, (name, radix) in enumerate(
            zip(config.edge_attributes, radices, strict=True)
        ):
            col = csr.attr[:, j]
            bad = (col < 0) | (col >= radix)
            if bad.any():
                raise ValueError(
                    f"edge_attrs column {j} ({name!r}) contains code "
                    f"{int(col[bad][0])}, outside [0, {radix}) implied by "
                    f"config.edge_codes[{name!r}]"
                )
    # Collapse the per-attribute columns into one integer per edge, mixed
    # radix. Derived from the *config*, never from the values present in this
    # batch: dense_rows here would renumber the alphabet whenever a batch is
    # missing a value, so a model's class ids would mean one thing at fit time
    # and another at predict time (spec 2026-09-02, section 3).
    edge_code = np.zeros(csr.attr.shape[0], np.int64)
    for j, radix in enumerate(radices):
        edge_code = edge_code * radix + csr.attr[:, j]
```

Then replace the provisional line 105 with:

```python
            pair = base[csr.dst] * n_edge_types + edge_code
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_refine.py -v
```

Expected: all PASS, including the pre-existing `test_edge_attr_outside_the_bond_alphabet_is_rejected` (its `match="edge_attr"` is still a substring of the new `edge_attrs column 0 (...)` message).

- [ ] **Step 5: Full suite, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: **314 passed, 1 skipped** (309 + 5).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: collapse edge attribute columns with a config-driven mixed radix

refine folds edge_attrs' columns into the single integer the pair encoding
needs, using radices taken from the config rather than from the batch.
dense_rows over edge_attrs would have been the obvious spelling and is
wrong: it numbers from the values present in the batch it is given, so a
predict batch missing a bond type would renumber the alphabet and every
class id would silently change meaning.

The per-level hot path is unchanged. The range check is now per column and
names the offending attribute. A batch whose column count disagrees with
edge_attributes raises instead of being read as a prefix."
```

---

## Task 4: `_BOND_ATTRS` registry and corpus discovery

**Files:**
- Modify: `src/sieve/io/rdkit_adapter.py` — replace `_BOND_ATTRS_PROVISIONAL`, add `_observed_edge_values`, extend `_discover_codes_chunk` and `build_codes`
- Modify: `charge_experiments/charge_experiments/predictors/sieve_predictor.py:85`
- Test: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: `SieveConfig.edge_attributes` / `edge_codes` (Task 2).
- Produces:
  - `_BOND_ATTRS: dict[str, Callable]` — `f(bond) -> str`. Entries: `bond_type`, `conjugated`.
  - `_MOL_BOND_ATTRS: dict[str, Callable]` — `f(mol) -> Sequence[str]`, one value per bond in RDKit bond-index order. Empty until Task 6.
  - `_observed_edge_values(name, mols) -> set[str]`
  - `build_codes(mols, attributes, *, edge_attributes=("bond_type",), n_jobs=None)` returning `(codes, edge_codes)` with `edge_codes` nested.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rdkit_adapter.py`:

```python
def test_edge_vocabularies_are_discovered_from_the_corpus():
    """Bond codes used to be a hardcoded four-entry table; anything else (a
    dative bond, say) collapsed onto code 0 with no signal. They are now
    discovered like node attributes: sorted observed values get 0..k-1, and k
    is reserved for the unseen."""
    from sieve.io.rdkit_adapter import build_codes

    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "C=C", "C#N"]]
    _, edge_codes = build_codes(mols, ["element"], edge_attributes=("bond_type",))
    assert set(edge_codes) == {"bond_type"}
    assert set(edge_codes["bond_type"]) == {"SINGLE", "DOUBLE", "TRIPLE"}
    assert sorted(edge_codes["bond_type"].values()) == [0, 1, 2]


def test_conjugated_is_an_edge_attribute():
    from sieve.io.rdkit_adapter import build_codes

    mols = [Chem.MolFromSmiles(s) for s in ["C=CC=C", "CCCC"]]
    _, edge_codes = build_codes(
        mols, ["element"], edge_attributes=("bond_type", "conjugated")
    )
    assert set(edge_codes["conjugated"]) == {"True", "False"}


def test_two_edge_attributes_reach_the_batch_as_separate_columns():
    from sieve.io.rdkit_adapter import build_codes

    smis = ["C=CC=C"]
    mols = [Chem.MolFromSmiles(s) for s in smis]
    codes, edge_codes = build_codes(
        mols, ["element"], edge_attributes=("bond_type", "conjugated")
    )
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edge_codes,
        edge_attributes=("bond_type", "conjugated"),
        max_wl_depth=2,
    )
    b = from_smiles(smis, config=cfg)
    assert b.edge_attrs.shape == (b.n_edges, 2)
    # butadiene: every bond is conjugated, bond orders are not all equal
    conj_col = b.edge_attrs[:, 1]
    assert len(set(conj_col.tolist())) == 1
    assert len(set(b.edge_attrs[:, 0].tolist())) == 2


def test_an_unseen_bond_type_gets_the_reserved_code():
    """A bond type absent from the fitting corpus must take the reserved code
    above the observed maximum, so it fails to match and backs off rather than
    colliding with a seen type."""
    from sieve.io.rdkit_adapter import build_codes

    codes, edge_codes = build_codes(
        [Chem.MolFromSmiles("CCO")], ["element"], edge_attributes=("bond_type",)
    )
    assert set(edge_codes["bond_type"]) == {"SINGLE"}
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edge_codes,
        edge_attributes=("bond_type",),
        max_wl_depth=1,
    )
    b = from_smiles(["C=C"], config=cfg)  # DOUBLE was never seen
    reserved = len(edge_codes["bond_type"])
    assert (b.edge_attrs[:, 0] == reserved).all()


@pytest.mark.parametrize("n_jobs", [None, 1, 2, 4])
def test_edge_vocabulary_discovery_is_deterministic_under_n_jobs(n_jobs):
    """Discovery is a set union -- commutative and associative -- and codes are
    assigned from the sorted union, so chunk count cannot affect numbering."""
    from sieve.io.rdkit_adapter import build_codes

    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "C=C", "C#N", "c1ccccc1"]]
    _, baseline = build_codes(mols, ["element"], edge_attributes=("bond_type",))
    _, got = build_codes(
        mols, ["element"], edge_attributes=("bond_type",), n_jobs=n_jobs
    )
    assert got == baseline
```

- [ ] **Step 2: Run them to verify they fail**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_rdkit_adapter.py -k "edge_vocab or conjugated or two_edge or unseen_bond" -v
```

Expected: FAIL — `build_codes() got an unexpected keyword argument 'edge_attributes'`.

- [ ] **Step 3: Add the registries**

In `src/sieve/io/rdkit_adapter.py`, delete `_BOND_ATTRS_PROVISIONAL` and add, immediately after the `_MOL_ATTRS` block:

```python
# Per-bond attribute providers: ``f(bond) -> str``. The edge-side counterpart
# of ``_ATTRS``, reached through ``SieveConfig.edge_attributes``. A separate
# namespace from the atom registries -- which is why the ring attributes below
# carry a ``bond_`` prefix.
_BOND_ATTRS = {
    "bond_type": lambda b: str(b.GetBondType()),
    "conjugated": lambda b: str(b.GetIsConjugated()),
}


# Molecule-level bond attribute providers: ``f(mol) -> Sequence[str]``, one
# value per bond in RDKit *bond*-index order (the atom-side ``_MOL_ATTRS`` uses
# atom-index order). Same rationale: whole-molecule precomputation runs once
# per ``Mol``. A name lives in exactly one of the two edge registries.
_MOL_BOND_ATTRS = {}
```

- [ ] **Step 4: Add edge value discovery**

Next to `_observed_values`:

```python
def _observed_edge_values(name: str, mols) -> set[str]:
    """The set of raw string values edge attribute ``name`` takes across
    ``mols`` -- the edge-side twin of ``_observed_values``."""
    if name in _MOL_BOND_ATTRS:
        f = _MOL_BOND_ATTRS[name]
        return {v for m in mols for v in f(m)}
    f = _BOND_ATTRS[name]
    return {f(b) for m in mols for b in m.GetBonds()}
```

- [ ] **Step 5: Extend the discovery worker and `build_codes`**

Replace `_discover_codes_chunk` with:

```python
def _discover_codes_chunk(
    blobs: list[bytes], attributes, edge_attributes
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Worker body for ``build_codes``: the per-attribute *set* of observed
    values for one chunk of molecules, node attributes and edge attributes
    separately (molecule-level providers run their own precompute -- CIP
    labeling, ring perception -- on the worker's copy).

    Returns raw sets, not code tables -- assigning dense integer codes needs
    the union across every chunk first, so numbering happens once in the
    parent after all chunks return (deterministic regardless of how many
    chunks there are or which finishes first).
    """
    mols = [_deserialize_mol(b) for b in blobs]
    return (
        {name: _observed_values(name, mols) for name in attributes},
        {name: _observed_edge_values(name, mols) for name in edge_attributes},
    )
```

Change `build_codes`'s signature and body:

```python
def build_codes(
    mols,
    attributes,
    *,
    edge_attributes=("bond_type",),
    n_jobs: int | None = None,
):
```

Add to its docstring, after the existing first paragraph:

```
    ``edge_attributes`` names the bond attributes to discover, resolved through
    ``_BOND_ATTRS``/``_MOL_BOND_ATTRS``. It may be empty, which yields an empty
    ``edge_codes`` and a pure-topology refinement. Edge vocabularies were
    hardcoded before 2026-09; discovering them means an unusual bond type gets
    the reserved unknown code instead of silently colliding on 0.
```

Replace the discovery block (lines 358-376) with:

```python
    edge_attributes = tuple(edge_attributes)
    if n_jobs is None or n_jobs == 1 or len(mols) < 2:
        seen_by_attr = {name: _observed_values(name, mols) for name in attributes}
        seen_by_edge = {
            name: _observed_edge_values(name, mols) for name in edge_attributes
        }
    else:
        n_chunks = 4 * effective_n_jobs(n_jobs)
        bounds = _chunk_boundaries_by_atom_count(mols, n_chunks)
        blob_chunks = [[_serialize_mol(m) for m in mols[s:e]] for s, e in bounds]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_discover_codes_chunk)(blobs, attributes, edge_attributes)
            for blobs in blob_chunks
        )
        seen_by_attr = {
            name: set().union(*(r[0][name] for r in results)) for name in attributes
        }
        seen_by_edge = {
            name: set().union(*(r[1][name] for r in results))
            for name in edge_attributes
        }

    codes = {
        name: {v: i for i, v in enumerate(sorted(seen_by_attr[name]))}
        for name in attributes
    }
    edge_codes = {
        name: {v: i for i, v in enumerate(sorted(seen_by_edge[name]))}
        for name in edge_attributes
    }
    return codes, edge_codes
```

- [ ] **Step 6: Point the featurization loop at the real registry**

In `_from_rdkit_sequential`, replace the provisional getter list with a resolver that mirrors the atom-side `atom_cols`/`mol_cols` split:

```python
    edge_tables = [dict(config.edge_codes[name]) for name in config.edge_attributes]
    edge_unknowns = [len(t) for t in edge_tables]
    bond_getters = []  # (column index, f(bond) -> str)
    mol_bond_cols = []  # (column index, f(mol) -> Sequence[str])
    for k, name in enumerate(config.edge_attributes):
        mol_f = _MOL_BOND_ATTRS.get(name)
        bond_f = _BOND_ATTRS.get(name)
        if mol_f is not None:
            mol_bond_cols.append((k, mol_f))
        elif bond_f is not None:
            bond_getters.append((k, bond_f))
        else:
            raise ValueError(f"unknown edge attribute {name!r}")
```

Inside the per-molecule loop, next to `mol_vals`:

```python
        mol_bond_vals = {k: f(mol) for k, f in mol_bond_cols}
```

Replace the bond loop:

```python
        for bi, b in enumerate(mol.GetBonds()):
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            row = [0] * len(config.edge_attributes)
            for k, bond_f in bond_getters:
                row[k] = edge_tables[k].get(bond_f(b), edge_unknowns[k])
            for k, vals in mol_bond_vals.items():
                row[k] = edge_tables[k].get(vals[bi], edge_unknowns[k])
            src += [u, v]
            dst += [v, u]
            attr += [row, row]
```

Note `bi` is the RDKit bond index and `mol.GetBonds()` yields in bond-index order, which is what `_MOL_BOND_ATTRS` providers key on.

- [ ] **Step 7: Update the downstream caller**

In `charge_experiments/charge_experiments/predictors/sieve_predictor.py`, the `build_codes` call at line 85 already receives the nested `edge_codes` and passes it straight through, so no change is required — but confirm:

```bash
grep -n "build_codes\|edge_codes\|edge_attributes" charge_experiments/charge_experiments/predictors/sieve_predictor.py
```

If `SieveConfig(...)` there does not pass `edge_attributes`, it defaults to `("bond_type",)`, which matches `build_codes`' default. Leave it.

- [ ] **Step 8: Run tests, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: **322 passed, 1 skipped** (314 + 4 + a 4-way parametrized test).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: _BOND_ATTRS registry and corpus-discovered edge vocabularies

Edge attributes resolve through _BOND_ATTRS (per-bond) and _MOL_BOND_ATTRS
(whole-molecule), the edge-side twins of _ATTRS/_MOL_ATTRS, and build_codes
discovers their vocabularies from the corpus the way node attributes are
discovered -- sorted union, dense codes from 0, one reserved code for the
unseen, unaffected by n_jobs chunking.

This retires the hardcoded four-entry bond table, whose fallback mapped any
other bond type (DATIVE, UNSPECIFIED) onto code 0 with no signal that it had
happened. First new attribute: conjugated."
```

---

## Task 5: End-to-end empty edge schema

`refine` already handles `edge_attributes=()` after Task 3, and `build_codes` after Task 4. This task proves the whole path — adapter through fit through predict — and pins it.

**Files:**
- Test: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4. Produces no new names.

- [ ] **Step 1: Write the failing test**

```python
def test_empty_edge_schema_end_to_end_refines_on_topology_alone():
    """edge_attributes=() is the pure-topology control arm. The adapter must
    emit an (n_edges, 0) array, and fit/predict must run without touching a
    bond attribute anywhere. Butadiene and butane have identical topology, so
    with no edge attributes their atoms must fall into the same classes --
    and differ once bond_type is declared."""
    from sieve.io.rdkit_adapter import build_codes

    smis = ["C=CC=C", "CCCC"]
    mols = [Chem.MolFromSmiles(s) for s in smis]

    codes, edge_codes = build_codes(mols, ["element"], edge_attributes=())
    assert edge_codes == {}
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edge_codes,
        edge_attributes=(),
        max_wl_depth=2,
    )
    b = from_smiles(smis, config=cfg)
    assert b.edge_attrs.shape == (b.n_edges, 0)

    from sieve.refine import refine

    labels = refine(b, cfg)[-1].labels
    # atom i of butadiene and atom i of butane are topologically identical
    assert labels[0] == labels[4]
    assert labels[1] == labels[5]
```

- [ ] **Step 2: Run it**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_rdkit_adapter.py::test_empty_edge_schema_end_to_end_refines_on_topology_alone -v
```

Expected: PASS if Tasks 1–4 are correct. **If it fails, do not weaken the test** — the failure is a real gap in the empty-schema path (most likely `np.array(attr).reshape` producing the wrong shape when `attr` rows are empty lists).

- [ ] **Step 3: Fix the adapter's empty-schema reshape if needed**

`np.array([[], []], np.int64)` already has shape `(2, 0)`, but `np.array([], np.int64)` (zero edges *and* zero columns) has shape `(0,)`. Make the construction explicit:

```python
        edge_attrs=np.array(attr, np.int64).reshape(
            len(attr), len(config.edge_attributes)
        ),
```

- [ ] **Step 4: Run the full suite, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: **323 passed, 1 skipped** (322 + 1).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: pin the empty edge schema end to end

edge_attributes=() runs adapter through fit through refine with an
(n_edges, 0) array and no bond attribute touched anywhere. Butadiene and
butane, topologically identical, fall into the same classes -- the
pure-topology control arm working as designed."
```

---

## Task 6: Molecule-level bond ring attributes

**Files:**
- Modify: `src/sieve/io/rdkit_adapter.py` — three providers + `_MOL_BOND_ATTRS` entries
- Test: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: `_MOL_BOND_ATTRS` (Task 4).
- Produces: edge attributes `bond_in_ring`, `bond_min_ring_size`, `bond_num_ring_memberships`.

- [ ] **Step 1: Write the failing test**

```python
def test_bond_ring_attributes():
    """The three edge-side ring attributes, on indane (fused cyclopentane +
    benzene): the fusion bond sits in two rings and reports the smaller one,
    an acyclic bond reports the sentinels. Sentinels follow the atom-side
    convention -- "none" for a ring *size* that does not exist, "0" for a
    count, which is a meaningful number."""
    from sieve.io.rdkit_adapter import build_codes

    # Indane (fused cyclopentane + benzene) with a methyl, which supplies the
    # acyclic bond. Verified to parse -- putting the methyl on a fusion carbon
    # instead ("C1CCc2ccccc21C") makes it 5-valent and RDKit returns None.
    smis = ["CC1CCc2ccccc21"]
    mols = [Chem.MolFromSmiles(s) for s in smis]
    attrs = ("bond_in_ring", "bond_min_ring_size", "bond_num_ring_memberships")
    codes, edge_codes = build_codes(mols, ["element"], edge_attributes=attrs)

    assert set(edge_codes["bond_in_ring"]) == {"True", "False"}
    assert "none" in edge_codes["bond_min_ring_size"]
    assert {"5", "6"} <= set(edge_codes["bond_min_ring_size"])
    assert {"0", "1", "2"} <= set(edge_codes["bond_num_ring_memberships"])

    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edge_codes,
        edge_attributes=attrs,
        max_wl_depth=1,
    )
    b = from_smiles(smis, config=cfg)
    mol = mols[0]

    def bond_row(pred):
        """First batch row for a bond satisfying `pred`. The adapter emits two
        rows per bond (both directions), in bond-index order."""
        for bi, bond in enumerate(mol.GetBonds()):
            if pred(bond):
                return b.edge_attrs[2 * bi]
        raise AssertionError("no bond matched")

    value_of = [
        {v: k for k, v in edge_codes[name].items()} for name in attrs
    ]
    # Membership in both ring sizes identifies the fusion bond uniquely. A
    # degree-based predicate does not: the methyl-bearing ring carbon also has
    # degree 3, so its bond to the fusion carbon would match too -- and that
    # bond is in one ring, not two.
    fusion = bond_row(lambda bd: bd.IsInRingSize(5) and bd.IsInRingSize(6))
    assert value_of[0][fusion[0]] == "True"
    assert value_of[1][fusion[1]] == "5"
    assert value_of[2][fusion[2]] == "2"

    acyclic = bond_row(lambda bd: not bd.IsInRing())
    assert value_of[0][acyclic[0]] == "False"
    assert value_of[1][acyclic[1]] == "none"
    assert value_of[2][acyclic[2]] == "0"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_rdkit_adapter.py::test_bond_ring_attributes -v
```

Expected: FAIL with `KeyError: 'bond_in_ring'` from `_observed_edge_values`.

- [ ] **Step 3: Implement the providers**

Add above the `_MOL_BOND_ATTRS` dict in `src/sieve/io/rdkit_adapter.py`:

```python
def _bond_ring_info(mol):
    """``RingInfo`` after SSSR perception. RDKit caches the result on the
    ``Mol``, so the three providers below and the atom-side ring attributes
    share one perception per molecule."""
    from rdkit import Chem

    Chem.GetSymmSSSR(mol)
    return mol.GetRingInfo()


def _bond_in_ring(mol) -> list[str]:
    """Whether each bond lies in any SSSR ring. Exactly derivable from either
    provider below -- kept because it is the cheapest of the three, but a
    config should declare one ring attribute, not several: they multiply the
    edge alphabet without adding information."""
    ri = _bond_ring_info(mol)
    return [str(bool(ri.NumBondRings(i))) for i in range(mol.GetNumBonds())]


def _bond_min_ring_size(mol) -> list[str]:
    """Size of the smallest SSSR ring each bond belongs to, ``"none"`` when
    the bond is acyclic -- a ring size of zero is meaningless, so this takes
    the sentinel rather than "0" (the atom-side ``min_ring_size`` convention).
    """
    ri = _bond_ring_info(mol)
    return [
        str(ri.MinBondRingSize(i)) if ri.NumBondRings(i) else "none"
        for i in range(mol.GetNumBonds())
    ]


def _bond_num_ring_memberships(mol) -> list[str]:
    """Number of SSSR rings each bond belongs to -- ``"2"`` or more marks a
    ring-fusion bond. Acyclic bonds get ``"0"``: a count of zero is meaningful,
    unlike a ring size of zero (the atom-side ``num_ring_memberships``
    convention)."""
    ri = _bond_ring_info(mol)
    return [str(ri.NumBondRings(i)) for i in range(mol.GetNumBonds())]
```

Fill the registry:

```python
_MOL_BOND_ATTRS = {
    "bond_in_ring": _bond_in_ring,
    "bond_min_ring_size": _bond_min_ring_size,
    "bond_num_ring_memberships": _bond_num_ring_memberships,
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest tests/test_rdkit_adapter.py::test_bond_ring_attributes -v
```

Expected: PASS.

- [ ] **Step 5: Full suite, lint, format, type-check**

```bash
export PATH="/home/craabreu/miniforge3/bin:$PATH"
uv run pytest -q
uv run ruff check src tests charge_experiments
uv run ruff format --check src tests charge_experiments
uv run ty check src tests charge_experiments
```

Expected: **324 passed, 1 skipped** (323 + 1).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: bond_in_ring, bond_min_ring_size, bond_num_ring_memberships

Three _MOL_BOND_ATTRS providers sharing the SSSR perception the atom-side
ring attributes already trigger. Sentinels follow the atom-side convention:
\"none\" for a ring size that does not exist, \"0\" for a count.

Prefixed because the edge registries are a separate namespace from the atom
ones -- bond_min_ring_size and min_ring_size are different attributes, and
the prefix keeps a config line and a grep unambiguous."
```

---

## Task 7: Reconcile `design.md`

`design.md` describes this interface as though it already existed. Bring it in line and record the decisions.

**Files:**
- Modify: `design.md` §9.2 (line ~945), §10.1 (line ~996), §11.1 (line ~1078), §7.2 (line ~772)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update §9.2's config JSON**

At line ~945, replace `"edge_attributes": ["bond_type"],` with both fields as they now exist:

```json
  "edge_attributes": ["bond_type"],
  "edge_codes": {"bond_type": {"SINGLE": 0, "DOUBLE": 1}},
```

and extend the `schema_version` bullet to name `edge_attributes` explicitly:

```
- **`schema_version`** is a digest over everything that affects what a class *means* — the attribute
  levels and their order, the edge attributes **and their order**, every code table, the neighbor
  schema, depth.
```

- [ ] **Step 2: Update §10.1's `SieveConfig`**

At line ~996, make the dataclass match the implementation:

```python
@dataclass(frozen=True)
class SieveConfig:
    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]   # graded order, §3.5
    attribute_codes: Mapping[str, Mapping[str, int]]
    edge_codes: Mapping[str, Mapping[str, int]]     # name -> {value: code}
    max_wl_depth: int
    edge_attributes: tuple[str, ...] = ("bond_type",)   # flat, may be empty
    neighbor_depth: int | None = None               # §3.6; None = no coarsening
    minimum_support: int = 1
    shrinkage_strength: float | None = None          # None = raw means, §4.2
    chunk_size: int | None = None                   # §4.1; None = whole corpus
```

- [ ] **Step 3: Update §11.1's `NodeBatch`**

At line ~1078, correct the field name (the shape was already right):

```python
    edge_attrs: np.ndarray     # (n_edges, n_edge_attr) int64
```

Add after the "Edges are stored **both directions**" sentence:

```
`edge_attrs` carries one column per configured edge attribute, in
`edge_attributes` order. A zero-width schema — `edge_attributes = ()` — is legal and
gives an `(n_edges, 0)` array, which makes refinement depend on topology alone.
```

- [ ] **Step 4: Extend §7.2 with the collapse**

After the `pair = labels[dst] * n_edge_types + bond` code block (line ~772), add:

```
**Where `bond` comes from.** Edge attributes are a *flat* set — unlike the graded
`attribute_levels`, every one of them contributes at full resolution at every level. The
`(n_edges, n_edge_attr)` row collapses to the single `bond` code above by mixed radix, each
attribute contributing its vocabulary plus one reserved unknown, so `n_edge_types` is the
product of those radices (and 1 for an empty schema, degenerating `pair` to the neighbor label).

The radices come from the **config**, never from the batch. Numbering them from the values
present in a given batch — `dense_rows` over `edge_attrs`, the obvious spelling — would renumber
the alphabet whenever a batch happened to be missing a value, so a model's class ids would mean
one thing at fit time and another at predict time. This is the same hazard `schema_version` (§9.2)
guards at the config level, arriving through the batch instead.
```

- [ ] **Step 5: Check for remaining drift**

```bash
grep -n "edge_attr\b\|edge_codes\|n_edge_types\|edge_attributes" design.md
```

Read each hit and confirm it matches the implementation. §3.6's "Edge attributes stay at full resolution in both arms" (line ~267) is still true and needs no change.

- [ ] **Step 6: Commit**

```bash
git add design.md
git commit -m "docs(design): reconcile edge attribute schema with the implementation

9.2, 10.1 and 11.1 described edge_attributes and a 2-D edge_attrs as though
they existed; they now match what is built. 7.2 gains the mixed-radix
collapse and the reason its radices come from the config rather than the
batch."
```

---

## Verification Checklist

Run before declaring the plan complete:

- [ ] `uv run pytest -q` → 324 passed, 1 skipped (baseline 302 + 22 new). Counts are a sanity check, not a contract: if yours differs, reconcile it against the tests each task adds before assuming something is broken.
- [ ] `uv run ruff check src tests charge_experiments` → clean
- [ ] `uv run ruff format --check src tests charge_experiments` → clean
- [ ] `uv run ty check src tests charge_experiments` → clean
- [ ] `grep -rn "edge_attr\b" --include=*.py . | grep -v .venv` → no hits
- [ ] `git diff main --stat` reviewed — `src/sieve/predict.py` and `src/sieve/merge.py` are **not** in it
- [ ] Every spec section 1–8 maps to a task (1→4,6; 2→2; 3→3; 4→1; 5→4; 6→2; 7→1,3,4,5,6; 8→7)
