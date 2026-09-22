# Cis/Trans Featurisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Sieve's refinement a radius-honest cis/trans code, so a double bond's geometry separates classes without any non-local information entering a label.

**Architecture:** The adapter extracts a fixed geometric relation per stereogenic double bond into a new optional `NodeBatch` array. A new module computes a stereo-blind content-rank fingerprint per radius. At each WL round `refine` uses the level-(k−2) fingerprint to decide which substituent wins at each end, turns that into a ternary code, and folds it into the existing `(neighbor label, bond)` pair encoding. Nothing is stored on a signature row, so `merge.py`, `dedupe.py` and `predict.py` are untouched.

**Tech Stack:** Python 3.11+, NumPy, RDKit, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-cis-trans-featurisation-design.md`

## Global Constraints

- The fingerprint folds the **static** `edge_code` only, never the stereo trit. A stereo-aware ordering breaks the merge (spec §5.1).
- No code is emitted before WL round *k* = 2 (spec §5.3).
- Read the relation from `Chem.FindPotentialStereo`, never `bond.GetStereo()` (spec §4).
- `n_edge_types` must be one constant across every level and every shard, or `merge._translate`'s `divmod` breaks (spec §6).
- With `stereo=()` the `schema_version` digest must be **byte-identical** to today's, so models on disk stay valid. The payload key is added conditionally (spec §6).
- Run tests with `.venv/bin/python -m pytest`. The repo root is the working directory.

---

## File Structure

| file | responsibility |
|---|---|
| `src/sieve/config.py` (modify) | the `stereo` field, `stereo_radices`, `n_edge_types`, the digest |
| `src/sieve/batch.py` (modify) | the `stereo_bonds` array and its validation |
| `src/sieve/stereo.py` (create) | content-rank fingerprints and code resolution — all the new logic |
| `src/sieve/refine.py` (modify) | wiring only: call into `stereo.py`, fold the trit |
| `src/sieve/io/rdkit_adapter.py` (modify) | extract the relation from RDKit |
| `tests/test_stereo.py` (create) | unit tests for `stereo.py` |
| `tests/test_config.py`, `test_batch.py`, `test_rdkit_adapter.py`, `test_refine.py`, `test_merge.py` (modify) | each layer's own tests |

`stereo.py` is a separate module rather than more code in `refine.py` because `refine.py` is currently one readable function and the fingerprint plus resolution is about 80 lines that have their own tests.

---

### Task 1: Config field, radix, and digest

**Files:**
- Modify: `src/sieve/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SieveConfig.stereo: tuple[str, ...]` (default `()`), `SieveConfig.stereo_radices: tuple[int, ...]`, module constants `STEREO_TRACKS: tuple[str, ...]` and `STEREO_RADIX: int = 4`. `n_edge_types` keeps its name and meaning.

- [x] **Step 1: Capture today's digest so the compatibility test is real**

Run this first and paste the literal into the test below:

```bash
.venv/bin/python -c "
from sieve.config import SieveConfig
cfg = SieveConfig(target_dim=1, attribute_levels=(('element',),),
                  attribute_codes={'element': {'C': 0, 'H': 1}},
                  edge_codes={'bond_type': {'SINGLE': 0, 'DOUBLE': 1}}, max_wl_depth=3)
print(cfg.schema_version)"
```

- [x] **Step 2: Write the failing tests**

```python
def _ref_config(**kw):
    from sieve.config import SieveConfig
    base = dict(target_dim=1, attribute_levels=(("element",),),
                attribute_codes={"element": {"C": 0, "H": 1}},
                edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}}, max_wl_depth=3)
    base.update(kw)
    return SieveConfig(**base)


def test_stereo_defaults_off_and_leaves_the_digest_untouched():
    # Paste the literal from Step 1. A fitted model on disk carries this
    # digest; changing it silently invalidates every one of them.
    assert _ref_config().schema_version == "PASTE_DIGEST_FROM_STEP_1"
    assert _ref_config().stereo == ()
    assert _ref_config().stereo_radices == ()


def test_enabling_a_stereo_track_changes_the_digest():
    off, on = _ref_config(), _ref_config(stereo=("cis_trans",))
    assert on.schema_version != off.schema_version


def test_a_stereo_track_widens_the_edge_alphabet_by_four():
    off, on = _ref_config(), _ref_config(stereo=("cis_trans",))
    assert on.n_edge_types == off.n_edge_types * 4
    assert on.stereo_radices == (4,)


def test_an_unknown_stereo_track_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="unknown stereo track"):
        _ref_config(stereo=("helical",))
```

- [x] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -k stereo -v`
Expected: FAIL — `SieveConfig.__init__() got an unexpected keyword argument 'stereo'`

- [x] **Step 4: Implement**

In `src/sieve/config.py`, beside the other module constants:

```python
STEREO_RADIX = 4  # none, cis, trans, and the reserved unknown
STEREO_TRACKS = ("cis_trans",)
```

Add the field to `SieveConfig`, after `neighbor_depth` so it precedes the
prediction-time settings:

```python
    stereo: tuple[str, ...] = ()
```

In `__post_init__`, beside the other validation:

```python
        for track in self.stereo:
            if track not in STEREO_TRACKS:
                raise ValueError(
                    f"unknown stereo track {track!r}; known: {list(STEREO_TRACKS)}"
                )
```

Add the property beside `edge_radices`:

```python
    @property
    def stereo_radices(self) -> tuple[int, ...]:
        """One radix per enabled stereo track, each a code alphabet of
        ``{none, cis, trans}`` plus the reserved unknown.

        Kept separate from ``edge_radices``, which describes the *static*
        adapter columns: a stereo code is recomputed every round and has no
        column in ``edge_attrs``.
        """
        return tuple(STEREO_RADIX for _ in self.stereo)
```

Change `n_edge_types` to include them:

```python
        return math.prod(self.edge_radices) * math.prod(self.stereo_radices)
```

In `schema_version`, add the key **conditionally**, after the `edge_codes` entry:

```python
        if self.stereo:
            # Added only when non-empty so that every digest computed before
            # stereo existed stays byte-identical, and the models carrying it
            # stay valid (design.md 9.2).
            payload["stereo"] = list(self.stereo)
```

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS, including the pre-existing tests.

- [x] **Step 6: Commit**

```bash
git add src/sieve/config.py tests/test_config.py
git commit -m "feat(config): a stereo track widens the edge alphabet and the digest"
```

---

### Task 2: `NodeBatch.stereo_bonds` and its validation

**Files:**
- Modify: `src/sieve/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `NodeBatch.stereo_bonds: np.ndarray | None = None`, shape `(n_stereo, 7)` int64, columns `[a, b, a1, a2, b1, b2, cis]`. `-1` permitted only in `a2`/`b2`. Validated in `__post_init__` by `_check_stereo_bonds`.

- [x] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from sieve.batch import NodeBatch


def _ethene_batch(stereo_bonds=None):
    """C1=C2 with one substituent on each end: atoms 0-1 sp2, 2 on 0, 3 on 1."""
    src = np.array([0, 1, 0, 2, 1, 3], np.int64)
    dst = np.array([1, 0, 2, 0, 3, 1], np.int64)
    return NodeBatch(
        node_attrs=np.zeros((4, 1), np.int64),
        edge_src=src, edge_dst=dst,
        edge_attrs=np.zeros((6, 1), np.int64),
        graph_id=np.zeros(4, np.int64),
        stereo_bonds=stereo_bonds,
    )


def test_stereo_bonds_defaults_to_none_and_changes_nothing():
    assert _ethene_batch().stereo_bonds is None


def test_a_well_formed_stereo_bond_is_accepted():
    rows = np.array([[0, 1, 2, -1, 3, -1, 1]], np.int64)
    assert _ethene_batch(rows).stereo_bonds.shape == (1, 7)


def test_stereo_bonds_rejects_a_wrong_width():
    with pytest.raises(ValueError, match="must have shape"):
        _ethene_batch(np.zeros((1, 5), np.int64))


def test_stereo_bonds_rejects_an_out_of_range_atom():
    rows = np.array([[0, 1, 2, -1, 99, -1, 1]], np.int64)
    with pytest.raises(ValueError, match="out of range"):
        _ethene_batch(rows)


def test_stereo_bonds_rejects_a_pair_that_is_not_an_edge():
    # 2-3 are not bonded to each other.
    rows = np.array([[2, 3, 0, -1, 1, -1, 1]], np.int64)
    with pytest.raises(ValueError, match="not an edge"):
        _ethene_batch(rows)


def test_stereo_bonds_rejects_a_controlling_atom_that_is_not_adjacent():
    # atom 3 hangs off atom 1, so it cannot control end 0.
    rows = np.array([[0, 1, 3, -1, 3, -1, 1]], np.int64)
    with pytest.raises(ValueError, match="not adjacent"):
        _ethene_batch(rows)


def test_stereo_bonds_rejects_a_missing_first_controlling_atom():
    rows = np.array([[0, 1, -1, 2, 3, -1, 1]], np.int64)
    with pytest.raises(ValueError, match="only in the second slot"):
        _ethene_batch(rows)


def test_stereo_bonds_rejects_a_non_boolean_relation():
    rows = np.array([[0, 1, 2, -1, 3, -1, 7]], np.int64)
    with pytest.raises(ValueError, match="cis column"):
        _ethene_batch(rows)
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_batch.py -k stereo -v`
Expected: FAIL — `NodeBatch.__init__() got an unexpected keyword argument 'stereo_bonds'`

- [x] **Step 3: Implement**

Add the field to `NodeBatch`, after `elements`:

```python
    stereo_bonds: np.ndarray | None = None  # (n_stereo, 7) int64
```

Call the check from `__post_init__`, after `_check_edges()`:

```python
        self._check_stereo_bonds()
```

And add the method:

```python
    def _check_stereo_bonds(self) -> None:
        """Validate the stereogenic-double-bond table, if present.

        Columns are ``[a, b, a1, a2, b1, b2, cis]``: the two sp2 atoms, each
        end's controlling substituents, and whether ``a1`` and ``b1`` lie on
        the same side. Every check here is a corpus bug that would otherwise
        produce a plausible *wrong* code rather than an error -- the code is
        consumed into a class label, so nothing downstream can notice.
        """
        sb = self.stereo_bonds
        if sb is None:
            return
        if sb.ndim != 2 or sb.shape[1] != 7:
            raise ValueError(
                f"stereo_bonds must have shape (n_stereo, 7), got {sb.shape}"
            )
        n = self.node_attrs.shape[0]
        a, b, a1, a2, b1, b2, cis = (sb[:, j] for j in range(7))
        if ((a1 < 0) | (b1 < 0)).any():
            raise ValueError(
                "stereo_bonds: -1 marks an absent substituent and is allowed "
                "only in the second slot of each end (columns a2, b2)"
            )
        for name, col in (("a", a), ("b", b), ("a1", a1), ("b1", b1)):
            if ((col < 0) | (col >= n)).any():
                raise ValueError(f"stereo_bonds column {name} is out of range [0, {n})")
        for name, col in (("a2", a2), ("b2", b2)):
            bad = (col < -1) | (col >= n)
            if bad.any():
                raise ValueError(f"stereo_bonds column {name} is out of range [-1, {n})")
        if ((cis < 0) | (cis > 1)).any():
            raise ValueError("stereo_bonds: the cis column must be 0 or 1")

        # Adjacency, vectorized: one sorted key per directed edge.
        key = self.edge_src * n + self.edge_dst
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]

        def present(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            want = u * n + v
            pos = np.searchsorted(sorted_key, want)
            pos = np.clip(pos, 0, sorted_key.shape[0] - 1)
            return sorted_key[pos] == want

        if not present(a, b).all() or not present(b, a).all():
            raise ValueError(
                "stereo_bonds: (a, b) is not an edge in both directions"
            )
        for end, sub, label in ((a, a1, "a1"), (b, b1, "b1")):
            if not present(end, sub).all():
                raise ValueError(f"stereo_bonds: {label} is not adjacent to its end")
        for end, sub, label in ((a, a2, "a2"), (b, b2, "b2")):
            have = sub >= 0
            if have.any() and not present(end[have], sub[have]).all():
                raise ValueError(f"stereo_bonds: {label} is not adjacent to its end")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_batch.py -v`
Expected: PASS, including the pre-existing tests.

- [x] **Step 5: Commit**

```bash
git add src/sieve/batch.py tests/test_batch.py
git commit -m "feat(batch): carry stereogenic double bonds, validated on construction"
```

---

### Task 3: Extract the relation in the adapter

**Files:**
- Modify: `src/sieve/io/rdkit_adapter.py`
- Test: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Consumes: `NodeBatch.stereo_bonds` from Task 2.
- Produces: `_stereo_bond_rows(mol) -> list[tuple[int, int, int, int, int, int, int]]`, atom indices local to `mol`. `from_rdkit` and `from_smiles` populate `stereo_bonds` when `config.stereo` is non-empty.

- [x] **Step 1: Write the failing tests**

```python
def test_stereo_rows_read_cis_trans_not_cip():
    """The store gives STEREOCIS/STEREOTRANS and SMILES gives STEREOE/STEREOZ,
    so reading bond.GetStereo() would import CIP on the from_smiles path.
    FindPotentialStereo gives the same local answer on both."""
    from rdkit import Chem
    from sieve.io.rdkit_adapter import _stereo_bond_rows

    trans = _stereo_bond_rows(Chem.MolFromSmiles("C/C=C/C"))
    cis = _stereo_bond_rows(Chem.MolFromSmiles(r"C/C=C\C"))
    assert len(trans) == 1 and len(cis) == 1
    assert trans[0][:6] == cis[0][:6]        # same atoms, same controlling pairs
    assert trans[0][6] != cis[0][6]          # opposite relation


def test_an_unspecified_double_bond_yields_no_row():
    from rdkit import Chem
    from sieve.io.rdkit_adapter import _stereo_bond_rows

    assert _stereo_bond_rows(Chem.MolFromSmiles("CC=CC")) == []


def test_a_non_stereogenic_double_bond_yields_no_row():
    from rdkit import Chem
    from sieve.io.rdkit_adapter import _stereo_bond_rows

    assert _stereo_bond_rows(Chem.MolFromSmiles("CC(C)=CC")) == []


def test_an_end_with_two_substituents_reports_both():
    from rdkit import Chem
    from sieve.io.rdkit_adapter import _stereo_bond_rows

    rows = _stereo_bond_rows(Chem.MolFromSmiles(r"C/C(F)=C(Cl)/C"))
    assert len(rows) == 1
    _, _, a1, a2, b1, b2, _ = rows[0]
    assert a2 >= 0 and b2 >= 0


def test_from_smiles_populates_stereo_bonds_only_when_configured():
    from dataclasses import replace
    from sieve.io.rdkit_adapter import from_smiles

    off = from_smiles("C/C=C/C", config=simple_config())
    on = from_smiles("C/C=C/C", config=replace(simple_config(), stereo=("cis_trans",)))
    assert off.stereo_bonds is None
    assert on.stereo_bonds.shape == (1, 7)
```

There is no pytest fixture: the suite uses `tests/helpers.py::simple_config(**kw)`,
which is `dataclasses.replace` over a base `SieveConfig`. So write
`simple_config()` for the stereo-blind config and
`simple_config(stereo=("cis_trans",))` for the enabled one, importing it with
`from tests.helpers import simple_config`.

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rdkit_adapter.py -k stereo -v`
Expected: FAIL — `ImportError: cannot import name '_stereo_bond_rows'`

- [x] **Step 3: Implement the extractor**

```python
_RDKIT_NO_ATOM = 0xFFFFFFFF  # what controllingAtoms uses for an absent slot


def _stereo_bond_rows(mol) -> list[tuple[int, int, int, int, int, int, int]]:
    """Stereogenic double bonds as ``[a, b, a1, a2, b1, b2, cis]``.

    Read through ``FindPotentialStereo`` rather than ``bond.GetStereo()``.
    The two are not interchangeable: a molecule parsed from SMILES carries
    ``STEREOE``/``STEREOZ``, which are CIP-derived and therefore depend on
    atoms arbitrarily far away, while the store carries the local
    ``STEREOCIS``/``STEREOTRANS``. ``FindPotentialStereo`` returns
    ``Bond_Cis``/``Bond_Trans`` relative to its own controlling atoms on both
    paths, which is the only form this featurization may read.

    Bonds reported ``Unspecified`` yield no row and therefore never receive a
    code. Ends carrying more than two substituents are skipped by RDKit's own
    representation, which names at most two per end.
    """
    from rdkit import Chem

    rows: list[tuple[int, int, int, int, int, int, int]] = []
    for element in Chem.FindPotentialStereo(mol):
        if element.type != Chem.StereoType.Bond_Double:
            continue
        if element.specified != Chem.StereoSpecified.Specified:
            continue
        controlling = [
            -1 if int(x) == _RDKIT_NO_ATOM else int(x)
            for x in element.controllingAtoms
        ]
        a1, a2, b1, b2 = controlling
        if a1 < 0 or b1 < 0:
            continue
        bond = mol.GetBondWithIdx(int(element.centeredOn))
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        cis = 1 if element.descriptor == Chem.StereoDescriptor.Bond_Cis else 0
        rows.append((int(a), int(b), a1, a2, b1, b2, cis))
    return rows
```

- [x] **Step 4: Populate the batch**

In `_from_rdkit_sequential`, alongside the per-molecule node offset that already
exists, accumulate the rows with that offset applied, and pass them to the
`NodeBatch` constructor. Offset every atom column but leave `-1` alone:

```python
    # stereo rows are collected only when a track is enabled; the offset
    # applies to the six atom columns and must not touch the -1 sentinels
    # or the cis flag.
    if config.stereo:
        for a, b, a1, a2, b1, b2, cis in _stereo_bond_rows(mol):
            shifted = [v + offset if v >= 0 else -1 for v in (a, b, a1, a2, b1, b2)]
            stereo_rows.append((*shifted, cis))
```

and at construction:

```python
        stereo_bonds=(
            np.array(stereo_rows, np.int64).reshape(-1, 7) if config.stereo else None
        ),
```

`reshape(-1, 7)` keeps the shape correct when a corpus has no stereogenic bond
at all, where `np.array([])` would otherwise be `(0,)` and fail Task 2's check.

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rdkit_adapter.py -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add src/sieve/io/rdkit_adapter.py tests/test_rdkit_adapter.py
git commit -m "feat(adapter): extract the local cis/trans relation per double bond"
```

---

### Task 4: Content-rank fingerprints

**Files:**
- Create: `src/sieve/stereo.py`
- Test: `tests/test_stereo.py`

**Interfaces:**
- Consumes: `CSRLayout` from `sieve.batch`, `_row_keys` from `sieve.dedupe`.
- Produces: `content_ranks(node_attrs: np.ndarray, csr: CSRLayout, edge_code: np.ndarray, n_rounds: int) -> list[np.ndarray]` returning `n_rounds + 1` arrays of `uint64`, index `j` being the radius-*j* fingerprint.

- [x] **Step 1: Write the failing tests**

```python
import numpy as np

from sieve.batch import NodeBatch
from sieve.stereo import content_ranks


def _chain(n_nodes, attrs):
    """A path graph 0-1-2-..., edges in both directions."""
    src = np.repeat(np.arange(n_nodes - 1), 2)
    dst = src.copy()
    src[0::2] = np.arange(n_nodes - 1); dst[0::2] = np.arange(1, n_nodes)
    src[1::2] = np.arange(1, n_nodes); dst[1::2] = np.arange(n_nodes - 1)
    return NodeBatch(
        node_attrs=np.array(attrs, np.int64).reshape(n_nodes, -1),
        edge_src=src, edge_dst=dst,
        edge_attrs=np.zeros((src.shape[0], 1), np.int64),
        graph_id=np.zeros(n_nodes, np.int64),
    )


def test_fingerprint_zero_separates_exactly_the_attribute_rows():
    batch = _chain(4, [[0], [1], [1], [0]])
    csr = batch.csr()
    fp = content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 0)
    assert len(fp) == 1
    assert fp[0][0] == fp[0][3] and fp[0][1] == fp[0][2]
    assert fp[0][0] != fp[0][1]


def test_fingerprints_grow_with_radius():
    # 0-1-2-3 with identical attributes: ends differ from the middle at r=1.
    batch = _chain(4, [[0], [0], [0], [0]])
    csr = batch.csr()
    fp = content_ranks(batch.node_attrs, csr, np.zeros(csr.dst.shape[0], np.int64), 2)
    assert len(fp) == 3
    assert fp[0][0] == fp[0][1]     # radius 0: all identical
    assert fp[1][0] != fp[1][1]     # radius 1: degree differs
    assert fp[1][0] == fp[1][3]     # radius 1: the two ends agree


def test_fingerprints_are_independent_of_batch_composition():
    """The whole point: an id is batch-local, a fingerprint is not."""
    alone = _chain(4, [[0], [1], [1], [0]])
    csr_a = alone.csr()
    fp_a = content_ranks(alone.node_attrs, csr_a,
                         np.zeros(csr_a.dst.shape[0], np.int64), 2)

    # The same path graph, with a second disconnected copy appended.
    src = np.concatenate([alone.edge_src, alone.edge_src + 4])
    dst = np.concatenate([alone.edge_dst, alone.edge_dst + 4])
    together = NodeBatch(
        node_attrs=np.concatenate([alone.node_attrs, alone.node_attrs]),
        edge_src=src, edge_dst=dst,
        edge_attrs=np.zeros((src.shape[0], 1), np.int64),
        graph_id=np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int64),
    )
    csr_t = together.csr()
    fp_t = content_ranks(together.node_attrs, csr_t,
                         np.zeros(csr_t.dst.shape[0], np.int64), 2)
    for j in range(3):
        assert np.array_equal(fp_a[j], fp_t[j][:4])
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_stereo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sieve.stereo'`

- [x] **Step 3: Implement**

Create `src/sieve/stereo.py`:

```python
"""Content-rank fingerprints and the cis/trans code built on them.

Kept out of ``refine.py`` because it has its own tests and ``refine`` is one
readable function; ``refine`` calls in and folds the result.
"""

from __future__ import annotations

import numpy as np

from sieve.batch import CSRLayout
from sieve.dedupe import _row_keys


def _mix(*columns: np.ndarray) -> np.ndarray:
    """Mix several int64/uint64 columns into one uint64 per row.

    Reuses ``dedupe._row_keys`` rather than a second mixer so the birthday
    argument ``dense_rows`` already makes for itself covers these keys too.
    """
    stacked = np.stack([c.view(np.int64) for c in columns], axis=1)
    return _row_keys(stacked)


def content_ranks(
    node_attrs: np.ndarray,
    csr: CSRLayout,
    edge_code: np.ndarray,
    n_rounds: int,
) -> list[np.ndarray]:
    """Radius-resolved content fingerprints, ``fp[j]`` for radius ``j``.

    ``fp_0`` is the attribute row; ``fp_j`` mixes ``fp_{j-1}`` of the atom with
    the sorted multiset of its neighbors' ``fp_{j-1}`` paired with the bond.
    A fingerprint is a function of the atom's radius-*j* environment and
    nothing else, so two atoms with the same environment share it in any
    batch, and adding molecules to the batch cannot change it. Class ids do
    not have that property: ``dense_rows`` numbers them by hashing the row and
    its docstring states that no caller may rely on that numbering.

    ``edge_code`` is the **static** edge alphabet, never a stereo code. A
    stereo-aware fingerprint would let the code reorder the substituents it is
    ranking, so remapping or mirroring could change which substituent wins and
    a merged model would disagree with a whole one.
    """
    n = node_attrs.shape[0]
    fp = [_row_keys(node_attrs)]
    width = max(int(csr.max_deg), 1)
    for _ in range(n_rounds):
        prev = fp[-1]
        nb = _mix(prev[csr.dst], edge_code)
        pad = np.zeros((n, width), np.uint64)
        pad[csr.src, csr.slot] = nb
        pad.sort(axis=1)
        fp.append(_mix(prev, *(pad[:, j] for j in range(width))))
    return fp
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_stereo.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/sieve/stereo.py tests/test_stereo.py
git commit -m "feat(stereo): content-rank fingerprints, invariant to batch composition"
```

---

### Task 5: Resolve the ternary code

**Files:**
- Modify: `src/sieve/stereo.py`
- Test: `tests/test_stereo.py`

**Interfaces:**
- Consumes: `content_ranks` from Task 4.
- Produces: `directed_positions(csr: CSRLayout, n_nodes: int, stereo_bonds: np.ndarray) -> tuple[np.ndarray, np.ndarray]` and `cis_trans_codes(stereo_bonds: np.ndarray, fp: np.ndarray) -> np.ndarray` returning one code per stereo bond in `{0, 1, 2}`.

- [x] **Step 1: Write the failing tests**

```python
from sieve.stereo import cis_trans_codes


def test_the_stored_relation_is_used_when_both_winners_are_the_first():
    rows = np.array([[0, 1, 2, -1, 3, -1, 1]], np.int64)   # cis, no ties possible
    fp = np.arange(4, dtype=np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [1]       # 1 == cis


def test_the_relation_flips_when_exactly_one_winner_differs():
    # End a has substituents 2 and 4; 4 outranks 2, so a's winner is not a1.
    rows = np.array([[0, 1, 2, 4, 3, -1, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 20], np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [2]       # 2 == trans


def test_the_relation_is_restored_when_both_winners_differ():
    rows = np.array([[0, 1, 2, 4, 3, 5, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 20, 30], np.uint64)
    assert cis_trans_codes(rows, fp).tolist() == [1]


def test_a_tie_at_either_end_defers():
    rows = np.array([[0, 1, 2, 4, 3, -1, 1]], np.int64)
    fp = np.array([0, 0, 10, 0, 10], np.uint64)            # 2 and 4 tie
    assert cis_trans_codes(rows, fp).tolist() == [0]       # 0 == none


def test_an_end_with_one_substituent_cannot_tie():
    rows = np.array([[0, 1, 2, -1, 3, -1, 0]], np.int64)
    fp = np.zeros(4, np.uint64)                            # everything ties
    assert cis_trans_codes(rows, fp).tolist() == [2]
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_stereo.py -k cis_trans -v`
Expected: FAIL — `ImportError: cannot import name 'cis_trans_codes'`

- [x] **Step 3: Implement**

Append to `src/sieve/stereo.py`:

```python
CODE_NONE, CODE_CIS, CODE_TRANS = 0, 1, 2


def cis_trans_codes(stereo_bonds: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """One code in ``{none, cis, trans}`` per stereogenic double bond.

    At each end the substituent with the larger fingerprint wins; equal
    fingerprints mean the bond is not distinguishable at this radius and the
    feature defers rather than guessing. The stored relation holds between the
    *first* controlling atom of each end, so it flips exactly when one winner
    -- and not both -- is the second.
    """
    a1, a2, b1, b2, cis = (stereo_bonds[:, j] for j in (2, 3, 4, 5, 6))
    zero = np.zeros(1, np.uint64)[0]

    def winner_is_first(first: np.ndarray, second: np.ndarray):
        has = second >= 0
        f1 = fp[first]
        f2 = np.where(has, fp[np.where(has, second, 0)], zero)
        return (~has) | (f1 > f2), has & (f1 == f2)

    a_first, a_tie = winner_is_first(a1, a2)
    b_first, b_tie = winner_is_first(b1, b2)
    flipped = a_first ^ b_first
    same_side = cis.astype(bool) ^ flipped
    return np.where(
        a_tie | b_tie, CODE_NONE, np.where(same_side, CODE_CIS, CODE_TRANS)
    ).astype(np.int64)


def directed_positions(
    csr: CSRLayout, n_nodes: int, stereo_bonds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Positions of each stereo bond's two directed halves in CSR order.

    Resolved once, before the round loop: the bond set never changes, only
    the code on it does. Positions index the CSR-ordered edge arrays, which
    is the order ``refine`` folds the codes into.
    """
    key = csr.src * n_nodes + csr.dst
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    def locate(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        want = u * n_nodes + v
        pos = np.searchsorted(sorted_key, want)
        if (pos >= sorted_key.shape[0]).any() or (sorted_key[pos] != want).any():
            raise ValueError("stereo_bonds names a pair that is not an edge")
        return order[pos]

    a, b = stereo_bonds[:, 0], stereo_bonds[:, 1]
    return locate(a, b), locate(b, a)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_stereo.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/sieve/stereo.py tests/test_stereo.py
git commit -m "feat(stereo): resolve the ternary code from a radius's fingerprints"
```

---

### Task 6: Wire it into the refinement

**Files:**
- Modify: `src/sieve/refine.py`
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no new public names. `refine(batch, config)` gains the behaviour.

- [x] **Step 1: Write the failing tests**

```python
def _butene_batch(cis: bool, config):
    from sieve.io.rdkit_adapter import from_smiles
    return from_smiles(r"C/C=C\C" if cis else "C/C=C/C", config=config)


def test_cis_and_trans_2_butene_separate_but_not_before_radius_two():
    """The radius rule, pinned. cis/trans is a fact about a four-atom span, so
    a radius-1 class asserting it would be naming an atom outside its own
    neighborhood."""
    from sieve.refine import refine

    cis = refine(_butene_batch(True, stereo_config), stereo_config())
    trans = refine(_butene_batch(False, stereo_config), stereo_config())
    assert sorted(cis[1].signatures.tolist()) == sorted(trans[1].signatures.tolist())
    assert sorted(cis[2].signatures.tolist()) != sorted(trans[2].signatures.tolist())


def test_a_non_stereogenic_double_bond_never_separates():
    from sieve.io.rdkit_adapter import from_smiles
    from sieve.refine import refine

    a = refine(from_smiles("CC(C)=CC", config=stereo_config()), stereo_config())
    b = refine(from_smiles("CC(C)=CC", config=stereo_config()), stereo_config())
    for la, lb in zip(a, b, strict=True):
        assert np.array_equal(la.signatures, lb.signatures)


def test_a_molecule_with_no_stereogenic_bond_is_unaffected_by_the_track():
    """Enabling the track must not perturb a molecule it has nothing to say
    about: same class count at every level, stereo on or off."""
    from sieve.io.rdkit_adapter import from_smiles
    from sieve.refine import refine
    from tests.helpers import simple_config

    off_cfg = simple_config(max_wl_depth=3)
    on_cfg = stereo_config()
    off = refine(from_smiles("CCCC", config=off_cfg), off_cfg)
    on = refine(from_smiles("CCCC", config=on_cfg), on_cfg)
    assert [lv.n_classes for lv in off] == [lv.n_classes for lv in on]


def test_a_configured_track_without_stereo_bonds_raises():
    import pytest
    from dataclasses import replace
    from sieve.io.rdkit_adapter import from_smiles
    from sieve.refine import refine

    batch = from_smiles("C/C=C/C", config=replace(stereo_config(), stereo=()))
    with pytest.raises(ValueError, match="config.stereo"):
        refine(batch, stereo_config())
```

There is no `stereo_config` fixture and none should be added. Use
`tests/helpers.py::simple_config`, whose base has `max_wl_depth=2` — too shallow
here, since the code first fires at *k* = 2. Write a module-level helper in
`tests/test_refine.py`:

```python
from tests.helpers import simple_config

def stereo_config(**kw):
    return simple_config(stereo=("cis_trans",), max_wl_depth=3, **kw)
```

and call `stereo_config()` in each test below, replacing the `stereo_config`
parameter in the signatures.

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_refine.py -k stereo -v`
Expected: FAIL — the cis and trans chains are identical at every level.

- [x] **Step 3: Implement**

In `src/sieve/refine.py`, import at the top:

```python
from sieve.config import LEVEL_WL, SieveConfig
from sieve.stereo import cis_trans_codes, content_ranks, directed_positions
```

After `edge_code` is built and before the `for offset, kind in enumerate(kinds)`
loop:

```python
    # --- stereo codes (spec 2026-09-22) ----------------------------------
    # Recomputed every round from the content rank, never stored: what
    # reaches a signature row is the pair encoding, which merge can remap.
    stereo_radix = math.prod(config.stereo_radices)
    fingerprints: list[np.ndarray] = []
    pos_ab = pos_ba = None
    if config.stereo:
        if batch.stereo_bonds is None:
            raise ValueError(
                f"config.stereo is {list(config.stereo)} but the batch carries "
                "no stereo_bonds; the adapter was run with a stereo-blind config"
            )
        n_wl = sum(1 for k in kinds if k == LEVEL_WL)
        fingerprints = content_ranks(batch.node_attrs, csr, edge_code, max(n_wl - 2, 0))
        pos_ab, pos_ba = directed_positions(csr, n, batch.stereo_bonds)
```

`math` needs importing at the top of the module if it is not already there.

Inside the loop, replace the `LEVEL_WL` body's first line:

```python
        if kind == LEVEL_WL:
            wl_round += 1
            full = edge_code
            if config.stereo:
                # The gather reaches distance 2, so an honest code needs
                # radius-(k-2) identities. At k = 1 there is no such radius:
                # the far substituent is two bonds away, outside a radius-1
                # neighborhood, so the feature stays silent rather than
                # asserting something the level cannot support.
                j = wl_round - 2
                stereo_code = np.zeros(edge_code.shape[0], np.int64)
                if j >= 0:
                    codes = cis_trans_codes(batch.stereo_bonds, fingerprints[j])
                    stereo_code[pos_ab] = codes
                    stereo_code[pos_ba] = codes
                full = edge_code * stereo_radix + stereo_code
            pair = base[csr.dst] * n_edge_types + full
```

Initialise `wl_round = 0` immediately before the loop.

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_refine.py -v`
Expected: PASS

- [x] **Step 5: Run the whole suite — this task touches the hot path**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, no regressions.

- [x] **Step 6: Commit**

```bash
git add src/sieve/refine.py tests/test_refine.py
git commit -m "feat(refine): fold a radius-honest cis/trans code into the pair encoding"
```

---

### Task 7: The merge monoid and the reflection control

**Files:**
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: the whole feature. Adds no production code; if a test here fails, the fix belongs in Tasks 4-6.

- [x] **Step 1: Write the failing tests**

```python
def test_a_stereo_model_merges_identically_to_a_whole_one():
    """The property the construction exists to preserve. Same shape as
    test_a_pickled_shard_merges_identically_to_a_local_one just above."""
    import dataclasses

    import numpy as np
    from rdkit import Chem

    import sieve
    from sieve.io.rdkit_adapter import from_rdkit

    cfg = stereo_config()
    smiles = ["C/C=C/C", r"C/C=C\\C", "C/C=C/CC", r"C/C=C\\CC",
              "C/C(F)=C(Cl)/C", r"C/C(F)=C(Cl)\\C", "CCCC", "CC(C)C"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rng = np.random.default_rng(0)
    b = from_rdkit(mols, y=None, config=cfg)
    # NodeBatch is a frozen dataclass; y is per-node and n_nodes is only known
    # once the batch exists, so attach the targets with replace().
    b = dataclasses.replace(b, y=rng.normal(size=(b.n_nodes, 1)))

    first = b.graph_id < 4
    a, c = sieve.fit(b[first], cfg), sieve.fit(b[~first], cfg)
    whole = sieve.fit(b, cfg)
    merged = a.merge(c)

    counts = [lv.n_classes for lv in whole.levels]
    assert [lv.n_classes for lv in merged.levels] == counts
    np.testing.assert_allclose(preds(merged, b), preds(whole, b))


def test_a_stereo_model_merges_identically_over_three_shards():
    """Two shards can pass by luck; three is the check that the ordering is
    genuinely batch-independent rather than symmetric in one split."""
    import dataclasses

    import numpy as np
    from rdkit import Chem

    import sieve
    from sieve.io.rdkit_adapter import from_rdkit

    cfg = stereo_config()
    smiles = ["C/C=C/C", r"C/C=C\\C", "C/C=C/CC", r"C/C=C\\CC",
              "C/C(F)=C(Cl)/C", r"C/C(F)=C(Cl)\\C", "CCCC", "CC(C)C",
              "C/C=C/Br"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rng = np.random.default_rng(1)
    b = from_rdkit(mols, y=None, config=cfg)
    # NodeBatch is a frozen dataclass; y is per-node and n_nodes is only known
    # once the batch exists, so attach the targets with replace().
    b = dataclasses.replace(b, y=rng.normal(size=(b.n_nodes, 1)))

    shards = [b[b.graph_id < 3], b[(b.graph_id >= 3) & (b.graph_id < 6)],
              b[b.graph_id >= 6]]
    merged = sieve.fit(shards[0], cfg)
    for shard in shards[1:]:
        merged = merged.merge(sieve.fit(shard, cfg))
    whole = sieve.fit(b, cfg)

    assert [lv.n_classes for lv in merged.levels] == [
        lv.n_classes for lv in whole.levels
    ]
    np.testing.assert_allclose(preds(merged, b), preds(whole, b))


```python
def test_enantiomers_are_not_separated_by_the_cis_trans_code():
    """E/Z survives reflection; this half must be blind to handedness, which
    is what keeps it independent of the mirror quotient the tetrahedral half
    will need."""
    import numpy as np
    from sieve.io.rdkit_adapter import from_smiles
    from sieve.refine import refine

    for left, right in [("C[C@H](F)/C=C/C", "C[C@@H](F)/C=C/C"),
                        (r"C[C@H](F)/C=C\C", r"C[C@@H](F)/C=C\C")]:
        a = refine(from_smiles(left, config=stereo_config()), stereo_config())
        b = refine(from_smiles(right, config=stereo_config()), stereo_config())
        for la, lb in zip(a, b, strict=True):
            assert np.array_equal(np.sort(la.signatures, axis=0),
                                  np.sort(lb.signatures, axis=0))


def test_the_fingerprint_never_sees_the_stereo_trit():
    """The one-character mistake with no error message: folding `full`
    instead of `edge_code` into the fingerprint makes the ordering
    stereo-dependent, so mirroring or remapping reorders substituents and a
    merged model disagrees on a handful of atoms."""
    import inspect

    from sieve import refine as refine_module

    source = inspect.getsource(refine_module)
    call = source.split("content_ranks(")[1].split(")")[0]
    assert "edge_code" in call and "full" not in call
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_merge.py -k stereo -v`
Expected: FAIL until Tasks 1-6 are complete; if they pass immediately, the
fixtures are not exercising the feature — check `stereo_config.stereo`.

- [x] **Step 3: Fix whatever they catch**

No new production code is planned here. A failure means one of:
- codes differ between whole and sharded fits → the fingerprint is seeing
  something batch-local (Task 4);
- enantiomers separate → `stereo_bonds` is picking up atom stereo (Task 3);
- class counts differ but predictions match → `n_edge_types` is not constant
  across levels (Task 1).

- [x] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tests/test_merge.py
git commit -m "test(stereo): the merge monoid holds and reflection is untouched"
```

---

## Execution notes

All 7 tasks completed inline; 299 tests pass, no regressions. Two things
came up during execution that the plan did not anticipate, both fixed within
the scope of the task where they surfaced rather than deferred:

- **Task 2** grew to include `NodeBatch.__getitem__` and `concat_batches`,
  neither of which carried `stereo_bonds` through. Undiscovered, Task 6/7's
  slicing tests (`b[first]`, `b[graph_id < 3]`) would have silently disabled
  stereo on every sharded fit. Fixed the same way `elements`/`y` already are:
  filtered/remapped on slicing, offset and all-or-none checked on concat.
- **Task 3**'s adapter wiring needed the same `inv[]` permutation the
  existing per-atom loop already applies for a custom `node_order`, not a
  plain node-count offset as the plan's prose suggested. Caught by a test
  that reverses a molecule's atom order and checks the emitted indices.

## Not in this plan

Per spec §9: the tetrahedral half, the mirror quotient, the post-fit vocabulary
stage, the non-local bound arms, cumulenes, and any change to `collapse_key` or
the curation.

Per spec §10, two measurements are owed but are not implementation: characterising
the 190 `Unspecified` double bonds, and re-measuring `cis-trans-geometry.md`'s
separation and cost figures on the rebuilt store under the honest-radius rule.
