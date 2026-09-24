# Tetrahedral Handedness: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `"tetrahedral"` stereo track, a handedness trit in the aware row beside the cis/trans trit, exactly mirror-invariant through an M row set and `mirror_of`, and Study E comparing it with Study D's cis/trans arm.

**Architecture:** The adapter emits `NodeBatch.stereo_centres` from local chiral tags. `refine` computes the trit from the stereo-blind content fingerprint at radius *k* − 1, folds it into the aware row's edge code, and builds a third row set M (the enantiomer's aware row) into the same union vocabulary. Statistics accrue to a class and its mirror, `aware_variance` counts each mirror orbit once, merge and persistence carry `mirror_of` like `blind_of`, and prediction is unchanged.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse, RDKit, pytest, ruff, ty; bash workflow.

**Spec:** `docs/superpowers/specs/2026-09-24-tetrahedral-handedness-design.md`

## Global Constraints

- One PR: this branch, `tetrahedral-spec` (PR #43), carries the spec, this plan, the code and Study E.
- `stereo=()` and `stereo=("cis_trans",)` models are unchanged: same classes, statistics, digests, saved keys. `stereo_centres`, `mirror_labels`, `mirror_of` are `None` on them.
- `STEREO_TRACKS = ("cis_trans", "tetrahedral")`; `stereo` is normalised to that order, duplicates rejected.
- Stereo digits in `STEREO_TRACKS` order, each `STEREO_RADIX = 4` wide: `full = edge_code * stereo_radix + stereo_code`, `stereo_code = ct * 4 + tet` with both tracks, the single digit with one.
- Tetrahedral codes: `CODE_NONE = 0`, `CODE_PLUS = 1`, `CODE_MINUS = 2`, `3` reserved.
- `stereo_centres` rows: `[v, n0, n1, n2, n3, parity]`, `parity ∈ {−1, +1}` (CCW = +1), `n3 = -1` for a virtual neighbour; an absent position contributes no inversion (it sorts last, as in RDKit's reference order).
- Round *k* reads `fp_{k-1}` (tetrahedral) and `fp_{k-2}` (cis/trans); `content_ranks` yields `n_wl - 1` rounds when the tetrahedral track is on, `max(n_wl - 2, 0)` otherwise.
- "Identical" means partitions and `count`/`mean`/`msd` bit-identical, derived estimates and predictions `rtol=1e-12`.
- Gate: `.venv/bin/ruff check src tests experiments`, `.venv/bin/ruff format --check src tests experiments`, `.venv/bin/ty check src tests experiments` (two known `assert_array_equal` diagnostics), `.venv/bin/python -m pytest -q`. Tests that need pandas use `pytest.importorskip("pandas")`.

---

### File map

| File | Change |
|---|---|
| `src/sieve/config.py` | `STEREO_TRACKS`, normalisation |
| `src/sieve/batch.py` | `stereo_centres`: field, validation, slicing, concat |
| `src/sieve/io/rdkit_adapter.py` | `_stereo_centre_rows`; per-track gating |
| `src/sieve/stereo.py` | `CODE_PLUS/MINUS`, `tetrahedral_codes`, `mirror_codes`, `centre_positions`, `FingerprintWindow` |
| `src/sieve/refine.py` | per-track codes, fingerprint window, M rows, `mirror_labels`/`mirror_of` |
| `src/sieve/level.py` | `FrozenLevel.mirror_of`, `mirror_targets`, `fit_level` mirror accrual |
| `src/sieve/continuation.py` | `aware_variance` counts each orbit once |
| `src/sieve/merge.py`, `src/sieve/model.py` | carry `mirror_of` |
| `experiments/experiments/stereo_subsets.py`, `cli.py` | tetrahedral subsets, `--check` for masks, stale-key detection |
| `experiments/workflows/cv_charges.sh` | masks guard; Study E |
| `tests/test_tetrahedral.py` | new: spec §7 |

---

### Task 1: Config

**Files:** `src/sieve/config.py`; spec §2.2 wording; Test: `tests/test_tetrahedral.py` (created here).

**Produces:** `STEREO_TRACKS == ("cis_trans", "tetrahedral")`; `SieveConfig.stereo` normalised.

- [ ] **Step 1: failing tests.** Create `tests/test_tetrahedral.py`:

```python
"""Tetrahedral handedness (spec 2026-09-24-tetrahedral-handedness-design.md, §7)."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import sieve
from sieve.config import KIND_AWARE, KIND_BLIND, SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit
from sieve.level import blind_targets, class_kinds
from sieve.refine import refine

BOTH = ("cis_trans", "tetrahedral")


def _mols(smiles, *, hs=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.AddHs(m) for m in mols] if hs else mols


def _config(mols, *, stereo=BOTH, depth=4, **kw):
    codes, edges = build_codes(mols, ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=depth,
        stereo=stereo,
        **kw,
    )


def _batch(mols, cfg, *, seed=0, node_order=None):
    b = from_rdkit(mols, y=None, config=cfg, node_order=node_order)
    y = np.random.default_rng(seed).normal(size=(b.n_nodes, 1))
    return dataclasses.replace(b, y=y)


def _stripped(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out


def _inverted(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.InvertChirality()
    return out


def test_track_order_is_normalised_and_duplicates_refused():
    mols = _mols(["CC"])
    a = _config(mols, stereo=("tetrahedral", "cis_trans"))
    b = _config(mols, stereo=("cis_trans", "tetrahedral"))
    assert a.stereo == BOTH and a.schema_version == b.schema_version
    with pytest.raises(ValueError, match="twice"):
        _config(mols, stereo=("cis_trans", "cis_trans"))


def test_three_digests_and_cis_trans_unchanged():
    mols = _mols(["CC"])
    digests = {
        _config(mols, stereo=s).schema_version
        for s in ((), ("cis_trans",), BOTH)
    }
    assert len(digests) == 3
    assert _config(mols, stereo=BOTH).stereo_radices == (4, 4)
```

- [ ] **Step 2:** `.venv/bin/python -m pytest -q tests/test_tetrahedral.py` → the normalisation test fails (`a.stereo` keeps the given order) and the duplicate is accepted.
- [ ] **Step 3: implement.** In `config.py`: `STEREO_TRACKS = ("cis_trans", "tetrahedral")`. In `__post_init__`, right after the loop that rejects unknown tracks:

```python
        if len(set(self.stereo)) != len(self.stereo):
            raise ValueError(f"stereo lists a track twice: {list(self.stereo)}")
        # One spelling per configuration, so equal configs hash equally; the
        # digits of the stereo code follow this order too (refine).
        object.__setattr__(
            self, "stereo", tuple(t for t in STEREO_TRACKS if t in self.stereo)
        )
```

  Spec §2.2: replace "A virtual hydrogen takes the fixed value 0, which sorts first." with "An absent position contributes no inversion: it sorts last, as it sits last in RDKit's reference order."
- [ ] **Step 4:** tests pass; `tests/test_config.py` passes.
- [ ] **Step 5: commit** `feat(config): the tetrahedral track name and a canonical track order`.

---

### Task 2: `NodeBatch.stereo_centres`

**Files:** `src/sieve/batch.py`. Test: `tests/test_tetrahedral.py`.

**Produces:** `NodeBatch.stereo_centres: np.ndarray | None` (`(n, 6)` int64), validated, sliced, concatenated.

- [ ] **Step 1: failing tests**

```python
from sieve.batch import NodeBatch, concat_batches


def _star_batch(centres):
    """Atom 0 bonded to atoms 1..4; two copies side by side."""
    src = [0, 1, 0, 2, 0, 3, 0, 4]
    dst = [1, 0, 2, 0, 3, 0, 4, 0]
    src += [s + 5 for s in src]
    dst += [d + 5 for d in dst]
    return NodeBatch(
        node_attrs=np.zeros((10, 1), np.int64),
        edge_src=np.array(src),
        edge_dst=np.array(dst),
        edge_attrs=np.zeros((16, 1), np.int64),
        graph_id=np.array([0] * 5 + [1] * 5),
        stereo_centres=np.array(centres, np.int64).reshape(-1, 6),
    )


@pytest.mark.parametrize(
    "row, message",
    [
        ([0, 1, 2, 3, 6, 1], "adjacent"),
        ([0, 1, -1, 3, 4, 1], "only in n3"),
        ([0, 1, 2, 3, -1, 1], "degree"),
        ([0, 1, 2, 3, 4, 0], "parity"),
        ([0, 1, 2, 3, 99, 1], "range"),
    ],
)
def test_stereo_centres_are_validated(row, message):
    with pytest.raises(ValueError, match=message):
        _star_batch([row])


def test_a_centre_listed_twice_is_refused():
    with pytest.raises(ValueError, match="once"):
        _star_batch([[0, 1, 2, 3, 4, 1], [0, 4, 3, 2, 1, 1]])


def test_stereo_centres_follow_slicing_and_concat():
    b = _star_batch([[0, 1, 2, 3, 4, 1], [5, 6, 7, 8, 9, -1]])
    second = b[b.graph_id == 1]
    np.testing.assert_array_equal(second.stereo_centres, [[0, 1, 2, 3, 4, -1]])
    both = concat_batches([second, second])
    np.testing.assert_array_equal(
        both.stereo_centres, [[0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, -1]]
    )
    with pytest.raises(ValueError, match="stereo_centres"):
        concat_batches([second, dataclasses.replace(second, stereo_centres=None)])
```

  (A three-neighbour row `[0, 1, 2, 3, -1, 1]` on a degree-4 atom is the "degree" case; a real three-neighbour centre is exercised through the adapter in Task 3.)
- [ ] **Step 2:** run → `TypeError: unexpected keyword 'stereo_centres'`.
- [ ] **Step 3: implement.** Field after `stereo_bonds`:

```python
    stereo_centres: np.ndarray | None = None  # (n_centres, 6) int64
```

  `__post_init__` calls `self._check_stereo_centres()` after `_check_stereo_bonds()`:

```python
    def _check_stereo_centres(self) -> None:
        """Validate the tetrahedral-centre table, if present.

        Columns are ``[v, n0, n1, n2, n3, parity]``: the centre, its
        neighbours in the order its chiral tag refers to, and +1 for CCW or
        -1 for CW. ``n3 = -1`` marks a virtual fourth neighbour (an implicit
        hydrogen or a lone pair). As for stereo_bonds, every check is a
        corpus bug that would otherwise become a plausible wrong code.
        """
        sc = self.stereo_centres
        if sc is None:
            return
        if sc.ndim != 2 or sc.shape[1] != 6:
            raise ValueError(
                f"stereo_centres must have shape (n_centres, 6), got {sc.shape}"
            )
        n = self.node_attrs.shape[0]
        v, nb, parity = sc[:, 0], sc[:, 1:5], sc[:, 5]
        if (nb[:, :3] < 0).any():
            raise ValueError("stereo_centres: -1 is allowed only in n3")
        for col, lo in ((v, 0), (nb[:, :3], 0), (nb[:, 3], -1)):
            if ((col < lo) | (col >= n)).any():
                raise ValueError(f"stereo_centres: an index is out of range [{lo}, {n})")
        if not np.isin(parity, (-1, 1)).all():
            raise ValueError("stereo_centres: parity must be -1 or +1")
        if np.unique(v).size != v.size:
            raise ValueError("stereo_centres: each centre may be listed once")
        present = nb >= 0
        degree = np.bincount(self.edge_src, minlength=n)
        if not np.array_equal(present.sum(axis=1), degree[v]):
            raise ValueError(
                "stereo_centres: a row must name every neighbour, so the number "
                "of present n_i must equal the centre's degree"
            )
        key = np.sort(self.edge_src * n + self.edge_dst)
        want = (np.repeat(v, 4) * n + nb.ravel())[present.ravel()]
        pos = np.clip(np.searchsorted(key, want), 0, max(key.size - 1, 0))
        if key.size == 0 or not (key[pos] == want).all():
            raise ValueError("stereo_centres: a neighbour is not adjacent to its centre")
```

  In `__getitem__`, beside the stereo_bonds block:

```python
        stereo_centres = None
        if self.stereo_centres is not None:
            sc = self.stereo_centres
            cols = sc[:, :5]
            has = cols >= 0
            selected = np.where(has, mask[np.where(has, cols, 0)], True)
            row_keep = selected.all(axis=1)
            remapped = np.where(has, remap[np.where(has, cols, 0)], -1)
            stereo_centres = np.concatenate(
                [remapped[row_keep], sc[row_keep, 5:6]], axis=1
            )
```

  and pass `stereo_centres=stereo_centres` to `_with_trusted_edges`. In `concat_batches`: the all-or-none check (`"stereo_centres is set on some but not all parts"`), a list filled per part with

```python
        if p.stereo_centres is not None:
            sc = p.stereo_centres
            cols = sc[:, :5]
            cols = np.where(cols >= 0, cols + node_off, -1)
            stereo_centres.append(np.concatenate([cols, sc[:, 5:6]], axis=1))
```

  and `stereo_centres=(np.concatenate(stereo_centres) if has_stereo_centres == {True} else None)`. `grep -n "_with_trusted_edges(" src tests experiments` and pass `stereo_centres` wherever it is called (it requires every field).
- [ ] **Step 4:** tests pass; `tests/test_batch.py` passes.
- [ ] **Step 5: commit** `feat(batch): stereo_centres, validated and carried through slicing and concat`.

---

### Task 3: The adapter

**Files:** `src/sieve/io/rdkit_adapter.py`. Test: `tests/test_tetrahedral.py`.

**Produces:** `_stereo_centre_rows(mol) -> list[tuple[int, int, int, int, int, int]]`; `from_rdkit` fills `stereo_centres` when `"tetrahedral" in config.stereo`, `stereo_bonds` only when `"cis_trans" in config.stereo`.

- [ ] **Step 1: failing tests**

```python
from sieve.io.rdkit_adapter import _stereo_centre_rows


def test_enantiomers_give_opposite_parity_on_identical_rows():
    (r,) = _stereo_centre_rows(_mols(["F[C@H](Cl)Br"])[0])
    (s,) = _stereo_centre_rows(_mols(["F[C@@H](Cl)Br"])[0])
    assert r[:5] == s[:5] and r[5] == -s[5]


def test_a_sulfoxide_has_a_virtual_fourth_neighbour():
    (row,) = _stereo_centre_rows(_mols(["C[S@](=O)CC"])[0])
    assert row[4] == -1


def test_untagged_and_two_neighbour_centres_give_no_row():
    assert _stereo_centre_rows(_mols(["CC(C)CC"])[0]) == []
    assert _stereo_centre_rows(Chem.MolFromSmiles("C[P@H]CC")) == []


def test_the_batch_carries_centres_only_under_the_track():
    mols = _mols(["F[C@H](Cl)Br", "C/C=C/C"])
    both = from_rdkit(mols, config=_config(mols))
    ct = from_rdkit(mols, config=_config(mols, stereo=("cis_trans",)))
    tet = from_rdkit(mols, config=_config(mols, stereo=("tetrahedral",)))
    assert both.stereo_centres.shape == (1, 6) and both.stereo_bonds.shape == (1, 7)
    assert ct.stereo_centres is None
    assert tet.stereo_bonds is None and tet.stereo_centres.shape == (1, 6)


def test_parallel_featurisation_carries_centres():
    mols = _mols(["F[C@H](Cl)Br", "N[C@@H](C)C(=O)O"] * 4)
    cfg = _config(mols)
    seq = from_rdkit(mols, config=cfg)
    par = from_rdkit(mols, config=cfg, n_jobs=2)
    np.testing.assert_array_equal(seq.stereo_centres, par.stereo_centres)
```

- [ ] **Step 2:** run → ImportError `_stereo_centre_rows`.
- [ ] **Step 3: implement.** Beside `_stereo_bond_rows`:

```python
def _stereo_centre_rows(mol) -> list[tuple[int, int, int, int, int, int]]:
    """Tagged tetrahedral centres as ``[v, n0, n1, n2, n3, parity]``, in raw
    RDKit atom-index order (callers apply ``inv[]``, as for stereo bonds).

    Read from the local chiral tag, never CIP and never
    ``FindPotentialStereo``, whose stereogenicity test uses the
    whole-molecule ranking: a tagged atom that is not stereogenic at some
    radius simply gets ``none`` there. The tag is a parity in the order of
    ``atom.GetBonds()``, which is written into the row explicitly so nothing
    depends on the CSR order matching it. A centre with three bonds has a
    virtual fourth neighbour (an implicit H or a lone pair), which RDKit's
    tag places last (measured 2026-09-24); any other bond count is skipped.
    """
    from rdkit import Chem

    parity_of = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: -1,
    }
    rows = []
    for atom in mol.GetAtoms():
        parity = parity_of.get(atom.GetChiralTag())
        if parity is None:
            continue
        v = atom.GetIdx()
        nbrs = [b.GetOtherAtomIdx(v) for b in atom.GetBonds()]
        if len(nbrs) == 3:
            nbrs.append(-1)
        elif len(nbrs) != 4:
            continue
        rows.append((v, nbrs[0], nbrs[1], nbrs[2], nbrs[3], parity))
    return rows
```

  In `_from_rdkit_sequential`: `if config.stereo:` becomes `if "cis_trans" in config.stereo:` around the stereo-bond block; add `centre_rows: list[tuple[int, ...]] = []` and

```python
        if "tetrahedral" in config.stereo:
            for v, n0, n1, n2, n3, parity in _stereo_centre_rows(mol):
                centre_rows.append(
                    (
                        off + int(inv[v]),
                        off + int(inv[n0]),
                        off + int(inv[n1]),
                        off + int(inv[n2]),
                        -1 if n3 < 0 else off + int(inv[n3]),
                        parity,
                    )
                )
```

  and in the returned batch `stereo_bonds=(... if "cis_trans" in config.stereo else None)`, `stereo_centres=(np.array(centre_rows, np.int64).reshape(-1, 6) if "tetrahedral" in config.stereo else None)`. The parallel path reassembles through `concat_batches`, which Task 2 taught.
- [ ] **Step 4:** tests pass; `tests/test_rdkit_adapter.py` passes.
- [ ] **Step 5: commit** `feat(adapter): tetrahedral centres from local chiral tags`.

---

### Task 4: The code

**Files:** `src/sieve/stereo.py`. Test: `tests/test_tetrahedral.py`.

**Produces:** `CODE_PLUS, CODE_MINUS = 1, 2`; `tetrahedral_codes(stereo_centres, fp) -> (n_centres,) int64`; `mirror_codes(codes)`; `centre_positions(csr, n, stereo_centres) -> (pos, row)`; `FingerprintWindow(generator).at(radius) -> np.ndarray`.

- [ ] **Step 1: failing tests** (the hand-checked arithmetic that pins the convention)

```python
from sieve.stereo import (
    CODE_MINUS,
    CODE_NONE,
    CODE_PLUS,
    FingerprintWindow,
    mirror_codes,
    tetrahedral_codes,
)


def _row(n3, parity):
    return np.array([[0, 1, 2, 3, n3, parity]], np.int64)


@pytest.mark.parametrize(
    "fp, n3, parity, expected",
    [
        ([0, 10, 20, 30, 40], 4, 1, CODE_PLUS),     # already sorted: even
        ([0, 20, 10, 30, 40], 4, 1, CODE_MINUS),    # one inversion
        ([0, 20, 10, 30, 40], 4, -1, CODE_PLUS),    # CW flips it back
        ([0, 40, 30, 20, 10], 4, 1, CODE_PLUS),     # six inversions: even
        ([0, 10, 20, 30, 0], -1, 1, CODE_PLUS),     # virtual n3 adds none
        ([0, 20, 10, 30, 0], -1, 1, CODE_MINUS),
        ([0, 10, 10, 30, 40], 4, 1, CODE_NONE),     # a tie defers
        ([0, 10, 20, 30, 10], -1, 1, CODE_PLUS),    # a virtual n3 never ties
    ],
)
def test_the_code_is_parity_times_the_sorting_sign(fp, n3, parity, expected):
    got = tetrahedral_codes(_row(n3, parity), np.array(fp, np.uint64))
    assert got.tolist() == [expected]


def test_mirror_codes_swap_plus_and_minus_only():
    codes = np.array([CODE_NONE, CODE_PLUS, CODE_MINUS])
    assert mirror_codes(codes).tolist() == [CODE_NONE, CODE_MINUS, CODE_PLUS]


def test_the_window_serves_two_consecutive_radii():
    w = FingerprintWindow(iter([np.array([r]) for r in range(5)]))
    assert w.at(1).tolist() == [1] and w.at(0).tolist() == [0]
    assert w.at(3).tolist() == [3] and w.at(2).tolist() == [2]
    with pytest.raises(ValueError, match="radius"):
        w.at(0)
```

- [ ] **Step 2:** run → ImportError.
- [ ] **Step 3: implement** in `stereo.py`:

```python
CODE_PLUS, CODE_MINUS = 1, 2  # tetrahedral: parity times the sorting sign


def tetrahedral_codes(stereo_centres: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """One code in ``{none, plus, minus}`` per tetrahedral centre.

    The sign is the tag's parity times the sign of the permutation that sorts
    the present neighbours by fingerprint, counted as inversions in reference
    order. An absent fourth position contributes no inversion: it sits last
    in RDKit's reference order, and any fixed place would flip every such
    sign alike. Two present neighbours with equal fingerprints make the
    centre unresolvable at this radius, and the code defers to ``none``.
    """
    nb = stereo_centres[:, 1:5]
    present = nb >= 0
    f = np.where(present, fp[np.where(present, nb, 0)], np.uint64(0))
    tie = np.zeros(nb.shape[0], bool)
    odd = np.zeros(nb.shape[0], bool)
    for i in range(4):
        for j in range(i + 1, 4):
            both = present[:, i] & present[:, j]
            tie |= both & (f[:, i] == f[:, j])
            odd ^= both & (f[:, i] > f[:, j])
    sign = np.where(odd, -1, 1) * stereo_centres[:, 5]
    return np.where(
        tie, CODE_NONE, np.where(sign > 0, CODE_PLUS, CODE_MINUS)
    ).astype(np.int64)


def mirror_codes(codes: np.ndarray) -> np.ndarray:
    """The codes of the mirror image: plus and minus swap, none stays."""
    out = codes.copy()
    out[codes == CODE_PLUS] = CODE_MINUS
    out[codes == CODE_MINUS] = CODE_PLUS
    return out


def centre_positions(
    csr: CSRLayout, n_nodes: int, stereo_centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CSR positions of every out-edge of a centre, and that centre's row.

    The code rides on all of a centre's out-edges, so it enters the centre's
    own signature row and no other; resolved once, before the round loop.
    """
    row_of = np.full(n_nodes, -1, np.int64)
    row_of[stereo_centres[:, 0]] = np.arange(stereo_centres.shape[0])
    pos = np.flatnonzero(row_of[csr.src] >= 0)
    return pos, row_of[csr.src[pos]]


class FingerprintWindow:
    """Serves ``fp_r`` and ``fp_{r-1}`` from a generator consumed in order.

    With both tracks on, round *k* reads ``fp_{k-1}`` (tetrahedral) and
    ``fp_{k-2}`` (cis/trans). ``content_ranks`` yields strictly in order and
    retaining every radius costs ~1.6 GB on the train split, so this keeps
    one fingerprint of history and refuses anything older.
    """

    def __init__(self, fingerprints: Iterator[np.ndarray]) -> None:
        self._it = fingerprints
        self._radius = -1
        self._cur: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    def at(self, radius: int) -> np.ndarray:
        while self._radius < radius:
            self._prev, self._cur = self._cur, next(self._it)
            self._radius += 1
        if radius == self._radius and self._cur is not None:
            return self._cur
        if radius == self._radius - 1 and self._prev is not None:
            return self._prev
        raise ValueError(
            f"radius {radius} is no longer held (window at {self._radius})"
        )
```

- [ ] **Step 4:** tests pass; `tests/test_stereo.py` passes.
- [ ] **Step 5: commit** `feat(stereo): the tetrahedral code, its mirror, and a two-radius window`.

---

### Task 5: Refinement: the trit and the M rows

**Files:** `src/sieve/refine.py`, `src/sieve/level.py` (only `mirror_targets`). Test: `tests/test_tetrahedral.py`: change the header import to `from sieve.level import blind_targets, class_kinds, mirror_targets`, and add `from sieve.io.rdkit_adapter import _stereo_centre_rows` if Task 3 did not already.

**Produces:** `LevelLabels(..., mirror_labels=None, mirror_of=None)`, property `mirror`; `sieve.level.mirror_targets(level) -> np.ndarray[int64]` (identity when `mirror_of is None`).

- [ ] **Step 1: failing tests**

```python
def _centre_labels(levels, atom):
    return [int(lv.labels[atom]) for lv in levels]


def test_the_radius_rule_under_element_only_attributes():
    """fp_0 hashes the whole attribute row, so the rule is only visible with
    element-only attributes: there CHFClBr's four neighbours differ at
    radius 0 and alanine's two carbons do not."""
    for smiles, first in (("F[C@H](Cl)Br", 1), ("N[C@@H](C)C(=O)O", 2)):
        mols = _mols([smiles])
        mirror = _inverted(mols)
        cfg = _config(mols + mirror, depth=3)
        lv = refine(from_rdkit(mols + mirror, config=cfg), cfg)
        (row,) = _stereo_centre_rows(mols[0])
        v, n = row[0], mols[0].GetNumAtoms()
        for k in range(1, 4):
            same = lv[k].labels[v] == lv[k].labels[v + n]
            assert same == (k < first), (smiles, k)


def test_ties_defer_at_every_radius():
    m = Chem.MolFromSmiles("CC(C)Cl")
    m.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    mols = [Chem.AddHs(m)]
    cfg = _config(mols, depth=4)
    for lv in refine(from_rdkit(mols, config=cfg), cfg):
        np.testing.assert_array_equal(lv.labels, lv.blind)


def test_five_orderings_of_alanine_agree():
    """Random SMILES of one molecule, so every copy is the same enantiomer."""
    base = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    smiles = list(Chem.MolToRandomSmilesVect(base, 5, randomSeed=0))
    for hs in (True, False):
        mols = _mols(smiles, hs=hs)
        cfg = _config(mols, depth=3)
        lv = refine(from_rdkit(mols, config=cfg), cfg)
        offs = np.cumsum([0] + [m.GetNumAtoms() for m in mols])
        centres = [o + _stereo_centre_rows(m)[0][0] for o, m in zip(offs, mols, strict=False)]
        for level in lv:
            assert len({int(level.labels[c]) for c in centres}) == 1


def test_renumbering_leaves_the_classes_unchanged():
    mols = _mols(["N[C@@H](C)C(=O)O", "F[C@H](Cl)Br"])
    rng = np.random.default_rng(0)
    orders = [rng.permutation(m.GetNumAtoms()) for m in mols]
    cfg = _config(mols + mols, depth=3)
    natural = [np.arange(m.GetNumAtoms()) for m in mols]
    b = from_rdkit(mols + mols, config=cfg, node_order=natural + orders)
    sizes = [m.GetNumAtoms() for m in mols]
    n = sum(sizes)
    # Raw atom a of molecule j sits at first[j] + a in the natural copy and at
    # n + first[j] + position-of-a-in-order_j in the permuted copy.
    first = np.cumsum([0, *sizes[:-1]])
    natural_pos = np.concatenate([f + np.arange(k) for f, k in zip(first, sizes, strict=True)])
    permuted_pos = np.concatenate(
        [n + f + np.argsort(o) for f, o in zip(first, orders, strict=True)]
    )
    for lv in refine(b, cfg):
        np.testing.assert_array_equal(lv.labels[natural_pos], lv.labels[permuted_pos])


def test_mirror_rows_and_maps_are_consistent():
    mols = _mols(["N[C@@H](C)C(=O)O", r"C/C=C\[C@H](F)Cl", "CCCC"])
    cfg = _config(mols)
    for lv in refine(from_rdkit(mols, config=cfg), cfg):
        mt, bt = mirror_targets(lv), blind_targets(lv)
        np.testing.assert_array_equal(mt[mt], np.arange(lv.n_classes))
        np.testing.assert_array_equal(mt[lv.labels], lv.mirror)
        np.testing.assert_array_equal(bt[lv.mirror], lv.blind)
        assert np.all(class_kinds(lv)[lv.mirror] & KIND_AWARE)
```

- [ ] **Step 2:** run → the radius-rule and mirror tests fail (no tetrahedral code yet).
- [ ] **Step 3: implement.**

  `level.py`, beside `blind_targets`:

```python
def mirror_targets(level) -> np.ndarray:
    """Per-class mirror image; a stored ``None`` means the identity."""
    if level.mirror_of is None:
        return np.arange(level.n_classes, dtype=np.int64)
    return level.mirror_of
```

  and `FrozenLevel` gains `mirror_of: np.ndarray | None = None  # (nc,) int64` after `blind_of` (Task 6 fills it).

  `refine.py`: `LevelLabels` gains `mirror_labels: np.ndarray | None = None` and `mirror_of: np.ndarray | None = None`, plus

```python
    @property
    def mirror(self) -> np.ndarray:
        """Each atom's class in its molecule's enantiomer: its own class when
        the tetrahedral track is off."""
        return self.labels if self.mirror_labels is None else self.mirror_labels
```

  `_union_level(sig_aware, sig_blind, sig_mirror=None)`:

```python
    n = sig_blind.shape[0]
    differs = np.flatnonzero((sig_aware != sig_blind).any(axis=1))
    stack = [sig_blind, sig_aware[differs]]
    mdiff = None
    if sig_mirror is not None:
        # A mirror row differs from its aware row only where a handedness
        # code is in reach; elsewhere it would deduplicate onto it anyway.
        mdiff = np.flatnonzero((sig_mirror != sig_aware).any(axis=1))
        stack.append(sig_mirror[mdiff])
    labels, uniq = dense_rows(np.concatenate(stack))
    blind = labels[:n]
    aware = blind.copy()
    aware[differs] = labels[n : n + differs.size]
    kind = np.zeros(uniq.shape[0], np.uint8)
    kind[blind] |= KIND_BLIND
    kind[aware] |= KIND_AWARE
    blind_of = np.arange(uniq.shape[0], dtype=np.int64)
    blind_of[aware] = blind
    mirror = mirror_of = None
    if mdiff is not None:
        mirror = aware.copy()
        mirror[mdiff] = labels[n + differs.size :]
        # An M class is the aware class of an enantiomer's atom.
        kind[mirror] |= KIND_AWARE
        blind_of[mirror] = blind
        mirror_of = np.arange(uniq.shape[0], dtype=np.int64)
        mirror_of[aware] = mirror
        mirror_of[mirror] = aware
    return LevelLabels(
        aware, uniq, uniq[:, 0].astype(np.int32), blind, kind, blind_of,
        mirror, mirror_of,
    )
```

  Stereo setup, replacing the `if config.stereo:` block:

```python
    cis_trans = "cis_trans" in config.stereo
    tetrahedral = "tetrahedral" in config.stereo
    stereo_bonds = stereo_centres = None
    pos_ab = pos_ba = tet_pos = tet_row = None  # bound with their tables
    window: FingerprintWindow | None = None
    if config.stereo:
        if cis_trans:
            if batch.stereo_bonds is None:
                raise ValueError(
                    f"config.stereo is {list(config.stereo)} but the batch "
                    "carries no stereo_bonds; the adapter was run with a "
                    "stereo-blind config"
                )
            stereo_bonds = batch.stereo_bonds
            pos_ab, pos_ba = directed_positions(csr, n, stereo_bonds)
        if tetrahedral:
            if batch.stereo_centres is None:
                raise ValueError(
                    f"config.stereo is {list(config.stereo)} but the batch "
                    "carries no stereo_centres; the adapter was run without "
                    "the tetrahedral track"
                )
            stereo_centres = batch.stereo_centres
            tet_pos, tet_row = centre_positions(csr, n, stereo_centres)
        n_wl = sum(1 for k in kinds if k == LEVEL_WL)
        # Round k reads fp_{k-1} (tetrahedral) and fp_{k-2} (cis/trans).
        rounds = max(n_wl - 1, 0) if tetrahedral else max(n_wl - 2, 0)
        window = FingerprintWindow(
            content_ranks(batch.node_attrs, csr, edge_code, rounds)
        )

    def stereo_full(ct: np.ndarray, tet: np.ndarray) -> np.ndarray:
        # Digits in STEREO_TRACKS order, one per enabled track.
        code = np.zeros(edge_code.shape[0], np.int64)
        if cis_trans:
            code = code * STEREO_RADIX + ct
        if tetrahedral:
            code = code * STEREO_RADIX + tet
        return edge_code * stereo_radix + code
```

  In the WL branch, replacing the stereo block:

```python
            if window is not None:
                e = edge_code.shape[0]
                ct = np.zeros(e, np.int64)
                tet = np.zeros(e, np.int64)
                tet_mirror = np.zeros(e, np.int64)
                if stereo_centres is not None:
                    assert tet_pos is not None and tet_row is not None
                    # The ranked neighbours sit one bond away, so radius
                    # k-1 is honest and the code can fire from round 1.
                    codes = tetrahedral_codes(stereo_centres, window.at(wl_round - 1))
                    tet[tet_pos] = codes[tet_row]
                    tet_mirror[tet_pos] = mirror_codes(codes)[tet_row]
                if stereo_bonds is not None and wl_round >= 2:
                    assert pos_ab is not None and pos_ba is not None
                    codes = cis_trans_codes(stereo_bonds, window.at(wl_round - 2))
                    ct[pos_ab] = codes
                    ct[pos_ba] = codes
                parent = levels[parents[offset]]
                zero = np.zeros(e, np.int64)
                sig_aware = _wl_rows(base, csr, stereo_full(ct, tet), n, n_edge_types)
                sig_blind = _wl_rows(parent.blind, csr, stereo_full(zero, zero), n, n_edge_types)
                sig_mirror = (
                    _wl_rows(parent.mirror, csr, stereo_full(ct, tet_mirror), n, n_edge_types)
                    if stereo_centres is not None
                    else None
                )
                levels.append(_union_level(sig_aware, sig_blind, sig_mirror))
                continue
```

  Imports: `STEREO_RADIX` from config; `FingerprintWindow, centre_positions, mirror_codes, tetrahedral_codes` from stereo. Keep the existing comments on recursive blinding.
- [ ] **Step 4:** tests pass; `tests/test_refine.py tests/test_stereo_refines_blind.py tests/test_merge.py` pass (cis/trans-only behaviour unchanged).
- [ ] **Step 5: commit** `feat(refine): the tetrahedral trit in the aware row, and the mirror row set`.

---

### Task 6: Statistics

**Files:** `src/sieve/level.py`, `src/sieve/continuation.py`. Test: `tests/test_tetrahedral.py`.

- [ ] **Step 1: failing tests**

```python
CORPUS = [
    "N[C@@H](C)C(=O)O", "C[C@H](Br)[C@H](C)Br", "C[C@H](Br)[C@@H](C)Br",
    "F[C@H](Cl)Br", "C[S@](=O)CC", r"C/C=C\[C@H](F)Cl", "C/C=C/C", "CCCC",
]
EB = {"class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}


def test_a_class_and_its_mirror_hold_identical_statistics():
    mols = _mols(CORPUS)
    cfg = _config(mols)
    model = sieve.fit(_batch(mols, cfg), cfg)
    seen = 0
    for lv in model.levels:
        mt = mirror_targets(lv)
        chiral = np.flatnonzero(mt != np.arange(lv.n_classes))
        seen += chiral.size
        np.testing.assert_array_equal(lv.count[chiral], lv.count[mt[chiral]])
        np.testing.assert_array_equal(lv.mean[chiral], lv.mean[mt[chiral]])
        np.testing.assert_array_equal(lv.msd[chiral], lv.msd[mt[chiral]])
    assert seen


def test_self_mirror_classes_count_each_atom_once():
    """meso-2,3-dibromobutane: rows naming a and M(a) are unchanged by
    reflection, and must not be counted twice."""
    mols = _mols(["C[C@H](Br)[C@@H](C)Br"])
    cfg = _config(mols)
    batch = _batch(mols, cfg)
    model, labels = sieve.fit(batch, cfg), refine(batch, cfg)
    for lv, fl in zip(labels, model.levels, strict=True):
        mt = mirror_targets(fl)
        for c in np.flatnonzero((class_kinds(fl) == KIND_AWARE) & (mt == np.arange(fl.n_classes))):
            assert fl.count[c] == int((lv.labels == c).sum())


def test_blind_classes_stay_the_stereo_blind_fit():
    mols = _mols(CORPUS)
    s_cfg, b_cfg = _config(mols, **EB), _config(mols, stereo=(), **EB)
    s_batch, b_batch = _batch(mols, s_cfg), _batch(mols, b_cfg)
    s, b = sieve.fit(s_batch, s_cfg), sieve.fit(b_batch, b_cfg)
    for ls, lb, fs, fb in zip(refine(s_batch, s_cfg), refine(b_batch, b_cfg), s.levels, b.levels, strict=True):
        pairs = np.unique(np.stack([ls.blind, lb.labels], 1), axis=0)
        ids = pairs[:, 0]
        np.testing.assert_array_equal(fs.count[ids], fb.count[pairs[:, 1]])
        np.testing.assert_array_equal(fs.mean[ids], fb.mean[pairs[:, 1]])


def test_aware_variance_counts_each_mirror_orbit_once():
    from sieve.continuation import _aware_members

    mols = _mols(["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O", "N[C@@H](CC)C(=O)O"])
    cfg = _config(mols, **EB)
    model = sieve.fit(_batch(mols, cfg), cfg)
    chiral_seen = 0
    for lv in model.levels:
        aware = np.flatnonzero(class_kinds(lv) & KIND_AWARE)
        mt = mirror_targets(lv)
        members = _aware_members(lv)
        # Exactly one member per orbit: every aware class or its mirror is
        # present, never both unless it is its own mirror.
        orbits = {min(int(c), int(mt[c])) for c in aware}
        assert sorted(orbits) == sorted(int(c) for c in members)
        chiral_seen += int((mt[aware] != aware).sum())
    assert chiral_seen
```

- [ ] **Step 2:** run → the statistics tests fail: mirror classes hold no atoms yet.
- [ ] **Step 3: implement.** `fit_level`:

```python
    if level.kind is not None:
        differs = level.labels != level.blind
        extra_labels, extra_y = [level.labels[differs]], [y[differs]]
        if level.mirror_labels is not None:
            # Each atom also counts toward its class's mirror, once: an
            # achiral environment has M(a) = a and is not added again.
            moved = level.mirror_labels != level.labels
            extra_labels.append(level.mirror_labels[moved])
            extra_y.append(y[moved])
        lab, yy = np.concatenate(extra_labels), np.concatenate(extra_y)
        if lab.size:
            c2, m2, s2 = _reduce(lab, yy, nc)
            only = level.kind == KIND_AWARE
            count[only], mean[only], msd[only] = c2[only], m2[only], s2[only]
    return FrozenLevel(
        level.signatures, count, mean, msd, level.parent,
        level.kind, level.blind_of, level.mirror_of,
    )
```

  `continuation.py`: add

```python
def _aware_members(lvl) -> np.ndarray:
    """Aware-flagged classes, one per mirror orbit: a class and its mirror
    hold identical statistics, and counting both would add a zero-variance
    sibling that biases tau^2_aware downward."""
    aware = np.flatnonzero(class_kinds(lvl) & KIND_AWARE)
    return aware[aware <= mirror_targets(lvl)[aware]]
```

  and use it in `aware_variance` in place of `np.flatnonzero(class_kinds(lvl) & KIND_AWARE)`.
- [ ] **Step 4:** tests pass; `tests/test_stereo_refines_blind.py tests/test_continuation.py tests/test_shrinkage.py` pass.
- [ ] **Step 5: commit** `feat(level): a class and its mirror pool their statistics; tau^2 counts each orbit once`.

---

### Task 7: Merge, persistence, and the invariance tests

**Files:** `src/sieve/merge.py`, `src/sieve/model.py`. Test: `tests/test_tetrahedral.py`.

- [ ] **Step 1: failing tests**

```python
def _predict(model, mols, cfg):
    return sieve.predict_detailed(model, from_rdkit(mols, config=cfg))


@pytest.mark.parametrize("rule", [{}, EB], ids=["none", "eb"])
def test_embedding_without_chiral_tags_is_the_cis_trans_model(rule):
    mols = _stripped(_mols(CORPUS))
    both, ct = _config(mols, **rule), _config(mols, stereo=("cis_trans",), **rule)
    pb = _predict(sieve.fit(_batch(mols, both), both), mols, both)
    pc = _predict(sieve.fit(_batch(mols, ct), ct), mols, ct)
    np.testing.assert_allclose(pb.value, pc.value, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(pb.stereo_refined, pc.stereo_refined)


@pytest.mark.parametrize("rule", [{}, EB], ids=["none", "eb"])
def test_predictions_are_mirror_invariant(rule):
    mols = _mols(CORPUS)
    cfg = _config(mols, **rule)
    model = sieve.fit(_batch(mols, cfg), cfg)
    np.testing.assert_allclose(
        _predict(model, _inverted(mols), cfg).value,
        _predict(model, mols, cfg).value,
        rtol=1e-12, atol=1e-15,
    )


def test_an_unseen_enantiomer_is_answered_like_the_trained_one():
    r, s = _mols(["N[C@@H](C)C(=O)O"]), _mols(["N[C@H](C)C(=O)O"])
    cfg = _config(r + s)
    model = sieve.fit(_batch(r, cfg), cfg)
    pr, ps = _predict(model, r, cfg), _predict(model, s, cfg)
    np.testing.assert_allclose(pr.value, ps.value, rtol=1e-12, atol=1e-15)
    assert ps.stereo_refined.any()


def test_diastereomers_separate():
    a, meso = _mols(["C[C@H](Br)[C@H](C)Br"]), _mols(["C[C@H](Br)[C@@H](C)Br"])
    cfg = _config(a + meso, depth=4)
    lv = refine(from_rdkit(a + meso, config=cfg), cfg)[-1]
    c = _stereo_centre_rows(a[0])[0][0]
    assert lv.labels[c] != lv.labels[c + a[0].GetNumAtoms()]


def test_a_fit_on_the_mirrored_corpus_predicts_the_same():
    mols = _mols(CORPUS)
    cfg = _config(mols, **EB)
    orig = sieve.fit(_batch(mols, cfg), cfg)
    mirr = sieve.fit(_batch(_inverted(mols), cfg), cfg)
    np.testing.assert_allclose(
        _predict(orig, mols, cfg).value, _predict(mirr, mols, cfg).value,
        rtol=1e-12, atol=1e-15,
    )


@pytest.mark.parametrize("cuts", [(3,), (2, 5)], ids=["two", "three"])
def test_merge_monoid_with_enantiomers_split(cuts):
    import itertools

    mols = _mols(CORPUS + ["N[C@H](C)C(=O)O"])  # alanine's two hands in different shards
    cfg = _config(mols, **EB)
    batch = _batch(mols, cfg)
    edges = [0, *cuts, len(mols)]
    merged = None
    for lo, hi in itertools.pairwise(edges):
        part = sieve.fit(batch[(batch.graph_id >= lo) & (batch.graph_id < hi)], cfg)
        merged = part if merged is None else merged.merge(part)
    whole = sieve.fit(batch, cfg)
    assert [lv.n_classes for lv in merged.levels] == [lv.n_classes for lv in whole.levels]
    np.testing.assert_allclose(
        sieve.predict(merged, batch), sieve.predict(whole, batch), rtol=1e-12, atol=1e-15
    )
    for lm in merged.levels:
        mt = mirror_targets(lm)
        np.testing.assert_array_equal(mt[mt], np.arange(lm.n_classes))


def test_save_and_load_keep_mirror_of(tmp_path):
    mols = _mols(CORPUS)
    cfg = _config(mols)
    model = sieve.fit(_batch(mols, cfg), cfg)
    model.save(tmp_path / "m.npz")
    loaded = sieve.SieveModel.load(tmp_path / "m.npz")
    for a, b in zip(model.levels, loaded.levels, strict=True):
        np.testing.assert_array_equal(mirror_targets(a), mirror_targets(b))


def test_predictions_do_not_depend_on_a_pentavalent_neighbour():
    mols = _mols(CORPUS)
    p = _mols(["COP12(OC)NC(=O)O[C@]1(C(F)(F)F)c1ccccc1O2"])
    cfg = _config(mols + p, **EB)
    model = sieve.fit(_batch(mols + p, cfg), cfg)
    alone = sieve.predict(model, from_rdkit(mols, config=cfg))
    beside = sieve.predict(model, from_rdkit(mols + p, config=cfg))
    np.testing.assert_array_equal(alone, beside[: alone.shape[0]])
```

- [ ] **Step 2:** run → the merge and save/load tests fail (`mirror_of` dropped); the invariance tests may already pass, which is expected since prediction is unchanged.
- [ ] **Step 3: implement.** `merge_level`, after the `blind_of` block:

```python
    mirror_of = None
    if a.mirror_of is not None or b.mirror_of is not None:
        # Remapped like blind_of, within this level; the empty model reads
        # as the identity.
        mirror_of = np.arange(n_new, dtype=np.int64)
        mirror_of[:m] = mirror_targets(a)
        b_mirror_of = remap[mirror_targets(b)].astype(np.int64)
        if np.any(both) and not np.array_equal(mirror_of[i][both], b_mirror_of[both]):
            raise AssertionError("mirror_of disagreement: a class changed its mirror")
        mirror_of[i] = np.where(nA > 0, mirror_of[i], b_mirror_of)
    return FrozenLevel(uniq, count, mean, msd, parent, class_kind, blind_of, mirror_of), remap
```

  `model.save`: `if lvl.mirror_of is not None: arrays[f"level_{k}_mirror_of"] = lvl.mirror_of`; `load`: the eighth `FrozenLevel` argument `data[f"level_{k}_mirror_of"] if f"level_{k}_mirror_of" in data.files else None`.
- [ ] **Step 4:** all of `tests/test_tetrahedral.py` passes; `tests/test_merge.py tests/test_io.py` pass.
- [ ] **Step 5: commit** `feat(merge): carry mirror_of through merge and persistence`.

---

### Task 8: Study E

**Files:** `experiments/experiments/stereo_subsets.py`, `experiments/experiments/cli.py`, `experiments/workflows/cv_charges.sh`, `experiments/tests/test_stereo_subsets.py`.

- [ ] **Step 1: failing tests** (append):

```python
def test_tetrahedral_subsets_mark_distance_to_a_tagged_centre():
    from experiments.stereo_subsets import HAS_TET, NEAR_TET1, subset_masks

    mol = _mol("N[C@@H](C)CCO")  # heavy atoms 0..5, centre 1
    m = subset_masks(mol)
    assert m.shape == (6, mol.GetNumAtoms()) and m[HAS_TET].all()
    np.testing.assert_array_equal(m[NEAR_TET1, :6], [True, True, True, True, False, False])


def test_a_mask_table_from_an_older_subset_list_is_refused(tmp_path):
    from experiments.stereo_subsets import build_mask_table, load_mask_table

    root = _store(tmp_path, SMILES)
    path = build_mask_table("s", stores_root=root, n_jobs=1)
    z = dict(np.load(path))
    z["subsets"] = z["subsets"][:3]
    np.savez(path, **z)
    with pytest.raises(ValueError, match="rebuild"):
        load_mask_table(path)


def test_a_sidecar_missing_a_subset_is_stale(tmp_path):
    from experiments.stereo_subsets import METRICS_FILE, missing_scores

    runs = tmp_path / "runs"
    d = _cv_run(runs, "e", repeat=0, fold=0, method="a", depth=5, metrics={})
    (d / "predictions.npz").write_bytes(b"x")
    (d / METRICS_FILE).write_text(json.dumps({"has_ez/rmse": 1.0}))
    assert missing_scores(runs, {"e": ["a"]}, depth=5) == [d]
```

- [ ] **Step 2:** run → ImportError `HAS_TET`.
- [ ] **Step 3: implement.**
  - `SUBSETS = ("has_ez", "near_ez2", "near_ez1", "has_tet", "near_tet2", "near_tet1")`, with `HAS_TET, NEAR_TET2, NEAR_TET1 = 3, 4, 5`; `subset_masks` returns `(6, n)`, the three new rows from atoms whose chiral tag is `CHI_TETRAHEDRAL_CW/CCW`, by the same distance rule; the module docstring lists the new subsets.
  - The existing `test_masks_mark_distance_to_the_nearest_stereogenic_double_bond` asserts `masks.shape == (3, ...)`; change it to `(6, ...)`.
  - `_packed` packs generally: `np.bitwise_or.reduce([m[s].astype(np.uint8) << s for s in range(len(SUBSETS))])`.
  - `build_mask_table` also writes `subsets=np.array(SUBSETS)`; `load_mask_table` raises `ValueError(f"{path} holds subsets {...}, not {SUBSETS}; rebuild it with stereo-subset-masks")` on a mismatch or a missing key.
  - `missing_scores` also reports a run whose sidecar lacks any `f"{name}/rmse"` for the current `SUBSETS`.
  - `stereo-subset-masks --check`: exit 0 only if the file exists and `load_mask_table` accepts it.
  - Workflow: `study-d-subset-masks`'s guard becomes `"'$PYTHON' -m experiments stereo-subset-masks '$STORE' --check"`. After `study-d-report`, add Study E, mirroring Study D:

```bash
# ===========================================================================
# Study E: the tetrahedral track on top of cis/trans
# ===========================================================================
#
# docs/superpowers/specs/2026-09-24-tetrahedral-handedness-design.md section 9.
# Both tracks against Study D's cis/trans arm, reused and not re-run, at the
# same depth, repeats and K, under both estimators. Primary metric: RMSE
# within two bonds of a tagged tetrahedral centre. near_ez2 is the check on
# the one-chain simplification (spec section 8).
STUDY_E_CONFIG_LABEL=element-ctt-eb
SIEVE_STUDY_E=sieve-cv-study-e
STUDY_E_PARAMS=$(
  echo "$SIEVE_PREDICTOR_PARAMS" | "$PYTHON" -c '
import json, sys
params = json.load(sys.stdin)
params["stereo"] = ["cis_trans", "tetrahedral"]
print(json.dumps(params))
'
)
STUDY_E_VARIANTS='[
  {"method": "sieve-element-ctt-continuation",    "class_estimator": "continuation", "shrinkage_weight": null},
  {"method": "sieve-element-ctt-continuation-eb", "class_estimator": "continuation", "shrinkage_weight": "empirical_bayes"}
]'
STUDY_E_METHODS=sieve-element-ctt-continuation,sieve-element-ctt-continuation-eb

fit_one_study_e_shard() {
  "$PYTHON" -m experiments cv-fit-sieve-shards "$STORE" \
    --n-shards "$N_SHARDS" --max-depth "$STUDY_D_DEPTH" --shard "$1" \
    --codes-path "$CODES_PATH" --config-label "$STUDY_E_CONFIG_LABEL" \
    --predictor-params "$STUDY_E_PARAMS" \
    $COLLAPSE_FLAG
}
export -f fit_one_study_e_shard
export STUDY_E_CONFIG_LABEL STUDY_E_PARAMS

dispatch_study_e_shards() {
  all_shard_ids | xargs -P "$STUDY_D_SHARD_JOBS" -n 1 \
    bash -c 'fit_one_study_e_shard "$1"' --
}

step study-e-shard-fits \
  "shard_fits_count_is fit-sieve-$STUDY_E_CONFIG_LABEL-w$STUDY_D_DEPTH-s $N_SHARDS" -- \
  dispatch_study_e_shards

run_study_e_repeat() {
  "$PYTHON" -m experiments cv-run-sieve "$STORE" \
    --n-shards "$N_SHARDS" --k "$K" \
    --depths "$STUDY_D_DEPTH" --repeats "$1" \
    --codes-path "$CODES_PATH" --config-label "$STUDY_E_CONFIG_LABEL" \
    --fit-depth "$STUDY_D_DEPTH" \
    --predictor-params "$STUDY_E_PARAMS" \
    --variants "$STUDY_E_VARIANTS" \
    $MODEL_CACHE_FLAG \
    --normalization equal_weighted --method sieve-element-ctt-continuation \
    $COLLAPSE_FLAG \
    --experiment "$SIEVE_STUDY_E" \
    --save-predictions
}
export -f run_study_e_repeat
export SIEVE_STUDY_E STUDY_E_VARIANTS

study_e_runs_done() {
  local m
  for m in $(echo "$STUDY_E_METHODS" | tr ',' ' '); do
    method_depth_runs_count_is "$SIEVE_STUDY_E" "$m" "$STUDY_D_DEPTH" \
      "$((K * $(n_items "$STUDY_D_REPEATS")))" || return 1
  done
}

dispatch_study_e_repeats() {
  echo "$STUDY_D_REPEATS" | tr ',' '\n' \
    | xargs -P "$STUDY_D_JOBS" -n 1 bash -c 'run_study_e_repeat "$1"' --
}

step study-e-runs "study_e_runs_done" -- dispatch_study_e_repeats

STUDY_E_SELECT="--select $SIEVE_STUDY_D=$STUDY_D_METHODS --select $SIEVE_STUDY_E=$STUDY_E_METHODS"

step study-e-subset-scores \
  "'$PYTHON' -m experiments score-stereo-subsets '$STORE' $STUDY_E_SELECT --depth $STUDY_D_DEPTH --check" -- \
  "$PYTHON" -m experiments score-stereo-subsets "$STORE" $STUDY_E_SELECT \
    --depth "$STUDY_D_DEPTH"

STUDY_E_REPORT="$FIGURES_DIR/study-e"
STUDY_E_METRICS="near_tet2/rmse near_tet1/rmse has_tet/rmse near_tet2/mae near_ez2/rmse rmse mae"

study_e_report_is_up_to_date() {
  [ -f "$STUDY_E_REPORT.txt" ] || return 1
  [ -z "$(find "experiments/runs/$SIEVE_STUDY_D" "experiments/runs/$SIEVE_STUDY_E" \
            -name subset_metrics.json -newer "$STUDY_E_REPORT.txt" -print -quit)" ]
}

run_study_e_report() {
  local metric_flags="" m
  for m in $STUDY_E_METRICS; do metric_flags="$metric_flags --metric $m"; done
  "$PYTHON" -m experiments stereo-report \
    --pair "continuation=$SIEVE_STUDY_D:sieve-element-ct-continuation,$SIEVE_STUDY_E:sieve-element-ctt-continuation" \
    --pair "cont+EB=$SIEVE_STUDY_D:sieve-element-ct-continuation-eb,$SIEVE_STUDY_E:sieve-element-ctt-continuation-eb" \
    --depth "$STUDY_D_DEPTH" --k "$K" $metric_flags \
    --out "$STUDY_E_REPORT"
}

step study-e-report "study_e_report_is_up_to_date" -- run_study_e_report
```

- [ ] **Step 4:** `experiments/tests/test_stereo_subsets.py` passes; `bash -n experiments/workflows/cv_charges.sh`.
- [ ] **Step 5: commit** `feat(experiments): tetrahedral subsets and Study E`.

---

### Task 9: Gate, docs, PR, run

- [ ] Spec status: "implemented (plan: `docs/superpowers/plans/2026-09-24-tetrahedral-handedness.md`)"; `docs/superpowers/specs/2026-09-23-stereo-refines-the-blind-class-design.md` §9 points at the new spec.
- [ ] Docstrings mentioning a single stereo track (`refine.py` stereo block, `config.stereo_radices`) updated to both.
- [ ] The full gate (Global Constraints); `git status` clean apart from `message.md`, which is not committed.
- [ ] Push `tetrahedral-spec`; retitle PR #43 to cover implementation and Study E; merge when CI is green.
- [ ] On `main`: `CV_UNTIL=study-e-report bash experiments/workflows/cv_charges.sh > experiments/results/workflow-study-e.log 2>&1`, watching step events. The masks rebuild, Study B and D runs are rescored for the new keys, and Study D's report regenerates identically before Study E runs.
- [ ] Report `experiments/docs/figures/study-e.txt`, and measure spec §10.1 (the fraction of centres resolved per radius) from a fitted shard.
