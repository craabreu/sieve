# HOSE-Code Lookup Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plain HOSE-code lookup predictor as a comparison arm, so the manuscript can score the closest published precedent on the same corpus as every other arm.

**Architecture:** A pure key module turns a full HOSE code into per-radius keys by string prefix; a predictor accumulates a mean and a count per key at every radius during `fit`, and at `predict` walks the radius down to the deepest key with enough support. No continuation estimate, no shrinkage, no new harness machinery — it registers as an ordinary `Predictor` and runs through the existing `cv.py`/`compare.py` pipeline.

**Tech Stack:** Python, numpy, RDKit, `hosegen` (Stefan Kuhn's HOSE-code generator), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-hose-baseline-design.md`

## Global Constraints

- Backoff only. Do **not** port the continuation estimate (`src/sieve/continuation.py`) or the empirical-Bayes shrinkage (`src/sieve/shrinkage.py`) onto HOSE codes.
- Generate the full code **once per atom** at the deepest radius and cut shorter keys from it as string prefixes. Never call `get_Hose_codes` once per radius — the spec's §4 measures why.
- Leave the stereo parameters off: `usestereo`, `wedgebond`, `strict`, `ringsize` all stay at their defaults.
- Do **not** implement `NormalizablePredictor`. This arm is scored as it predicts.
- Defaults: `max_radius = 5`, `n_min = 1`. The generator caps at **12** spheres and raises a bare `IndexError` beyond that, so `max_radius` is validated against that ceiling in the constructor.
- The predictor name registered in the harness is `"hose"`.
- A depth sweep **regenerates** codes at each candidate `max_radius`. The containment that lets Sieve read a shallow setting out of a deep fit, and DASH out of truncated paths, does not hold for HOSE codes: branch ordering consults what lies beyond the sphere, so a deeper generation re-renders shallower spheres and 8.5% of *k*-prefixes change. The predictor needs no flag for this — a sweep is one instance per radius — but the code cache is reusable across folds at a fixed `max_radius` and **never** across settings. Spec §7.
- `hosegen` is an optional dependency. Nothing outside `predictors/hose.py` may import it at module scope, and every test touching it is skipped when it is absent.

## File Structure

| File | Responsibility |
|---|---|
| `experiments/experiments/predictors/hose_keys.py` | **Create.** The prefix-key construction, pure string handling, no optional imports so it is testable in the fast suite. |
| `experiments/experiments/predictors/hose.py` | **Create.** The predictor: `fit`, `predict`, the matched-radius diagnostic, and registration. |
| `experiments/experiments/predictors/__init__.py` | **Modify.** One lazy-import branch, following the existing `dash`/`sieve` pattern. |
| `pyproject.toml` (repo root) | **Modify.** A `hose` extra. The root pyproject finds packages in both `src` and `experiments`; there is no `experiments/pyproject.toml`. |
| `experiments/configs/hose-charge-example.yaml` | **Create.** A runnable config mirroring `dash-charge-example.yaml`. |
| `experiments/tests/test_hose_keys.py` | **Create.** Fast, no optional dependency. |
| `experiments/tests/test_predictor_hose.py` | **Create.** Synthetic `MoleculeSet`, skipped without `hosegen`. |
| `experiments/tests/test_predictor_hose_optional.py` | **Create.** End-to-end through `run()`, gated on the real prepared store. |

---

### Task 1: The prefix-key construction

**Files:**
- Create: `experiments/experiments/predictors/hose_keys.py`
- Test: `experiments/tests/test_hose_keys.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sphere_prefix(code: str, k: int) -> str` and the module constant `DELIMITERS: list[str]`.

- [ ] **Step 1: Write the failing test**

Create `experiments/tests/test_hose_keys.py`:

```python
"""Prefix keys for HOSE codes. No optional dependency: these are strings."""

from __future__ import annotations

from experiments.predictors.hose_keys import DELIMITERS, MAX_SPHERES, sphere_prefix

# hosegen.HoseGenerator().get_Hose_codes(ethanol, 0, max_radius=6)
ETHANOL_METHYL = "C-4;HHHC(HHO/H/)//"
PHENOL_CH = "C-3;H*C*C(H,H*C,*C/*CO,H*&/H*&,H)//"


def test_sphere_prefix_cuts_at_each_sphere_boundary():
    assert sphere_prefix(ETHANOL_METHYL, 1) == "C-4;HHHC("
    assert sphere_prefix(ETHANOL_METHYL, 2) == "C-4;HHHC(HHO/"
    assert sphere_prefix(ETHANOL_METHYL, 3) == "C-4;HHHC(HHO/H/"
    assert sphere_prefix(ETHANOL_METHYL, 4) == "C-4;HHHC(HHO/H/)"
    assert sphere_prefix(ETHANOL_METHYL, 5) == "C-4;HHHC(HHO/H/)/"


def test_each_key_is_a_prefix_of_the_next():
    """The property the backoff rests on: a class at radius k+1 sits inside
    exactly one class at radius k."""
    for code in (ETHANOL_METHYL, PHENOL_CH):
        for k in range(1, 8):
            assert sphere_prefix(code, k + 1).startswith(sphere_prefix(code, k))


def test_a_code_shorter_than_k_spheres_is_returned_unchanged():
    assert sphere_prefix("C-4;HHHC(", 5) == "C-4;HHHC("


def test_zero_spheres_is_the_empty_key():
    assert sphere_prefix(ETHANOL_METHYL, 0) == ""


def test_the_delimiter_table_matches_the_generator_cap():
    """12 is hosegen's own ceiling: it indexes sphere_delimiters directly, so
    a thirteenth sphere raises IndexError inside it."""
    assert len(DELIMITERS) == MAX_SPHERES == 12


def test_a_child_key_never_has_two_parents():
    """Spec acceptance check 2, over real codes hardcoded so the fast suite
    can run it without the generator: grouping by the radius-(k+1) key must
    never turn up two distinct radius-k keys."""
    codes = [
        "C-4;HHHC(HHO/H/)//",                   # ethanol, methyl C
        "C-4;HHCO(HHH,H//)//",                  # ethanol, methylene C
        "C-3;H*C*C(*CO,H*C/H*C,H,H*&/H*&)//",   # phenol, C bearing OH
        "C-3;H*C*C(H,H*C,*C/*CO,H*&/H*&,H)//",  # phenol, ortho CH
        "C-4;HHHC(=OO/,C/HHH)//",               # methyl acetate, acetyl CH3
        "C-3;=OCO(,HHH,C/HHH/)//",              # methyl acetate, carbonyl C
    ]
    for k in range(1, 6):
        parent_of: dict[str, str] = {}
        for code in codes:
            child, parent = sphere_prefix(code, k + 1), sphere_prefix(code, k)
            assert parent_of.setdefault(child, parent) == parent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/tests/test_hose_keys.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'experiments.predictors.hose_keys'`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/experiments/predictors/hose_keys.py`:

```python
"""Per-radius keys for HOSE codes, cut as prefixes of one full code.

Sphere ``i`` of a HOSE code is closed by ``DELIMITERS[i - 1]``, the positional
sequence ``hosegen.HoseGenerator.sphere_delimiters`` emits, so the first ``k``
spheres are a literal prefix of the whole string. That is what makes a key at
radius ``k + 1`` determine the key at radius ``k``, and hence every class sit
inside exactly one coarser class.

Generating a code separately at each radius does *not* have that property --
the branch grouping is rendered differently, and classes acquire more than one
parent. See §4 of docs/superpowers/specs/2026-09-16-hose-baseline-design.md,
which measures it. This module exists so that mistake cannot be made by
accident.
"""

from __future__ import annotations

# hosegen.HoseGenerator.sphere_delimiters, copied exactly: "(" closes sphere
# 1, ")" closes sphere 4, "/" closes every other. The generator indexes this
# list directly, so 12 is a hard ceiling -- asking it for a thirteenth sphere
# raises a bare IndexError from inside it, with no message of its own.
DELIMITERS: list[str] = ["(", "/", "/", ")"] + ["/"] * 8
MAX_SPHERES: int = len(DELIMITERS)


def sphere_prefix(code: str, k: int) -> str:
    """The first ``k`` spheres of ``code``, as a prefix of it.

    Returns ``code`` unchanged when it holds fewer than ``k`` spheres, so a
    query deeper than a molecule reaches degrades to its deepest description
    rather than raising."""
    pos = 0
    for i in range(k):
        nxt = code.find(DELIMITERS[i], pos)
        if nxt < 0:
            return code
        pos = nxt + 1
    return code[:pos]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/tests/test_hose_keys.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check experiments/experiments/predictors/hose_keys.py experiments/tests/test_hose_keys.py
git add experiments/experiments/predictors/hose_keys.py experiments/tests/test_hose_keys.py
git commit -m "feat(experiments): per-radius HOSE keys cut as prefixes of one code"
```

---

### Task 2: The predictor

**Files:**
- Create: `experiments/experiments/predictors/hose.py`
- Modify: `experiments/experiments/predictors/__init__.py`
- Modify: `pyproject.toml` (repo root)
- Test: `experiments/tests/test_predictor_hose.py`

**Interfaces:**
- Consumes: `sphere_prefix` from Task 1; `experiments.data.MoleculeSet`; `experiments.predictors.base.Prediction`; `experiments.predictors.register`.
- Produces: `HoseLookupPredictor(max_radius: int = 5, n_min: int = 1)` with `name = "hose"`, `fit(train, val, *, rng) -> None`, `predict(test) -> Prediction`, and the registry entry `"hose"`.

- [ ] **Step 1: Write the failing test**

Create `experiments/tests/test_predictor_hose.py`:

```python
"""HoseLookupPredictor against a synthetic MoleculeSet. Skipped without the
optional hosegen dependency."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tests.helpers import synthetic_molecule_set

pytest.importorskip("hosegen", reason="optional dependency: pip install -e '.[hose]'")


def _predictor(**kw):
    from experiments.predictors.hose import HoseLookupPredictor

    return HoseLookupPredictor(**kw)


def test_predicting_the_training_set_reproduces_its_class_means():
    """Every atom matches its own deepest class, so each prediction is the
    mean of the training atoms sharing that atom's full code."""
    train = synthetic_molecule_set(n_mol=8, seed=0)
    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(train).atom_value

    assert got.shape == (train.n_atoms,)
    assert np.isfinite(got).all()
    # the class means are an average of the targets, so they cannot leave the
    # range of the targets
    y = train.atom_target
    assert got.min() >= y.min() - 1e-12
    assert got.max() <= y.max() + 1e-12


def test_an_unseen_element_falls_back_to_the_global_mean():
    train = synthetic_molecule_set(n_mol=6, seed=1)
    # CCCl is in the synthetic set; CCBr is not, so bromine is unseen
    from rdkit import Chem

    from experiments.data import MoleculeSet

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("CCBr", params)
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("MBIScharge", 0.0)
    test = MoleculeSet(mols=[mol], atom_property="MBIScharge")

    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(test).atom_value

    bromine = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == "Br"]
    assert len(bromine) == 1
    assert got[bromine[0]] == pytest.approx(float(np.mean(train.atom_target)))


def test_n_min_refuses_a_class_below_the_threshold():
    """With a threshold above any class's support, every atom falls all the
    way back to the global mean."""
    train = synthetic_molecule_set(n_mol=4, seed=2)
    p = _predictor(max_radius=5, n_min=10_000)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(train).atom_value
    assert got == pytest.approx(np.full(train.n_atoms, float(np.mean(train.atom_target))))


def test_predict_before_fit_raises():
    train = synthetic_molecule_set(n_mol=2, seed=3)
    with pytest.raises(RuntimeError, match="fit must be called"):
        _predictor().predict(train)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="max_radius"):
        _predictor(max_radius=0)
    with pytest.raises(ValueError, match="n_min"):
        _predictor(n_min=0)


def test_a_radius_above_the_generator_ceiling_is_refused_up_front():
    """hosegen caps at 12 spheres and raises a bare IndexError past it. Catch
    that in the constructor rather than hours into a fit."""
    with pytest.raises(ValueError, match="12"):
        _predictor(max_radius=13)


def test_registered_under_its_name():
    from experiments.predictors import build

    p = build("hose", {"max_radius": 3})
    assert p.name == "hose"
    assert p.max_radius == 3


def test_counts_from_two_disjoint_halves_add_up_to_the_union():
    """Spec acceptance check 4. The accumulation is a plain sum over atoms, so
    fitting two halves and adding their counts must reproduce fitting the
    union. This checks the accumulation, not a property of the method."""
    from experiments.data import MoleculeSet

    whole = synthetic_molecule_set(n_mol=8, seed=5)
    half_a = MoleculeSet(mols=whole.mols[:4], atom_property=whole.atom_property)
    half_b = MoleculeSet(mols=whole.mols[4:], atom_property=whole.atom_property)

    fitted = []
    for part in (half_a, half_b, whole):
        p = _predictor(max_radius=5)
        p.fit(part, part, rng=np.random.default_rng(0))
        fitted.append(p)
    first, second, union = fitted

    for k in range(1, 6):
        keys = set(first._tables[k]) | set(second._tables[k])
        assert keys == set(union._tables[k])
        for key in keys:
            counted = (
                first._tables[k].get(key, (0.0, 0))[1]
                + second._tables[k].get(key, (0.0, 0))[1]
            )
            assert counted == union._tables[k][key][1]


def test_two_conformers_of_one_molecule_get_the_same_prediction():
    """Spec acceptance check 5. HOSE codes read the 2D graph, so conformers are
    indistinguishable to this arm and to the Sieve arms alike, which puts a
    floor under the error that neither can cross."""
    from rdkit import Chem

    from experiments.data import MoleculeSet

    train = synthetic_molecule_set(n_mol=8, seed=0)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mols = []
    for _ in range(2):
        mol = Chem.MolFromSmiles("CCO", params)
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 0.0)
        mols.append(mol)
    test = MoleculeSet(mols=mols, atom_property="MBIScharge")

    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    got = p.predict(test).atom_value

    n = mols[0].GetNumAtoms()
    assert got[:n] == pytest.approx(got[n:])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/tests/test_predictor_hose.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'experiments.predictors.hose'` (or the whole file skipped if `hosegen` is not installed — install it first, see Step 3)

- [ ] **Step 3: Declare the optional dependency**

In the repo-root `pyproject.toml`, add to `[project.optional-dependencies]`, after the `sklearn` entry:

```toml
# The HOSE-code baseline arm (experiments/predictors/hose.py). Stefan Kuhn's
# generator, a Python port of the Java CDK HOSECodeGenerator, not on PyPI.
# Two traps: the distribution is named hose-code-generator while the import is
# hosegen, and its install_requires names xmlrunner, which drags in unittest2
# and fails to build on modern Python. hosegen never imports xmlrunner, so
# install it with --no-deps; rdkit and numpy, all it actually needs, are
# already here:
#   pip install --no-deps "hose-code-generator @ git+https://github.com/Ratsemaat/HOSE-code-generator"
hose = ["hose-code-generator"]
```

Then install it into the working environment. Both flags matter: the
distribution name is not the import name, and `--no-deps` is what avoids the
`xmlrunner` -> `unittest2` build failure.

```bash
pip install --no-deps "hose-code-generator @ git+https://github.com/Ratsemaat/HOSE-code-generator"
```

- [ ] **Step 4: Write minimal implementation**

Create `experiments/experiments/predictors/hose.py`:

```python
"""A plain HOSE-code lookup baseline: the inference rule of Bremser's register
and of NMRShiftDB, scored on this corpus.

Backoff only. There is deliberately no continuation estimate and no shrinkage
here; see docs/superpowers/specs/2026-09-16-hose-baseline-design.md §1. The arm
answers what the incumbent lookup tradition does, and is not an ablation of
Sieve -- HOSE codes carry bond order and aromaticity, which the fitted Sieve
models in this series do not.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from experiments.data import MoleculeSet
from experiments.predictors import register
from experiments.predictors.base import Prediction
from experiments.predictors.hose_keys import MAX_SPHERES, sphere_prefix


class HoseLookupPredictor:
    name: ClassVar[str] = "hose"

    def __init__(self, max_radius: int = 5, n_min: int = 1) -> None:
        if max_radius < 1:
            raise ValueError(f"max_radius must be >= 1, got {max_radius}")
        if max_radius > MAX_SPHERES:
            raise ValueError(
                f"max_radius must be <= {MAX_SPHERES}, the generator's own "
                f"ceiling, got {max_radius}"
            )
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        self.max_radius = max_radius
        self.n_min = n_min
        self._tables: list[dict[str, tuple[float, int]]] | None = None
        self._global_mean: float | None = None

    def _codes(self, mols: list[Any]) -> Iterator[str]:
        """One full code per atom, in the flattened atom order of ``mols``.

        Generated once at ``max_radius``; every shorter key is cut from it by
        ``sphere_prefix``. The generator is imported here, not at module
        scope, so the harness loads without the optional dependency."""
        from hosegen import HoseGenerator

        generator = HoseGenerator()
        for mol in mols:
            for atom in mol.GetAtoms():
                yield generator.get_Hose_codes(
                    mol, atom.GetIdx(), max_radius=self.max_radius
                )

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        if train.n_conformers == 0:
            raise ValueError("hose requires a non-empty train split")
        target = train.atom_target
        sums: list[defaultdict[str, float]] = [
            defaultdict(float) for _ in range(self.max_radius + 1)
        ]
        counts: list[defaultdict[str, int]] = [
            defaultdict(int) for _ in range(self.max_radius + 1)
        ]
        for value, code in zip(target, self._codes(train.mols), strict=True):
            for k in range(1, self.max_radius + 1):
                key = sphere_prefix(code, k)
                sums[k][key] += float(value)
                counts[k][key] += 1
        self._tables = [
            {key: (sums[k][key] / counts[k][key], counts[k][key]) for key in counts[k]}
            for k in range(self.max_radius + 1)
        ]
        self._global_mean = float(np.mean(target))

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._tables is None or self._global_mean is None:
            raise RuntimeError("fit must be called before predict")
        atom_value: NDArray[np.float64] = np.empty(test.n_atoms, dtype=np.float64)
        for i, code in enumerate(self._codes(test.mols)):
            value = self._global_mean
            for k in range(self.max_radius, 0, -1):
                entry = self._tables[k].get(sphere_prefix(code, k))
                if entry is not None and entry[1] >= self.n_min:
                    value = entry[0]
                    break
            atom_value[i] = value
        return Prediction(atom_value=atom_value)


def _build(params: Mapping[str, Any]) -> HoseLookupPredictor:
    return HoseLookupPredictor(**params)


register("hose", _build)
```

- [ ] **Step 5: Register it lazily**

In `experiments/experiments/predictors/__init__.py`, inside `build`, add after the `dash_pretrained` branch:

```python
    if name == "hose" and name not in REGISTRY:
        import experiments.predictors.hose  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest experiments/tests/test_predictor_hose.py tests/test_hose_keys.py -v`
Expected: all passed, none skipped

- [ ] **Step 7: Lint and commit**

```bash
python -m ruff check experiments/experiments/predictors/hose.py experiments/experiments/predictors/__init__.py experiments/tests/test_predictor_hose.py
git add experiments/experiments/predictors/hose.py experiments/experiments/predictors/__init__.py pyproject.toml experiments/tests/test_predictor_hose.py
git commit -m "feat(experiments): a HOSE-code lookup baseline arm"
```

---

### Task 3: The matched-radius diagnostic

The fraction of atoms answered at each radius is the most informative number this arm produces, and the manuscript wants the same figure for Sieve. Expose it rather than recomputing it later.

**Files:**
- Modify: `experiments/experiments/predictors/hose.py`
- Test: `experiments/tests/test_predictor_hose.py`

**Interfaces:**
- Consumes: `HoseLookupPredictor` from Task 2.
- Produces: `HoseLookupPredictor.matched_radius -> NDArray[np.int64]`, one entry per atom of the most recent `predict` call, holding the radius that answered each atom and `0` where the global mean did.

- [ ] **Step 1: Write the failing test**

Append to `experiments/tests/test_predictor_hose.py`:

```python
def test_matched_radius_records_where_each_atom_was_answered():
    train = synthetic_molecule_set(n_mol=8, seed=0)
    p = _predictor(max_radius=5)
    p.fit(train, train, rng=np.random.default_rng(0))
    p.predict(train)

    radius = p.matched_radius
    assert radius.shape == (train.n_atoms,)
    # every training atom's own full code is in the table, so all are answered
    # at the deepest radius
    assert (radius == 5).all()


def test_matched_radius_is_zero_for_the_global_mean_fallback():
    train = synthetic_molecule_set(n_mol=4, seed=2)
    p = _predictor(max_radius=5, n_min=10_000)
    p.fit(train, train, rng=np.random.default_rng(0))
    p.predict(train)
    assert (p.matched_radius == 0).all()


def test_matched_radius_before_predict_raises():
    train = synthetic_molecule_set(n_mol=2, seed=3)
    p = _predictor()
    p.fit(train, train, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="predict must be called"):
        _ = p.matched_radius
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest experiments/tests/test_predictor_hose.py -k matched_radius -v`
Expected: FAIL, `AttributeError: 'HoseLookupPredictor' object has no attribute 'matched_radius'`

- [ ] **Step 3: Write minimal implementation**

In `experiments/experiments/predictors/hose.py`, add to `__init__`:

```python
        self._matched_radius: NDArray[np.int64] | None = None
```

Replace the body of `predict` with:

```python
    def predict(self, test: MoleculeSet) -> Prediction:
        if self._tables is None or self._global_mean is None:
            raise RuntimeError("fit must be called before predict")
        atom_value: NDArray[np.float64] = np.empty(test.n_atoms, dtype=np.float64)
        matched: NDArray[np.int64] = np.zeros(test.n_atoms, dtype=np.int64)
        for i, code in enumerate(self._codes(test.mols)):
            value, answered_at = self._global_mean, 0
            for k in range(self.max_radius, 0, -1):
                entry = self._tables[k].get(sphere_prefix(code, k))
                if entry is not None and entry[1] >= self.n_min:
                    value, answered_at = entry[0], k
                    break
            atom_value[i] = value
            matched[i] = answered_at
        self._matched_radius = matched
        return Prediction(atom_value=atom_value)
```

and add the accessor after `predict`:

```python
    @property
    def matched_radius(self) -> NDArray[np.int64]:
        """The radius that answered each atom of the most recent ``predict``,
        ``0`` where the global mean did. One entry per atom, in the same order
        as ``Prediction.atom_value``."""
        if self._matched_radius is None:
            raise RuntimeError("predict must be called before matched_radius")
        return self._matched_radius
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest experiments/tests/test_predictor_hose.py -v`
Expected: all passed

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check experiments/experiments/predictors/hose.py experiments/tests/test_predictor_hose.py
git add experiments/experiments/predictors/hose.py experiments/tests/test_predictor_hose.py
git commit -m "feat(experiments): record which radius answered each atom"
```

---

### Task 4: A runnable config, end to end

**Files:**
- Create: `experiments/configs/hose-charge-example.yaml`
- Test: `experiments/tests/test_predictor_hose_optional.py`

**Interfaces:**
- Consumes: the `"hose"` registry entry from Task 2.
- Produces: a config the CLI accepts, and proof the arm survives the real `run()` pipeline.

- [ ] **Step 1: Write the failing test**

Create `experiments/tests/test_predictor_hose_optional.py`:

```python
"""End-to-end HoseLookupPredictor test through the real run() pipeline, gated
on the real, already-split dash-molecules store and on the optional hosegen
dependency. Mirrors test_predictor_sieve_optional.py."""

from __future__ import annotations

import pytest

from experiments.tests.helpers import real_store_has_columns

pytestmark = pytest.mark.skipif(
    not real_store_has_columns("dash-molecules", "split"),
    reason="real dash-molecules store not prepared and split locally",
)


def test_hose_charge_predictor_runs_end_to_end_via_run(tmp_path):
    pytest.importorskip("hosegen")
    from experiments.config import (
        DataCfg,
        ExperimentCfg,
        PredictorCfg,
        RunCfg,
        TargetCfg,
    )
    from experiments.runner import run

    cfg = ExperimentCfg(
        run=RunCfg(experiment="hose-charge-optional", seed=0),
        data=DataCfg(store="dash-molecules", split_column="split"),
        target=TargetCfg(atom_property="MBIScharge", molecule_property="net_charge"),
        predictor=PredictorCfg(name="hose", params={"max_radius": 5}),
    )
    result = run(cfg, runs_root=tmp_path, allow_dirty=True, tracking=None, limit=200)
    assert result.metrics["n_test_conformers"] >= 0
```

- [ ] **Step 2: Run test to verify it fails or skips for the right reason**

Run: `python -m pytest experiments/tests/test_predictor_hose_optional.py -v -rs`
Expected: either FAIL (store present, predictor not wired into `run`) or SKIP with the reason "real dash-molecules store not prepared and split locally". A skip for any other reason means the gate is wrong.

- [ ] **Step 3: Write the config**

Create `experiments/configs/hose-charge-example.yaml`:

```yaml
run:
  experiment: hose-charges
  seed: 0
data:
  store: dash-molecules
  split_column: split
target:
  atom_property: MBIScharge
  molecule_property: net_charge
  label: charge (e)
predictor:
  name: hose
  params:
    max_radius: 5
    n_min: 1
```

- [ ] **Step 4: Run the test and the config**

Run: `python -m pytest experiments/tests/test_predictor_hose_optional.py -v -rs`
Expected: PASS where the store is prepared, SKIP where it is not.

Then, where the store is prepared, run the config itself and confirm it completes:

```bash
python -m experiments run experiments/configs/hose-charge-example.yaml
```

If the CLI's subcommand differs, read `experiments/experiments/cli.py` for the
real invocation rather than guessing.

- [ ] **Step 5: Commit**

```bash
git add experiments/configs/hose-charge-example.yaml experiments/tests/test_predictor_hose_optional.py
git commit -m "feat(experiments): a runnable HOSE arm config, checked end to end"
```

---

### Task 5: Measure one shard before committing to the full run

The spec's §6 measures 314 µs/atom at radius 5 on one core, single-threaded.
That predicts hours for the whole corpus, so find out on one shard whether that
holds here before starting a cross-validation that might run for a day.

**Files:**
- Create: `docs/superpowers/plans/2026-09-16-hose-baseline-results.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a recorded wall-clock and error figure, so the decision to run the full comparison is made on evidence.

- [ ] **Step 1: Time the featurization on one shard**

```bash
python - <<'PY'
import time
import numpy as np
from experiments.predictors.hose import HoseLookupPredictor
from experiments.tests.helpers import synthetic_molecule_set

ms = synthetic_molecule_set(n_mol=200, seed=0)
p = HoseLookupPredictor(max_radius=5)
t0 = time.perf_counter()
p.fit(ms, ms, rng=np.random.default_rng(0))
fit_s = time.perf_counter() - t0
t0 = time.perf_counter()
p.predict(ms)
predict_s = time.perf_counter() - t0
print(f"{ms.n_atoms} atoms: fit {fit_s:.2f}s, predict {predict_s:.2f}s, "
      f"{1e6 * fit_s / ms.n_atoms:.0f} us/atom fitting")
PY
```

- [ ] **Step 2: Fit and score one real shard**

Use the config from Task 4 against a single shard rather than the full
cross-validation. Read `experiments/experiments/cv.py` for how a single shard is
selected, and follow whatever `test_predictor_sieve_optional.py` does to reach
the real store.

- [ ] **Step 3: Record what came out**

Create `docs/superpowers/plans/2026-09-16-hose-baseline-results.md` holding: atoms
per second measured here, wall clock for one shard, the RMSE and R² on that
shard, and the distribution of `matched_radius` over its atoms. Numbers only,
with the command that produced each.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-09-16-hose-baseline-results.md
git commit -m "docs: one-shard timing and error for the HOSE arm"
```

- [ ] **Step 5: Price the sweep as well as the single fit**

If a Study-A-style depth sweep is wanted, it costs more here than it does for
the other two arms, because each point regenerates. From the spec's §7 table,
one pass at radius 10 is 1085 µs/atom while native passes over *k*=1…10 total
4491 µs/atom, a factor of 4.1. Multiply your measured per-atom rate from Step 1
by the corpus size both ways and record both figures, so the choice between a
single setting and a full sweep is made on wall clock rather than on hope.

- [ ] **Step 6: Stop and report**

Do **not** start the full cross-validation or the sweep. Report the numbers and
let the author decide whether the comparison is worth its wall clock, and at
which `max_radius` — the spec notes NMRShiftDB and Kuhn et al. both use six
spheres while this defaults to five, and both are worth reporting.

---

## Open questions for the author

These are decisions the plan does not make:

1. **Sphere count.** Defaults to 5, matching the refinement radius the
   manuscript selects for Sieve. NMRShiftDB and Kuhn et al. both use 6. The
   manuscript's Sec. 2.1 defers the number to its results section, so whichever
   is run has to be stated there.
2. **Whether the arm enters Table 3 and Fig. 4**, or the manuscript records that
   the comparison was not made. `TODO.md` in the manuscript repo tracks this.
