# DASH Charges Experiment Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold `charges_experiments/`, a second, fully independent experiment
series that reproduces DASH's MBIS atomic-partial-charge prediction task on
DASH's own published training SDF, with a data pipeline and two predictors
(DASH-tree, Sieve).

**Architecture:** Three phases, in dependency order: (1) the data pipeline —
vendored clustering, config/data/metrics plumbing, the `sieve` core adapter
change, and the runner/CLI harness, exercised end-to-end on synthetic data;
(2) the DASH-tree charge predictor, ported from `cosmo_experiments`'s
profile predictor down to a scalar target; (3) the Sieve charge predictor,
built directly on `sieve.io.rdkit_adapter`'s new `y_from_atom_prop` option.
Every module is written fresh for this series (no imports from
`sieve_experiments`/`cosmo_experiments`); the cosmo series is read only as a
reference for shape and convention.

**Tech Stack:** Python 3.11+, RDKit, numpy, pandas/pyarrow, PyYAML, pytest.
`sieve` (this repo's core package) is a real dependency, same as
`cosmo_experiments`. No cosmolayer, no torch, no mlflow-required plotting
(this series' spec drops the plots module entirely).

**Spec:** `docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`

## Global Constraints

- `charges_experiments/` is a fully independent package (`charge_experiments`)
  — zero imports from `sieve_experiments`/`cosmo_experiments`, at the harness
  level. Both series share only the core `sieve` package as a dependency.
- Charge target column is `MBIScharge` (verified against the real SDF at
  `~/tmp/dash_molecules/dashMoleculesSDF_v2.sdf`: the per-record property tag
  is exactly `>  <MBIScharge>`, one pipe-delimited float per atom, in
  molblock atom order).
- Source data: `~/tmp/dash_molecules/dashMoleculesSDF_v2.sdf`, 8,278,301,584
  bytes, md5 `305f521c6b422546bdf09c1e87eb922d`, from
  `https://www.research-collection.ethz.ch/server/api/core/bitstreams/4e827dd2-65a0-4305-9118-480ef5fce0b5/content`.
- Every conformer keeps its own row. No SMILES anywhere in this series' store
  or predictors — each row's `Mol` (`Mol.ToBinary()`/`Chem.Mol(blob)`) is the
  row payload, and the target rides on the `Mol`'s own atoms
  (`atom.SetDoubleProp("MBIScharge", value)`).
- Splitting: Butina clustering + a fraction-target cluster split, computed
  with `_chalcedon` (vendored from `cosmolayer.store._chalcedon`, itself
  vendored from `rowansci/chalcedon` to dodge that package's
  `python>=3.14` floor) — never RDKit's own Butina, never a reimplementation.
  Achiral fingerprints; clusters/splits grouped by `chembl_id`, never split
  across a molecule's own conformers.
- Metrics: MAE, RMSE, **R²** (no `max_abs_residual` — this reverses the cosmo
  series' `charge_metrics`, whose R²-skip was about near-zero-variance *net
  molecular* charge, not per-atom `MBIScharge`). A molecule-level
  charge-conservation check is a secondary diagnostic only.
- No GNN/Chemprop predictor and no size-/scaffold-biased second split in this
  pass — out of scope per the spec.

---

## File Structure

```
charges_experiments/
  charge_experiments/
    __init__.py
    __main__.py                # python -m charge_experiments ...
    cli.py                     # run / prepare-store / summarize subcommands
    config.py                  # YAML config loading, --set overrides
    data.py                    # MoleculeSet (Mol-blob backed), Mol<->blob helpers
    metrics.py                 # MAE/RMSE/R2, charge-conservation check
    runner.py                  # execute/_execute_inner, manifest+MLflow plumbing
    prepare_store.py           # download+verify SDF, parse to parquet, cluster+split
    _chalcedon/                # vendored from cosmolayer.store._chalcedon
      __init__.py
      tanimoto_similarity.py
      butina_cluster.py
      greedy_cluster_split.py
      NOTICE
    predictors/
      __init__.py               # lazy registry
      base.py                     # Predictor protocol, Prediction dataclass
      dash.py                       # ported tree-matching/back-off, scalar charge
      sieve_predictor.py             # SieveConfig(target_dim=1, ...)
  tests/
    conftest.py
    helpers.py                  # synthetic_molecule_set fixture
    test_charge_data.py
    test_charge_metrics.py
    test_charge_config.py
    test_charge_chalcedon.py
    test_charge_prepare_store.py
    test_charge_smoke.py
    test_charge_predictor_dash.py
    test_charge_predictor_dash_optional.py
    test_charge_predictor_sieve.py
    test_charge_predictor_sieve_optional.py
  configs/
  docs/
  pins.toml
  README.md
```

Modified existing files: root `pyproject.toml`, root `.gitignore`,
`src/sieve/io/rdkit_adapter.py`, `tests/test_rdkit_adapter.py`.

---

## Phase 1 — Data pipeline

### Task 1: Vendor `_chalcedon`

**Files:**
- Create: `charges_experiments/charge_experiments/_chalcedon/__init__.py`
- Create: `charges_experiments/charge_experiments/_chalcedon/tanimoto_similarity.py`
- Create: `charges_experiments/charge_experiments/_chalcedon/butina_cluster.py`
- Create: `charges_experiments/charge_experiments/_chalcedon/greedy_cluster_split.py`
- Create: `charges_experiments/charge_experiments/_chalcedon/NOTICE`

**Interfaces:**
- Produces: `_chalcedon.butina_cluster.butina_cluster(fingerprints, cutoff=0.65, ...) -> NDArray[np.intp]`
  and `_chalcedon.greedy_cluster_split.greedy_cluster_split(cluster_ids, fractions) -> dict[str, NDArray[np.intp]]`,
  both used by Task 6's `prepare_store.py`.

- [ ] **Step 1: Copy the three vendored modules verbatim**

Copy byte-for-byte from
`/home/craabreu/github-repos/cosmolayer/cosmolayer/store/_chalcedon/`:
`tanimoto_similarity.py`, `butina_cluster.py`, `greedy_cluster_split.py`.

```bash
mkdir -p charges_experiments/charge_experiments/_chalcedon
cp /home/craabreu/github-repos/cosmolayer/cosmolayer/store/_chalcedon/tanimoto_similarity.py \
   /home/craabreu/github-repos/cosmolayer/cosmolayer/store/_chalcedon/butina_cluster.py \
   /home/craabreu/github-repos/cosmolayer/cosmolayer/store/_chalcedon/greedy_cluster_split.py \
   charges_experiments/charge_experiments/_chalcedon/
```

- [ ] **Step 2: Fix the one intra-package import in `butina_cluster.py`**

The copied file imports
`from cosmolayer.store._chalcedon.tanimoto_similarity import (...)`. Edit it
to the new package's own path:

```python
from charge_experiments._chalcedon.tanimoto_similarity import (
    Precision,
    TanimotoSimilarity,
)
```

- [ ] **Step 3: Write `__init__.py`**

```python
"""Vendored subset of chalcedon (Butina clustering + fraction-target
cluster split). See ``NOTICE``."""
```

- [ ] **Step 4: Write `NOTICE`**

```
This directory vendors `tanimoto_similarity.py`, `butina_cluster.py`
(with a small, marked modification: an added `progress` parameter gating
the tqdm bars), and `greedy_cluster_split.py` (unmodified) from:

    https://github.com/rowansci/chalcedon
    commit 92da3cc5bd6ffb0d397cb49ea556f168d1d38b7e

by way of an intermediate vendored copy at
`cosmolayer/store/_chalcedon/` (https://github.com/craabreu/cosmolayer),
whose own NOTICE documents the same upstream commit and modification.
This copy is unmodified relative to that intermediate copy except for the
one import path fixed to this package's own module layout
(`cosmolayer.store._chalcedon.tanimoto_similarity` ->
`charge_experiments._chalcedon.tanimoto_similarity`).

chalcedon requires Python >=3.14 as installed from PyPI, but its source
uses no 3.14-only syntax; vendoring lets charge_experiments (Python
>=3.11) use its faster Butina implementation without raising its own
Python floor, and without taking cosmolayer itself as a dependency (this
series has no other reason to depend on cosmolayer -- see the design
spec's "Code sharing between series" decision).

Original license below.

MIT License

Copyright (c) 2026 Elias Mann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Write a smoke test for the vendored glue**

Create `charges_experiments/tests/test_charge_chalcedon.py`:

```python
"""The vendored _chalcedon modules import cleanly under this package's own
module path (the one thing Task 1's copy could get wrong) and reproduce the
doctested behavior from their upstream docstrings."""

from __future__ import annotations

import numpy as np


def test_butina_cluster_matches_upstream_doctest_example():
    from charge_experiments._chalcedon.butina_cluster import butina_cluster

    fingerprints = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.uint8
    )
    assert butina_cluster(fingerprints, cutoff=0.5).tolist() == [1, 1, 0, 0]


def test_greedy_cluster_split_matches_upstream_doctest_example():
    from charge_experiments._chalcedon.greedy_cluster_split import (
        greedy_cluster_split,
    )

    ids = np.array([0, 0, 0, 1, 1, 2, 3])
    result = greedy_cluster_split(ids, {"train": 0.6, "test": 0.4})
    assert result["train"].tolist() == [0, 1, 2, 5]
    assert result["test"].tolist() == [3, 4, 6]
```

- [ ] **Step 6: Run the test** (will fail with `ModuleNotFoundError:
  charge_experiments` until Task 2 registers the package with pytest/uv —
  acceptable to defer running this to the end of Task 2; note that here)

- [ ] **Step 7: Commit**

```bash
git add charges_experiments/charge_experiments/_chalcedon charges_experiments/tests/test_charge_chalcedon.py
git commit -m "feat(charges): vendor cosmolayer's _chalcedon clustering module

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 2: Package scaffolding and repo registration

**Files:**
- Create: `charges_experiments/charge_experiments/__init__.py`
- Create: `charges_experiments/charge_experiments/__main__.py`
- Create: `charges_experiments/README.md`
- Create: `charges_experiments/pins.toml`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing new.
- Produces: `charge_experiments` importable as a package; `pytest` collects
  `charges_experiments/tests`; `python -m charge_experiments` resolves (even
  though `cli.py` doesn't exist until Task 8 — `__main__.py` imports it
  lazily inside `main()`, matching `sieve_experiments/__main__.py`'s shape,
  so this task can land before `cli.py` exists as long as nothing calls
  `main()` yet).

- [ ] **Step 1: `charge_experiments/__init__.py`**

```python
"""charge_experiments: DASH atomic-partial-charge (MBIScharge) prediction
harness -- a second, fully independent experiment series alongside
cosmo_experiments (sigma-profile prediction). See
docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md.
"""

from __future__ import annotations
```

- [ ] **Step 2: `charge_experiments/__main__.py`**

```python
"""``python -m charge_experiments <command> ...``"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from charge_experiments.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Update `pyproject.toml`**

Add `"charges_experiments"` to `packages.find`'s `where`, add
`"charges_experiments/tests"` to `testpaths`, add
`"charges_experiments/**/*.py"` to ruff's `include`, add
`"charges_experiments"` to `[tool.ty.environment].extra-paths`:

```toml
[tool.setuptools.packages.find]
where = ["src", "cosmo_experiments", "charges_experiments"]
```
```toml
[tool.pytest.ini_options]
# cosmo_experiments/tests and charges_experiments/tests each hold one
# series' own tests, kept contained under their own series directory.
testpaths = ["tests", "cosmo_experiments/tests", "charges_experiments/tests"]
filterwarnings = ["error::RuntimeWarning"]
```
```toml
[tool.ruff]
line-length = 88
target-version = "py311"
include = [
    "src/**/*.py",
    "tests/**/*.py",
    "cosmo_experiments/**/*.py",
    "charges_experiments/**/*.py",
]
```
```toml
[tool.ty.environment]
extra-paths = ["cosmo_experiments", "charges_experiments"]
```

No new `[tool.ty.overrides]` entry is needed yet (charges_experiments has no
pandas/mlflow-typed module comparable to `runner.py`'s `np.savez` false
positive until Task 8 proves otherwise — revisit then if `ty` actually
flags it).

- [ ] **Step 4: Update `.gitignore`**

Find the existing block:
```
# Experiment outputs and external checkouts (see cosmo_experiments/README.md).
# Note: cosmo_experiments/pins.toml is tracked and lives one level up,
...
cosmo_experiments/runs/
cosmo_experiments/mlruns/
cosmo_experiments/cache/
cosmo_experiments/external/
cosmo_experiments/results/
```
Add immediately after it:
```

# Same shape for the charges series (see charges_experiments/README.md).
# charges_experiments/pins.toml is tracked, same as cosmo_experiments/pins.toml.
charges_experiments/runs/
charges_experiments/mlruns/
charges_experiments/cache/
charges_experiments/results/
charges_experiments/stores/
```

(`charges_experiments/stores/` is new relative to the cosmo series' list:
this series' `molecules.parquet` — parsed from an 8.3GB SDF — lives under
its own `stores/` the same way `cosmo_experiments`' chaos-store does under
the shared top-level `stores/`, which is already git-ignored elsewhere in
this file; add the charges-local one explicitly for clarity even if
redundant with any existing top-level `stores/` ignore.)

- [ ] **Step 5: `charges_experiments/README.md`**

```markdown
# charges_experiments

A second, independent experiment series: predicting DASH's MBIS atomic
partial charges (the `MBIScharge` SDF property) on DASH's own published
training data (`dashMoleculesSDF_v2.sdf`, ETH Research Collection).

Fully independent of `cosmo_experiments` (sigma-profile prediction) at the
harness level -- no shared package, only the core `sieve` dependency in
common. See
`docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md`
for the full design.

## Usage

    uv run python -m charge_experiments prepare-store
    uv run python -m charge_experiments run --config configs/dash-charge-example.yaml
    uv run python -m charge_experiments summarize
```

- [ ] **Step 6: `charges_experiments/pins.toml`**

```toml
# Pinned external repos for the charges experiments (see predictors/dash.py).
# Mirrors cosmo_experiments/pins.toml's [dash_tree] entry -- both series use
# the same rinikerlab/DASH-tree clone (its published topology, not its
# training pipeline), but each series pins and clones it independently
# (charges_experiments/external/, not shared with cosmo_experiments/external/)
# to keep the two series' import/build graphs fully separate.

[dash_tree]
url = "https://github.com/rinikerlab/DASH-tree.git"
commit = "6cf1b2351c4674e602153dd493c06d9c020fc9ce"  # main, 2026-08-24
license = "MIT"
notes = """
See cosmo_experiments/pins.toml's own [dash_tree] entry for the full story
(preload=True gotcha, out-of-vocabulary atom failure mode, match_new_atom's
O(n^2) neighbor-dict rebuild) -- identical here, since this series uses the
same pinned commit's same match_new_atom/DASHTree machinery, just to
predict a different (scalar, not profile) property.
"""
```

- [ ] **Step 7: Verify the package resolves and Task 1's test collects**

```bash
uv run python -c "import charge_experiments; print('ok')"
uv run pytest charges_experiments/tests/test_charge_chalcedon.py -v
```

Expected: both tests from Task 1 now PASS.

- [ ] **Step 8: Commit**

```bash
git add charges_experiments/charge_experiments/__init__.py \
  charges_experiments/charge_experiments/__main__.py \
  charges_experiments/README.md charges_experiments/pins.toml \
  pyproject.toml .gitignore
git commit -m "feat(charges): scaffold charge_experiments package, register in pyproject

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 3: `data.py` — Mol-blob `MoleculeSet`

**Files:**
- Create: `charges_experiments/charge_experiments/data.py`
- Create: `charges_experiments/tests/helpers.py`
- Create: `charges_experiments/tests/test_charge_data.py`

**Interfaces:**
- Produces:
  `mol_to_blob(mol) -> bytes`, `blob_to_mol(blob: bytes) -> Chem.Mol`,
  `molecule_sum(per_atom, mol_id, n_molecules) -> NDArray[np.float64]`,
  `MoleculeSet` dataclass with fields `chembl_id: list[str]`,
  `conf_id: list[str]`, `mols: list[Any]`, `net_charge: NDArray[np.float64]`,
  `split: list[str] | None`, properties `n_conformers`, `num_atoms`,
  `n_atoms`, `atom_mol_id`, `atom_charge`, method `select(mask)`.
  `REPO_ROOT`, `DEFAULT_STORES_ROOT = REPO_ROOT / "charges_experiments" / "stores"`.
- Consumed by: Task 6 (`prepare_store.py` writes/reads rows this shape
  describes), Task 8 (`runner.py`), Task 10/12 (predictors).

- [ ] **Step 1: Write the failing round-trip test**

```python
# charges_experiments/tests/test_charge_data.py
"""Pure-rdkit tests for data.py's Mol-blob serialize/deserialize round trip
and MoleculeSet -- no store, no download needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _mol_with_charges(smiles: str, charges: list[float]):
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    assert mol.GetNumAtoms() == len(charges)
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)
    return mol


def test_mol_to_blob_round_trip_preserves_atom_properties():
    from charge_experiments.data import blob_to_mol, mol_to_blob

    mol = _mol_with_charges("CO", [-0.1, 0.1])
    blob = mol_to_blob(mol)
    assert isinstance(blob, bytes)

    restored = blob_to_mol(blob)
    assert restored.GetNumAtoms() == 2
    restored_charges = [a.GetDoubleProp("MBIScharge") for a in restored.GetAtoms()]
    assert restored_charges == pytest.approx([-0.1, 0.1])


def test_mol_to_blob_round_trip_preserves_chiral_tags():
    from rdkit import Chem

    from charge_experiments.data import blob_to_mol, mol_to_blob

    mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
    assert mol is not None
    original_tags = [a.GetChiralTag() for a in mol.GetAtoms()]
    assert any(t != Chem.ChiralType.CHI_UNSPECIFIED for t in original_tags)

    restored = blob_to_mol(mol_to_blob(mol))
    restored_tags = [a.GetChiralTag() for a in restored.GetAtoms()]
    assert restored_tags == original_tags
```

- [ ] **Step 2: Run it to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'charge_experiments.data'`.

- [ ] **Step 3: Write `data.py`**

```python
"""Molecule/atom data for the charges experiment harness.

``MoleculeSet``, ``molecule_sum``, ``mol_to_blob``/``blob_to_mol`` are pure
rdkit + numpy -- no pandas, no network -- so they are importable and
testable without touching the real (8.3GB source / parsed parquet) store.
See charges_experiments/tests/test_charge_data.py and the
``synthetic_molecule_set`` fixture in charges_experiments/tests/helpers.py.

Unlike cosmo_experiments' MoleculeSet, there is no SMILES field anywhere: a
conformer's target (``MBIScharge``) rides directly on its own RDKit ``Mol``
as a real atom property, and the store persists the serialized ``Mol``
itself -- see the design spec's "Store row format" decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES_ROOT = REPO_ROOT / "charges_experiments" / "stores"
DEFAULT_CACHE_DIR = REPO_ROOT / "charges_experiments" / "cache"

# AtomProps carries MBIScharge (set via atom.SetDoubleProp); MolProps is
# cheap to include too and covers any future mol-level property. Chiral
# tags and 3D conformer coordinates are intrinsic Atom/Mol fields, not
# properties, so they survive ToBinary()/Chem.Mol() regardless of
# propertyFlags -- passed explicitly here anyway, rather than relying on
# rdkit's own global pickle-property default, so this doesn't silently
# break if that default ever changes upstream.
_PICKLE_PROPS = None  # set lazily in mol_to_blob (see its own docstring)


def mol_to_blob(mol: Any) -> bytes:
    """Serialize ``mol`` to bytes, preserving atom/mol properties (which is
    where ``MBIScharge`` lives) and stereo/chiral tags (intrinsic, always
    preserved)."""
    from rdkit import Chem

    return mol.ToBinary(
        Chem.PropertyPickleOptions.AtomProps | Chem.PropertyPickleOptions.MolProps
    )


def blob_to_mol(blob: bytes) -> Any:
    """Deserialize a blob written by ``mol_to_blob`` back into a ``Mol``."""
    from rdkit import Chem

    return Chem.Mol(blob)


def molecule_sum(
    per_atom: NDArray[np.floating], mol_id: NDArray[np.int64], n_molecules: int
) -> NDArray[np.float64]:
    """Sum per-atom values into per-conformer rows. A plain sum: the
    conformer's own net charge is the sum of its atoms' real partial
    charges, no averaging or normalization involved."""
    per_atom = np.asarray(per_atom, dtype=np.float64)
    out = np.zeros(n_molecules, dtype=np.float64)
    np.add.at(out, mol_id, per_atom)
    return out


@dataclass(frozen=True)
class MoleculeSet:
    """One split's worth of conformers. Each entry in ``mols`` is one
    conformer's own RDKit ``Mol``, atoms carrying ``MBIScharge`` as a real
    double property -- there is no separate, position-aligned target array
    to keep in sync (contrast cosmo_experiments' ``MoleculeSet``, which
    carries ``mol_profile``/``atom_profile`` as arrays parallel to
    ``smiles``)."""

    chembl_id: list[str]
    conf_id: list[str]
    mols: list[Any]
    net_charge: NDArray[np.float64]
    split: list[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.mols)
        if len(self.chembl_id) != n:
            raise ValueError("chembl_id must have one entry per conformer")
        if len(self.conf_id) != n:
            raise ValueError("conf_id must have one entry per conformer")
        if len(self.net_charge) != n:
            raise ValueError("net_charge must have one entry per conformer")
        if self.split is not None and len(self.split) != n:
            raise ValueError("split must have one entry per conformer")

    @property
    def n_conformers(self) -> int:
        return len(self.mols)

    @property
    def num_atoms(self) -> NDArray[np.int64]:
        return np.array([m.GetNumAtoms() for m in self.mols], dtype=np.int64)

    @property
    def n_atoms(self) -> int:
        return int(self.num_atoms.sum()) if self.mols else 0

    @property
    def atom_mol_id(self) -> NDArray[np.int64]:
        """Conformer index of each atom, e.g. [0,0,0,1,1,2,...]."""
        return np.repeat(np.arange(self.n_conformers), self.num_atoms)

    @property
    def atom_charge(self) -> NDArray[np.float64]:
        """Per-atom ``MBIScharge`` ground truth, flattened across every
        conformer's own atom order."""
        if not self.mols:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(
            [
                np.array(
                    [a.GetDoubleProp("MBIScharge") for a in m.GetAtoms()],
                    dtype=np.float64,
                )
                for m in self.mols
            ]
        )

    def select(self, mol_mask: NDArray[np.bool_]) -> MoleculeSet:
        """The sub-set of conformers where ``mol_mask`` is True. The only
        place a split mask is applied."""
        mol_mask = np.asarray(mol_mask, dtype=bool)
        idx = np.flatnonzero(mol_mask)
        return MoleculeSet(
            chembl_id=[self.chembl_id[i] for i in idx],
            conf_id=[self.conf_id[i] for i in idx],
            mols=[self.mols[i] for i in idx],
            net_charge=np.asarray(self.net_charge)[mol_mask],
            split=None if self.split is None else [self.split[i] for i in idx],
        )
```

- [ ] **Step 4: Run the tests again**

```bash
uv run pytest charges_experiments/tests/test_charge_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Write the `synthetic_molecule_set` fixture**

```python
# charges_experiments/tests/helpers.py
"""Fixtures shared across charges_experiments' test suite."""

from __future__ import annotations

import numpy as np


def synthetic_molecule_set(n_mol: int = 8, seed: int = 0):
    """A small, fully-populated ``MoleculeSet`` for fast harness tests --
    real RDKit ``Mol`` objects (small alkanes/alcohols), each atom carrying a
    fabricated but deterministic ``MBIScharge``, net_charge computed to be
    exactly consistent with it (sum of atom charges), so both the input and
    any rollup can be checked exactly."""
    from rdkit import Chem

    from charge_experiments.data import MoleculeSet, molecule_sum

    rng = np.random.default_rng(seed)
    base_smiles = ["CO", "CCO", "CCC", "CC(C)O", "CCCC", "CC(=O)O", "CCN", "CCCl"]
    smiles = [base_smiles[i % len(base_smiles)] for i in range(n_mol)]

    mols = []
    num_atoms = []
    for i, smi in enumerate(smiles):
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        assert mol is not None, smi
        n_atoms = mol.GetNumAtoms()
        charges = rng.normal(scale=0.2, size=n_atoms)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("MBIScharge", float(charge))
        mols.append(mol)
        num_atoms.append(n_atoms)

    atom_charge = np.concatenate(
        [np.array([a.GetDoubleProp("MBIScharge") for a in m.GetAtoms()]) for m in mols]
    )
    mol_id = np.repeat(np.arange(n_mol), num_atoms)
    net_charge = molecule_sum(atom_charge, mol_id, n_mol)

    chembl_id = [f"CHEMBL{1000 + i // 2}" for i in range(n_mol)]  # 2 conformers/id
    conf_id = [f"conf_{i % 2:02d}" for i in range(n_mol)]

    return MoleculeSet(
        chembl_id=chembl_id, conf_id=conf_id, mols=mols, net_charge=net_charge
    )
```

- [ ] **Step 6: Add a test exercising the fixture + `select`**

Append to `charges_experiments/tests/test_charge_data.py`:

```python
def test_synthetic_molecule_set_select_preserves_alignment():
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=0)
    assert mset.n_conformers == 8
    assert mset.n_atoms == int(mset.num_atoms.sum())

    mask = np.array([True, False, True, False, True, False, True, False])
    sub = mset.select(mask)
    assert sub.n_conformers == 4
    assert sub.chembl_id == [mset.chembl_id[i] for i in range(8) if mask[i]]
    np.testing.assert_array_equal(sub.net_charge, mset.net_charge[mask])
    # atom_charge stays consistent with net_charge after selection
    from charge_experiments.data import molecule_sum

    resummed = molecule_sum(sub.atom_charge, sub.atom_mol_id, sub.n_conformers)
    np.testing.assert_allclose(resummed, sub.net_charge, atol=1e-8)
```

- [ ] **Step 7: Run the whole data test file**

```bash
uv run pytest charges_experiments/tests/test_charge_data.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add charges_experiments/charge_experiments/data.py charges_experiments/tests/helpers.py charges_experiments/tests/test_charge_data.py
git commit -m "feat(charges): add Mol-blob MoleculeSet (data.py)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 4: `metrics.py`

**Files:**
- Create: `charges_experiments/charge_experiments/metrics.py`
- Create: `charges_experiments/tests/test_charge_metrics.py`

**Interfaces:**
- Consumes: `charge_experiments.data.molecule_sum`.
- Produces: `regression_metrics(y_true, y_pred) -> dict[str, float]` (keys
  `mae`, `rmse`, `r2`), `charge_conservation_metrics(atom_charge_pred,
  mol_id, net_charge_true, n_molecules) -> dict[str, float]` (keys
  `mae`, `rmse`, `r2`, all `charge_conservation/`-prefixed by the caller).

- [ ] **Step 1: Write the failing test**

```python
# charges_experiments/tests/test_charge_metrics.py
"""Pure-numpy metrics tests -- hand-computed numbers, no rdkit needed."""

from __future__ import annotations

import numpy as np
import pytest


def test_regression_metrics_mae_rmse_r2_hand_computed():
    from charge_experiments.metrics import regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 5.0])

    out = regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(np.sqrt((0 + 0 + 0 + 4) / 4))
    ss_res = 4.0
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    assert out["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_regression_metrics_perfect_prediction_is_r2_one():
    from charge_experiments.metrics import regression_metrics

    y = np.array([-0.3, 0.1, 0.5, -0.1])
    out = regression_metrics(y, y.copy())
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)


def test_regression_metrics_empty_input_is_nan_not_a_crash():
    """pyproject.toml promotes RuntimeWarning to an error, so a bare
    np.mean(empty) must never be reached."""
    from charge_experiments.metrics import regression_metrics

    out = regression_metrics(np.array([]), np.array([]))
    assert np.isnan(out["mae"])
    assert np.isnan(out["rmse"])
    assert np.isnan(out["r2"])


def test_charge_conservation_metrics_sums_atoms_per_conformer():
    from charge_experiments.metrics import charge_conservation_metrics

    # Two conformers: atoms [0,0,1,1,1] -> conformer 0 has 2 atoms, conformer 1 has 3.
    mol_id = np.array([0, 0, 1, 1, 1])
    atom_charge_pred = np.array([0.2, -0.1, 0.05, 0.05, -0.2])
    net_charge_true = np.array([0.0, 0.0])

    out = charge_conservation_metrics(atom_charge_pred, mol_id, net_charge_true, 2)
    pred_sums = np.array([0.1, -0.1])
    expected_mae = float(np.mean(np.abs(pred_sums - net_charge_true)))
    assert out["mae"] == pytest.approx(expected_mae)


def test_charge_conservation_metrics_perfect_conservation_is_zero_error():
    from charge_experiments.metrics import charge_conservation_metrics

    mol_id = np.array([0, 0, 1])
    atom_charge_pred = np.array([0.5, -0.5, 1.0])
    net_charge_true = np.array([0.0, 1.0])

    out = charge_conservation_metrics(atom_charge_pred, mol_id, net_charge_true, 2)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_metrics.py -v
```

Expected: FAIL, `ModuleNotFoundError: No module named 'charge_experiments.metrics'`.

- [ ] **Step 3: Write `metrics.py`**

```python
"""Shared metrics for the charges experiment harness.

Pure numpy: no rdkit, no pandas, no mlflow -- unit-testable with hand
computed numbers (see charges_experiments/tests/test_charge_metrics.py).

Unlike cosmo_experiments' ``charge_metrics`` (MAE/RMSE/max_abs_residual, no
R2 -- net *molecular* charge clusters near zero, destabilizing ss_tot), this
series' primary target is per-atom ``MBIScharge``, which has real spread, so
R2 is informative and reported; ``max_abs_residual`` is dropped. See the
design spec's Metrics decision.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import molecule_sum


def regression_metrics(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> dict[str, float]:
    """MAE/RMSE/R2 flattened over all elements.

    NaN for every key on empty input rather than a RuntimeWarning-turned-
    error from averaging zero elements -- pyproject.toml promotes
    RuntimeWarning to an error.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def charge_conservation_metrics(
    atom_charge_pred: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    net_charge_true: NDArray[np.floating],
    n_molecules: int,
) -> dict[str, float]:
    """Secondary diagnostic: how well each conformer's summed predicted atom
    charges reproduce its own molblock ``M CHG`` total (``net_charge`` --
    unlike cosmo_experiments' sigma-derived "charge", there is no sign flip
    here: ``MBIScharge`` is a real atomic partial charge, and a conformer's
    atoms should sum to its own formal charge directly).
    """
    pred_sum = molecule_sum(atom_charge_pred, mol_id, n_molecules)
    return regression_metrics(net_charge_true, pred_sum)
```

- [ ] **Step 4: Run again**

```bash
uv run pytest charges_experiments/tests/test_charge_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charges_experiments/charge_experiments/metrics.py charges_experiments/tests/test_charge_metrics.py
git commit -m "feat(charges): add metrics.py (MAE/RMSE/R2, charge conservation)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 5: `config.py`

**Files:**
- Create: `charges_experiments/charge_experiments/config.py`
- Create: `charges_experiments/tests/test_charge_config.py`

**Interfaces:**
- Produces: `RunCfg(experiment, seed, tags)`, `DataCfg(store, split_column,
  train_split="train", val_split="val", eval_split="test")`,
  `PredictorCfg(name, params)`, `ExperimentCfg(run, data, predictor)`,
  `load_config(path, overrides=()) -> ExperimentCfg`,
  `apply_overrides(raw, overrides) -> dict`, `to_dict(cfg) -> dict`,
  `to_flat_params(cfg) -> dict[str, str]`.
- Consumed by: Task 8 (`runner.py`, `cli.py`).

This mirrors `cosmo_experiments/sieve_experiments/config.py` exactly, minus
the `scheme` field (no sigma-averaging scheme concept in this series) and
`VALID_SPLIT_COLUMNS` narrowed to this series' one split column.

- [ ] **Step 1: Write the failing test**

```python
# charges_experiments/tests/test_charge_config.py
from __future__ import annotations

import pytest
import yaml


def _write_yaml(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base_raw():
    return {
        "run": {"experiment": "charge-smoke", "seed": 0},
        "data": {"store": "dash-molecules", "split_column": "split"},
        "predictor": {"name": "global_mean", "params": {}},
    }


def test_load_config_round_trips_a_minimal_yaml(tmp_path):
    from charge_experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    assert cfg.run.experiment == "charge-smoke"
    assert cfg.run.seed == 0
    assert cfg.data.store == "dash-molecules"
    assert cfg.data.split_column == "split"
    assert cfg.data.train_split == "train"
    assert cfg.data.val_split == "val"
    assert cfg.data.eval_split == "test"
    assert cfg.predictor.name == "global_mean"


def test_load_config_rejects_unknown_top_level_key(tmp_path):
    from charge_experiments.config import load_config

    raw = _base_raw()
    raw["bogus"] = 1
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_load_config_rejects_invalid_split_column(tmp_path):
    from charge_experiments.config import load_config

    raw = _base_raw()
    raw["data"]["split_column"] = "not_a_real_column"
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValueError, match="split_column"):
        load_config(path)


def test_load_config_applies_set_overrides(tmp_path):
    from charge_experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path, overrides=["predictor.params.max_wl_depth=3"])
    assert cfg.predictor.params["max_wl_depth"] == 3


def test_to_dict_and_to_flat_params_round_trip(tmp_path):
    from charge_experiments.config import to_dict, to_flat_params
    from charge_experiments.config import load_config

    path = _write_yaml(tmp_path, _base_raw())
    cfg = load_config(path)
    d = to_dict(cfg)
    assert d["run"]["experiment"] == "charge-smoke"
    flat = to_flat_params(cfg)
    assert flat["data.store"] == "dash-molecules"
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_config.py -v
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write `config.py`**

```python
"""Run configuration: YAML in, frozen dataclasses out. Mirrors
cosmo_experiments/sieve_experiments/config.py's shape; this series drops the
`scheme` field (no sigma-averaging-scheme concept here) and has exactly one
valid split column (`split` -- this series builds no size-biased second
split, per the design spec's "Out of scope" list)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_SPLIT_COLUMNS = ("split",)

_RUN_KEYS = {"experiment", "seed", "tags"}
_DATA_KEYS = {"store", "split_column", "train_split", "val_split", "eval_split"}
_PREDICTOR_KEYS = {"name", "params"}
_TOP_KEYS = {"run", "data", "predictor"}


@dataclass(frozen=True)
class RunCfg:
    experiment: str
    seed: int
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataCfg:
    store: str
    split_column: str
    train_split: str = "train"
    val_split: str = "val"
    eval_split: str = "test"

    def __post_init__(self) -> None:
        if self.split_column not in VALID_SPLIT_COLUMNS:
            raise ValueError(
                f"data.split_column must be one of {VALID_SPLIT_COLUMNS}, "
                f"got {self.split_column!r}"
            )


@dataclass(frozen=True)
class PredictorCfg:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentCfg:
    run: RunCfg
    data: DataCfg
    predictor: PredictorCfg


def _check_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"unknown key(s) in {where}: {sorted(extra)}")


def _parse_scalar(text: str) -> Any:
    """Best-effort str -> int/float/bool for ``--set`` overrides."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a raw (pre-validation) config dict."""
    out = copy.deepcopy(dict(raw))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be 'key.path=value', got {override!r}")
        path, value = override.split("=", 1)
        keys = path.split(".")
        node = out
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _parse_scalar(value)
    return out


def _build(raw: Mapping[str, Any]) -> ExperimentCfg:
    _check_keys(raw, _TOP_KEYS, "config")
    for section in ("run", "data", "predictor"):
        if section not in raw:
            raise ValueError(f"config is missing required section {section!r}")

    run_raw = raw["run"]
    _check_keys(run_raw, _RUN_KEYS, "run")
    run = RunCfg(
        experiment=run_raw["experiment"],
        seed=run_raw["seed"],
        tags=dict(run_raw.get("tags", {})),
    )

    data_raw = raw["data"]
    _check_keys(data_raw, _DATA_KEYS, "data")
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
    _check_keys(predictor_raw, _PREDICTOR_KEYS, "predictor")
    predictor = PredictorCfg(
        name=predictor_raw["name"], params=dict(predictor_raw.get("params", {}))
    )

    return ExperimentCfg(run=run, data=data, predictor=predictor)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentCfg:
    """Load and validate a YAML config file, applying ``--set`` overrides."""
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(raw)


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = str(value)


def to_dict(cfg: ExperimentCfg) -> dict[str, Any]:
    """The resolved config as a plain nested dict -- what gets written to
    ``config.resolved.yaml`` in a run directory."""
    return {
        "run": {
            "experiment": cfg.run.experiment,
            "seed": cfg.run.seed,
            "tags": dict(cfg.run.tags),
        },
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "eval_split": cfg.data.eval_split,
        },
        "predictor": {"name": cfg.predictor.name, "params": dict(cfg.predictor.params)},
    }


def to_flat_params(cfg: ExperimentCfg) -> dict[str, str]:
    """Flatten a config to dot-separated string params, for MLflow logging."""
    out: dict[str, str] = {}
    _flatten("run", {"experiment": cfg.run.experiment, "seed": cfg.run.seed}, out)
    for k, v in cfg.run.tags.items():
        out[f"run.tags.{k}"] = str(v)
    _flatten(
        "data",
        {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "eval_split": cfg.data.eval_split,
        },
        out,
    )
    _flatten("predictor.name", cfg.predictor.name, out)
    _flatten("predictor.params", dict(cfg.predictor.params), out)
    return out
```

- [ ] **Step 4: Run again**

```bash
uv run pytest charges_experiments/tests/test_charge_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add charges_experiments/charge_experiments/config.py charges_experiments/tests/test_charge_config.py
git commit -m "feat(charges): add config.py (YAML load, --set overrides)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 6: `prepare_store.py` — download, streaming parse, cluster+split

**Files:**
- Create: `charges_experiments/charge_experiments/prepare_store.py`
- Create: `charges_experiments/tests/test_charge_prepare_store.py`

**Interfaces:**
- Consumes: `charge_experiments.data.mol_to_blob`,
  `charge_experiments._chalcedon.butina_cluster.butina_cluster`,
  `charge_experiments._chalcedon.greedy_cluster_split.greedy_cluster_split`.
- Produces: `download_dash_sdf(dest_dir, *, url=DOWNLOAD_URL) -> Path`,
  `parse_dash_molecules(sdf_path, out_path) -> None` (writes
  `molecules.parquet` with columns `chembl_id, conf_id, mol, net_charge`, no
  `split` column yet), `assign_splits(store_dir, *, train=0.8, val=0.1,
  test=0.1) -> str` (adds `split`, overwrites `molecules.parquet`, returns
  the summary text), `prepare_store(store_name, *, stores_root, train=0.8,
  val=0.1, test=0.1) -> None` (the idempotent orchestrator, mirroring
  `cosmo_experiments/sieve_experiments/prepare_store.py`'s own `prepare_store`).
- Consumed by: Task 8's `cli.py` (`prepare-store` subcommand), Task 8/10/12's
  `_optional` tests.

- [ ] **Step 1: Write the failing tests for the pure-logic pieces**

The SDF parsing/download pieces need a real (large) file to test
end-to-end — that's Task 6's Step 6, an `_optional` test gated on the real
SDF's presence. This step covers what's testable without it: stereo
assignment and the parquet-round-trip shape, using tiny hand-built SDF text
and a tiny synthetic fingerprint/cluster case.

```python
# charges_experiments/tests/test_charge_prepare_store.py
"""Fast-suite tests for prepare_store.py's pure-logic pieces -- no download,
no real 8.3GB SDF needed. The real end-to-end parse/cluster/split path is
covered by test_charge_prepare_store_optional.py, gated on that file's
presence."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

_TINY_SDF = """\

     RDKit          3D

  4  3  0  0  0  0  0  0  0  0999 V2000
   -0.6091    0.0044    0.4110 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5413    0.0039    0.4779 N   0  0  0  0  0  4  0  0  0  0  0  0
    1.7164    0.0116    0.5566 O   0  0  0  0  0  1  0  0  0  0  0  0
   -1.6488   -0.0199    0.3290 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
  2  3  1  0
  1  4  1  0
M  CHG  2   2   1   3  -1
M  END
>  <CHEMBL_ID>  (1)
CHEMBL185198

>  <CONF_ID>  (1)
conf_00

>  <MBIScharge>  (1)
-0.3034|0.3806|-0.3975|0.3204

$$$$
"""


def test_parse_dash_molecules_writes_one_row_per_record(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text(_TINY_SDF)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"
    assert df.loc[0, "conf_id"] == "conf_00"
    assert df.loc[0, "net_charge"] == pytest.approx(0.0)  # M CHG: +1 and -1 cancel

    from charge_experiments.data import blob_to_mol

    mol = blob_to_mol(df.loc[0, "mol"])
    assert mol.GetNumAtoms() == 4
    charges = [a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()]
    assert charges == pytest.approx([-0.3034, 0.3806, -0.3975, 0.3204])


def test_parse_dash_molecules_handles_multiple_records(tmp_path):
    from charge_experiments.prepare_store import parse_dash_molecules

    sdf_path = tmp_path / "tiny.sdf"
    two_records = _TINY_SDF + _TINY_SDF.replace("CHEMBL185198", "CHEMBL999999").replace(
        "conf_00", "conf_01"
    )
    sdf_path.write_text(two_records)
    out_path = tmp_path / "molecules.parquet"

    parse_dash_molecules(sdf_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 2
    assert set(df["chembl_id"]) == {"CHEMBL185198", "CHEMBL999999"}


def test_assign_splits_never_splits_a_chembl_id_across_splits(tmp_path):
    from charge_experiments.prepare_store import assign_splits, parse_dash_molecules

    # Three distinct chembl_ids, each with 2 conformers of the same tiny
    # molecule text (connectivity-identical, so Butina clustering behavior
    # doesn't matter here -- what's under test is that the split join
    # groups by chembl_id, not by row).
    records = []
    for i in range(3):
        for conf in ("conf_00", "conf_01"):
            records.append(
                _TINY_SDF.replace("CHEMBL185198", f"CHEMBL{i}").replace(
                    "conf_00", conf
                )
            )
    sdf_path = tmp_path / "tiny.sdf"
    sdf_path.write_text("".join(records))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    molecules_path = store_dir / "molecules.parquet"
    parse_dash_molecules(sdf_path, molecules_path)

    assign_splits(store_dir, train=1 / 3, val=1 / 3, test=1 / 3)

    import pandas as pd

    df = pd.read_parquet(molecules_path)
    assert "split" in df.columns
    per_id = df.groupby("chembl_id")["split"].nunique()
    assert (per_id == 1).all()
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_prepare_store.py -v
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write `prepare_store.py`**

```python
"""Download DASH's published training SDF, parse it (streaming, never
loading the whole 8.3GB file into memory), and cluster+split it.

Mirrors cosmo_experiments/sieve_experiments/prepare_store.py's own
download-verify-idempotent shape and its download/split separation
(``download_chaos_store``/``split_chaos_store``/``prepare_store`` there ->
``download_dash_sdf``/``parse_dash_molecules``/``assign_splits``/
``prepare_store`` here), adapted for a plain (non-zip) file download and a
streaming SDF parse instead of a cosmolayer SegmentStore load.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DOWNLOAD_URL = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "4e827dd2-65a0-4305-9118-480ef5fce0b5/content"
)
EXPECTED_BYTES = 8_278_301_584
EXPECTED_MD5 = "305f521c6b422546bdf09c1e87eb922d"
SDF_FILENAME = "dashMoleculesSDF_v2.sdf"

CHUNK_SIZE = 1 << 20  # 1 MiB
PARQUET_BATCH_SIZE = 50_000  # rows buffered before each parquet write

logger = logging.getLogger("charge_experiments")


def download_dash_sdf(dest_dir: Path, *, url: str = DOWNLOAD_URL) -> Path:
    """Stream ``url`` to ``dest_dir / SDF_FILENAME``, verifying size and md5
    against the published values. Idempotent: does nothing but log if a
    correctly-sized file is already present (a full md5 pass over 8.3GB on
    every call would be needlessly slow; size is a fast first check, and a
    truncated/corrupted re-download is caught by md5 the next time this
    function actually re-downloads)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / SDF_FILENAME

    if out_path.exists() and out_path.stat().st_size == EXPECTED_BYTES:
        logger.info("%s already present with the expected size; skipping download", out_path)
        return out_path

    md5 = hashlib.md5()
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)
            md5.update(chunk)

    actual_bytes = out_path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            f"downloaded {out_path.name}: {actual_bytes} bytes, expected "
            f"{EXPECTED_BYTES}; download incomplete or corrupted"
        )
    actual_md5 = md5.hexdigest()
    if actual_md5 != EXPECTED_MD5:
        out_path.unlink()
        raise ValueError(
            f"downloaded {out_path.name} md5 {actual_md5} != expected "
            f"{EXPECTED_MD5}; download corrupted"
        )
    logger.info("downloaded %s (md5 %s)", out_path, actual_md5)
    return out_path


def _assign_stereo_if_needed(mol: Any) -> None:
    """If ``mol`` has an unassigned stereocenter (not already fully
    specified by the molblock's own parity bits), perceive stereo from its
    own 3D coordinates. Mutates ``mol`` in place, matching
    ``Chem.AssignStereochemistry``'s own convention."""
    from rdkit import Chem

    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if any(tag == "?" for _, tag in centers):
        Chem.AssignStereochemistryFrom3D(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _parse_one_record(mol: Any) -> dict[str, Any] | None:
    """Extract one row's worth of data from an already-parsed rdkit ``Mol``
    (one ``ForwardSDMolSupplier`` record). Returns ``None`` (and logs a
    warning) for a record missing ``MBIScharge``/``CHEMBL_ID``/``CONF_ID``
    or whose atom count disagrees with its ``MBIScharge`` count, rather than
    raising -- a handful of malformed records should not abort an
    hours-long parse of an 8.3GB file."""
    from charge_experiments.data import mol_to_blob

    if mol is None:
        return None
    for required in ("CHEMBL_ID", "CONF_ID", "MBIScharge"):
        if not mol.HasProp(required):
            logger.warning("record missing %s property; skipping", required)
            return None

    charges = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
    if len(charges) != mol.GetNumAtoms():
        logger.warning(
            "MBIScharge has %d values but molecule has %d atoms; skipping (chembl_id=%s)",
            len(charges),
            mol.GetNumAtoms(),
            mol.GetProp("CHEMBL_ID"),
        )
        return None

    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("MBIScharge", charge)

    _assign_stereo_if_needed(mol)

    from rdkit import Chem

    return {
        "chembl_id": mol.GetProp("CHEMBL_ID"),
        "conf_id": mol.GetProp("CONF_ID"),
        "mol": mol_to_blob(mol),
        "net_charge": float(Chem.GetFormalCharge(mol)),
    }


def parse_dash_molecules(sdf_path: Path, out_path: Path) -> None:
    """Stream-parse ``sdf_path`` (never loading it whole into memory) into
    ``out_path``, a parquet file with columns ``chembl_id, conf_id, mol,
    net_charge`` (no ``split`` column yet -- see ``assign_splits``). Written
    in batches via a ``pyarrow.parquet.ParquetWriter`` so peak memory is
    bounded by ``PARQUET_BATCH_SIZE`` rows, not the whole (multi-million-row)
    dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem

    schema = pa.schema(
        [
            ("chembl_id", pa.string()),
            ("conf_id", pa.string()),
            ("mol", pa.binary()),
            ("net_charge", pa.float64()),
        ]
    )

    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    n_written = 0
    n_skipped = 0
    try:
        with open(sdf_path, "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, sanitize=True, removeHs=False)
            for mol in supplier:
                row = _parse_one_record(mol)
                if row is None:
                    n_skipped += 1
                    continue
                batch.append(row)
                if len(batch) >= PARQUET_BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, schema)
                    writer.write_table(table)
                    n_written += len(batch)
                    batch = []
            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema)
                writer.write_table(table)
                n_written += len(batch)
    finally:
        if writer is not None:
            writer.close()

    logger.info("parsed %d records (%d skipped) from %s", n_written, n_skipped, sdf_path)


def _achiral_fingerprints(mols: list[Any], *, radius: int = 2, n_bits: int = 2048):
    """Dense achiral Morgan fingerprint matrix, one row per mol -- the input
    shape ``_chalcedon.butina_cluster`` expects."""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    out = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=False
        )
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def assign_splits(
    store_dir: Path, *, train: float = 0.8, val: float = 0.1, test: float = 0.1
) -> str:
    """Compute (or refresh) the ``split`` column on ``store_dir /
    'molecules.parquet'`` and overwrite it in place; return the summary
    text. Clustering fingerprints come from each unique ``chembl_id``'s
    first-seen conformer only (any one conformer's connectivity suffices --
    clustering is graph-level, computed achiral so different stereoisomers
    of the same 2D graph land in the same cluster). Splits are then assigned
    per-cluster via the vendored ``greedy_cluster_split`` and joined back
    onto every row by ``chembl_id``, so a molecule's conformers/
    stereoisomers never span two splits."""
    assert abs(train + val + test - 1) < 1e-6, "the fractions must sum to 1"
    import pandas as pd

    from charge_experiments._chalcedon.butina_cluster import butina_cluster
    from charge_experiments._chalcedon.greedy_cluster_split import (
        greedy_cluster_split,
    )
    from charge_experiments.data import blob_to_mol

    molecules_path = store_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path)

    first_seen = df.drop_duplicates(subset="chembl_id", keep="first")
    unique_chembl_ids = first_seen["chembl_id"].to_numpy()
    first_mols = [blob_to_mol(b) for b in first_seen["mol"]]
    fingerprints = _achiral_fingerprints(first_mols)

    cluster_ids = butina_cluster(fingerprints, cutoff=0.65)
    split_by_index = greedy_cluster_split(
        cluster_ids, fractions={"train": train, "val": val, "test": test}
    )
    chembl_to_split: dict[str, str] = {}
    for split_name, indices in split_by_index.items():
        for i in indices:
            chembl_to_split[unique_chembl_ids[i]] = split_name

    df["split"] = df["chembl_id"].map(chembl_to_split)

    summary = (
        df.groupby("split")
        .agg(n_conformers=("chembl_id", "size"), n_chembl_ids=("chembl_id", "nunique"))
        .reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_conformers"] / len(df)
    summary_text = summary.to_string()

    df.to_parquet(molecules_path)
    return summary_text


def prepare_store(
    store_name: str,
    *,
    stores_root: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> None:
    """Ensure ``store_name`` is downloaded, parsed, and has a ``split``
    column. Idempotent at each stage, mirroring
    cosmo_experiments/sieve_experiments/prepare_store.py's own
    ``prepare_store``."""
    store_dir = stores_root / store_name
    store_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = download_dash_sdf(store_dir)

    molecules_path = store_dir / "molecules.parquet"
    if not molecules_path.exists():
        parse_dash_molecules(sdf_path, molecules_path)
    else:
        logger.info("%s already parsed; skipping", molecules_path)

    import pyarrow.parquet as pq

    already_split = "split" in pq.ParquetFile(molecules_path).schema.names
    if already_split:
        logger.info("%s already has a split column; nothing to do", molecules_path)
        return

    summary_text = assign_splits(store_dir, train=train, val=val, test=test)
    (store_dir / "split_summary.txt").write_text(summary_text + "\n")
    logger.info("wrote split for %s:\n%s", store_name, summary_text)
```

- [ ] **Step 4: Run the fast-suite prepare_store tests**

```bash
uv run pytest charges_experiments/tests/test_charge_prepare_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Write the download/full-pipeline `_optional` test**

```python
# charges_experiments/tests/test_charge_prepare_store_optional.py
"""End-to-end prepare_store test against the real, already-downloaded
dashMoleculesSDF_v2.sdf. Skipped entirely if that file is absent -- matches
cosmo_experiments' *_optional.py pattern for tests needing the real store."""

from __future__ import annotations

from pathlib import Path

import pytest

_REAL_SDF = Path.home() / "tmp" / "dash_molecules" / "dashMoleculesSDF_v2.sdf"

pytestmark = pytest.mark.skipif(
    not _REAL_SDF.exists(), reason="real dashMoleculesSDF_v2.sdf not present locally"
)


def test_parse_first_n_records_of_the_real_sdf(tmp_path):
    """Doesn't parse the whole 8.3GB file (would take too long for a test
    run) -- copies just the first two records' worth of bytes (up to the
    second '$$$$' terminator) into a small temp file and parses that,
    exercising the real parser against the real file's real property names
    and formatting."""
    from charge_experiments.prepare_store import parse_dash_molecules

    text = _REAL_SDF.read_text(errors="replace")
    end = text.index("$$$$", text.index("$$$$") + 1) + len("$$$$\n")
    sample_path = tmp_path / "sample.sdf"
    sample_path.write_text(text[:end])

    out_path = tmp_path / "molecules.parquet"
    parse_dash_molecules(sample_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 2
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"
```

- [ ] **Step 6: Run the optional test**

```bash
uv run pytest charges_experiments/tests/test_charge_prepare_store_optional.py -v
```

Expected: PASS (the real file is present at
`~/tmp/dash_molecules/dashMoleculesSDF_v2.sdf` on this machine).

- [ ] **Step 7: Commit**

```bash
git add charges_experiments/charge_experiments/prepare_store.py \
  charges_experiments/tests/test_charge_prepare_store.py \
  charges_experiments/tests/test_charge_prepare_store_optional.py
git commit -m "feat(charges): add prepare_store.py (download, streaming parse, cluster+split)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 7: `sieve.io.rdkit_adapter`'s `y_from_atom_prop` option (core library change)

**Files:**
- Modify: `src/sieve/io/rdkit_adapter.py`
- Modify: `tests/test_rdkit_adapter.py`

**Interfaces:**
- Produces: `from_rdkit(mols, y=None, *, config, node_order=None,
  y_from_atom_prop=None) -> NodeBatch` — backward-compatible; existing
  callers passing `y` explicitly are unaffected.
- Consumed by: Task 12 (`sieve_predictor.py`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rdkit_adapter.py`:

```python
def test_y_from_atom_prop_reads_scalar_per_atom():
    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit

    mols = []
    for smi, charges in [("CO", [-0.2, 0.2]), ("CCO", [-0.1, 0.05, 0.05])]:
        mol = Chem.MolFromSmiles(smi)
        for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
            atom.SetDoubleProp("q", charge)
        mols.append(mol)

    cfg = cfg_for(["CO", "CCO"])
    b = from_rdkit(mols, config=cfg, y_from_atom_prop="q")

    assert b.y is not None
    assert b.y.shape == (5, 1)
    expected = [-0.2, 0.2, -0.1, 0.05, 0.05]
    assert b.y[:, 0].tolist() == pytest.approx(expected)


def test_y_from_atom_prop_respects_node_order():
    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    charges = [-0.2, 0.2]
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("q", charge)

    cfg = cfg_for(["CO"])
    reversed_order = [np.array([1, 0])]
    b = from_rdkit([mol], config=cfg, node_order=reversed_order, y_from_atom_prop="q")

    assert b.y[:, 0].tolist() == pytest.approx([0.2, -0.2])


def test_y_and_y_from_atom_prop_are_mutually_exclusive():
    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit

    mol = Chem.MolFromSmiles("CO")
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("q", 0.0)
    cfg = cfg_for(["CO"])

    with pytest.raises(ValueError, match="y_from_atom_prop"):
        from_rdkit([mol], y=np.zeros((2, 1)), config=cfg, y_from_atom_prop="q")
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest tests/test_rdkit_adapter.py -k y_from_atom_prop -v
```

Expected: FAIL, `TypeError: from_rdkit() got an unexpected keyword argument 'y_from_atom_prop'`.

- [ ] **Step 3: Modify `from_rdkit`**

```python
def from_rdkit(
    mols,
    y=None,
    *,
    config: SieveConfig,
    node_order=None,
    y_from_atom_prop: str | None = None,
) -> NodeBatch:
    if y is not None and y_from_atom_prop is not None:
        raise ValueError("pass either y or y_from_atom_prop, not both")

    flat = [a for g in config.attribute_levels for a in g]
    n = sum(m.GetNumAtoms() for m in mols)
    node_attrs = np.zeros((n, len(flat)), np.int64)
    elements = np.zeros(n, np.int64)
    graph_id = np.zeros(n, np.int64)
    y_out = np.zeros((n, 1), np.float64) if y_from_atom_prop is not None else None
    src, dst, attr = [], [], []
    off = 0
    for gi, mol in enumerate(mols):
        order = (
            list(range(mol.GetNumAtoms()))
            if node_order is None
            else list(node_order[gi])
        )
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
            if y_out is not None:
                y_out[g, 0] = a.GetDoubleProp(y_from_atom_prop)
        for b in mol.GetBonds():
            u = off + int(inv[b.GetBeginAtomIdx()])
            v = off + int(inv[b.GetEndAtomIdx()])
            c = config.edge_codes.get(str(b.GetBondType()), 0)
            src += [u, v]
            dst += [v, u]
            attr += [c, c]
        off += mol.GetNumAtoms()
    return NodeBatch(
        node_attrs=node_attrs,
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attr=np.array(attr, np.int64),
        graph_id=graph_id,
        y=y if y is not None else y_out,
        elements=elements,
    )
```

- [ ] **Step 4: Run the tests again**

```bash
uv run pytest tests/test_rdkit_adapter.py -v
```

Expected: all PASS (including the pre-existing tests — this is a strictly
additive change).

- [ ] **Step 5: Commit**

```bash
git add src/sieve/io/rdkit_adapter.py tests/test_rdkit_adapter.py
git commit -m "feat(rdkit_adapter): add y_from_atom_prop option to from_rdkit

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 8: `runner.py` + `cli.py` + smoke test

**Files:**
- Create: `charges_experiments/charge_experiments/runner.py`
- Create: `charges_experiments/charge_experiments/cli.py`
- Create: `charges_experiments/tests/conftest.py`
- Create: `charges_experiments/tests/test_charge_smoke.py`

**Interfaces:**
- Consumes: `charge_experiments.config.ExperimentCfg/to_dict/to_flat_params`,
  `charge_experiments.data.MoleculeSet/molecule_sum`,
  `charge_experiments.metrics.regression_metrics/charge_conservation_metrics`,
  `charge_experiments.predictors.build` (registry — built in Task 9, but
  only `global_mean` needs to exist for this task's smoke test, added here
  since it has no optional dependency, matching
  `sieve_experiments.predictors.global_mean`'s role in the cosmo series).
- Produces: `execute(cfg, mset, masks, *, runs_root, allow_dirty=False,
  tracking=None, data_seconds=0.0) -> RunResult(run_dir, metrics, manifest)`,
  `run(cfg, *, runs_root, allow_dirty=False, tracking=None, limit=None) ->
  RunResult`, `DEFAULT_RUNS_ROOT`, `DEFAULT_TRACKING_URI`. `cli.py`'s
  `build_parser()`/`main()`.

This is a much smaller `execute`/`_execute_inner` than the cosmo series':
no plots, no profile/area rollup, no train/val-vs-test parity panels — just
fit, predict, score (regression_metrics + charge_conservation_metrics),
write manifest/metrics/predictions, optional MLflow log.

- [ ] **Step 1: Write `predictors/__init__.py`, `predictors/base.py`, and
  `predictors/global_mean.py`** (needed for this task's smoke test; the
  `Predictor` protocol itself is this task's dependency, not Task 9's — Task
  9 only adds the `dash` registry branch)

```python
# charges_experiments/charge_experiments/predictors/base.py
"""The predictor seam: one interface, one scalar-per-atom output.

Unlike cosmo_experiments' base.py, there is no AtomPredictor/
MoleculePredictor split and no profile/area/charge rollup machinery: every
predictor in this series predicts one scalar (MBIScharge) per atom
directly, and the molecule-level charge-conservation check
(metrics.charge_conservation_metrics) is computed by the caller
(runner.py), not by the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet


@dataclass(frozen=True)
class Prediction:
    """What every predictor returns: one predicted ``MBIScharge`` per atom,
    in ``test``'s own atom order (``test.atom_mol_id``-aligned)."""

    atom_charge: NDArray[np.float64]


class Predictor(Protocol):
    name: ClassVar[str]

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None: ...

    def predict(self, test: MoleculeSet) -> Prediction: ...
```

```python
# charges_experiments/charge_experiments/predictors/global_mean.py
"""The simplest possible baseline: predict the training set's own mean
MBIScharge for every atom. No optional dependency, so this is the one
predictor registered eagerly (see predictors/__init__.py)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet
from charge_experiments.predictors.base import Prediction


class GlobalMeanPredictor:
    name = "global_mean"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        if train.n_conformers == 0:
            raise ValueError("global_mean requires a non-empty train split")
        self._mean = float(np.mean(train.atom_charge))

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._mean is None:
            raise RuntimeError("fit must be called before predict")
        atom_charge: NDArray[np.float64] = np.full(test.n_atoms, self._mean)
        return Prediction(atom_charge=atom_charge)
```

```python
# charges_experiments/charge_experiments/predictors/__init__.py
"""Predictor registry. Lazy imports: a predictor module needing an optional
dependency (rdkit, the DASH-tree clone) registers itself via ``register``
only when its own module is imported. Only ``global_mean`` (no optional
deps) is registered eagerly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from charge_experiments.predictors.base import Predictor
from charge_experiments.predictors.global_mean import GlobalMeanPredictor

_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Predictor]] = {
    "global_mean": lambda params: GlobalMeanPredictor(**params),
}

REGISTRY: Mapping[str, Callable[[Mapping[str, Any]], Predictor]] = _BUILDERS


def register(name: str, builder: Callable[[Mapping[str, Any]], Predictor]) -> None:
    _BUILDERS[name] = builder


def build(name: str, params: Mapping[str, Any]) -> Predictor:
    # One branch per optional-dependency predictor module, added as built:
    # predictors/dash.py -> "dash" (Task 10), predictors/sieve_predictor.py
    # -> "sieve" (Task 12).
    if name == "dash" and name not in REGISTRY:
        import charge_experiments.predictors.dash  # noqa: F401
    if name == "sieve" and name not in REGISTRY:
        import charge_experiments.predictors.sieve_predictor  # noqa: F401
    if name not in REGISTRY:
        raise ValueError(f"unknown predictor {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](params)
```

- [ ] **Step 2: Write `runner.py`**

```python
"""Run one charges experiment: config + data in, a run directory (and
optionally an MLflow record) out.

``execute`` is the core, testable pipeline (an already-built ``MoleculeSet``
and split masks in, so the smoke test never touches the real store).
``run`` is the real entry point: loads the store via
``data.load_molecule_set`` then calls ``execute``. Mirrors
cosmo_experiments/sieve_experiments/runner.py's shape, much smaller: no
plots, no profile/area rollup, no train/val parity-plot bookkeeping.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from charge_experiments import metrics as metrics_mod
from charge_experiments.config import ExperimentCfg, to_dict, to_flat_params
from charge_experiments.data import REPO_ROOT, MoleculeSet, molecule_sum
from charge_experiments.predictors import build
from charge_experiments.predictors.base import Prediction

DEFAULT_TRACKING_URI = f"file:{REPO_ROOT / 'charges_experiments' / 'mlruns'}"
DEFAULT_RUNS_ROOT = REPO_ROOT / "charges_experiments" / "runs"

logger = logging.getLogger("charge_experiments")


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, float]
    manifest: dict[str, Any]


def _git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": dirty,
        "describe": _run(["git", "describe", "--always", "--dirty"]),
    }


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("sieve", "numpy", "scipy", "pyyaml", "rdkit", "pandas", "pyarrow", "mlflow"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _run_name(cfg: ExperimentCfg) -> str:
    d = cfg.data
    return f"{cfg.predictor.name}-{d.split_column}-s{cfg.run.seed}"


def _savez_run(path: Path, test: MoleculeSet, pred: Prediction, /) -> None:
    np.savez(
        path,
        chembl_id=np.array(test.chembl_id),
        conf_id=np.array(test.conf_id),
        num_atoms=test.num_atoms,
        net_charge=test.net_charge,
        atom_charge_true=test.atom_charge,
        atom_charge_pred=pred.atom_charge,
    )


def _score(test: MoleculeSet, pred: Prediction) -> dict[str, float]:
    out = metrics_mod.regression_metrics(test.atom_charge, pred.atom_charge)
    out["n_test_atoms"] = float(test.n_atoms)
    out["n_test_conformers"] = float(test.n_conformers)
    conservation = metrics_mod.charge_conservation_metrics(
        pred.atom_charge, test.atom_mol_id, test.net_charge, test.n_conformers
    )
    out.update({f"charge_conservation/{k}": v for k, v in conservation.items()})
    return out


def execute(
    cfg: ExperimentCfg,
    mset: MoleculeSet,
    masks: dict[str, NDArray[np.bool_]],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    data_seconds: float = 0.0,
) -> RunResult:
    """Run the pipeline against an already-loaded ``mset``/``masks``."""
    git_info = _git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
        )

    random.seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)

    train = mset.select(masks[cfg.data.train_split])
    val = mset.select(masks[cfg.data.val_split])
    test = mset.select(masks[cfg.data.eval_split])

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / cfg.run.experiment / f"{_run_name(cfg)}__{stamp}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(run_dir / "stdout.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    try:
        return _execute_inner(
            cfg,
            train,
            val,
            test,
            rng=rng,
            run_dir=run_dir,
            started=started,
            git_info=git_info,
            tracking=tracking,
            data_seconds=data_seconds,
        )
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def _score_extra_split(predictor: Any, mset: MoleculeSet, *, split: str) -> dict[str, float]:
    """Predict + score train/val the same way test is scored, mirroring
    cosmo_experiments' own train/val-alongside-test convention. Empty split
    -> no keys, not a crash."""
    if mset.n_conformers == 0:
        return {}
    pred = predictor.predict(mset)
    score = _score(mset, pred)
    return {f"{split}/{k}": v for k, v in score.items()}


def _execute_inner(
    cfg: ExperimentCfg,
    train: MoleculeSet,
    val: MoleculeSet,
    test: MoleculeSet,
    *,
    rng: np.random.Generator,
    run_dir: Path,
    started: datetime,
    git_info: dict[str, Any],
    tracking: str | None,
    data_seconds: float,
) -> RunResult:
    predictor = build(cfg.predictor.name, cfg.predictor.params)

    t0 = time.perf_counter()
    predictor.fit(train, val, rng=rng)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_metrics = _score_extra_split(predictor, train, split="train")
    train_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_metrics = _score_extra_split(predictor, val, split="val")
    val_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = predictor.predict(test)
    predict_s = time.perf_counter() - t0

    run_metrics = _score(test, pred)
    run_metrics.update(train_metrics)
    if train_metrics:
        run_metrics["time/train_predict_s"] = train_predict_s
    run_metrics.update(val_metrics)
    if val_metrics:
        run_metrics["time/val_predict_s"] = val_predict_s
    run_metrics["time/fit_s"] = fit_s
    run_metrics["time/predict_s"] = predict_s
    run_metrics["time/data_s"] = data_seconds

    manifest = {
        "schema_version": 1,
        "run_name": _run_name(cfg),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": {"fit": fit_s, "predict": predict_s, "data": data_seconds},
        "git": git_info,
        "seed": cfg.run.seed,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "n_train_conformers": train.n_conformers,
            "n_val_conformers": val.n_conformers,
            "n_test_conformers": test.n_conformers,
            "n_train_atoms": train.n_atoms,
            "n_test_atoms": test.n_atoms,
        },
        "config": to_dict(cfg),
    }
    match_stats = getattr(predictor, "match_stats", None)
    if match_stats:
        manifest["match_stats"] = match_stats

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(to_dict(cfg)))
    (run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, sort_keys=True))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _savez_run(run_dir / "predictions.npz", test, pred)

    if tracking is not None:
        _log_mlflow(cfg, run_metrics, run_dir, tracking)

    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


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
    with mlflow.start_run(run_name=_run_name(cfg)):
        tags = {
            "predictor": cfg.predictor.name,
            "split_column": cfg.data.split_column,
            "store": cfg.data.store,
            "seed": str(cfg.run.seed),
            "run_dir": str(run_dir),
            **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
        }
        mlflow.set_tags(tags)
        mlflow.log_params(to_flat_params(cfg))
        clean_metrics = {
            k: v for k, v in run_metrics.items() if isinstance(v, float) and not np.isnan(v)
        }
        mlflow.log_metrics({f"test/{k}": v for k, v in clean_metrics.items()})
        mlflow.log_artifacts(str(run_dir))


def load_molecule_set(
    store_name: str,
    *,
    split_column: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    stores_root: Path | None = None,
) -> tuple[MoleculeSet, dict[str, NDArray[np.bool_]]]:
    """Load ``molecules.parquet`` for ``store_name`` into a ``MoleculeSet``
    plus split masks."""
    from charge_experiments.data import DEFAULT_STORES_ROOT, blob_to_mol

    import pandas as pd

    root = stores_root if stores_root is not None else DEFAULT_STORES_ROOT
    store_dir = root / store_name
    df = pd.read_parquet(store_dir / "molecules.parquet")
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    mols = [blob_to_mol(b) for b in df["mol"]]
    mset = MoleculeSet(
        chembl_id=list(df["chembl_id"]),
        conf_id=list(df["conf_id"]),
        mols=mols,
        net_charge=df["net_charge"].to_numpy(dtype=np.float64),
        split=list(df[split_column]),
    )
    masks = {name: (df[split_column] == name).to_numpy() for name in splits}
    return mset, masks


def run(
    cfg: ExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    limit: int | None = None,
) -> RunResult:
    """Load the real store, then run the pipeline (see ``execute``)."""
    t0 = time.perf_counter()
    mset, masks = load_molecule_set(
        cfg.data.store,
        split_column=cfg.data.split_column,
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.eval_split),
        limit=limit,
    )
    data_seconds = time.perf_counter() - t0

    return execute(
        cfg,
        mset,
        masks,
        runs_root=runs_root,
        allow_dirty=allow_dirty,
        tracking=tracking,
        data_seconds=data_seconds,
    )
```

- [ ] **Step 3: Write `cli.py`**

```python
"""``python -m charge_experiments <command> ...``"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from charge_experiments.config import load_config
from charge_experiments.data import DEFAULT_STORES_ROOT
from charge_experiments.runner import DEFAULT_RUNS_ROOT, DEFAULT_TRACKING_URI, run

SUMMARY_COLUMNS = [
    "run_name",
    "predictor",
    "split_column",
    "seed",
    "n_test_atoms",
    "mae",
    "rmse",
    "r2",
    "charge_conservation/mae",
    "charge_conservation/rmse",
    "charge_conservation/r2",
    "train/mae",
    "train/r2",
    "val/mae",
    "val/r2",
    "time/fit_s",
    "time/predict_s",
    "time/data_s",
    "git_commit",
    "run_dir",
]


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.set)
    tracking = None if args.no_tracking else DEFAULT_TRACKING_URI
    result = run(
        cfg,
        runs_root=DEFAULT_RUNS_ROOT,
        allow_dirty=args.allow_dirty,
        tracking=tracking,
        limit=args.limit,
    )
    print(f"run written to {result.run_dir}")
    for key in sorted(result.metrics):
        print(f"  {key}: {result.metrics[key]}")
    return 0


def _cmd_prepare_store(args: argparse.Namespace) -> int:
    from charge_experiments.prepare_store import prepare_store

    prepare_store(args.store, stores_root=DEFAULT_STORES_ROOT)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    del args
    runs_root = DEFAULT_RUNS_ROOT
    rows = []
    for manifest_path in sorted(runs_root.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        metrics_path = run_dir / "metrics.json"
        run_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        row = {
            "run_name": manifest.get("run_name", ""),
            "predictor": manifest.get("config", {}).get("predictor", {}).get("name", ""),
            "split_column": manifest.get("data", {}).get("split_column", ""),
            "seed": manifest.get("seed", ""),
            "git_commit": manifest.get("git", {}).get("commit", ""),
            "run_dir": str(run_dir),
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                value = run_metrics.get(key, "")
                row[key] = (
                    "" if value == "" else f"{value:.6g}" if isinstance(value, float) else value
                )
        rows.append(row)

    rows.sort(key=lambda r: (r["split_column"], r["predictor"], r["seed"]))

    out_path = runs_root.parents[0] / "results" / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="charge_experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one experiment from a YAML config")
    p_run.add_argument("--config", required=True, type=Path)
    p_run.add_argument(
        "--set", action="append", default=[], metavar="key.path=value",
        help="override a config value; may be passed multiple times",
    )
    p_run.add_argument("--limit", type=int, default=None, help="use only the first N conformers")
    p_run.add_argument("--allow-dirty", action="store_true", help="run with an uncommitted git tree")
    p_run.add_argument("--no-tracking", action="store_true", help="skip MLflow logging for this run")
    p_run.set_defaults(func=_cmd_run)

    p_prepare = sub.add_parser("prepare-store", help="download, parse, and split the DASH molecules SDF")
    p_prepare.add_argument("store", nargs="?", default="dash-molecules")
    p_prepare.set_defaults(func=_cmd_prepare_store)

    p_summary = sub.add_parser("summarize", help="collect runs/**/metrics.json into a CSV")
    p_summary.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: `tests/conftest.py`**

```python
"""Empty on purpose -- present so pytest treats charges_experiments/tests as
a package root the same way cosmo_experiments/tests does, for
`from charge_experiments.tests.helpers import ...`-style imports."""
```

- [ ] **Step 5: Write the smoke test**

```python
# charges_experiments/tests/test_charge_smoke.py
"""End-to-end smoke test on a synthetic store -- no download, no network,
no mlflow required (tracking=None)."""

from __future__ import annotations

import json

import numpy as np
from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
from charge_experiments.runner import execute

from charge_experiments.tests.helpers import synthetic_molecule_set


def _tiny_cfg() -> ExperimentCfg:
    return ExperimentCfg(
        run=RunCfg(experiment="charge-smoke-test", seed=0, tags={"stage": "smoke"}),
        data=DataCfg(
            store="synthetic", split_column="split",
            train_split="train", val_split="val", eval_split="test",
        ),
        predictor=PredictorCfg(name="global_mean", params={}),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=n_mol, p=[0.6, 0.2, 0.2])
    return {name: labels == name for name in ("train", "val", "test")}


def test_smoke_pipeline_writes_every_artifact(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)

    run_dir = result.run_dir
    assert run_dir.is_dir()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "stdout.log").exists()


def test_smoke_metrics_include_r2_and_charge_conservation(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)

    assert np.isfinite(result.metrics["mae"])
    assert np.isfinite(result.metrics["rmse"])
    assert "r2" in result.metrics
    assert "charge_conservation/mae" in result.metrics
    assert "max_abs_residual" not in result.metrics


def test_smoke_rejects_dirty_tree_by_default(tmp_path, monkeypatch):
    import charge_experiments.runner as runner_mod
    import pytest

    monkeypatch.setattr(
        runner_mod, "_git_info",
        lambda repo_root: {"commit": "deadbeef", "branch": "main", "dirty": True, "describe": "deadbeef-dirty"},
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()

    with pytest.raises(RuntimeError, match="dirty"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None)


def test_smoke_handles_an_empty_test_split(tmp_path):
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    all_train = np.ones(10, dtype=bool)
    none = np.zeros(10, dtype=bool)
    masks = {"train": all_train, "val": none, "test": none}
    cfg = _tiny_cfg()

    result = execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)
    assert result.metrics["n_test_conformers"] == 0
    assert np.isnan(result.metrics["mae"])


def test_smoke_metrics_json_matches_returned_metrics(tmp_path):
    mset = synthetic_molecule_set(n_mol=15, seed=2)
    masks = _synthetic_masks(15, seed=3)
    cfg = _tiny_cfg()

    result = execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)
    on_disk = json.loads((result.run_dir / "metrics.json").read_text())
    assert on_disk.keys() == result.metrics.keys()
    for key in on_disk:
        if isinstance(on_disk[key], float) and np.isnan(on_disk[key]):
            assert np.isnan(result.metrics[key])
        else:
            assert on_disk[key] == result.metrics[key]
```

- [ ] **Step 6: Run the smoke suite**

```bash
uv run pytest charges_experiments/tests/test_charge_smoke.py -v
```

Expected: all PASS.

- [ ] **Step 7: Run the whole fast suite so far**

```bash
uv run pytest charges_experiments/tests -v
uv run ruff check charges_experiments/
```

Expected: all PASS, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add charges_experiments/charge_experiments/runner.py \
  charges_experiments/charge_experiments/cli.py \
  charges_experiments/charge_experiments/predictors \
  charges_experiments/tests/conftest.py \
  charges_experiments/tests/test_charge_smoke.py
git commit -m "feat(charges): add runner.py, cli.py, global_mean predictor, smoke test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

Phase 1 is now complete and independently verifiable: `python -m
charge_experiments run --config <cfg> --no-tracking` works end-to-end
against a real (once prepared) store using `global_mean`.

---

## Phase 2 — DASH-tree charge predictor

### Task 9: Clone the pinned DASH-tree checkout for this series

**Files:**
- No tracked files (this task's output, `charges_experiments/external/`, is
  git-ignored — add the ignore entry here since Task 2 didn't need it yet).
- Modify: `.gitignore`

**Interfaces:**
- Produces: `charges_experiments/external/DASH-tree/` on disk, a plain git
  clone at the commit `charges_experiments/pins.toml`'s `[dash_tree]`
  section pins (`6cf1b2351c4674e602153dd493c06d9c020fc9ce`), matching
  `cosmo_experiments/external/DASH-tree`'s own role — cloned independently
  per series (never shared), per the "no shared harness code" constraint.

- [ ] **Step 1: Add the ignore entry**

In `.gitignore`, extend the block added in Task 2:

```
charges_experiments/external/
```

- [ ] **Step 2: Clone the pinned commit**

```bash
mkdir -p charges_experiments/external
git clone https://github.com/rinikerlab/DASH-tree.git charges_experiments/external/DASH-tree
git -C charges_experiments/external/DASH-tree checkout 6cf1b2351c4674e602153dd493c06d9c020fc9ce
```

- [ ] **Step 3: Verify `uv sync` doesn't try to absorb it as a workspace member**

`pyproject.toml`'s existing `[tool.uv.workspace] exclude =
["cosmo_experiments/external/*"]` does not cover this new path — add it:

```toml
[tool.uv.workspace]
exclude = ["cosmo_experiments/external/*", "charges_experiments/external/*"]
```

```bash
uv sync --extra dev --extra chem
```

Expected: completes without trying to build/install `serenityff` from
`charges_experiments/external/DASH-tree`.

- [ ] **Step 4: Commit the ignore/exclude changes** (the clone itself is
  git-ignored, not committed)

```bash
git add .gitignore pyproject.toml
git commit -m "chore(charges): gitignore/exclude charges_experiments/external

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 10: `predictors/dash.py` — scalar DASH-tree charge predictor

**Files:**
- Create: `charges_experiments/charge_experiments/predictors/dash.py`
- Create: `charges_experiments/tests/test_charge_predictor_dash.py`

**Interfaces:**
- Consumes: `charge_experiments.data.MoleculeSet`,
  `charge_experiments.predictors.register`,
  `charge_experiments.predictors.base.Prediction`.
- Produces: `populate_tree_with_charge_property(tree, paths, atom_charge) ->
  LiteralTreeChargeProperties(charge_column, fallback_charge)`,
  `predict_via_data_storage_walk(tree, paths, props) ->
  NDArray[np.float64]` (one predicted charge per atom, in `paths`' order),
  `DASHChargePredictor` implementing the `Predictor` protocol, registered as
  `"dash"`.
- Consumed by: Task 11's `_optional` test, `predictors/__init__.py`'s
  `"dash"` branch (already wired in Task 8).

This mirrors `cosmo_experiments/sieve_experiments/predictors/dash.py`'s
two-layer split (pure numpy tree-walk logic, separately testable from real
`DASHTree`/rdkit machinery) but with one scalar per node instead of a
51-bin profile array, and no atom-map-order/SMILES bookkeeping — this
series' `MoleculeSet.mols` are already-parsed `Mol` objects in their own
canonical atom order.

- [ ] **Step 1: Write the failing tests for the pure-logic layer**

```python
# charges_experiments/tests/test_charge_predictor_dash.py
"""Pure-numpy/pandas tests for dash.py's tree-populate/predict-walk logic --
a fake tree-like object stands in for a real DASHTree, so these need no
rdkit and no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _FakeTree:
    """A minimal stand-in exposing only ``data_storage`` (a dict of
    branch_idx -> pandas DataFrame), the one attribute
    populate_tree_with_charge_property/predict_via_data_storage_walk touch."""

    def __init__(self, branch_sizes: dict[int, int]):
        self.data_storage = {
            branch: pd.DataFrame(index=range(size)) for branch, size in branch_sizes.items()
        }


def test_populate_tree_with_charge_property_writes_node_means():
    from charge_experiments.predictors.dash import populate_tree_with_charge_property

    tree = _FakeTree({0: 3})
    # Two atoms both matched at path [(0, 1)] (root only); charges 0.2 and 0.4.
    paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.4])

    props = populate_tree_with_charge_property(tree, paths, atom_charge)

    df = tree.data_storage[0]
    assert df.loc[1, props.charge_column] == pytest.approx(0.3)
    assert pd.isna(df.loc[0, props.charge_column])
    assert pd.isna(df.loc[2, props.charge_column])
    assert props.fallback_charge == pytest.approx(0.3)


def test_predict_via_data_storage_walk_prefers_deepest_populated_node():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    train_paths = [[(0, 1)], [(0, 1), (0, 2)]]  # atom0: shallow only; atom1: shallow+deep
    atom_charge = np.array([0.1, 0.5])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    # Predict for an atom matched at both node 1 (populated) and node 2
    # (also populated, deepest) -> should use node 2's own mean (0.5), not
    # node 1's blended mean.
    test_paths = [[(0, 1), (0, 2)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.5)


def test_predict_via_data_storage_walk_backs_off_to_shallower_node():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 4})
    train_paths = [[(0, 1)]]
    atom_charge = np.array([0.7])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    # Deepest node (3) was never populated at train time -> back off to node 1.
    test_paths = [[(0, 1), (0, 3)]]
    predicted = predict_via_data_storage_walk(tree, test_paths, props)
    assert predicted[0] == pytest.approx(0.7)


def test_predict_via_data_storage_walk_falls_back_to_global_mean_for_unmatched_atom():
    from charge_experiments.predictors.dash import (
        populate_tree_with_charge_property,
        predict_via_data_storage_walk,
    )

    tree = _FakeTree({0: 2})
    train_paths = [[(0, 1)], [(0, 1)]]
    atom_charge = np.array([0.2, 0.6])
    props = populate_tree_with_charge_property(tree, train_paths, atom_charge)

    predicted = predict_via_data_storage_walk(tree, [[]], props)
    assert predicted[0] == pytest.approx(0.4)  # global mean, atom has empty path
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_dash.py -v
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write `predictors/dash.py`**

```python
"""DASH-tree charge predictor: DASH-tree's published topology
(``DASHTree.match_new_atom``, unmodified) with a back-off step reproducing
``DASHTree.get_property_noNAN``'s own missing-value fallback (deepest ->
shallowest, first populated node wins, else the global mean) -- ported from
cosmo_experiments/sieve_experiments/predictors/dash.py, adapted to a
**scalar** target (this series' own ``MBIScharge``, not a 51-bin profile)
and to this series' Mol-blob store (no atom-map-order/SMILES bookkeeping:
``MoleculeSet.mols`` are already-parsed ``Mol`` objects in their own atom
order, so tree-matching iterates them directly).

Two layers, deliberately split so the algorithm is testable without either
optional dependency (see charges_experiments/tests/test_charge_predictor_dash.py
for the pure-logic layer; the real-tree/real-rdkit layer is
_optional-tested only, in test_charge_predictor_dash_optional.py):

- ``populate_tree_with_charge_property``/``predict_via_data_storage_walk``
  -- pure numpy + pandas over pre-computed tree paths and an already-loaded
  ``DASHTree``'s own storage.
- ``DASHChargePredictor`` -- wires those onto real atoms: rdkit for
  iterating each conformer's own atoms and ``DASHTree.match_new_atom`` for
  the tree path.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction

logger = logging.getLogger("charge_experiments")

PathKey = tuple[int, int]
NodePath = list[PathKey]

# pins.toml's [dash_tree]: a plain git clone, cloned independently of
# cosmo_experiments' own copy (see Task 9) -- see that pins.toml entry for
# why (no shared harness code between series).
_DASH_TREE_ROOT = REPO_ROOT / "charges_experiments" / "external" / "DASH-tree"
if _DASH_TREE_ROOT.exists() and str(_DASH_TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASH_TREE_ROOT))


@dataclass(frozen=True)
class LiteralTreeChargeProperties:
    """What ``populate_tree_with_charge_property`` writes onto a
    ``DASHTree``'s own ``data_storage``, and what
    ``predict_via_data_storage_walk`` needs to read it back."""

    charge_column: str
    fallback_charge: float


def populate_tree_with_charge_property(
    tree: Any, paths: list[NodePath], atom_charge: NDArray[np.floating]
) -> LiteralTreeChargeProperties:
    """Populate an already-loaded ``DASHTree``'s own storage with our own
    per-node mean ``MBIScharge`` over every node on every atom's path. A
    node with zero matching atoms gets no entry and stays ``NaN`` --
    exactly DASH's own ``get_property_noNAN`` missing-value semantics."""
    n = len(paths)
    if len(atom_charge) != n:
        raise ValueError("paths and atom_charge must have the same length")
    atom_charge = np.asarray(atom_charge, dtype=np.float64)
    charge_column = "dash_charge_mean"

    charge_sum: dict[PathKey, float] = {}
    count: dict[PathKey, int] = {}
    for path, charge in zip(paths, atom_charge, strict=True):
        for key in path:
            charge_sum[key] = charge_sum.get(key, 0.0) + float(charge)
            count[key] = count.get(key, 0) + 1

    by_branch: dict[int, list[int]] = {}
    for branch_idx, node_id in count:
        by_branch.setdefault(branch_idx, []).append(node_id)

    for branch_idx, node_ids in by_branch.items():
        df = tree.data_storage[branch_idx]
        n_rows = len(df)
        node_ids_arr = np.array(node_ids, dtype=np.int64)
        means = np.array(
            [charge_sum[(branch_idx, nid)] / count[(branch_idx, nid)] for nid in node_ids]
        )
        values = np.full(n_rows, np.nan)
        values[node_ids_arr] = means
        df[charge_column] = values

    return LiteralTreeChargeProperties(
        charge_column=charge_column, fallback_charge=float(atom_charge.mean())
    )


def predict_via_data_storage_walk(
    tree: Any, paths: list[NodePath], props: LiteralTreeChargeProperties
) -> NDArray[np.float64]:
    """Predict by walking each atom's matched path deepest -> shallowest
    directly against ``tree.data_storage`` and using the first node whose
    row is populated -- the same fallback ``DASHTree.get_property_noNAN``
    itself implements."""
    n = len(paths)
    predicted = np.empty(n, dtype=np.float64)

    arrays: dict[int, NDArray[np.float64]] = {}
    for branch_idx in {path[0][0] for path in paths if path}:
        branch_df = tree.data_storage[branch_idx]
        if props.charge_column in branch_df.columns:
            arrays[branch_idx] = branch_df[props.charge_column].to_numpy(dtype=np.float64)

    for i, path in enumerate(paths):
        value = None
        arr = arrays.get(path[0][0]) if path else None
        if arr is not None:
            for _, node_id in reversed(path):
                candidate = arr[node_id]
                if not np.isnan(candidate):
                    value = candidate
                    break
        predicted[i] = props.fallback_charge if value is None else value

    return predicted


def _default_neighbor_dict_factory(mol: Any, af: Any) -> Any:
    from serenityff.charge.tree.dash_tools import init_neighbor_dict

    return init_neighbor_dict(mol, af=af)


def _atom_paths(
    mset: MoleculeSet,
    tree: Any,
    *,
    max_depth: int,
    attention_threshold: float,
    neighbor_dict_factory: Any = _default_neighbor_dict_factory,
) -> tuple[list[NodePath], dict[str, int]]:
    """``DASHTree.match_new_atom`` for every atom in ``mset``, in each
    conformer's own atom order (no atom-map-order decoding needed here --
    ``mset.mols`` are already-parsed ``Mol`` objects in their canonical
    order). Two failure modes are tolerated and counted, mirroring
    cosmo_experiments' own ``_atom_paths``: the whole molecule, when
    ``init_neighbor_dict`` raises (one out-of-vocabulary atom feature tuple
    takes the whole molecule down); a single atom, when ``match_new_atom``
    itself raises."""
    paths: list[NodePath] = []
    n_unmatched_atoms = 0
    n_unmatched_molecules = 0

    for mol in mset.mols:
        n_atoms = mol.GetNumAtoms()
        try:
            neighbor_dict = neighbor_dict_factory(mol, tree.atom_feature_type)
        except Exception:
            paths.extend([] for _ in range(n_atoms))
            n_unmatched_molecules += 1
            n_unmatched_atoms += n_atoms
            continue

        for j in range(n_atoms):
            try:
                raw = tree.match_new_atom(
                    j, mol, max_depth=max_depth,
                    attention_threshold=attention_threshold, neighbor_dict=neighbor_dict,
                )
                path = [(raw[0], node_id) for node_id in raw[1:]]
            except Exception:
                path = []
                n_unmatched_atoms += 1
            paths.append(path)

    stats = {
        "n_atoms": len(paths),
        "n_conformers": mset.n_conformers,
        "n_unmatched_atoms": n_unmatched_atoms,
        "n_unmatched_molecules": n_unmatched_molecules,
    }
    return paths, stats


class DASHChargePredictor:
    """DASH-tree charge baseline: published topology + our own per-node
    MBIScharge mean and missing-value back-off. See module docstring.

    ``preload`` defaults to True (see cosmo_experiments/pins.toml's GOTCHA 1
    -- on-demand loading has an ordering bug that raises on every H atom at
    the pinned commit).
    """

    name = "dash"

    def __init__(
        self,
        *,
        max_depth: int = 16,
        attention_threshold: float = 5.2,
        tree_folder_path: str | None = None,
        preload: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.attention_threshold = attention_threshold
        self.tree_folder_path = tree_folder_path
        self.preload = preload
        self.match_stats: dict[str, dict[str, int]] = {}
        self._tree: Any = None
        self._props: LiteralTreeChargeProperties | None = None

    def _load_tree(self) -> Any:
        if self._tree is None:
            from serenityff.charge.tree.dash_tree import DASHTree

            kwargs: dict[str, Any] = {"preload": self.preload, "verbose": False}
            if self.tree_folder_path is not None:
                kwargs["tree_folder_path"] = self.tree_folder_path
            self._tree = DASHTree(**kwargs)
        return self._tree

    def _paths_for(self, mset: MoleculeSet, *, split: str) -> list[NodePath]:
        tree = self._load_tree()
        paths, stats = _atom_paths(
            mset, tree, max_depth=self.max_depth, attention_threshold=self.attention_threshold
        )
        self.match_stats[split] = stats
        if stats["n_unmatched_atoms"]:
            logger.warning(
                "DASH could not match %d/%d %s atoms (%d/%d conformers rejected outright)",
                stats["n_unmatched_atoms"], stats["n_atoms"], split,
                stats["n_unmatched_molecules"], stats["n_conformers"],
            )
        return paths

    def fit(self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator) -> None:
        del val, rng
        tree = self._load_tree()
        paths = self._paths_for(train, split="train")
        self._props = populate_tree_with_charge_property(tree, paths, train.atom_charge)

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._props is None:
            raise RuntimeError("fit must be called before predict")
        tree = self._load_tree()
        paths = self._paths_for(test, split="test")
        atom_charge = predict_via_data_storage_walk(tree, paths, self._props)
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> DASHChargePredictor:
    return DASHChargePredictor(**params)


register("dash", _build)
```

- [ ] **Step 4: Run the pure-logic tests**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_dash.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add charges_experiments/charge_experiments/predictors/dash.py charges_experiments/tests/test_charge_predictor_dash.py
git commit -m "feat(charges): add DASHChargePredictor (scalar charge, pure-logic layer)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 11: DASH optional end-to-end test

**Files:**
- Create: `charges_experiments/tests/test_charge_predictor_dash_optional.py`

**Interfaces:**
- Consumes: `charge_experiments.predictors.dash.DASHChargePredictor`,
  `charge_experiments.tests.helpers.synthetic_molecule_set`.

- [ ] **Step 1: Write the optional test**

```python
# charges_experiments/tests/test_charge_predictor_dash_optional.py
"""End-to-end DASHChargePredictor test against the real pinned DASH-tree
clone. Skipped if that clone (charges_experiments/external/DASH-tree,
Task 9) is absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists(), reason="charges_experiments/external/DASH-tree not cloned"
)


def test_dash_charge_predictor_fits_and_predicts_on_synthetic_molecules():
    from charge_experiments.predictors.dash import DASHChargePredictor
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    rng = np.random.default_rng(0)
    predictor = DASHChargePredictor()
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))
    assert "train" in predictor.match_stats
    assert predictor.match_stats["train"]["n_conformers"] == mset.n_conformers
```

- [ ] **Step 2: Run it**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_dash_optional.py -v
```

Expected: PASS once Task 9's clone exists (SKIPPED otherwise, never a
failure).

- [ ] **Step 3: Commit**

```bash
git add charges_experiments/tests/test_charge_predictor_dash_optional.py
git commit -m "test(charges): add DASHChargePredictor optional end-to-end test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

Phase 2 is now complete: `predictor.name: dash` runs end-to-end through
`charge_experiments run`.

---

## Phase 3 — Sieve charge predictor

### Task 12: `predictors/sieve_predictor.py`

**Files:**
- Create: `charges_experiments/charge_experiments/predictors/sieve_predictor.py`
- Create: `charges_experiments/tests/test_charge_predictor_sieve.py`

**Interfaces:**
- Consumes: `sieve.config.SieveConfig`, `sieve.io.rdkit_adapter.build_codes`,
  `sieve.io.rdkit_adapter.from_rdkit(..., y_from_atom_prop=...)` (Task 7),
  `charge_experiments.predictors.register`,
  `charge_experiments.predictors.base.Prediction`.
- Produces: `SievePredictor` implementing the `Predictor` protocol,
  registered as `"sieve"`.

- [ ] **Step 1: Write the failing tests for the pure-logic layer**

```python
# charges_experiments/tests/test_charge_predictor_sieve.py
"""Fast-suite tests for sieve_predictor.py's config-building and
batch-building helpers -- real rdkit, real sieve.fit/predict, but no store,
no DASH-tree clone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def test_build_config_learns_codes_from_training_mols():
    from charge_experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _build_config,
    )
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=3,
        minimum_support=1,
        shrinkage_strength=None,
    )
    assert config.target_dim == 1
    assert "element" in config.attribute_codes


def test_batch_for_reads_mbis_charge_directly_off_the_mols():
    from charge_experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _batch_for,
        _build_config,
    )
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=4, seed=1)
    config = _build_config(
        mset.mols, attributes=DEFAULT_ATTRIBUTES, target_dim=1,
        max_wl_depth=3, minimum_support=1, shrinkage_strength=None,
    )
    batch = _batch_for(mset.mols, config, with_target=True)
    assert batch.n_nodes == mset.n_atoms
    np.testing.assert_allclose(batch.y[:, 0], mset.atom_charge)


def test_sieve_charge_predictor_fits_and_predicts_end_to_end():
    from charge_experiments.predictors.sieve_predictor import SievePredictor
    from charge_experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=8, seed=2)
    rng = np.random.default_rng(0)
    predictor = SievePredictor(max_wl_depth=2, minimum_support=1)
    predictor.fit(mset, mset, rng=rng)
    pred = predictor.predict(mset)

    assert pred.atom_charge.shape == (mset.n_atoms,)
    assert np.all(np.isfinite(pred.atom_charge))
```

- [ ] **Step 2: Run to see it fail**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_sieve.py -v
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write `predictors/sieve_predictor.py`**

```python
"""A basic Sieve charge predictor: this project's own hierarchical
regressogram (``sieve.fit``/``sieve.predict``) wired onto per-atom
``MBIScharge`` prediction directly via
``sieve.io.rdkit_adapter.from_rdkit``'s ``y_from_atom_prop`` option -- no
separate, position-aligned ``y`` array, since the target already rides on
each conformer's own ``Mol`` (see the design spec's "Sieve's target
ingestion" decision). No SMILES, no atom-map-order recovery: ``mset.mols``
are already the canonical, deserialized ``Mol`` objects, so
``node_order=None`` (natural atom order) is exactly right.

Attribute set mirrors DASH's own atom feature tuple (see
predictors/dash.py's docstring and cosmo_experiments' own precedent),
``max_wl_depth``/``minimum_support`` are starting values, deliberately not
tuned -- see the design spec's "Out of scope" list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charge_experiments.data import MoleculeSet
from charge_experiments.predictors import register
from charge_experiments.predictors.base import Prediction

DEFAULT_ATTRIBUTES = ("element", "degree", "formal_charge", "aromatic", "num_h")


def _build_config(
    train_mols: list[Any],
    *,
    attributes: tuple[str, ...],
    target_dim: int,
    max_wl_depth: int,
    minimum_support: int,
    shrinkage_strength: float | None,
) -> Any:
    """Learn ``attribute_codes``/``edge_codes`` from the training corpus and
    freeze them into a ``SieveConfig``."""
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes

    codes, edge_codes = build_codes(train_mols, attributes)
    return SieveConfig(
        target_dim=target_dim,
        attribute_levels=(attributes,),
        attribute_codes=codes,
        edge_codes=edge_codes,
        max_wl_depth=max_wl_depth,
        minimum_support=minimum_support,
        shrinkage_strength=shrinkage_strength,
    )


def _batch_for(mols: list[Any], config: Any, *, with_target: bool) -> Any:
    """Build a ``NodeBatch`` for ``mols`` under an already-fitted
    ``config``. ``node_order`` is left ``None``: each ``Mol``'s own atom
    order is already this series' canonical order."""
    from sieve.io.rdkit_adapter import from_rdkit

    return from_rdkit(
        mols, config=config, y_from_atom_prop="MBIScharge" if with_target else None
    )


class SievePredictor:
    """A basic Sieve charge baseline: one attribute level, a handful of
    Weisfeiler-Lehman refinement rounds, no shrinkage -- the first,
    deliberately unengineered Sieve baseline for this series."""

    name = "sieve"

    def __init__(
        self,
        *,
        attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
        max_wl_depth: int = 3,
        minimum_support: int = 1,
        shrinkage_strength: float | None = None,
    ) -> None:
        self.attributes = tuple(attributes)
        self.max_wl_depth = max_wl_depth
        self.minimum_support = minimum_support
        self.shrinkage_strength = shrinkage_strength
        self._config: Any = None
        self._model: Any = None

    def fit(self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator) -> None:
        del val, rng
        import sieve

        self._config = _build_config(
            train.mols, attributes=self.attributes, target_dim=1,
            max_wl_depth=self.max_wl_depth, minimum_support=self.minimum_support,
            shrinkage_strength=self.shrinkage_strength,
        )
        batch = _batch_for(train.mols, self._config, with_target=True)
        self._model = sieve.fit(batch, self._config)

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._model is None or self._config is None:
            raise RuntimeError("fit must be called before predict")
        import sieve

        batch = _batch_for(test.mols, self._config, with_target=False)
        atom_charge_2d = sieve.predict(self._model, batch)
        atom_charge: NDArray[np.float64] = np.asarray(atom_charge_2d, dtype=np.float64)[:, 0]
        return Prediction(atom_charge=atom_charge)


def _build(params: Mapping[str, Any]) -> SievePredictor:
    return SievePredictor(**params)


register("sieve", _build)
```

- [ ] **Step 4: Run the tests again**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_sieve.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add charges_experiments/charge_experiments/predictors/sieve_predictor.py charges_experiments/tests/test_charge_predictor_sieve.py
git commit -m "feat(charges): add SievePredictor (y_from_atom_prop-based charge target)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

---

### Task 13: Sieve optional end-to-end test + full-suite verification

**Files:**
- Create: `charges_experiments/tests/test_charge_predictor_sieve_optional.py`

**Interfaces:**
- Consumes: `charge_experiments.predictors.sieve_predictor.SievePredictor`,
  `charge_experiments.runner.execute`, `charge_experiments.config.*`.

- [ ] **Step 1: Write the optional test**

Unlike DASH (gated on an external clone), Sieve's charge predictor has no
extra optional dependency beyond rdkit — so this "optional" test is gated
on the real, prepared store instead (matching cosmo_experiments'
`test_experiment_predictor_sieve_optional.py`'s own reason for existing:
real-store scale, not a missing dependency).

```python
# charges_experiments/tests/test_charge_predictor_sieve_optional.py
"""End-to-end SievePredictor test through the real run() pipeline, gated on
the real, already-split dash-molecules store (Task 6's prepare_store, run
manually -- this test does not download/parse the 8.3GB SDF itself)."""

from __future__ import annotations

from pathlib import Path

import pytest

from charge_experiments.data import DEFAULT_STORES_ROOT

_STORE_DIR = DEFAULT_STORES_ROOT / "dash-molecules"

pytestmark = pytest.mark.skipif(
    not (_STORE_DIR / "molecules.parquet").exists(),
    reason="real dash-molecules store not prepared locally",
)


def test_sieve_charge_predictor_runs_end_to_end_via_run(tmp_path):
    from charge_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from charge_experiments.runner import run

    cfg = ExperimentCfg(
        run=RunCfg(experiment="sieve-charge-optional", seed=0),
        data=DataCfg(store="dash-molecules", split_column="split"),
        predictor=PredictorCfg(name="sieve", params={"max_wl_depth": 2}),
    )
    result = run(cfg, runs_root=tmp_path, allow_dirty=True, tracking=None, limit=200)
    assert result.metrics["n_test_conformers"] >= 0
```

- [ ] **Step 2: Run it**

```bash
uv run pytest charges_experiments/tests/test_charge_predictor_sieve_optional.py -v
```

Expected: SKIPPED (the real store isn't prepared as part of this plan —
`prepare_store` is a long-running, separate operational step) unless
someone has already run `prepare-store` locally.

- [ ] **Step 3: Run the entire charges_experiments fast suite one more time**

```bash
uv run pytest charges_experiments/tests -v
uv run pytest tests/test_rdkit_adapter.py -v
uv run ruff check charges_experiments/ src/sieve/io/rdkit_adapter.py
uv run ty check charges_experiments/ 2>&1 | tail -50
```

Expected: every non-`_optional`/skipped test PASSES; ruff clean; `ty`
clean or only pre-existing/expected unresolved-import notes for
`serenityff`/`pandas`/`pyarrow` (add a `[tool.ty.overrides]` entry mirroring
`pyproject.toml`'s existing `allowed-unresolved-imports` list, scoped to
`charges_experiments`, only if `ty` actually flags these — verify first
rather than pre-emptively adding an override nothing needs).

- [ ] **Step 4: Commit**

```bash
git add charges_experiments/tests/test_charge_predictor_sieve_optional.py
git commit -m "test(charges): add SievePredictor optional end-to-end test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011QxBZCsbaugJns9fYNVTdu"
```

Phase 3 is now complete: both predictors (`dash`, `sieve`) run end-to-end
through `charge_experiments run` against a real, prepared store, and the
whole series is registered in CI (`pyproject.toml`'s `testpaths`/`ruff
include`) without importing anything from `cosmo_experiments`/
`sieve_experiments`.

---

## Self-Review

**Spec coverage:**
- Code sharing / independence: Task 2 (registration alongside, not shared
  with, cosmo_experiments), verified by zero `sieve_experiments`/
  `cosmo_experiments` imports anywhere in `charge_experiments`.
- Charge target column (`MBIScharge`): Task 6 (`_parse_one_record`), Task 3
  (`MoleculeSet.atom_charge`).
- Source data / download+verify: Task 6 (`download_dash_sdf`).
- Per-conformer rows, no dedup/averaging: Task 6 (`parse_dash_molecules`
  emits one row per SDF record).
- No SMILES, Mol-blob row payload: Task 3 (`mol_to_blob`/`blob_to_mol`),
  Task 6 (`_parse_one_record` writes `mol_to_blob(mol)` directly).
- Stereo from 3D when not already specified: Task 6
  (`_assign_stereo_if_needed`).
- Sieve's `y_from_atom_prop` core-library change: Task 7.
- Splitting via vendored `_chalcedon`, achiral fingerprints, grouped by
  `chembl_id`: Task 1 (vendor), Task 6 (`assign_splits`).
- DASH-tree and Sieve predictors, no GNN baseline: Task 10, Task 12; no
  `chemprop`-shaped module anywhere in this plan.
- Metrics: MAE/RMSE/R², no max_abs_residual, charge-conservation
  diagnostic: Task 4.
- Testing layering (fast unit tests, core-library test, `_optional` gated
  suites, runner smoke test): Tasks 3/4/5/6 (fast), Task 7 (core), Task 6/11/13
  (`_optional`), Task 8 (smoke).
- Directory layout matches the spec's tree exactly (Task 1/2/3/4/5/6/8/10/12
  file paths).
- Out-of-scope items (Chemprop, second biased split, hyperparameter tuning)
  are absent from every task — confirmed by scanning the task list above.

**Placeholder scan:** no TBD/TODO, no "add appropriate error handling", no
"similar to Task N" instead of literal code; every step with code shows the
actual code. The one deliberately-deferred item (a `[tool.ty.overrides]`
entry in Task 13, Step 3) is phrased as "add only if `ty` actually flags
this" with a concrete check command, not an unresolved placeholder.

**Type consistency:** `Prediction.atom_charge` (Task 8's `base.py`) is the
one field every predictor's `predict()` returns (Task 8 `global_mean`, Task
10 `dash`, Task 12 `sieve`) — checked consistent across all three. `Predictor`
protocol's `fit(self, train, val, *, rng)`/`predict(self, test)` signature
matches every implementation. `MoleculeSet` field names (`chembl_id`,
`conf_id`, `mols`, `net_charge`, `split`, `atom_mol_id`, `atom_charge`,
`num_atoms`, `n_conformers`, `n_atoms`) are used identically across Tasks 3,
4, 6, 8, 10, 12. `_chalcedon.butina_cluster`/`greedy_cluster_split` call
signatures in Task 6 match Task 1's vendored copies exactly.
