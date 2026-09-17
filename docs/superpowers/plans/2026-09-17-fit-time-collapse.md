# Fit-Time Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse molecules that are equivalent under this series' model premise — a molecule's conformers, exact duplicate structures, and enantiomers — at fit time only, leaving held-out rows one per conformer so the metric still checks the premise.

**Architecture:** A pure `collapse_key(mol)` function defines the equivalence. A store op writes that key plus three counts as columns, once. A `collapse_molecule_set()` transform groups a training `MoleculeSet` by key and averages targets through a canonical atom correspondence. The CV drivers apply that transform to the training side only. Nothing in `sieve` core, DASH's stats, or HOSE's tables changes.

**Tech Stack:** Python 3.12, RDKit, NumPy, PyArrow/pandas, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md` — read it before Task 1. The plan argues from it.

## Global Constraints

- Run every command from the repo root with the project venv: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/ty`. A bare `python` resolves to a different interpreter and will fail on `import experiments`.
- Before committing, all three must pass: `.venv/bin/ruff check src tests experiments`, `.venv/bin/ruff format --check src tests experiments`, `.venv/bin/ty check src tests experiments`. `ty` must report exactly **2 diagnostics** (two pre-existing `assert_array_equal` ones, in `experiments/tests/test_data.py:164` and `tests/test_batch.py:419`). More than 2 means you introduced one.
- The installed `ty` is newer than CI's, so a clean local `ty` is a **lower bound** on what CI reports. If a new `np.savez`-style splat is added, expect a numpy-stubs false positive and add the file to the existing `[[tool.ty.overrides]]` include list in `pyproject.toml` whose comment begins "numpy-stubs gap".
- HOSE tests need the optional `hosegen`; it is installed. If an import fails, reinstall with `/home/craabreu/miniforge3/bin/uv pip install --python .venv/bin/python --no-deps "hose-code-generator @ git+https://github.com/Ratsemaat/HOSE-code-generator"`.
- Do **not** run anything against the real `dash-molecules` store. Every test in this plan uses tiny synthetic stores. The real store has ~1M conformers and any full pass takes 10+ minutes.
- Commit after every task. End commit messages with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 1: The equivalence key

**Files:**
- Create: `experiments/experiments/collapse.py`
- Test: `experiments/tests/test_collapse.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `collapse_key(mol: Any) -> str` and `mirror_mol(mol: Any) -> Any`, both in `experiments.collapse`. Tasks 2 and 3 import them.

- [ ] **Step 1: Write the failing test**

Create `experiments/tests/test_collapse.py`:

```python
"""The equivalence key: what it merges, and what it must not."""

from __future__ import annotations

import pytest

from rdkit import Chem


def _mol(smiles: str):
    """A molecule with explicit hydrogens, as the store holds them."""
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_same_molecule_written_differently_shares_a_key():
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("CCO")) == collapse_key(_mol("OCC"))


def test_enantiomers_share_a_key():
    """Measured interchangeable (0.00780 per-atom RMS, the noise floor), and
    no featurization in this series reads chirality."""
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("N[C@@H](C)C(=O)O")) == collapse_key(
        _mol("N[C@H](C)C(=O)O")
    )


def test_diastereomers_do_not_share_a_key():
    """Not mirror images, and measured to differ by 0.00698 above the
    enantiomer control -- merging them would hide a real error."""
    from experiments.collapse import collapse_key

    a = _mol("C[C@H](O)[C@H](N)C")
    b = _mol("C[C@H](O)[C@@H](N)C")
    assert collapse_key(a) != collapse_key(b)


def test_ez_isomers_do_not_share_a_key():
    """mirror_mol leaves bond stereo untouched, so E and Z keys differ."""
    from experiments.collapse import collapse_key

    assert collapse_key(_mol("C/C=C/C")) != collapse_key(_mol("C/C=C\\\\C"))


def test_achiral_molecule_key_is_its_own_canonical_smiles():
    from experiments.collapse import collapse_key

    m = _mol("c1ccccc1")
    assert collapse_key(m) == Chem.MolToSmiles(m)


def test_mirror_of_a_mirror_is_the_original():
    from experiments.collapse import mirror_mol

    m = _mol("N[C@@H](C)C(=O)O")
    twice = mirror_mol(mirror_mol(m))
    assert Chem.MolToSmiles(twice) == Chem.MolToSmiles(m)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -v`
Expected: every test FAILS with `ModuleNotFoundError: No module named 'experiments.collapse'`.

- [ ] **Step 3: Implement the minimal code to make the tests pass**

Create `experiments/experiments/collapse.py`:

```python
"""Which molecules this series' model premise makes equivalent.

A molecule's conformers, an exactly duplicated structure, and an enantiomer
are indistinguishable to every arm here: they share one graph, and no
featurization in this series reads chirality. Diastereomers and E/Z isomers
are NOT equivalent -- they are measurably different molecules (spec section 1)
and merging them would define the target as a mean over different chemistry.

See docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md section 3.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

_CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW
_CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CCW


def mirror_mol(mol: Any) -> Any:
    """A copy with every tetrahedral centre inverted, bond stereo untouched.

    Bond stereo is deliberately preserved: a mirror image has the opposite
    handedness at every stereocentre but the same E/Z geometry, which is what
    makes E/Z isomers survive the key while enantiomers do not.
    """
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == _CW:
            atom.SetChiralTag(_CCW)
        elif tag == _CCW:
            atom.SetChiralTag(_CW)
    return out


def collapse_key(mol: Any) -> str:
    """The equivalence class of ``mol``, as a canonical string.

    ``min`` over the molecule and its mirror picks the same representative of
    an enantiomer pair from either side, so the two share a key without any
    pairwise comparison.
    """
    return min(Chem.MolToSmiles(mol), Chem.MolToSmiles(mirror_mol(mol)))
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint, then commit**

```bash
.venv/bin/ruff format experiments/experiments/collapse.py experiments/tests/test_collapse.py
.venv/bin/ruff check src tests experiments
.venv/bin/ty check src tests experiments   # must say "Found 2 diagnostics"
git add experiments/experiments/collapse.py experiments/tests/test_collapse.py
git commit -m "feat(experiments): the collapse equivalence key

min(canonical_smiles(mol), canonical_smiles(mirror(mol))) merges a molecule's
conformers, exact duplicate structures and enantiomers, and merges nothing
else: diastereomers differ in canonical SMILES and are not each other's
mirror, and mirror_mol leaves bond stereo untouched so E/Z isomers keep
distinct keys.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Annotate a store with the key and its counts

**Files:**
- Modify: `experiments/experiments/store_ops.py` (append a new function at end of file)
- Modify: `experiments/experiments/cli.py` (add `_cmd_annotate_collapse` beside `_cmd_to_united_atom`, and a parser beside the `to-united-atom` one at line ~1335)
- Test: `experiments/tests/test_collapse.py` (append)

**Interfaces:**
- Consumes: `collapse_key` from Task 1.
- Produces: `annotate_collapse(store: str, *, stores_root: Path) -> dict[str, int]` in `experiments.store_ops`, returning `{"rows": int, "keys": int}`. Task 4's docs reference the CLI name `annotate-collapse`.

- [ ] **Step 1: Write the failing test**

Append to `experiments/tests/test_collapse.py`:

```python
def _write_store(tmp_path, rows):
    """A minimal store: one row per (smiles, dash_id, split, cluster, shard)."""
    import pandas as pd
    from experiments.data import mol_to_blob

    recs = []
    for smiles, did, split, cluster, shard in rows:
        m = _mol(smiles)
        for atom in m.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 0.1 * atom.GetAtomicNum())
        recs.append(
            {
                "mol": mol_to_blob(m),
                "dash_id": did,
                "conf_id": "c0",
                "net_charge": 0.0,
                "split": split,
                "cluster": cluster,
                "shard": shard,
            }
        )
    store = tmp_path / "s"
    store.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(store / "molecules.parquet")
    return "s", tmp_path


def test_annotate_collapse_adds_key_and_counts(tmp_path):
    """Two conformers of one molecule, plus its enantiomer, plus an unrelated
    molecule: the first three share a key, the fourth does not."""
    import pandas as pd

    from experiments.store_ops import annotate_collapse

    store, root = _write_store(
        tmp_path,
        [
            ("N[C@@H](C)C(=O)O", "d1", "train", 0, "s00"),
            ("N[C@@H](C)C(=O)O", "d1", "train", 0, "s00"),
            ("N[C@H](C)C(=O)O", "d2", "train", 0, "s00"),
            ("CCO", "d3", "train", 1, "s01"),
        ],
    )
    out = annotate_collapse(store, stores_root=root)
    assert out == {"rows": 4, "keys": 2}

    df = pd.read_parquet(root / store / "molecules.parquet")
    assert set(df["collapse_key"]).__len__() == 2
    ala = df[df["dash_id"].isin(["d1", "d2"])]
    assert set(ala["n_collapsed"]) == {3}
    assert set(ala["n_molecules"]) == {2}
    assert set(ala["n_enantiomer_forms"]) == {2}
    other = df[df["dash_id"] == "d3"]
    assert set(other["n_collapsed"]) == {1}
    assert set(other["n_enantiomer_forms"]) == {1}


def test_annotate_collapse_is_idempotent(tmp_path):
    from experiments.store_ops import annotate_collapse

    store, root = _write_store(tmp_path, [("CCO", "d1", "train", 0, "s00")])
    first = annotate_collapse(store, stores_root=root)
    second = annotate_collapse(store, stores_root=root)
    assert first == second


def test_annotate_collapse_refuses_a_group_that_straddles_a_shard(tmp_path):
    """Verified not to happen on the real corpus; asserted so a future one
    cannot break it silently (spec section 4)."""
    import pytest

    from experiments.store_ops import annotate_collapse

    store, root = _write_store(
        tmp_path,
        [
            ("CCO", "d1", "train", 0, "s00"),
            ("CCO", "d2", "train", 0, "s01"),
        ],
    )
    with pytest.raises(ValueError, match="straddles"):
        annotate_collapse(store, stores_root=root)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -k annotate -v`
Expected: FAIL with `ImportError: cannot import name 'annotate_collapse'`.

- [ ] **Step 3: Implement the minimal code to make the tests pass**

Append to `experiments/experiments/store_ops.py`:

```python
def annotate_collapse(store: str, *, stores_root: Path) -> dict[str, int]:
    """Add ``collapse_key`` and its three counts to an existing store.

    In place, like ``prepare-store --n-shards`` adding ``cluster``/``shard``.
    Idempotent: a store already carrying the columns is recounted, not
    rewritten differently.

    Refuses a key group that straddles a ``split``, ``cluster`` or ``shard``.
    On the real corpus none does -- identical molecules share a fingerprint,
    so Butina cannot separate them, and both the split and the sharding are by
    whole cluster -- and asserting it means a future corpus that breaks the
    property stops rather than silently averaging across a fold boundary.
    """
    import pandas as pd

    from experiments.collapse import collapse_key
    from experiments.data import blob_to_mol

    path = Path(stores_root) / store / "molecules.parquet"
    df = pd.read_parquet(path)
    df["collapse_key"] = [collapse_key(blob_to_mol(b)) for b in df["mol"]]

    for column in ("split", "cluster", "shard"):
        if column not in df.columns:
            continue
        spread = df.groupby("collapse_key")[column].nunique()
        bad = spread[spread > 1]
        if len(bad):
            raise ValueError(
                f"collapse group straddles {column!r}: "
                f"{list(bad.index[:3])} (and {max(len(bad) - 3, 0)} more)"
            )

    grouped = df.groupby("collapse_key")
    df["n_collapsed"] = grouped["collapse_key"].transform("size").astype("int32")
    df["n_molecules"] = grouped["dash_id"].transform("nunique").astype("int32")
    df["n_enantiomer_forms"] = (
        df.assign(_canon=[Chem.MolToSmiles(blob_to_mol(b)) for b in df["mol"]])
        .groupby("collapse_key")["_canon"]
        .transform("nunique")
        .astype("int32")
    )
    df.to_parquet(path)
    return {"rows": int(len(df)), "keys": int(df["collapse_key"].nunique())}
```

Add `from rdkit import Chem` to `store_ops.py`'s imports if it is not already there.

Then add the CLI command. In `experiments/experiments/cli.py`, beside `_cmd_to_united_atom`:

```python
def _cmd_annotate_collapse(args: argparse.Namespace) -> int:
    from experiments.store_ops import annotate_collapse

    out = annotate_collapse(args.store, stores_root=DEFAULT_STORES_ROOT)
    print(f"{out['rows']} rows -> {out['keys']} collapse keys")
    return 0
```

and beside the `to-united-atom` parser (~line 1335):

```python
    p_annotate = sub.add_parser(
        "annotate-collapse",
        help="add collapse_key and its counts to a store, in place -- the "
        "equivalence a fit collapses over (conformers, exact duplicates, "
        "enantiomers). Idempotent; refuses a group straddling a split, "
        "cluster or shard",
    )
    p_annotate.add_argument("store", nargs="?", default="dash-molecules")
    p_annotate.set_defaults(func=_cmd_annotate_collapse)
```

If `DEFAULT_STORES_ROOT` is not already defined in `cli.py`, use the same expression the `to-united-atom` command uses for its `stores_root`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -v`
Expected: 9 passed.

Also check the CLI is wired: `.venv/bin/python -m experiments annotate-collapse --help`
Expected: usage text mentioning `collapse_key`.

- [ ] **Step 5: Lint, then commit**

```bash
.venv/bin/ruff format experiments/experiments/store_ops.py experiments/experiments/cli.py experiments/tests/test_collapse.py
.venv/bin/ruff check src tests experiments
.venv/bin/ty check src tests experiments
git add experiments/experiments/store_ops.py experiments/experiments/cli.py experiments/tests/test_collapse.py
git commit -m "feat(experiments): annotate a store with collapse_key and its counts

Adds collapse_key plus n_collapsed, n_molecules and n_enantiomer_forms to an
existing store in place, the pattern prepare-store --n-shards already uses for
cluster/shard. The canonical-SMILES pass is paid once rather than per fit.

Refuses a key group that straddles a split, cluster or shard. None does on the
real corpus; asserting it stops a future corpus from silently averaging across
a fold boundary.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Collapse a MoleculeSet

**Files:**
- Modify: `experiments/experiments/collapse.py`
- Test: `experiments/tests/test_collapse.py` (append)

**Interfaces:**
- Consumes: `collapse_key`, `mirror_mol` from Task 1; `collapse_key` column present in `MoleculeSet.ids` (put there automatically by `load_molecule_set`, which routes unrecognised store columns into `ids`).
- Produces: `collapse_molecule_set(mset: MoleculeSet, *, weight_by_collapse: bool = False) -> MoleculeSet` in `experiments.collapse`. Task 4 calls it.

**Why the atom correspondence is not the identity:** two records of one structure may order their atoms differently, and an enantiomer's atoms do not correspond positionally to its mirror partner's. Every member is therefore re-ordered into the canonical order of whichever of itself or its mirror matches the key (spec section 5).

- [ ] **Step 1: Write the failing test**

Append to `experiments/tests/test_collapse.py`:

```python
def _charged(smiles: str, charges: dict[int, float]):
    """A molecule whose MBIScharge is set per atom index."""
    m = _mol(smiles)
    for atom in m.GetAtoms():
        atom.SetDoubleProp("MBIScharge", charges.get(atom.GetIdx(), 0.0))
    return m


def test_collapse_averages_conformers_of_one_molecule():
    import numpy as np

    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0})
    b = _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0})
    key = collapse_key(a)
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={"collapse_key": [key, key], "dash_id": ["d1", "d1"],
             "conf_id": ["c0", "c1"]},
    )
    out = collapse_molecule_set(mset)
    assert out.n_conformers == 1
    heavy = out.atom_target[:3]
    np.testing.assert_allclose(sorted(heavy), sorted([2.0, 3.0, 4.0]))


def test_collapse_leaves_distinct_keys_alone():
    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0})
    b = _charged("CCC", {0: 1.0})
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={"collapse_key": [collapse_key(a), collapse_key(b)],
             "dash_id": ["d1", "d2"], "conf_id": ["c0", "c0"]},
    )
    assert collapse_molecule_set(mset).n_conformers == 2


def test_collapse_matches_enantiomer_atoms_through_the_mirror():
    """The correspondence is the point: averaging positionally would pair
    atoms that are not the same atom."""
    import numpy as np

    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    a = _charged("N[C@@H](C)C(=O)O", {})
    b = _charged("N[C@H](C)C(=O)O", {})
    for atom in a.GetAtoms():
        atom.SetDoubleProp("MBIScharge", float(atom.GetAtomicNum()))
    for atom in b.GetAtoms():
        atom.SetDoubleProp("MBIScharge", float(atom.GetAtomicNum()))
    key = collapse_key(a)
    assert key == collapse_key(b)
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={"collapse_key": [key, key], "dash_id": ["d1", "d2"],
             "conf_id": ["c0", "c0"]},
    )
    out = collapse_molecule_set(mset)
    assert out.n_conformers == 1
    # charges were set to the atomic number, identical under any correct
    # correspondence, so every averaged value must still be an integer
    np.testing.assert_allclose(out.atom_target, np.round(out.atom_target))


def test_collapse_equals_fractional_weighting_for_the_mean():
    """The equivalence that justifies the design (spec section 5): the
    collapsed class mean equals the 1/n_collapsed-weighted mean of all rows."""
    import numpy as np

    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    rows = [
        _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
        _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
        _charged("CCO", {0: 5.0, 1: 6.0, 2: 7.0}),
    ]
    key = collapse_key(rows[0])
    mset = MoleculeSet(
        mols=rows,
        atom_property="MBIScharge",
        ids={"collapse_key": [key] * 3, "dash_id": ["d1"] * 3,
             "conf_id": ["c0", "c1", "c2"]},
    )
    collapsed = sorted(collapse_molecule_set(mset).atom_target[:3])
    # the 1/n-weighted mean of all rows, computed independently per atom index
    fractional = sorted(
        sum(m.GetAtomWithIdx(i).GetDoubleProp("MBIScharge") for m in rows) / len(rows)
        for i in range(3)
    )
    np.testing.assert_allclose(collapsed, fractional)
    np.testing.assert_allclose(collapsed, [3.0, 4.0, 5.0])


def test_weight_by_collapse_repeats_each_representative():
    """The migration setting: weighting by n_collapsed must reproduce the
    uncollapsed row count and leave the mean unchanged (spec section 9)."""
    import numpy as np

    from experiments.collapse import collapse_key, collapse_molecule_set
    from experiments.data import MoleculeSet

    rows = [
        _charged("CCO", {0: 1.0, 1: 2.0, 2: 3.0}),
        _charged("CCO", {0: 3.0, 1: 4.0, 2: 5.0}),
        _charged("CCC", {0: 9.0}),
    ]
    keys = [collapse_key(m) for m in rows]
    mset = MoleculeSet(
        mols=rows,
        atom_property="MBIScharge",
        ids={"collapse_key": keys, "dash_id": ["d1", "d1", "d2"],
             "conf_id": ["c0", "c1", "c0"]},
    )
    plain = collapse_molecule_set(mset)
    weighted = collapse_molecule_set(mset, weight_by_collapse=True)

    assert plain.n_conformers == 2                 # two keys
    assert weighted.n_conformers == 3              # CCO twice, CCC once
    # the CCO representative carries the same averaged target either way
    np.testing.assert_allclose(
        sorted(plain.atom_target[: rows[0].GetNumAtoms()]),
        sorted(weighted.atom_target[: rows[0].GetNumAtoms()]),
    )
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -k collapse_ -v`
Expected: FAIL with `ImportError: cannot import name 'collapse_molecule_set'`.

- [ ] **Step 3: Implement the minimal code to make the tests pass**

First extend the import block at the **top** of
`experiments/experiments/collapse.py` — appending imports mid-file trips ruff's
E402:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import CanonicalRankAtoms

from experiments.data import MoleculeSet
```

Then append to the same file:

```python
def _canonical_order(mol: Any, key: str) -> np.ndarray:
    """Positions in the key's canonical order -> atom indices in ``mol``.

    A member whose own canonical SMILES is not the key is its mirror, so the
    ranking is taken on the mirrored copy; chiral-tag inversion does not move
    atom indices, so the resulting order applies to ``mol`` unchanged.
    """
    ranked = mol if Chem.MolToSmiles(mol) == key else mirror_mol(mol)
    return np.argsort(np.asarray(CanonicalRankAtoms(ranked)))


def collapse_molecule_set(
    mset: MoleculeSet, *, weight_by_collapse: bool = False
) -> MoleculeSet:
    """One conformer per collapse key, carrying that key's mean target.

    For FITTING only. Held-out sets are never collapsed: averaging a test
    target would encode the equivalence premise into the metric, so the metric
    could no longer detect the premise being false (spec section 2).

    ``weight_by_collapse`` repeats each representative ``n_collapsed`` times,
    reproducing the uncollapsed, atom-weighted fit. It exists for the
    migration check and is not the recommended setting.
    """
    keys = mset.ids.get("collapse_key")
    if keys is None:
        raise ValueError(
            "MoleculeSet has no 'collapse_key'; run `experiments "
            "annotate-collapse <store>` first"
        )
    dash = mset.ids.get("dash_id") or [""] * mset.n_conformers
    conf = mset.ids.get("conf_id") or [""] * mset.n_conformers

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(str(k), []).append(i)

    mols: list[Any] = []
    ids: dict[str, list[Any]] = {name: [] for name in mset.ids}
    values: list[float] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda i: (str(dash[i]), str(conf[i])))
        orders = [_canonical_order(mset.mols[i], key) for i in members]
        stacked = np.stack(
            [
                np.array(
                    [
                        mset.mols[i].GetAtomWithIdx(int(a)).GetDoubleProp(
                            mset.atom_property
                        )
                        for a in order
                    ]
                )
                for i, order in zip(members, orders, strict=True)
            ]
        )
        mean = stacked.mean(axis=0)

        rep_i = members[0]
        rep = Chem.Mol(mset.mols[rep_i])
        for position, atom_idx in enumerate(orders[0]):
            rep.GetAtomWithIdx(int(atom_idx)).SetDoubleProp(
                mset.atom_property, float(mean[position])
            )
        repeats = len(members) if weight_by_collapse else 1
        for _ in range(repeats):
            mols.append(rep)
            for name in mset.ids:
                ids[name].append(mset.ids[name][rep_i])
            if mset.molecule_value is not None:
                values.append(float(mset.molecule_value[rep_i]))

    return MoleculeSet(
        mols=mols,
        atom_property=mset.atom_property,
        molecule_property=mset.molecule_property,
        molecule_value=(None if mset.molecule_value is None else np.array(values)),
        ids=ids,
        split=None,
    )
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -v`
Expected: 14 passed.

- [ ] **Step 5: Lint, then commit**

```bash
.venv/bin/ruff format experiments/experiments/collapse.py experiments/tests/test_collapse.py
.venv/bin/ruff check src tests experiments
.venv/bin/ty check src tests experiments
git add experiments/experiments/collapse.py experiments/tests/test_collapse.py
git commit -m "feat(experiments): collapse a MoleculeSet by equivalence key

One conformer per key, carrying that key's mean target. The atom
correspondence is not the identity -- two records of one structure may order
atoms differently, and an enantiomer's atoms do not correspond positionally to
its mirror partner's -- so every member is re-ordered into the canonical order
of whichever of itself or its mirror matches the key.

For fitting only. weight_by_collapse repeats each representative n_collapsed
times, reproducing the uncollapsed atom-weighted fit for the migration check.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Apply the collapse in the CV drivers

**Files:**
- Modify: `experiments/experiments/cv.py` — `fit_sieve_shard`, `fit_dash_shard`, `fit_hose_shard`, and their `run_*_shard_fits` wrappers
- Modify: `experiments/experiments/cli.py` — add `--collapse` and `--weight-by-collapse` to `cv-fit-sieve-shards`, `cv-fit-dash-shards`, `cv-fit-hose-shards`
- Test: `experiments/tests/test_collapse.py` (append)

**Interfaces:**
- Consumes: `collapse_molecule_set` from Task 3.
- Produces: keyword arguments `collapse: bool = False` and `weight_by_collapse: bool = False` on `fit_sieve_shard`, `fit_dash_shard`, `fit_hose_shard` and each `run_*_shard_fits`.

**Where the call goes:** each `fit_*_shard` already builds `train = mset.select(masks[shard])`. Insert the collapse immediately after that line and nowhere else — the held-out path in `run_*_cv` must remain untouched.

- [ ] **Step 1: Write the failing test**

Append to `experiments/tests/test_collapse.py`:

```python
def test_fit_sieve_shard_collapses_only_the_training_side(tmp_path):
    """The fit sees one row per key; the store and every held-out path still
    hold one row per conformer."""
    import json

    import pandas as pd

    from experiments.cv import fit_sieve_shard
    from experiments.predictors.sieve_predictor import _build_config, save_codes
    from experiments.store_ops import annotate_collapse
    from experiments.tests.helpers import synthetic_molecule_set

    store, root = _write_store(
        tmp_path,
        [
            ("CCO", "d1", "train", 0, "s00"),
            ("CCO", "d1", "train", 0, "s00"),
            ("CCC", "d2", "train", 0, "s00"),
        ],
    )
    annotate_collapse(store, stores_root=root)

    whole = synthetic_molecule_set(n_mol=4, seed=0)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=1,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    out = fit_sieve_shard(
        store=store,
        shard="s00",
        depth=1,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=tmp_path / "runs",
        stores_root=root,
        allow_dirty=True,
        collapse=True,
    )
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert manifest["n_train_conformers"] == 2      # 3 rows -> 2 keys
    assert manifest["collapse"] is True

    # the store itself is untouched
    assert len(pd.read_parquet(root / store / "molecules.parquet")) == 3
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -k fit_sieve_shard_collapses -v`
Expected: FAIL with `TypeError: fit_sieve_shard() got an unexpected keyword argument 'collapse'`.

- [ ] **Step 3: Write the minimal implementation**

In `experiments/experiments/cv.py`, for **each** of `fit_sieve_shard`, `fit_dash_shard` and `fit_hose_shard`:

1. Add to the signature, after `atom_property`:

```python
    collapse: bool = False,
    weight_by_collapse: bool = False,
```

2. Immediately after the existing `train = mset.select(masks[shard])` line, insert:

```python
    if collapse:
        from experiments.collapse import collapse_molecule_set

        train = collapse_molecule_set(train, weight_by_collapse=weight_by_collapse)
```

3. In that function's `manifest.json` payload dict, add two entries beside `"shard"`:

```python
                "collapse": collapse,
                "weight_by_collapse": weight_by_collapse,
```

Then thread both through each `run_*_shard_fits` wrapper: add the same two keyword arguments with the same defaults, and pass them to the `fit_*_shard` call inside.

In `experiments/experiments/cli.py`, add to each of the three `cv-fit-*-shards` parsers:

```python
    p_cv_fit_sieve.add_argument(
        "--collapse",
        action="store_true",
        help="fit on one row per collapse_key (conformers, exact duplicates "
        "and enantiomers merged). Requires `annotate-collapse` first. The "
        "held-out side is never collapsed",
    )
    p_cv_fit_sieve.add_argument(
        "--weight-by-collapse",
        action="store_true",
        help="with --collapse, weight each key by n_collapsed, reproducing "
        "the uncollapsed atom-weighted fit. For the migration check only",
    )
```

(substituting `p_cv_fit_dash` and `p_cv_fit_hose` for the other two), and pass
`collapse=args.collapse, weight_by_collapse=args.weight_by_collapse` in each
corresponding `_cmd_cv_fit_*_shards` function.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py experiments/tests/test_cv.py -v`
Expected: all pass, 15 in `test_collapse.py`.

- [ ] **Step 5: Lint, then commit**

```bash
.venv/bin/ruff format experiments/experiments/cv.py experiments/experiments/cli.py experiments/tests/test_collapse.py
.venv/bin/ruff check src tests experiments
.venv/bin/ty check src tests experiments
git add experiments/experiments/cv.py experiments/experiments/cli.py experiments/tests/test_collapse.py
git commit -m "feat(experiments): --collapse on the shard fits, training side only

Each fit_*_shard collapses its training MoleculeSet by collapse_key after
selecting its shard. The held-out path in run_*_cv is untouched: collapsing a
test target would encode the equivalence premise into the metric, so the
metric could no longer detect the premise being false.

The manifest records collapse and weight_by_collapse, so a run's provenance
says which premise it was fitted under.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The held-out irreducible-floor diagnostic

**Files:**
- Modify: `experiments/experiments/cv.py` — `_write_cv_run`
- Test: `experiments/tests/test_collapse.py` (append)

**Interfaces:**
- Consumes: nothing from Task 3; reads `held_out.ids["collapse_key"]` directly.
- Produces: a `floor/rmse` entry in every CV run's `metrics.json` when the held-out set carries `collapse_key`; absent otherwise.

**What it measures:** within a held-out fold, rows sharing a key are identical to every arm here, so any spread among their targets is error no graph-based model can avoid. Reporting it turns the equivalence premise into a measured number per study (spec section 7).

- [ ] **Step 1: Write the failing test**

Append to `experiments/tests/test_collapse.py`:

```python
def test_held_out_floor_is_the_within_key_scatter():
    import numpy as np

    from experiments.collapse import collapse_key, held_out_floor
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0, 1: 1.0, 2: 1.0})
    b = _charged("CCO", {0: 3.0, 1: 3.0, 2: 3.0})
    c = _charged("CCC", {0: 5.0})
    key_ab, key_c = collapse_key(a), collapse_key(c)
    mset = MoleculeSet(
        mols=[a, b, c],
        atom_property="MBIScharge",
        ids={"collapse_key": [key_ab, key_ab, key_c],
             "dash_id": ["d1", "d2", "d3"], "conf_id": ["c0", "c0", "c0"]},
    )
    # the CCO pair differs by 2.0 on every atom, so each deviates by 1.0;
    # CCC is alone and contributes nothing
    floor = held_out_floor(mset)
    n_cco_atoms = a.GetNumAtoms() * 2
    expected = np.sqrt(n_cco_atoms * 1.0**2 / (n_cco_atoms + c.GetNumAtoms()))
    assert floor == pytest.approx(expected)


def test_held_out_floor_is_zero_without_duplicates():
    from experiments.collapse import collapse_key, held_out_floor
    from experiments.data import MoleculeSet

    a = _charged("CCO", {0: 1.0})
    b = _charged("CCC", {0: 2.0})
    mset = MoleculeSet(
        mols=[a, b],
        atom_property="MBIScharge",
        ids={"collapse_key": [collapse_key(a), collapse_key(b)],
             "dash_id": ["d1", "d2"], "conf_id": ["c0", "c0"]},
    )
    assert held_out_floor(mset) == pytest.approx(0.0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py -k floor -v`
Expected: FAIL with `ImportError: cannot import name 'held_out_floor'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `experiments/experiments/collapse.py`:

```python
def held_out_floor(mset: MoleculeSet) -> float:
    """RMS of the within-key deviations across a held-out set.

    Rows sharing a collapse key are identical to every arm in this series, so
    any spread among their targets is error no graph-based model can avoid.
    Returns 0.0 when no key repeats, and 0.0 when the set carries no
    ``collapse_key`` at all.
    """
    keys = mset.ids.get("collapse_key")
    if keys is None or mset.n_atoms == 0:
        return 0.0
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(str(k), []).append(i)

    sse = 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        key = str(keys[members[0]])
        orders = [_canonical_order(mset.mols[i], key) for i in members]
        stacked = np.stack(
            [
                np.array(
                    [
                        mset.mols[i].GetAtomWithIdx(int(a)).GetDoubleProp(
                            mset.atom_property
                        )
                        for a in order
                    ]
                )
                for i, order in zip(members, orders, strict=True)
            ]
        )
        sse += float(((stacked - stacked.mean(axis=0)) ** 2).sum())
    return float(np.sqrt(sse / mset.n_atoms))
```

Then in `experiments/experiments/cv.py`, inside `_write_cv_run`, after the
metrics dict is assembled and before it is written, add:

```python
    from experiments.collapse import held_out_floor

    floor = held_out_floor(held_out)
    if floor:
        metrics["floor/rmse"] = floor
```

Locate the metrics dict by searching `_write_cv_run` for where
`_score_raw_and_normalized(...)` is assigned; add the lines just after the
`train/` family is merged in.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `.venv/bin/pytest experiments/tests/test_collapse.py experiments/tests/test_cv.py -v`
Expected: all pass, 17 in `test_collapse.py`.

- [ ] **Step 5: Lint, then commit**

```bash
.venv/bin/ruff format experiments/experiments/collapse.py experiments/experiments/cv.py experiments/tests/test_collapse.py
.venv/bin/ruff check src tests experiments
.venv/bin/ty check src tests experiments
git add experiments/experiments/collapse.py experiments/experiments/cv.py experiments/tests/test_collapse.py
git commit -m "feat(experiments): report the held-out irreducible floor

Rows sharing a collapse key are identical to every arm here, so any spread
among their held-out targets is error no graph-based model can avoid.
Recorded as floor/rmse on every CV run whose held-out set carries the key.

This is what turns the equivalence premise into a measured number per study
rather than a standing assumption: if it drifts upward, the premise is
failing on new data.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full-suite check and the migration procedure

**Files:**
- Modify: `experiments/README.md` — append a "Collapsing equivalent molecules" subsection under "Running an experiment"

**Interfaces:**
- Consumes: the CLI surface from Tasks 2 and 4.
- Produces: no code. Documents the migration check the spec requires before any result is re-run.

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest experiments/tests tests -q`
Expected: all pass. Baseline before this plan was **685 passed, 5 skipped**; this plan adds 17 tests, so expect **702 passed, 5 skipped**. Any failure is a regression from this plan — fix it before continuing.

- [ ] **Step 2: Run all three lint steps**

```bash
.venv/bin/ruff check src tests experiments
.venv/bin/ruff format --check src tests experiments
.venv/bin/ty check src tests experiments
```
Expected: "All checks passed!", "N files already formatted", "Found 2 diagnostics".

- [ ] **Step 3: Document the migration check**

Append to `experiments/README.md`, under the "Running an experiment" section:

```markdown
### Collapsing equivalent molecules

A molecule's conformers, an exactly duplicated structure and an enantiomer are
indistinguishable to every predictor here, so counting them separately
reweights class means for no informational reason. Annotate the store once:

    uv run python -m experiments annotate-collapse dash-molecules

then pass `--collapse` to any `cv-fit-*-shards` command. The collapse applies
to the **training** side only; held-out rows stay one per conformer, so the
metric still measures per-conformer error and still detects the equivalence
premise failing. Every CV run whose store carries the key also records
`floor/rmse`, the within-key scatter on its own held-out set -- the error no
graph-based model can avoid.

Diastereomers and E/Z isomers are deliberately NOT merged: they are
measurably different molecules.

**Before re-running any published result**, check the migration: fit one arm
with `--collapse --weight-by-collapse` and confirm it reproduces that arm's
existing number exactly. That validates the annotation, the grouping and the
representative selection at once. `--collapse` alone changes the fit, which is
the point; `minimum_support` then counts distinct structures, so the present
value of 12 (chosen as "at least four molecules") becomes 4.

See `docs/superpowers/specs/2026-09-17-fit-time-collapse-design.md`.
```

- [ ] **Step 4: Verify the README renders the commands correctly**

Run: `grep -A 3 "annotate-collapse dash-molecules" experiments/README.md`
Expected: the command appears inside the indented code block.

- [ ] **Step 5: Commit**

```bash
git add experiments/README.md
git commit -m "docs(experiments): how to collapse, and the migration check

Records that the collapse is training-side only and why, that diastereomers
and E/Z isomers are deliberately excluded, and that --weight-by-collapse must
reproduce an existing number exactly before any published result is re-run.

Notes the minimum_support semantics change: it now counts distinct structures,
so 12 (chosen as 'at least four molecules') becomes 4.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope for this plan

Named so the executor does not drift into them:

- Re-running Studies A, B or C under the collapsed premise. A separate decision with its own cost.
- Re-tuning the empirical-Bayes arms for the new counts. Needed before the shrinkage variants mean anything post-collapse, but it is an experiment, not an implementation.
- Any change to `sieve` core, `merge.py`, DASH's `compute_node_stats`, or HOSE's tables. The collapse lives entirely in batch construction.
- Fractional (`1/n_collapsed`) weighting as a general mechanism. Spec section 5 records what it would need and why it is not equivalent under shrinkage.
- Running `annotate-collapse` against the real `dash-molecules` store. That is a ~10-minute pass and a deliberate, separate step.
