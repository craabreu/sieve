# Experiments Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the DASH-charge-specific `charge_experiments` harness into a dataset-agnostic `experiments` harness whose config names the atomic property to train and predict, leaving the data-preparation script as the only dataset-specific component.

**Architecture:** Two phases. Phase 1 (Task 1) is a pure mechanical rename — package, tests, configs, path constants, packaging metadata — proven correct by the existing suite passing unchanged. Phase 2 (Tasks 2-9) introduces a `target:` config section and threads the configured property name through the data model, predictors, metrics, normalizers and plots, renaming charge-specific identifiers as it goes. Each task is test-first and ends green.

**Tech Stack:** Python 3.12, numpy, pandas/pyarrow, RDKit (optional dep), matplotlib, MLflow (optional), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-09-experiments-generalization-design.md`

## Global Constraints

- **No changes to `src/sieve/**`.** The sieve core is out of scope; the harness adapts to its existing API (`from_rdkit(..., y_from_atom_prop=<str>)`, scalar target, y shape `(n, 1)`).
- **No store rewrites.** Every existing `molecules.parquet` under `stores/` stays byte-identical. The harness reads whichever columns the config names.
- **Data does not move.** `stores/`, `runs/`, `cache/`, `external/`, `mlflow_artifacts/`, `results/`, `mlflow_runs.db` stay physically under `charge_experiments/` and are reached from `experiments/` through symlinks.
- **Scalar target only.** Exactly one atom property per run.
- **Metric keys rename outright:** `charge_conservation/{mae,rmse,r2}` → `sum_constraint/{mae,rmse,r2}`. No legacy alias in the aggregator; old run dirs will show blanks in those `summarize` columns, and that is accepted.
- **`target.atom_property` is required; `target.molecule_property` and `target.label` are optional.** `label` defaults to `atom_property`.
- **Charge numbers must not change.** Any DASH-charge run's metric *values* must be numerically identical before and after; only key names and field names change.
- **Run tests with `uv run pytest`** from the repo root. Full suite: `uv run pytest`. Lint: `uv run ruff check .` and `uv run ruff format --check .`.
- **Commit after every task.** Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt
  ```

## File Structure

**Phase 1 moves (git mv, contents otherwise unchanged):**

| From | To |
|---|---|
| `charge_experiments/charge_experiments/` | `experiments/experiments/` |
| `charge_experiments/configs/`, `docs/`, `tests/`, `README.md`, `pins.toml` | `experiments/…` |

**Phase 2 responsibilities:**

- `experiments/experiments/config.py` — YAML → frozen dataclasses. Gains `TargetCfg`; owns all cross-key validation (e.g. `normalization` requires `molecule_property`).
- `experiments/experiments/data.py` — `MoleculeSet`: conformers + the configured property names. Gains `atom_property`, `molecule_property`, `ids`; `atom_charge`→`atom_target`, `net_charge`→`molecule_value`.
- `experiments/experiments/predictors/base.py` — `Prediction.atom_value`, `RawPrediction.atom_value`.
- `experiments/experiments/metrics.py` — `sum_constraint_metrics`.
- `experiments/experiments/normalize.py` — dataset-agnostic sum-constraint schemes (names and formulas unchanged).
- `experiments/experiments/runner.py` — loading, scoring, npz, panels; gates every molecule-level output on `molecule_property`.
- `experiments/experiments/prepare_dash.py` *(new, from `prepare_store.py`)* — DASH-only: download, parse, cluster, split, `prepare_store()`.
- `experiments/experiments/store_ops.py` *(new, from `prepare_store.py`)* — dataset-agnostic `subsample_store`, `partition_store`, `to_united_atom_store`.
- `experiments/experiments/cli.py` — argparse; `SUMMARY_COLUMNS`; imports from the two split modules.

---

### Task 1: Rename the package to `experiments`

Pure mechanical move. No behavior changes anywhere — the suite passing unchanged is the proof.

**Files:**
- Move: `charge_experiments/{charge_experiments,configs,docs,tests,README.md,pins.toml}` → `experiments/…`
- Modify: `pyproject.toml`, `.gitignore`
- Modify (path constants): `experiments/experiments/data.py:25-26`, `experiments/experiments/runner.py:46,53-54`, `experiments/experiments/predictors/dash.py:55`, `experiments/experiments/tests/conftest.py`

- [ ] **Step 1: Move the tracked tree**

```bash
cd /data3/craabreu/github_repos/sieve
mkdir -p experiments
git mv charge_experiments/charge_experiments experiments/experiments
git mv charge_experiments/configs experiments/configs
git mv charge_experiments/docs experiments/docs
git mv charge_experiments/tests experiments/tests
git mv charge_experiments/README.md experiments/README.md
git mv charge_experiments/pins.toml experiments/pins.toml
```

- [ ] **Step 2: Create the data symlinks**

The ~25GB of data stays where it is; `experiments/` reaches it by link.

```bash
cd /data3/craabreu/github_repos/sieve/experiments
for d in stores runs cache external mlflow_artifacts results; do
  ln -s ../charge_experiments/$d $d
done
ln -s ../charge_experiments/mlflow_runs.db mlflow_runs.db
ls -l | grep '\->'    # verify seven links
test -d stores/dash-molecules-10fold-1 && echo "stores resolve"
```

- [ ] **Step 3: Rewrite the package name everywhere**

`charge_experiments` appears as an import path, a logger name, a CLI prog name, and in prose. Replace all of it, then fix prose by hand where the *old directory* is genuinely meant (the symlink targets in `.gitignore`).

```bash
cd /data3/craabreu/github_repos/sieve
grep -rl 'charge_experiments' --include='*.py' --include='*.yaml' --include='*.toml' \
  experiments/ pyproject.toml | xargs sed -i 's/charge_experiments/experiments/g'
grep -rn 'charge_experiments' experiments/ --include='*.py' --include='*.toml'   # expect none
```

- [ ] **Step 4: Fix the path constants**

The `sed` above already rewrote these strings; confirm each resolves through a symlink.

```python
# experiments/experiments/data.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORES_ROOT = REPO_ROOT / "experiments" / "stores"
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments" / "cache"
```

```python
# experiments/experiments/runner.py
_MLFLOW_RUNS_DB = REPO_ROOT / "experiments" / "mlflow_runs.db"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "experiments" / "mlflow_artifacts"
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
```

```python
# experiments/experiments/predictors/dash.py
_DASH_TREE_ROOT = REPO_ROOT / "experiments" / "external" / "DASH-tree"
```

Verify:

```bash
uv run python -c "
from experiments.data import DEFAULT_STORES_ROOT
from experiments.runner import DEFAULT_RUNS_ROOT
print(DEFAULT_STORES_ROOT.exists(), DEFAULT_RUNS_ROOT.exists())"
# expect: True True
```

- [ ] **Step 5: Update `pyproject.toml`**

Four places name the old directory:

```toml
[tool.setuptools.packages.find]
where = ["src", "experiments"]

[tool.setuptools.sdist]           # (whichever table currently holds it)
exclude = ["experiments/external/*"]

[tool.pytest.ini_options]
testpaths = ["tests", "experiments/tests"]

[tool.coverage.run]
include = ["src/**/*.py", "tests/**/*.py", "experiments/**/*.py"]
```

plus the ruff per-file ignore and exclude, which become
`experiments/experiments/_chalcedon/**`, and the `omit`/exclude entry that
currently names `charge_experiments/charge_experiments/data.py` →
`experiments/experiments/data.py`.

- [ ] **Step 6: Update `.gitignore`**

Keep the existing `charge_experiments/` entries (the real data still lives there) and add the symlink names so git never stages them:

```gitignore
# Experiment outputs reached through experiments/ (symlinks into
# charge_experiments/, which still physically holds the data).
experiments/runs
experiments/mlflow_runs.db
experiments/mlflow_artifacts
experiments/cache
experiments/results
experiments/stores
experiments/external
```

- [ ] **Step 7: Rename the test files**

Drop the `charge_` infix, except where the test really is about DASH.

```bash
cd /data3/craabreu/github_repos/sieve/experiments/tests
for f in test_charge_*.py; do
  git mv "$f" "$(echo "$f" | sed 's/^test_charge_/test_/')"
done
ls
```

Expected names include `test_config.py`, `test_data.py`, `test_cli.py`,
`test_metrics.py`, `test_normalize.py`, `test_plots.py`, `test_runner_mlflow_optional.py`,
`test_smoke.py`, `test_predictor_dash.py`, `test_predictor_dash_pretrained_optional.py`,
`test_predictor_sieve.py`, `test_prepare_store.py`, `test_tree_artifact.py`, `test_chalcedon.py`.

- [ ] **Step 8: Run the full suite**

```bash
cd /data3/craabreu/github_repos/sieve && uv run pytest -q
```

Expected: every test that passed before passes now, same count. Any failure here is a rename bug — fix it before continuing; do not adjust an assertion to fit.

- [ ] **Step 9: Lint**

```bash
uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 10: Smoke-test the CLI against a real store**

```bash
uv run python -m experiments run \
  --config experiments/configs/sieve-charge-example.yaml \
  --set data.store=dash-molecules-10fold-1 --limit 200
```

Expected: a run directory under `experiments/runs/` (i.e. through the symlink), with `metrics.json`, `manifest.json`, `parity_panel.png`.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: rename charge_experiments to experiments

Pure mechanical move: package, tests, configs, path constants and
packaging metadata. Data stays under charge_experiments/, reached by
symlinks from experiments/.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 2: `TargetCfg` in the config

**Files:**
- Modify: `experiments/experiments/config.py`
- Modify: every file in `experiments/configs/*.yaml`
- Test: `experiments/tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```python
  @dataclass(frozen=True)
  class TargetCfg:
      atom_property: str
      molecule_property: str | None = None
      label: str | None = None
      @property
      def axis_label(self) -> str: ...   # label or atom_property
  # ExperimentCfg gains: target: TargetCfg   (positional, after data)
  ```

- [ ] **Step 1: Write the failing tests**

Add to `experiments/tests/test_config.py`:

```python
def _minimal_raw():
    return {
        "run": {"experiment": "e", "seed": 0},
        "data": {"store": "s", "split_column": "split"},
        "target": {"atom_property": "MBIScharge"},
        "predictor": {"name": "global_mean"},
    }


def test_target_section_parsed():
    cfg = _build(_minimal_raw())
    assert cfg.target.atom_property == "MBIScharge"
    assert cfg.target.molecule_property is None
    assert cfg.target.axis_label == "MBIScharge"


def test_target_label_overrides_axis_label():
    raw = _minimal_raw()
    raw["target"]["label"] = "charge (e)"
    assert _build(raw).target.axis_label == "charge (e)"


def test_missing_target_section_raises():
    raw = _minimal_raw()
    del raw["target"]
    with pytest.raises(ValueError, match="target"):
        _build(raw)


def test_missing_atom_property_raises():
    raw = _minimal_raw()
    raw["target"] = {}
    with pytest.raises(ValueError, match="atom_property"):
        _build(raw)


def test_unknown_target_key_raises():
    raw = _minimal_raw()
    raw["target"]["units"] = "e"
    with pytest.raises(ValueError, match="units"):
        _build(raw)


def test_normalization_without_molecule_property_raises():
    raw = _minimal_raw()
    raw["normalization"] = "equal_weighted"
    with pytest.raises(ValueError, match="molecule_property"):
        _build(raw)


def test_normalization_with_molecule_property_ok():
    raw = _minimal_raw()
    raw["target"]["molecule_property"] = "net_charge"
    raw["normalization"] = "equal_weighted"
    assert _build(raw).normalization == "equal_weighted"


def test_target_round_trips_through_to_dict_and_flat_params():
    raw = _minimal_raw()
    raw["target"]["molecule_property"] = "net_charge"
    cfg = _build(raw)
    assert to_dict(cfg)["target"] == {
        "atom_property": "MBIScharge",
        "molecule_property": "net_charge",
        "label": None,
    }
    flat = to_flat_params(cfg)
    assert flat["target.atom_property"] == "MBIScharge"
    assert flat["target.molecule_property"] == "net_charge"
```

Make sure the file imports what these use (`pytest`, `_build`, `to_dict`, `to_flat_params` from `experiments.config`).

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest experiments/tests/test_config.py -q
```

Expected: failures — `unknown key(s) in config: ['target']`, and `AttributeError: 'ExperimentCfg' object has no attribute 'target'`.

- [ ] **Step 3: Implement `TargetCfg`**

In `experiments/experiments/config.py`:

```python
_TARGET_KEYS = {"atom_property", "molecule_property", "label"}
_TOP_KEYS = {
    "run", "data", "target", "predictor",
    "normalization", "tree_stats_load_path", "save_tree_stats",
}


@dataclass(frozen=True)
class TargetCfg:
    """What this run predicts. ``atom_property`` is the name of the per-atom
    property carried on each stored ``Mol`` (the DASH series' own
    ``MBIScharge``, some other dataset's own scalar) -- it is read as the
    training target and predicted per atom.

    ``molecule_property`` names a per-molecule column of the store's parquet
    holding the value each molecule's atom values should sum to (the DASH
    series' ``net_charge``). Optional: when omitted the run has no sum
    constraint, so no ``sum_constraint/*`` metrics, no residual panel, and
    no ``normalization``. ``label`` is plot-axis text only, defaulting to
    ``atom_property``."""

    atom_property: str
    molecule_property: str | None = None
    label: str | None = None

    @property
    def axis_label(self) -> str:
        return self.label if self.label is not None else self.atom_property
```

Add `target: TargetCfg` to `ExperimentCfg` immediately after `data`, and in `_build`:

```python
    for section in ("run", "data", "target", "predictor"):
        if section not in raw:
            raise ValueError(f"config is missing required section {section!r}")

    target_raw = raw["target"]
    _check_keys(target_raw, _TARGET_KEYS, "target")
    if "atom_property" not in target_raw:
        raise ValueError("target.atom_property is required")
    target = TargetCfg(
        atom_property=target_raw["atom_property"],
        molecule_property=target_raw.get("molecule_property"),
        label=target_raw.get("label"),
    )
```

and, after the existing `normalization` membership check:

```python
    if normalization is not None and target.molecule_property is None:
        raise ValueError(
            f"normalization={normalization!r} needs a per-molecule total to "
            "redistribute against; set target.molecule_property"
        )
```

Extend `to_dict` with

```python
        "target": {
            "atom_property": cfg.target.atom_property,
            "molecule_property": cfg.target.molecule_property,
            "label": cfg.target.label,
        },
```

and `to_flat_params` with

```python
    _flatten(
        "target",
        {
            "atom_property": cfg.target.atom_property,
            "molecule_property": cfg.target.molecule_property,
            "label": cfg.target.label,
        },
        out,
    )
```

- [ ] **Step 4: Run the config tests**

```bash
uv run pytest experiments/tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Add `target:` to every shipped config**

Each file in `experiments/configs/` gets, between `data:` and `predictor:`:

```yaml
target:
  atom_property: MBIScharge
  molecule_property: net_charge
  label: charge (e)
```

```bash
ls experiments/configs/*.yaml | while read f; do grep -q '^target:' "$f" || echo "MISSING: $f"; done
```

Expected: no output.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -q
```

Other tests build `ExperimentCfg` directly or from raw dicts; add `target=TargetCfg(atom_property="MBIScharge", molecule_property="net_charge")` (or the `target:` block) wherever construction now fails. Do not weaken an assertion to compensate.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(experiments): config-named target atom property

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 3: Generalize `MoleculeSet`

**Files:**
- Modify: `experiments/experiments/data.py`
- Modify: `experiments/experiments/runner.py` (`load_molecule_set`, `_savez_run`, and every `test.atom_charge`/`test.net_charge` reference)
- Modify: `experiments/experiments/predictors/{global_mean,dash,dash_pretrained,sieve_predictor}.py` (target reads only)
- Modify: `experiments/tests/helpers.py`
- Test: `experiments/tests/test_data.py`

**Interfaces:**
- Consumes: `TargetCfg` (Task 2).
- Produces:
  ```python
  @dataclass(frozen=True)
  class MoleculeSet:
      mols: list[Any]
      atom_property: str
      molecule_property: str | None = None
      molecule_value: NDArray[np.float64] | None = None
      ids: Mapping[str, list[str | None]] = <empty dict>
      split: list[str] | None = None
      # properties: n_conformers, num_atoms, n_atoms, atom_mol_id,
      #             atom_target -> NDArray[np.float64]
      # select(mol_mask) -> MoleculeSet
  # helpers.synthetic_molecule_set(n_mol=8, seed=0, atom_property="MBIScharge")
  ```

- [ ] **Step 1: Write the failing tests**

Add to `experiments/tests/test_data.py` (whose header already has `numpy as np`
and `pytest`; add `from experiments.data import MoleculeSet` and
`from experiments.tests.helpers import synthetic_molecule_set`):

```python
def test_atom_target_reads_the_configured_property():
    from rdkit import Chem
    from experiments.data import MoleculeSet

    mol = Chem.MolFromSmiles("CO")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp("alpha", float(i))
    mset = MoleculeSet(mols=[mol], atom_property="alpha")
    assert mset.atom_target.tolist() == [0.0, 1.0]


def test_missing_atom_property_raises_naming_it():
    from rdkit import Chem
    from experiments.data import MoleculeSet

    mset = MoleculeSet(mols=[Chem.MolFromSmiles("CO")], atom_property="alpha")
    with pytest.raises(KeyError, match="alpha"):
        _ = mset.atom_target


def test_molecule_value_is_none_when_unconfigured():
    mset = synthetic_molecule_set(n_mol=4)
    bare = MoleculeSet(mols=mset.mols, atom_property=mset.atom_property)
    assert bare.molecule_value is None
    assert bare.molecule_property is None


def test_ids_are_carried_and_subset_by_select():
    mset = synthetic_molecule_set(n_mol=4)
    assert set(mset.ids) == {"chembl_id", "conf_id", "dash_id"}
    sub = mset.select(np.array([True, False, True, False]))
    assert sub.n_conformers == 2
    assert sub.ids["conf_id"] == [mset.ids["conf_id"][0], mset.ids["conf_id"][2]]
    assert sub.atom_property == mset.atom_property
    assert sub.molecule_property == mset.molecule_property


def test_select_keeps_molecule_value_none_when_unset():
    mset = synthetic_molecule_set(n_mol=4)
    bare = MoleculeSet(mols=mset.mols, atom_property=mset.atom_property)
    assert bare.select(np.array([True, False, True, False])).molecule_value is None


def test_ids_length_mismatch_raises():
    mset = synthetic_molecule_set(n_mol=4)
    with pytest.raises(ValueError, match="conf_id"):
        MoleculeSet(
            mols=mset.mols,
            atom_property=mset.atom_property,
            ids={"conf_id": ["a", "b"]},
        )
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest experiments/tests/test_data.py -q
```

Expected: `TypeError` on the `MoleculeSet(...)` constructor calls (`chembl_id` still required, `atom_property` unknown).

- [ ] **Step 3: Rewrite `MoleculeSet`**

In `experiments/experiments/data.py`:

```python
@dataclass(frozen=True)
class MoleculeSet:
    """One split's worth of conformers. Each entry in ``mols`` is one
    conformer's own RDKit ``Mol``, its atoms carrying ``atom_property`` as a
    real double property -- there is no separate, position-aligned target
    array to keep in sync.

    ``molecule_property``/``molecule_value`` are the optional per-molecule
    total the atom values should sum to (the DASH series' ``net_charge``);
    both are ``None`` for a dataset with no such constraint.

    ``ids`` carries every remaining store column as per-conformer
    provenance -- the DASH stores' ``chembl_id``/``conf_id``/``dash_id``,
    some other dataset's own keys. Nothing in the harness groups or
    clusters by them; they are written straight through to
    ``predictions.npz``.
    """

    mols: list[Any]
    atom_property: str
    molecule_property: str | None = None
    molecule_value: NDArray[np.float64] | None = None
    ids: Mapping[str, list[str | None]] = field(default_factory=dict)
    split: list[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.mols)
        if self.molecule_value is not None and len(self.molecule_value) != n:
            raise ValueError("molecule_value must have one entry per conformer")
        if (self.molecule_property is None) != (self.molecule_value is None):
            raise ValueError(
                "molecule_property and molecule_value must be set together"
            )
        for key, values in self.ids.items():
            if len(values) != n:
                raise ValueError(f"ids[{key!r}] must have one entry per conformer")
        if self.split is not None and len(self.split) != n:
            raise ValueError("split must have one entry per conformer")

    @property
    def atom_target(self) -> NDArray[np.float64]:
        """Per-atom ground truth for ``atom_property``, flattened across
        every conformer's own atom order."""
        if not self.mols:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(
            [self._mol_target(m) for m in self.mols]
        )

    def _mol_target(self, mol: Any) -> NDArray[np.float64]:
        out = np.empty(mol.GetNumAtoms(), dtype=np.float64)
        for i, atom in enumerate(mol.GetAtoms()):
            if not atom.HasProp(self.atom_property):
                raise KeyError(
                    f"atom {i} of a stored conformer has no property "
                    f"{self.atom_property!r} -- the store was prepared for a "
                    "different target"
                )
            out[i] = atom.GetDoubleProp(self.atom_property)
        return out

    def select(self, mol_mask: NDArray[np.bool_]) -> MoleculeSet:
        """The sub-set of conformers where ``mol_mask`` is True. The only
        place a split mask is applied."""
        mol_mask = np.asarray(mol_mask, dtype=bool)
        idx = np.flatnonzero(mol_mask)
        return MoleculeSet(
            mols=[self.mols[i] for i in idx],
            atom_property=self.atom_property,
            molecule_property=self.molecule_property,
            molecule_value=(
                None
                if self.molecule_value is None
                else np.asarray(self.molecule_value)[mol_mask]
            ),
            ids={k: [v[i] for i in idx] for k, v in self.ids.items()},
            split=None if self.split is None else [self.split[i] for i in idx],
        )
```

`n_conformers`, `num_atoms`, `n_atoms`, `atom_mol_id` keep their current bodies. Add `from collections.abc import Mapping` and `from dataclasses import dataclass, field` to the imports.

- [ ] **Step 4: Update the loader and npz writer**

In `experiments/experiments/runner.py`:

```python
def load_molecule_set(
    store_name: str,
    *,
    target: TargetCfg,
    split_column: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    stores_root: Path | None = None,
) -> tuple[MoleculeSet, dict[str, NDArray[np.bool_]]]:
    """Load ``molecules.parquet`` for ``store_name`` into a ``MoleculeSet``
    plus split masks. Every column that isn't ``mol``, the split column or
    ``target.molecule_property`` is carried through as an identifier."""
    import pandas as pd

    from experiments.data import DEFAULT_STORES_ROOT, blob_to_mol

    root = stores_root if stores_root is not None else DEFAULT_STORES_ROOT
    df = pd.read_parquet(root / store_name / "molecules.parquet")
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    mol_prop = target.molecule_property
    if mol_prop is not None and mol_prop not in df.columns:
        raise ValueError(
            f"store {store_name!r} has no column {mol_prop!r} "
            f"(target.molecule_property); columns: {sorted(df.columns)}"
        )
    id_columns = [c for c in df.columns if c not in ("mol", split_column, mol_prop)]

    mset = MoleculeSet(
        mols=[blob_to_mol(b) for b in df["mol"]],
        atom_property=target.atom_property,
        molecule_property=mol_prop,
        molecule_value=(
            None if mol_prop is None else df[mol_prop].to_numpy(dtype=np.float64)
        ),
        ids={c: list(df[c]) for c in id_columns},
        split=list(df[split_column]),
    )
    masks = {name: (df[split_column] == name).to_numpy() for name in splits}
    return mset, masks
```

and its caller in `run()` gains `target=cfg.target`.

```python
def _savez_run(path: Path, test: MoleculeSet, pred: Prediction, /) -> None:
    arrays: dict[str, Any] = {k: np.array(v) for k, v in test.ids.items()}
    arrays["num_atoms"] = test.num_atoms
    arrays["atom_target_true"] = test.atom_target
    arrays["atom_target_pred"] = pred.atom_charge  # renamed in Task 4
    if test.molecule_value is not None:
        arrays["molecule_value"] = test.molecule_value
    np.savez(path, **arrays)
```

- [ ] **Step 5: Update every remaining reader**

Replace `.atom_charge` on a `MoleculeSet` with `.atom_target`, and `.net_charge` with `.molecule_value`, in:
`runner.py` (`_score`, `_normalize`, `_build_parity_panels`), `predictors/global_mean.py` (`fit`), `predictors/dash.py` (`fit`), `predictors/dash_pretrained.py` (`predict`), and the tests. Leave `Prediction.atom_charge` alone — Task 4 renames it.

```bash
grep -rn '\.atom_charge\|\.net_charge' experiments/ --include='*.py' | grep -v 'pred\.\|raw\.\|Prediction('
```

Expected after the edits: no hits.

- [ ] **Step 6: Update the test fixture**

In `experiments/tests/helpers.py`:

```python
def synthetic_molecule_set(n_mol: int = 8, seed: int = 0, atom_property: str = "MBIScharge"):
    """A small, fully-populated ``MoleculeSet`` for fast harness tests --
    real RDKit ``Mol`` objects (small alkanes/alcohols), each atom carrying a
    fabricated but deterministic value of ``atom_property``, with
    ``molecule_value`` computed to be exactly its per-molecule sum, so both
    the input and any rollup can be checked exactly."""
```

with the body's `"MBIScharge"` literals replaced by `atom_property`, and the return:

```python
    return MoleculeSet(
        mols=mols,
        atom_property=atom_property,
        molecule_property="net_charge",
        molecule_value=net_charge,
        ids={
            "chembl_id": chembl_id,
            "conf_id": conf_id,
            "dash_id": [None] * n_mol,
        },
    )
```

- [ ] **Step 7: Run the full suite**

```bash
uv run pytest -q
```

Expected: PASS. Tests that assert on `predictions.npz` keys need the new names (`atom_target_true`, `atom_target_pred`, `molecule_value`); that is the intended change, not a test to weaken.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(experiments): MoleculeSet carries its own property names

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 4: `Prediction.atom_value`, and predictors read the configured property

**Files:**
- Modify: `experiments/experiments/predictors/base.py`
- Modify: `experiments/experiments/predictors/{global_mean,sieve_predictor,dash,dash_pretrained}.py`
- Modify: `experiments/experiments/runner.py`
- Test: `experiments/tests/test_predictor_base.py`, `experiments/tests/test_predictor_sieve.py`

**Interfaces:**
- Consumes: `MoleculeSet.atom_property`, `MoleculeSet.atom_target` (Task 3).
- Produces:
  ```python
  @dataclass(frozen=True)
  class Prediction:      atom_value: NDArray[np.float64]
  @dataclass(frozen=True)
  class RawPrediction:   atom_value: NDArray[np.float64]; atom_std: NDArray[np.float64]
  # sieve_predictor._batch_for(mols, config, *, with_target, atom_property, n_jobs=None)
  ```

- [ ] **Step 1: Write the failing tests**

In `experiments/tests/test_predictor_base.py`:

```python
def test_prediction_field_is_atom_value():
    from experiments.predictors.base import Prediction

    pred = Prediction(atom_value=np.zeros(3))
    assert pred.atom_value.shape == (3,)
```

In `experiments/tests/test_predictor_sieve.py`:

```python
def test_sieve_fits_a_non_charge_property():
    from experiments.predictors.sieve_predictor import SievePredictor

    mset = synthetic_molecule_set(n_mol=8, atom_property="alpha")
    predictor = SievePredictor(max_wl_depth=1, minimum_support=1)
    predictor.fit(mset, mset, rng=np.random.default_rng(0))
    pred = predictor.predict(mset)
    assert pred.atom_value.shape == (mset.n_atoms,)
    assert np.isfinite(pred.atom_value).all()
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest experiments/tests/test_predictor_base.py experiments/tests/test_predictor_sieve.py -q
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'atom_value'`, and the sieve test failing because `_batch_for` asks RDKit for `MBIScharge` on atoms that only carry `alpha`.

- [ ] **Step 3: Rename the field**

```bash
cd /data3/craabreu/github_repos/sieve
grep -rl 'atom_charge' experiments/ --include='*.py' \
  | xargs sed -i 's/\batom_charge\b/atom_value/g'
grep -rn 'atom_charge' experiments/ --include='*.py'   # expect none
```

Then fix the docstrings this touched by hand: `predictors/base.py`'s module and class docstrings should say "one scalar per atom (the run's own `target.atom_property`)" rather than naming `MBIScharge`, and `Prediction`'s docstring should read:

```python
@dataclass(frozen=True)
class Prediction:
    """What every predictor returns: one predicted value of the run's own
    ``target.atom_property`` per atom, in ``test``'s own atom order
    (``test.atom_mol_id``-aligned)."""

    atom_value: NDArray[np.float64]
```

`NormalizableChargePredictor` becomes `NormalizablePredictor` (same `sed` idiom); update its references in `runner.py` and the tests.

- [ ] **Step 4: Thread the property into the sieve adapter call**

In `experiments/experiments/predictors/sieve_predictor.py`:

```python
def _batch_for(
    mols: list[Any],
    config: Any,
    *,
    with_target: bool,
    atom_property: str,
    n_jobs: int | None = None,
) -> Any:
    """Build a ``NodeBatch`` for ``mols`` under an already-fitted
    ``config``. ``node_order`` is left ``None``: each ``Mol``'s own atom
    order is already this series' canonical order."""
    from sieve.io.rdkit_adapter import from_rdkit

    return from_rdkit(
        mols,
        config=config,
        y_from_atom_prop=atom_property if with_target else None,
        n_jobs=n_jobs,
    )
```

Its three call sites pass the set's own name: `fit` → `atom_property=train.atom_property`, `predict_raw` → `atom_property=test.atom_property`, `predict_loo_raw` → `atom_property=train.atom_property`. Also update `predict_loo_raw`'s docstring sentence "Every MoleculeSet in this series carries MBIScharge on its Mols" → "Every MoleculeSet carries its own ``atom_property`` on its Mols".

`DEFAULT_ATTRIBUTES` is untouched — attribute levels are model inputs, not the target (spec decision 2).

- [ ] **Step 5: Run the tests**

```bash
uv run pytest experiments/tests/test_predictor_base.py experiments/tests/test_predictor_sieve.py -q
```

Expected: PASS.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(experiments): Prediction.atom_value; sieve reads the configured property

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 5: `sum_constraint` metrics, gated on `molecule_property`

**Files:**
- Modify: `experiments/experiments/metrics.py:61-75`
- Modify: `experiments/experiments/runner.py` (`_score`)
- Modify: `experiments/experiments/cli.py:29-31` (`SUMMARY_COLUMNS`)
- Test: `experiments/tests/test_metrics.py`, `experiments/tests/test_smoke.py`

**Interfaces:**
- Consumes: `MoleculeSet.molecule_value` (Task 3), `Prediction.atom_value` (Task 4).
- Produces: `metrics.sum_constraint_metrics(atom_value_pred, mol_id, molecule_value_true, n_molecules) -> dict[str, float]`; run metrics keyed `sum_constraint/{mae,rmse,r2,n_nan}`.

- [ ] **Step 1: Write the failing tests**

In `experiments/tests/test_metrics.py`, adapt the existing `charge_conservation_metrics` test to the new name and add:

```python
def test_sum_constraint_metrics_against_hand_computed_numbers():
    from experiments.metrics import sum_constraint_metrics

    # two molecules, 2 atoms each; predicted sums 1.0 and 4.0 vs true 1.5, 3.0
    pred = np.array([0.4, 0.6, 2.0, 2.0])
    mol_id = np.array([0, 0, 1, 1])
    true = np.array([1.5, 3.0])
    out = sum_constraint_metrics(pred, mol_id, true, 2)
    assert out["mae"] == pytest.approx(0.75)
    assert out["rmse"] == pytest.approx(np.sqrt((0.5**2 + 1.0**2) / 2))
```

In `experiments/tests/test_smoke.py`, rename the existing
`test_smoke_metrics_include_r2_and_charge_conservation` to
`..._and_sum_constraint` and change its last assertion to
`assert "sum_constraint/mae" in result.metrics`. Then add:

```python
def test_no_sum_constraint_metrics_without_a_molecule_property(tmp_path):
    from dataclasses import replace

    from experiments.config import TargetCfg
    from experiments.data import MoleculeSet

    full = synthetic_molecule_set(n_mol=20, seed=0)
    mset = MoleculeSet(
        mols=full.mols, atom_property=full.atom_property, ids=full.ids
    )
    masks = _synthetic_masks(20, seed=1)
    cfg = replace(_tiny_cfg(), target=TargetCfg(atom_property="MBIScharge"))

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert np.isfinite(result.metrics["mae"])
    assert not any(k.startswith("sum_constraint/") for k in result.metrics)
```

`_tiny_cfg()` (Task 2 already gave it a `target=TargetCfg(atom_property="MBIScharge", molecule_property="net_charge")`) is reused via `dataclasses.replace` so the two cases differ in exactly one field.

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest experiments/tests/test_metrics.py experiments/tests/test_smoke.py -q
```

Expected: `ImportError: cannot import name 'sum_constraint_metrics'`.

- [ ] **Step 3: Rename the metric function**

In `experiments/experiments/metrics.py`:

```python
def sum_constraint_metrics(
    atom_value_pred: NDArray[np.floating],
    mol_id: NDArray[np.int64],
    molecule_value_true: NDArray[np.floating],
    n_molecules: int,
) -> dict[str, float]:
    """Secondary diagnostic, for a dataset whose per-atom values are
    expected to sum to a known per-molecule total (``target.
    molecule_property``): how well each molecule's summed predictions
    reproduce it. For the DASH charge series that total is the conformer's
    own molblock ``M CHG`` sum, and there is no sign flip -- ``MBIScharge``
    is a real atomic partial charge, and a conformer's atoms should sum to
    its formal charge directly.
    """
    pred_sum = molecule_sum(atom_value_pred, mol_id, n_molecules)
    return regression_metrics(molecule_value_true, pred_sum)
```

Update the module docstring's `charge_metrics` comparison sentence to name `sum_constraint_metrics`.

- [ ] **Step 4: Gate scoring on the constraint**

In `experiments/experiments/runner.py`:

```python
def _score(test: MoleculeSet, pred: Prediction) -> dict[str, float]:
    out = metrics_mod.regression_metrics(test.atom_target, pred.atom_value)
    out["n_test_atoms"] = float(test.n_atoms)
    out["n_test_conformers"] = float(test.n_conformers)
    if test.molecule_value is not None:
        constraint = metrics_mod.sum_constraint_metrics(
            pred.atom_value, test.atom_mol_id, test.molecule_value, test.n_conformers
        )
        out.update({f"sum_constraint/{k}": v for k, v in constraint.items()})
    return out
```

- [ ] **Step 5: Update the summary columns**

In `experiments/experiments/cli.py`, `SUMMARY_COLUMNS` entries become
`"sum_constraint/mae"`, `"sum_constraint/rmse"`, `"sum_constraint/r2"`.

- [ ] **Step 6: Run the tests**

```bash
uv run pytest experiments/tests/test_metrics.py experiments/tests/test_smoke.py -q
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(experiments): sum_constraint metrics, optional per-molecule total

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 6: Generalize the normalizers' vocabulary

Behavior-preserving: identical arithmetic, dataset-neutral names. The existing `test_normalize.py` numbers are the proof.

**Files:**
- Modify: `experiments/experiments/normalize.py`
- Modify: `experiments/experiments/runner.py` (`_normalize`), `experiments/experiments/predictors/dash_pretrained.py` (call site)
- Test: `experiments/tests/test_normalize.py`

**Interfaces:**
- Produces: `NORMALIZERS[name](raw_value, raw_std, molecule_value, mol_id, n_conformers) -> NDArray[np.float64]`, keys `std_weighted`, `equal_weighted` (unchanged).

- [ ] **Step 1: Add a test that the schemes are property-agnostic**

In `experiments/tests/test_normalize.py`:

```python
def test_equal_weighted_hits_the_requested_total_for_any_quantity():
    from experiments.normalize import equal_weighted_normalize

    raw = np.array([1.0, 2.0, 3.0])          # sums to 6.0
    mol_id = np.array([0, 0, 0])
    total = np.array([9.0])                   # nothing charge-like about it
    out = equal_weighted_normalize(raw, np.ones(3), total, mol_id, 1)
    assert out.sum() == pytest.approx(9.0)
```

- [ ] **Step 2: Run it**

```bash
uv run pytest experiments/tests/test_normalize.py -q
```

Expected: PASS already — the arithmetic never cared. This test is the guard for the rename that follows.

- [ ] **Step 3: Rename the parameters and rewrite the prose**

```bash
cd /data3/craabreu/github_repos/sieve
sed -i 's/\braw_charge\b/raw_value/g; s/\bnet_charge\b/molecule_value/g; \
        s/\btot_charge_tree\b/tot_value_tree/g' experiments/experiments/normalize.py
```

Then rewrite the module docstring by hand:

```python
"""Sum-constraint schemes: redistribute a predictor's raw, unnormalized
per-atom values so each molecule's atoms sum to a known per-molecule total
(``config.TargetCfg.molecule_property``). Kept independent of any predictor
so a run can apply one to an already-computed raw prediction without
re-matching or re-fitting anything (see ``config.ExperimentCfg.
normalization`` and ``runner._predict``).

Both schemes come from DASH's own post-hoc charge conservation (the paper's
eq. 4 and ``get_molecules_partial_charges``'s ``symmetric`` branch), but
neither is charge-specific: they need only a per-atom value, a per-atom
std, and a per-molecule total.

Every entry in ``NORMALIZERS`` shares one signature, ``(raw_value, raw_std,
molecule_value, mol_id, n_conformers) -> atom_value``, even though
``equal_weighted_normalize`` ignores ``raw_std`` entirely -- this lets
calling code stay normalization-agnostic.
"""
```

Keep every provenance comment (the `_DEFAULT_STD_VALUE = 0.1` note, the sign-discrepancy finding in `dash_pretrained.py`) verbatim — those are findings about a real upstream implementation, not charge vocabulary.

Update `runner._normalize`:

```python
def _normalize(raw: Any, mset: MoleculeSet, *, normalization: str) -> Prediction:
    """Apply ``normalize.NORMALIZERS[normalization]`` to a predictor's raw
    walk output (``predictors.base.RawPrediction``, from ``predict_raw``/
    ``predict_loo_raw``). Requires ``mset.molecule_value``; ``config``
    already refuses a ``normalization`` without ``target.molecule_property``,
    so this is an invariant, not a user-facing error."""
    assert mset.molecule_value is not None
    atom_value = NORMALIZERS[normalization](
        raw.atom_value,
        raw.atom_std,
        mset.molecule_value,
        mset.atom_mol_id,
        mset.n_conformers,
    )
    return Prediction(atom_value=atom_value)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest experiments/tests/test_normalize.py -q && uv run pytest -q
```

Expected: PASS, with the pre-existing numeric assertions unchanged — that is the proof the arithmetic survived.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(experiments): normalizers are dataset-agnostic sum constraints

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 7: Parity panels use the configured label, and drop the residual panel when unconstrained

**Files:**
- Modify: `experiments/experiments/runner.py:293-347` (`_build_parity_panels`) and `_write_plots`
- Test: `experiments/tests/test_plots.py`

**Interfaces:**
- Consumes: `TargetCfg.axis_label` (Task 2), `MoleculeSet.molecule_value` (Task 3).
- Produces: `_build_parity_panels(test, pred, run_metrics, *, label: str) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write the failing tests**

In `experiments/tests/test_plots.py`:

```python
def test_panels_use_the_configured_label():
    mset = synthetic_molecule_set(n_mol=6)
    pred = Prediction(atom_value=mset.atom_target + 0.01)
    panels = _build_parity_panels(mset, pred, {"mae": 0.01}, label="alpha (a.u.)")
    assert panels[0]["quantity"] == "alpha (a.u.)"
    assert panels[0]["title"] == "atom alpha (a.u.)"


def test_no_residual_panel_without_a_molecule_total():
    mset = synthetic_molecule_set(n_mol=6)
    bare = MoleculeSet(mols=mset.mols, atom_property=mset.atom_property)
    pred = Prediction(atom_value=bare.atom_target + 0.5)
    panels = _build_parity_panels(bare, pred, {"mae": 0.5}, label="charge (e)")
    assert len(panels) == 1
    assert all(p.get("kind") != "histogram" for p in panels)
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest experiments/tests/test_plots.py -q
```

Expected: `TypeError: _build_parity_panels() got an unexpected keyword argument 'label'`.

- [ ] **Step 3: Implement**

In `experiments/experiments/runner.py`:

```python
def _build_parity_panels(
    test: MoleculeSet,
    pred: Prediction,
    run_metrics: dict[str, float],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Which panels a run's ``parity_panel.png`` gets: the per-atom target
    always (a hexbin parity plot -- ``test`` is assumed non-empty,
    ``_write_plots`` checks that before calling this); the sum-constraint
    residual as a secondary diagnostic, when the run has a per-molecule
    total to compare against -- a 1-D histogram of the per-conformer
    residual (predicted atom values summed, minus the molecule's own
    ``molecule_value``), not a parity scatter, since a residual is one
    number per conformer, not a true/predicted pair. The same residual
    ``metrics.sum_constraint_metrics`` already scores (its own ``err =
    y_pred - y_true`` is this exact quantity). Pure numpy -- no matplotlib
    -- so this is testable independent of ``plots.py`` actually rendering
    anything."""
    panels: list[dict[str, Any]] = []

    atom_true, atom_pred = _finite_pair(test.atom_target, pred.atom_value)
    if atom_true.size:  # every atom NaN (e.g. a pretrained baseline that
        # matched nothing in this split) -- an empty hexbin panel would
        # crash on its own .min()/.max() axis limits, so skip it, not fake it
        panels.append(
            {
                "y_true": atom_true,
                "y_pred": atom_pred,
                "quantity": label,
                "title": f"atom {label}",
                "metrics": {
                    k: v for k, v in run_metrics.items() if k in ("mae", "rmse", "r2")
                },
            }
        )

    if test.molecule_value is None:
        return panels

    pred_total = molecule_sum(pred.atom_value, test.atom_mol_id, test.n_conformers)
    residual = pred_total - test.molecule_value
    residual = residual[~np.isnan(residual)]
    # A predictor whose own normalization already satisfies the constraint
    # exactly (e.g. std_weighted/equal_weighted -- residuals at float
    # round-off, ~1e-16) has nothing worth plotting here: a histogram of
    # that is a single spike carrying no information, not a diagnostic.
    # Same threshold plots._histogram_subplot's own degenerate-bin-count
    # guard uses, kept in sync deliberately.
    if residual.size and np.max(np.abs(residual)) >= 1e-6:
        panels.append(
            {
                "kind": "histogram",
                "values": residual,
                "xlabel": f"molecule {label} residual",
                "title": "sum constraint",
                "metrics": {
                    k.removeprefix("sum_constraint/"): v
                    for k, v in run_metrics.items()
                    if k.startswith("sum_constraint/")
                },
            }
        )
    return panels
```

`_write_plots` already takes `cfg`; it passes `label=cfg.target.axis_label`.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest experiments/tests/test_plots.py -q && uv run pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(experiments): parity panels follow target.label and the sum constraint

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 8: Split `prepare_store.py` into `prepare_dash.py` and `store_ops.py`

Pure code motion — no logic changes. This is what makes "the prep script is the only dataset-specific piece" literally true.

**Files:**
- Create: `experiments/experiments/prepare_dash.py` (from `prepare_store.py`: `DOWNLOAD_URL`, `EXPECTED_BYTES`, `SDF_FILENAME`, `_USER_AGENT`, `CHUNK_SIZE`, `PARQUET_BATCH_SIZE`, `download_dash_sdf`, `_assign_stereo_if_needed`, `_parse_one_record`, `parse_dash_molecules`, `_achiral_fingerprints`, `assign_splits`, `prepare_store`)
- Create: `experiments/experiments/store_ops.py` (from `prepare_store.py`: `subsample_store`, `partition_store`, `_write_subsample`, `_to_united_atom`, `to_united_atom_store`)
- Delete: `experiments/experiments/prepare_store.py`
- Modify: `experiments/experiments/cli.py:86-140`
- Modify: `experiments/tests/test_prepare_store.py` → split into `test_prepare_dash.py` and `test_store_ops.py`; `test_prepare_store_optional.py` → `test_prepare_dash_optional.py`

- [ ] **Step 1: Move the DASH-only half**

```bash
cd /data3/craabreu/github_repos/sieve
git mv experiments/experiments/prepare_store.py experiments/experiments/prepare_dash.py
```

Then cut the four store-operation functions plus `_write_subsample` and `_to_united_atom` out of it into a new `experiments/experiments/store_ops.py`, carrying their imports. Give the new file this docstring:

```python
"""Dataset-agnostic operations on an already-prepared store: draw smaller
representative stores from a big one (``subsample_store``), divide one into
disjoint folds (``partition_store``), or emit a heavy-atom-only version
(``to_united_atom_store``). Every one of these acts on the store parquet's
own columns, so any dataset whose prep script emits that format inherits
them -- see experiments/README.md's "adding a dataset".
"""
```

and narrow `prepare_dash.py`'s docstring to the DASH pipeline alone (download → parse → cluster → split → `prepare_store`).

- [ ] **Step 2: Update the CLI imports**

In `experiments/experiments/cli.py`:

```python
def _cmd_prepare_store(args: argparse.Namespace) -> int:
    from experiments.prepare_dash import prepare_store
    ...

def _cmd_subsample_store(args: argparse.Namespace) -> int:
    from experiments.store_ops import subsample_store
    ...

def _cmd_partition_store(args: argparse.Namespace) -> int:
    from experiments.store_ops import partition_store
    ...

def _cmd_to_united_atom(args: argparse.Namespace) -> int:
    from experiments.store_ops import to_united_atom_store
    ...
```

- [ ] **Step 3: Split the tests**

```bash
cd /data3/craabreu/github_repos/sieve/experiments/tests
git mv test_prepare_store.py test_prepare_dash.py
git mv test_prepare_store_optional.py test_prepare_dash_optional.py
```

Move every test exercising `subsample_store`, `partition_store` or
`to_united_atom_store` out of `test_prepare_dash.py` into a new
`test_store_ops.py`, updating imports to `experiments.store_ops`. No
assertion changes.

- [ ] **Step 4: Verify no stale references**

```bash
cd /data3/craabreu/github_repos/sieve
grep -rn 'prepare_store' experiments/ --include='*.py' | grep -v 'def prepare_store\|prepare_dash import prepare_store\|_cmd_prepare_store'
```

Expected: no hits (the CLI subcommand name `prepare-store` stays as it is — it is a user-facing name, not an import path).

- [ ] **Step 5: Run the tests**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 6: Verify the CLI still resolves both halves**

```bash
uv run python -m experiments --help
uv run python -m experiments subsample-store --help
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(experiments): split prepare_store into prepare_dash and store_ops

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

### Task 9: Documentation

**Files:**
- Modify: `experiments/README.md`
- Modify: `experiments/experiments/__init__.py` (module docstring)

- [ ] **Step 1: Reframe `experiments/README.md`**

Open with the harness/dataset split rather than DASH:

```markdown
# experiments

A node-level regression harness for molecular stores: a run names an atomic
property, fits a predictor on the train split, and scores per-atom
predictions against it. Nothing in the harness is specific to one dataset --
a dataset arrives as a data-preparation script that writes the store format
below, plus a config naming its property.

DASH's MBIS atomic partial charges (`MBIScharge`, from
`dashMoleculesSDF_v2.sdf`) are the series this harness was built for, and
`prepare_dash.py` is its prep script; `docs/dash_molecules_sdf.md` records
what running it against the real published SDF turned up.
```

Then add, before the existing "Usage" section:

```markdown
## Adding a dataset

1. Write `experiments/<dataset>_prep.py` producing
   `stores/<name>/molecules.parquet` with:
   - `mol` -- `data.mol_to_blob(mol)` bytes, one conformer per row, its
     atoms carrying your target property (`atom.SetDoubleProp`);
   - `split` -- `"train"` / `"val"` / `"test"` per row;
   - optionally a per-molecule column your atoms' values should sum to;
   - any further columns, carried through as per-conformer identifiers.
2. Name it in a config:
   ```yaml
   target:
     atom_property: my_property
     molecule_property: my_total   # omit if there is no sum constraint
     label: my property (units)    # optional, plot axes
   ```
3. Everything else already works: `store_ops.py`'s subsample / partition /
   united-atom commands, every predictor, the metrics, the plots.

Omitting `molecule_property` disables the `sum_constraint/*` metrics, the
residual panel, and the `normalization` key -- a config that sets
`normalization` without it is rejected at load.
```

Update every `python -m charge_experiments` in the file to `python -m experiments`, and the `charge_experiments/runs/...` example path to `experiments/runs/...`.

- [ ] **Step 2: Update the package docstring**

```python
# experiments/experiments/__init__.py
"""experiments: a node-level regression harness for molecular stores. A run
names the atomic property it trains and predicts (``config.TargetCfg``);
the data-preparation script is the only dataset-specific component. Built
for, and still principally used for, DASH atomic partial charges
(``MBIScharge``) -- see
docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md
and docs/superpowers/specs/2026-09-09-experiments-generalization-design.md.
"""
```

- [ ] **Step 3: Verify the documented commands actually run**

```bash
cd /data3/craabreu/github_repos/sieve
uv run python -m experiments --help
uv run python -m experiments run \
  --config experiments/configs/sieve-charge-example.yaml \
  --set data.store=dash-molecules-10fold-1 --limit 500
```

Expected: a run directory with `metrics.json` containing `mae`, `rmse`, `r2`
and `sum_constraint/mae`; `config.resolved.yaml` containing the `target:`
block; `parity_panel.png` with two panels.

- [ ] **Step 4: Full suite and lint, one last time**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(experiments): reframe around the harness/dataset split

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q3YSwLeZ5k6oqRcTmW4Wzt"
```

---

## Verification checklist

After Task 9, confirm against the spec:

- [ ] `experiments/` holds all tracked code; `charge_experiments/` holds only untracked data; the seven symlinks resolve.
- [ ] `uv run python -m experiments run --config …` reproduces a known earlier run's `mae`/`rmse`/`r2` to the last digit (only the key `charge_conservation/*` → `sum_constraint/*` differs).
- [ ] A config with no `target.molecule_property` runs, scores, and plots — one panel, no `sum_constraint/*` keys.
- [ ] A config with `normalization` and no `target.molecule_property` is rejected at load.
- [ ] `grep -rn 'MBIScharge' experiments/experiments/ --include='*.py'` hits only DASH-specific modules (`prepare_dash.py`, `predictors/dash*.py`, `tree_artifact.py`) and docstrings that name it as an example.
