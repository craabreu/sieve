# Baseline experiment harness — design

## Context

The end goal is to predict **molecule sigma profiles for molecules larger than
those seen in training**. Per-atom prediction from the local atom environment is
the route to that goal, and Sieve is the model being developed for it.

Today the repo cannot support that claim reproducibly. `scripts/train_chaos_*.py`
are argparse CLIs that print results to the screen and write PNG files next to
themselves, which then get committed. There is no metrics file, no saved model,
no run record, no seed control, and no lint or type coverage on those files.
`cosmolayer` — needed by every experiment — is not a declared dependency, and
`uv.lock` was git-ignored while CI used pip. A result could not be reproduced
from a clean checkout, let alone compared fairly against outside work.

This design adds one root subdirectory, `experiments/`, holding a small, tested
workflow: a YAML config goes in, a per-run directory plus a local MLflow record
comes out. **Milestone 1 covers two external baselines only** — DASH-tree and
COSMO-NET, plus a trivial floor predictor. Sieve itself is deferred to a later
milestone; the predictor interface is built so it drops in as one more file with
no change to the harness.

## Decisions

| Topic | Decision |
|---|---|
| Milestone 1 content | Baselines only: DASH-tree (Stage A: refit, not rebuild) and COSMO-NET, plus a global-mean floor |
| Headline split | `biased_split` (train on small molecules, test on larger). Random `split` reported beside it |
| Primary metric | Wasserstein-1 on the **normalized** molecule sigma profile; each row divided by **its own** sum — a pure shape distance, decoupled from area error |
| Secondary metrics | Total molecule area (MAE/RMSE/R²) when a predictor supplies it; total molecule charge (MAE/RMSE, **no R²**) when a predictor supplies it |
| Total molecule charge | A **known input** (the molecule's formal charge). Per-atom charges are reconciled to it; the charge metrics score the total **before** reconciliation, i.e. how well the raw output would have reproduced the given total |
| DASH | Staged. Stage A (this milestone): keep the published tree topology, fit our own per-node statistics on the training split, predict by support-based back-off — the same algorithm Sieve itself uses, applied to DASH's hierarchy. Stage B (later, separate work): rebuild the tree topology with the attention GNN on our data |
| COSMO-NET | Pinned git clone at a fixed commit, retrained on our splits, run as a subprocess in its own virtual environment to avoid dependency conflicts with the core project |
| Val split | Used. COSMO-NET early-stops on `val`; DASH's `minimum_support` is chosen on `val`. `test` is touched once, at the end |
| Seeds | 3 seeds for COSMO-NET (stochastic training); 1 for DASH Stage A (the fit is deterministic) |
| Scheme | `cosmo-sac-2010` only, for this milestone |
| Config | YAML files, parsed into frozen dataclasses that reject unknown keys |
| Tracking | A local MLflow file store, plus a self-contained per-run directory that does not depend on MLflow being present |
| Runner | A Python CLI (`python -m sieve_experiments ...`) plus a Makefile for the common targets |

Two facts keep this milestone small:

- The cosmolayer sigma grid is **51 points, −0.025 … +0.025 e/Å², bin width
  0.001**, identical across all three averaging schemes, and identical to
  COSMO-NET's own grid. No re-binning is needed anywhere in the pipeline.
- DASH-properties (Lehner et al., *J. Chem. Phys.* 161, 074103, 2024) reuses one
  tree shape across many atomic properties. Stage A therefore needs no GNN
  retraining: match each atom to a node path with the published tree, fit our own
  statistics at each node, and back off toward the root when support is thin —
  which is exactly Sieve's own algorithm, applied to someone else's hierarchy.

## Architecture

```
experiments/
  README.md                     how to run; what each metric means
  Makefile                      the runner front-end
  pins.toml                     external repo URLs + commit SHAs + python versions
  cosmonet-requirements.txt     uv-pip-compiled lock for the COSMO-NET venv
  configs/                      global-mean-biased, dash-{biased,random},
                                cosmonet-{biased,random}.yaml
  results/summary.csv           the one committed results table
  sieve_experiments/            the importable package
    config.py  data.py  metrics.py  plots.py  runner.py  cli.py  prepare_store.py
    predictors/  base.py  global_mean.py  dash.py  cosmonet.py  __init__.py (REGISTRY)
  runs/  mlruns/  cache/  external/      all git-ignored
```

`pins.toml` and `cosmonet-requirements.txt` are tracked at `experiments/`, not
inside `external/`, because git cannot un-ignore a file that sits inside an
ignored directory. Tests live in the existing `tests/` directory so
`testpaths` and the ruff `include` list need no new machinery.

### The predictor seam

```python
class Predictor(Protocol):
    name: ClassVar[str]
    def fit(self, train, val, *, rng) -> None
    def predict(self, test) -> Prediction

class AtomPredictor(ABC):      # DASH, later Sieve — base class does the rollup
    charge_reconciliation: str = "std_weighted"
    def fit_atoms(...); def predict_atoms(...) -> AtomPrediction

class MoleculePredictor(ABC):  # COSMO-NET
    def predict_molecules(...) -> Prediction
```

`Prediction` requires only `mol_profile`; every other field (`mol_area`,
`mol_charge_raw`, atom-level arrays) is optional, and `metrics.molecule_metrics`
skips the corresponding metric block when a field is absent. That is what lets a
molecule-level model (COSMO-NET) and an atom-level model (DASH, later Sieve)
share one harness. The atom → molecule rollup is a plain sum, defined once in
`predictors/base.py::roll_up`, never an average or a normalization — valid
because unnormalized atom profiles partition the molecule profile
(`design.md` §11.4).

Charge reconciliation happens on the atom array, then the total is re-summed, so
after reconciliation the molecule total matches the known net charge by
construction. The charge metrics therefore score `mol_charge_raw`, the
pre-reconciliation total, which is the honest answer to "how well would the raw
prediction have reproduced the given total."

### Metrics

Promoted out of the duplicated code in `scripts/train_chaos_sigma_profile.py`
into `experiments/sieve_experiments/metrics.py` (pure numpy, no cosmolayer, no
rdkit, so it is independently unit-testable):

| key | definition |
|---|---|
| `profile/w1_norm_mean` **(primary)** | mean per-row Wasserstein-1 after dividing each row by its own sum |
| `profile/w1_norm_area_weighted` | same rows, weighted by true molecule area |
| `profile/w1_abs_mean`, `/mae`, `/rmse`, `/r2` | on unnormalized profiles, for continuity with today's numbers |
| `area/mae`, `/rmse`, `/r2` | total molecule area, when supplied |
| `charge/mae`, `/rmse`, `/max_abs_residual` | total molecule charge, when supplied. **No R²** — net charge is ~0 for nearly every molecule in the store, making R² meaningless |
| `n_test`, `n_degenerate` | molecule count; rows whose true or predicted profile sums to ≤ 0 are excluded from the normalized metrics |

Two implementation traps, both load-bearing: `pyproject.toml` sets
`filterwarnings = ["error::RuntimeWarning"]`, so row normalization must mask a
zero sum explicitly rather than divide and suppress the warning; and MLflow
rejects non-ASCII metric keys, so the existing `"R²"` key is renamed to `r2` at
the source.

### Data

`MoleculeSet` (in `data.py`) is one frozen dataclass holding molecule- and
atom-level truth plus a `select(mask)` method — the only place a split mask is
applied. Molecule-level truth (profile, area, charge) comes straight from
`SegmentStore.compute_molecule_sigma_profiles(scheme)` and is cached once per
store/scheme to `experiments/cache/` (~22 MB). Atom-level truth is recomputed per
run rather than cached, since a full cache would be ~400 MB; `time/data_s` is
recorded in every manifest so that choice can be revisited on evidence. Net
charge is `Chem.GetFormalCharge(mol)` — an exact integer, and the known input
described above.

### Reproducibility

- `cosmolayer` becomes a declared dependency behind a new `experiments` extra
  (never in core deps — it pulls in torch, lightning, open3d), pinned via
  `[tool.uv.sources]` to a commit SHA.
- `uv.lock` is tracked (removed from `.gitignore`) and CI now runs
  `uv sync --locked` instead of `pip install -e`.
- `experiments/**/*.py` is added to the ruff `include` list and to the
  pre-commit hook and CI's `ty check`, matching `src/` and `tests/` coverage.
  `scripts/` and `stores/` stay out of scope — legacy, untouched this milestone.
- COSMO-NET runs in its own uv-managed virtual environment, built from a
  `uv pip compile`-locked requirements file, entirely separate from the core
  project environment, to avoid its TensorFlow/DeepChem/PyTorch stack
  conflicting with anything else.
- Every run writes a `manifest.json` recording the git commit, seed, package
  versions, external pins, resolved config, timings, and split statistics
  (including `train_mean_heavy_atoms` / `test_mean_heavy_atoms`, the one-line
  proof that a `biased_split` run really is an extrapolation test).

## Testing

Fast, CI-safe tests (no cosmolayer, no mlflow, no network) cover the metrics
module with analytic cases (a pinned Wasserstein-1 unit test on two one-hot
vectors, scale invariance of the normalized form, a degenerate-row NaN with no
warning), config loading and rejection of unknown keys, the `MoleculeSet` rollup
and all three charge-reconciliation modes, and a full pipeline smoke test on a
synthetic store. Optional-data tests, guarded like the existing
`tests/test_benchmark.py`, check that summed atom profiles reproduce
`compute_molecule_sigma_profiles` on the real store — the single highest-value
test in the milestone, since the atom → molecule path fails silently otherwise.

## Explicitly out of scope for this milestone

- Rewiring `scripts/train_chaos_*.py` onto the new metrics module.
- The Sieve predictor itself (the seam is built, the implementation is not).
- DASH Stage B (rebuilding the tree topology).

## Risks

1. DASH atom matching (~2.1M atoms through a Python API with HDF5 lookups) may
   be slow; mitigated with a path cache, a process pool, and a `--limit` timing
   probe before committing to a full run.
2. DASH's property trees auto-download from an external repository with no
   version pinned by the API; the harness hashes the local directory to detect
   drift, though it cannot prevent it.
3. COSMO-NET's dependency stack may not resolve cleanly on the development
   machine's platform; this is the largest schedule risk in the milestone.
4. COSMO-NET's output row order is not guaranteed — results are joined back by
   SMILES, never by position.
5. The `cosmolayer` git pin uses SSH and will not resolve on a machine without
   the corresponding key; it is kept out of CI by design for this reason.
