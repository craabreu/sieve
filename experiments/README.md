# Sieve baseline experiment harness

A reproducible workflow for Milestone 1: two external baselines (DASH-tree,
COSMO-NET) on the chaos-store sigma-profile prediction task, evaluated on
Sieve's own `biased_split` extrapolation split. Design doc:
`docs/superpowers/specs/2026-08-24-baseline-experiment-harness-design.md`.

## Status (2026-08-24) — read this before continuing the work

**Done and verified (T0–T7 of the design doc's task table):**

- `sieve_experiments/metrics.py` — Wasserstein-1 + regression metrics, 18 unit
  tests with hand-computed expected values.
- `sieve_experiments/config.py` — YAML config, rejects unknown keys, 5 tracked
  configs in `configs/`.
- `sieve_experiments/data.py` — `MoleculeSet`, `molecule_sum`, `select`,
  `load_molecule_set` (real-store loader with a molecule-truth cache).
- `sieve_experiments/predictors/base.py` — the `Predictor` seam
  (`AtomPredictor`/`MoleculePredictor`), charge reconciliation, `roll_up`.
  `predictors/global_mean.py` is the floor predictor and the smoke-test
  fixture.
- `sieve_experiments/runner.py`, `cli.py`, `plots.py`, `__main__.py` — the
  full pipeline: `python -m sieve_experiments run --config <path>` writes a
  run directory (`config.resolved.yaml`, `metrics.json`, `manifest.json`,
  `predictions.npz`, `plots/`, `stdout.log`) and optionally logs to a local
  MLflow store. Verified end-to-end against the **real** local chaos-store
  (not just synthetic data): `global_mean` on `biased_split` correctly shows
  `test_mean_num_atoms (50.9) > val (40.0) > train (31.0)` in its manifest —
  proof the extrapolation split works as intended.
- `sieve_experiments/prepare_store.py` — idempotent; verified against the
  already-prepared local `stores/chaos-store/`.
- 183 fast tests passing (`uv run pytest -q`, no optional deps needed) + 8
  optional-data tests passing against the real store (need `cosmolayer` +
  `stores/chaos-store/` present — see "Running against real data" below).

**In progress — DASH-tree predictor (T8), stopped mid-implementation:**

The DASH-tree API has been fully researched (see `pins.toml`'s `[dash_tree]`
notes and the design doc) but `predictors/dash.py` has **not been written
yet**. What's known and ready to use:

- DASH-tree is pinned at `experiments/pins.toml`'s `[dash_tree]` commit.
  Clone it with `git clone https://github.com/rinikerlab/DASH-tree.git
  experiments/external/DASH-tree && cd experiments/external/DASH-tree &&
  git checkout <pinned commit>`.
- **Do not `pip install` it** — use a `sys.path.insert(0, ...)` shim in
  `predictors/dash.py` instead (see `pins.toml` for why).
- `DASHTree(preload=False)` needs no network access — the default
  MBIS-charge tree topology ships inside the clone
  (`serenityff/charge/data/default_dash_tree/`, ~300MB, 122 branch files).
- Matching API: `tree.match_new_atom(rdkit_atom_idx, mol, max_depth=...,
  attention_threshold=..., attention_increment_threshold=...)` returns a
  `list[int]`: `[branch_idx, 0, node_id_1, node_id_2, ...]` (deepest last).
  `(branch_idx, node_id)` is the unique key to fit our own per-node
  statistics against — that IS "Stage A": walk training atoms to their paths,
  accumulate count/mean per `(branch_idx, node_id)` (prune below
  `minimum_support`), then at predict time walk the path deepest→shallowest
  and take the first retained key, else a global fallback mean. This is
  Sieve's own back-off algorithm applied to DASH's published tree shape.
- **Atom index mapping**: the chaos-store's SMILES are atom-mapped in COSMO
  file order. To get the RDKit atom index matching flat position `j` within
  a molecule, parse with `Chem.SmilesParserParams(); params.removeHs =
  False`, then `order = np.argsort([a.GetAtomMapNum() for a in
  mol.GetAtoms()])`; the RDKit index is `order[j]`. This mirrors
  `src/sieve/io/cosmolayer_adapter.py`'s exact convention — reuse it rather
  than re-deriving it.
- Runtime deps for matching-only (no torch): `numpy`, `pandas`, `rdkit`,
  `tables`, `tqdm`, `pillow` — already declared in pyproject.toml's
  `experiments` extra.
- **Important gotcha already hit and fixed**: cloning any repo with its own
  `pyproject.toml` under `experiments/` gets silently absorbed by `uv` as an
  implicit workspace member unless excluded. This repo's root
  `pyproject.toml` already has `[tool.uv.workspace] exclude =
  ["experiments/external/*"]` — keep it if you ever move that path.

**Not started — COSMO-NET predictor (T9):**

- `[cosmonet]` in `pins.toml` has the repo URL but **no pinned commit yet** —
  a clone was interrupted by a disk-space incident before `git rev-parse
  HEAD` could be recorded. Re-clone, record the commit, and fill in
  `pins.toml`. The checkout is ~4.8GB (includes CSAC-2002/2010 data CSVs) —
  make sure there's real disk headroom first.
- `cosmonet-requirements.txt` has not been compiled yet. It needs a
  CUDA-enabled torch build matching the GPU server ("estes" — see the
  `gpu-server-estes` memory entry: CUDA 13.2 driver, Ubuntu 22.04, Python
  3.10.12, dual RTX 4090, no `uv` installed there yet). Compile it there or
  with a matching `--extra-index-url` (e.g. `https://download.pytorch.org/
  whl/cu124`) targeting Python 3.10 / linux-x86_64.

**Not started — T10** (`summarize` polish, results table, this README's
final form).

**Known external gap:** Zenodo record 22050672 (the chaos-store's official
source, used by `prepare_store.download_chaos_store`) returned HTTP 404
("not registered") from this dev machine on 2026-08-24 — not a rate limit.
`prepare_store.py` handles this by trust-on-first-download hashing rather
than checking a published checksum (see its module docstring). Check whether
the record resolves from wherever you're running next; if so, consider
filling in `EXPECTED_ZIP_SHA256` in `prepare_store.py` from Zenodo's
published checksum.

## Running today's working pieces

```bash
uv sync --locked --extra dev --extra chem --extra experiments   # heavy: pulls cosmolayer, torch (via cosmolayer), rdkit, pandas, mlflow
uv run pytest -q                                                 # fast suite + optional-data suite (skips gracefully without the store)

# a real run against the local store (already present at stores/chaos-store/):
uv run python -m sieve_experiments run \
    --config experiments/configs/global-mean-biased.yaml \
    --allow-dirty --no-tracking --limit 500
```

`--allow-dirty` is needed on an uncommitted tree; drop it once committed.
Runs land in `experiments/runs/` (git-ignored).
