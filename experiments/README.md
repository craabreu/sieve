# Sieve baseline experiment harness

A reproducible workflow for Milestone 1: two external baselines (DASH-tree,
COSMO-NET) on the chaos-store sigma-profile prediction task, evaluated on
Sieve's own `biased_split` extrapolation split. Design doc:
`docs/superpowers/specs/2026-08-24-baseline-experiment-harness-design.md`.

## Status (2026-08-25, updated) — read this before continuing the work

**Done and verified (T0–T8 of the design doc's task table):**

- `sieve_experiments/metrics.py` — Wasserstein-1 + regression metrics, hand-
  computed expected values, including empty-eval-split edge cases (a tiny
  `--limit` run can land zero molecules in val/test; every mean/max here is
  now guarded rather than raising, since `pyproject.toml` promotes
  `RuntimeWarning` to an error).
- `sieve_experiments/config.py` — YAML config, rejects unknown keys, 5 tracked
  configs in `configs/`.
- `sieve_experiments/data.py` — `MoleculeSet`, `molecule_sum`, `select`,
  `load_molecule_set` (real-store loader with a molecule-truth cache),
  `select_atoms_by_smiles` + `load_atom_truth` (atom-level truth for a split,
  joined back onto it by SMILES — never by position — since
  `load_molecule_set` never populates `MoleculeSet.atom_*`; see the module
  docstring).
- `sieve_experiments/predictors/base.py` — the `Predictor` seam
  (`AtomPredictor`/`MoleculePredictor`), charge reconciliation, `roll_up`.
  `predictors/global_mean.py` is the floor predictor and the smoke-test
  fixture. `predictors/dash.py` is DASH Stage A (below).
- `sieve_experiments/runner.py`, `cli.py`, `plots.py`, `__main__.py` — the
  full pipeline: `python -m sieve_experiments run --config <path>` writes a
  run directory (`config.resolved.yaml`, `metrics.json`, `manifest.json`,
  `predictions.npz`, `plots/`, `stdout.log`) and optionally logs to a local
  MLflow store. Verified end-to-end against the **real** local chaos-store
  (not just synthetic data): `global_mean` on `biased_split` correctly shows
  `test_mean_num_atoms > val > train` in its manifest — proof the
  extrapolation split works as intended.
- `sieve_experiments/prepare_store.py` — idempotent; verified against a fresh
  download (Zenodo record 22050672 now resolves — see "Known external gap"
  below, updated). Verifies both Zenodo's published md5 and a
  trust-on-first-download sha256.
- 215 tests passing (`uv run pytest -q`) against a fully-populated local
  environment (real `stores/chaos-store/` + the pinned DASH-tree clone under
  `experiments/external/`); 7 skipped are unrelated (`test_benchmark.py`'s
  own separate benchmark store, not chaos-store). Without those two present,
  the suite gracefully skips the optional-data tests instead of failing.

**Done and verified — DASH-tree predictor (T8):**

`predictors/dash.py` is written and end-to-end tested against the real store
and the real pinned DASH-tree clone
(`experiments/tests/test_experiment_predictor_dash_optional.py`). Two layers:

- `fit_backoff`/`predict_backoff` — pure numpy over pre-computed
  `(branch_idx, node_id)` tree paths, no optional deps, fast-suite tested
  (`experiments/tests/test_experiment_predictor_dash.py`). Walk training
  atoms to their paths and accumulate per node (prune below
  `minimum_support`), then at predict time walk the path deepest→shallowest,
  take the first retained node (else the unconditional global mean), and
  reconstruct a profile from it. Sieve's own back-off algorithm, applied to
  DASH's published tree shape.

  Each atom is decomposed into **shape** (its profile, shifted so its own
  sigma-centroid sits at zero and divided by its own area — location- and
  scale-invariant) / **location** (that sigma-centroid) / **magnitude**
  (area) *before* averaging within a node, not averaged as raw unnormalized
  vectors — atoms sharing a tree node rarely sit at the same sigma-centroid,
  and bin-wise-averaging their raw profiles smears/widens the result by
  however much those locations spread (measured 5–35% width inflation on
  real chaos-store tree-node groups). `predict_backoff` reconstructs a
  prediction by shifting the averaged shape template back out to a
  predicted location and scaling by a predicted area.

  `location_mode` (predictor param, default `"charge"`) picks how that
  scalar location comes out of a node's stats: `"charge"` divides the mean
  charge by the mean area (the natural way to combine an intensive quantity
  across a heterogeneous population — total/total, using the same additive
  charge the reconciliation machinery already relies on); `"sigma"` instead
  averages each atom's own sigma-centroid directly. The two differ whenever
  areas and locations both vary within a node (mean(charge)/mean(area) ≠
  mean(charge/area) in general) — kept as a config option rather than
  picking one, since it's a genuine, undecided modeling choice.
- `DASHBackoffPredictor` — wires that onto real atoms: RDKit for the
  atom-index mapping (`src/sieve/io/cosmolayer_adapter.py`'s
  atom-mapped-SMILES convention, reused verbatim) and
  `DASHTree.match_new_atom` for the tree path. Needs `store`/`scheme` in its
  own `predictor.params` (see `configs/dash-{biased,random}.yaml`) —
  duplicating `data.store`/`data.scheme` — because atom-level truth for the
  training split has to be loaded independently
  (`data.load_atom_truth`; `load_molecule_set` never populates it).

**First full-store pass, 2026-08-25 (superseding the earlier `--limit 300`
numbers below).** `--config configs/dash-biased.yaml`, no `--limit`, full
53,079-molecule chaos-store, `attention_threshold=5.2` (the paper's tuned
value — see `pins.toml`'s `[dash_tree]` notes), `location_mode="charge"`
(the default). A `--limit 5000` timing probe ran first (design.md risk #1):
24s wall, extrapolated the full run to ~2-3 min, which held (`real 1m58s`).
Test-split (`n_test` 5333 molecules) results, DASH vs. `global_mean` floor
(`configs/global-mean-biased.yaml`, same split, `real 2.3s`):

| metric | dash_backoff | global_mean |
| --- | --- | --- |
| `profile/w1_norm_mean` | 0.449 | 1.024 |
| `area/r2` | 0.949 | 0.415 |
| `atom/profile/w1_norm_mean` | 1.030 | — (no atom truth for global_mean's params) |
| `charge/mae` | 0.102 | 0.00694 |

DASH clearly wins on profile shape and area; `global_mean` beats DASH on
molecule-level `charge/mae` for a mechanical reason, confirmed against
`predictions.npz`'s `net_charge` array: **5296/5333 (99.3%) of test
molecules have `net_charge` exactly 0** (chaos-store is almost entirely
formally-neutral molecules). `global_mean`'s charge reconciliation
effectively predicts a constant zero, so its "MAE" is just
`mean(|true screening charge|)` = 0.00694 by construction — bit-for-bit the
number reported (unaffected by the sign-convention fix below, since
`mean(|0 - x|)` is sign-invariant). It isn't evidence `global_mean` models
charge well; it's an artifact of the metric on a near-degenerate label
distribution. `time/fit_s` 96.8s (includes the ~10s tree preload),
`time/predict_s` 14.3s, `time/data_s` 0.41s.

**Charge sign-convention bug, found and fixed 2026-08-25.** Every "charge"
metric above (and the charge-reconciliation target every predictor uses)
was silently comparing a sigma-derived charge against `net_charge` with the
wrong sign. COSMO's screening charge is the charge the dielectric
continuum induces on the cavity surface, which *opposes* the solute's own
enclosed charge — confirmed empirically on chaos-store: molecules with
`net_charge == +1` average `Sum(sigma * mol_profile) == -1.005`, not `+1`
(correlation -0.997 against `net_charge`, +0.997 against `-net_charge`).
Added `MoleculeSet.screening_charge` (`= -net_charge`) as the one correct
reconciliation/scoring target and fixed the three call sites that used
`net_charge` directly (`reconcile_charge` via `roll_up`, `global_mean`'s
`mol_charge`, `runner.py`'s `charge_true`), plus a real-data regression
test (`test_screening_charge_is_the_negated_net_charge_on_real_molecules`).
Effect on the numbers, re-run after the fix: DASH's `charge/mae` improved
0.108 → **0.102**; `global_mean`'s is unchanged (0.00694, expected — see
above); COSMO-NET's improved 3× (0.0172 → **0.00546**, see the T9 section
below) since it makes a real, nontrivial charge prediction that was being
scored against the wrong-signed target. Only 54/53,079 molecules are
charged, which is why the aggregate MAE shift is modest for DASH even
though the fix is substantively correct — the per-molecule effect on those
54 is large.

**Coverage caveat — read before quoting any DASH number.** DASH cannot match
every chaos-store atom: `init_neighbor_dict` rejects atoms whose feature
tuple is outside DASH's published vocabulary (boron, Si, Ge, Sb, Te), and it
runs over the whole molecule, so one such atom disqualifies **all** of that
molecule's atoms. Measured on the full store: **train 52103/1168845 atoms
(4.5%) from 2082/42459 molecules (4.9%)**; **test 2114/203063 atoms (1.0%)
from 51/5333 molecules (1.0%)** — close to the earlier `--limit 300`
estimate (~4%), and confirmed not to be a small-sample artifact. Those atoms
fall back to the unconditional global mean — i.e. part of any DASH score is
really the floor predictor's score. Every run records this per split in its
manifest's `match_stats` and logs a WARNING; the results table should carry
it alongside the metrics rather than quoting DASH numbers as if coverage
were 100%.

<details>
<summary>Earlier <code>--limit 300</code> smoke numbers (superseded above,
kept for the timing-probe methodology note)</summary>

A real CLI run (`--config configs/dash-biased.yaml --limit 300`) beats the
`global_mean` floor on the same slice: `profile/w1_norm_mean` 0.00054
(`location_mode="charge"`, the default) vs. 0.00079, `area/r2` 0.96 vs. 0.69.
`fit_s` ≈11s at this size, ~10s of which is the one-time DASHTree preload;
`predict_s` is 0.05s. This predates the `attention_threshold` fix (was 10,
now 5.2) and an earlier metrics-module revision, so the raw numbers aren't
directly comparable to the full-store pass above — kept only as the
worked example for "run a `--limit` timing probe before committing to a
full run" (design.md risk #1).

</details>

**Gotchas hit while building T8** (full detail in `pins.toml`'s
`[dash_tree]` notes):

- At the pinned commit, `DASHTree(preload=False)` (the on-demand-load mode
  originally planned, since it needs no network access) raises `KeyError` on
  every hydrogen atom — an ordering bug in `_get_init_layer`'s H-atom
  special case. Fixed by defaulting `DASHBackoffPredictor(preload=True)`
  (~10s, ~300MB into memory once, still no network). Note this is a
  *different* failure from the coverage caveat above — an early draft of
  this README conflated the two.
- `match_new_atom` rebuilds the whole molecule's neighbor dict on every call
  unless one is passed via `neighbor_dict=` — O(n_atoms²) per molecule.
  `_atom_paths` hoists it per molecule (as DASH's own
  `_get_allAtoms_nodePaths` does): **8.5× faster, bit-identical metrics**.
- The atom-index mapping (flat store position `j` → RDKit index `order[j]`)
  is guarded by a real alignment test against the store's own `element`
  column, mirroring `cosmolayer_adapter.py`'s `check_alignment`. A
  transposed mapping still produces perfectly finite metrics, so nothing
  else in the suite would catch it; the inverse convention mismatches ~36%
  of atoms, so the guard genuinely discriminates.
- `fit_backoff`/`predict_backoff` live in a module (`dash.py`) that also
  defines a `Path` type alias for tree paths — that shadowed
  `pathlib.Path`'s import; renamed the alias to `NodePath`. Worth watching
  for in any module that both imports `pathlib.Path` and wants a short type
  alias name.
- Two pre-existing "empty eval split" crashes surfaced by exercising a real
  small `--limit` run end-to-end: `metrics.regression_metrics`/
  `charge_metrics` raised on 0-row input (now return NaN), and
  `runner._write_plots` crashed on an empty test set via
  `plots.parity_hexbin`'s `min()`/`max()` (now skips plotting when
  `test.n_molecules == 0`). Neither is DASH-specific — any predictor hits
  these on a `--limit` small enough that `biased_split`'s val/test land
  empty (e.g. `--limit 50` on chaos-store, still exercised as a regression
  test in `test_experiment_smoke.py`/`test_experiment_metrics.py`).

**Done and verified — COSMO-NET predictor (T9), 2026-08-25:**

- `[cosmonet]` in `pins.toml` has a pinned commit (`366839a0e6f9...`,
  re-cloned after the earlier disk-space incident — 106GB free this time,
  no issue; actual checkout is ~14GB, not ~4.8GB, see `pins.toml` for why).
- Dependency resolution (**design.md risk #3, the milestone's largest
  schedule risk**) is **done and clean** — `experiments/
  cosmonet-requirements.txt` has the full pin, repro steps, and a gotcha
  (`torch-geometric` is required by `DMPNNModel` but missing from the
  repo's own `requirements.txt`).
- `sieve_experiments/cosmonet_data.py` (`category_labels` +
  `write_cosmonet_csv`) converts a chaos-store `MoleculeSet` into COSMO-NET's
  exact CSAC-CSV input format — confirmed the 51-point grid matches
  COSMO-NET's own exactly, so it's a small join, not a real transform.
- Trained the repo's own DMPNN (`Training/DMPNN/DMPNN-Train-pSigma.py
  --splitter 4`, its documented default) on the **full** chaos-store,
  `biased_split`-labeled, 100 epochs (~4h10m on "estes"'s GPU — confirmed
  via `nvidia-smi` and deepchem's own cuda auto-detection, not a silent CPU
  fallback). **The train/val/test split was certified against our own
  `biased_split` column molecule-for-molecule, twice (after the 1-epoch
  probe and again after the full run): 53,079/53,079 rows matched, 0
  mismatches both times** — not just aggregate counts lining up.
- `sieve_experiments/predictors/cosmonet.py` (`CosmonetPredictor`) wires
  the trained checkpoint's own saved prediction CSV into the `Predictor`
  seam as a SMILES-keyed lookup (not live re-inference — see its module
  docstring for why that's numerically identical for any in-store molecule,
  and where it stops generalizing). It **re-certifies every molecule's
  CATEGORY against the run's own split at fit/predict time**, every run —
  the same check done manually for this pass, now a permanent guard against
  silently reusing a checkpoint trained on a *different* split_column
  (which would leak train-set molecules into a "test" evaluation
  undetectably). 7 tests, all passing, including both leakage-guard cases.
- A real `--config configs/cosmonet-biased.yaml` harness run reproduces the
  manually-verified numbers bit-for-bit (2.7s wall — pure lookup, no
  training in the harness itself):

  | metric | dash_backoff | global_mean | **cosmonet** |
  | --- | --- | --- | --- |
  | `profile/w1_norm_mean` | 0.449 | 1.024 | **0.224** |
  | `area/r2` | 0.949 | 0.415 | 0.775 |
  | `area/mae` | 8.41 | 31.30 | 15.56 |
  | `charge/mae` | 0.102 | 0.00694* | **0.00546** |

  Numbers above are post charge-sign-fix (see the DASH section's callout
  above) — re-run after the fix, not just recomputed. COSMO-NET's
  `charge/mae` improved 3× from the pre-fix number (0.0172 → 0.00546) since
  it makes a real, nontrivial charge prediction (unlike `global_mean`'s
  constant zero), so the wrong-signed target mattered far more for it than
  for DASH. \* `global_mean`'s low `charge/mae` is a metric artifact, not a
  real win — see the git history for the full explanation (99.3% of
  chaos-store test molecules are exactly neutral). COSMO-NET wins clearly
  on profile shape (about half DASH's W1) but loses to DASH on area —
  expected, since this DMPNN was trained with a flat per-bin MSE loss
  (`torch.nn.functional.mse_loss`, all 51 sigma bins weighted equally,
  confirmed from `deepchem.models.torch_models.dmpnn`'s source), with no
  explicit shape/location/magnitude decomposition the way DASH Stage A has.
- **Known limitations, honestly scoped:** the active training used the
  script's smaller `MODEL_HPARAMS` (`hidden_size=51, ffn_num_layers=1`) —
  a larger, apparently-tuned block (`hidden_size=512, ffn_num_layers=5,
  dropout=0.232`) is commented out just above it in the source, unexplored
  here. `cosmonet-random.yaml` (the sanity split) has no trained checkpoint
  yet — only `biased_split` has been trained so far.

**Not started — T10** (`summarize` polish, results table, this README's
final form).

**Known external gap, resolved 2026-08-24:** Zenodo record 22050672 (the
chaos-store's official source) returned HTTP 404 on an earlier dev machine;
it resolves normally now. `prepare_store.py` now verifies both Zenodo's
published md5 (`EXPECTED_ZIP_MD5`, filled in from
`curl -s https://zenodo.org/api/records/22050672`) at download time and a
trust-on-first-download sha256, recorded at `stores/chaos-store/
.download.sha256` — but only checked once: the sha256 is of the downloaded
zip, deleted right after extraction, so a later run that finds the store
already present just reports the recorded hash rather than re-verifying it
against the extracted directory (nothing left to re-hash it against).

## Running today's working pieces

```bash
uv sync --locked --extra dev --extra chem --extra experiments   # heavy: pulls cosmolayer, torch (via cosmolayer), rdkit, pandas, mlflow
uv run python -m sieve_experiments prepare-store                # downloads + splits chaos-store into stores/ (git-ignored, ~8GB)
uv run pytest -q                                                 # fast suite + optional-data suite (skips gracefully without the store)

# DASH-tree clone, needed for predictors/dash.py's optional-data tests and
# real dash_backoff runs (see pins.toml's [dash_tree] for the pinned commit):
git clone https://github.com/rinikerlab/DASH-tree.git experiments/external/DASH-tree
git -C experiments/external/DASH-tree checkout 6cf1b2351c4674e602153dd493c06d9c020fc9ce

# a real run against the local store:
uv run python -m sieve_experiments run \
    --config experiments/configs/global-mean-biased.yaml \
    --allow-dirty --no-tracking --limit 500

# DASH Stage A, once the clone above is present:
uv run python -m sieve_experiments run \
    --config experiments/configs/dash-biased.yaml \
    --allow-dirty --no-tracking --limit 500
```

`--allow-dirty` is needed on an uncommitted tree; drop it once committed.
Runs land in `experiments/runs/` (git-ignored). A small `--limit` (e.g. 50)
can land zero molecules in val/test on `biased_split` — metrics/plots handle
that gracefully (NaN, no plot) rather than crashing, but the run then has no
real signal in it; `--limit 200`+ reliably gives a non-empty val/test on
chaos-store.
