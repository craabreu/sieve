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

**Profile-mode experiment, 2026-08-25.** Does the shape/location/area
decomposition (`fit_backoff`/`predict_backoff`, `profile_mode="decomposed"`,
the default) earn its keep over the simplest alternative — bin-wise-average
each tree node's raw, unnormalized atom profiles directly, with area/charge
only ever *derived* from the resulting profile (`sum`, `profile @
sigma_values`), never fit as their own quantities
(`fit_backoff_raw`/`predict_backoff_raw`, `profile_mode="raw"`)? Both run
full-store, `biased_split` (`configs/dash-biased.yaml` vs.
`configs/dash-raw-biased.yaml`):

| profile_mode | `atom/profile/w1_norm_mean` | `profile/w1_norm_mean` | `area/r2` | `charge/mae` |
| --- | --- | --- | --- | --- |
| decomposed (default) | 1.030 | 0.449 | 0.949 | 0.102 |
| raw | 1.058 | **0.442** | 0.949 | 0.102 |

`area/r2`/`charge/mae` are identical (to floating-point noise) either way —
not a coincidence: molecule-level area/charge are additive rollups of
atom-level area/charge, and "raw"'s derived charge (mean-profile @
sigma_values) equals mean(individual atom charges) exactly by linearity,
the same number "decomposed" fits directly. The decomposition changes
nothing about area/charge, only how the profile bins are shaped. There, the
two modes diverge in an unexpected direction: at the **atom** level "raw"
is worse (1.058 vs 1.030 — the blur `fit_backoff`'s docstring predicts from
averaging raw profiles across a node's differing sigma-centroids), but at
the **molecule** level (after summing every atom into its molecule) "raw"
comes out slightly *better* (0.442 vs 0.449) — "decomposed"'s own
reconstruction (shifting a shape template to a *predicted* location, itself
only ever an imperfect point estimate) introduces its own per-atom
placement error that doesn't obviously wash out any better than raw's blur
does once atoms are summed. `profile_mode="decomposed"` stays the default
(atom-level accuracy is the more defensible thing to optimize for, and the
molecule-level gap for "raw" is small), but `"raw"` stays available as a
documented, cheaper alternative. Full writeup in `pins.toml`'s
`[dash_tree]` notes.

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
- **Correction, 2026-08-25 (found while investigating T10's W1 gap below):**
  the claim above — that training used `MODEL_HPARAMS`'s smaller block
  (`hidden_size=51, ffn_num_layers=1, dropout=0.1`) — is **wrong**.
  `DMPNNModel.__init__`'s real parameter names (checked via
  `inspect.signature`) are `enc_hidden`, `ffn_layers`, `enc_dropout_p`/
  `ffn_dropout_p` — not `hidden_size`, `ffn_num_layers`, `dropout`. Those
  three keys (plus `message_passing_steps`) get silently absorbed into
  `**kwargs` and forwarded to `TorchModel.__init__`, never reaching the
  encoder. Confirmed directly against our own trained checkpoint's weights:
  `encoder.W_i.weight` is `(300, 147)` — deepchem's own defaults
  (`enc_hidden=300`, `atom_fdim=133 + bond_fdim=14`), not the paper's
  35/12-dim features either (`atom_fdim=35` alone is also ineffective
  without `use_default_fdim=False`) — and `ffn.linears.0.weight` is
  `(300, 300)`, a 3-layer FFN, not the claimed single layer. **T9's actual
  model is deepchem's own default-sized DMPNN** (bigger and richer than
  either the paper describes or T10 faithfully implements), not the paper's
  claimed architecture — the numbers above are real, just mischaracterized
  until now. `cosmonet-random.yaml` (the sanity split) has no trained
  checkpoint yet — only `biased_split` has been trained so far.

**Done and verified — Chemprop reimplementation of COSMO-NET (T10), 2026-08-25.**
Replaces what was originally planned as T10 (renumbered, now T12 below —
originally to T11, then T11 itself was reassigned to per-atom Chemprop).

COSMO-NET-Paper's own repo has **three independently-confirmed
reproducibility gaps**, found while investigating T9's negative-sigma-bin
issue, that make it unsuitable as the sole D-MPNN sigma-profile baseline
going forward:

1. The shipped `Sigma_saved_model/StratifiedCATEGORY_CV5/` checkpoint's own
   weights don't match the hyperparameters printed in its own committed
   training log (log: `hidden_size=51, ffn_num_layers=1`; checkpoint
   state_dict: `hidden_size=300, ffn_num_layers=3`) — root cause found
   2026-08-25 (see T9's correction note above): `MODEL_HPARAMS`'s keys
   mostly don't match real `DMPNNModel.__init__` parameter names, so they
   never reach the encoder and every run silently builds deepchem's own
   default-shaped model regardless of what the log echoes back. (The
   commit-timing observation — `.pt` files added in a separate, later
   commit than the log/results — was real but turned out not to be the
   cause; it's a coincidence, not a swap between two different runs.)
2. The atom featurizer needed to reproduce that checkpoint's actual input
   dimension (`atom_fdim=35`, not deepchem's stock 133-dim) was never
   published — independently confirmed by a second party in the repo's own
   GitHub issue #1 (opened 2026-08-09, still unanswered).
3. The published training script (`DMPNN-Train-pSigma.py`) has **no**
   non-negativity enforcement anywhere in its prediction path (no
   softplus/clamp/clip between `model.predict()` and the CSV write), yet
   its own README claims a softplus-patched FFN was used, and its own
   published `Results/pSigma-DMPNN.csv` has exactly 0% negative
   sigma-profile bins across all 5 folds — the published script cannot
   reproduce its own repo's published results.

T9 (COSMO-NET, above) stays as-is: a real, honestly-documented baseline
trained with what's actually publishable in that repo (stock deepchem, no
softplus) — its negative-bin issue (19.6% of test-set bins, confirmed) is
now a known, explained limitation, not silently swept under the rug.

T10 is a from-scratch D-MPNN sigma-profile predictor built on
[Chemprop](https://github.com/chemprop/chemprop) (the actively-maintained,
standard open-source D-MPNN implementation) instead of the
deepchem-wrapped DMPNN stack, without depending on COSMO-NET-Paper's own
unreproducible artifacts:

- Dependency resolution (`uv sync --extra chemprop`, a new opt-in extra)
  was clean on the first try — chemprop 2.3.1 is compatible with the main
  venv's existing torch 2.13.0+cu130 (CUDA available) and lightning 2.6.5,
  both already present transitively via `cosmolayer`. No separate venv,
  unlike T9 — training runs in-process.
- **Revised 2026-08-25** (prompted by the user pointing out that T10's
  original W1-vs-T9 comparison "makes no sense"): T10 originally targeted
  the architecture the peer-reviewed paper's own text specifies (Tables
  1–2, Section 2.2.2 — `hidden_size=51`, `ffn_num_layers=1`, `dropout=0.1`,
  sum pooling, 12-dim bond features). Investigating why that scored worse
  than T9 found T9's actual trained model was never running that
  architecture at all (see T9's correction note above) — every real run of
  `DMPNN-Train-pSigma.py`, ours included, silently trains deepchem's own
  default-shaped DMPNN instead, because most of `MODEL_HPARAMS`'s keys
  aren't real parameter names on the installed deepchem version. **T10 now
  targets that real, empirically-verified architecture instead** — the one
  thing in this whole picture with direct evidence (trained checkpoint
  weights, from both the paper repo's own shipped one and our independent
  T9 run) behind it, rather than prose that neither run actually followed:

  | | originally targeted (paper's text) | now targeted (verified real) |
  | --- | --- | --- |
  | `hidden_size` | 51 | **300** |
  | FFN layers (total) | 1 | **3** |
  | dropout | 0.1 | **0.0** |
  | bond features | 12-dim (Table 2) | **14-dim** (deepchem's stock, never patched) |
  | atom features | 35-dim (Table 1) | 35-dim, **unchanged** — the one part of the paper's claim that was genuinely realized |
  | readout | sum (the paper's own eqn 11) | **mean** — `aggregation` isn't in `MODEL_HPARAMS` at all |

  Getting the FFN layer count right needed tracing *both* frameworks'
  layer-counting conventions precisely, since they disagree by one:
  deepchem's `PositionwiseFeedForward` treats its `n_layers` as the total
  Linear-layer count; chemprop's own `MLP.build(n_layers=N)` instead yields
  `N+1` total layers. So deepchem's real `ffn_layers=3` needs chemprop's
  `n_layers=2` — not 3, and not the original 0. Locked in by a dedicated
  test that counts actual `nn.Linear` submodules rather than trusting
  either framework's parameter name at face value. The bond side no longer
  needs a custom featurizer at all: deepchem's real (never-patched) 14-dim
  bond features turn out to be structurally identical to chemprop's own
  built-in `MultiHotBondFeaturizer()` default, so that's used directly.
  Not replicated: the real run's exact optimizer/LR schedule
  (`ExponentialDecay`, set directly in the script rather than through the
  broken `MODEL_HPARAMS` path, so genuinely applied) vs. chemprop's own
  default Adam + Noam-like warmup — a documented, deliberately-deferred
  remaining deviation.
- **Non-negativity is structural, not a post-hoc clip**: softplus sits on
  the FFN's raw output *before* unscaling; since min-max scaling's
  `y_min ≥ 0` for every profile bin, `softplus(x) > 0` composed with
  unscaling guarantees `p(σ) > 0` for every prediction, by construction —
  exactly the demonstration the paper's own repo could not produce (§1's
  gap 3, above). This part of the design is unaffected by the revision
  above. 10 fast tests + 3 real end-to-end optional tests (bond-featurizer
  dimension check, FFN layer-count check, and a fit/predict run asserting
  zero negative bins), all passing.
- One real bug caught in testing: `pl.Trainer(accelerator="auto")` alone
  silently launched multi-GPU DDP on this machine's 2 GPUs, splitting the
  validation/test batch across ranks — surfaced immediately as a metrics
  shape mismatch (10 true rows vs. 5 predicted). Fixed with an explicit
  `devices=1`.
- The revised architecture has 401K trainable params (vs. the original's
  12.1K). A `--limit 5000` timing probe: 122.3s — barely different from
  the original, much smaller model's 127.8s at the same scale (small-batch
  GPU training here is overhead-bound, not compute-bound).
- **Full-store run (revised architecture): 1337.8s (22.3 min)**, matching
  the probe's extrapolation almost exactly. **T10 now beats T9** on both
  profile shape and area — unsurprising once the correction above is
  understood: T10 now runs the real architecture T9 accidentally used,
  built deliberately (proper softplus, proper min-max scaling, no
  silently-broken kwargs) rather than by accident.

  | metric | dash_backoff | global_mean | cosmonet (T9) | **chemprop_dmpnn (T10)** |
  | --- | --- | --- | --- | --- |
  | `profile/w1_norm_mean` | 0.449 | 1.024 | 0.224 | **0.2194** |
  | `area/r2` | 0.949 | 0.415 | 0.775 | **0.828** |
  | `charge/mae` | 0.102 | 0.00694* | 0.00546 | 0.0201 |
  | negative sigma bins | — | — | 19.6% | **0%** |

  The comparison that originally "made no sense" is fully resolved: T10
  and T9 were never comparable before (different real architectures under
  an identical-looking config — T9 secretly 300 hidden units/3 FFN
  layers/mean pooling, T10 faithfully 51/1/sum), and now they are (same
  real architecture, T10 the deliberate version of it). T10 is still the
  only one of the two whose non-negativity is a structural guarantee
  rather than a discovered gap — now it also happens to score better,
  though that was never the point.
- **Loss-function experiment**: `ChempropDMPNNPredictor` now has a
  `loss_mode` option (`"mse"` default, `"w1_normalized"`, `"mse_cumsum"`) —
  tried against the default MSE. `w1_normalized` (Wasserstein-1 on each
  profile normalized by its own row sum — a pure shape loss) reached the
  **best profile shape of any config in this milestone**,
  `profile/w1_norm_mean` **0.2071** — but `area/r2` and `charge/mae` are
  **not reported** for it: normalizing away all magnitude information
  before computing the loss gives the model zero gradient for scale, so
  area/charge were never something it was trying (and failing) to predict
  — quoting an R²/MAE there would misrepresent an untrained byproduct as a
  real predictive failure. `mse_cumsum` (MSE on raw cumulative sums, tried
  at `--limit 5000` scale only — a full run wasn't judged worth it) does
  supervise magnitude and landed worse on shape and only marginally better
  on area than plain MSE. Neither replaces the default; both stay
  available as documented options. Full numbers and the mechanistic
  explanation in `pins.toml`'s `[chemprop]` notes.

See `pins.toml`'s `[chemprop]` notes for the full gotcha writeup, including
the first (now-superseded) paper-faithful run's own numbers, kept as the
record of what "faithfully implementing the paper's own claimed
hyperparameters" actually produces.

**Done and verified — per-atom Chemprop (T11), 2026-08-25.**

`predictors/chemprop_atom.py` — a `MolAtomBondMPNN` predicting one 51-bin
profile per **atom**, off that atom's own message-passing hidden state.
`agg=None`, no molecule-level and no bond-level head; molecule results come
from the harness's ordinary `roll_up` of the atom predictions, so they are a
pure readout of atom quality rather than a separately-trained head. This is
the second atom-level predictor after DASH Stage A (T8), and the comparison
against it is the point: learned per-atom vs averaged-over-a-published-tree.

Otherwise it follows T10's recipe exactly — raw unshifted, unnormalized
bins, per-bin min-max scaling fit on the training split only, plain MSE —
so T11-vs-T10 isolates the atom-vs-molecule head and T11-vs-DASH isolates
learned-vs-averaged.

- **The finding: softplus does not transfer to the atom level.** T10's
  softplus output **collapses entirely here** — it predicts identically zero
  and cannot overfit even 20 molecules. This is structural, not tuning.
  `softplus(x) = 0` only as `x → −∞`, and atom profiles are **81.4% exact
  zeros** (each atom occupies ~9.5 of the 51 bins; a *molecule* profile is
  far denser — 50.2% zeros, 25.4 bins populated). MSE therefore drives most
  outputs toward exactly zero, pushing pre-activations to −∞, which is
  precisely where softplus's own derivative `sigmoid(x)` vanishes. The head
  dies. T10 never hit this because molecule targets are dense and never need
  the output driven to exact zero. Measured on a 20-molecule overfit (true
  mean atom area 7.676):

  | output activation | w1 | predicted area | negative bins |
  | --- | --- | --- | --- |
  | softplus (T10's) | 6.717 | 0.000 | 0 — dead |
  | **squared (`x²`)** | **0.878** | **7.859** | **0** |
  | abs | 0.947 | 7.939 | 0 |
  | plain linear | 0.825 | 7.786 | 13,110 |

  `x²` is the adopted default: **equally structurally non-negative** (the
  activation still sits on the raw output before unscaling, and every bin's
  training minimum is exactly 0, so `y_min ≥ 0`), but exact zero is reachable
  at the finite point `x = 0`, so no gradient dies. It costs essentially
  nothing against an unconstrained linear head, which would forfeit the
  non-negativity guarantee that was T10's whole selling point. Selectable via
  `output_activation`; `"softplus"` stays available so the collapse stays
  reproducible.
- **Atom features**: chemprop's own `MultiHotAtomFeaturizer.v2()` (72-dim,
  Z = 1–36 plus 53), *not* T10's `PaperAtomFeaturizer`. The paper's Table 1
  vocabulary has **no hydrogen**, and 56.8% of chaos-store atoms are hydrogen
  once the graph keeps explicit H — which it must, since the store carries
  atom-level truth for every atom. Covers every chaos-store element except
  Sb and Te.
- **Atom ordering** — the one thing that would silently produce plausible but
  meaningless numbers. chemprop's `make_mol(..., reorder_atoms=True)`
  renumbers by atom-map number, which *is* the store's own flat atom order,
  so no manual permutation is needed (unlike `dash.py`). Guarded three ways:
  a per-molecule count + map-order assert, a test against the store's own
  `element` column (dropping `reorder_atoms=True` changes the element
  sequence for 38 of 40 molecules and mismatches 16.0% of atoms, so it
  genuinely discriminates), and a reversed-molecule-order prediction test.
- **A shape check cannot catch a shuffled predict loader**: `build_dataloader`
  shuffles *molecules*, and each molecule's atoms travel with it as a
  contiguous block, so a shuffled loader returns the identical
  `(n_atoms, 51)` shape with blocks permuted. The guard is a per-molecule
  atom-count sequence check recovered from each batch's own `bmg.batch`.
- **100% atom coverage**, unlike DASH (~1.0% of test atoms fall back to the
  global mean) — see the coverage caveat in T8. The comparison mildly favours
  T11 for that reason.

**Full-store results, 2026-08-25.** `--config configs/chemprop-atom-biased.yaml`,
no `--limit`, `n_test` 5333 molecules / 203,063 atoms. `time/fit_s` 2048s
(34 min, matching the `--limit 5000` probe's extrapolation), `time/predict_s`
6.5s. **0/271,983 rolled-up bins negative**; predicted molecule areas span
103–551 against a true 114–520.

| metric | DASH decomposed | DASH raw | **chemprop_atom (T11)** | chemprop_dmpnn (T10) |
| --- | --- | --- | --- | --- |
| `atom/profile/w1_norm_mean` | **1.030** | 1.058 | 1.055 | — (molecule-level) |
| `atom/area/r2` | 0.945 | 0.945 | **0.956** | — |
| `atom/charge/mae` | 0.00752 | 0.00752 | **0.00726** | — |
| `profile/w1_norm_mean` | 0.449 | 0.442 | 0.380 | **0.219** |
| `area/r2` | **0.949** | 0.949 | 0.943 | 0.828 |
| `charge/mae` | 0.102 | 0.102 | 0.0792 | **0.0201** |

Reading these honestly:

- **T11 beats DASH on atom area and atom charge**, and is essentially tied
  with it on atom *shape* (1.055 vs 1.030 — DASH decomposed is marginally
  ahead, DASH raw marginally behind). A learned per-atom model does not
  obviously beat averaging over DASH's published tree at the atom level,
  which is a more interesting result than if it had.
- **But T11 is clearly ahead once rolled up to molecules** (`profile/
  w1_norm_mean` 0.380 vs DASH's 0.449/0.442) — the same atom-level-vs-
  molecule-level decoupling the `profile_mode` experiment turned up in T8:
  the two granularities' shape errors do not move together, because per-atom
  errors can cancel or compound when summed into a molecule.
- **T10 still owns molecule-level shape and charge** (0.219, 0.0201) by a
  wide margin — unsurprising, since it optimizes the molecule profile
  directly, whereas T11's molecule numbers are an unoptimized by-product of
  summing atoms. **T11 owns area** (0.943 vs 0.828). So the atom-level head
  buys area accuracy and per-atom detail at a real cost in molecule-level
  shape; it does not dominate T10, and shouldn't be reported as if it did.

Full gotcha writeup in `pins.toml`'s `[chemprop]` notes.

**Not started — T12** (`summarize` polish, results table, this README's
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
