# Sieve baseline experiment harness

A reproducible workflow for Milestone 1: two external baselines (DASH-tree,
COSMO-NET) on the chaos-store sigma-profile prediction task, evaluated on
Sieve's own `biased_split` extrapolation split. Design doc:
`docs/superpowers/specs/2026-08-24-baseline-experiment-harness-design.md`.

## Status (2026-08-26, updated) — read this before continuing the work

**Note on task numbering:** T9 (a SMILES-keyed lookup against an
externally-trained COSMO-NET checkpoint) was retired and folded into the
COSMO-NET section below (T10, "reimplemented on Chemprop") once T10 was
shown to fully supersede it — a trustworthy, checkpoint-verified-
architecture baseline trained end-to-end on our own data, with no
dependency on the external clone. T10/T11/T12 keep their existing labels
for continuity with commit history and run-directory naming; the numbering
is not closed up.

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
- 292 tests passing (`uv run pytest -q`) against a fully-populated local
  environment (real `stores/chaos-store/` + the pinned DASH-tree clone under
  `experiments/external/`); 7 skipped are unrelated (`test_benchmark.py`'s
  own separate benchmark store, not chaos-store). Without those two present,
  the suite gracefully skips the optional-data tests instead of failing.

**Done and verified — DASH (T8):**

`predictors/dash.py` is written and end-to-end tested against the real store
and the real pinned DASH-tree clone
(`experiments/tests/test_experiment_predictor_dash_optional.py`).
`DASHPredictor` reproduces DASH's own published algorithm as literally as
possible, per an explicit design goal ("rely on their code as much as
possible"): DASH-tree's own topology (`DASHTree.match_new_atom`, unmodified)
for atom matching, and a back-off step that reproduces DASH's own real
prediction-time fallback (`DASHTree.get_property_noNAN`'s deepest→shallowest
walk, first populated node wins, else the global mean over all training
atoms) — no support/count threshold anywhere, matching DASH's own code
(`Node.prune()`, the one piece of code that would implement one, is
confirmed dead: no caller anywhere in the repo).

- `populate_tree_with_sigma_properties` writes the raw (undecomposed) mean
  sigma profile + charge std directly onto `tree.data_storage[branch_idx]`,
  one row per node — the same shape DASH-properties' own paper describes its
  own population process taking (re-matching atoms, averaging into existing
  nodes). Fast-suite tested against a fake `data_storage`, no rdkit/DASH-tree
  clone needed (`experiments/tests/test_experiment_predictor_dash.py`).
- `predict_via_data_storage_walk` reimplements `get_property_noNAN`'s exact
  fallback semantics as a fast, self-contained function rather than calling
  it live — ~130× faster (0.35µs/call vs. ~47µs/call), the difference
  between a ~2 minute and ~39 minute full-store predict step. Verified
  bit-for-bit equivalent to a literal-call prototype's own output on a real
  run before the swap, not just argued.
- Wired onto real atoms via RDKit for the atom-index mapping
  (`src/sieve/io/cosmolayer_adapter.py`'s atom-mapped-SMILES convention,
  reused verbatim) and `DASHTree.match_new_atom` for the tree path. Needs
  `store`/`scheme` in its own `predictor.params` (see
  `configs/dash-{biased,random}.yaml`) — duplicating `data.store`/
  `data.scheme` — because atom-level truth for the training split has to be
  loaded independently (`data.load_atom_truth`; `load_molecule_set` never
  populates it).

Full engineering history — the two real bugs found only by running
end-to-end, DASH's paper-says-median-code-says-mean discrepancy, and the
built/measured/replaced performance story — is in `pins.toml`'s
`[dash_tree]` notes; not duplicated here.

**Full-store results, 2026-08-26.** `--config configs/dash-biased.yaml`, no
`--limit`, full 53,079-molecule chaos-store, `attention_threshold=5.2` (the
paper's tuned value), n_test 5333 molecules / 203,063 atoms, fit 93.5s,
predict 16.2s, **0/271,983 rolled-up bins negative**:

| metric | dash | global_mean |
| --- | --- | --- |
| `profile/w1_norm_mean` | **0.407** | 1.024 |
| `area/r2` | **0.952** | 0.415 |
| `atom/profile/w1_norm_mean` | 1.012 | — (no atom truth for global_mean's params) |
| `charge/mae` | **0.0922** | 0.00694* |

\* `global_mean`'s low `charge/mae` is a metric artifact, not a real win:
**5296/5333 (99.3%) of test molecules have `net_charge` exactly 0**
(chaos-store is almost entirely formally-neutral molecules), so
`global_mean`'s charge reconciliation effectively predicts a constant zero
and its "MAE" is just `mean(|true screening charge|)` by construction.

Two Sieve-invented back-off variants — shape/location/area decomposition,
and a plain bin-wise mean with a `minimum_support=5` safety threshold on top
of the published topology — were tried and measured along the way. Both
underperformed the design above on every metric shown, at both
granularities, and have since been retired (their code and configs
deleted) — the finding that settled removing `minimum_support` entirely
was that it was measurably *costing* accuracy, not buying it. Full
comparison table and the "why" in `pins.toml`'s `[dash_tree]` notes.

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
The COSMO-NET lookup baseline this bug also affected improved 3× on charge
once fixed (see the `chemprop_cosmonet` section below for that history).
Only 54/53,079 molecules are charged, so the aggregate MAE shift from this
fix is modest even though it's substantively correct — the per-molecule
effect on those 54 is large.

**Coverage caveat — read before quoting any DASH number.** DASH cannot match
every chaos-store atom: `init_neighbor_dict` rejects atoms whose feature
tuple is outside DASH's published vocabulary (boron, Si, Ge, Sb, Te), and it
runs over the whole molecule, so one such atom disqualifies **all** of that
molecule's atoms. Measured on the full store: **train 52103/1168845 atoms
(4.5%) from 2082/42459 molecules (4.9%)**; **test 2114/203063 atoms (1.0%)
from 51/5333 molecules (1.0%)**. Those atoms fall back to the unconditional
global mean — i.e. part of any DASH score is really the floor predictor's
score. Every run records this per split in its manifest's `match_stats` and
logs a WARNING; the results table should carry it alongside the metrics
rather than quoting DASH numbers as if coverage were 100%.

**Profile-mode experiment (summary; full writeup in `pins.toml`'s
`[dash_tree]` notes).** Does the shape/location/area decomposition earn its
keep over the simplest alternative — bin-wise-averaging each tree node's
raw, unnormalized atom profiles directly? Both retired variants above were
compared full-store: area/charge came out identical between them (a
consequence of linearity, not a coincidence), but profile *shape* diverged
in an unexpected direction — decomposition was better at the atom level but
slightly worse at the molecule level, since summing atoms into a molecule
doesn't let either mode's own approximation error wash out predictably.
Neither variant survives in the current design, which supersedes both (see
the full-store results above).

**All-atom vs. united-atom (summary; full writeup in `pins.toml`'s
`[chaos_store_ua]` notes).** A question surfaced while building T11: is
DASH all-atom or united-atom (implicit H)? DASH *predicts* a value for
every hydrogen but *represents* one purely as an attribute of its heavy-atom
neighbor, so a real united-atom store (`coarse-grain-store`, built via a
genuine cosmolayer perf fix —
[cosmolayer#55](https://github.com/craabreu/cosmolayer/pull/55), 40-90 min
→ 21.7s) was built to test DASH (and T11) at that native granularity fairly.

| metric | DASH AA | DASH UA |
| --- | --- | --- |
| test molecules rejected outright | 1.0% | **4.6%** |
| `atom/profile/w1_norm_mean` | 1.030 | 1.551 |
| `profile/w1_norm_mean` | 0.449 | 0.812 |
| `charge/mae` | 0.102 | 0.153 |

DASH-UA is worse across every metric — root cause: DASH's own feature tuple
uses `atom.GetDegree()` (RDKit's explicit-neighbor-only count), and its
published tree was built entirely from all-atom (explicit-H) data where
degree structurally includes bonded H. On a united-atom store, degree drops
by however many H's got merged away, presenting the published vocabulary
tuples it was never built to see — confirmed directly in DASH's own paper's
"Atom Features" section, not just the code. **Decision: DASH is AA-only for
every comparison in this milestone** — `dash-biased.yaml`/`dash-random.yaml`
only. This is a genuine limitation of DASH's own hand-built feature scheme,
not evidence that hydrogens are harder to predict in general — T11 (whose
feature scheme has no analogous defect) genuinely improves on the same
united-atom store; see the T11 section below.

**Gotchas hit while building T8** (full detail in `pins.toml`'s
`[dash_tree]` notes):

- At the pinned commit, `DASHTree(preload=False)` (the on-demand-load mode
  originally planned, since it needs no network access) raises `KeyError` on
  every hydrogen atom — an ordering bug in `_get_init_layer`'s H-atom
  special case. Fixed by defaulting `DASHPredictor(preload=True)`
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
- `dash.py` also defines a type alias for tree paths that once shadowed
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

**Done and verified — COSMO-NET, reimplemented on Chemprop (T10),
2026-08-25.** Replaces what was originally planned as T10 (renumbered, now
T12 below — originally to T11, then T11 itself was reassigned to per-atom
Chemprop). Formerly two predictors — a since-retired SMILES-keyed lookup
against an externally-trained checkpoint (T9), and this one — collapsed
into a single, trustworthy baseline once investigating T9's negative-bins
finding showed the external repo's own checkpoint could not be trusted at
all (see below); T9's own numbers are kept as historical record in
`pins.toml`'s `[cosmonet_investigation]` notes, not reproduced here.

`predictors/chemprop_cosmonet.py` (`ChempropCosmonetPredictor`) is a
Chemprop D-MPNN reimplementation of COSMO-NET's sigma-profile model — not
an independent architecture inspired by the same idea, but a direct
reproduction of the *real* architecture COSMO-NET-Paper's own published
checkpoint was actually trained with, reverse-engineered from the
checkpoint's own weight shapes. COSMO-NET-Paper's own repo (a real training
run was carried out against it, on our own full chaos-store data, before
this was understood — see `pins.toml`'s `[cosmonet_investigation]` notes)
turned out to have **three independently-confirmed reproducibility gaps**,
found while investigating a negative-sigma-bin discrepancy, that make it
unsuitable to depend on directly:

1. The shipped `Sigma_saved_model/StratifiedCATEGORY_CV5/` checkpoint's own
   weights don't match the hyperparameters printed in its own committed
   training log (log: `hidden_size=51, ffn_num_layers=1`; checkpoint
   state_dict: `hidden_size=300, ffn_num_layers=3`). Root cause:
   `DMPNNModel.__init__`'s real parameter names (checked via
   `inspect.signature`) are `enc_hidden`, `ffn_layers`, `enc_dropout_p`/
   `ffn_dropout_p` — not `hidden_size`, `ffn_num_layers`, `dropout`. Those
   three keys (plus `message_passing_steps`) get silently absorbed into
   `**kwargs` and forwarded to `TorchModel.__init__`, never reaching the
   encoder — every run, including one carried out independently on our own
   data, silently builds deepchem's own default-shaped model regardless of
   what the log echoes back. Confirmed directly against real trained
   checkpoint weights (both the paper repo's own shipped one and the
   independent run): `encoder.W_i.weight` is `(300, 147)` — deepchem's own
   defaults (`enc_hidden=300`, `atom_fdim=133 + bond_fdim=14`), not the
   paper's 35/12-dim features either (`atom_fdim=35` alone is also
   ineffective without `use_default_fdim=False`) — and `ffn.linears.0.weight`
   is `(300, 300)`, a 3-layer FFN, not the claimed single layer.
2. The atom featurizer needed to reproduce that checkpoint's actual input
   dimension (`atom_fdim=35`, not deepchem's stock 133-dim) was never
   published — independently confirmed by a second party in the repo's own
   GitHub issue #1 (opened 2026-08-09, still unanswered). The mechanism is a
   patched deepchem module (`GraphConvConstants.ATOM_FDIM` hardcoded to 35),
   not a script-level flag.
3. The published training script (`DMPNN-Train-pSigma.py`) has **no**
   non-negativity enforcement anywhere in its prediction path (no
   softplus/clamp/clip between `model.predict()` and the CSV write), yet
   its own README claims a softplus-patched FFN was used, and its own
   published `Results/pSigma-DMPNN.csv` has exactly 0% negative
   sigma-profile bins across all 5 folds — the published script cannot
   reproduce its own repo's published results.

This is exactly why `chemprop_cosmonet.py` reproduces the architecture from
checkpoint-weight evidence rather than trusting either the paper's own text
or the training script's own printed hyperparameters — both turned out to
be wrong, independently of each other. Built on
[Chemprop](https://github.com/chemprop/chemprop) (the actively-maintained,
standard open-source D-MPNN implementation) instead of the deepchem-wrapped
DMPNN stack, trained end-to-end on our own chaos-store data, with no
dependency on the external clone or its saved-CSV lookup convention (the
21GB clone has since been deleted from disk — its findings are fully
captured here and in `pins.toml`):

- Dependency resolution (`uv sync --extra chemprop`, a new opt-in extra)
  was clean on the first try — chemprop 2.3.1 is compatible with the main
  venv's existing torch 2.13.0+cu130 (CUDA available) and lightning 2.6.5,
  both already present transitively via `cosmolayer`. No separate venv,
  training runs in-process.
- **Revised 2026-08-25** (prompted by the user pointing out that an early
  W1 comparison "makes no sense"): originally targeted the architecture the
  peer-reviewed paper's own text specifies (Tables 1–2, Section 2.2.2 —
  `hidden_size=51`, `ffn_num_layers=1`, `dropout=0.1`, sum pooling, 12-dim
  bond features). Investigating that gap led directly to gap 1 above — the
  independently-trained checkpoint was never running that architecture at
  all. **This predictor now targets that real, empirically-verified
  architecture instead** — the one thing in this whole picture with direct
  evidence (trained checkpoint weights, from both the paper repo's own
  shipped one and the independent run) behind it, rather than prose that
  neither run actually followed:

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
  the probe's extrapolation almost exactly.

  | metric | dash | global_mean | **chemprop_cosmonet** |
  | --- | --- | --- | --- |
  | `profile/w1_norm_mean` | 0.407 | 1.024 | **0.2194** |
  | `area/r2` | 0.952 | 0.415 | **0.828** |
  | `charge/mae` | 0.0922 | 0.00694* | 0.0201 |
  | negative sigma bins | — | — | **0%** |

  Non-negativity here is a structural guarantee (softplus before
  unscaling, `y_min ≥ 0` for every profile bin) rather than a discovered
  gap — the retired lookup baseline (T9) had a real, unexplained 19.6%
  negative-bin rate on its own published-checkpoint predictions; see
  `pins.toml`'s `[cosmonet_investigation]` notes for that history.
- **Loss-function experiment**: `ChempropCosmonetPredictor` now has a
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
  explanation in `pins.toml`'s `[chemprop_cosmonet]` notes.

See `pins.toml`'s `[chemprop_cosmonet]` notes for the full gotcha writeup,
including the first (now-superseded) paper-faithful run's own numbers, kept
as the record of what "faithfully implementing the paper's own claimed
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

| metric | DASH decomposed (retired) | DASH raw (retired) | **chemprop_atom (T11)** | chemprop_cosmonet (T10) |
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
  molecule-level decoupling T8's own profile-mode experiment found (atom
  shape and molecule shape moved in *opposite* directions between DASH's two
  retired variants — see T8's section above): the two granularities' shape
  errors do not move together, because per-atom errors can cancel or
  compound when summed into a molecule.
- **T10 still owns molecule-level shape and charge** (0.219, 0.0201) by a
  wide margin — unsurprising, since it optimizes the molecule profile
  directly, whereas T11's molecule numbers are an unoptimized by-product of
  summing atoms. **T11 owns area** (0.943 vs 0.828). So the atom-level head
  buys area accuracy and per-atom detail at a real cost in molecule-level
  shape; it does not dominate T10, and shouldn't be reported as if it did.

**T11 on the united-atom store, 2026-08-25.** Unlike DASH (AA-only per the
decision in T8's section — see there for why), T11's feature scheme has no
representation-dependence defect to correct for, so `chemprop-atom-ua-
biased.yaml` is a legitimate comparison. Full-store, `biased_split`,
n_test 5333 molecules / 108,347 atoms, fit 19.6 min, **0/271,983 rolled-up
bins negative**:

| metric | T11 AA | T11 UA |
| --- | --- | --- |
| `atom/profile/w1_norm_mean` | 1.055 | **0.989** |
| `atom/area/r2` | 0.956 | **0.976** |
| `atom/charge/mae` | **0.0073** | 0.0107 |
| `profile/w1_norm_mean` | 0.380 | **0.330** |
| `area/r2` | 0.943 | **0.949** |
| `charge/mae` | 0.0792 | **0.0673** |

T11 genuinely *improves* on the united-atom store — at both granularities,
on every metric except atom-level charge. This is a real signal, not an
artifact of the kind that sank DASH-UA: confirmed above that T11's
featurizer has no analogous representation-dependence. Plausible mechanism
(not investigated further): a UA atom's target is a heavy atom's own
surface plus its former hydrogens' merged surface — a "chunkier," less
sparse/degenerate regression target than an individual hydrogen's own
often-small profile.

Full gotcha writeup in `pins.toml`'s `[chemprop_cosmonet]` notes.

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
# real dash runs (see pins.toml's [dash_tree] for the pinned commit):
git clone https://github.com/rinikerlab/DASH-tree.git experiments/external/DASH-tree
git -C experiments/external/DASH-tree checkout 6cf1b2351c4674e602153dd493c06d9c020fc9ce

# a real run against the local store:
uv run python -m sieve_experiments run \
    --config experiments/configs/global-mean-biased.yaml \
    --allow-dirty --no-tracking --limit 500

# DASH, once the clone above is present:
uv run python -m sieve_experiments run \
    --config experiments/configs/dash-biased.yaml \
    --allow-dirty --no-tracking --limit 500

# Chemprop reimplementation of COSMO-NET (T10), needs `uv sync --extra
# chemprop`; --set overrides max_epochs for a fast smoke check:
uv run python -m sieve_experiments run \
    --config experiments/configs/chemprop-cosmonet-biased.yaml \
    --allow-dirty --no-tracking --limit 500 \
    --set predictor.params.max_epochs=1

# united-atom store (H merged into their heavy-atom neighbor), needed for
# chemprop-atom-ua-biased.yaml -- requires chaos-store already prepared
# above; idempotent, see pins.toml's [chaos_store_ua]:
uv run python -m sieve_experiments coarse-grain-store chaos-store
```

`--allow-dirty` is needed on an uncommitted tree; drop it once committed.
Runs land in `experiments/runs/` (git-ignored). A small `--limit` (e.g. 50)
can land zero molecules in val/test on `biased_split` — metrics/plots handle
that gracefully (NaN, no plot) rather than crashing, but the run then has no
real signal in it; `--limit 200`+ reliably gives a non-empty val/test on
chaos-store.
