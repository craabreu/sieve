# Sieve baseline experiment harness

A reproducible workflow for Milestone 1: two external baselines (DASH-tree,
COSMO-NET) on the chaos-store sigma-profile prediction task, evaluated on
Sieve's own `biased_split` extrapolation split. Design doc:
`docs/superpowers/specs/2026-08-24-baseline-experiment-harness-design.md`.

`experiments/docs/` holds the detailed engineering history behind each
predictor below (investigations, bugs found, full-store result tables,
retired approaches) — internal working notes, not part of the published
harness; safe to delete before release. This README and `pins.toml` are
the parts meant to stay.

## Headline results

Full chaos-store (53,079 molecules), `biased_split` (the extrapolation
split — train on the smallest molecules, evaluate on the largest), test
split 5333 molecules / 203,063 atoms.

| metric | global_mean | dash (T8) | chemprop_cosmonet (T10) | chemprop_atom (T11) |
| --- | --- | --- | --- | --- |
| `profile/w1_norm_mean` | 1.024 | 0.407 | **0.2194** | 0.380 |
| `area/r2` | 0.415 | 0.952 | 0.828 | **0.943** |
| `charge/mae` | 0.00694* | 0.0922 | **0.0201** | 0.0792 |
| `atom/profile/w1_norm_mean` | — (no atom truth) | 1.012 | — (molecule-level only) | **1.055**\*\* |
| `atom/area/r2` | — | — (no atom truth) | — (molecule-level only) | 0.956 |
| `atom/charge/mae` | — | — (no atom truth) | — (molecule-level only) | 0.00726 |
| negative sigma bins (rolled-up) | — | 0% | 0% | 0% |

\* `global_mean`'s low `charge/mae` is a metric artifact, not a real win —
99.3% of test molecules have `net_charge` exactly 0, so it's effectively
scoring a constant-zero prediction. \*\* Not directly comparable to DASH's
atom-level number: DASH covers ~99% of test atoms (its own published
feature vocabulary rejects a small fraction — see the DASH section's
coverage caveat below); chemprop_atom covers 100%. `chemprop_atom` wins
molecule-level `area/r2` and atom-level shape/area/charge;
`chemprop_cosmonet` wins molecule-level profile shape and charge by a wide
margin, since it optimizes the molecule profile directly rather than
rolling one up from atoms.

## Predictors

### global_mean

`predictors/global_mean.py` — the floor predictor: predicts the training
split's mean profile for every molecule. No optional dependencies; also
the smoke-test fixture for the harness itself.

### dash (T8)

`predictors/dash.py`'s `DASHPredictor` reproduces DASH-tree's published
algorithm as literally as possible: its own topology
(`DASHTree.match_new_atom`, unmodified) for atom matching, and a back-off
step that reproduces its own real prediction-time fallback
(`DASHTree.get_property_noNAN`'s deepest→shallowest walk, first populated
node wins, else the global mean) — no support/count threshold anywhere,
matching DASH's own code. Two Sieve-invented back-off variants were tried
and measurably underperformed this design on every metric; both are
retired. DASH cannot match ~1% of test atoms (its own published feature
vocabulary rejects a small fraction — boron, Si, Ge, Sb, Te); those atoms
fall back to the global mean, recorded per-run in `manifest.json`'s
`match_stats`. AA-only in every comparison in this milestone — its own
`GetDegree()` feature field is not representation-invariant on a
united-atom store (see below).

Full design story, engineering history, and results: `docs/dash.md`.

### chemprop_cosmonet (T10)

`predictors/chemprop_cosmonet.py`'s `ChempropCosmonetPredictor` is a
Chemprop D-MPNN reimplementation of COSMO-NET's sigma-profile model — not
an independent architecture inspired by the same idea, but a direct
reproduction of the *real* architecture COSMO-NET-Paper's own published
checkpoint was actually trained with (`hidden_size=300`, 3 FFN layers, no
dropout, mean pooling), reverse-engineered from the checkpoint's own
weight shapes rather than trusted from the paper's text or the training
script's own printed hyperparameters — both turned out to be wrong. This
predictor supersedes an earlier SMILES-keyed lookup baseline built
directly on COSMO-NET-Paper's own repo, retired once its checkpoint proved
untrustworthy (see `docs/cosmonet_investigation.md`). Non-negativity is
structural: softplus before unscaling, guaranteed given every profile
bin's training minimum is exactly 0.

Full architecture-revision story, dependency notes, and the loss-mode
experiment: `docs/chemprop_cosmonet.md`.

### chemprop_atom (T11)

`predictors/chemprop_atom.py` predicts one 51-bin profile per **atom**
(off that atom's own message-passing hidden state via a `MolAtomBondMPNN`
with `agg=None`), rather than per molecule — molecule numbers are a pure
`roll_up` of atom predictions, not a separately-trained head. Otherwise
follows `chemprop_cosmonet`'s recipe exactly, so it isolates the
atom-vs-molecule head (vs. T10) and learned-vs-averaged (vs. DASH).
Headline finding: `chemprop_cosmonet`'s softplus output structurally
collapses at the atom level (atom profiles are 81.4% exact zeros, driving
pre-activations to −∞ where softplus's own gradient vanishes) — fixed
with a squared (`x²`) output activation, equally non-negative but
reachable at a finite point. 100% atom coverage, unlike DASH. Genuinely
improves on the united-atom store (unlike DASH, which does not — see
below).

Full gotcha writeup and results: `docs/chemprop_atom.md`.

### The united-atom store

`chaos-store-ua` (built by `python -m sieve_experiments
coarse-grain-store`) merges each hydrogen into its heavy-atom neighbor,
letting DASH and `chemprop_atom` be scored at DASH's own native
(united-atom) representation. DASH is unambiguously worse there — its own
`atom.GetDegree()` feature field is not representation-invariant, a real
limitation of its hand-built feature scheme (confirmed against the primary
source, not just the code). `chemprop_atom` has no analogous defect and
genuinely improves there, at both granularities.

Full investigation, including a real quadratic-time performance bug found
and fixed upstream in cosmolayer: `docs/chaos_store_ua.md`.

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

# collect every run's metrics.json into one CSV:
uv run python -m sieve_experiments summarize
```

`--allow-dirty` is needed on an uncommitted tree; drop it once committed.
Runs land in `experiments/runs/` (git-ignored). A small `--limit` (e.g. 50)
can land zero molecules in val/test on `biased_split` — metrics/plots handle
that gracefully (NaN, no plot) rather than crashing, but the run then has no
real signal in it; `--limit 200`+ reliably gives a non-empty val/test on
chaos-store.
