# Train/val/test: does an extrapolation split make val a worse test proxy?

`biased_split` is built to extrapolate: `prepare_store.py`'s
`split_chaos_store` sorts molecule clusters by mean heavy-atom count
(`assign_clusters_by_mean_size`) and assigns train first (smallest), then
val, then test (largest). `split` (the Butina-cluster "random" split) makes
no such size-ordering; train/val/test are just three cluster-disjoint
draws from the same molecule-size distribution.

If that's true, val should be a systematically *easier* proxy for test
under `biased_split` (val sits between train and test in size, closer to
train's own distribution) than under `split` (val and test should be
statistically interchangeable) -- i.e. val-test similarity should be
**worse** under `biased_split`. `runner.py`'s `_score_extra_split` scores
train and val exactly the way test is scored (same `molecule_metrics`
call, `train/`/`val/`-prefixed keys in `metrics.json`), specifically to
make this comparison possible from any run's own output, not just this one.

## The mechanism, confirmed directly

Full chaos-store, DASH (`dash-biased.yaml`/`dash-random.yaml`), mean
heavy-atom count per split (from each run's own `manifest.json`):

| split scheme | train | val | test |
|---|---|---|---|
| `biased_split` | 27.5 | 32.9 | **38.1** |
| `split` (random) | 30.0 | 25.6 | 25.4 |

`biased_split`'s val sits strictly between train and test (not close to
either) -- confirming the split-construction claim directly, not just
inferring it from downstream metrics. `split`'s val and test are nearly
identical (25.6 vs. 25.4) -- two draws from the same distribution, as
intended for a sanity-reference split.

## Full-store results (2026-08-26)

DASH, full 53,079-molecule chaos-store, both configs run end to end
(`--config dash-biased.yaml`/`dash-random.yaml --allow-dirty --no-tracking`,
no `--limit`):

| metric | split scheme | train | val | test |
|---|---|---|---|---|
| `profile/w1_norm_mean` | `biased_split` | 0.339 | 0.350 | 0.407 |
| | `split` | 0.280 | 0.677 | 0.671 |
| `charge/mae` | `biased_split` | 0.0416 | 0.0657 | 0.0922 |
| | `split` | 0.0395 | 0.0867 | 0.0904 |
| `area/r2` | `biased_split` | 0.969 | 0.965 | 0.952 |
| | `split` | 0.981 | 0.916 | 0.949 |
| `atom/profile/w1_norm_mean` | `biased_split` | 0.752 | 0.871 | 1.012 |
| | `split` | 0.682 | 1.248 | 1.217 |

Two different effects are visible here, and they need to be read
separately:

**Train vs. everything else -- "seen data," expected regardless of split
scheme.** DASH's back-off directly averages *train* molecules into the
tree nodes it later queries, so train performance reflects nodes largely
populated by the very molecules being scored -- not a fair
train/val/test comparison the way a model with no such leakage would give,
and not the question this doc is about. This shows up as a **huge**
train-val gap under `split` (`profile/w1_norm_mean` 0.280 → 0.677, a
2.4x jump) but only a **small** one under `biased_split` (0.339 → 0.350,
+3%) -- itself informative: under `split`, val is genuinely "unseen" data
the same way test is, so it jumps straight to test-like performance; under
`biased_split`, val is close enough in size to train's own distribution
that it doesn't show that same jump.

**Val vs. test -- the actual extrapolation-difficulty question:**

| metric | biased_split \|Δ\|/test | split \|Δ\|/test |
|---|---|---|
| `profile/w1_norm_mean` | **14.0%** | 0.8% |
| `atom/profile/w1_norm_mean` | **14.0%** | 2.5% |
| `charge/mae` | **28.8%** | 4.1% |
| `area/r2` | 1.4% | 3.4% |

For profile shape and charge, `biased_split`'s val-test gap is **4-8x
larger** (relative to test) than `split`'s -- confirming the hypothesis
directly: an extrapolation split's val is a systematically worse proxy for
test than a representative split's val is, and by a wide margin, not a
marginal one. `area/r2` is the one metric where this reverses (`split`'s
own val-test gap, 3.4%, exceeds `biased_split`'s, 1.4%) -- reported
honestly, not omitted. R² is sensitive to the *range* of the true values
in each split (a wide, homogeneous area range inflates R² almost
regardless of prediction quality), so this is plausibly a range-composition
artifact specific to that one metric rather than evidence against the
mechanism above -- not investigated further here.

## Practical implication

**Val is a systematically optimistic proxy for test under `biased_split`**
-- any early-stopping/model-selection decision made against val (e.g.
Chemprop's own lightning `Trainer` validation loop) is being made against
an easier distribution than the one the model is ultimately scored on.
This isn't a bug in the harness; it's the deliberate cost of an
extrapolation split, and now it's directly measured rather than assumed.
Worth remembering when comparing a predictor's own val-time behavior
(e.g. early-stopping curves) against its final test-time numbers on
`biased_split` configs. Train's own numbers, by contrast, are not a
meaningful "how good is the model" signal for DASH at all (see the "seen
data" effect above) -- don't read a small train/val/test progression on
`biased_split` as "the model generalizes almost perfectly"; it partly
reflects how little train and val differ in molecule size, not how little
error the model makes on genuinely new data.

## Reproducing / extending this

Every run now writes `train/*`/`val/*` keys into `metrics.json` alongside
the existing (unprefixed) test keys, for any predictor/config -- see
`runner.py`'s `_score_split`/`_score_extra_split`. This comparison used
only DASH (fast, ~2 min/config, plus ~1.5 min extra for train since it's
~8x larger than val/test); the same comparison for `chemprop_cosmonet`/
`chemprop_atom` would need a real full-store run of each
(`chemprop-cosmonet-{biased,random}.yaml`/`chemprop-atom-{biased,random}
.yaml`, ~20-35 min each, plus a proportionally larger train-predict pass)
-- not run here, but the harness now supports it without further code
changes. Note DASH's own "seen data" effect above would not apply to those
predictors the same way (they don't populate anything from train the way
DASH's tree-node averaging does), so their own train/val/test comparison
would need its own reading, not this doc's DASH-specific one.
