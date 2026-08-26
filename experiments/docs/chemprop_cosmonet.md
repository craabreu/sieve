# chemprop_cosmonet (T10) — engineering history

Full deep-dive behind `predictors/chemprop_cosmonet.py`'s
`ChempropCosmonetPredictor`. The short version lives in `README.md`'s
Predictors section and `pins.toml`'s `[chemprop_cosmonet]` notes (pin
metadata, a short summary); this file is everything that didn't need to
stay there. Background on *why* the real architecture had to be
reverse-engineered from checkpoint weights in the first place is in
`docs/cosmonet_investigation.md`.

## Origin: reproducing the paper, then reproducing what actually got trained

Originally built to reproduce COSMO-NET's published architecture (Naseri
Boroujeni et al., *Mol. Syst. Des. Eng.*, 2026, DOI 10.1039/d6me00088f)
directly from the paper's Tables 1-2 and Section 2.2.2. Revised once
investigating the COSMO-NET-Paper repo (see `docs/cosmonet_investigation.md`)
showed the checkpoint that repo actually ships was never trained with that
architecture at all — three independently-confirmed reproducibility gaps
made the external repo unsuitable to depend on directly:

1. The shipped `Sigma_saved_model/StratifiedCATEGORY_CV5/` checkpoint's
   weights (`hidden_size=300, ffn_num_layers=3`) contradict the
   hyperparameters printed in its own committed training log
   (`hidden_size=51, ffn_num_layers=1`) — root cause: the training
   script's `MODEL_HPARAMS` keys mostly don't match real
   `DMPNNModel.__init__` parameter names on the installed deepchem, so
   they get silently absorbed into `TorchModel`'s `**kwargs` and never
   reach the encoder — every run, not just this checkpoint, builds
   deepchem's own default-shaped model regardless of what the log echoes
   back.
2. The atom featurizer needed to reproduce that checkpoint (`atom_fdim=35`
   vs. deepchem's stock 133) was never published — independently
   confirmed in that repo's own GitHub issue #1, unanswered since
   2026-08-09; the mechanism is a patched deepchem module
   (`GraphConvConstants.ATOM_FDIM` hardcoded to 35), not a script-level
   flag. Narrower than it first looked, too: `BOND_FDIM` stayed at
   deepchem's stock 14 in their checkpoint as well (49-35=14) — only the
   atom side was ever patched, so even their own shipped model never used
   the paper's own Table 2 (12-dim bond features) either.
3. The published training script has no non-negativity enforcement
   anywhere in its prediction path, yet its own published results CSV has
   exactly 0% negative bins across all 5 folds.

## Dependency resolution

`uv sync --extra chemprop` (a new opt-in extra, into the *main* venv, not
a separate one — unlike the retired lookup baseline's COSMO-NET clone
venv) resolved cleanly on the first try, no conflicts. rdkit stayed at
2026.03.5 (chemprop's own transitive pin `cuik_molmaker_pin==2026.3.5`
requires exactly that rdkit version, and it happened to already match — a
future rdkit bump could require watching this). torch (2.13.0+cu130, CUDA
available) and lightning (2.6.5) were already present transitively via
cosmolayer's `experiments` extra, so this added no new heavy stack, just
chemprop itself plus its own tail (myerson, descriptastorus, astartes,
cuik_molmaker_pin, padelpy, mordredcommunity, aimsim-core, xarray — none of
which this predictor actually uses; they come along with the `chemprop`
import regardless).

## Architecture notes

See `predictors/chemprop_cosmonet.py`'s module docstring for the full
reasoning. Two real gotchas found while building this, beyond ordinary API
mapping:

- chemprop's `MLP.build(n_layers=...)` counts *additional* hidden layers
  beyond the direct input→output projection — the paper's
  "ffn_num_layers=1" (one FFN layer, no hidden layer) is chemprop's
  `n_layers=0`, confirmed by inspecting `MLP.build` directly (`n_layers=0`
  → one Linear; `n_layers=1` → two). Using chemprop's `n_layers=1` would
  silently build a deeper FFN than the paper describes.
- `pl.Trainer(accelerator="auto")` alone silently launched multi-GPU DDP
  on this machine's 2 GPUs, splitting the val/test batch across ranks —
  caught as a metrics shape mismatch in the first end-to-end test run
  (`profile_true` had 10 rows, `profile_pred` had 5, half lost to the
  other rank). Fixed with an explicit `devices=1`.

chemprop's built-in atom featurizer cannot hit the paper's exact 35
dimensions (mandatory "unknown" pad bits push it to 41), so
`PaperAtomFeaturizer` is a custom, duck-typed class (not a
`chemprop.featurizers.base.VectorFeaturizer` subclass, so it stays
importable and unit-testable without chemprop — confirmed empirically
that chemprop accepts plain duck-typed featurizers with no isinstance
check anywhere in the call path). The bond side, after the revision below,
uses chemprop's own built-in `MultiHotBondFeaturizer()` directly — its
14-dim default turns out to be structurally identical to deepchem's real
(never-patched) bond featurization, so no custom class is needed there at
all.

Non-negativity is structural, not a post-hoc clip: softplus sits on the
FFN's raw scaled-space output, before `UnscaleTransform` (min-max, built
manually since chemprop's own `normalize_targets()` is z-score only).
Since min-max scaling's `y_min >= 0` for every profile bin, `softplus(x) >
0` composed with unscaling guarantees `p(sigma) > 0` for every prediction
— verified end to end: the first real fit/predict run through the harness
(`--limit 40, max_epochs=1`) produced zero negative bins.

## First full-store run (paper-faithful architecture)

`hidden_size=51, ffn_n_layers=0, dropout=0.1`, sum pooling, atom/bond=35/12:
1363.5s, `profile/w1_norm_mean` 0.397, `area/r2` 0.731, `charge/mae`
0.0538, 0/271,983 negative bins (0.0000%) verified directly against
`predictions.npz`, vs the retired lookup baseline's measured 19.6%.
Superseded by the revision below — kept here as the record of what
"faithfully implementing the paper's own claimed hyperparameters" actually
produces.

## Revision: targeting the real, verified architecture

Prompted by the user pointing out that an early W1 comparison "makes no
sense" — an initial guess of "optimizer/init differences" turned out to be
wrong; the real explanation follows from `docs/cosmonet_investigation.md`'s
correction — the checkpoint's actual trained model was never running the
paper's claimed architecture at all; it's deepchem's own silently-defaulted
DMPNN. Verified precisely, not by inference:

- `hidden_size`: 51 → 300 (`enc_hidden`'s real default).
- FFN depth: the paper's "ffn_num_layers=1" → deepchem's real
  `ffn_layers=3`. Getting the chemprop-side count right required tracing
  *both* frameworks' layer-counting conventions directly, since they
  disagree by one: deepchem's `PositionwiseFeedForward.__init__` treats
  `n_layers` as the *total* Linear-layer count (confirmed by reading its
  source: `n_layers=3` builds exactly 3 `nn.Linear`); chemprop's own
  `MLP.build(n_layers=N)` instead yields `N+1` total layers (an
  *additional*-hidden-layers count, confirmed separately for the
  original, now-superseded `ffn_num_layers=1` case). So deepchem's real
  `ffn_layers=3` needs chemprop's `n_layers=2`, not 3 and not the original
  0. Locked in by a dedicated test
  (`test_default_ffn_has_the_deepchem_equivalent_layer_count`) that counts
  actual `nn.Linear` submodules rather than trusting either framework's
  parameter name.
- dropout: 0.1 → 0.0 (`enc_dropout_p`/`ffn_dropout_p` both default to
  0.0).
- Bond features: the paper's own 12-dim Table 2 → deepchem's real,
  never-patched 14-dim stock featurization — structurally identical to
  chemprop's own built-in `MultiHotBondFeaturizer()` default, so that's
  used directly now instead of a custom `PaperBondFeaturizer` (removed).
- Readout: sum → mean pooling. Not just an ignored kwarg landing back on a
  default that happens not to matter (like `depth=3`, which coincidentally
  matches the paper's own intent) — `aggregation` isn't in `MODEL_HPARAMS`
  at all (grepped, absent), so this is the real model doing something the
  paper's own eqn 11 explicitly says it doesn't do.
- Atom features (35-dim, Table 1) are unchanged — this is the one part of
  the paper's claimed architecture that genuinely was realized (patched
  into deepchem's own `GraphConvConstants.ATOM_FDIM`), so
  `PaperAtomFeaturizer` stays exactly as it was.
- Not replicated: the real run's optimizer/LR schedule
  (`ExponentialDecay(initial_rate=0.001, decay_rate=0.95,
  decay_steps=1000)`, set directly in the script rather than via the
  broken `MODEL_HPARAMS` dict, so genuinely applied) vs. chemprop's own
  default Adam + Noam-like warmup. Would need subclassing
  `MPNN.configure_optimizers` — a documented, deliberately-deferred
  remaining deviation.

New architecture: 401K params (vs. the original's 12.1K) — confirmed via
lightning's own model summary table at fit time. A `--limit 5000` timing
probe (100 real epochs): 122.3s — barely different from the original,
much smaller model's 127.8s at the same scale (small-batch GPU training
here is overhead-bound, not compute-bound, so 33× more parameters barely
moves wall time), and already promising: `profile/w1_norm_mean` 0.301 at
this scale vs. the original architecture's 0.577.

**Full-store run (biased_split, 100 epochs), revised architecture:**
1337.8s (22.3 min, matching the probe's extrapolation almost exactly).
`profile/w1_norm_mean` 0.2194, `area/r2` 0.828, `charge/mae` 0.0201,
0/271,983 negative bins (0.0000%) verified directly against
`predictions.npz` — structural non-negativity holds regardless of
architecture size, exactly as expected (softplus+min-max unscaling
guarantees it independent of hidden_size/depth/dropout).

This predictor now beats the retired lookup baseline on both profile shape
(0.2194 vs. 0.224) and area (0.828 vs. 0.775) — unsurprising once the
architecture correction is understood: it now runs the same real
architecture the checkpoint accidentally used, built cleanly (proper
softplus non-negativity, proper min-max scaling from train-set stats only,
no silently-broken kwargs) rather than by accident. The comparison that
"made no sense" is fully resolved: they were never comparable before
(different real architectures under an identical-looking config), and now
they are (same real architecture, this one built deliberately).

## Loss-mode experiment

The user asked to test two alternative training losses against the
default per-bin MSE — `w1_normalized` (Wasserstein-1 on each profile
normalized by its own row sum, a pure shape loss) and `mse_cumsum` (MSE on
the profiles' raw cumulative sums, a joint shape+magnitude loss) — see
`predictors/chemprop_cosmonet.py`'s module docstring for both formulas and
the `_UnscalingRowLoss` base class (both need real, unscaled units to mean
anything physically, since per-bin min-max scaling makes a row-sum/cumsum
on the *scaled* tensor meaningless).

A `--limit 5000` probe for both, then a full-store run for `w1_normalized`
only (`mse_cumsum`'s probe result didn't justify the ~22 min full-store
cost — user's call, not run):

| loss_mode | scale | w1_norm_mean | area/r2 | charge/mae |
|---|---|---|---|---|
| mse (default) | 5000 | 0.301 | -0.157 | -- |
| w1_normalized | 5000 | 0.266 | not meaningful* | not meaningful* |
| mse_cumsum | 5000 (probe only) | 0.564 | 0.058 | -- |
| mse (default) | full store | 0.2194 | 0.828 | 0.0201 |
| w1_normalized | full store | **0.2071** | not meaningful* | not meaningful* |

(*) `area/r2` and `charge/mae` are not reported as numbers for
`w1_normalized`, deliberately — quoting -1822/-243 (the raw values this
run actually produced) would misrepresent what happened. This loss gives
the network literally zero gradient for overall scale (both prediction
and target are divided by their own row sum before the loss ever sees
them), so area was never something the model was trying and failing to
predict — it's an untrained, free-floating byproduct of whatever scale the
optimizer happened to wander to, not a real predictive failure on a real
target. Reporting an R²/MAE on it invites exactly the wrong reading (that
the model attempted magnitude and got it badly wrong), so it's correctly
described here as "not meaningful," not scored. (For the record: full-store
predicted areas landed in 1250-5437 vs. a true range of 114-520 —
illustrating *why* it's unscored, not a metric to cite.)

`w1_normalized`'s full-store profile shape is the best of any config tried
in this whole milestone (0.2071, beating even the default-loss run's
0.2194). 0/271,983 negative bins confirmed on this run too — non-negativity
is architecture-level (softplus), not loss-level, and holds regardless of
which of the three loss_modes is used (also covered by a dedicated
parametrized optional test).

`mse_cumsum`'s probe-scale numbers (`w1_norm_mean` 0.564, worse shape than
both alternatives; `area/r2` 0.058, better than plain mse's -0.157 at the
same scale but a marginal gain) didn't justify a full run. Recorded as a
probe-only result, not a full-store one.

**Takeaway**: neither alternative loss is a straightforward improvement
over the default MSE for this architecture/task. `w1_normalized` would
need an explicit magnitude term (e.g. combined with an area MSE, mirroring
`SigmaProfileLoss`'s shape+area+charge combination) to be usable on its
own; `mse_cumsum`'s cost (worse shape) wasn't worth its benefit (marginal
area) even at probe scale. Kept as documented `loss_mode` options on
`ChempropCosmonetPredictor` for future experimentation, not adopted as a
new default — `chemprop-cosmonet-biased.yaml`/`chemprop-cosmonet-random.yaml`
stay on `loss_mode="mse"`.
