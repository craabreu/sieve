# The COSMO-NET-Paper investigation — engineering history

Historical investigation record. The clone this describes
(`snaserib/COSMO-NET-Paper`) was deleted from disk (21GB, git-ignored)
once the findings below were fully captured in prose — nothing operational
here (training invocations, CSV paths, checkpoint directories) is
reproducible on demand any more. What remains load-bearing is the
architectural evidence: this is the reasoning behind
`predictors/chemprop_cosmonet.py`'s whole reason to exist, reproducing the
*real* checkpoint-verified architecture rather than trusting either the
paper's text or the training script's own printed hyperparameters. The
short version lives in `README.md`'s Predictors section; `pins.toml`'s
`[cosmonet_investigation]` just points here.

## The training run

A real training run was carried out on our own chaos-store data (D-MPNN
training script, `--splitter 4` imposing our biased_split as its CATEGORY
column; full 53079-molecule store, 42459/5287/5333 train/val/test) via the
repo's own `DMPNN-Train-pSigma.py`, GPU-confirmed (cu124 torch, driver
595.71.05, "estes"). Final checkpoint (~4h10m, 100 epochs): Test MAE
0.827, RMSE 1.777, R2 0.970 (raw units) — confirming a real, capable model
trained end-to-end, not just a config-time sanity check. Loss function
(checked from source): plain per-bin MSE, all 51 sigma-profile bins as
independent, equally-weighted regression targets — no distributional loss
(W1) and no shape/location/magnitude split the way DASH Stage A has.

## Correction: MODEL_HPARAMS was never actually applied

Found while investigating `chemprop_cosmonet`'s profile W1 gap.
`MODEL_HPARAMS`'s active block (`hidden_size=51, ffn_num_layers=1,
dropout=0.1`) was **never actually applied**. Checked
`inspect.signature(DMPNNModel.__init__)` against `MODEL_HPARAMS`'s own
keys: only `n_tasks`, `atom_fdim`, and `ffn_activation` are real parameter
names on the installed deepchem version. `dropout`, `message_passing_
steps`, `hidden_size`, and `ffn_num_layers` are not — `DMPNNModel.__init__`
forwards any unmatched kwarg straight to `TorchModel.__init__`
(`super().__init__(model, ..., **kwargs)`), never to the `DMPNN` encoder it
just built two lines above from its own named defaults (`enc_hidden=300`,
`depth=3`, `ffn_layers=3`, `enc_dropout_p=ffn_dropout_p=0.0`).

`message_passing_steps=3` deserves its own callout: unlike the other
three, its silent failure is invisible from the trained model's behavior,
because deepchem's own unrelated default for `depth` (message-passing
steps, its real parameter name) also happens to be 3. The script never
actually passes `message_passing_steps` through *at all* — if the paper's
authors had written `message_passing_steps=5` instead, the real model
would still have silently trained with `depth=3`, and the log would print
"5" regardless. It's not "deepchem correctly interpreting
message_passing_steps=3"; it's deepchem never seeing that key, landing on
`depth=3` by coincidence. The bug is real and identical in kind to the
other three — it just doesn't produce a visible symptom for this one
specific value.

`atom_fdim=35` alone is also ineffective without `use_default_fdim=False`
(deepchem's own docstring: "if True [the default], self.atom_fdim ...
[is] initialized ... from the GraphConvConstants class" — ignoring the
passed value). Confirmed directly against our own trained checkpoint's
real weights: `encoder.W_i.weight` is `(300, 147)` = enc_hidden 300,
atom_fdim 133 + bond_fdim 14 (deepchem's stock RDKit featurization, not
the paper's own 35/12 — Table 1/2); `ffn.linears.0.weight` is `(300,
300)`, i.e. a 3-layer FFN (300→300→300→51), not the claimed single layer.

This also fully explains the "checkpoint doesn't match its own log"
finding without needing a "swap between commits" theory that seemed like
the explanation at the time: every run of this script — ours included, run
independently, months later, from a fresh venv — silently builds this same
deepchem-default-shaped model regardless of what `MODEL_HPARAMS` or the
commented-out "larger, apparently-tuned block" claim; the training log
just echoes `MODEL_HPARAMS`'s dict verbatim, whether or not those values
took effect. The commit-timing observation (`.pt` files added 44 min after
the log/results) was real but not causal — a coincidence, not the
mechanism. The trained model here is therefore deepchem's own default DMPNN
(bigger and richer — 300 hidden units, 3 FFN layers, 133-dim atom
features, no dropout — than either the paper describes or than
`chemprop_cosmonet` faithfully implements), not the paper's claimed
architecture. Its own numbers are real and worth keeping (a real model
really was trained on chaos-store and really does predict
non-negative-mostly profiles), just mischaracterized until this correction
as "the paper's hyperparameters".

## Why the authors' own checkpoint is (300, 49), not (300, 147)

Their shipped `StratifiedCATEGORY_CV5` checkpoint's `encoder.W_i.weight` is
`(300, 49)` = 35 + 14, not our `(300, 147)` = 133 + 14 — both used
`use_default_fdim=True` (never overridden anywhere, confirmed by grep), so
by the mechanism above both should read `GraphConvConstants.ATOM_FDIM`/
`BOND_FDIM` unconditionally, ignoring the `atom_fdim=35` kwarg entirely.
Checked `DMPNNEncoderLayer.__init__` directly:

```python
if use_default_fdim:
    self.atom_fdim = GraphConvConstants.ATOM_FDIM
    self.concat_fdim = GraphConvConstants.ATOM_FDIM + GraphConvConstants.BOND_FDIM
else:
    self.atom_fdim = atom_fdim
    self.concat_fdim = atom_fdim + bond_fdim
```

On our installed (stock, unpatched PyPI) deepchem,
`GraphConvConstants.ATOM_FDIM=133`, `BOND_FDIM=14` (confirmed directly) —
hence our `(300, 147)`. The only way their run got `(300, 49)` through this
same code path is if `GraphConvConstants.ATOM_FDIM` itself equalled 35 in
whatever deepchem they actually used — i.e. they didn't touch the
training script's logic at all, they patched deepchem's own
`dmpnn_featurizer.py` module, hardcoding `GraphConvConstants.ATOM_FDIM =
35` (and presumably narrowing `ATOM_FEATURES['atomic_num']` to their
8-element vocabulary), so that `use_default_fdim=True` silently does the
"right" thing only because they'd redefined what "default" means. This is
exactly GitHub issue #1's own description: "the README says
dmpnn_featurizer.py was patched (ALLOWED_ATOMS = 8 elements, ATOM_FDIM
hardcoded to 35)... the patched file isn't in the repo."

**Corollary**, not previously noted: `BOND_FDIM` stayed at deepchem's stock
14 in *both* checkpoints (49-35=14, 147-133=14) — only the atom side was
ever patched. So even the authors' own shipped, published model never
actually used the paper's own Table 2 (12-dim bond features) either — a
fourth, narrower gap layered on top of the featurizer gap above, not just
"the featurizer file is missing" but "even the file that IS missing only
covers half of what Tables 1-2 together describe."

## The retired lookup predictor

A first, now-retired predictor (`CosmonetPredictor`,
`predictors/cosmonet.py`) was built directly on top of this training run:
a SMILES-keyed lookup against the training script's own saved
`Results/pSigma-{model}.csv`, not live re-inference, reproducing
`profile/w1_norm_mean` 0.224 on chaos-store's `biased_split`. Retired once
`predictors/chemprop_cosmonet.py` (T10) superseded it — a trustworthy,
checkpoint-verified-architecture baseline trained end-to-end on our own
data, with no dependency on the external clone or its saved-CSV lookup
convention. That number is kept here as the historical record of what the
lookup-based approach achieved; it is not reproducible without the deleted
clone and is not a claim about `chemprop_cosmonet`'s own performance (see
`docs/chemprop_cosmonet.md` for that).

Its published training script also had **no** non-negativity enforcement
anywhere in its prediction path (no softplus/clamp/clip between
`model.predict()` and the CSV write), yet its own README claimed a
softplus-patched FFN was used, and its own published
`Results/pSigma-DMPNN.csv` had exactly 0% negative sigma-profile bins
across all 5 folds — the published script cannot reproduce its own repo's
published results. On our own independently-trained checkpoint, the
lookup predictor's real negative-bin rate was 19.6% of test-set bins.
