# chemprop_atom (T11) — engineering history

Full deep-dive behind `predictors/chemprop_atom.py`. The short version
lives in `README.md`'s Predictors section; `pins.toml`'s
`[chemprop_cosmonet]` (chemprop_atom shares its pin — same package,
same version) just points here. United-atom-store results are in
`docs/chaos_store_ua.md`.

## What it is

`predictors/chemprop_atom.py` — a `MolAtomBondMPNN` predicting one 51-bin
profile per **atom**, off that atom's own message-passing hidden state,
with `agg=None` and no molecule/bond head at all. Molecule-level numbers
come from the harness's own `roll_up` of those atom predictions, so they
are a pure readout of atom quality rather than a separately-trained head.
This is the second atom-level predictor after DASH (T8), and the
comparison against it is the point: learned per-atom vs.
averaged-over-a-published-tree.

Otherwise it follows `chemprop_cosmonet`'s recipe exactly — raw unshifted,
unnormalized bins, per-bin min-max scaling fit on the training split only,
plain MSE — so this-vs-`chemprop_cosmonet` isolates the atom-vs-molecule
head and this-vs-DASH isolates learned-vs-averaged.

## GOTCHA 1 (the big one): softplus does not transfer to the atom level

`chemprop_cosmonet`'s softplus output collapses completely here. A
softplus atom model predicts identically zero and cannot overfit even 20
molecules. Mechanism, and it is purely structural, not a tuning problem:
`softplus(x) = 0` only as `x → -inf`, and atom profiles are **81.4% exact
zeros** (each atom occupies only ~9.5 of the 51 bins; a molecule profile
is far denser — 50.2% zeros, 25.4 bins populated). MSE therefore drives
the majority of outputs toward exactly zero, pushing pre-activations to
-inf, which is precisely where softplus's own derivative `sigmoid(x)`
vanishes — the head dies and never recovers. `chemprop_cosmonet` never hit
this because molecule-level targets are dense and never need the output
driven to exact zero. Measured, 20-molecule overfit, true mean atom area
7.676:

| output activation | w1 | predicted area | negative bins |
|---|---|---|---|
| softplus (chemprop_cosmonet's) | 6.717 | 0.000 | 0 (dead) |
| **squared (x²)** | **0.878** | **7.859** | **0** |
| abs | 0.947 | 7.939 | 0 |
| plain linear | 0.825 | 7.786 | 13110 |

`x²` is the adopted default: equally structurally non-negative (the
activation still sits on the raw output, before unscaling, and every
profile bin's train min is exactly 0 so `y_min >= 0`), but a target of
exactly zero is reachable at the finite point `x = 0`, so no gradient
dies. It costs essentially nothing against an unconstrained linear head
(0.878 vs. 0.825) which would emit 13k negative bins and forfeit the
structural guarantee that was `chemprop_cosmonet`'s whole selling point.
Selectable via `output_activation`; `"softplus"` stays available so the
collapse above stays reproducible. The FFN factory is shared
(`chemprop_cosmonet.nonneg_regression_ffn_class`) so the
activation-before-unscale invariant is defined exactly once.

## GOTCHA 2: atom featurizer

`chemprop_cosmonet`'s `PaperAtomFeaturizer` (the paper's Table 1, 8
elements {C,N,O,S,F,P,Cl,Br}) is unusable per-atom — it has **no
hydrogen**, and 56.8% of chaos-store atoms are hydrogen once the graph
keeps explicit H (which it must, since the store has atom-level truth for
every atom). `chemprop_atom` uses chemprop's own
`MultiHotAtomFeaturizer.v2()` (72-dim, Z=1-36 plus 53), covering every
chaos-store element except Sb and Te.

## GOTCHA 3: atom ordering — the silent-corruption risk

Chemprop's own `make_mol(smi, keep_h=True, add_h=False,
reorder_atoms=True)` renumbers atoms by atom-map number, which *is* the
store's own flat atom order, so **no manual permutation is needed** —
unlike `predictors/dash.py`, which does its own
`np.argsort(GetAtomMapNum())`. Verified but never assumed. Dropping
`reorder_atoms=True` changes the element sequence for 38 of 40 test
molecules and mismatches 16.0% of atoms, so the guard genuinely
discriminates. Note also that `CuikmolmakerDataset` must *not* be used: it
ignores `_reorder_atoms` and featurizes in raw textual order (a known
chemprop bug), which would silently misalign every atom-level target.

Guarded three ways: a per-molecule count + map-order assert, a test
against the store's own `element` column, and a reversed-molecule-order
prediction test.

## GOTCHA 4: a shape check cannot catch a shuffled predict loader

`build_dataloader` shuffles *molecules*, and each molecule's atoms travel
with it as a contiguous block — so a shuffled loader returns the exact
same `(n_atoms, 51)` array shape, just with molecule blocks permuted. The
guard is therefore a per-molecule atom-count sequence check recovered from
each batch's own `bmg.batch`, not an assert on the array shape.

## Coverage

**100% atom coverage**, unlike DASH (~1.0% of test atoms fall back to the
global mean — see `docs/dash.md`'s coverage caveat). The comparison mildly
favours `chemprop_atom` for that reason.

## Full-store results

`--config configs/chemprop-atom-biased.yaml`, no `--limit`, `n_test` 5333
molecules / 203,063 atoms. `time/fit_s` 2048s (34 min, matching the
`--limit 5000` probe's extrapolation), `time/predict_s` 6.5s.
**0/271,983 rolled-up bins negative**; predicted molecule areas span
103-551 against a true 114-520.

| metric | DASH decomposed (retired) | DASH raw (retired) | **chemprop_atom** | chemprop_cosmonet |
|---|---|---|---|---|
| atom/profile/w1_norm_mean | **1.030** | 1.058 | 1.055 | -- (molecule-level) |
| atom/area/r2 | 0.945 | 0.945 | **0.956** | -- |
| atom/charge/mae | 0.00752 | 0.00752 | **0.00726** | -- |
| profile/w1_norm_mean | 0.449 | 0.442 | 0.380 | **0.219** |
| area/r2 | **0.949** | 0.949 | 0.943 | 0.828 |
| charge/mae | 0.102 | 0.102 | 0.0792 | **0.0201** |

Reading these honestly:

- **`chemprop_atom` beats DASH on atom area and atom charge**, and is
  essentially tied with it on atom *shape* (1.055 vs. 1.030 — DASH
  decomposed is marginally ahead, DASH raw marginally behind). A learned
  per-atom model does not obviously beat averaging over DASH's published
  tree at the atom level, which is a more interesting result than if it
  had.
- **But it's clearly ahead once rolled up to molecules** (`profile/
  w1_norm_mean` 0.380 vs. DASH's 0.449/0.442) — the same atom-level-vs-
  molecule-level decoupling DASH's own profile-mode experiment found (see
  `docs/dash.md`): the two granularities' shape errors do not move
  together, because per-atom errors can cancel or compound when summed
  into a molecule.
- **`chemprop_cosmonet` still owns molecule-level shape and charge**
  (0.219, 0.0201) by a wide margin — unsurprising, since it optimizes the
  molecule profile directly, whereas `chemprop_atom`'s molecule numbers
  are an unoptimized by-product of summing atoms. `chemprop_atom` owns
  area (0.943 vs. 0.828). So the atom-level head buys area accuracy and
  per-atom detail at a real cost in molecule-level shape; it does not
  dominate `chemprop_cosmonet`, and shouldn't be reported as if it did.

## United-atom store

See `docs/chaos_store_ua.md`'s "T11-UA" section for the full united-atom
comparison — `chemprop_atom` genuinely improves on the united-atom store,
at both granularities and on nearly every metric, unlike DASH.
