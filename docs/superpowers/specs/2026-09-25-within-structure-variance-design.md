# Within-Structure Variance in the Predictive Variance

**Date:** 2026-09-25
**Status:** design, awaiting review; not implemented
**Revises:** `design-update-v2.md` §3 (the three-term predictive variance) by one added
term and one changed constant. The three terms and their estimators are kept.

---

## 1. Why

`uncertainty.predictive_variance` was calibrated before the store rebuild, the current
curation rule, conformer collapse and continuation + EB. On today's incumbent (element,
continuation + EB, depth 5, Study B repeat 0) it is five times overconfident at the
deepest matched radius:

| k\* | share of atoms | E[z²], current |
|---:|---:|---:|
| 1–2 | 4% | 0.87–0.95 |
| 3 | 13% | 1.14 |
| 4 | 19% | 1.86 |
| 5 | 63% | 5.06 |

Overall E[z²] is 3.74, and coverage at a nominal 95% is 82.8%.

**Cause: collapse.** Training fits one row per collapse key, carrying the key's mean
target, so class statistics contain no conformer-to-conformer spread. Held-out atoms are
individual conformers, and they do. The missing spread is the **within-structure
variance** σ²_w: the expected squared deviation of one conformer's atomic charge from
its structure's mean over conformers and symmetry-equivalent atoms. It mixes three
things: genuine conformational dependence, symmetry breaking by geometry, and
single-conformer anomalies in the reference.

**Evidence** (2026-09-25, all on train-split CV folds; the test split was never
touched):

- Fitting σ² = φ(k\*) + λ(k\*)·s² to held-out squared errors gives φ(5) = 1.12 × 10⁻⁴,
  against a within-structure variance of 1.01 × 10⁻⁴ from the store's floor cache: a
  ratio of 1.11. The error floor at depth is σ²_w.
- The class mean's standard error s/√N ranks errors worse than s at every radius, and
  adds nothing to s² in a regression. The limiting factor is not the precision of the
  class mean.
- **The fix, "form B" = current variance + σ²_w**, rotated over the five folds of repeat 0
  (chosen on four, scored on the fifth):

| weights | NLL | E[z²] | coverage 50 / 95 / 99% | normalised RMSE | normalised MAE |
|---|---:|---:|---|---:|---:|
| current | −2.046 | 3.73 | 0.459 / 0.830 / 0.894 | 0.018986 | 0.010684 |
| **form B** | −2.831 | 0.97 | 0.665 / 0.955 / 0.981 | 0.018875 | 0.010630 |
| equal weights | | | | 0.019573 | 0.010929 |

  Form B beats the current variance in normalised RMSE in 5/5 folds (mean −1.11 × 10⁻⁴,
  −0.6%), and equal weights in 5/5 (−7.0 × 10⁻⁴, −3.6%). Normalisation weights the
  molecular charge residual by σ², so only ratios within a molecule matter there.
- **Re-tuning** on the other four folds picked α^v = 10 and α^t = 1 in every fold, by both
  NLL and normalised RMSE, and a = 0.5 by NLL. The gain over α^v = 30 is small (NLL −0.006,
  normalised RMSE −0.03%) but consistent.
- **Ablation:** the selection term is required (without it, shallow radii are 25–90%
  under-dispersed). The mean-estimation term is nearly redundant, and stays (§6).

## 2. What changes

1. **The predictive variance** becomes form B, the current three terms plus σ²_w:

   σ²ᵢ = within-class(shrunk, α^v) + a·selection(α^t) + mean-estimation + σ²_w

2. **`uncertainty.ALPHA_V`** changes from 30 to 10. `ALPHA_T = 1` and
   `SELECTION_WEIGHT = 0.5` are unchanged.
3. **σ²_w comes from the fit itself** (phase 1), carried by the collapsed molecules, and
   optionally per class (phase 2, adopted only if measured better).

Nothing changes for a model that carries no within-structure statistics: it adds σ²_w = 0,
which is today's behaviour apart from the α^v default.

## 3. Phase 1: pooled σ²_w

### 3.1 Collapse attaches the numbers

`experiments.collapse.collapse_molecule_set` already aligns each key's conformers
(`aligned_values`) and takes orbit means (`orbit_means`). It additionally writes two atom
properties on the representative, beside the target:

- `<atom_property>__within_sse`: Σ over members *m* of (v[m, i] − orbit_mean[i])², for
  reference atom *i*;
- `<atom_property>__within_n`: the number of members.

Summed over atoms and keys, these reproduce `collapse.floor_components`' `sse` and
`n_atoms` exactly: the same deviations from the same orbit means. Under
`weight_by_collapse`, the representative is repeated `n` times and carries `sse/n` and
`1` per copy, so the sums stay invariant.

### 3.2 The adapter carries them

`from_rdkit(..., within_from_atom_prop=None)` names the target property. When set, the
adapter reads the two companion properties into new optional `NodeBatch` fields:

- `within_sse`, shape `(n_nodes, d)`, float64;
- `within_n`, shape `(n_nodes,)`, float64.

They are validated (finite, non-negative, `within_n ≥ 1` wherever the SSE is positive) and
carried through `__getitem__` and `concat_batches` like `y`.

### 3.3 The model stores two additive numbers

`SieveModel` gains `within_sse: np.ndarray | None` (`(d,)`) and `within_n: float`,
defaulting to `None` and `0`. `fit` sums the batch's arrays when present. `merge_models`
adds them. `save`/`load` write them under a `within` key only when present, so existing
files load unchanged and write back unchanged.

**σ²_w = within_sse / within_n**, per target dimension, and 0 when `within_n == 0`.

These are statistics, not configuration: `schema_version` does not change, and models with
and without them merge (the sums simply cover the rows that carried them).

### 3.4 Attaching them to an existing model

`SieveModel.with_within_structure(sse, n)` returns a copy carrying the given sums. The
experiments use it to give the cached fold models, which were fitted without them, the
pooled value from the floor cache of each fold's **training** shards. No refit is needed for
phase 1.

### 3.5 The predictive variance reads it

`uncertainty.predictive_variance(model, ...)` adds σ²_w to every class's variance, and to
the global fallback for unmatched atoms. `SievePredictor`'s normalisers already read
`predictive_variance` when the config enables it, so normalisation inherits form B with no
further change.

## 4. Phase 2: per-class σ²_w (an experiment, not a commitment)

`fit_level` accumulates `within_sse` and `within_n` per class, with the same membership as
`y`: blind labels for every atom, and aware labels for aware-only classes, as design D
does. `merge_level` adds them. The predictive variance uses

σ²_w,c = (within_sse_c + β·σ²_w) / (within_n_c + β),

shrunk toward the pooled σ²_w, with β fixed by the same rotation over repeat 0.

**Adopted only if** it beats pooled σ²_w out of fold on both NLL and normalised RMSE.
Otherwise it is not merged. Phase 2 requires refitting shards, one arm (the incumbent)
first.

## 5. Tests

Written before the code.

1. **Collapse:** for a small multi-conformer set, the attached per-atom SSE and counts sum
   to `floor_components`' `sse` and `n_atoms`; under `weight_by_collapse`, the sums are
   unchanged.
2. **Adapter and batch:** the arrays are read, validated (a negative SSE is refused),
   sliced and concatenated.
3. **Model:** `fit` sums them; `merge` of two shards equals fitting their union (the merge
   monoid); `save`/`load` round-trip them; a file without them loads with `within_n == 0`;
   `schema_version` is identical with and without them.
4. **Predictive variance:** with σ²_w = 0 it is bit-identical to today's at the new α^v;
   with σ²_w > 0 it is today's plus σ²_w exactly, including the global fallback.
5. **`with_within_structure`:** it sets the sums, and changes neither predictions nor the
   digest.
6. **Reproduction:** on repeat 0 folds 0 and 1, the library's form B reproduces the
   analysis numbers above (E[z²] and normalised RMSE to four figures).

## 6. Not in scope

- **Dropping the mean-estimation term.** The ablation found it nearly redundant, but
  removing it is a separate simplification, measured separately.
- **Intervals.** Form B calibrates the variance, not the distribution: coverage is 66% at a
  nominal 50% and 98% at a nominal 99%, because the errors are heavy-tailed (excess kurtosis
  58). Intervals, if wanted, need conformal calibration by k\*, a separate design.
- **Reported normalised metrics.** The studies score unnormalised predictions; whether to
  report normalised charges is a paper decision.
