# Design: curating anomalous MBIS records in the DASH-charges store

**Status:** draft, awaiting review
**Date:** 2026-09-11
**Scope:** one annotation stage in `experiments/experiments/prepare_dash.py`, two
new columns in `molecules.parquet`, one opt-in config key. No change to any
predictor, to `sieve`, or to how metrics are computed.

## Problem

A small number of conformers in `dashMoleculesSDF_v2.sdf` carry MBIS charges that
are physically impossible. The clearest case, `CHEMBL1303779` / `QMUGS500_92125`:

| conformer | atom 17 (sp2 aromatic CH) | whole-molecule agreement |
|---|---|---|
| conf_00 | −0.124 | — |
| **conf_01** | **+2.975** | 34 of 37 atoms differ from siblings by >0.05 e |
| conf_02 | −0.130 | — |

The underlying quantum chemistry is healthy: across the three conformers the DFT
total energy agrees to 1 mHartree, the HOMO–LUMO gap to three decimals, and the
DFT Mulliken, DFT Löwdin and GFN2 Mulliken charges on that same atom read
−0.006/+0.011/−0.006, +0.003/+0.008/+0.003 and −0.023/−0.020/−0.023. Only the
MBIS partition is wrong, and only for one conformer. All three still sum to the
formal charge to 1e-4, so the sum rule does not detect it.

Prevalence on the test split (103,033 conformers), by deviation from the sibling
median: **2.12%** above 0.1 e, **0.375%** above 0.3 e, **0.137%** above 0.5 e.
Failures occur in both source pipelines, about 2.3× more often in the
QMugs-derived half.

**Impact.** Excluding conformers above 0.5 e changes the full-corpus Sieve
depth-6 test metrics by: MAE 0.011544 → 0.011406 (−1.2%), RMSE 0.023259 →
0.021581 (−7.2%), **excess kurtosis 476 → 60**, max error 3.09 → 1.14 e. So the
headline accuracy barely moves, while every squared-error and distribution-shape
statistic moves a lot. Any claim about the error distribution computed on the
uncurated store is dominated by ~100 records.

## The criterion

For each molecule, take the per-atom median charge across its conformers; flag a
conformer if any atom deviates from it by more than a threshold.

```
key   = dash_id                                   # NOT chembl_id
med   = median over conformers of the same key    # per atom
score = max_i |q_i - med_i|                       # per conformer, in e
flag  = score > 0.3
```

**`dash_id`, not `chembl_id`.** 49.6% of the corpus (511,116 of 1,029,785
conformers) has no `CHEMBL_ID` — those are the `Rest_*` records, for which
`parse_record` synthesizes `conf_id` counted per `dash_id`. `dash_id` is present
on every record, yields 348,935 molecules at 2–3 conformers each, and never spans
two `chembl_id`s. Keying on `chembl_id` silently drops half the corpus and, in
the analysis that produced this spec, made the measured prevalence look 43%
lower at the 0.3 e threshold (221 records instead of 386). `partition_store`
already keys molecule grouping on `chembl_id.fillna(dash_id)`, which is why the
existing splits have no leakage.

**Threshold 0.3 e.** The null distribution is extraordinarily tight: the robust
scale of per-atom sibling deviation is ~0.001 e for C, N, O, H and the halogens,
and 0.0058 e for P. A 0.3 e cut therefore sits 200–300 robust scales into an
empty tail — it is conservative by construction rather than tuned to a decision
boundary. The threshold is also readable as chemistry: *an atom's charge should not
shift by more than three tenths of an electron between conformers of the same
molecule.*

**Every molecule in this store has ≥2 conformers** (minimum 2, median 3), so the
criterion applies to every record; there is no singleton fallback to specify.

## Posture: flag, never drop

Records are never removed. The store gains a score and a flag; each experiment
decides what to do with them, and the default is to use everything.

Rationale: the threshold stays tunable after the fact, a sensitivity analysis
costs nothing, results already on disk stay comparable, and published numbers on
the DASH-charges dataset remain comparable with anyone else's. Dropping at parse
time would bake the threshold into a 9.6 GB artifact and make every existing run
non-comparable.

## Schema

`molecules.parquet` gains two columns, and the fold stores inherit them
automatically — `partition_store` reads the whole frame and writes row subsets,
so it carries unknown columns through without modification.

| column | type | meaning |
|---|---|---|
| `mbis_sibling_dev` | float64 | `max_i \|q_i − median_over_siblings(q_i)\|`, in e |
| `mbis_anomaly_flag` | bool | `mbis_sibling_dev > 0.3` |

Storing the score, not only the verdict, is what keeps the threshold revisable
and the sensitivity analysis free.

## Where it runs

A new stage in `prepare_store`, after the split stage, with its own guard:

```
download → parse → split → annotate
```

Each stage already skips when its output exists; `annotate` follows the same
pattern (skip when both columns are present). It cannot live in `parse_record`,
which is streaming and sees one record at a time, whereas the criterion needs all
conformers of a molecule at once. It is a pass over the parsed store, not over
the 7.8 GB SDF. Runtime on the full corpus has not been measured; the
equivalent pass over the 103k-conformer
test split takes ~1 minute, dominated by deserializing each conformer's `Mol`,
so expect ~10 minutes for 1.03M — worth confirming before it is called cheap.

`prepare_store` currently returns early once the split column exists; that early
return becomes a guarded stage so the annotation stage can follow it.

Existing stores are annotated by re-running `prepare-store`, which will skip
download, parse and split and execute only the new stage. Fold stores must then
be rebuilt, or annotated by the same pass — the spec prefers rebuilding, since
`partition_store` is already idempotent and deterministic at seed 0.

## How experiments consume it

One optional config key, default `false`, so nothing changes unless asked:

```yaml
data:
  exclude_anomalous: false   # drop rows with mbis_anomaly_flag before use
```

When true, rows are filtered in `load_molecule_set`, before the split masks are
applied, so it affects train, val and eval consistently. `metrics.json` records
the resolved value (it rides in the config, so `to_flat_params` logs it for
free), and the run manifest therefore says which population a number refers to.

**Recommended usage, and it differs by purpose.** Headline accuracy metrics:
report on the unfiltered store, for comparability. Any analysis of error
distribution, tails, or uncertainty calibration: filter, because the unfiltered
statistic is dominated by ~100 corrupted records. RMSE-based comparisons: report
both, since the exclusion moves Sieve's RMSE by 7.2% and DASH's by 4.5% and thus
changes the reported gap.

## What the criterion was validated against

The flag is defined from the reference data alone, never from model error. That
is what makes post-hoc curation defensible rather than circular, and it is worth
one sentence in any paper that uses it. The validations below were run *after*
the criterion was fixed.

**Independent quantum-chemical cross-check.** Both source pipelines carry GFN2-xTB
Mulliken charges (`XTB_MulikenCharge` in `Rest`, `GFN2:MULLIKEN_CHARGES` in
QMugs — verified to be the same quantity on a molecule present in both sources:
agreement 0.075 e max, while the two records' own `MBIScharge` fields differ by
0.017 e). Regressing MBIS on that reference per DASH-tuple class and taking the
maximum robust residual gives an anomaly detector with AUC 0.9986 against the
sibling flag. Conversely, at that detector's ~80%-precision operating point, the
0.3 e sibling rule recovers **89.4%** of what it finds. Only **9 conformers in
103,033 (0.009%)** are strongly flagged by the cross-check while their siblings
agree — the criterion's structural blind spot, measured rather than assumed.

**Model error, as corroboration only.** Sieve's own max per-conformer error
separates the groups cleanly: median 0.056 e on ordinary records, 0.358 e on
sibling-flagged records the cross-check calls clean, 0.481 e on records both
methods flag.

**False positives are real chemistry, and acceptable.** Of 386 records flagged at
0.3 e, 306 (79.3%) are confirmed by the cross-check. The remaining ~80 have
unremarkable charge magnitudes but twice the corpus-median dipole change between
conformers — molecules whose charge distribution genuinely varies with geometry.
Excluding them costs 0.08% of data and removes records a graph-only model cannot
predict anyway. Retaining them would leave corrupted labels in the metric. The
spec accepts them.

## Alternatives rejected, with measurements

Every alternative below was implemented and scored against the same independent
label; all are rejected in favour of the plain rule.

| alternative | result |
|---|---|
| Cross-scheme z as the *primary* criterion | AUC 0.9986, the best detector — but needs a 7.8 GB SDF re-parse, ~38 per-class calibration constants, and a second data source. Kept as validation, not as the criterion. |
| Dipole difference ‖Σ Δqᵢ(rᵢ−r̄)‖ | AUC 0.9556, *worse* than the per-atom cross-check; ANDing it with stage 1 cut recall to 73%. A global summary is blind to a large error near the centroid. |
| Dipole *ratio* \|μ_MBIS\|/\|μ_xTB\| | Rejected: normal records reach 5.18 against the anomaly's 5.79. Ratios blow up for near-zero denominators. |
| Per-element charge z (intrinsic) | AUC 0.769 with element classes; 0.900 with DASH-tuple classes and mean aggregation. Useful only if a future store lacks auxiliary charges. |
| Per-bond-type charge jump | AUC 0.867 after per-bond-type normalization (0.751 raw). Real polar bonds reach 2.5 e, which swamps the signal. |
| Per-element normalization of the sibling deviation | AUC 0.9623 vs 0.9605 raw; at matched budget identical at the operating point. The null spread is element-independent, so there is nothing to normalize. |
| `q_i / (q_i + Σ_neighbours q_j)` | AUC 0.655. The denominator is a nearly neutral quantity — below 0.05 for 14.7% of atoms — so the ratio ranges over ±10³. |
| `q_i / (\|q_i\| + Σ\|q_j\|)` | AUC 0.9599 but only 44% recall at matched budget: the anomaly inflates the denominator, cancelling itself. |
| Model prediction error | AUC 0.9837, but circular for evaluating those same models, and Sieve and DASH are both graph-only, so they share exactly the blind spot that produces the false-positive class. |

## Testing

Unit tests in `experiments/tests/`, following the existing `test_store_ops.py`
patterns, on small synthetic frames rather than the real store:

1. A molecule whose conformers agree scores ~0 and is not flagged.
2. A molecule with one perturbed conformer flags exactly that conformer, not its
   siblings.
3. Grouping uses `dash_id`: records sharing a `dash_id` but with `chembl_id` null
   are grouped together, and a `chembl_id`-keyed grouping would not.
4. The annotate stage is idempotent — running twice leaves the columns unchanged
   and does not re-derive them.
5. `partition_store` propagates both columns into the fold stores.
6. `exclude_anomalous: true` removes exactly the flagged rows and leaves the
   split proportions otherwise untouched.

Verification beyond unit tests: re-run the annotation on the real store and
confirm the corpus-wide counts match those recorded here (2.12% / 0.375% /
0.137% at 0.1 / 0.3 / 0.5 e on the test split).

## Out of scope

- Dropping records, at parse time or anywhere else.
- Re-running any completed experiment. The existing runs stay as they are; the
  flag exists so future analyses can choose.
- The cross-scheme, dipole and intrinsic detectors as production code. They were
  built to validate the criterion and stay as throwaway analysis scripts.
- Reporting the anomalies upstream to the DASH authors. Worth doing, and the
  extracted SDFs exist, but it is not a code change.

## Open questions

1. **Threshold.** 0.3 e is recommended; 0.5 e halves the false positives and
   drops recall to 79.8%. Since the score is stored, this is revisable without
   re-deriving anything, and a sensitivity curve over 0.1–1.0 e belongs in any
   paper that uses the flag.
2. **Fold stores.** Rebuild them after annotation, or annotate them in place?
   Rebuilding is simpler and deterministic; annotating in place avoids
   invalidating the ~1.8 GB of depth-6 shards keyed to the current folds.
3. **Whether `exclude_anomalous` should default to `true` for new experiment
   series** once the flag exists, with the current default kept only for
   reproducing existing runs.
