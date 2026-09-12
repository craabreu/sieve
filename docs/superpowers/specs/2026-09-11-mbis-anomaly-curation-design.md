# Design: curating anomalous MBIS records in the DASH-charges store

**Status:** draft, awaiting review
**Date:** 2026-09-11
**Scope:** one statistic retained during parsing plus one new annotation stage,
both in `experiments/experiments/prepare_dash.py`; three new columns in
`molecules.parquet`; one opt-in config key. No change to any predictor, to
`sieve`, or to how metrics are computed.

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

### The mechanism is documented, not hypothetical

MBIS is an *iterative* stockholder partition: the pro-atom parameters are refined
to self-consistency [Verstraelen2016MBIS]. Its reference implementation stops
when the largest change in any charge between iterations falls below `threshold`
(default 1e-06) or when `maxiter` (default 500) is reached, and the HORTON
documentation for the method states plainly that **"if no convergence is reached
in the end, no warning is given"**
(<https://theochem.github.io/horton/2.1.1/lib/mod_horton_part_mbis.html>). A
record that hits the iteration cap is therefore written out with the last
iteration's charges, indistinguishable from a converged one. MBIS is also known
not to converge uniquely, unlike Hirshfeld-I.

That also explains the one observation that otherwise looks contradictory — the
sum rule holding to 1e-4 on a record where 34 of 37 atoms are wrong. This part is
inference from the method's structure rather than a quoted result: a stockholder
scheme divides the density by pro-atom weights that sum to 1 at *every* iteration,
so the total charge is conserved by construction whether or not the parameters
have converged. Non-convergence corrupts how the density is split *between*
atoms while leaving the total exact.

The QMugs half of the corpus was computed with Psi4 at ωB97X-D/def2-SVP
[Isert2022QMugs]. Psi4 exposes `MBIS_MAXITER` and `MBIS_D_CONVERGENCE`, and also
computes per-atom valence widths and volume ratios — quantities that would
diagnose a failed partition directly. None of them survive into the distributed
SDF, which carries only the resulting charges.

**So this spec reconstructs, from published outputs, a convergence flag the
producing pipeline had and discarded.** That is worth stating because it explains
both why every detector we built tops out short of perfect and where the real fix
belongs: upstream, in a pipeline that records whether each MBIS solve converged.

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

Two signals. The sibling comparison **screens**; an independent cross-check
**confirms**; exclusion requires both.

```
# screen -- needs only the stored charges
key    = dash_id                                  # NOT chembl_id
med    = median over conformers of the same key   # per atom
sib    = max_i |q_i - med_i|                      # per conformer, in e

# confirm -- needs the auxiliary GFN2-xTB charges the SDF already carries
xtb_z  = max_i |q_i - (a_el * x_i + b_el)| / s_el # robust residual, per atom

flag   = sib > 0.3  AND  xtb_z > (99th percentile of the corpus)
```

**Why confirmation is required, and not optional.** The screen is
model-independent in its definition but *not* independent of model difficulty: a
sibling disagreement means the charge varies with conformation, and a graph-only
model cannot predict conformational variation by construction. Screening alone
therefore selects records this model class must fail on — including records whose
labels are perfectly valid. Requiring an independent quantum-chemical
contradiction converts the criterion from "records our models find hard" into
"records where two partition schemes disagree about the same wavefunction", which
is a statement about the data rather than about the model.

The cost of getting this wrong is small but real, and was measured. Of 386
records the screen flags at 0.3 e, 300 are confirmed and 86 are valid-but-hard.
Dropping only the confirmed 300 moves Sieve's full-corpus test MAE by −1.76% and
DASH's by −1.33%; dropping only the 86 moves them by −0.24% and −0.18%. So **88%
of the effect comes from records with independent evidence of corruption**, and
the valid-but-hard residue is about a tenth of it and near-identical across the
two model families. Confirmation removes even that residue.

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

`molecules.parquet` gains three columns, and the fold stores inherit them
automatically — `partition_store` reads the whole frame and writes row subsets,
so it carries unknown columns through without modification.

| column | type | meaning |
|---|---|---|
| `mbis_sibling_dev` | float64 | screen: `max_i \|q_i − median_over_siblings(q_i)\|`, in e |
| `mbis_xtb_z` | float64 | confirmation: max robust residual of MBIS against GFN2-xTB Mulliken |
| `mbis_anomaly_flag` | bool | `mbis_sibling_dev > 0.3 AND mbis_xtb_z > p99` |

Storing both scores, not only the verdict, is what keeps the thresholds revisable
and the sensitivity analysis free — and it lets an analysis separate the
confirmed-corrupt population from the valid-but-conformationally-hard one, which
are different things and should not be conflated.

## Where it runs

Each statistic is computed where its inputs exist.

```
download → parse (+ mbis_xtb_z) → split → annotate (+ mbis_sibling_dev, flag)
```

`mbis_xtb_z` is computed **in `parse_record`**, which is the only place the
auxiliary charges exist — that function currently clears every mol-level property
right after reading `MBIScharge`, discarding `XTB_MulikenCharge` /
`GFN2:MULLIKEN_CHARGES` along with the rest. It needs only the record itself, so
it fits the streaming parse.

`mbis_sibling_dev` and the flag are computed in a new `annotate` stage after the
split, because the sibling median needs all conformers of a molecule at once.
Each stage skips when its output is already present, like the existing ones.

That annotate pass reads the parsed store, not the 7.8 GB SDF. Its runtime on the
full corpus has not been measured; the equivalent pass over the 103k-conformer
test split takes ~1 minute, dominated by deserializing each conformer's `Mol`, so
expect ~10 minutes for 1.03M — worth confirming before calling it cheap.

`prepare_store` currently returns early once the split column exists; that early
return becomes a guarded stage so the annotation stage can follow it.

Existing stores need a one-off backfill for `mbis_xtb_z`, since they were parsed
before the column existed and the auxiliary charges are no longer in the store: a
pass over the SDF keyed by `(dash_id, record order within that id)`, which is how
the analysis behind this spec matched them and took 147 s for the full corpus.
`mbis_sibling_dev` needs no backfill — re-running `prepare-store` will skip
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

**Rule, not recommendation: headline accuracy metrics are always reported on the
unfiltered store.** MAE, RMSE and R² as the primary numbers for any model come
from every record, so no reader has to trust the curation to trust the headline,
and the numbers stay comparable with anyone else's on this dataset. Filtered
numbers are reported *alongside*, never instead.

Filtering is for analysis that the corrupted records genuinely invalidate:
distribution shape, tails, uncertainty calibration. There the unfiltered
statistic is dominated by ~100 records — excess kurtosis 476 against 60 — and
reporting it would characterise the dataset's failures rather than the model.

RMSE deserves its own note whichever way it is reported: exclusion moves Sieve's
by 7.2% and DASH's by 4.5%, so the *gap* between methods depends on the choice in
a way MAE's does not.

**A framing the spec states rather than implies.** The valid-but-hard records —
those the screen catches and the cross-check clears — are unlearnable by *any*
graph-based model: their inputs are graph-identical to their siblings' and their
labels differ, so no function of the graph can fit both. Including them measures
the dataset's conformational spread, not the model. That is a sound reason to
exclude them from a distribution analysis, and it is *not* a reason to exclude
them from an accuracy number. Requiring cross-check confirmation means the flag
does not exclude them at all; the screen score is stored so an analysis can
isolate them deliberately if it wants to.

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
separates the three groups cleanly: median 0.056 e on ordinary records, 0.358 e
on screen-flagged records the cross-check clears, 0.481 e on records both methods
flag. Note the middle number is what the "you are removing hard cases" objection
is about — those records *are* six times harder than average for the model — and
it is exactly why the flag requires confirmation and the headline metric is never
filtered.

**What the screen catches that is not corruption.** Of 386 records the screen
flags at 0.3 e, ~300 (79%) are confirmed by the cross-check. The rest have
unremarkable charge magnitudes but twice the corpus-median dipole change between
conformers — molecules whose charge distribution genuinely varies with geometry.
Under this spec they are **not** flagged for exclusion: they keep a high
`mbis_sibling_dev` and a low `mbis_xtb_z`, which is exactly the signature that
identifies them, and an analysis interested in conformational sensitivity can
select them on that.

## Alternatives rejected, with measurements

Every alternative below was implemented and scored against the same independent
label; all are rejected in favour of the plain rule.

| alternative | result |
|---|---|
| Cross-scheme z as the *primary* criterion | AUC 0.9986, the best detector — but needs a 7.8 GB SDF re-parse, ~38 per-class calibration constants, and a second data source. Kept as validation, not as the criterion. |
| Dipole difference ‖Σ Δqᵢ(rᵢ−r̄)‖ | AUC 0.9556, *worse* than the per-atom cross-check; ANDing it with stage 1 cut recall to 73%. A global summary is blind to a large error near the centroid. |
| Dipole *ratio* \|μ_MBIS\|/\|μ_xTB\| | Rejected: normal records reach 5.18 against the anomaly's 5.79. Ratios blow up for near-zero denominators. A dipole check also has a known systematic baseline — MBIS charges overestimate molecular dipole and quadrupole moments by ~10% [Mikkelsen2025MBISMultipole] — though the offset measured here was ~35% (ratio 1.35 median over 273 QMugs records), a discrepancy this spec does not explain. |
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
- Reporting the anomalies upstream to the DASH and QMugs authors. Worth doing,
  the extracted SDFs exist, and the literature above makes the ask concrete:
  whether their MBIS runs recorded convergence, since the reference
  implementation's default is to reach `maxiter` silently. Not a code change
  here.

## Open questions

1. **Thresholds.** 0.3 e for the screen; the 99th percentile of the corpus for
   the confirmation. Both scores are stored, so both are revisable without
   re-deriving anything, and a two-way sensitivity table belongs in any paper
   that uses the flag.
2. **Which cross-scheme variant to implement.** The 300/86 split quoted above was
   measured with a *per-DASH-tuple-class* regression (≈38 constants, AUC 0.9986).
   A *per-element* regression is far simpler (≈11 constants) and nearly as good
   (AUC 0.9967), and for a confirmation role that is likely enough — but the
   confirmed set would shift slightly, so the 300/86 numbers must be re-measured
   against whichever variant is implemented rather than carried over.
3. **Fold stores.** Rebuild them after annotation, or annotate them in place?
   Rebuilding is simpler and deterministic; annotating in place avoids
   invalidating the ~1.8 GB of depth-6 shards keyed to the current folds.
4. **Whether `exclude_anomalous` should default to `true` for new experiment
   series** once the flag exists, with the current default kept only for
   reproducing existing runs.

## References

Registered in `references/doi_list.txt`; the bibliography is generated, never
hand-edited.

- `Verstraelen2016MBIS` — the MBIS method, 10.1021/acs.jctc.6b00456
- `Mikkelsen2025MBISMultipole` — multipole-constrained MBIS, 10.1021/acs.jctc.4c01297
- `Isert2022QMugs` — the QMugs dataset, 10.1038/s41597-022-01390-7
- `Lehner2023DASH` — the DASH tree (already registered), 10.1021/acs.jcim.3c00800

Not registered, deliberately: the HORTON and Psi4 documentation pages cited above
are software documentation with no DOI, and the 2017 Comment on the MBIS paper
(arXiv:1701.01714) is cited in discussion but its published DOI could not be
verified, so no guess was registered for it.
