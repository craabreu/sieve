# HOSE baseline: one-shard timing and error

Task 5 of `2026-09-16-hose-baseline.md`. One fold, run to decide whether the
full cross-validation is worth its wall clock. **Nothing here carries a
confidence interval and nothing here is a published number.**

## The split

Whole cluster-clean shards from the store's own `shard` column, so no molecule
cluster straddles the division:

| | shards | conformers | atoms |
|---|---|---:|---:|
| train | s00, s01 | 37,084 | 1,468,088 |
| test | s49 | 18,475 | 771,849 |

That is 2 of the 50 training shards, so **4% of the training data the
manuscript's own runs use**. Every absolute error below is correspondingly
worse than the manuscript's, which reaches 0.0200 at radius 5, and the arms sit
closer together than they would at full scale.

## Error, on identical data

Sieve as the manuscript configures it: element only, bond types on edges.

| arm | RMSE | R² | MAE |
|---|---:|---:|---:|
| HOSE lookup, 5 spheres | 0.03927 | 0.98507 | 0.01892 |
| Sieve, pooled, *r*=5 | 0.03830 | 0.98579 | 0.01883 |
| Sieve, continuation, *r*=5 | 0.03672 | 0.98694 | 0.01831 |
| Sieve, continuation + EB, *r*=5 | **0.03651** | **0.98709** | **0.01819** |

Sieve beats the HOSE baseline on every reading, including the pooled one, and
the two estimator corrections widen the margin: continuation alone accounts for
a 4.1% reduction in RMSE against the pooled reading, and shrinkage another
0.6%. Against HOSE the full estimator is 7.0% lower.

### With the predictor's richer default attributes

The same arms run with `DEFAULT_ATTRIBUTES` (element, degree, formal charge,
aromatic, num_h) rather than element alone:

| arm | RMSE | R² | MAE |
|---|---:|---:|---:|
| Sieve, pooled, *r*=5 | 0.04077 | 0.98390 | 0.01962 |
| Sieve, continuation, *r*=5 | 0.03853 | 0.98563 | 0.01892 |
| Sieve, continuation + EB, *r*=5 | 0.03837 | 0.98574 | 0.01882 |

Every one is **worse** than its element-only counterpart, and the pooled
reading falls behind HOSE. At 4% of the training data the richer attributes
fragment classes faster than the extra information pays for, which is the
behaviour the depth sweep already shows in the other direction. Worth
re-checking at full scale before drawing any conclusion from it.

## Cost

| arm | fit | predict |
|---|---:|---:|
| HOSE lookup, 5 spheres | 578.4 s | 309.8 s |
| Sieve, element only, *r*=5 | 15.2 s | 6.7 s |

**38x on fit, 46x on predict.** HOSE's cost is featurization: 394 µs/atom
fitting and 401 µs/atom predicting, against the 314 µs/atom benchmarked in the
spec on a single drug-like molecule. Sieve's whole fit runs at about 10 µs per
atom.

## Where atoms were answered

`HoseLookupPredictor.matched_radius` over the 771,849 test atoms:

| radius | atoms | share |
|---:|---:|---:|
| 0 (global mean) | 382 | 0.05% |
| 1 | 26,050 | 3.4% |
| 2 | 153,977 | 20.0% |
| 3 | 222,333 | 28.8% |
| 4 | 178,761 | 23.2% |
| 5 (deepest) | 190,346 | 24.7% |

Only a quarter of atoms reach the deepest radius and the mode is 3, at 4% of
the training data. Almost nothing falls all the way to the global mean, so the
corpus covers the chemistry; what it does not cover is the *deep* chemistry.
This is the diagnostic the manuscript's `TODO.md` wants for Sieve as well.

## Commands

```bash
# environment, from the repo root
python -m venv .venv && .venv/bin/pip install -e ".[dev,chem]"
.venv/bin/pip install "pandas>=2" "pyarrow>=14"
.venv/bin/pip install --no-deps \
  "hose-code-generator @ git+https://github.com/Ratsemaat/HOSE-code-generator"
```

The two run scripts are not committed: they build a `MoleculeSet` per shard
from `molecules.parquet`, fit each arm, and score with
`experiments.metrics.regression_metrics`. `experiments/stores/dash-molecules/`
here is a symlink to the prepared store in the sibling `sieve` checkout.

## What this does and does not settle

Settles: the arm runs, end to end, on the real corpus; HOSE's cost is
featurization and it is large; the manuscript's configuration beats the closest
published precedent on one fold.

Does not settle: anything with an interval. One fold, 4% of the data, no
repeats, no Tukey. A full cross-validation is what Table 3 would need, and the
spec's §7 prices a sweep on top of that.

Open, and for the author: whether to run the arm at 6 spheres as NMRShiftDB and
Kuhn et al. do rather than at 5, and whether the richer-attribute result above
survives at full scale.
