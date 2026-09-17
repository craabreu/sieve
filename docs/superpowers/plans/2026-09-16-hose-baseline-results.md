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

Sieve in the configuration the studies use: **element on nodes, nothing on
edges**, so refinement past level 0 is pure graph topology.

| arm | RMSE | R² | MAE |
|---|---:|---:|---:|
| HOSE lookup, 5 spheres | 0.03927 | 0.98507 | 0.01892 |
| Sieve, pooled, *r*=5 | 0.03587 | 0.98754 | 0.01844 |
| Sieve, continuation, *r*=5 | 0.03497 | 0.98816 | 0.01809 |
| Sieve, continuation + EB, *r*=5 | **0.03478** | **0.98829** | **0.01798** |

Sieve beats the HOSE baseline on every reading. Against HOSE the full
estimator is **11.4%** lower in RMSE; the corrections account for part of that,
continuation taking 2.5% off the pooled reading and shrinkage a further 0.5%.

Note what HOSE sees that Sieve here does not: its codes carry bond order and
aromaticity, while these Sieve arms carry the element and the graph. The
comparison is not feature-matched, and it is the *incumbent* that holds the
richer description.

### Attributes hurt, monotonically, at this data volume

The same fold, same radius, same continuation + EB estimator, varying only what
each atom and bond carries:

| node attributes | edge attributes | RMSE |
|---|---|---:|
| element | none | **0.03478** |
| element | bond type | 0.03651 |
| element, degree, formal charge, aromatic, num_h | bond type | 0.03837 |

Every addition makes it worse, and the richest setting falls behind HOSE
(0.03927) on the pooled reading. At 4% of the training data the extra
attributes fragment classes faster than the information they add pays for.
Whether that survives at full scale is unknown and worth checking before it is
repeated anywhere.

## Cost

| arm | fit | predict |
|---|---:|---:|
| HOSE lookup, 5 spheres | 578.4 s | 309.8 s |
| Sieve, element on nodes, none on edges, *r*=5 | 11.9 s | not timed separately |

**49x on fit.** Predict was timed only for a richer Sieve setting, at 6.7 s
against HOSE's 309.8 s, so the prediction ratio is of the same order but is
not quoted here as a measurement of this configuration. HOSE's cost is featurization: 394 µs/atom
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
