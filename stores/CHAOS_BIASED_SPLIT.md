# chaos-store `biased_split`: a size-skewed train/val/test split

`stores/chaos-store/molecules.parquet` carries two molecule-level splits:

- **`split`** -- a plain train/val/test split, built by `SegmentStore`'s own
  clustering (Butina clustering, Morgan fingerprints, Tanimoto cutoff 0.65).
  Molecule size is roughly the same across train/val/test.
- **`biased_split`** -- val and test are skewed toward larger molecules than
  train, so a model can be checked for extrapolation to sizes it barely saw
  in training. This document explains how `biased_split` is built and why.

Both are produced by `stores/download_chaos_store.py`.

## The basic idea

1. Cluster molecules so near-duplicates land in the same cluster (this keeps
   a near-duplicate from leaking across the train/test boundary).
2. Sort clusters from smallest to largest molecules.
3. Walk the sorted clusters, filling train first, then val, then test, each
   up to its target fraction of the total molecule count
   (`assign_clusters_by_mean_size` in `download_chaos_store.py`).

Because clusters are filled in size order, train ends up with the smallest
80% of molecules (by this walk), and val/test get the larger remainder --
in particular test, filled last, gets whatever is left, which is
consistently the largest slice.

## Why this needed a fix

Step 1 originally reused the store's own `cluster_id`, which comes from
Butina clustering on **plain Morgan fingerprints** (radius 2, 2048 bits).
Morgan fingerprints encode local shape (rings, branches, nearby atom types),
not overall size -- two molecules can share many of these local shapes even
when one is much bigger than the other. So a single cluster routinely held
both small and large molecules (checked directly: a cluster's minimum and
maximum heavy-atom count often span most of the dataset's whole range).

Since step 3 assigns a *whole cluster* to one split, a cluster spanning a
wide size range blurs the boundary between splits no matter how step 3
sorts or slices -- the resulting `biased_split` looked barely different from
the unbiased `split`.

## What was tried

Appending extra bits for heavy-atom count to the Morgan fingerprint before
re-clustering, so size affects which molecules land together:

| encoding (window) | clusters (6k-molecule sample) | size-weighted mean cluster (max - min) |
|---|---|---|
| plain Morgan (no size bits) | 1,975 | 8.4 |
| + thermometer (bits 1..count all on) | 299 | 13.7 -- **worse** |
| + one-hot (1 bit for the exact count) | 2,177 | 7.6 |
| + coarse, width 3 (count ± 1) | 2,020 | 6.1 |
| + coarse, width 5 (count ± 2) | **1,559** | **5.7 -- best** |
| + coarse, width 9 (count ± 4) | 785 | 6.5 -- turning worse again |

**Thermometer coding backfired.** Tanimoto similarity is
shared-bits / total-bits-either-has. A thermometer block is mostly "on"
bits (e.g. 20 of 32 bits for a 20-atom molecule), so two same-size molecules
share nearly all of that block regardless of how different their actual
structures are -- it glued unrelated molecules together into a few giant,
size-mixed clusters instead of separating them.

**One-hot was safe but weak.** Only 1 of 32 extra bits turns on, so it never
inflates similarity the way thermometer did, but it also barely moves a
Tanimoto score that already has dozens of structural bits in it.

**Coarse coding (bits {n-w, ..., n+w} on for a count of n) was the balance
point.** A small, capped overlap lets nearby sizes share a little
similarity -- enough to matter -- without flooding the score the way an
unbounded thermometer block does. Widening the window kept helping up to
about `w = 2`, then started reproducing the thermometer failure mode (fewer
clusters, growing size ranges), just more slowly since the overlap is
capped instead of unbounded.

## What `biased_split` actually uses

`SIZE_BAND_HALF_WIDTH = 2` in `download_chaos_store.py`: for a molecule
with `n` heavy atoms, bits `n-2` through `n+2` (out of a 32-bit block,
clipped to `[0, 32)`) are appended to its Morgan fingerprint before
re-clustering with the same Butina cutoff (0.65) the store itself uses.
Clusters are then grouped by **median** heavy-atom count (steadier than the
mean against one outlier member) and walked as in step 3 above.

Result, full chaos-store (~53k molecules):

| split | mean heavy atoms (old, plain-Morgan clusters) | mean heavy atoms (current, coarse-coded clusters) |
|---|---|---|
| train | 12.68 | 12.11 |
| val | 16.79 | 19.02 |
| test | 20.32 | 22.33 |

The train-to-test gap grew from 7.64 to 10.22 heavy atoms, and
`stores/characterize.py`'s `biased_split` histogram (see
`chaos-store-biased-split.png`) shows train and val/test as visibly
separate distributions rather than three overlapping bell curves.

## Regenerating

```bash
python stores/download_chaos_store.py     # rebuilds split + biased_split
python stores/characterize.py chaos-store  # regenerates all plots, including
                                            # one histogram per *split* column
```

## Known limits

- Re-clustering is still Butina clustering, still built around **shape**
  similarity; the size band only nudges it. A cluster can still span a
  double-digit heavy-atom range on occasion.
- `val` and `test` are meant to differ from each other, not just from
  `train`: val is the middle band, filled after train and before test, and
  test -- filled last, from whatever is left -- ends up the largest of the
  three (see the mean heavy-atom-count table above, where test consistently
  comes out above val).
- A stronger split would need either an explicit size gap between splits
  (dropping a middle band of clusters from every split) or a genuinely
  different clustering distance (structure and size combined with an
  explicit, tunable weight, rather than mixed into one fingerprint) --
  both left for later.
