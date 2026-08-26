# The united-atom store — engineering history

Full deep-dive behind `chaos-store-ua` (built by `python -m
sieve_experiments coarse-grain-store`) and its use in testing DASH (T8) and
`chemprop_atom` (T11) at their native atom granularity. The short version
lives in `README.md`'s Predictors section; `pins.toml`'s `[chaos_store_ua]`
just points here.

## Why build it

Prompted by a question about DASH's own atom granularity: is it all-atom or
united-atom (implicit H)? Investigated directly against real chaos-store
data and DASH-tree's source (`serenityff/charge/tree/dash_tree.py`,
`atom_features.py`) while building T11:

- DASH *is* all-atom in what it **predicts** — every hydrogen gets its own
  prediction from its own dedicated tree branch (branch 37, the single
  feature tuple `(1,1,0,False,0)` every H maps to — `AtomFeatures.
  feature_list` has exactly one H entry out of 122).
- But it is united-atom in how it **represents**: `_get_init_layer`
  explicitly redirects an H atom's subgraph descent to its heavy-atom
  neighbor, with the authors' own comment "skip Hs as they are only
  treated implicitly". Measured on 2930 real chaos-store atoms: 0
  subgraphs ever contain a hydrogen. Consequence: an H's prediction is a
  pure function of its heavy atom's environment — 60.4% of hydrogens in a
  sample of real chaos-store molecules share a bit-identical tree path
  with another H in the same molecule (e.g. both H's of a CH2), so they
  get identical predictions.

Since 56.8% of chaos-store atoms are hydrogen, `atom/*` metrics are
majority-weighted by exactly the atoms DASH represents most crudely — an
all-atom DASH number and a genuinely united-atom prediction problem are
being conflated under one metric. The fix is not a predictor option: it's
giving DASH (and T11) an atom population that actually matches DASH's own
united-atom design, via a real united-atom store, so the same all-atom vs.
united-atom question can be asked of every atom-level predictor uniformly.

## How it's built

Built via cosmolayer's own `SegmentStore.coarse_grain()`, not a bespoke
merge: every hydrogen `Chem.RemoveHs` would actually remove gets folded
into its heavy-atom neighbor; a hydrogen RDKit conservatively keeps (a
non-default isotope, or a neighbor with non-tetrahedral stereo it can't
safely represent implicitly) survives as its own atom, same as any heavy
atom — verified with a synthetic deuterium case (`[2H]`), since chaos-store
itself has zero such survivors in every subsample checked (2000
molecules).

**Why this preserves a controlled comparison**: `coarse_grain` drops no
segments — only atom partitioning changes. Verified on a 400-molecule
subsample: molecule-level sigma profiles and areas come back bit-identical
between the AA and UA stores. `molecules_df`'s `split`/`biased_split`
columns are also carried through unchanged (not recomputed) —
`prepare_ua_store` (`prepare_store.py`) enforces this by deriving the UA
store from an already-split AA store rather than re-clustering, so AA and
UA runs use the exact same molecules in the exact same train/val/test
roles. Molecule-level metrics are therefore directly comparable AA-vs-UA
the same way as any other config change; atom-level metrics are *not*
comparable across AA/UA (different atom population, different targets) and
must be reported as separate blocks, never the same column.

**No predictor changes needed for atom ordering/parsing** (separate from
the DASH feature-vocabulary issue below), verified rather than assumed:

- Both predictors' SMILES/atom-order paths (DASH's plain
  `Chem.SmilesParserParams(removeHs=False)`, T11's `make_mol(keep_h=True,
  add_h=False, reorder_atoms=True)`) parse UA SMILES correctly with zero
  code changes: verified 200/200 real coarse-grained molecules,
  element-order-aligned against the UA store's own `atoms_df`, for both
  paths, including the synthetic deuterium-survivor case. `keep_h`/`add_h`
  need care though (a real footgun, worth remembering for any future
  predictor): `keep_h` only *retains* explicit H already written in the
  SMILES text (`params.removeHs = not keep_h`); it never *adds* one.
  `add_h=True` calls `Chem.AddHs`, which explicitly instantiates every
  implicit H as a brand-new atom with no atom-map number — on a UA store
  that would silently balloon the united atoms back toward all-atom and
  break `reorder_atoms`'s map-number sort entirely. Neither predictor sets
  `add_h`; T11 already used `keep_h=True, add_h=False`, so it needed no
  change at all for UA support.
- T10 (molecule-level) needs no re-run on the UA store: it never touches
  atom-level truth, and its plain `Chem.MolFromSmiles(smi)` (default
  `removeHs=True`) produces the identical heavy-atom graph whether parsing
  an AA or a UA SMILES (coarse-graining never touches heavy-atom
  connectivity, only merges peripheral H into their neighbor's implicit-H
  count).

## Build performance: a real quadratic-time bug found and fixed

The "~0.2s per 400 molecules, ~30s extrapolated for the full store"
estimate from an early subsample timing was *wrong*, not just imprecise —
linearly extrapolating a subsample timing hid a real algorithmic bug
rather than measuring it. The first real full-store `coarse-grain-store
chaos-store` run was killed after 30+ min of steady 100% CPU with no sign
of finishing (confirmed not hung via `/proc`: advancing wall clock, stable
~3.5GB RSS, still mid-load with no output written yet).

**Root cause**, found by reading cosmolayer's own source directly:
`SegmentStore.coarse_grain()`'s atom-index remap loop recomputed
`segment_molecule == m` — a full scan of the *entire* segment array — once
per molecule, plus two more full-length boolean-mask operations per
iteration. O(n_molecules × n_segments), not O(n_segments): chaos-store has
~53,079 molecules and ~99.7M segments, so the loop did on the order of
5.29e12 element-compares total, vs 2.96e8 on the 400-molecule subsample
that "validated" the fast estimate — 17,861× more work, not the 132× a
molecule-count-only extrapolation predicts. Segments are already
contiguous per molecule by construction (that's what `segment_offsets` is
for), so this is a genuine, fixable performance bug, not something
inherent to coarse-graining.

**Fix**: cosmolayer PR [#55](https://github.com/craabreu/cosmolayer/pull/55)
(branch `fix/coarse-grain-quadratic-remap`) slices each molecule's own
contiguous segment range directly from `segment_offsets` instead of
rescanning the full array. Bit-identical output (all 129 of cosmolayer's
own store tests pass unchanged, including the 5 existing CoarseGrain ones);
added a new regression test using differently-sized fixture molecules so a
boundary slip in the slicing itself would be caught. Measured on synthetic
data at chaos-store's real segment density: 339× faster at n_mols=2000,
1650× at n_mols=8000, with the ratio growing (confirming O(n²) vs O(n), not
a constant factor) — extrapolated full-store: the old loop alone is
40-90 minutes; the new one is under a second.

**Status**: PR #55 merged; sieve's own pin bumped accordingly. Real
full-store `chaos-store-ua` build: **21.7s total** (was projected at
40-90+ minutes pre-fix). 53,079 molecules, 1,546,081 atoms → 736,106
(52.4% reduction). Verified on the full store (not just a subsample):
molecules/split/biased_split identical to chaos-store, molecule-level
sigma profiles and areas bit-identical. Idempotent re-run: 0.4s.

`prepare_ua_store` (`prepare_store.py`) is idempotent like `prepare_store`:
skipped if the destination already has a `biased_split` column. CLI:
`python -m sieve_experiments coarse-grain-store chaos-store` (default dest
`<source>-ua`).

## DASH-UA: a real, useful negative result

**Correction (2026-08-25):** an earlier claim — "DASH's own atom features
already encode H-count... DASH-on-UA is the same model, just finally
scored at its own native granularity" — was wrong. It was based on
verifying only `GetTotalNumHs(includeNeighbors=True)` invariance (true),
never checking DASH's *other* feature field, `atom.GetDegree()` (not
true). RDKit's `GetDegree()` counts explicit bonded neighbors only — it
does *not* include implicit hydrogens. DASH's published tree was built
entirely from all-atom (explicit-H) COSMO representations, where a heavy
atom's degree structurally includes any bonded H as an explicit neighbor.
On the UA store the same atom's degree drops by however many H's got
folded away (implicit H isn't counted), producing a `(element, degree,
charge, aromatic, nH)` tuple the published vocabulary may never have
seen — either an outright `KeyError` (`init_neighbor_dict` rejects the
whole molecule) or a match against a less-relevant node.

Confirmed directly, e.g. a phosphine-oxide P atom (`...O[PH](C)=O`): AA
tuple `(15, 4, 0, False, 1)` (degree counts the explicit H) vs UA tuple
`(15, 3, 0, False, 1)` (same atom, degree excludes the folded H) — only
the AA tuple is in DASH's vocabulary. Measured on a 300-molecule
diagnostic sample: 22/300 molecules rejected on UA vs 13/300 on AA, 10 of
which succeed on AA and fail *only* on UA, every one via this exact
mechanism.

**Confirmed in the primary source**, not just the code (Lehner et al., *J.
Chem. Inf. Model.* 63, 6014-6028 (2023), "Atom Features" section, p. 6017):
the paper lists "Number of bonds (1, 2, 3, 4, 5)" and "Number of attached
hydrogens (0, 1, 2, 3)" as two *separate*, independent feature fields, and
gives H2 as a worked example: "a hydrogen atom with one bond and a formal
charge of zero has the atom type 'H 1 0 False 0'" — one bond (to the other
H), zero attached hydrogens (H itself has none). This is only a coherent
design if every atom, hydrogen included, is meant to be an explicit graph
node — exactly `GetDegree()`'s semantics. So this isn't an implementation
quirk to route around; it's DASH's own designed input contract. A
united-atom (implicit-H) input is outside that contract by construction,
not a different but equally valid way to run the same method.

(Separately investigated and *ruled out* as a comparable issue: does
chemprop's `MultiHotAtomFeaturizer.v2()` have an analogous defect, since
its own nH field also depends on RDKit call defaults? No — its "number of
Hs" field uses `GetTotalNumHs()` without `includeNeighbors=True`, which
correctly reports 0 for every T11 heavy atom (T11 always keeps H
explicit) — but this is *not* a bug, it's the standard graph-featurization
convention: "number of hydrogens not otherwise represented as graph
nodes." When every H is already an explicit node with its own features and
bonds, 0 implicit H is exactly correct, not lost information — message
passing sees the real H nodes directly. DASH has no such mechanism (no
message passing, a discrete tree lookup only), which is precisely why the
same category of representation-dependence is fatal for DASH's
`GetDegree()` field but not for T11's `GetTotalNumHs()` field.)

**Full-store results** (produced by the now-deleted `dash-ua-biased.yaml`
— kept here as the historical record since the finding above is now fully
captured in prose and doesn't need to stay reproducible on demand;
`biased_split`, n_test 5333 molecules / 108,347 atoms):

| metric | DASH AA (decomposed) | DASH UA |
|---|---|---|
| coverage: train atoms unmatched | 4.5% | 9.0% |
| coverage: train molecules rejected outright | 4.9% | 10.6% |
| coverage: test molecules rejected outright | 1.0% | 4.6% |
| atom/profile/w1_norm_mean | 1.030 | 1.551 |
| atom/area/r2 | 0.945 | 0.906 |
| atom/charge/mae | 0.00752 | 0.0193 |
| profile/w1_norm_mean | 0.449 | 0.812 |
| area/r2 | 0.949 | 0.872 |
| charge/mae | 0.102 | 0.153 |

DASH is unambiguously worse on chaos-store-ua, across every metric —
exactly the mechanism above (`GetDegree()`'s representation-dependence
feeding the AA-only-trained tree systematically out-of-vocabulary tuples),
not a fair "same model, native granularity" comparison. This is a genuine,
useful negative result: it demonstrates DASH's published tree/vocabulary
is *not* representation-invariant, a real limitation of its hand-built
feature scheme worth knowing about independent of this milestone. It does
*not* mean "hydrogens are hard to predict" or "united-atom targets are
harder" in any general sense — see T11-UA's results below, for a model
whose feature scheme does not have this defect.

**Decision (user's call): DASH is AA-only for every comparison in this
milestone** — `dash-biased.yaml`/`dash-random.yaml` only, never on a
united-atom store. The DASH-UA run above stays on record as the evidence
for the mechanistic finding, but its numbers must never appear in a
results table alongside `dash`/`chemprop_cosmonet`/`chemprop_atom` as if
it were a comparable configuration. The config that produced it,
`dash-ua-biased.yaml` (marked "NOT A BENCHMARK CONFIG" in its own header
for this same reason), has since been deleted. T11 (`chemprop_atom`) is
unaffected by this decision — its feature scheme does not have DASH's
defect, so `chemprop-atom-ua-biased.yaml` remains a legitimate comparison.

## T11-UA: a real positive result

Full-store, `biased_split`, n_test 5333 molecules / 108,347 atoms, fit
1174s / 19.6 min, **0/271,983 rolled-up bins negative**, predicted areas
103-533 vs true 114-520:

| metric | T11 AA | T11 UA |
|---|---|---|
| atom/profile/w1_norm_mean | 1.055 | **0.989** |
| atom/area/r2 | 0.956 | **0.976** |
| atom/charge/mae | **0.0073** | 0.0107 |
| profile/w1_norm_mean | 0.380 | **0.330** |
| area/r2 | 0.943 | **0.949** |
| charge/mae | 0.0792 | **0.0673** |

Unlike DASH, T11 genuinely improves on the united-atom store, at both
granularities and on nearly every metric (the one exception is atom-level
charge/mae, worse on UA) — consistent with the featurizer-invariance
finding above: T11 has no analogous defect to correct for, so this is a
real signal, not an artifact. Plausible mechanism (not further
investigated): a UA atom's target profile is a heavy atom's own surface
plus its former hydrogens' merged surface — a "chunkier," typically less
sparse/degenerate target than an individual H's own often-small profile,
which may simply be an easier per-atom regression target. `atom/n_test`
drops from 203,063 (AA) to 108,347 (UA) as expected (52.4% fewer atoms,
matching the store's own reduction); `atom/n_degenerate` is 87 in both
runs — likely genuinely-buried heavy atoms (zero exposed surface either
way), not investigated further.
