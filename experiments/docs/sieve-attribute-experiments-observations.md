# Sieve atom-attribute experiments: observations

Running notebook of nontrivial findings from exploring which atom
attributes/WL settings predict DASH's MBIS atomic partial charge well.
Unless noted otherwise: predictor is `sieve` (this repo's own
regressogram), `edge_attributes: []` (no bond information at all), pooled
across the five disjoint `dash-molecules-60k-{1..5}` stores (seed 0,
`minimum_support: 1`), test-split MAE/R² reported as the mean across
the 5 folds (`n_runs=5` everywhere below unless noted). None of the
configs behind these numbers are committed to git yet (see each section);
this file itself starts as an uncommitted doc too.

## Attribute correlation ratio (η²) against atom charge — full `dash-molecules`

43,248,638 atoms, 1,029,785 conformers, the *entire* store (not a
subsample). η² is the fraction of atom-charge variance explained by
grouping atoms on that attribute alone — mathematically identical to the
R² a depth-0, single-attribute Sieve baseline measures out-of-sample
(confirmed: full-store in-sample η² and the 60k-store out-of-sample R²
below agree to ~0.01-0.02 for every attribute checked).

| attribute | # values | η² |
|---|---:|---:|
| element | 11 | **0.4601** |
| electronegativity | 5 | 0.4345 |
| group | 6 | 0.3832 |
| valence_electrons | 6 | 0.3832 |
| period | 5 | 0.2297 |
| hybridization | 6 | 0.2179 |
| block | 2 | 0.2128 |
| degree | 5 | 0.1792 |
| num_ring_memberships | 7 | 0.0353 |
| min_ring_size | 21 | 0.0304 |
| aromatic | 2 | 0.0141 |
| formal_charge | 3 | 0.0061 |
| chirality | 5 | 0.0007 |
| num_h | 1 | ~0.0000 |

**`element` wins outright** among single attributes. `num_h` carries zero
signal alone on this (all-atom) store because H is already its own
`element` bucket — likely different on a united-atom store.

### `group` and `valence_electrons` are 100% redundant — but only on this dataset

Checking the actual `(element, group, valence_electrons)` triples present:
every element observed (`H C N O F P S Cl Br` — all main-group, no
transition metals) maps to a unique `(group, valence_electrons)` pair, and
the mapping is a perfect bijection (`group=17 ↔ ve=7`, `group=16 ↔ ve=6`,
...). This is **not a general property of the two attributes** — for
main-group elements, IUPAC group number and valence-electron count are
definitionally the same thing; a dataset containing transition metals
would break the bijection (group and valence-electron count diverge
there). It holds here specifically because `dash-molecules` is drug-like
organic chemistry.

## DASH baseline (for reference)

Pooled across the same 5 stores, `dash` predictor:

| | mae | r² |
|---|---:|---:|
| test, unnormalized | 0.0190 | 0.987 |
| test, `std_weighted` | 0.0193 | 0.988 |

`std_weighted` conserves molecule charge exactly (`charge_conservation/mae`
~7.5e-16, float round-off) at a small atom-level MAE cost.

## Sieve, single attribute, WL depth 0→6 (no edge attributes)

`max_wl_depth` counts **only WL refinement rounds**, not attribute-level
count — total refinement levels = `len(attribute_levels) + max_wl_depth`
(`SieveConfig.level_kinds`). At depth 0 with one attribute level, "depth"
and "total levels" coincide; they diverge once graded levels are used
(below).

| depth | element mae/r² | electronegativity mae/r² |
|---:|---|---|
| 0 | 0.1511 / 0.447 | 0.1532 / 0.425 |
| 1 | 0.0526 / 0.937 | 0.0555 / 0.929 |
| 2 | 0.0253 / 0.985 | 0.0271 / 0.983 |
| 3 | 0.0170 / 0.9903 | 0.0176 / 0.989 |
| 4 | 0.0156 / 0.9907 | 0.0160 / 0.9900 |
| 5 | 0.0153 / **0.9907** | 0.0157 / 0.9900 |
| 6 | 0.0153 / 0.9907 | 0.0157 / 0.9900 |

`element` beats `electronegativity` (0.5-step rounded Pauling EN) at every
depth — expected, since EN's 5 buckets merge elements `element` keeps
distinct. Both plateau hard by depth 4-5; depth 5→6 is noise-level
movement. `--set predictor.params.attributes=[electronegativity]` requires
a YAML config file, not `--set` alone (`--set` only parses scalars, not
lists — `--set predictor.params.attributes=[electronegativity]` silently
becomes the *string* `"[electronegativity]"` and crashes inside
`build_codes` on the literal `"["` character; caught and killed before it
ran the real sweep).

## Sieve, `[group, element, hybridization]` jointly (one flat attribute level)

| depth | mae | r² |
|---:|---:|---:|
| 0 | 0.1474 | 0.500 |
| 1 | 0.0486 | **0.944** |
| 2 | 0.0237 | 0.985 |
| 3 | 0.0169 | 0.989 |
| 4 | 0.0158 | 0.9895 |
| 5 | 0.0157 | 0.9895 |
| 6 | 0.0157 | 0.9894 |

**Crossover with plain `element`**: richer at low depth (hybridization adds
real independent signal, matching its 0.218 solo η²), but **plain
`element` wins from depth 3 onward** (0.9907 vs 0.9895 at depth 6) —
likely a sparsity effect: three jointly-refined attributes create finer
WL classes at the same nominal depth than one attribute does, so
higher-depth classes get less training support each, with no floor
against it (`minimum_support: 1`).

(`edge_attributes` wasn't reachable on `SievePredictor` at all before this
session — `_build_config` always called `build_codes` with its hardcoded
default `("bond_type",)`, never threading a caller's choice through to
either `build_codes` or `SieveConfig` itself. Fixed and committed:
`charge_experiments` `bf014ca`.)

## Graded attribute levels: `[group] → [element] → [hybridization]`, no `neighbor_depth`

Numerically **identical to the flat joint version above, at every single
depth** (verified to 6 significant figures). Not a coincidence: without
`neighbor_depth` set, a WL round's parent is always *the last attribute
level* regardless of how many levels preceded it
(`SieveConfig.level_parents`), and a graded chain's class key at that last
level is still the joint tuple `(group, element, hybridization)` — the
same partition the flat single-level version builds directly. So grading
vs. flattening the same attribute set only matters once `neighbor_depth`
is involved (below) — otherwise it's a no-op on the final classes (though
not on backoff availability for genuinely unmatched atoms, which never
triggered here: every combination in test was already seen in train).

## `neighbor_depth`: what it is, what it isn't

`neighbor_depth` lets WL neighbors contribute a **coarser**, independently
refining chain than the center, instead of the same full attribute
schema at every round (design.md §3.6 — "evaluated, not adopted": measured
there only as *coverage*, never against a real target). It's an index
into `attribute_levels` (`1 <= neighbor_depth <= len(attribute_levels)`);
`neighbor_depth=0` is invalid — `SieveConfig` raises `"neighbor_depth must
be between 1 and N ... got 0"`, because it means "the coarse chain's first
WL round branches off attribute level `neighbor_depth - 1`," and there is
no level `-1` to point at. Topology-only WL (uniform initial labeling, no
attribute at all) is a perfectly valid *general* WL procedure, but wasn't
expressible in Sieve as it stood: `SieveConfig.__post_init__` requires
`attribute_levels` non-empty with `>= 1` attribute per level, so there was
no way to start *any* chain — main or coarse — from a constant label.

**Fix, added this session**: `is_atom`, a genuinely constant attribute
(`"atom"` for every atom, real or RDKit's own dummy atom) — see
`src/sieve/io/rdkit_adapter.py`. As `attribute_levels[0]` it puts every
atom in one class at level 0, so WL from there on is driven purely by
topology. Committed to `sieve` core (not `charge_experiments`):
commit `1701def`. Tested directly (`test_is_atom_attribute_is_constant...`
in `tests/test_rdkit_adapter.py`) and full sieve suite green (155 passed,
1 pre-existing skip).

**A load-bearing side effect, confirmed but not yet acted on**: `is_atom`
at level 0 is mathematically the same thing as `SieveModel`'s own
separately-tracked `global_mean`/`global_count`/`global_msd` fields
(`model.py`) — both are "the level with no parent, that always matches,
whose mean is over everyone," and `shrinkage.py` already uses
`global_mean` as level 0's own shrinkage target via the identical
`parents[k] < 0` branch that `predict.py` uses for the final backoff.
Confirmed empirically below (depth-0 R² is exactly 0, i.e. exactly the
`global_mean` predictor). Not implemented: collapsing the two into one
mechanism would need `predict.py`'s `matched_level == -1` sentinel
semantics reworked and `merge.py`'s separate global-field pooling folded
into ordinary level-merging, plus a schema/format-version bump — flagged
as real, separate sieve-core work, deliberately not started.

## Sieve, topology-only (`attributes: [is_atom]`, no edge attributes)

| depth | mae | r² |
|---:|---:|---:|
| 0 | 0.2473 | **~0.0000** (`-4.9e-14`, float noise) |
| 1 | 0.1868 | 0.179 |
| 2 | 0.1477 | 0.375 |
| 3 | 0.1286 | 0.476 |
| 4 | 0.1124 | 0.537 |
| 5 | 0.1054 | **0.544** |
| 6 | 0.1021 | 0.540 |

Depth-0 R²=0 is the direct numerical proof of the `is_atom ≡ global_mean`
equivalence above — a constant predictor's R² is 0 by definition.

**Genuinely surprising result**: pure, unlabeled molecular-graph topology
(no element, no attributes at all, not even bond type) reaches **r²=0.544
by depth 5** — *higher* than `element` alone with zero WL refinement
(r²=0.447). A charge's position in the molecular graph carries more signal
than raw atom identity does, at least compared against a zero-depth
attribute baseline. Plateaus (even dips slightly) past depth 5, same
shape as every other attribute set tried.

## Graded `[element] → [hybridization]`, `neighbor_depth=1` (element-only WL neighbors)

| depth | mae | r² |
|---:|---:|---:|
| 0 | 0.1474 | 0.500 |
| 1 | 0.0518 | 0.939 |
| 2 | 0.0253 | 0.985 |
| 3 | 0.0170 | 0.9903 |
| 4 | 0.0156 | **0.9908** |
| 5 | 0.0153 | 0.9908 |
| 6 | 0.0153 | 0.9907 |

Best of both worlds, and the first real (non-coverage) confirmation of
design.md §3.6's hypothesis: keeps the joint config's depth-0/1 boost from
`hybridization` (0.500 r² at depth 0, matching the joint/graded-no-neighbor
runs above), while *avoiding* the joint config's higher-depth sparsity
penalty — element-only neighbors keep the WL neighbor alphabet small, so
this edges out plain `element` by depth 4 (0.9908 vs 0.9907) instead of
falling behind it the way the fully-fine joint/graded-no-neighbor config
did (capped at 0.9895).

## Joint (multi-attribute) η², full store

Same full-store methodology, but grouping atoms on attribute *tuples*:

Every two-way `(element, X)` pair:

| joint attribute | combos | η² | vs element alone |
|---|---:|---:|---:|
| element | 11 | 0.4601 | — |
| **element + degree** | 26 | **0.5427** | **+0.0826** |
| element + hybridization | 22 | 0.5091 | +0.0490 |
| element + min_ring_size | 94 | 0.4833 | +0.0232 |
| element + formal_charge | 21 | 0.4712 | +0.0111 |
| element + aromatic | 17 | 0.4699 | +0.0098 |
| element + num_h | 11 | 0.4601 | −0.0000 |
| element + electronegativity | 11 | 0.4601 | −0.0000 |

Deeper stacks:

| combination | # combos | η² |
|---|---:|---:|
| element + hybridization | 22 | 0.5091 |
| element + degree + hybridization | 37 | 0.5480 |
| element + hybridization + aromatic | 29 | 0.5750 |
| element + hybridization + aromatic + formal_charge | 49 | 0.5872 |
| element + hybridization + aromatic + min_ring_size | 192 | 0.5992 |
| element + degree + aromatic | 35 | 0.6141 |
| element + hybridization + aromatic + formal_charge + min_ring_size + num_ring_memberships | 382 | 0.6170 |
| **DASH tuple** (element + degree + formal_charge + aromatic + num_h) | **50** | **0.6260** |
| ALL 14 atom-local attributes | 617 | 0.6600 |

**`hybridization` is nearly subsumed by `degree`.** It adds only **+0.0053**
on top of `element+degree`, while `degree` adds **+0.0389** on top of
`element+hybridization`. Not merely "degree wins pairwise" — degree almost
entirely *contains* it. Chemically consistent: in an all-atom graph,
coordination number plus element pins down hybridization for most organic
atoms.

**A clean additive decomposition:**

| step | η² | gain |
|---|---:|---:|
| element | 0.4601 | — |
| + degree | 0.5427 | +0.0826 |
| + aromatic | 0.6141 | +0.0714 |
| + formal_charge | 0.6260 | +0.0119 |
| + *all nine others* | 0.6600 | +0.0340 |

Four attributes reach 95% of the all-attribute ceiling.

**DASH's feature tuple is well-designed, and efficiency is the point.**
η²=0.6260 from **50 classes**, against the all-attribute ceiling's 0.6600
from **617** — 95% of the signal at 8% of the class count. A deliberately
"WL-orthogonal" stack (adding ring attributes, which 1-WL provably cannot
compute) scores *lower* (0.6170) while using 382 classes. Since class
fragmentation is exactly what degrades high-depth performance (see the
`formal_charge` resonance section), **signal per class is the metric that
matters**, and DASH's choice optimizes it well.

**The WL caveat, which η² cannot see.** `degree` — the largest contributor
after `element` — is precisely what WL recovers for free: WL round 1
computes `h₁(v) = H(h₀(v), MULTISET{h₀(u)})`, and the *cardinality* of that
multiset **is** the degree. So `degree`-as-an-attribute should be
redundant at `max_wl_depth >= 1`, and the η² ranking should invert at
depth: `degree`'s +0.0826 evaporating while `hybridization` (not
recoverable — every run here uses `edge_attributes: []`, so WL sees no
bond orders at all) becomes the useful route to that information instead.
Including `degree` is not *zero* benefit though: it also injects
*neighbours'* degrees into the multiset one round earlier than plain WL
would reach them — a head start, not new information, which is the same
speed-vs-support trade seen throughout.

*Selection principle for Sieve (as opposed to DASH): prefer attributes WL
cannot compute.* WL recovers `degree` and `num_h` (H-neighbour count);
1-WL provably cannot count cycles, so `aromatic`/ring attributes carry
genuinely non-WL information; `element`, `formal_charge` and (absent
`bond_type` edges) `hybridization` are not graph-derivable at all.

### Confirmed: `degree`'s value evaporates with WL depth, then goes negative

`[element, degree]` vs `[element]`, same five stores, test r²:

| depth | element | element + degree | delta |
|---:|---:|---:|---:|
| 0 | 0.4472 | 0.5390 | **+0.0917** |
| 1 | 0.9370 | 0.9486 | +0.0116 |
| 2 | 0.9850 | 0.9856 | +0.0006 |
| 3 | 0.9903 | 0.9894 | −0.0009 |
| 4 | 0.9907 | 0.9896 | −0.0011 |
| 5 | 0.9907 | 0.9895 | −0.0012 |
| 6 | 0.9907 | 0.9895 | −0.0012 |

All three predicted effects appear:

1. **WL recovers it.** The +0.0917 depth-0 advantage (η² predicted
   +0.0826) collapses 87% after one WL round.
2. **The depth-1 residual is the head start**, not retained information:
   +0.0116 survives one round because `degree`-as-attribute also injects
   *neighbours'* degrees a round earlier than plain WL reaches them. Gone
   by depth 2.
3. **Past depth 2 it is a net cost** — sign flips negative and stays
   there (−0.0012, ~2-3x the fold-to-fold std of 0.0004, consistent in
   sign across four depths). Same support-fragmentation mechanism as the
   `formal_charge` resonance splitting and the
   `[group,element,hybridization]` high-depth degradation.

**The practical lesson**: `degree` is the best atom-local partner for
`element` by η² (+0.0826) *and* the one WL makes free. Screening
attributes on atoms alone would have led straight to it; only the depth
sweep reveals the inversion. η² ranks raw signal; what matters in a WL
model is signal WL cannot itself derive.

### `[element] → [degree]`, `neighbor_depth=1`: exactly neutral

Same five stores, test r², against both baselines:

| depth | element | `[element,degree]` flat | `[element]→[degree]` nd=1 | vs element | vs flat |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4472 | 0.5390 | 0.5390 | +0.0918 | −0.0000 |
| 1 | 0.9370 | 0.9486 | 0.9370 | −0.0000 | −0.0116 |
| 2 | 0.9850 | 0.9856 | 0.9850 | +0.0000 | −0.0006 |
| 3 | 0.9903 | 0.9894 | 0.9902 | −0.0001 | +0.0008 |
| 4 | 0.9907 | 0.9896 | 0.9907 | +0.0000 | +0.0011 |
| 5 | 0.9907 | 0.9895 | 0.9907 | −0.0000 | +0.0012 |
| 6 | 0.9907 | 0.9895 | 0.9907 | −0.0000 | +0.0012 |

Two separable effects:

1. **`neighbor_depth` removed the fragmentation penalty.** The flat
   version's −0.0012 deficit at depth 4-6 is fully recovered (+0.0011 /
   +0.0012 vs flat). Keeping `degree` out of the *neighbour* alphabet is
   what removed the cost.
2. **It converges to plain `element` exactly** (±0.0000 at every depth
   >= 1). Explicable as a genuine identity: the centre starts as
   `(element, degree)` and refines against element-seeded coarse
   descriptors of its neighbours — but `degree` *is* the cardinality of
   that same multiset, so `(element,degree) + element-WL-neighbours`
   carries identical information to `element + element-WL-neighbours`,
   i.e. plain `element` WL. Same partition, same predictions.

## The 0.9907 plateau is overfitting, not missing information

Three independent `neighbor_depth=1` configurations converge to **exactly
0.9907** at depth >= 3 — `[element]→[degree]`, `[element]→[hybridization]`,
and `[element]→[hybridization]→[aromatic]`:

| depth | element | `el→hyb→arom` nd=1 | std | vs element |
|---:|---:|---:|---:|---:|
| 0 | 0.4472 | **0.5519** | 0.0020 | **+0.1047** |
| 1 | 0.9370 | 0.9419 | 0.0003 | +0.0049 |
| 2 | 0.9850 | 0.9852 | 0.0003 | +0.0002 |
| 3 | 0.9903 | 0.9902 | 0.0004 | −0.0001 |
| 4 | 0.9907 | 0.9907 | 0.0004 | +0.0000 |
| 5 | 0.9907 | 0.9907 | 0.0004 | −0.0000 |
| 6 | 0.9907 | 0.9906 | 0.0004 | −0.0001 |

**The cause is overfitting.** Train vs test r² for the `element` sweep:

| depth | train r² | test r² | gap |
|---:|---:|---:|---:|
| 1 | 0.9402 | 0.9370 | 0.0032 |
| 2 | 0.9883 | 0.9850 | 0.0033 |
| 3 | 0.9965 | 0.9903 | 0.0062 |
| 4 | 0.9985 | 0.9907 | 0.0078 |
| 5 | 0.9991 | 0.9907 | 0.0084 |
| 6 | 0.9993 | 0.9907 | 0.0086 |

Textbook shape: train climbs monotonically, test flatlines from depth 4,
and the gap widens with every added round. Every attribute configuration
converging to the same 0.9907 is therefore a shared **generalisation**
wall, not a shared information wall.

**The untouched lever.** Every run in this series used the most permissive
possible regularisation, and neither knob was ever varied:

- `minimum_support: 1` — a class holding a *single* training atom is used
  directly as an estimate.
- `shrinkage_strength: None` — no shrinkage of class means toward their
  parents (`shrinkage.shrunk_means` degenerates to
  `where(n > 0, lvl.mean, parent_est)`).

Both are `SieveConfig` fields exposed by `SievePredictor`. Sweeping them at
fixed depth is the experiment most likely to move test r² off 0.9907 —
unlike any further attribute engineering.

### Two retracted claims (both wrong)

1. **"The missing information is in the edges."** Because three
   centre-attribute configurations converged to 0.9907, an earlier draft
   concluded WL must lack bond orders (every sweep uses
   `edge_attributes: []`) and proposed `[element]` +
   `edge_attributes: [bond_type]` as decisive. Wrong: the model already
   fits *training* data far past what it generalises, so it does not lack
   features. Adding `bond_type` refines the partition further and should
   make overfitting worse.

2. **"Train r² exceeds the conformational ceiling, so Sieve is memorising
   conformers."** Wrong twice over.
   - *Mechanically impossible*: conformers of one molecule have identical
     graphs, so corresponding atoms get identical features and always land
     in the same class at any depth. A class always contains all
     conformers of any atom it contains, so its mean is the conformer mean.
     Sieve cannot separate conformers even in principle.
   - *The comparison was across two different datasets*: the 0.99869
     ceiling was computed on the full `dash-molecules` store (2.94
     conformers/molecule), but **every experiment ran on the
     `dash-molecules-60k-*` stores, which hold exactly one conformer per
     molecule** (`subsample-store` defaults to
     `conformers_per_molecule: 1`). On a one-conformer store there is no
     conformational variance at all, so the graph-only ceiling there is
     **1.0** and no ceiling argument applies.

   The conformational-ceiling figure (0.99869, within-conformer variance
   = 0.131% of total) remains valid *for the full store*, and is worth
   knowing before any run that uses multiple conformers per molecule. It
   says nothing about the sweeps recorded here.

## Edge attributes shadow atom attributes — a caveat to this whole series

Every sweep above uses `edge_attributes: []`. That matters, because RDKit
propagates the *same* perception to bonds, and WL round 1 sees the multiset
of `(neighbour label, edge label)` pairs — so an edge attribute enters
exactly where the corresponding atom attribute would.

Tested on `dash-molecules-60k-1` (2,526,987 atoms) by asking whether the
WL-round-1 view `(element, multiset of incident edge labels)` *determines*
the atom attribute:

| atom attribute | edge attribute | views | ambiguous | atoms affected |
|---|---|---:|---:|---|
| `aromatic` | `bond_type` | 48 | **0** | **0 (0.000%)** |
| `min_ring_size` | `bond_min_ring_size` | 270 | **0** | **0 (0.000%)** |
| `num_ring_memberships` | `bond_num_ring_memberships` | 85 | **0** | **0 (0.000%)** |
| `hybridization` | `bond_type` | 48 | 4 | 143,211 (5.667%) |
| `hybridization` | `conjugated` | 43 | 5 | 80,000 (3.166%) |

`aromatic` is **exactly** recoverable: RDKit types every aromatic ring bond
`AROMATIC`, so "atom is aromatic" ⟺ "has an incident AROMATIC bond". Both
ring attributes are likewise exactly shadowed by their bond-level
counterparts.

`hybridization` is 94.3% determined at round 1. All four ambiguous views
are lone-pair conjugation cases — atoms whose hybridization depends on
*what the neighbour is*, not on their own bonds:

| view | hybridizations | atoms | chemistry |
|---|---|---:|---|
| `N` + 3×SINGLE | SP2 / SP3 | 72,703 | amine (sp³) vs amide/aniline N (sp²) |
| `O` + 2×SINGLE | SP2 / SP3 | 67,882 | ether vs ester/aryl ether O |
| `O` + 1×SINGLE | SP2 / SP3 | 2,498 | hydroxyl vs phenol/enol O |
| `S` + DOUBLE,2×SINGLE | SP2D / SP3 | 128 | sulfoxide/sulfone |

Extending to a depth-2 view drops ambiguity to **0.679%** — one extra WL
round resolves 88% of what remains.

**Consequence.** `aromatic` and `hybridization` looked valuable in the η²
decomposition (+0.0714 and +0.0490) *only because no edge attributes were
enabled*. With `bond_type` edges, atom-level `aromatic` carries literally
zero information and is pure class fragmentation — the same trap `degree`
fell into once WL ran. The clean test, not yet run: `[element]` with
`edge_attributes: [bond_type]` versus `[element, hybridization, aromatic]`
with `edge_attributes: []`.

This generalises the selection principle: **prefer attributes that neither
WL nor an enabled edge attribute can derive.**

**`neighbor_depth` cannot manufacture information; it can only stop you
paying fragmentation for information you already have.** Applied to a
WL-recoverable attribute the best it can do is return exactly to
baseline — which it does, precisely.

**Correction to the `[element] → [hybridization]` / `neighbor_depth=1`
section above**: its 0.9908 at depth 4-5 is nominally above `element`'s
0.9907, but +0.0001 is well inside the fold-to-fold std (~0.0004). That
should not have been read as a win. **Neither `neighbor_depth` variant
tested actually beats plain `element` at depth** — one is exactly
neutral, the other indistinguishable from neutral. design.md §3.6's
hypothesis remains unconfirmed against a real target; what is now
established is only the narrower claim that `neighbor_depth` removes the
fragmentation cost of extra centre attributes.

**Two exact zeros are proofs, not measurements.** `element + num_h` and
`element + electronegativity` both equal 0.4601 to four decimals — neither
adds a single distinguishable class beyond `element`. Numerical
confirmation that bucketed EN is a strict *coarsening* of element
(EN = f(element), so element ∧ EN = element) and that `num_h` is fully
derivable from the explicit-H graph on an all-atom store. Both would
behave differently on a united-atom store, where `num_h` is the only
surviving trace of the removed hydrogens.

**`degree` is the strongest partner for `element`** (+0.0826), beating
`hybridization` (+0.0490). Chemically sensible: charge is set by
electronegativity differences *summed over bonds*, so the count of
bonding partners sits directly on the causal path, while hybridization
only modulates effective EN. This partly vindicates DASH's own feature
tuple `(element, degree, formal_charge, aromatic, num_h)` — omitting
hybridization looks odd chemically, but degree really is the better
pairwise partner. (Their `num_h` is dead weight on an all-atom store.)

**`aromatic` is a near-pure interaction term**, and the three-cell
decomposition makes it unambiguous:

| context | `aromatic`'s contribution |
|---|---:|
| alone | η² = 0.0141 |
| added to `element` | **+0.0098** |
| added to `element + hybridization` | **+0.0659** |

Nearly worthless until `hybridization` is present, then the largest single
contributor in the stack. Chemically exact: `aromatic=False` lumps sp³
alkane carbon together with sp² alkene carbon, so the split is
uninformative because *both sides are heterogeneous*. Once hybridization
has isolated sp², aromatic-vs-not within sp² is precisely the
delocalized-(benzene) vs localized-(alkene) π contrast, which carries
genuinely different densities. No marginal or pairwise screen would have
found this — only the conditional decomposition does.

**Ceiling of atom-local attributes.** Stacking every useful atom-local
attribute caps at η²≈0.587, while `element` + 4 WL rounds reaches r²≈0.991
(out-of-sample). MBIS charge is a *relational* property — driven by
electronegativity differences across bonds — so no purely atom-local
feature set substitutes for neighbor information. WL is doing the
chemistry.

## `formal_charge` is not resonance-invariant, and that is a real cost

RDKit's perception is more resonance-aware than naive Lewis reading, so
the three attributes differ:

- `hybridization`: **invariant** in practice. Both carboxylate oxygens get
  `SP2`; amide N gets `SP2` (conjugation perceived), not `SP3`.
- `aromatic`: **Kekulé-invariant by construction** — `C1=CC=CC=C1` and
  `c1ccccc1` produce byte-identical attributes after sanitization.
- `formal_charge`: **not invariant.** Resonance-equivalent atoms get
  different values: carboxylate O's (0 / −1), nitro O's (0 / −1),
  guanidinium N's (+1 / 0 / 0).

Measured on `dash-molecules-60k-1`:

| group | n | mean q, O(formal_charge=0) | mean q, O(formal_charge=−1) | formal_charge differs |
|---|---:|---:|---:|---|
| carboxylate | 15 | −0.7839 | −0.7832 | always |
| nitro | 1872 | −0.4492 | −0.4504 | always |

The paired oxygens' charge distributions agree to ~0.001 — as resonance
demands — yet `formal_charge` splits them into different classes every
time. (Charged carboxylates are rare here, n=15; most acids appear in
neutral `-C(=O)OH` form. Nitro is the practically relevant case.)

**Why η² can't see the harm.** Refining a partition can only weakly
increase in-sample SS_between, so splitting two *identical* populations
costs η² nothing — which is why `formal_charge` looked merely weak
(0.0061 alone, +0.0122 marginal) rather than actively harmful. The damage
is to **support per class**: one population of 2N atoms becomes two of N,
for zero signal. That is a concrete mechanism for the high-depth sparsity
degradation measured above (`[group,element,hybridization]` capping at
r²=0.9895 while plain `element` reached 0.9907). In-sample η² is blind to
it; out-of-sample R² at depth is not.

Related-but-distinct invariance failures still open: **tautomers**
(2-pyridone vs 2-hydroxypyridine genuinely change aromaticity *and*
hybridization — but those are different molecules, not resonance forms)
and **aromaticity model dependence** across toolkits (RDKit vs Daylight vs
MDL disagree on borderline rings). Precedent in this codebase for caring
about exactly this class of problem: the `chirality` attribute
deliberately uses rigorous CIP labeling rather than raw RDKit tags, for
representation-invariance.

## Graded `[is_atom] → [element]`, `neighbor_depth=1` (topology-only WL neighbors)

| depth | mae | r² |
|---:|---:|---:|
| 0 | 0.1511 | 0.447 |
| 1 | 0.1409 | 0.539 |
| 2 | 0.0896 | 0.737 |
| 3 | 0.0703 | 0.803 |
| 4 | 0.0615 | 0.821 |
| 5 | 0.0582 | **0.823** |
| 6 | 0.0570 | 0.821 |

A real negative result, and it sharpens the reading of the section above.
Plateaus at r²≈0.823 — well above topology-alone (0.544) and
`element`-alone-no-WL (0.447), but far *below* plain `element`+WL
(0.9907) or `[element] → [hybridization]`/`neighbor_depth=1` (0.9908).
The difference is what the *coarse chain's own base* encodes: here it's
`is_atom` (constant), so neighbors never carry any element identity at
any WL round, only abstracted connectivity shape — center knows its own
element but nothing concrete about what surrounds it. In the
`[element] → [hybridization]` case, the coarse chain's base was real
`element`, so neighbors still carried true chemical identity, just not
refined further by `hybridization`. **`neighbor_depth`'s payoff depends
entirely on what the coarse level itself still encodes** — coarsening
all the way down to a constant discards something WL evidently can't
route around from the center alone, even after 6 rounds.
