# DASH (T8) — engineering history

Full deep-dive behind `predictors/dash.py`'s `DASHPredictor`. The short
version lives in `README.md`'s Predictors section and `pins.toml`'s
`[dash_tree]` notes (pin metadata, GOTCHA 1-3, paper parameters); this file
is everything that didn't need to stay there.

## The design goal: reproduce DASH literally

The user asked for a predictor that reproduces DASH literally, relying on
their own code as much as possible — not a parallel reimplementation of
DASH's back-off semantics, however faithful. Investigated piece by piece
against the pinned DASH-tree clone, not assumed:

- Shape/location/area decomposition is *not* an algorithmic deviation from
  DASH — it's a choice of which property gets fed into the same
  populate-and-average machinery, no different in kind from DASH-properties
  choosing to populate nodes with Mulliken vs. RESP charge. Irrelevant to
  what makes something "literal".
- No DASH function populates an *existing* tree with a new property.
  DASH-properties itself gets new properties onto nodes by re-matching
  atoms and averaging (its own paper, Sec II.C: "this population process
  does not include any fitting... each node's property is simply the
  median of all properties of all atoms matching this node") — the same
  shape `populate_tree_with_sigma_properties` takes. Some accumulation
  code is unavoidably ours; nothing exists to call instead.
- **Paper says median, code computes mean.** Grepped the whole DASH-tree
  repo for `median` — zero matches anywhere in `serenityff/`. The actual
  population code (`DevelopNode.update_average`/`get_DASH_data_from_
  dev_node`, `tree_develop/develop_node.py`) computes `np.nanmean`/
  `np.nanstd`. A real paper/code discrepancy, not a translation nuance.
  "Rely on their code" means mean, not median — this predictor computes
  the mean, matching what DASH's own code actually does.
- `DASHTree.get_property_noNAN` *is* DASH's own real prediction-time
  fallback: walks a matched node path deepest → shallowest, returns the
  first non-NaN value, consults no count/support threshold anywhere.
  `Node.prune()` (`node.py`) — the one piece of code that would implement
  a count threshold — is dead code, confirmed via grep: no caller anywhere
  in the repo except its own commented-out recursive self-call.

## Built, measured, then replaced

An initial version called `get_property_noNAN` literally, once per scalar
property (51 profile bins + charge_std, 52 calls/atom), so the back-off
step was unambiguously DASH's own code executing. Measured directly: ~47
microseconds/call (pandas `.iloc` overhead, not compute); a `--limit 5000`
probe's predict step alone took 195s, extrapolating to ~39 minutes for the
full store's 203,063 test atoms. The user weighed that cost against strict
literalness and chose speed — "It doesn't matter if it's mathematically
equivalent. I want a simple function." — so the final design is a verified
byte-for-byte-equivalent reimplementation of `get_property_noNAN`'s own
walk, not a live call into it.

**Two real bugs found along the way**, both only surfaced by actually
running the code end to end (never by the fast-suite unit tests alone,
which never touch a real `DASHTree`):

1. `get_property_noNAN` expects DASH's own raw path format (`[branch_idx,
   0, node_id_1, node_id_2, ...]`, a flat list — the literal
   `match_new_atom` return value), not this module's own `NodePath`
   conversion of it (`[(branch_idx, 0), (branch_idx, node_id_1), ...]`, a
   list of pairs, used everywhere else in `dash.py` as a dict key). Calling
   `get_property_noNAN(matched_node_path=path)` with our format raised
   `KeyError: (34, 0)` (`path[0]` is a tuple, not a plain branch-index int),
   then a second failure inside DASH's own except handler (`invalid literal
   for int() with base 10: '(34, 0)'`) trying to lazy-load a tree file
   named after the tuple's string repr. Moot now that we no longer call
   `get_property_noNAN`, but a real fact about their API worth knowing.
2. A genuine full-store-only failure: a branch with *zero* training atoms
   never gets `populate_tree_with_sigma_properties`'s new columns written
   at all (it only ever touches branches its own accumulation actually
   visited). A `--limit 5000` probe never happened to route a test atom
   into such a branch; the full 53,079-molecule store did, and hit
   `KeyError: "None of [Index(['sigma_bin_0', ...])] are in the
   [columns]"`. Fixed by treating a branch with no new columns exactly
   like any other "nothing here" case — immediate fallback, not a crash.
   Locked in with a regression test
   (`test_predict_via_data_storage_walk_falls_back_for_a_branch_with_no_
   new_columns`).

**Final design:** `populate_tree_with_sigma_properties` writes the plain
(raw, undecomposed) mean profile + charge_std per node directly onto
`tree.data_storage[branch_idx]`. `predict_via_data_storage_walk`
reimplements `get_property_noNAN`'s exact semantics (deepest → shallowest,
first populated node wins, else the global-mean fallback) rather than
calling it 52 times per atom. A first attempt at this
(`df[cols].iloc[node_id]` inside the per-atom walk) was measured *worse*
than the literal-call version — 467 microseconds/call, since re-selecting
52 columns from a wide DataFrame on every single lookup dominates. Fixed by
converting each touched branch's relevant columns to a plain numpy array
*once* (`df[cols].to_numpy()`), outside the atom loop, then indexing that
array per atom — 0.35 microseconds/call, ~1300× faster than the naive
DataFrame-slicing version and ~130× faster than `get_property_noNAN`
itself. Verified equivalent to the literal-call version's own output,
bit-for-bit, on a real `--limit 5000` run before and after the swap (same
`area/mae` to 15 significant figures), not just argued.

Two more decisions worth recording:

- `charge_std_floor=0.1` matches `DASHTree.get_molecules_partial_charges`'s
  own `default_std_value` — not the `1e-12` every other predictor here
  uses. Added as a new `reconcile_charge`/`roll_up`/`AtomPredictor`
  parameter (`predictors/base.py`), additive and defaulting to `1e-12` for
  every existing predictor's unchanged behavior.
- `charge_reconciliation="std_weighted"`'s formula was already verified
  algebraically identical to `get_molecules_partial_charges`'s own
  redistribution (residual weighted by each atom's std over the molecule's
  total std) — kept as our own function rather than calling theirs
  directly, since `get_molecules_partial_charges` bundles matching +
  scalar-charge lookup + reconciliation into one molecule-level convenience
  function with no separable reconciliation-only piece, and only knows
  about a single charge property — it doesn't fit this harness's composable
  atom-then-rollup architecture (profile + area + charge together).

## Full-store results (2026-08-26)

`biased_split`, n_test 5333 molecules / 203,063 atoms, fit 93.5s, predict
16.2s, **0/271,983 rolled-up bins negative**, predicted areas 88.5-553.2 vs
true 114.5-519.8. For the record, the first two columns are the two
Sieve-invented back-off variants tried along the way — shape/location/area
decomposition (`fit_backoff`/`predict_backoff`), and a plain bin-wise mean
with a `minimum_support=5` safety threshold on top of the published
topology (`fit_backoff_raw`/`predict_backoff_raw`) — both since retired,
their code and configs deleted; the third is what `DASHPredictor` ships:

| metric | DASH decomposed (retired) | DASH raw, min_support=5 (retired) | **DASH (shipped)** |
|---|---|---|---|
| atom/profile/w1_norm_mean | 1.030 | 1.058 | 1.012 |
| atom/area/r2 | 0.945 | 0.945 | 0.944 |
| atom/charge/mae | 0.00752 | 0.00752 | 0.00757 |
| profile/w1_norm_mean | 0.449 | 0.442 | **0.407** |
| area/r2 | 0.949 | 0.949 | 0.952 |
| charge/mae | 0.102 | 0.102 | 0.0922 |

DASH's own real back-off (no support threshold, just the missing-value
fallback) lands *better* than both of Sieve's own retired variants on every
single metric shown, at both granularities — not just close. The only
structural difference from "raw, min_support=5" is the support threshold
itself: same property (raw, undecomposed profile), same
attention_threshold, same tree. So Sieve's own `minimum_support=5` safety
threshold, motivated by distrust of thinly-supported nodes, was measurably
*costing* accuracy, not buying it — a genuinely useful, unexpected finding:
the thing added to make DASH "more careful" made it worse, on this task, at
this scale. This is the finding that settled removing `minimum_support`
(and the decomposed/raw split) entirely rather than keeping either as an
option. Not investigated further (why a thinly-supported node's raw mean
would be *more* informative than backing off to a coarser, better-supported
ancestor is not obvious) — worth a note for whoever revisits Stage A.

### Profile-mode experiment (the atom-vs-molecule decoupling)

Does the shape/location/area decomposition earn its keep over the simplest
alternative — bin-wise-averaging each tree node's raw, unnormalized atom
profiles directly, with area/charge only ever *derived* from the resulting
profile (`sum`, `profile @ sigma_values`), never fit as their own
quantities? Both retired variants above were compared full-store,
`biased_split`:

`area/r2`/`charge/mae` are identical (to floating-point noise) either way —
not a coincidence: molecule-level area/charge are additive rollups of
atom-level area/charge, and "raw"'s derived charge (mean-profile @
sigma_values) equals mean(individual atom charges) exactly by linearity of
the dot product over the mean — precisely the same number "decomposed"
fits directly. The decomposition changes nothing about area/charge, only
how the profile bins are shaped.

Where the modes genuinely differ is profile shape, and in an unexpected
direction: at the **atom** level "raw" is worse (1.058 vs 1.030 — the blur
bin-wise-averaging raw profiles across a node's differing sigma-centroids
predicts), but at the **molecule** level (after summing every atom into its
molecule) "raw" comes out slightly *better* (0.442 vs 0.449). The likely
mechanism: "decomposed" mode's own reconstruction step (shift a shape
template out to a *predicted* location, itself only ever an imperfect
point estimate) introduces its own per-atom placement error that doesn't
obviously wash out any better than raw's blur does once atoms are summed.
Not investigated further — worth a note for whoever revisits Stage A, since
the natural assumption ("more structure = better") does not hold at the
molecule level for this metric. This is exactly the atom-vs-molecule
decoupling referenced elsewhere in this project (e.g. T11's own results):
the two granularities' shape metrics do not move together, because
per-atom errors can cancel or compound when summed.

## Charge sign-convention bug (found here, fixed project-wide)

Found and fixed 2026-08-25. Every "charge" metric (and the
charge-reconciliation target every predictor uses) was silently comparing a
sigma-derived charge against `net_charge` with the wrong sign. COSMO's
screening charge is the charge the dielectric continuum induces on the
cavity surface, which *opposes* the solute's own enclosed charge —
confirmed empirically on chaos-store: molecules with `net_charge == +1`
average `Sum(sigma * mol_profile) == -1.005`, not `+1` (correlation -0.997
against `net_charge`, +0.997 against `-net_charge`). Added
`MoleculeSet.screening_charge` (`= -net_charge`) as the one correct
reconciliation/scoring target and fixed the three call sites that used
`net_charge` directly (`reconcile_charge` via `roll_up`, `global_mean`'s
`mol_charge`, `runner.py`'s `charge_true`), plus a real-data regression
test (`test_screening_charge_is_the_negated_net_charge_on_real_molecules`).
Effect on the numbers, re-run after the fix: DASH's `charge/mae` improved
0.108 → 0.102 (the retired `dash_backoff` design's own number at the time);
the retired COSMO-NET lookup baseline's improved 3× (0.0172 → 0.00546, see
`docs/cosmonet_investigation.md`) since it makes a real, nontrivial charge
prediction that was being scored against the wrong-signed target. Only
54/53,079 molecules are charged, which is why the aggregate MAE shift is
modest for DASH even though the fix is substantively correct — the
per-molecule effect on those 54 is large.

## Coverage caveat — read before quoting any DASH number

DASH cannot match every chaos-store atom: `init_neighbor_dict` rejects
atoms whose feature tuple is outside DASH's published vocabulary (boron,
Si, Ge, Sb, Te), and it runs over the whole molecule, so one such atom
disqualifies *all* of that molecule's atoms. Measured on the full store:
**train 52103/1168845 atoms (4.5%) from 2082/42459 molecules (4.9%)**;
**test 2114/203063 atoms (1.0%) from 51/5333 molecules (1.0%)**. Those
atoms fall back to the unconditional global mean — i.e. part of any DASH
score is really the floor predictor's score. Every run records this per
split in its manifest's `match_stats` and logs a WARNING; a results table
should carry it alongside the metrics rather than quoting DASH numbers as
if coverage were 100%.

## All-atom vs. united-atom

See `docs/chaos_store_ua.md` for the full united-atom store investigation
and DASH-UA's full-store results. Summary: DASH *predicts* a value for
every hydrogen but *represents* one purely as an attribute of its
heavy-atom neighbor (RDKit's `atom.GetDegree()`, one of DASH's own feature
fields, structurally includes bonded H — so a united-atom store presents
tuples DASH's published vocabulary was never built to see).
**Decision: DASH is AA-only for every comparison in this milestone** —
`dash-biased.yaml`/`dash-random.yaml` only.

## Other gotchas hit while building T8

- At the pinned commit, `DASHTree(preload=False)` (the on-demand-load mode
  originally planned, since it needs no network access) raises `KeyError`
  on every hydrogen atom — an ordering bug in `_get_init_layer`'s H-atom
  special case. Fixed by defaulting `DASHPredictor(preload=True)` (~10s,
  ~300MB into memory once, still no network). This is a *different*
  failure from the coverage caveat above — an early draft of this project's
  notes conflated the two.
- `match_new_atom` rebuilds the whole molecule's neighbor dict on every
  call unless one is passed via `neighbor_dict=` — O(n_atoms²) per
  molecule. `_atom_paths` hoists it per molecule (as DASH's own
  `_get_allAtoms_nodePaths` does): 8.5× faster, bit-identical metrics.
- The atom-index mapping (flat store position `j` → RDKit index `order[j]`)
  is guarded by a real alignment test against the store's own `element`
  column, mirroring `cosmolayer_adapter.py`'s `check_alignment`. A
  transposed mapping still produces perfectly finite metrics, so nothing
  else in the suite would catch it; the inverse convention mismatches ~36%
  of atoms, so the guard genuinely discriminates.
- `dash.py` also once defined a type alias for tree paths that shadowed
  `pathlib.Path`'s import; renamed the alias to `NodePath`. Worth watching
  for in any module that both imports `pathlib.Path` and wants a short
  type alias name.
- Two pre-existing "empty eval split" crashes surfaced by exercising a real
  small `--limit` run end-to-end: `metrics.regression_metrics`/
  `charge_metrics` raised on 0-row input (now return NaN), and
  `runner._write_plots` crashed on an empty test set via
  `plots.parity_hexbin`'s `min()`/`max()` (now skips plotting when
  `test.n_molecules == 0`). Neither is DASH-specific — any predictor hits
  these on a `--limit` small enough that `biased_split`'s val/test land
  empty.
