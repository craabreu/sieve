# A second, independent experiment series: DASH atomic partial charges

## Context

The `experiments/` tree so far is one series: predicting COSMO sigma
profiles on chaos-store, comparing DASH-tree, COSMO-NET (Chemprop), and
Sieve's own predictor. The user wants to start a second series,
reproducing the *other* prediction task from the original DASH paper —
atomic partial charges (MBIS charges) — on the DASH paper's own training
data, with the intent of publishing this series' results independently,
and before the sigma-profile series.

Because it is meant to be publishable on its own, this series must not be
entangled with the sigma-profile series' import graph: no shared package,
even where the two would otherwise look similar.

The sigma-profile tree has already been renamed from `experiments/` to
top-level `cosmo_experiments/` (freeing the `experiments` name, and kept
around, tracked, purely as reference material for this series — deletable
later if it turns out not to be needed). This spec covers scaffolding the
new, sibling `charges_experiments/` series: its data pipeline and its
first two predictors (DASH-tree, Sieve).

## Decisions (settled by conversation)

| Question | Decision |
|---|---|
| Code sharing between series | None at the harness level. `charges_experiments/` is a fully independent package (`charge_experiments`), no imports from `sieve_experiments`/`cosmo_experiments`. Registered in the same root `pyproject.toml` (packages.find, testpaths, ruff include, ty overrides) alongside the cosmo series — same repo, same CI, zero cross-imports. (Both series *do* depend on the core `sieve` package, same as today — that's a shared dependency, not shared harness code.) |
| Charge target column | `MBIScharge` (the SDF's per-atom MBIS charge property), not the DFT Lowdin/Mulliken columns also present. |
| Source data | `~/tmp/dash_molecules/dashMoleculesSDF_v2.sdf` (8.3GB, DASH's own published training set, ETH Research Collection), fetched via the download logic already in `~/tmp/dash_molecules/download_dash_molecules.sh` (streaming wget, size + md5 check), ported into the new series' `prepare-store` step the same way `prepare_store.py` already streams+verifies the chaos-store zip. |
| Per-conformer rows | Every conformer keeps its own row — **not** averaged, **not** deduplicated to one conformer per molecule. Two reasons: (a) a topology-only predictor's own tree/regressogram already aggregates repeated observations per node (mean/variance/shrinkage) more informatively than a pre-average would; (b) different conformers of the same `chembl_id` can genuinely perceive to different stereoisomers once stereo is assigned from 3D coordinates — they are not noisy repeats of one fixed graph. |
| Store row format | **No SMILES.** Each conformer's target rides directly on its own RDKit `Mol` object as a real atom property (`atom.SetDoubleProp("MBIScharge", ...)`), and the store persists the serialized `Mol` itself (`Mol.ToBinary()`, which preserves atom properties and stereo/chiral tags) rather than converting through a SMILES string. This removes the atom-map-number/array-position bookkeeping the cosmo series needs entirely — there is no separate array to keep aligned, since the target and the graph are the same object. Stereochemistry is assigned from the conformer's own 3D coordinates (`Chem.AssignStereochemistryFrom3D`) when not already fully specified by the molblock's own parity bits, before the `Mol` is serialized, so the stored `Mol`'s stereo/chiral tags reflect that conformer's real, perceived stereochemistry. |
| Sieve's target ingestion | `src/sieve/io/rdkit_adapter.py`'s `from_rdkit` gains a `y_from_atom_prop: str | None` option: when set, `y` is read per-atom (`a.GetDoubleProp(name)`) inside the loop that already walks each atom to build `node_attrs`, instead of being supplied as a separately-indexed array. This is a small, backward-compatible core-library change (the cosmo series' own callers are unaffected — they keep passing `y` explicitly), shared by both series since both go through `from_rdkit`. |
| Splitting | Butina clustering, then a single cluster-based split (train/val/test) — no size-biased second split this time (unlike the cosmo series' `split`/`biased_split` pair). Clustering and the split itself are done with `cosmolayer`'s `_chalcedon` module (`butina_cluster` + `greedy_cluster_split`), vendored wholesale into `charge_experiments` rather than taken as a `cosmolayer` dependency or reimplemented — see "Vendoring `_chalcedon`" below. `greedy_cluster_split`'s LPT scheduling replaces the cosmo series' `assign_clusters_by_mean_size`, since there's no size-biasing goal this time. Clustering fingerprints are computed **achiral** (no `useChirality`) specifically so that different stereoisomers of the same 2D graph get bit-identical fingerprints (Tanimoto distance exactly 0) and are therefore guaranteed — not merely likely — to land in the same cluster, and so the same split. Clusters (and therefore splits) are assigned by grouping all rows sharing a `chembl_id`; a molecule's conformers/stereoisomers never span two splits. |
| Predictors (first pass) | DASH-tree charge predictor and a Sieve charge predictor only. No GNN/Chemprop baseline yet — deliberately kept small, matching "publish before cosmo." Revisit once this comparison is working. |
| Metrics | Scalar-target only: MAE, RMSE, R² on `MBIScharge`, reimplemented locally (not imported from the cosmo series' `metrics.py`) since there is no profile/W1/JSD machinery to share, and computed fresh rather than copying the cosmo series' `charge_metrics` (which skips R² because net *molecular* charge clusters near zero and destabilizes `ss_tot`) — per-atom `MBIScharge` doesn't have that near-zero-variance issue, so R² is informative here. No max-abs-residual. A molecule-level net-charge-conservation check (sum of predicted atom charges vs. the molblock's own `M CHG` total) is worth keeping as a sanity metric, mirroring the cosmo series' `screening_charge` reconciliation idea, but scoped down — no `charge_reconciliation` modes needed unless a real predictor turns out to need one. |
| Implementation approach | Copy-then-adapt for the domain-agnostic harness *skeleton* only: `cli.py`'s subcommand structure, `config.py` (YAML + `--set` overrides), `runner.py`'s scaffolding (manifest writing, git-dirty check, MLflow tracking, `execute`/`_execute_inner` shape), `predictors/base.py`'s protocol, and `prepare_store.py`'s download-and-verify pattern. Write fresh, using the cosmo series only as a reference to read (not copy-paste): `data.py` (a much smaller `MoleculeSet`-equivalent — no profile/area trio), `metrics.py` (MAE/RMSE/max-abs-residual only), and the predictors themselves — `dash.py`'s tree-matching/back-off *algorithm* is worth porting deliberately (particularly the atom-map-order convention and the WL-style topology matching), but its profile-specific machinery (`LiteralTreeProperties`, sigma-grid handling) is not. |

## Architecture

### Directory layout

```
cosmo_experiments/           # existing, already renamed -- kept as reference
  sieve_experiments/
  tests/
  configs/
  docs/
  pins.toml
  README.md
  ...
charges_experiments/         # new, fully independent, top-level sibling
  charge_experiments/
    __init__.py
    cli.py                # run / prepare-store / summarize subcommands
    config.py              # YAML config loading, --set overrides
    data.py                 # AtomRecord/MoleculeSet-equivalent for scalar charge
    metrics.py               # MAE / RMSE / R2, charge-conservation check
    runner.py                 # execute/_execute_inner, manifest+MLflow plumbing
    prepare_store.py           # download dashMoleculesSDF_v2.sdf, parse to parquet
    _chalcedon/                 # vendored from cosmolayer.store._chalcedon (see NOTICE)
      __init__.py
      tanimoto_similarity.py
      butina_cluster.py
      greedy_cluster_split.py
      NOTICE
    predictors/
      __init__.py               # lazy registry (same pattern as cosmo's)
      base.py                     # Predictor protocol for scalar-per-atom output
      dash.py                       # ported tree-matching/back-off, charge target
      sieve_predictor.py             # SieveConfig(target_dim=1, ...)
  tests/
  configs/
  docs/
  pins.toml
  README.md
```

`pyproject.toml` gains a second set of the same reference kinds the cosmo
rename already needed (`packages.find`'s `where`, `testpaths`, ruff
`include`, `ty` `overrides`/`extra-paths`), pointed at
`charges_experiments/...` alongside the existing `cosmo_experiments/...`
entries. `.gitignore` gains
`charges_experiments/{runs,mlruns,cache,results}/` alongside the existing
`cosmo_experiments/...` entries.

### Vendoring `_chalcedon`

`cosmolayer` already vendors Butina clustering and a fraction-target
cluster split from `chalcedon` (rowansci/chalcedon) at
`cosmolayer/store/_chalcedon/` (`tanimoto_similarity.py`,
`butina_cluster.py`, `greedy_cluster_split.py`, plus a `NOTICE`
documenting the upstream commit and the one modification —
`butina_cluster`'s added `progress` parameter). Upstream `chalcedon`
itself pins `python>=3.14`; `cosmolayer` vendors it precisely to avoid
inheriting that floor. `charge_experiments` is a fully independent
series (per the code-sharing decision above) and has no reason to take
on `cosmolayer` as a dependency just for clustering/splitting — chaos-
store and `cosmolayer`'s `SegmentStore` play no role in this series'
data pipeline. So `charge_experiments/_chalcedon/` copies the same
three modules (plus `NOTICE`, with its provenance note extended to
name `cosmolayer`'s copy as the immediate source and the original
`chalcedon` commit as the ultimate origin) verbatim, unmodified from
`cosmolayer`'s copy. This is the first vendored-code instance in this
repo (`cosmolayer` itself is a real dependency elsewhere, pinned via
git in the root `pyproject.toml`; nothing is currently vendored
in-tree) — no existing in-repo convention to match beyond the
`NOTICE`-file shape `cosmolayer` itself already uses.

### Data pipeline

1. **Download** (`charge_experiments prepare-store` or similar): port
   `download_dash_molecules.sh`'s logic into Python, matching
   `prepare_store.py`'s existing conventions — stream in chunks, verify
   against the published size (8,278,301,584 bytes) and md5
   (`305f521c6b422546bdf09c1e87eb922d`), idempotent (skip if the parsed
   store already exists).
2. **Parse** (streaming `Chem.ForwardSDMolSupplier`, never loading the
   8.3GB file into memory at once): for each record —
   - Read the molblock's atoms/bonds/`M CHG` charges and the
     `>  <MBIScharge>` property block (pipe- or whitespace-delimited floats,
     one per atom, in molblock atom order).
   - If the mol's stereo isn't already fully specified from parity bits,
     call `Chem.AssignStereochemistryFrom3D` using the molblock's own 3D
     coordinates, then `Chem.AssignStereochemistry(cleanIt=True,
     force=True)`.
   - Write each atom's MBIS charge directly onto the `Mol` as a real
     atom property (`atom.SetDoubleProp("MBIScharge", value)`), so the
     target and the graph are the same object — no separate array, no
     atom-map-order bookkeeping.
   - Serialize the `Mol` (`Mol.ToBinary()`, which preserves atom
     properties and stereo/chiral tags) and emit one row: `chembl_id`,
     `conf_id`, `mol` (binary blob), `net_charge` (sum of the molblock's
     own `M CHG` values, kept as a plain column for the charge-
     conservation check without needing to deserialize every row).
3. **Cluster + split**: compute an achiral Morgan fingerprint directly
   from each unique `chembl_id`'s first-seen conformer `Mol` (no SMILES
   round-trip needed — `AllChem.GetMorganFingerprint(mol, radius,
   useChirality=False)` works straight on the `Mol`; any one conformer's
   connectivity suffices, since clustering is graph-level, not
   stereo-level), cluster with the vendored `_chalcedon.butina_cluster`,
   then assign clusters to train/val/test with the vendored
   `_chalcedon.greedy_cluster_split(cluster_ids, fractions)` (LPT
   scheduling by cluster size, no size-biasing goal this time — see
   "Vendoring `_chalcedon`" below), then join the `chembl_id -> split`
   assignment back onto every row (every conformer of a `chembl_id`
   inherits that `chembl_id`'s split).
4. Write `molecules.parquet` (one row per atom-bearing conformer: `chembl_id`,
   `conf_id`, `mol` binary blob, `net_charge`, `split`) plus a
   `split_summary.txt` alongside it, mirroring `prepare_store.py`'s
   existing pattern. Exact column layout is an implementation-plan detail;
   the binary-`Mol`-blob-as-row-payload shape is fixed by this spec.

### Predictors

**`DASHChargePredictor`**: same two-phase shape as `cosmo_experiments`'s
`DASHPredictor` — build a WL-depth-graded matching tree from train
molecules' topology, but each tree node stores a **scalar** charge
statistic (mean, and enough state for the same missing-value back-off
`get_property_noNAN` reproduces: deepest → shallowest, first populated
node wins, else the global mean) instead of a 51-bin profile array.
`predict_atoms` walks the same paths and returns one float per atom.
Deserializes each row's `Mol` directly from its binary blob and reads
`MBIScharge` straight off the atom (`a.GetDoubleProp("MBIScharge")`) when
building the tree — no atom-map-order decoding needed here either, since
`cosmo_experiments`'s version only needed that convention to recover atom
order from a plain SMILES string, which this series no longer stores.

**Sieve charge predictor**: same shape as `cosmo_experiments`'s
`SievePredictor`, but built on `from_rdkit(mols, y_from_atom_prop=
"MBIScharge", config=config, ...)` directly on the deserialized `Mol`
objects — no `from_smiles`, no separate `y` array. `SieveConfig(
target_dim=1, ...)` since the target is now a scalar, not a 51-dim
profile. Attribute set and `max_wl_depth`/`minimum_support` starting
values worth revisiting once real data is in hand — not fixed by this
spec.

### Metrics

`charge_experiments/metrics.py`: `regression_metrics`-equivalent
(MAE/RMSE/R²), computed at atom level directly (no molecule-level
roll-up needed as the primary metric, since the target *is* already
per-atom — unlike the cosmo series, there's no profile to sum first).
Unlike the cosmo series' `charge_metrics` (which skips R² because net
*molecular* charge clusters near zero and destabilizes `ss_tot`),
per-atom `MBIScharge` has enough spread that R² is informative, so it's
included here; `max_abs_residual` is dropped. A molecule-level
charge-conservation check (`sum(predicted atom charges)` vs. the
molblock's own total) is reported as a secondary diagnostic, not the
headline metric.

### Testing

Mirrors the cosmo series' layered approach: fast unit tests for pure
functions (`data.py`'s `Mol`-blob serialize/deserialize round trip and
`MBIScharge`-atom-property parsing, `metrics.py`, the `prepare_store.py`
glue that calls the vendored `_chalcedon.butina_cluster` /
`greedy_cluster_split`) needing only rdkit; a test for the new `sieve.io.rdkit_adapter
.from_rdkit`'s `y_from_atom_prop` option, alongside its existing tests in
the core `sieve` package's own test suite (not this series' tests) since
it's a core-library change; an `_optional` suite gated on the real
downloaded SDF (skipped if absent, matching how
`test_experiment_predictor_sieve_optional.py`/`_dash_optional.py` already
gate on the real chaos-store) for end-to-end fit/predict round trips; a
smoke test for `runner.py`'s `execute()` shape using synthetic data, no
real download needed.

## Out of scope (this spec)

- The GNN/Chemprop baseline for charges (deliberately deferred).
- Any second, size- or scaffold-biased split for the charges series
  (deliberately not building the cosmo series' `biased_split` equivalent
  here).
- Renumbering or otherwise touching the cosmo series' own milestone
  labels (T8-T13) — the earlier rename to `cosmo_experiments/` was a pure
  relocation, no content changes beyond path updates.
- The precise starting hyperparameters for either predictor (attribute
  set, `max_wl_depth`, `minimum_support`, DASH-tree's WL depth) — left to
  the implementation plan, to be tuned once real data is in hand.
