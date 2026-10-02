# Plan: DASH/QMugs and DASH/Extra as two training-ready stores

Status: approved with decisions D1-D5 (2026-10-01; see the end); implemented on branch
`dash-subset-stores` (`dash_subsets.py`, `dash_diagnostics.py`, `dash_curation.py`, CLI
`prepare-dash-subsets`).

## Goal

Replace the single `dash-molecules` store with two stores, `dash-qmugs` and `dash-extra`, built by
`prepare-store` from the one deposited SDF. Each store is curated as described in the manuscript
(sieve_paper, "The DASH Dataset"), split into train/test with cluster-clean CV shards, and readable
by the existing harness (`runner.load_molecule_set`, `cv.load_shards`) without code changes there.

Ordering constraints:

1. **Clustering before separation.** One Butina pass over the whole corpus, so both stores carry
   cluster ids from the same clustering.
2. **Splitting before separation (D1).** One cluster-level train/test split and one shard
   assignment over the whole corpus, so that a cluster, and therefore every structure deposited in
   both subsets, has the same split and shard in both stores. A model fitted on one store can then
   be tested on the other without leakage.
3. **Curation before splitting**, as today: the split is computed on the population that survives
   curation. Curation is defined per subset (geometry flags for both, copy rule and ESP cut for
   DASH/QMugs), but it runs on the combined staging store, before the split.

## Current state (branch `themol-store`)

`prepare_dash.prepare_store`: download -> `parse_dash_molecules` (stereo from 3D, geometry columns,
all SDF tags discarded) -> optional `molecules.parquet.uncurated` -> `curate_conformers` (0.4 e
isolated-atom rule) -> `assign_splits` (Butina 0.65 on achiral Morgan fingerprints of the first
conformer per `dash_id`, `greedy_cluster_split` 90/10, `n_shards` train shards). `collapse_key` is a
separate in-place step (`annotate-collapse`).

What the paper's curation needs that the store does not have:

- `DFT:ESP_AT_NUCLEI` per atom (DASH/QMugs only), dropped at parse.
- The subset label (derivable from the `DASH_IDX` prefix: `QMUGS500_*` vs `Rest_*`).
- `collapse_key` at parse time (the copy test groups by structure within a subset).
- Pair RMSD (symmetry-aware, reflection allowed), the charge discrepancy, triangles of copies, the
  copy rule, and the ESP score -- all implemented only as analysis scripts in sieve_paper
  (`pair_rmsd.py`, `pair_charges.py`, `qmugs_curation.py`, `curation2.esp_score`).

## Target pipeline

```
download_dash_sdf
  -> parse_dash_molecules        one pass over the SDF, all 1,029,785 records, into the staging store
  -> cluster_dash_molecules      Butina over unique achiral graphs, whole corpus -> `cluster` column
  -> diagnose (per subset)       pair RMSD -> copies; charge discrepancy for every pair
  -> curate (per subset)         geometry flags (both); copy rule + ESP cut (DASH/QMugs only)
  -> assign_splits               one train/test split and 50 shards over the surviving corpus
  -> separate_dash_subsets       dash-qmugs and dash-extra, each with split, shard and cluster
  -> write reports               split, curation and diagnostics summaries per store
```

### Store layout

Each stage writes its own file in the staging store and is skipped when that file exists, so a
change to one stage reruns only that stage and the ones after it (delete their files).

```
stores/dash-staging/
  dashMoleculesSDF_v2.sdf       (or --sdf-path)
  parsed.parquet                every record: training columns + subset, collapse_key,
                                canonical_smiles, graph_smiles
  record-fields.parquet         per record, the deposited fields of the D3 table, row-aligned
  clusters.parquet              Butina cluster per record (one pass over the corpus)
  pairs.parquet                 per pair within a structure and subset: RMSD, mirror flag,
                                MBIS and population-charge discrepancies, energy differences,
                                largest bond-length difference
  structures.parquet            per collapse key and subset: records, rotatable bonds
  curation.parquet              per record: curation_step ("" when kept), esp_score
  split.parquet                 per surviving record: split, shard
  diagnostics.json              pair, copy and curation counts
stores/dash-qmugs/, stores/dash-extra/
  molecules.parquet             curated, split; training-ready
  curation_summary.txt, split_summary.txt, diagnostics.json, built-at-commit.txt
```

Per-atom arrays stay out of `molecules.parquet`: `load_molecule_set` carries every non-`mol`
column as an identifier, and list columns of about 40 atoms per row would cost memory in every shard
fit for no training use. The training stores keep scalar provenance columns only (below); the
staging store keeps the rest, row-aligned with `parsed.parquet`, and records why each removed row
went in `curation_step` (`""`, `geometry`, `copy rule, witness`, `copy rule, ESP`, `ESP cut`).

**What the staging store keeps (D3).** Everything needed to rebuild the data-section figures of the
manuscript (pair-rmsd, pair-consistency, geometry-ratios, copy-rule, esp-curation,
conformers-per-structure, and the SI dipole-consistency figure) without re-reading the SDF:

| Field | Subset | Used by |
|---|---|---|
| `DFT:ESP_AT_NUCLEI` (per atom) | QMugs | ESP score, esp-curation |
| `DFT:LOWDIN_CHARGES`, `DFT:MULLIKEN_CHARGES` (per atom) | QMugs | pair-consistency, esp-curation |
| `GFN2:MULLIKEN_CHARGES` / `XTB_MulikenCharge` (per atom) | both | pair-consistency, copy-rule, esp-curation |
| `DFT:TOTAL_ENERGY`, `GFN2:TOTAL_ENERGY` | QMugs | pair-consistency |
| `MBIS_Energy`, `XTB_Energy` | Extra | pair-consistency |
| `DFT:DIPOLE` | QMugs | dipole-consistency (SI) |
| geometry columns (already parsed) | both | geometry-ratios |
| rotatable bonds per structure (computed) | both | pair-rmsd |

### Columns of a training store

Existing: `chembl_id, conf_id, dash_id, mol, net_charge`, the five geometry columns, `split`,
`cluster`, `shard`. Added: `subset` (`QMugs`/`Extra`), `collapse_key`, `collapse_smiles` and the
counts `annotate_collapse` writes today, and for DASH/QMugs `esp_score` (float). Removed records are
physically absent from the training stores and listed, with their reason, in the staging store.

## Work items

### 1. Parse (prepare_dash.py)

- Keep the fields of the D3 table and write them to `record-fields.parquet`, aligned by row; verify
  every per-atom list against the atom count, as `MBIScharge` is verified.
- Add `subset` from the `DASH_IDX` prefix; refuse any third prefix.
- Compute `collapse_key`/`collapse_smiles` during parse, as `prepare_themol` already does, and drop
  the separate `annotate-collapse` step for DASH (keep the command for old stores).

### 2. Cluster before separation (new `cluster_dash_molecules`)

- Fingerprint one representative per unique achiral graph (achiral canonical SMILES), not per
  `dash_id`: identifiers mix structures (8.98%), and identical graphs are then one point by
  construction rather than by Butina's tie behaviour. Map clusters back to every row.
- Same fingerprint and cutoff as today (`_achiral_fingerprints`, radius 2, 2048 bits, 0.65).
- The 3,952 structures deposited in both subsets share a cluster id and, since the split is taken
  over clusters before separation, the same split and shard in both stores.

### 3. Diagnose (new module `dash_diagnostics.py`, ported from sieve_paper)

- `pair_rmsd`: every pair of records sharing a `collapse_key` and a subset, heavy-atom RMSD via
  `rdMolAlign.GetBestRMS` against the record and its mirror (`collapse.mirror_mol`), minimum kept;
  spawn pool as in `geometry.annotate_geometry`. About 0.55 M pairs per subset.
- `COPY_RMSD = 0.17` angstrom splits copies from distinct conformers.
- `charge_discrepancy`: for each pair, sort the charges of each equivalence class
  (`collapse.aligned_values`) and take the largest difference (the manuscript's eq. charge
  discrepancy), for the MBIS charges and every population-charge set the pair carries; the absolute
  energy differences for every energy it carries; and the largest bond-length difference
  (`pair_bond_lengths.py`).
- Rotatable bonds per structure (RDKit default definition, hydrogens removed) into
  `structures.parquet`.
- Write `pairs.parquet`; `diagnostics.json` records the pair counts, copy fractions overall and for
  structures with at most two rotatable bonds, and the distinct-conformation counts.

### 4. Curate (new module `dash_curation.py`)

Both subsets:

- **Geometry flags**: drop rows with `stretched_bond | close_contact` (thresholds already in
  `geometry.py`). Expected: 0 in DASH/QMugs, 1,007 in DASH/Extra.

DASH/QMugs only (port of `sieve_paper/data_analysis/qmugs_curation.py`):

- **Triangle bounds**: from all triangles of mutual copies, `agree` = 99.9th percentile of the
  smallest side, `disagree` = its maximum. Computed, never hard-coded. Expected 0.078 / 0.179 e
  from 11,163 triangles.
- **ESP score**: per-atom potential of the other MBIS charges, `V_i = sum_j q_j / r_ij` (a.u.);
  robust least squares `phi_i ~ a_k + b_k q_i + c_k V_i` per atom class (element, degree, bonded
  hydrogens, aromaticity, formal charge; >= 300 atoms, else per element), trimmed at four robust
  sigmas over four iterations, fitted on all DASH/QMugs records; residual divided by the robust
  scale of its element; score = RMS over the record's atoms. Port `robust_lstsq`, the potential
  kernel and `esp_score` from `curation2.py`; derive the atom class from the stored `mol` rather than
  from `atom_attributes.py`'s cache.
- **Copy rule**: for copies beyond `disagree`, a third copy agreeing (< `agree`) with one member and
  disagreeing (> `disagree`) with the other condemns the other; with no single verdict, the member
  with the higher ESP score goes. Expected 809 disagreeing pairs, 265 witnessed, 666 removed.
- **ESP cut**: the 99.95th percentile of the score over healthy copies (in an agreeing pair, not
  condemned); remove every record above it without an agreeing copy. Expected cut 3.96, 3,268
  removed; 3,934 in total (0.76%).
- Record in `diagnostics.json`: the bounds, triangle count, witness counts, ESP agreement with the
  witnesses (265/265), detection of witnessed failures (84%), the transfer check (3,012 predicted vs
  3,217 observed), and the distinct-conformer tail before and after (1.09% -> 0.12% beyond 0.271 e).

DASH/Extra: nothing beyond the geometry flags (decided 2026-10-01).

The old `curate_conformers` (0.4 e rule) is no longer called on these stores; keep it for the legacy
`dash-molecules` store until that store is retired.

### 5. Split before separation (prepare_dash.assign_splits)

- Generalise `assign_splits` to take precomputed cluster ids (a `cluster` column) instead of
  re-clustering, and to group by `collapse_key` rather than `dash_id`, so that no structure spans two
  splits even when its records sit under several identifiers.
- One 90/10 train/test split and 50 train shards over the surviving corpus (D2: 50 per store, which
  a shared assignment gives both stores at once).
- Balance per store, not only overall: `greedy_cluster_split` balances the total count, so a cluster
  rich in one subset could skew that subset's fractions. Assign clusters with a two-component target
  (DASH/QMugs and DASH/Extra row counts) and report both stores' train/test fractions and shard sizes;
  `cluster-report` checks the per-store shard balance before the split is written.

### 6. Separate (new `separate_dash_subsets`)

- Write each subset's surviving rows, with `split`, `shard`, `cluster`, to its own store directory as
  `molecules.parquet`. Pure row filter; asserts that the two row sets partition the survivors.

### 7. Orchestration and CLI

- `prepare_dash_subsets(stores_root, *, staging="dash-staging", stores=("dash-qmugs", "dash-extra"),
  n_shards=..., sdf_path=None)`: idempotent per stage, keyed on the files above; refuses to re-split
  a store whose curation changed, as `prepare_store` does today.
- CLI: `prepare-store` gains `--subsets` (or a new `prepare-dash-subsets` command); `cluster-report`
  accepts the per-subset uncurated stores. Configs: add `dash-qmugs`/`dash-extra` examples; leave the
  `dash-molecules` configs untouched.

### 8. Tests

- Unit: subset label from prefix; record-field parse and count checks; clustering of identical achiral graphs
  into one cluster; separation partitions the rows; pair RMSD of a molecule and its mirror is zero;
  charge discrepancy invariant to permuting equivalent atoms; triangle bounds on a synthetic set;
  copy rule on synthetic triangles (witnessed, conflicting, unwitnessed with ESP tie-break); ESP fit
  recovering known coefficients on synthetic data; split keeps every `collapse_key` and cluster
  within one split, and a structure in both subsets gets the same split and shard in both stores.
- Integration (optional suite, like `test_prepare_dash_optional.py`): run the whole pipeline on a
  small SDF excerpt holding both schemas.
- Acceptance on the real file: the counts in `diagnostics.json` reproduce the manuscript's numbers
  listed in section 5, within the tolerance of Butina-independent quantities (they should match
  exactly, since curation does not depend on clustering).

### 9. Documentation

- README "adding a dataset" and `docs/dash_molecules_sdf.md`: the two schemas, the staging store,
  and the new stores.
- Paper side (sieve_paper): point the `data_analysis` scripts at `dash-staging` and check that every
  data-section figure regenerates unchanged from it.

## Decisions (2026-10-01)

- **D1** Clustering and splitting both precede separation: one cluster-level split and shard
  assignment shared by the two stores.
- **D2** 50 shards per store.
- **D3** The staging store keeps whatever the manuscript's figures need (table above).
- **D4** Names approved: `dash-staging`, `dash-qmugs`, `dash-extra`.
- **D5** Base the work on main. `themol-store` is main plus five commits with main unchanged since,
  so there is nothing to rebase; the work branch starts from main and brings in only the geometry
  columns (`a2f0225`) and the identity-column option of `annotate_collapse` (`f247ee5`), unless the
  THEMol and SPICE commits should come along.

## Cost estimate

One SDF parse (about 20 min, as today), one Butina pass (minutes on unique graphs), pair RMSD for
about 1.1 M pairs (about 10-20 min with 32 workers, as in sieve_paper), the ESP fit and copy rule
(minutes). The heavy steps are cached per stage, so a change to the curation re-runs only steps 4-6.
