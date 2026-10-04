# MLPepper and THEMol as training-ready stores

Status: implemented on branch `other-stores` (2026-10-04; results at the end), with decisions D1-D8 (2026-10-03),
mirroring the DASH and SPICE subset stores (`docs/dash-subset-stores-plan.md`,
`docs/spice-stores-plan.md`). The analysis behind every curation decision is in sieve_paper (SI
section "Diagnostics and Curation of the Other Datasets", THEMol and MLPepper;
`data_analysis/mlpepper_*.py`, `data_analysis/themol_*.py`).

## Goal

Three training stores, built through two staging stores and readable by the existing harness
(`runner.load_molecule_set`, `cv.load_shards`, `cv_charges.sh`) without changes there:

- `mlpepper-vacuum` and `mlpepper-water`, from the MLPepper v1.1 singlepoint view (Zenodo
  15801339, CC BY 4.0), through `mlpepper-staging`. The two phases are computed on the same
  geometry and are used as separate datasets over the same entries, as SPICE's two kinds of
  conformation are.
- `themol` (name open, D3), from THEMol's MBIS subset through `themol-staging`, curated as in
  the manuscript's SI. THEMol has a single geometry per record, so there is one store.

## MLPepper

### Source

One SQLite file, `MLPepper-RECAP-Optimized-Fragments-v1.1_singlepoint_view.sqlite` (5.1 GB,
md5 `8182af6fa1bff7e55b0d030a21bc7ac7`), a view of a QCArchive singlepoint dataset: 75,097
entries x 2 specifications = 150,194 records. Records, entries and specifications are
zstd-compressed msgpack blobs (`records.record`, `dataset_entries.entry`,
`dataset_specifications.specification`), joined by `dataset_records(entry_name,
specification_name, record_id)`. Decoding needs `zstandard` and `msgpack`; the sieve venv has
`msgpack` but not `zstandard`, so both go into the `charges` extra of `pyproject.toml` (D5).
The download is pinned to the Zenodo record and checked against its md5, like SPICE's.

Each record holds the molecule (symbols, geometry in bohr, connectivity, charge, multiplicity,
the mapped SMILES), the status, and the properties: total energy, ddX solvation energy (water
only), SCF dipole, MBIS charges and dipoles, Mulliken and Lowdin charges, Mayer and
Wiberg-Lowdin indices, grid electron count, Psi4 version and wall time.

### Pipeline

```
parse_staging        one row per entry: the store Mol built by prepare_themol.build_record_mol
                     from the mapped SMILES and the (shared) geometry, stereo_check, whether the
                     QCArchive connectivity equals the SMILES graph, net_charge, the geometry
                     columns, collapse_key, canonical_smiles, graph_smiles, and the MBIS charges
                     of both phases as list columns (q_vacuum, q_water; null when the record is
                     not complete); record-fields.parquet holds, per entry and phase, the status,
                     total energy, solvation energy, SCF dipole, Mulliken and Lowdin charges
cluster_records      dash_subsets.cluster_records: Butina 0.65 over unique achiral graphs (56k)
curate               incomplete (an entry without a complete record in both phases), geometry
diagnose_pairs       every pair of entries of one structure (at most ten entries per structure,
                     44,935 pairs): heavy-atom RMSD with symmetry and reflection, sorted MBIS
                     discrepancy and energy difference in each phase
assign_split         dash_subsets.assign_split over the kept entries, points = collapse keys,
                     a single stratum; one cluster-level 90/10 split and N train shards
separate             mlpepper-vacuum and mlpepper-water: the same rows, the Mol of each store
                     carrying that phase's charges as MBIScharge
```

### Curation channels

1. **incomplete** -- an entry without a complete record in both phases: the five vacuum records
   that ended in Psi4's `could not converge MBIS`. Removed from both stores. Expected: 5 entries.
2. **geometry** -- `stretched_bond | close_contact`, unchanged. Expected: 0.

Not curated: the exact duplicate conformers (heavy-atom RMSD below 0.01 A, same energy and
charges): their labels agree, so they only weight the minimum they share. `diagnostics.json`
records the copy fraction at the MLPepper cutoff of 0.01 A, which the SI places at the minimum of
the pair density, and the pair quantiles per phase.

### Columns of a training store

`entry, smiles` (the mapped SMILES)`, mol, net_charge, stereo_check, connectivity_agrees`, the
five geometry columns, `collapse_key, cluster, split, shard`, and `n_collapsed, n_molecules,
n_enantiomer_forms`. The phase is the store, as the kind is for SPICE; `energy` and
`solvation_energy` stay in the staging store's `record-fields.parquet`.

## THEMol

### Source and the existing parse

Eight HDF5 files (31 GB), already downloaded to `/data/craabreu/THEMol/MBIS` and already parsed
into `stores/themol-mbis/molecules.parquet` (3,082,151 rows, with `stereo_check`, the geometry
columns, `collapse_key` and its counts, no split). The parse costs hours, so `parse_staging`
derives the staging store from an existing uncurated parse when one is given (`--parsed-path`,
default `stores/themol-mbis/molecules.parquet`), adding the columns the curation needs from the
stored Mol blobs (`charge_sum`, `sensitive` = contains B, Si or P, `canonical_smiles`,
`graph_smiles`, `skeleton`); it falls back to `prepare_themol.parse_themol` from the HDF5 files
(downloaded when absent) otherwise.

### Pipeline

```
parse_staging        one row per record, as above
cluster_records      Butina 0.65 over unique heavy-atom skeletons (D4); hours, cached
curate               charge_sum, multi_anion, geometry (in this order of precedence)
diagnose_pairs       every pair of records of one structure (304,608 pairs): heavy-atom RMSD with
                     symmetry and reflection, sorted MBIS discrepancy, same_molecule
assign_split         dash_subsets.assign_split, points = collapse keys, a single stratum
write                themol/molecules.parquet with split, shard, cluster
```

### Curation channels

1. **charge_sum** -- a record whose MBIS charges do not add up to its net charge within 0.01 e.
   Expected: 1 record.
2. **multi_anion** -- a record with a net charge of -3 or below, or of -2 if it contains B, Si or
   P. THEMol computed its charges in vacuum with def2-TZVPD; the SI shows, from THEMol's own
   protonation series and from the structures shared with MLPepper, that the charge of a P, Si or
   B centre collapses on the second deprotonation while C, N and S change smoothly. Expected:
   2,481 records.
3. **geometry** -- `stretched_bond | close_contact`, unchanged. Expected: 3,124 raised, 2,751
   removed after the channels above.

Expected survivors: 3,076,918 records of 2,990,185 molecules; 8,688 at a net charge of -2.
Duplicates (one molecule under several uuids, 81,819 molecules) and copies (88,797 pairs below
0.17 A) are kept, as for MLPepper.

### Columns of the training store

`themol_id, h5_file, smiles, mol, net_charge, stereo_check`, the five geometry columns,
`collapse_key, cluster, split, shard`, and the three collapse counts. `charge_sum` and
`sensitive` stay in the staging store.

## Shared work items

1. `experiments/mlpepper_store.py`: download (md5-pinned), sqlite decoding, `parse_staging`,
   `curate`, `diagnose_pairs` (reusing `spice_subsets._pair_task` generalised to several charge
   sets and energies), `separate`, `prepare_mlpepper_stores` orchestration, idempotent per stage.
2. `experiments/themol_store.py`: `parse_staging` from the existing parse or the HDF5 files,
   `curate`, `diagnose_pairs`, `prepare_themol_store_curated` orchestration.
3. `dash_subsets.assign_split`: already takes a `subset` column as stratum; a single-stratum
   call needs no change. `cluster_records` gains a keyword for the column to fingerprint by (D4).
4. CLI: `prepare-mlpepper-stores` and `prepare-themol-stores` (names open), with `--sqlite-path`
   / `--parsed-path`, `--n-shards`, `--workers`, `--limit`.
5. `cv_charges.sh`: the `case` on `CV_STORE` gains the three stores and a `CV_MLPEPPER_SQLITE`
   path, as `CV_SPICE_HDF5`; depths and shard counts are DASH's defaults, to be re-read in Study A.
6. `pyproject.toml`: `zstandard>=0.22`, `msgpack>=1.0` in `charges`.
7. Tests, fast suite: a synthetic sqlite view (two entries x two specs, zstd-msgpack blobs, one
   vacuum record in error) and a synthetic THEMol HDF5 (a P dianion, a C dianion, a trianion, a
   record with a wrong charge sum, a stretched bond) exercising parse, curation, pairs, split and
   separation; `build_parser` defaults.
8. README: the two new commands beside `prepare-spice-subsets`.
9. Paper side (follow-up, sieve_paper): point `data_analysis/mlpepper_*.py` and `themol_*.py` at
   the staging stores and check that the SI figures regenerate unchanged.

## Decisions (2026-10-03)

- **D1** Branch: `other-stores` off `spice-stores` (PR #52 open), since the builders reuse
  `prepare_themol.build_record_mol`, `spice_subsets` and `dash_subsets`; PR #53 would target
  `spice-stores` or wait for #52 to merge.
- **D2** MLPepper names: `mlpepper-staging`, `mlpepper-vacuum`, `mlpepper-water`.
- **D3** THEMol names: `themol-staging` and `themol` for the curated training store, leaving
  `themol-mbis` as the uncurated parse that the paper's scripts read (as `dash-molecules` and
  `spice-2` were left). Alternative: rebuild `themol-mbis` in place as the curated store.
- **D4** THEMol clustering: plain Butina over the 2,004,504 unique heavy-atom skeletons (the graph
  without H, charges, bond orders, aromaticity and stereo, `themol_protonation._skeleton` in
  sieve_paper), fingerprinted with the same achiral Morgan radius 2, 2048 bits, cutoff 0.65. A
  protonation series then shares a cluster by construction. Measured on this machine (32 cores)
  with the vendored implementation: 50k skeletons 13.5 s, 100k 42.8 s, 200k 138 s, 7.2 GB peak,
  scaling as n^1.7, so the full run is 2-4 h and about 70 GB; the singleton fraction falls with n
  (52%, 46%, 41%), so the clustering is doing real work, and the cluster count grows as about
  sqrt(n), the largest cluster near 1% of the molecules. Sampling (Butina on a subset, the rest
  assigned to the nearest centre) was considered and is unnecessary at this cost. MLPepper's 56k
  graphs take the plain `dash_subsets.cluster_records`.
- **D5** Dependencies: `zstandard` and `msgpack` added to the `charges` extra, so the builder
  reads the sqlite view directly instead of depending on sieve_paper's extraction.
- **D6** THEMol curation exactly as decided for the SI (charge sum 0.01 e; Q <= -3, or Q = -2
  with B, Si or P; geometry flags); MLPepper: the five incomplete entries and the geometry flags.
- **D7** Pairs: all pairs within a structure for both datasets (as the SI did; 45k and 305k
  pairs), not SPICE's random 100 per structure.
- **D8** Shards: 50 for all three stores, as for DASH and SPICE, subject to `cluster-report`.

## Cost estimate

MLPepper: decoding 150k blobs and building 75k Mols, minutes with 32 workers; clustering 56k
graphs, seconds; 45k pairs, a minute. THEMol: the augment step over 3.08 M Mol blobs, about 10
min with 48 workers; 305k pairs, about 15 min; clustering 2-4 h (D4).

## Results of the real builds (2026-10-04)

MLPepper (43 s, 32 workers) reproduces the SI: 75,097 entries, 5 incomplete, 0 geometry, 75,092
kept; 44,935 pairs, 6,860 copies below 0.01 A; 4,183 clusters; split 90.1/9.9, 50 shards.

THEMol (3 h 25 min, 48 workers, almost all of it the skeleton Butina) reproduces the SI curation
(charge_sum 1, multi_anion 2,481, geometry 3,124 raised / 2,751 removed, 3,076,918 kept, 8,688
at Q <= -2) and pairs (304,608 pairs, 88,797 copies); 2,004,504 skeletons in 47,585 clusters;
split 90.1/9.9 by structure, 50 shards of 50,361-50,362 train structures. The molecule count
differs from the SI's first draft (2,987,645 against 2,990,185) because the store identifies a
molecule by the canonical SMILES of the stored Mol, whose stereo is perceived from 3D, rather than
of the deposited SMILES; the SI was updated to the store's count.
