# SPICE 2: high-energy and low-energy conformations as two training-ready stores

Status: implemented on branch `spice-stores` (`spice_subsets.py`, CLI `prepare-spice-subsets`),
mirroring the DASH subset stores (`docs/dash-subset-stores-plan.md`). The analysis behind every
decision is in sieve_paper (SI section "Diagnostics and Curation of the Other Datasets", SPICE 2;
`data_analysis/spice_*.py`).

## Goal

Two stores, `spice-high-energy` and `spice-low-energy`, built from `SPICE-2.0.1.hdf5` through a
shared `spice-staging` store, curated as in the manuscript's SI, split into train/test with
cluster-clean CV shards, and readable by the existing harness without changes there.

## The two kinds of conformation

SPICE generated 50 conformations per molecule: 25 snapshots of molecular dynamics at 500 K and,
from each, a low-energy conformation made by five L-BFGS iterations and 1 ps of dynamics at 100 K.
The generation script (`pubchem/createPubchem.py` of github.com/openmm/spice-dataset) stores the
snapshots as conformations 0-24 and the conformation relaxed from snapshot `c` as `c + 25`. The
downloader groups the QCArchive records by entry name, so `SPICE-2.0.1.hdf5` lists a molecule's
conformations in the order of these indices sorted as text: position `k` of a group of 50 holds
generation index `TEXT_ORDER[k] = sorted(range(50), key=str)[k]`. On the real file, 98.8% of the
molecules whose energies and forces split them 25/25 follow exactly this pattern, and a relaxed
conformation lies below its own snapshot in energy in 99.5% of cases. The kinds were sampled
differently and are used as separate datasets: no pair, curation test or store mixes them.

## Pipeline

```
parse_staging        single-molecule groups only (prepare_spice._parse_one_group), plus the
                     deposited group size, generation index, sampling, collapse_key,
                     canonical_smiles, graph_smiles; record-fields.parquet holds the total energy
                     and the largest component and RMS of the force
cluster_records      dash_subsets' Butina pass over unique achiral graphs
curate               incomplete molecules, then geometry flags
diagnose_pairs       at most 100 random pairs per structure and kind, kept conformations only:
                     heavy-atom RMSD (symmetry and reflection), sorted MBIS discrepancy, energy
                     difference -- the data of the SI pair figures
assign_split         dash_subsets.assign_split with the kind as the stratum; one cluster-level
                     90/10 split and 50 train shards shared by both stores
separate             spice-high-energy and spice-low-energy
```

Each stage writes its own file in `spice-staging` and is skipped when that file exists.

## Curation channels

1. **incomplete** -- every conformation of a molecule whose group does not hold exactly 50
   conformations, or of which the parse could not keep all 50. The downloader keeps only the
   calculations that completed and drops conformations with a force component above
   1 hartree/bohr (`downloader/config.yaml`, `max_force: 1.0`); the one group of 100 holds two
   generations of one molecule. The text-order mapping identifies the kinds only in a group of
   exactly 50. Expected: 307 molecules, 12,116 conformations.
2. **geometry** -- `stretched_bond | close_contact` of `experiments.geometry`, unchanged from DASH
   (1.25; 1.05 for atoms sharing a neighbour; 1.15 for atoms three or more bonds apart). Expected:
   1,111 conformations of 102 molecules.

Not curated, on the evidence in the SI: sibling-based charge cuts (no independent charge evidence,
as for DASH/Extra), energy outliers, and anything based on the deposited bond indices (both the
Wiberg-Lowdin and the Mayer indices misbehave in the diffuse def2-TZVPPD basis). The legacy 0.4 e
rule of `prepare_dash.curate_conformers` is not applied.

## Columns of a training store

`spice_id, conf_id, subset` (the SPICE subset name)`, smiles, mol, net_charge, stereo_check`, the
five geometry columns, `generation, sampling, collapse_key, cluster, split, shard`, and the counts
`n_collapsed, n_molecules, n_enantiomer_forms`.

## Decisions (2026-10-02)

- Only molecules with all 50 conformations are kept.
- High-energy and low-energy conformations are separate datasets.
- Curation channels: completeness and the DASH geometry flags, nothing else.
- One split shared by the two stores, so a molecule's two kinds never fall on opposite sides of a
  train/test boundary; 50 shards, as for DASH.
- The THEMol and SPICE builder commits of `themol-store` come along, since `prepare_spice` builds
  records with `prepare_themol.build_record_mol`.
