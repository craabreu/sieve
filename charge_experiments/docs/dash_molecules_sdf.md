# `dashMoleculesSDF_v2.sdf`: findings from the first real parse

Working notes from actually running `prepare-store` against the real,
8.3GB published SDF for the first time (2026-08-27). Kept here as a
standing record for whoever next touches `prepare_store.py` or reruns the
full pipeline -- everything below was found empirically, against the real
file, not assumed from the spec.

## The download needs a browser User-Agent

`download_dash_sdf`'s plain `urllib.request.urlopen(url)` gets a
**403 Forbidden** from the ETH Research Collection server -- it rejects
urllib's default User-Agent (`python-urllib/x.y`). The original
`download_dash_molecules.sh` bash script already worked around this
(`UA='Mozilla/5.0 ...'`); `prepare_store.py` didn't port that part over.
Fixed by sending the same UA string via `urllib.request.Request(url,
headers={"User-Agent": ...})` -- verified with a live request (`status:
200`) before touching the real 8.3GB download.

## `ForwardSDMolSupplier` auto-attaches every SDF property, not just the ones you read

Each SDF record's `>  <TAG>` blocks all get attached to the parsed `Mol`
as mol-level string properties automatically -- not just `CHEMBL_ID`/
`CONF_ID`/`MBIScharge`, which is all `_parse_one_record` ever reads, but
every `GFN2:*`/`DFT:*` quantum-chemistry field too (total/atomic/formation
energies, dipole/quadrupole moments, rotational constants, HOMO/LUMO,
Mulliken and Loewdin charges, bond orders, polarizability, dispersion
coefficients, ~40 fields in total). `mol_to_blob`'s
`PropertyPickleOptions.MolProps` serialized *all* of them into every
stored row's blob, unused -- `chembl_id`/`conf_id`/`net_charge` already
live in their own parquet columns, and nothing ever reads them back off a
deserialized `Mol`.

Measured impact on a real, in-progress parse: **output/input-consumed
ratio dropped from ~44.6% to ~8.0%** (a ~5.6x reduction) once
`_parse_one_record` clears every mol-level property (`mol.ClearProp(name)`
for `name in mol.GetPropNames()`) right before serializing. Atom-level
`MBIScharge` is unaffected -- it lives on the `Atom` objects, not the
`Mol`'s own property dict, and clearing mol-level props doesn't touch it.

Final `molecules.parquet` for the ChEMBL-only cohort (before the second
cohort below was added) was 537,303,775 bytes for 518,669 conformers, vs.
an extrapolated ~3.7GB if the bloat had been left in -- confirmed by
actually letting one (later-discarded) unfixed run finish parsing.

## The SDF holds two distinct record schemas, not one

Roughly half the file's records fail the `_parse_one_record` schema check
(missing `CHEMBL_ID`) -- not because they're malformed, but because
they're a **different, equally legitimate record schema**:

- **Records 1-518,669**: `CHEMBL_ID`/`CONF_ID`/`MBIScharge` -- the schema
  the parser was originally built against, matching the sample record
  inspected at the very start of the file.
- **Records 518,670-1,029,785** (~511,116 records, ~49.6% of the file): no
  `CHEMBL_ID`, no `CONF_ID`, but `DASH_IDX` (e.g. `"Rest_2"`),
  `MBIS_CHARGES` (identical values to `MBIScharge`, just duplicated under
  a different name), `MBIS_Energy`, `XTB_Energy`, `XTB_MulikenCharge`.

Confirmed against the DASH paper's own stated dataset composition
(Lehner et al., arXiv:2305.15981): the training set was assembled from
**four sources**, not just ChEMBL --

> "we generated an extended data set by collecting and filtering
> molecules from four different sources: (i) the QMugs data set, (ii) the
> training set from Ref. [22], (iii) lead-like molecule from ChEMBL
> version 30 (filtered as in Ref. [22]), and (iv) organic liquids from
> Refs. [35]-[38]."

> "The data set of 393,692 unique molecules (three conformers each, i.e.,
> 1,076,252 3D structures in total) was split randomly into a 90% subset
> for training..., while the remaining 10% (100,171 3D structures) served
> as validation set."

Only source (iii) is ChEMBL; sources (i)/(ii)/(iv) have no ChEMBL
identifier, which is exactly what `DASH_IDX = "Rest_N"` most plausibly
means -- the rest of the sources beyond ChEMBL. Sampling 3,000 of the
`DASH_IDX`-only records found ~1,012 unique `DASH_IDX` values, each shared
by ~3 rows -- the same "a few conformers per molecule" pattern
`CHEMBL_ID` rows show, confirming `DASH_IDX` plays the identical
per-molecule grouping role. `_Name` is empty for every sampled
`DASH_IDX`-only record, so it isn't a usable fallback identifier.

The paper's stated total (1,076,252 3D structures) vs. this SDF release's
actual total (1,029,785 records, both schemas, before any parse-time
skips) leaves a ~4.3% gap -- not reconciled; plausibly some records fail
RDKit sanitization entirely (`mol is None`, skipped silently, no warning
currently logged for that specific case) or this release differs slightly
from what was originally published.

### Decision: include both cohorts

Initially the parser only accepted `CHEMBL_ID` rows, which meant the
store held under half of DASH's own published training data. Extended to
accept both schemas:

- `MoleculeSet`/the parquet store now carry **both** `chembl_id` and
  `dash_id` columns -- exactly one is set per row, the other `None`.
- `DASH_IDX`-only records get a synthesized sequential `conf_id`
  (`dash_conf_counters`, keyed by `DASH_IDX`, hands out `"conf_0"`,
  `"conf_1"`, ... per group in file order), since that schema has no
  `CONF_ID` of its own. `conf_id` is purely informational downstream (no
  predictor or metric reads it back), so a synthesized label serves
  exactly as well as a real one.
- `assign_splits` clusters/splits by a coalesced `mol_key =
  chembl_id.fillna(dash_id)` instead of `chembl_id` alone, so a
  `DASH_IDX`-only molecule's conformers never span two splits either.

## Final real-corpus store stats

After including both cohorts, `prepare-store dash-molecules` against the
real file:

```
       n_conformers  n_molecules  fraction
split
train        823722       279148  0.799897
val          103030        34894  0.100050
test         103033        34893  0.100053
```

**1,029,785 total conformers** across **348,935 unique molecules** --
every record that survived `_parse_one_record` (both schemas), split with
fractions landing almost exactly on the requested 0.8/0.1/0.1 target.
`molecules.parquet`: 960,684,305 bytes. Verified no molecule's conformers
span two splits, in either identity scheme (see
`test_assign_splits_never_splits_a_chembl_id_across_splits`/
`..._never_splits_a_dash_id_across_splits`/
`..._handles_a_mixed_store_of_both_schemas` in
`charge_experiments/tests/test_charge_prepare_store.py`).

## Different conformers of the same molecule really do get different stereochemistry

The design spec's rationale for assigning stereochemistry independently
per conformer (from that conformer's own 3D coordinates, rather than once
per molecule from a shared 2D graph) is empirically real in this corpus,
not just a hypothetical edge case. Sampled 5,000 randomly-chosen
multi-conformer molecule groups (mixed across both identity schemes) and
compared each conformer's isomeric SMILES:

**386 of 5,000 groups (~7.7%) have conformers with genuinely different
stereochemistry.** Every difference found in this sample is a double-bond
(E/Z) geometry flip -- a `/`<->`\` change around a C=N, N=N, or C=C
linkage (imines, hydrazones, amidines) -- never an R/S chiral-center
mismatch or a connectivity difference. Examples (mol_key: differing
fragment):

- `CHEMBL1309432`: `/N=N/` vs. `/N=N\` on a diazo linkage
- `CHEMBL1467505`: same pattern on a thiohydrazide
- `CHEMBL2315738`, `CHEMBL3195155`, `CHEMBL3335241`, `CHEMBL3900264`,
  `CHEMBL4207103`: the same E/Z-flip pattern on their own imine/hydrazone
  bonds

This is exactly why clustering/splitting is done on **achiral**
fingerprints (`useChirality=False`): it's what keeps a molecule's
stereoisomeric conformer pairs grouped together in the same split, rather
than risking a train/test leak where the model sees one stereoisomer at
train time and a materially different one (different E/Z geometry, not
just noise) at test time.
