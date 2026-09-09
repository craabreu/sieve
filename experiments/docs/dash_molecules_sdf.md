# `dashMoleculesSDF_v2.sdf`: findings from the first real parse

Working notes from actually running `prepare-store` against the real,
8.3GB published SDF for the first time (2026-08-27). Kept here as a
standing record for whoever next touches `prepare_dash.py` or reruns the
full pipeline -- everything below was found empirically, against the real
file, not assumed from the spec.

## The download needs a browser User-Agent

`download_dash_sdf`'s plain `urllib.request.urlopen(url)` gets a
**403 Forbidden** from the ETH Research Collection server -- it rejects
urllib's default User-Agent (`python-urllib/x.y`). The original
`download_dash_molecules.sh` bash script already worked around this
(`UA='Mozilla/5.0 ...'`); `prepare_dash.py` didn't port that part over.
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

- **Records 1-518,669**: `CHEMBL_ID`/`CONF_ID`/`MBIScharge` plus the ~42
  `GFN2:*`/`DFT:*` quantum-chemistry fields -- the schema the parser was
  originally built against, matching the sample record inspected at the
  very start of the file.
- **Records 518,670-1,029,785** (~511,116 records, ~49.6% of the file): no
  `CHEMBL_ID`, no `CONF_ID`, no `GFN2:*`/`DFT:*` block, but
  `MBIS_CHARGES` (identical values to `MBIScharge`, just duplicated under
  a different name), `MBIS_Energy`, `XTB_Energy`, `XTB_MulikenCharge`.

**`DASH_IDX` is on *both* schemas -- it is not what distinguishes them.**
A full-file tag scan (49 distinct tags in total) settles this:

```
1029785  <MBIScharge>, <DASH_IDX>          <- every record
 518669  <CHEMBL_ID>, <CONF_ID>, 42x GFN2:*/DFT:*
 511116  <MBIS_CHARGES>, <MBIS_Energy>, <XTB_Energy>, <XTB_MulikenCharge>
```

`DASH_IDX` takes **348,935 distinct values -- exactly the corpus's unique-
molecule count** -- so it is the universal molecule key, and its prefix
encodes the source cohort: `QMUGS500_*` on the 518,669 `CHEMBL_ID`-bearing
records, `Rest_*` on the other 511,116, no third prefix. QMugs is itself
built from ChEMBL, which is why those molecules carry both identities; the
two keys agree 1:1 (176,969 unique `CHEMBL_ID`s against 176,969 unique
`QMUGS500_*` ids). So `Rest_N` is not "everything that is not ChEMBL" --
it is one named cohort of two.

An earlier revision of `_parse_one_record` read the two identities as
mutually exclusive (`if CHEMBL_ID ... elif DASH_IDX`) and so wrote
`dash_id = None` on half the corpus, losing the source-cohort label
entirely. Fixed: `DASH_IDX` is now read unconditionally. **Stores built
before that fix have `dash_id` NULL on every ChEMBL row** -- still valid
(`assign_splits` falls back to `chembl_id`, and the 1:1 agreement means
grouping is unchanged either way), but their cohort labels are only
recoverable by re-running `prepare-store`.

Confirmed against the DASH paper's own stated dataset composition
(Lehner et al., arXiv:2305.15981): the training set was assembled from
**four sources**, not just ChEMBL --

> "we generated an extended data set by collecting and filtering
> molecules from four different sources: (i) the QMugs data set, (ii) the
> training set from Ref. [22], (iii) lead-like molecule from ChEMBL
> version 30 (filtered as in Ref. [22]), and (iv) organic liquids from
> Refs. [35]-[38]."

> "The data set of 398,935 unique molecules (three conformers each, i.e.,
> 1,029,785 3D structures in total) was split randomly into a 90% subset
> for training..., while the remaining 10% (100,171 3D structures) served
> as validation set."

(That quote was previously recorded here as "393,692 ... 1,076,252",
which is not a sentence the paper contains -- it spliced the Conclusions'
molecule count onto the Data Availability statement's structure count.
Corrected against the PDF; the numbers matter, see the next section.)

Only source (iii) is ChEMBL; sources (i)/(ii)/(iv) have no ChEMBL
identifier, which is exactly what `DASH_IDX = "Rest_N"` most plausibly
means -- the rest of the sources beyond ChEMBL. Sampling 3,000 of the
`DASH_IDX`-only records found ~1,012 unique `DASH_IDX` values, each shared
by ~3 rows -- the same "a few conformers per molecule" pattern
`CHEMBL_ID` rows show, confirming `DASH_IDX` plays the identical
per-molecule grouping role. `_Name` is empty for every sampled
`DASH_IDX`-only record, so it isn't a usable fallback identifier.

### The paper reports two incompatible dataset sizes; this release matches one of them exactly

Checked line by line against the published PDF (JCIM 2023, 63, 6014-6028,
doi 10.1021/acs.jcim.3c00800). The paper states its own dataset size four
different ways, and they do not agree with each other. They fall into two
internally consistent families:

| | molecules | 3D structures | where in the paper |
|---|---:|---:|---|
| **A** (this release) | **348,935** | **1,029,785** | Methods, "The final data set contains..."; Results, "...1,029,785 3D structures in total" |
| **B** | 393,692 | 1,076,252 | Conclusions; Data Availability; Training Procedure (976,081 train + 100,171 val) |

**Family A is arithmetically exact against this store.** 348,935 x 3 =
1,046,805, minus the **17,020** molecules that carry only 2 conformers
instead of 3 (measured directly off `molecules.parquet`: 331,915 molecules
have exactly 3, 17,020 have 2, none have 1 or 4+) = **1,029,785** -- the
record count in this SDF, to the digit. So the release is not a truncated
or partially-failed copy of a larger set: it *is* family A, complete.

**Family B is internally consistent too, just about something else.**
976,081 + 100,171 = 1,076,252 exactly, so the Training Procedure's train
split and the Data Availability statement's total belong together. 393,692
molecules at "up to three conformers" is compatible with 1,076,252
structures (it implies 104,824 missing conformer slots). What family B
is *not* compatible with is family A: 1,076,252 - 1,029,785 = 46,467
structures, ~4.3%, unaccounted for.

**And one Results sentence mixes the two, which is how the inconsistency
shows.** It reads "398,935 unique molecules (three conformers each, i.e.,
1,029,785 3D structures in total)" -- but 398,935 x 3 = 1,196,805, not
1,029,785. 398,935 is 50,000 above family A's 348,935 and appears three
times (twice in the text, once in the Figure 5 caption); it matches
neither family and is most plausibly a typo for 348,935 that propagated.
The same sentence then calls 100,171 "the remaining 10%" of 1,029,785,
but 100,171 is 9.73% of that -- and it is the same 100,171 that pairs
exactly with family B's 976,081. So the validation-set size belongs to
family B while the sentence around it quotes family A's total. (9.73%
rather than 10% is itself unremarkable: the paper splits by *molecule*,
not by structure -- "the three conformers of a molecule were always kept
together" -- and molecules carrying only 2 conformers make the structure
fraction land slightly off 10% either way.)

**Standing conclusion.** Do not treat the 1,076,252 figure as evidence
that this parse lost ~4.3% of the file: nothing here is missing, and the
earlier hypothesis in this doc (silent RDKit sanitization failures,
`mol is None` skipped without a warning) is not needed to explain
anything. The residual open question is narrower and is the paper's, not
this pipeline's: what the 46,467-structure / 44,757-molecule difference
between the two families refers to. Possibly a pre-filtering count that
was never updated after the final filtering step described in Methods.

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
`experiments/tests/test_prepare_dash.py`).

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
