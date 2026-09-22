# Sieve — Store Build: Remaining Stereochemistry Defects

**Status:** handoff, actionable; nothing here is implemented
**Date:** 2026-09-21
**Scope:** what is still wrong in `prepare_dash`'s stereo handling after
`274761d fix(experiments): perceive bond stereo, and stop overwriting declared parities`,
how to fix each item, and how to verify the fix. Also §5, the document numbers that
change once the store is rebuilt, and §6, the scripts that produced them.
**Branch:** measured on `fix-dash-stereo-perception` @ `1603aee`, against
`experiments/stores/dash-molecules/molecules.parquet` as rebuilt 2026-09-20 20:05, with
`molecules.parquet.pre-stereofix.bak` as the before-comparison.
**Companion:** `curation-key-fix.md` covers the 0.4 e conformer criterion, which
groups on `dash_id` rather than on a structure key. The two interact: §1 below
increases the population that defect affects, so they should land together.
**Provenance:** every figure below is measured on the real store, on the date above.
Sample strides are stated per measurement. Nothing is estimated.

---

## 0. What the existing fix got right

Confirm this before changing anything, so a regression is visible.

| claim | measurement | result |
|---|---|---|
| bond stereo is now perceived | test split, groups split only by a bond-stereo declaration while geometrically identical | **0** (was 41 groups, all `STEREOANY` vs `STEREOCIS`/`STEREOTRANS` on C=N and N=N) |
| declared parities survive and are right | 51,377 rows (every 20th), declared tetrahedral tag vs its own coordinates | **0 mismatches** |
| the labelling artifact is essentially closed | test-split groups separated by `collapse_key` but geometrically identical in every conformer | **3 groups / 9 conformers**, was 51 / 309 |

Those 3 residual groups are item 2 below. They are the whole remaining artifact population.

---

## 1. The perception gate is blind to dependent stereocentres

**Biggest of the three. Fix this one first.**

### What is wrong

`_assign_stereo_if_needed` decides whether to consult coordinates with

```python
centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
needed = any(tag == "?" for _, tag in centers)
```

On a molecule carrying **explicit hydrogens** this returns `[]` for *dependent*
(para-) stereocentres — spiro, ring-fusion and bridgehead centres, whose two ring
branches are constitutionally identical so neither is a classical stereocentre while the
pair is stereogenic. The legacy implementation returns `[]` too. The store parses with
`removeHs=False`, so this is the live path, and the failure is silent: an empty list,
not an error.

Reproduce, on three real store records with stereo stripped to simulate a molblock that
declared none:

| dash_id | gate, explicit H | gate, on a `RemoveHs` copy |
|---|---|---|
| `Rest_95518` | `[]` → would not perceive | `[(5,'?'), (8,'?')]` → would perceive |
| `Rest_10716` | `[]` | `[(4,'?'), (7,'?')]` |
| `Rest_23243` | `[]` | `[(20,'?'), (21,'?')]` |

The same suppression hits `Chem.FindPotentialStereo`, verified on four molecules: the
stripped molecule yields the centres with implicit H and nothing after `AddHs`. Small
cases such as 1,4-dimethylcyclohexane survive `AddHs`, so this cannot be checked on toy
inputs — use real records.

### How big it is

Scanned every 50th row (20,551 rows), probing each record's heavy-atom copy for
potential tetrahedral centres and asking whether the stored `Mol` tagged them:

| | |
|---|---:|
| potential tetrahedral centres found | 21,195 |
| left unassigned by the store | **195 (0.92%)** |
| rows holding at least one | **121 (0.59%)** |

Composition of the 195: 122 three-coordinate N (90 of them in three rings, i.e.
bridgehead and configurationally locked), 73 four-coordinate C (63 in three rings).

### Why it matters

This is the *under*-splitting failure, and it is worse in kind than the over-splitting
one §2.0 of `stereochemistry.md` chased. An unassigned centre means `collapse_key`
merges genuine diastereomers, so the fit target becomes a mean over different
chemistry — precisely what `collapse.py`'s module docstring exists to prevent. It also
inflates the keyed floor for the affected groups.

### The fix

Run the gate — and only the gate — on a heavy-atom copy, mapping indices back. The
perception itself still runs on the real molecule.

```python
probe = Chem.Mol(mol)
for atom in probe.GetAtoms():
    atom.SetAtomMapNum(atom.GetIdx() + 1)
probe = Chem.RemoveHs(probe)
centers = Chem.FindMolChiralCenters(probe, includeUnassigned=True,
                                    useLegacyImplementation=False)
needed = any(tag == "?" for _, tag in centers)
```

Verified to recover exactly the centres the store already tags elsewhere in the same
molecules.

### How to verify

1. Re-run the scan above; `left unassigned` should fall to near zero, and any residue
   should be centres that are genuinely undeterminable from the coordinates.
2. Add a regression test using one of `Rest_95518`, `Rest_10716`, `Rest_23243`: strip
   stereo, run `_assign_stereo_if_needed`, assert the dependent centres come back tagged.
   A toy molecule will pass with or without the fix and is not a test.

---

## 2. Non-tetrahedral chiral tags split collapse keys

### What is wrong

`AssignStereochemistryFrom3D` assigns `CHI_TRIGONALBIPYRAMIDAL` and
`CHI_SQUAREPLANAR` to pentavalent phosphorus and to sulfonic-acid sulfur, and the
**permutation index varies between conformers of one molecule**, so one structure
receives several `collapse_key`s.

Observed on the test split:

| dash_id | keys | what differs |
|---|---:|---|
| `Rest_128717` | 3 | `[P@TB14]`, `[P@TB1]`, `[P@TB13]` on one phosphorane |
| `Rest_130841` | 3 | `[P@TB13]`, `[P@TB14]`, `[P@TB16]` |
| `Rest_12517` | 3 | `[S@TB15]`, no tag, `[S@TB17]` on an S(=O)(=O)(OH) |

It is compounded by `mirror_mol` (`experiments/experiments/collapse.py:27`), which
inverts only `CHI_TETRAHEDRAL_CW/CCW`. A molecule carrying a TB or SP tag therefore has
a "mirror" identical to itself in that tag, so `min(MolToSmiles(mol),
MolToSmiles(mirror_mol(mol)))` cannot merge its enantiomers either.

### How big it is

Every 4th row, 256,885 rows scanned: **4 rows (0.0016%)**, 4 distinct molecules — 3
`S deg=3 CHI_SQUAREPLANAR`, 1 `P deg=5 CHI_TRIGONALBIPYRAMIDAL`. Rare, but it is the
**entire remaining artifact population** on the test split after the 2026-09-20 fix.

### The fix

After the 3D pass, clear any chiral tag that is not tetrahedral:

```python
KEEP = {Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
for atom in mol.GetAtoms():
    if atom.GetChiralTag() not in KEEP:
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
```

These centres are not stereogenic in any sense MBIS charges distinguish, and the
alternative — teaching `mirror_mol` the TB/SP permutation algebra — is much more work
for the same outcome. Do it in `_assign_stereo_if_needed` rather than in `collapse.py`,
so the stored `Mol` and the key agree.

### How to verify

`Rest_128717`, `Rest_130841` and `Rest_12517` should each collapse to **one** key across
their conformers.

---

## 3. `STEREOANY` survives on ~3.4% of records — decide, do not just leave it

1,878 bonds across 51,377 sampled rows (every 20th); 1,769 rows, 3.44%.

**Not currently harmful.** The flag is consistent across a molecule's conformers, so it
splits no keys — the test-split measurement finds zero bond-declaration-only artifact
groups after the fix.

**But it is a silent hole in the planned E/Z feature.** `cis-trans-geometry.md` §2.1
reads `STEREOCIS`/`STEREOTRANS` with `GetStereoAtoms()`, so a `STEREOANY` bond yields no
code at all. 3.4% of records have double bonds outside that feature's reach and nothing
in the design documents accounts for it.

This needs a decision, not necessarily a change: either accept it and state the coverage
in `cis-trans-geometry.md` §7, or decide that a `STEREOANY` bond whose geometry *is*
determinable from the conformer should be perceived rather than left flagged.

---

## 4. A convention that must be stated, not assumed

RDKit's tetrahedral parity is the **determinant of the first three neighbour vectors
taken from the central atom, with neighbours in bond order**:

```python
v = positions[[n.GetIdx() for n in atom.GetNeighbors()][:3]] - positions[atom.GetIdx()]
volume = np.dot(np.cross(v[0], v[1]), v[2])      # CCW -> +, CW -> -
```

Calibrated against RDKit's own tags on 390 four-coordinate centres from the store:

| formulation | agreement |
|---|---:|
| determinant from the **central atom**, bond order | **390/390 (100%)** |
| determinant from the **first neighbour**, bond order | 0/390 |
| determinant from the central atom, neighbours sorted by index | 380/390 (97.4%) |

The from-first-neighbour reading is exactly anticorrelated, so it produces plausible
nonsense rather than an error; sorting neighbours by index rather than bond order breaks
2.6% of centres. `tetrahedral-chirality.md` §6's row "the tag matches the signed volume
in that order, 27/27" should say which formula it means, since the proposed feature's
correctness rests on it.

The same convention covers **three-coordinate stereocentres** — sulfoxides,
sulfilimines. RDKit has no lone-pair pseudoatom; it defines the parity with the central
atom as the fourth vertex, which is where the lone pair points. Verified:
`C[S@](=O)c1ccccc1` → `CHI_TETRAHEDRAL_CCW`, volume +4.101; `C[S@@](=O)c1ccccc1` →
`CHI_TETRAHEDRAL_CW`, volume −3.689; `AssignStereochemistryFrom3D` recovers both.
Phosphines are *not* perceived by default (`C[P@](C)c1ccccc1` comes back
`CHI_UNSPECIFIED`), so the trick covers S(IV) but not every three-coordinate centre.

---

## 5. What changes in the documents once the store is rebuilt

The test-split numbers in `stereochemistry.md` were all measured on the pre-fix store.
Rebuilt on both stores with one harness, which reproduces §2 exactly on the pre-fix side
(102,824 conformers, 4,211,148 atoms, 32,633 stereo-blind groups, 1,493 subset groups,
8,027 conformers, 7.81%, keyed 0.009168 / blind 0.009348):

| | pre-fix | bond-stereo fix | rebuilt store |
|---|---:|---:|---:|
| whole split, keyed floor | 0.009168 | 0.009108 | 0.009078 |
| whole split, stereo-blind floor | 0.009348 | 0.009348 | 0.009332 |
| whole split, gap | 0.000180 | 0.000240 | **0.000253** |
| subset groups | 1,493 | 1,998 | 1,976 |
| subset conformers | 8,027 | 9,397 | 9,257 |
| subset share of split | 7.81% | 9.14% | **9.00%** |
| subset keyed floor | 0.009040 | 0.008567 | 0.008440 |
| subset stereo-blind floor | 0.011161 | 0.011082 | 0.011144 |
| subset gap | 0.002121 | 0.002515 | **0.002704** |
| artifact groups / conformers | 51 / 309 | 3 / 9 | **0 / 0** |

The third column is the store rebuilt 2026-09-21, holding 102,855 test conformers and
4,209,344 atoms in 32,667 stereo-blind groups.

**Read the third column as a new measurement, not as a controlled increment.** The
rebuild recomputed the split, and Butina is not bit-exact across runs, so the test set is
a different set of molecules from the one the first two columns describe. The direction
of every movement is consistent across all three, and the artifact count falls to zero,
but a difference of one part in the fourth decimal between columns two and three is not
attributable to the gate and exotic-tag fixes alone.

Three consequences for the prose.

**§2.0's correction has the wrong sign.** It predicted the gap was 7.0% too large. The
fix makes it **19% larger**: the stereo-blind floor barely moves, because stripping
stereo is indifferent to the fix, while the keyed floor falls as 505 new subset groups
appear. §2.0 modelled the defect as groups that should be *merged*; perceiving bond
stereo from 3D mostly *splits*. The sentence "correcting it makes §2's argument stronger
rather than weaker" is right in conclusion and wrong in mechanism.

**The artifact population is gone, now completely.** On a geometric criterion — all
conformers sharing one signature of signed volumes and torsion classes, up to a global
reflection — it is 51 groups / 309 conformers pre-fix, 3 groups / 9 conformers after
the bond-stereo fix, and **0 groups / 0 conformers on the rebuilt store**. §2.0's
"~11% of the stereo-sensitive subset is a labelling artifact" does not survive on any of
the three.

**The two tracks re-weight.** Classifying subset groups by what differs geometrically:

| | pre-fix groups | bond-stereo fix | rebuilt store | share of conformers |
|---|---:|---:|---:|---:|
| E/Z only | 443 | 990 | 967 | **40.3%** |
| tetrahedral only | 857 | 857 | 884 | **51.9%** |
| both | 118 | 124 | 125 | 7.7% |
| neither | 75 | 27 | **0** | **0%** |

The "neither" bucket closing is the result worth keeping: on the rebuilt store every
stereo-sensitive group is explained by an E/Z difference, a tetrahedral one, or both,
with no residue that the two tracks fail to account for.

The tetrahedral bucket is *identical* before and after — the fix touched bond stereo and
stopped overwriting atom parities, so atom geometry did not move — while the E/Z bucket
more than doubles. §3's headline 68.9% tetrahedral / 27.4% E/Z becomes roughly
**50.7% / 40.5%**. The E/Z half, which both twin documents treat as the cheap
afterthought and whose local descriptor reaches only 56% of its bucket, is now worth
nearly as much as the tetrahedral half all of §6 is about.

Note the pre-fix E/Z bucket lands at 443 groups against §3's flag-based 449, which is
the cross-check that the criterion agrees.

**Not rebuilt:** §3's reachability rows (resolves at *k* ≤ 6 / *k* > 6 / automorphism
tie). Those need the WL refinement over stripped graphs and the per-group
differing-centre resolution test, which is a separate job from the floors.

---

---

## 6. The measurement scripts

`store-build-fixes-scripts/`, beside this file. Uncommitted, deliberately — they are the
harness the numbers above came from, kept so you can re-run rather than rebuild them.
Every one is run **from the repository root** (they do `sys.path.insert(0,
"experiments")`) and takes a `molecules.parquet` path, so each can be pointed at the
current store or at `molecules.parquet.pre-stereofix.bak` for a before/after pair.

| script | what it answers | invocation |
|---|---|---|
| `rebuild_stereo_numbers.py` | §5's floors, the stereo-sensitive subset, the geometric artifact count and the artifact-merged gap | `... <parquet> <tag> [row-limit]` — writes `/tmp/stereo-numbers-<tag>.json` |
| `compose.py` | §5's composition table (E/Z only, tetrahedral only, both, neither) | `... <parquet> <tag>` |
| `classify_residue.py` | splits the residual artifact groups into real conflicts and cases the geometric test cannot adjudicate | `... <parquet>` |
| `diagnose_artifacts.py` | what *declaration* separates the keys of a geometrically identical group — bond types, `STEREOANY` conflicts, atom environments | `... <parquet>` |
| `dump_unadjudicated.py` | prints the unadjudicated groups in full: positions, tags, per-member signed volumes, SMILES | `... <parquet>` |
| `check_match.py` | validates the cross-member atom matching itself (0 of 1,493 groups fail) — run this first if a number looks wrong | `... <parquet>` |
| `scan_unassigned.py` | §1's measurement: potential centres the store left unassigned | `... <parquet> <stride>` |
| `scan_exotic.py` | §2's measurement: records carrying a non-tetrahedral chiral tag | `... <parquet> <stride>` |
| `audit_store.py` | §0's checks in one pass: tag census, `STEREOANY` count, declared parity vs coordinates | `... <parquet> <stride>` |
| `lone_pair_probe.py` | §4's evidence that RDKit has no lone-pair pseudoatom and how it scores three-coordinate centres | no arguments |
| `dump_two.py` | the phosphorane groups of §2, printed per key | `... <parquet>` |
| `curation_risk.py` | `curation-key-fix.md`'s measurement: identifiers spanning several structures, structures with no same-structure sibling, and whether cross-structure pairs pass the 0.4 e test | `... <parquet>` |

Three things to know before trusting a re-run.

**The parity convention is load-bearing and easy to get backwards.** `geometry_signature`
in `rebuild_stereo_numbers.py` is the shared routine; §4 gives the calibration. An
inverted convention still compares members consistently, so it produces plausible
results rather than an error.

**`potential_centres` probes the heavy-atom graph on purpose** — see §1. Reverting that
to a probe on the explicit-H molecule silently loses every dependent stereocentre, which
is what made the first pass of these measurements undercount.

**Restricting to potential stereocentres matters.** Scoring every four-coordinate atom
pulls in symmetric centres such as a CH2, whose signed volume depends on which of two
equivalent hydrogens the canonical ranking happened to place first, and manufactures
differences that are not stereochemical.

Runtimes are minutes: the full test split is ~40 s per store for
`rebuild_stereo_numbers.py`, and the corpus scans are a few minutes at stride 20.

## 7. Order of work

1. Item 1, the gate. It is the only one that changes the fit targets.
2. Item 2, the exotic tags. Cheap, and it closes the artifact population.
3. `curation-key-fix.md` §3, the grouping key, before rebuilding.
4. Rebuild the store, **saving the uncurated parquet** (`curation-key-fix.md` §4);
   keep `molecules.parquet.pre-stereofix.bak` and add a
   `.pre-gatefix.bak` alongside it, since every number in §5 is a before/after pair.
5. Re-run §5's measurements and update `stereochemistry.md` §2, §2.0, §3 and §10.
6. Item 3, the `STEREOANY` decision, and item 4, the convention note in
   `tetrahedral-chirality.md` §6.

**Commit the measurement harness this time.** `stereochemistry.md`'s provenance note
records that its scripts were scratch and not committed, which is why every number above
had to be rebuilt from the descriptions rather than re-run. The scans in §1, §2 and §5 are each under 200 lines, ship in
`store-build-fixes-scripts/` (§6) and belong in `experiments/workflows/` or as tests.
