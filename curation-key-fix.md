# Sieve — The 0.4 e Curation Criterion Groups on the Wrong Key

**Status:** handoff, actionable; nothing here is implemented
**Date:** 2026-09-21
**Scope:** why `curate_conformers` groups conformers on an identifier that is not a
structure key, what that costs, why the obvious fix is not a one-line change, and what to
measure during the rebuild that cannot be measured now.
**Relationship to other documents:** `store-build-fixes.md` is the companion, covering
the stereo-perception defects in the same file. Read it first — its §1 fix *increases*
the population affected here, so the two should land together. `main.tex`
§\ref{sec:curation of anomalous conformers} states the criterion as the paper describes
it, and §5 below says which sentence becomes false.
**Provenance:** measured on `fix-dash-stereo-perception` @ `1603aee` against the
2026-09-20 store. The script is `store-build-fixes-scripts/curation_risk.py`.

---

## 1. What is wrong

`curate_conformers` (`experiments/experiments/prepare_dash.py`) reconstructs the DASH
authors' criterion: compare every *pair* of a molecule's conformers atom by atom, call a
pair agreeing when no atom's charge differs by more than $0.4~\mathrm{e}$, and remove a
conformer exactly when it agrees with none of its siblings. Its docstring states the
grouping plainly:

> Grouping is by ``dash_id``, the only identity present on every record: 49.6% of the
> corpus has no ``CHEMBL_ID``, and grouping on that would leave half the molecules
> uncorroborated and delete them.

The reason for not using `chembl_id` is right. The problem is that `dash_id` is not a
structure key either. An identifier records a deposit, and the depositors filed
chemically distinct species under single identifiers — predominantly diastereomers,
with E/Z isomers and tautomer normalisation behind them.

So for those identifiers the criterion is not comparing *the same atom in the same
molecule across conformers*. It is comparing an atom across **different molecules**.

**The stereo fix makes it bigger.** Identifiers covering more than one structure go from
7,452 (2.14%) before the 2026-09-20 rebuild to **12,470 (3.57%) after**, holding 37,121
conformers. Perceiving bond stereo resolves records into more distinct structures, so
every fix in `store-build-fixes.md` widens this population further.

## 2. What it actually costs — measured, and smaller than it looks

Run `store-build-fixes-scripts/curation_risk.py <parquet>`. On the current store, over
400 randomly sampled mixed identifiers:

| pairs compared | agree within 0.4 e | disagree |
|---|---:|---:|
| same-structure | 373 | **0** |
| cross-structure | 817 | **0** |

**Diastereomers under one identifier agree to well within the threshold.** So the
criterion is not deleting genuine chemistry wholesale, and the alarming reading — that
diastereomers look "anomalous" to each other and get discarded — is *not* what the
surviving data shows. Say so plainly rather than implying a larger defect than exists.

What the measurement does establish is the other direction:

> **14,815 structures survive with no same-structure sibling inside their identifier**,
> spread over all 12,470 mixed identifiers.

Each of those passed curation only because a *different molecule* certified it. The
MBIS failure mode is a wild outlier — the documented case puts $+2.975~\mathrm{e}$ on a
carbon its siblings place near $-0.13~\mathrm{e}$ — and such a record disagrees with
everything, so cross-structure corroboration would still catch it. The practical risk of
a missed failure is therefore low. The *stated guarantee*, however, is false; see §5.

## 3. The fix is not a one-line key swap

The obvious change is to group on `collapse_key`, which is computed at parse time and is
a structure key by construction. Do that and the criterion compares like with like.

**But the existing lone-conformer rule then deletes 14,815 structures.** The docstring
says:

> A molecule with a single conformer in the *input* has no pair to corroborate it and is
> therefore removed -- which never occurs in the real corpus (minimum 2 conformers per
> molecule), but is pinned by test rather than left to chance.

That premise holds for `dash_id` groups and fails for `collapse_key` groups: the 14,815
structures of §2 have exactly one conformer in their own group. Swapping the key without
touching the rule would discard every one of them.

So the change has two parts:

1. **Group on `collapse_key`** instead of `dash_id`.
2. **Keep a structure that has only one conformer**, rather than removing it. Its lack of
   a sibling is a property of how the corpus was deposited, not evidence that the record
   failed. Removing it would discard sound data for a reason that has nothing to do with
   the MBIS convergence failure the criterion exists to catch.

Part 2 weakens the "survivors come in pairs by construction" invariant to "survivors come
in pairs **whenever the deposit provided a pair**", which is the honest form and is what
the paper should say.

An alternative worth considering, if you would rather not weaken the invariant: keep
`dash_id` grouping but require agreeing pairs to share a `collapse_key`, and treat a
structure with no same-structure sibling as *unjudged* rather than as failed —
functionally the same outcome as part 2, but it keeps the pairing logic intact and makes
the exemption explicit in the code rather than implicit in a rule change.

## 4. What cannot be measured now, and must be measured during the rebuild

Everything in §2 is measured on the **curated** store, so it describes records that
survived. The complementary question — *were any records wrongly deleted?* — needs the
uncurated parse, which exists only transiently while `prepare_store` runs.

When you rebuild, after `parse_dash_molecules` and **before** `curate_conformers`, save
the uncurated parquet and answer:

1. Of the 2,247 conformers the criterion removes, how many belong to an identifier
   covering more than one structure?
2. Of those, how many disagree with their siblings only across a structure boundary, and
   agree with every conformer of their own structure? Those are wrongful deletions.
3. Of the 86 identifiers removed entirely, how many are mixed identifiers in which each
   structure had a single conformer? Those may be two sound molecules discarded for
   failing to corroborate each other.

**Measured on the 2026-09-21 rebuild** (`curation_deletions.py`, after the corrections
in `curation-audit-correction.md`). The replay reproduces the recorded 2,247 / 86
exactly, which is the check the rest rests on.

| | conformers |
|---|---:|
| removed by the old `dash_id` rule (replayed) | 2,247 |
| removed by the new structure-keyed rule (observed) | 2,230 |
| rescued — old deleted it, new keeps it | 50 |
| newly deleted — old kept it, new deletes it | 33 |

The net −17 in `curation_summary.txt` is the difference of two much larger movements and
should not be quoted alone.

1. Removed conformers inside a mixed identifier: **72 of 2,247**.
2. **Wrongly deleted — rescued *and* holding an agreeing same-structure conformer: 21.**
   For all 21 every agreeing partner lies in a different deposit, which is what makes the
   deletion attributable to the grouping rather than to the threshold. The other 29
   rescued records have no agreeing same-structure partner and are kept as unjudged solo
   structures, which is §3 part 2's deliberate policy rather than a rescue.
3. Identifiers dropped entirely that are mixed with every structure holding one
   conformer: **1 of 86 — `Rest_118752`**, two E/Z isomers of an amidine C=N bond whose
   charges differ by 1.752 e. The old rule read that as mutual disagreement and deleted
   both; they are different molecules, so the difference was chemistry, not an MBIS
   failure. Both survive in the rebuilt store.

> **Open discrepancy.** `curation-audit-correction.md` §2 gives **20** for item 2; this
> run measures **21**. Ruled out as explanations: a partner that does not itself survive
> curation (all 21 have surviving partners), and a threshold boundary (the tightest
> margin is 0.3982 e, none within 0.002 of 0.4). The 21 are listed in
> `experiments/results/curation-deletions.log`. Reconcile before either figure reaches
> the manuscript.

The prediction above — "small, possibly zero" — was right about the magnitude and wrong
about where to look. The 817/817 agreement rate says diastereomers under one identifier
do not appear anomalous to each other, so the damage was never there; it was in the other
half of the same mismatch, a structure split across deposits whose corroborating partner
the grouping could not reach.

## 5. What changes in the manuscript

`main.tex` §\ref{sec:curation of anomalous conformers} currently states:

> A further consequence of the rule is that survivors come in pairs by construction, so a
> molecule ends with zero, two, or three conformers and never with a lone, uncorroborated
> one.

Under `dash_id` grouping that sentence is false for 14,815 structures, each of which ends
with exactly one conformer and is corroborated only by a different molecule. It is the
one claim in the section that the data contradicts outright.

The same section describes the criterion as comparing "the same atom in the three
conformers" of a molecule, which is true only for identifiers holding a single structure.

Both are repaired by the fix rather than by rewording: with `collapse_key` grouping and
part 2 above, the criterion does compare the same atom in the same molecule.

The invariant, however, does not simply become true once conditioned — it has to be
restated, because structure grouping also **widens** what a group is. It pools every
deposit of a structure, and pools enantiomers too, since the key takes the smaller of a
molecule's canonical SMILES and its mirror's. Measured on the store: **17,474 groups
hold more than three conformers, up to 19 across seven identifiers, and every one of
those 17,474 spans more than one deposit.** So "a molecule ends with zero, two, or three
conformers" is false in the other direction as well, and the replacement is: a structure
ends with zero conformers, with the single one it was deposited with, or with at least
two.

**This widening is deliberate and the manuscript should say so**, in one or two sentences
in §\ref{sec:curation of anomalous conformers}. The DASH authors' phrase — "the
difference between the partial charge of the same atom in the three conformers" — plainly
means one deposit's own three, so pooling across deposits is a departure from the letter
of the description, and the paper presents the criterion as a faithful reconstruction.
It is nonetheless the right reading, and the argument is short: what the rule requires is
that the records compared be the same molecule computed the same way. The corpus is
homogeneous in level of theory, as §\ref{sec:the dash charges corpus} already
establishes, so two deposits of one structure are as comparable as two conformers of one
deposit; and MBIS charges are invariant under reflection, so an enantiomer's conformers
are comparable as well. Pooling strengthens the criterion rather than weakening it, since
a failed record then has more siblings to disagree with. What it gives up — the guarantee
that a corroborating sibling came from the same deposit — was never the property that
mattered, because a deposit is not a structure, which is the whole reason for the change.

### The widening requires aligning atoms before comparing

Found while implementing §3, and not obvious from the plan. The rule compares
charges **index-wise**, which was safe only because one deposit's conformers share
an atom ordering by construction. Pooling deposits removes that guarantee, and the
existing `a.shape != b.shape` guard cannot see the difference: of 400 sampled groups
spanning more than one deposit, **15 (3.75%, ~670 of 17,890) hold the same atoms in
a different order**.

Compared by raw index those pairs pit one atom against another, which fails in both
directions — a spurious disagreement that deletes a sound record, or a spurious
agreement that certifies a failed one. Neither is visible in the output; both look
like ordinary charge comparisons.

The fix is already in the codebase: `collapse._canonical_order(mol, key)`, the same
alignment `collapse_molecule_set` uses to average charges over exactly these groups.
Applying it resolves all 15 sampled cases, and makes curation and the fit mean the
same thing by "the same atom". Charges are stored in the key's canonical order at
read time, so the comparison itself is unchanged.

Two further notes for the rewrite. The curation subsection should say which key the
grouping uses and why, since `dash_id`-versus-structure is now a recurring distinction in
the paper. And whatever §4 measures belongs in the Supporting Information beside the
existing pairwise-versus-all-pairs analysis, as the evidence that the reconstruction is
faithful.

## 6. Order of work

1. Land `store-build-fixes.md` §1 and §2 first — they change which records are distinct
   structures, and therefore what `collapse_key` grouping means here.
2. Make the two-part change of §3, with a test pinning that a single-conformer structure
   inside a mixed identifier survives, and one pinning that a structure deposited twice
   forms a single group.
3. Rebuild, saving the uncurated parquet, and run §4's three counts.
4. Update `main.tex` per §5, and re-run `curation_risk.py` on the new store to confirm
   the cross-structure comparison no longer happens at all.
