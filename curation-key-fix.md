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

Given §2's result — cross-structure pairs agreeing 100% of the time — I expect these
counts to be small, possibly zero. Measure them anyway: they are the difference between
"the criterion is sound and we checked" and "the criterion is sound as far as we know".

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
part 2 above, the criterion does compare the same atom in the same molecule, and the
invariant becomes true as soon as it is stated conditionally.

Two further notes for the rewrite. The curation subsection should say which key the
grouping uses and why, since `dash_id`-versus-structure is now a recurring distinction in
the paper. And whatever §4 measures belongs in the Supporting Information beside the
existing pairwise-versus-all-pairs analysis, as the evidence that the reconstruction is
faithful.

## 6. Order of work

1. Land `store-build-fixes.md` §1 and §2 first — they change which records are distinct
   structures, and therefore what `collapse_key` grouping means here.
2. Make the two-part change of §3, with a test pinning that a single-conformer structure
   inside a mixed identifier survives.
3. Rebuild, saving the uncurated parquet, and run §4's three counts.
4. Update `main.tex` per §5, and re-run `curation_risk.py` on the new store to confirm
   the cross-structure comparison no longer happens at all.
