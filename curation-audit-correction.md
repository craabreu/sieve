# Sieve — The Curation Deletion Audit Reports Zero Where the Answer Is 20

**Status:** handoff, actionable; nothing here is implemented
**Date:** 2026-09-21
**Scope:** `store-build-fixes-scripts/curation_deletions.py` answers `curation-key-fix.md`
§4's second question with `0`, and the true answer is `20`. The cause is in the script,
not the data. This says what the right numbers are, how they were obtained, what to
change, and which claim in the manuscript they support.
**Relationship to other documents:** corrects `curation-key-fix.md` §4, whose expected
answers should be replaced with the table in §2 below. `store-build-fixes.md` is
unaffected.
**Provenance:** measured 2026-09-21 on the rebuild that finished at 16:11 — the
uncurated parse (`molecules.parquet.uncurated`, 1,029,785 rows) against the curated
store written from it (`molecules.parquet`, 1,027,555 rows).

---

## 1. What the audit got right

Keep these. The replay is faithful, which is the check everything else rests on:

> old rule: removed 2,247 conformer(s), 86 identifier(s) entirely

Exactly the recorded figures, so stereo perception moved no charge, no atom count and no
atom order, and nothing unrelated changed in the parse. Question 1 (72 of 2,247 removed
conformers sit in a mixed identifier) and question 3 (1 of 86 dropped identifiers is
mixed with every structure holding one conformer — `Rest_118752`) are both correct.

## 2. What it gets wrong

Question 2 reports **0 wrongful deletions**. The true figure is **20**.

Ground truth comes from diffing the uncurated parse against the curated store the same
run produced, which needs no inference about what either rule "would" do:

| | conformers |
|---|---:|
| removed by the old `dash_id` rule (replayed) | 2,247 |
| removed by the new structure-keyed rule (observed) | 2,230 |
| **rescued** — old deleted it, new keeps it | **50** |
| **newly deleted** — old kept it, new deletes it | **33** |

The net figure of −17 in `curation_summary.txt` is the difference of two much larger
movements and should not be quoted on its own.

**Of the 50 rescued, 20 have a same-structure conformer they agree with, and in all 20
cases that conformer is in a different deposit — none in the same one.** Those are
wrongful deletions in the strict sense: sound records with a genuine corroborating
sibling that `dash_id` grouping could not see, because the sibling was filed under
another identifier. The other 30 have no agreeing same-structure partner at all and are
kept as unjudged solo structures, which is the deliberate policy of
`curation-key-fix.md` §3 part 2 rather than a rescue.

**The 33 newly deleted are the converse, working as designed**: records that the old
rule kept only because a *different molecule* certified them, now judged against their
own structure and failing.

Note the two fixes are entangled in the 20. They are cross-deposit comparisons, so they
became possible only with the grouping change and only give the right answer because of
the atom alignment added in `61b2b82`; without it those pairs would have been compared
out of order and most would not have agreed. The alignment fix is therefore load-bearing
for this result, not a tidy-up.

## 3. Why the script says zero

Three defects, all in question 2's few lines:

```python
by_key.setdefault((df.at[r, "dash_id"], df.at[r, "key"]), []).append(r)
...
own = [s for s in by_key[(df.at[r, "dash_id"], df.at[r, "key"])] if s != r]
if own and all(agrees(r, s) for s in own):
```

1. **`by_key` is keyed on `(dash_id, key)`.** A same-structure conformer in another
   deposit is therefore invisible to it — and that is precisely the population the
   grouping change exists to rescue. All 20 real cases fall in this blind spot, which is
   why the count is not merely low but exactly zero.
2. **`all(...)` is the wrong quantifier.** The criterion keeps a conformer that agrees
   with *at least one* sibling, so the audit has to ask `any`, not `all`. As written it
   demands agreement with every sibling, which is a strictly stronger condition than the
   rule it is auditing.
3. **`own and` excludes records whose own structure has no other conformer**, i.e. the
   solo structures. Those are kept by the new rule and deleted by the old one, so by the
   docstring's own definition — "records the new grouping keeps and the old one threw
   away" — they belong in the answer, or the definition needs narrowing to say so.

## 4. What to change

**Compute the sets by diffing, not by inferring.** The curated store is the new rule's
own output, so `uncurated − curated` *is* the new rule's removal set exactly, with no
second implementation to keep faithful. Replay the old rule only, and take the new side
from the store:

```python
kept = set(zip(cur.dash_id, cur.conf_id))
new_removed = {i for i, (d, c) in enumerate(zip(unc.dash_id, unc.conf_id))
               if (d, c) not in kept}
rescued = old_removed - new_removed          # 50
newly_deleted = new_removed - old_removed    # 33
```

Then classify `rescued` by whether an agreeing same-structure conformer exists
**anywhere in the corpus**, keyed on `collapse_key` alone:

```python
by_key = collections.defaultdict(list)
for i, k in enumerate(keys):
    by_key[k].append(i)

partners = [s for s in by_key[keys[r]] if s != r and agrees(r, s)]
```

`partners` non-empty is the wrongful-deletion condition; whether every partner sits in a
different `dash_id` is what makes it attributable to the old grouping. Report
`newly_deleted` as well — the audit never asked for it, and it is the evidence that the
change tightened the criterion rather than merely loosening it.

Keep the replay and its `2,247 / 86` assertion. It is the only thing standing between
these counts and a silent change elsewhere in the parse.

## 5. What this supports in the manuscript

`curation-key-fix.md` §4 predicted these counts would be "small, possibly zero", on the
strength of cross-structure pairs agreeing 817 times out of 817. That prediction was
right about the magnitude and wrong about the direction of interest: the agreement rate
says diastereomers under one identifier do not look anomalous to each other, and the
damage was never there. It was in the *other* half of the same mismatch — a structure
split across deposits, whose corroborating partner the grouping could not reach.

So the honest sentence for §\ref{sec:curation of anomalous conformers}, or for the
Supporting Information beside the pairwise-versus-all-pairs analysis, is that the
criterion was reconstructed on a deposit identifier, that this deleted 20 sound records
whose corroborating conformer was deposited separately and retained 33 that only a
different molecule vouched for, and that grouping on the structure key corrects both.
Both figures are tiny against 1,029,785 conformers, and saying so is stronger than the
alternative of not having looked.

Do **not** write "no records were wrongly deleted" on the strength of the current
script's output.

## 6. Order of work

1. Fix question 2 as in §4, and re-run against `molecules.parquet.uncurated` — it is
   still on disk beside the curated store, and this is the last work that needs it.
2. Replace `curation-key-fix.md` §4's expected answers with §2's table.
3. Record `Rest_118752` somewhere durable; it is the one identifier where the old rule
   discarded two structures for failing to corroborate each other, and it is worth a
   line in the SI as the concrete example.
