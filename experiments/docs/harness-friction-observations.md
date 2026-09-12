# How the experiments are conducted: friction worth designing away

Observations from one long session (2026-09-10/12) that ran both charge
series end to end — fold sweeps, full-corpus sweeps, normalization stages —
and then spent most of its time on analysis the harness had no place to put.
Recorded before designing anything, so the discussion starts from what
actually happened rather than from remembered impressions.

Each item states what it is, what evidence came up, and why it matters.
**No solutions are proposed here.** Several of these interact, and a fix for
one may be the wrong shape once another is settled.

## 1. A run records its code but not its data

`manifest.json` carries `git` (commit, branch, dirty, describe), `packages`,
`python`, `seed`, and the full resolved `config`. What it records about the
data is only this:

```json
"data": {"store": "dash-molecules", "split_column": "split",
         "n_train_conformers": 823722, "n_test_conformers": 103033, ...}
```

A store *name* and some row counts. Nothing identifies which version of that
store produced the numbers — no content hash, no store build id, no record of
which SDF or which parse produced it.

This was harmless while the store was immutable. Conformer curation
(`prepare_dash.curate_conformers`, commit `d7ff3f1`) makes the store a thing
with versions: the same name now denotes 1,029,785 conformers before and
1,027,538 after. Every run currently on disk was computed against the former
and says nothing about it.

The refusal added in `prepare_store` — raising when a store was split before
curation existed — is a symptom of the same gap. The pipeline can only detect
the problem through a marker file's absence, because there is no fingerprint
to compare.

## 2. "Has this been done?" is inferred from the filesystem, in two places

Idempotency is a directory-existence check, implemented twice with different
mechanisms:

- the workflow scripts glob in shell —
  `compgen -G "experiments/runs/$EXPERIMENT/${batch}__*/metrics.json"`;
- `dash_depth_sweep` re-checks the same condition in Python
  (`sweep_done`/`fold_done`, matching `d{depth}-{label}__*/metrics.json`).

Both work, and the stub-dispatch tests confirm a second pass runs nothing.
But completion is *inferred* from artifacts rather than recorded, and the two
implementations can drift. It also makes "done" a property of a glob pattern:
the run-directory naming convention
(`{batch_id}__{predictor}-{store}-s{seed}__{stamp}__{uuid}`) is load-bearing
for correctness, not just for readability.

A related consequence: interrupting a sweep leaves a half-written run
directory that later counts as done if `metrics.json` happens to exist. This
session hit the adjacent case twice — a killed DASH sweep left a completed
`d1-full` directory that would have become a duplicate on restart, and it had
to be removed by hand.

## 3. Analysis is not part of the project

`experiments/results` is gitignored (`.gitignore:39`). Every analysis this
session produced lives outside the repo: **26 Python scripts** in a session
scratchpad under `/tmp`, producing figures that were copied into the ignored
`experiments/results/` tree.

What that covers is not incidental work — it is most of the session's
substantive output:

- the per-element and per-carbon-stratum error-distribution figures;
- the conformational charge-range and cross-scheme difference figures
  (whole corpus, 43.2M atoms);
- four anomaly detectors with measured ROC/AUC against a common label;
- the calibration and reconciliation experiments behind design.md §13's
  answers;
- the corpus passes establishing the curation's prevalence numbers.

Their conclusions reached the repo — in commit messages, `design.md`, and a
spec — but **nothing regenerates them**. A figure in the manuscript currently
has no reproducible provenance inside this project, and a reviewer's "how was
this computed?" has no answer that points at tracked code.

## 4. Figures carry no provenance

Related but distinct from (3): the figures themselves record nothing about
what produced them. No store version, no run ids, no git commit, no date.
`val_error_by_element.png` and `charge_delta_by_element.png` are 14.7M- and
43.2M-atom aggregates whose inputs are identified only by the prose in the
message that delivered them.

## 5. Numbers reach documents by hand

Every measured figure now in `design.md` §13, in the curation spec, and in
commit messages was read off a terminal and retyped. That includes the
numbers those documents rest on — `0.015346`, `AUC 0.9986`, `kurtosis 476 →
60`, `2,247 of 1,029,785`.

Two hand-transcription errors were caught during this session by re-deriving
them (a prevalence stated as 35% lower when it was 43%, and a threshold
described as "half a tenth of an electron" when 0.3 e is three tenths). Both
were caught by chance rather than by a check.

## 6. Folds are materialized copies that must stay in sync

`partition-store` writes ten physical stores (10 × 97 MB) derived from the
base store. They are cheap to rebuild and idempotent, but they are a second
copy of the data that is correct only relative to a particular base store —
and nothing ties them to it. Curating the base store silently invalidates
them, and the only signal is that a human remembers.

The same applies to everything keyed to a fold: the 90 fold-sweep runs, the
10 normalization runs, and ~1.8 GB of depth-6 `tree_stats.npz` shards.

## 7. The unit of execution is a process, and nothing is cached between them

Each run is a fresh process that re-reads and re-featurizes its store. On the
full-corpus Sieve sweep this was 11 processes over the same 824k conformers:

| stage | per run |
|---|---|
| `time/data_s` | ~40 s |
| `time/featurize_s` | 195–213 s |
| `time/fit_s` | 136–216 s |

So roughly **35–40 minutes of identical featurization** was repeated across
the sweep, against ~30 minutes of work that actually differed. The DASH
sweep avoids the analogous waste only because `dash_depth_sweep` exists as a
bespoke module that shares one fit and one walk across depths — a
special-case solution to a general problem.

## 8. Configuration is duplicated between YAML and shell

A workflow stage's identity lives in two places: a YAML config file
(`experiments/configs/sieve-charge-sweep.yaml`) and a pile of `--set`
overrides in the shell script. The split is principled — lists must be in
YAML because `--set` parses only scalars, a trap that silently turned
`attributes=element` into a 7-tuple of characters — but it means reading a
stage requires reading both, and the shell half is not validated by anything
except running it.

## What triggered writing this down

Conformer curation changes the store, so every number in
`experiments/runs/` is about to be recomputed. That makes this the cheapest
moment to change how the runs are produced, because the sunk cost of the
existing outputs is about to be spent anyway.
