# Sieve — Engineering Practice and Plans

Everything below concerns how Sieve is **built, checked, packaged, and handed to someone else** —
the machinery around the method rather than the method itself.

What Sieve computes is `design.md`, which is authoritative on representation, fit, merge, and query.
Where it sits among prior work is `literature.md`. Evaluation protocol, baselines, splitting
strategy, and the novelty argument are unsettled and live in `drafts/wllr.md`. None of those
questions are reopened here.

Each section states what is **true today**, what the **intent** is, and what is **open**. An intent
is a decision with a trigger, not a dated roadmap: it says what will be done and what event forces
it. Where today's state is a gap rather than a choice, the section says so.

| Topic | Where it stands | § |
|---|---|---|
| Typing | One checker (`ty`), pinned exactly, one live override | §1 |
| Testing | Two tiers; the benchmark tier runs nowhere but a maintainer's machine | §2 |
| CI | Lint and test on 3.11/3.13; does not enforce the reference registry | §3 |
| Documentation | Specs are strong, API docs absent, docstrings unenforced | §4 |
| Dependencies | Core is numpy + scipy; `cosmolayer` is a soft dependency in no extra | §5 |
| Distribution | Never published: `0.1.0`, no tags; MIT licensed but not declared in metadata | §6 |
| Workflow | The pre-commit hook is opt-in and fails open | §7 |

---

## 1. Typing

**Today.** `ty==0.0.72`, pinned exactly in the `dev` extra, is the only type checker. It runs over
`src` and `tests` in CI's lint job and again in the pre-commit hook. There is one live suppression:
`[[tool.ty.overrides]]` on `src/sieve/model.py` silences `invalid-argument-type`.
`tool.ty.analysis.allowed-unresolved-imports` lists `cosmolayer` and `cosmolayer.**`, because that
package cannot be installed from this repository (§5). No annotation-coverage rule is enforced —
ruff's `ANN` rules are unselected — so coverage is whatever happened to be written.

**Intent.** Keep exactly one checker. Adding mypy alongside `ty` would double the suppression
surface and the version-pinning burden for no added signal on a codebase this size. Treat the
override table as debt with a name: drive the `model.py` entry to zero rather than growing the
table, and require every new entry to carry a comment saying what it hides and why it is not fixable
at the call site. An override with no such comment is a bug report, not a configuration.

**Open.** `ty` is pre-1.0. The exact pin is correct for that reason and should not be loosened for
convenience — a floor would let a patch release introduce diagnostics that fail CI on an unrelated
commit. Relaxing to a compatible-release specifier is a decision for `ty` 1.0, not before. Separately:
whether to enforce annotation coverage on the public surface only — the eight names in
`sieve.__all__` — or leave it to review.

---

## 2. Testing and benchmarks

**Today.** Fifteen test modules under `tests/`, with `testpaths = ["tests"]`. The property tests in
`test_properties.py` are handwritten constructions — disjoint cycles, relabeled isomorphs — not
`hypothesis`; there is no property-based testing dependency. Only `test_rdkit_adapter.py` guards an
import, via `pytest.importorskip("rdkit")`.

`test_benchmark.py` is different in kind. It is skipped in full unless `cosmolayer` **and** `rdkit`
both import **and** `stores/cosmo_sample_10k_split` exists — and `stores/` is gitignored. It pins
reference numbers for the corpus (227,723 atoms; R² in (0.913, 0.923) for atomic area and
(0.929, 0.939) for charge; mean matched level 2.88 ± 0.35) alongside two cost regressions: a merge
must cost under half a full refit, and `fold` must beat a sequential reduce at 64 shards.

The consequence, stated plainly: **none of the benchmark tier runs in CI**, and nothing in CI
exercises the method's accuracy. See §3.

**Intent.** Keep the two tiers distinct and name them.

*Tier one* — every module except `test_benchmark.py` — is hermetic, fast, and must pass everywhere.
This is the gate. It already carries the load-bearing correctness properties that need no corpus,
including the 1-WL negative control (`test_two_wl_indistinguishable_graphs_do_collide`), which pins a
known expressiveness bound as intended behavior so that a future contributor does not "fix" it.

*Tier two* is the corpus benchmark: a regression detector for whoever has the store on disk. Its
statistical pins — the R² intervals, the class counts, the matched-level mean — are contracts about
the *method*, and a failure there is a real finding. Its two timing assertions are not contracts:
they are wall-clock comparisons that hold comfortably on a developer machine and would flake on
shared hardware. That distinction should survive contact with anyone who tries to automate the tier.

**Open.** Whether a small committed fixture corpus could put a cheap version of the statistical pins
into tier one, so that the method has *some* automated accuracy regression coverage, and what that
costs in repository size. If tier two is ever automated, the statistical pins move and the timing
checks either stay behind or are replaced by operation counts rather than seconds. Also open: whether
`hypothesis` earns its dependency for the merge monoid's associativity and identity laws, which are
exactly the shape property-based testing is good at.

---

## 3. Continuous integration

**Today.** `.github/workflows/ci.yml` runs on pushes to `main` and on all pull requests, with two
jobs. `lint` runs on 3.11: `ruff check`, `ruff format --check`, then `ty check`, all over `src` and
`tests`. `test` runs `pytest -v` on a 3.11/3.13 matrix. Both install `pip install -e ".[dev,chem]"`.
There is no dependency caching and no coverage measurement.

Two gaps follow from that installation line. First, `cosmolayer` is absent and `stores/` is not in
the repository, so `test_benchmark.py` skips in every job (§2). Second, **CI never runs the reference
registry check** that the pre-commit hook enforces — so a `--no-verify` commit, or a contributor who
never pointed `core.hooksPath` at `.githooks`, can land a broken bibliography with CI fully green.

**Intent.** CI is the authority, and the hook is a convenience that catches things earlier. Anything
the project actually requires belongs in CI, whether or not a hook also checks it. The reference
registry is required — `references/generate_bibtex.sh --check` is offline once the DOI cache is
warm, and belongs in the `lint` job, gated on the same paths the hook uses. The 3.11/3.13 matrix
covers the ends of the supported range and does not need filling in; `requires-python = ">=3.11"` is
what it is testing.

**Open.** Whether to cache pip downloads — RDKit dominates install time, and the trade is cache
maintenance against a slower but entirely predictable run. Whether coverage is worth measuring at
all, given that the risk in this codebase is a wrong array contract rather than an unvisited line;
if it is added, it should be reported and not gated on a threshold.

---

## 4. Documentation

**Today.** Three documents carry the project: `README.md` (what Sieve is, install, a worked SMILES
example, the merge monoid, and an explicit *Not implemented in v1* section), `design.md` (the
authoritative specification), and `literature.md` (positioning). `references/` holds a generated
bibliography, and the reference registry is the one documentation artifact under automated
validation.

Below that, coverage thins. Every module has a module docstring, but 33 of 51 public functions and
classes have one — `sklearn.py` (2 of 8) and `io/rdkit_adapter.py` (1 of 3) are the weakest. No
docstring rule is enforced: ruff's `D` rules are unselected. There is no API documentation build and
no rendered docs site; a user's only route to the callable surface is reading `src/`.

**Intent.** The specification documents are the project's strongest asset and stay hand-written and
authoritative — they are not a docs-site build target and should never be reduced to one. What is
missing is the layer beneath them: the eight names in `sieve.__all__` are the supported surface, and
every one of them should carry a docstring stating what it does, what it assumes about its inputs,
and which `design.md` section governs it. That is worth enforcing mechanically once it is true,
by selecting ruff's `D` rules for the public surface rather than by review.

**Open.** Whether a rendered API site is warranted at all before there is an external user (§6). If
one is ever built, whether it renders `design.md` alongside the API or leaves the specification as a
repository document — the two have different audiences and the merge is not obviously an improvement.

---

## 5. Dependencies and extras

**Today.** The core depends on `numpy>=1.24` and `scipy>=1.10` only. Three extras: `chem`
(`rdkit>=2023.3`), `sklearn` (`scikit-learn>=1.3`), and `dev` (`pytest`, `ruff>=0.16`, `ty==0.0.72`,
and `scikit-learn`, so the wrapper's tests run without a second extra).

`cosmolayer` is an unlisted dependency behind no extra at all. `src/sieve/io/cosmolayer_adapter.py` imports it,
`test_benchmark.py` requires it, and `ty` is explicitly told not to resolve it — but it appears in no
extra and cannot be installed from this repository. It is a private soft dependency, and today that
fact is recorded only as a type-checker exemption.

**Intent.** The core stays numpy + scipy. Everything that reads a domain format is an adapter behind
an extra, and an adapter's absence must degrade to an import error at the adapter, never to a
changed result in the core. `cosmolayer` should be named as what it is — a soft, externally-sourced
dependency — in this document and in the adapter's own docstring, so that a reader who cannot install
it understands they are looking at an optional path rather than a broken one.

**Open.** Whether `cosmolayer` ever becomes installable (a public release, a git URL, or a documented
private index) and therefore an extra like the others. Until then, whether `test_benchmark.py`'s
skip-unless-present behavior is the right default or should be a loud, explicitly-requested opt-in,
so that a maintainer notices when it silently stops running.

---

## 6. Distribution and versioning

**Today.** Sieve has never been distributed. `version = "0.1.0"` is set statically in
`pyproject.toml`, and there are **no git tags**. The build backend is setuptools with
`packages.find` over `src`. The README documents only editable installs; its `pip install -e ".[chem]"`
line installs RDKit without the `dev` tooling, which is correct for a user and wrong for a
contributor, and the two cases are not distinguished.

The project is **MIT licensed**: `LICENSE.md` is present and tracked. That license is not yet
declared in package metadata, though — `pyproject.toml` carries no `license` field and no
classifiers — so a built distribution would ship the file without stating the license in the
metadata that tooling reads.

**Intent.** Declare the license in `pyproject.toml`, not only on disk. Two routes, and the choice is
a version-floor decision: the SPDX form (`license = "MIT"`, PEP 639) is the modern one but needs the
build requirement raised from `setuptools>=68` to `>=77`, while the
`License :: OSI Approved :: MIT License` classifier works against the current floor. Prefer the SPDX
form and raise the floor, since nothing here depends on old setuptools.

Publication is no longer blocked, but the trigger stays the same: the first external user, not a
version number. The moment someone outside this repository is pointed at Sieve it needs a claimed
name, a tag, and a changelog. Versioning stays `0.x` and explicitly makes no stability promise while
`design.md` §13 still has open questions that could change stored formats; the serialization
`format_version` (design.md §9) is the compatibility signal that actually matters to a stored model,
and it moves independently of the package version.

**Open.** Whether to claim the `sieve` name on PyPI — it is a common word and may well be taken,
which would force a distribution name distinct from the import name. Whether releases are
tag-triggered through a workflow or cut by hand at this scale. Whether the version stays static in
`pyproject.toml` or is derived from tags. And whether `LICENSE.md` should simply be `LICENSE`:
setuptools' default `license-files` glob matches both, so this is a convention question rather than a
packaging one.

---

## 7. Contributor workflow

**Today.** `.githooks/pre-commit` runs `ruff check`, `ruff format --check`, and `ty check` whenever a
staged file is Python, then validates the reference registry when a document or anything under
`references/` is staged. It is careful where it counts: it validates the *index* rather than the
working tree, via `git checkout-index` into a temporary directory, which is the correct way to avoid
passing a broken staged file because the working copy happens to be clean.

It is also **opt-in and fails open**. It runs only after `git config core.hooksPath .githooks`, which
is documented in a comment inside the hook itself — where someone who has not installed it will not
read it. And each tool is guarded by a `command -v` check, so a clone without `ruff` or `ty` on the
path silently skips those checks and reports success. "It passed locally" is therefore not evidence
that anything ran.

**Intent.** The hook stays advisory and CI stays authoritative (§3); the failure-open behavior is
acceptable *only* under that split, and stops being acceptable the moment a check exists in the hook
but not in CI — which is exactly today's reference-registry gap. Hook installation belongs in a
contributor-facing setup note that a new clone will actually encounter, not solely in a comment
inside the file being installed.

**Open.** Whether to adopt `pre-commit` proper — it solves installation and tool provisioning, at the
cost of a dependency and a second place where lint versions are declared and can drift from
`pyproject.toml`. Whether the hook should announce a skipped check loudly rather than silently, which
is a small change and removes the misleading-success case regardless of what else is decided. Whether
`CONTRIBUTING.md` is warranted yet, or whether this section is that document for now.
