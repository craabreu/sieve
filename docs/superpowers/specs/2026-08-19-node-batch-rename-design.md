# Design: Generalize `AtomBatch` naming, clarify terse config fields

**Status:** approved, ready for implementation planning
**Date:** 2026-08-19
**Scope:** a naming pass over the public and internal API — no behavior change.

## Background

The codebase currently mixes three vocabularies for the same concept:

- The batch class is named `AtomBatch` (chemistry-specific).
- Its own fields say `node_attrs`, `n_atoms` (graph-generic: "node").
- `design.md`'s title and math notation (§1) define the method itself in terms of
  **vertices** ("Support-gated Inference over Enriched Vertex Environments," $v$ ranges
  over vertices).

`design.md §11.1` already states the core is "graph-library agnostic" — i.e. meant to work
on graphs in general, not only molecules — which `AtomBatch`'s chemistry-specific name works
against. Separately, `SieveConfig` exposes two fields whose names require reading `design.md`
to understand at all: `n_min` and `alpha`.

`design.md`'s own vocabulary (vertex, in the method's name and math) is out of scope here and
is not being changed — this pass only touches code identifiers and `design.md`'s
implementation-facing pseudocode/prose, not its abstract notation.

## Decision: node/edge, not vertex/edge or node/link

Three vocabularies were considered for the code-level (non-math) naming:

| Option | Verdict |
|---|---|
| vertex/edge | Rejected for code identifiers — reserved for `design.md`'s abstract math notation only, per the user's explicit preference. |
| **node/edge** | **Chosen.** `edge` is already the term used everywhere in the code (`edge_src`, `edge_dst`, `edge_attr`, `edge_codes`, `csr.attr`) and is already correct and standard for this domain (matches NetworkX, PyG, DGL, scipy CSR-graph conventions). Only the atom-specific side needs to change, to match the already-present `node_attrs` field. |
| node/link | Rejected — would require renaming `edge_*` throughout the codebase for no functional gain, and "link" is a weaker fit for this domain (social-network/D3.js convention, not standard in cheminformatics/graph-ML). |

## The rename table

| Old | New | Notes |
|---|---|---|
| `AtomBatch` (class, `src/sieve/batch.py`) | `NodeBatch` | |
| `n_atoms` (property/param, wherever it appears) | `n_nodes` | |
| `atom_counts`, `atom_order`, other `atom_*` batch-side identifiers | `node_*` equivalents | Local variables and parameters, not just class/dataclass fields. |
| `SieveConfig.n_bond` (property) | `n_edge_types` | This is the size of the edge-*type* alphabet (`max(edge_codes.values()) + 1`), not a count of edges — `n_edge`/`n_edges` was considered and rejected as misleading. |
| `SieveConfig.n_min` (field) | `minimum_support` | |
| `SieveConfig.alpha` (field) | `shrinkage_strength` | |

**Deliberately not renamed**, and why:

- `elements`, the rdkit/SMILES adapters (`io/rdkit_adapter.py`, `io/cosmolayer_adapter.py`) —
  these are genuinely atom/molecule-specific concerns, not generic graph concerns. Renaming
  them would blur a real distinction, not fix an uninformative one.
- `edge_codes` — already descriptive (the code→bond-type mapping); `edge` here is correct,
  and "bond" would reintroduce the chemistry-specific vocabulary this pass is removing from
  the *generic* graph type.
- `y` — sklearn/ML convention for targets (`design.md §10.2` plans a scikit-learn wrapper);
  renaming to `target`/`targets` would work against that convention, not toward clarity.
- `msd` (`FrozenLevel.msd`, `SieveModel.global_msd`) — looks like the same class of terse
  name as `n_min`/`alpha`, but it is a *deliberate* choice recorded in `design.md §4.1`:
  it must never be confused with the reported `variance` (`s²`), since the two are different
  quantities (population vs. Bessel-corrected). Renaming it toward anything variance-adjacent
  would undo a real design decision.
- `design.md`'s own vocabulary — "vertex" stays in the method's name and math notation
  (§0–§1); only implementation-facing pseudocode/prose referencing the renamed identifiers
  changes.

## Blast radius

Mechanical, not architectural: ~193 occurrences of the renamed identifiers across 26 files
(`src/sieve/*.py`, `src/sieve/io/*.py`, `src/sieve/sklearn.py`, every file in `tests/`), plus
the corresponding sections of `design.md` (§9.2's JSON example, §10.1's `SieveConfig`
pseudocode, §11.1's `AtomBatch` pseudocode, and prose mentions elsewhere). No new behavior;
the implementation plan is a rename pass with no fallback shim.

## Serialization compatibility

`n_min` and `alpha` are stored keys in the `.npz` config JSON blob
(`SieveModel.save`/`SieveModel.load` in `src/sieve/model.py`). Renaming them changes the blob
shape, so:

- Bump `FORMAT_VERSION` from `1` to `2` in `src/sieve/config.py`.
- No migration path — `SieveModel.load()` already refuses to load an unrecognized
  `format_version` outright (`design.md §9.2`'s stated purpose for the field: "a reader that
  does not recognize it must refuse to load, not guess"). An old `.npz` produced under
  `FORMAT_VERSION = 1` gets a clear `ValueError`, not a silent misparse.
- Confirmed acceptable: no released version exists yet, so there is nothing to preserve
  compatibility with.

## Testing

No new test *behavior* — this is a pure rename. The existing test suite is renamed alongside
the source (identifiers, fixture helpers in `tests/helpers.py`, assertion messages) and serves
as the verification. `tests/test_io.py::test_unknown_format_version_is_refused` already covers
the "unrecognized format version raises" path generically and needs no change beyond whatever
identifier renames touch it incidentally.

Verification for the implementation: full `pytest`, `ruff check`, `ruff format --check`, and
`ty check`, same bar as any change in this repo.
