# Sieve

**S**upport-gated **I**nference over **E**nriched **V**ertex **E**nvironments.

Sieve is a node-level regressor for labeled graphs (molecules, in practice)
that generalizes the classical regressogram to a *nested hierarchy* of
partitions: it refines each node's environment through graded attribute
levels and rounds of Weisfeiler–Lehman color refinement, fits mean and
variance per class at every level, and predicts by looking up the deepest
class that both matches a query node and has enough training support,
backing off to shallower — and ultimately global — statistics otherwise. See
`literature.md` §3 for how this sits among target encoders, hierarchical
regressograms, and WL kernels.

## Install

```bash
pip install -e ".[dev]"       # core + pytest
pip install -e ".[chem]"      # + RDKit, for the molecular adapter
```

## A worked example

```python
import numpy as np
from rdkit import Chem
import sieve
from sieve.config import SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_smiles

smiles = ["CCO", "CCC", "c1ccccc1", "CC(=O)O"]
y = np.array([[1.2], [0.9], [2.3], [1.7]])   # one label per atom's molecule here

mols = [Chem.MolFromSmiles(s) for s in smiles]
codes, edges = build_codes(mols, ["element", "degree", "aromatic"])
cfg = SieveConfig(
    target_dim=1,
    attribute_levels=(("element", "degree", "aromatic"),),
    attribute_codes=codes,
    edge_codes=edges,
    max_wl_depth=2,
    minimum_support=1,
)
batch = from_smiles(
    smiles, y=np.repeat(y, [m.GetNumAtoms() for m in mols], axis=0), config=cfg
)
model = sieve.fit(batch, cfg)
pred = sieve.predict_detailed(model, batch)
print(pred.value, pred.matched_level)
```

## The merge monoid

A fitted model is immutable; combining shards, streaming in chunks, and
parallel fitting are all the same operation:

```python
shards = [sieve.fit(b, cfg) for b in batches]
model = sum(shards, sieve.SieveModel.empty(cfg))   # or a.merge(b) pairwise
```

`config.chunk_size` uses exactly this path internally — chunking is a memory
decision, not a separate statistical code path.

## Inference-time options

Everything below is read at **prediction** time. None of it enters
`schema_version`, all of it goes through `SieveModel.with_params(...)`, and
none of it invalidates fitted statistics — so a whole comparison costs one
fit, not one fit per arm:

```python
model = sieve.fit(batch, cfg)
a = sieve.predict(model, q)
b = sieve.predict(model.with_params(class_estimator="continuation"), q)  # no refit
```

**`class_estimator`** — how a class's estimate is formed from what is stored
beneath it. There is no `None` here; `"pooled"` *is* the do-nothing option.

| value | estimate |
|---|---|
| `"pooled"` *(default)* | the class's own atom-weighted mean — the original rule |
| `"continuation"` | unweighted mean of the class's children's stored means, so each distinct child environment counts once (design.md §4.4) |
| `"continuation_recursive"` | as above but over the children's own continuation estimates; measured indistinguishable from `"continuation"`, kept for reproducibility |

**`shrinkage_weight`** — how a class blends toward its shrunk parent.
**`None` is the "no shrinkage" option**, and naming a rule is how shrinkage is
requested.

| value | weight on the class's own estimate |
|---|---|
| `None` *(default)* | **no shrinkage** |
| `"count"` | `N / (N + shrinkage_strength)` — needs a `shrinkage_strength` |
| `"diversity"` | Kneser–Ney's own λ, `min(α·C/N, 1)` on the parent — needs a `shrinkage_strength` |
| `"empirical_bayes"` | `C / (C + α)` with α estimated per level — **refuses** a `shrinkage_strength` |

**`shrinkage_strength`** (`float | None`, default `None`) — α for `"count"` and
`"diversity"`. A bare `shrinkage_strength` with no weight still means
`"count"`, so configs predating `shrinkage_weight` are unchanged.

**`minimum_support`** (`int ≥ 1`, default `1`) — a class needs this many
members to be matched; `1` is the no-thresholding option.

Note that "off" has four spellings across these: `shrinkage_weight=None`,
`shrinkage_strength=0.0`, `minimum_support=1`, and `class_estimator="pooled"`.

`chunk_size` sits in the same allowlist for a different reason — it changes
nothing about what a class *means*, but it is a fit-time memory decision
(design.md §4.1), not an inference knob.

`charge_experiments` adds one genuinely post-hoc option on top, applied to
`predict_raw` output rather than to the model: `normalization`, one of `None`
(default), `"std_weighted"`, or `"equal_weighted"`.

## Not implemented in v1

These are deliberate omissions, not oversights:

- **Vocabulary pruning** (design.md §8) — the spec calls it opt-in, not a
  default; nothing here needs it yet.
- **Full-covariance targets** (design.md §5.2) — the spec defers this
  upgrade; the merge monoid is a one-term change away from it when needed.
- **LOO under the newer estimators** — `predict_loo` supports
  `class_estimator="pooled"` with count-weighted shrinkage only; the
  correction needs the held-out node's child class identity, which the
  search loop does not retain. A deliberate scope cut, not a structural
  limit (design.md §4.4).

## Further reading

`design.md` is the authoritative specification for what Sieve computes and
why. `literature.md` places it among prior work (target encoding,
regressograms, WL kernels, COSMO-RS group contributions).
