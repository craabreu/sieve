# Sieve

**S**upport-gated **I**nference over **E**nriched **V**ertex **E**nvironments.

Sieve is a node-level regressor for labelled graphs (molecules, in practice)
that generalises the classical regressogram to a *nested hierarchy* of
partitions: it refines each node's environment through graded attribute
levels and rounds of Weisfeiler–Lehman colour refinement, fits mean and
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

## A fifteen-line worked example

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
cfg = SieveConfig(target_dim=1, attribute_levels=(("element", "degree", "aromatic"),),
                 attribute_codes=codes, edge_codes=edges, max_wl_depth=2, n_min=1)

batch = from_smiles(smiles, y=np.repeat(y, [m.GetNumAtoms() for m in mols], axis=0),
                    config=cfg)
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

## Not implemented in v1

These are deliberate omissions, not oversights:

- **Vocabulary pruning** (design.md §8) — the spec calls it opt-in, not a
  default; nothing here needs it yet.
- **The neighbour schema** (design.md §3.6) — the spec says it was
  evaluated, not adopted. The config field exists and raises
  `NotImplementedError` if set, so a future implementer has to make the
  decision deliberately rather than by accident.
- **Full-covariance targets** (design.md §5.2) — the spec defers this
  upgrade; the merge monoid is a one-term change away from it when needed.

## Further reading

`design.md` is the authoritative specification for what Sieve computes and
why. `literature.md` places it among prior work (target encoding,
regressograms, WL kernels, COSMO-RS group contributions).
