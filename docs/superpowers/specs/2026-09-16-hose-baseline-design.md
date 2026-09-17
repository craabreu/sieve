# A HOSE-code baseline

Specification for a plain HOSE-code lookup predictor, to be added to
`experiments/experiments/predictors/` as a comparison arm. Everything here was
checked against the sources it cites, and every number in it was measured
unless it says otherwise.

## 1. Scope

Build **only** the baseline inference rule:

> describe an atom by its HOSE code out to *k* spheres; average the reference
> charges of the training atoms carrying the same code; if none does, shorten
> by one sphere and try again.

Do **not** port the continuation estimate (`sieve/continuation.py`) or the
empirical-Bayes shrinkage (`sieve/shrinkage.py`) onto HOSE codes. Those are the
manuscript's own contributions and the point of the arm is what the incumbent
tradition does without them. A HOSE arm carrying them would answer a question
nobody asked and would need the parent map that §4 exists to warn you about.

The arm answers: **how does the closest published lookup method do on this
corpus?** It is not an ablation of Sieve. HOSE codes encode bond order and
aromaticity, the fitted Sieve models here use element only, so the two differ
in features as well as in construction. Say so in whatever writes up the
result.

## 2. Why it exists

The manuscript concedes, in Sec. 2.2.1, that nested atomic environments,
per-class statistics, averaging over matched atoms, and backing off when
nothing matches are all long-established practice:

- Bremser's original register already holds an averaged shift, a standard
  deviation and a count of entries per code, and already estimates from a
  neighbouring entry when the code is absent (Anal. Chim. Acta **103**,
  355–365, 1978, Table 4 and the "Automatic estimation of unknown spectra"
  section).
- NMRShiftDB retreats one sphere at a time until a code is matched, and wants
  ten values before quoting a range (J. Chem. Inf. Comput. Sci. **43**,
  1733–1739, 2003, §6).
- Kuhn et al. run exactly the rule above for proton shifts: "we created a
  six-sphere HOSE code for each atom in the test set and tried to match this
  HOSE code in the 'training' set. If one or more values were found the
  average was considered for prediction. If no matches were found, we backed up
  sphere by sphere" (BMC Bioinformatics **9**, 400, 2008).

`TODO.md` in the manuscript repo notes that only DASH is compared and that
reviewers will ask about alternatives. This is the alternative that the
manuscript itself spends a paragraph on, so it is the one worth having.

## 3. The generator

```bash
pip install git+https://github.com/Ratsemaat/HOSE-code-generator
```

Stefan Kuhn's Python port of the Java CDK `HOSECodeGenerator`, which also
carries the stereo-aware extension of Kuhn & Johnson, ACS Omega **4**,
7323–7329 (2019). API:

```python
from hosegen import HoseGenerator
g = HoseGenerator()
code = g.get_Hose_codes(mol, atom_idx, max_radius=5)   # mol is an RDKit Mol
```

Other parameters (`usestereo`, `wedgebond`, `strict`, `ringsize`) are for the
stereo-aware form. **Leave them off.** The charges in this corpus are computed
per conformer but HOSE codes without stereo are a function of the 2D graph, and
turning stereo on would introduce a dependence on wedge bonds that the SDF's
own geometry does not reliably carry.

Instantiate `HoseGenerator()` once and reuse it; it holds only lookup tables.

### Format

A code looks like this, for ethanol's methyl carbon at `max_radius=6`:

```
C-4;HHHC(HHO/H/)//
```

`C-4;` is the centre (element, then the count of bonded partners including
hydrogens, then charge and ring codes when present). `HHHC` is sphere 1. The
remainder is one segment per sphere, closed by a delimiter. The delimiter
sequence is positional and comes from `HoseGenerator.sphere_delimiters`:

```python
["(", "/", "/", ")", "/", "/", "/", ...]   # "/" thereafter
```

so sphere *i* is closed by `sphere_delimiters[i-1]`. A code generated at a
radius deeper than the molecule reaches simply has empty segments: ethanol's
methyl carbon is `C-4;HHHC(HHO/H/)` at radius 4 and gains only delimiters
after that, while a C24 chain's content keeps growing to radius 12.

**Twelve spheres is a hard ceiling.** `sphere_delimiters` holds exactly twelve
entries and the generator indexes it directly, so `max_radius=13` raises a bare
`IndexError: list index out of range` from inside it, with no message of its
own. Validate `max_radius` against that ceiling in the constructor rather than
discovering it partway through a fit.

## 4. The key construction, and the trap

**Generate the full code once per atom and cut the shorter keys out of it as
string prefixes. Never call `get_Hose_codes` once per radius.**

```python
DELIMITERS = ["(", "/", "/", ")"] + ["/"] * 12    # hosegen.sphere_delimiters

def sphere_prefix(code: str, k: int) -> str:
    """The first k spheres of a full HOSE code, as a prefix of it.

    Returns the code unchanged if it holds fewer than k spheres."""
    pos = 0
    for i in range(k):
        nxt = code.find(DELIMITERS[i], pos)
        if nxt < 0:
            return code
        pos = nxt + 1
    return code[:pos]
```

For `C-4;HHHC(HHO/H/)//` this yields

```
k=1  'C-4;HHHC('
k=2  'C-4;HHHC(HHO/'
k=3  'C-4;HHHC(HHO/H/'
k=4  'C-4;HHHC(HHO/H/)'
k=5  'C-4;HHHC(HHO/H/)/'
```

Each key is a prefix of the next, so two atoms agreeing at radius *k+1* agree
at radius *k* automatically. That is the property the backoff needs, and this
construction gives it for free.

### Why not generate per radius

Because the shorter code the generator produces on its own is *not* the
truncation of the longer one — the branch grouping is rendered differently —
and the resulting classes do not form a tree. Measured over 699 atoms from 40
drug-like molecules, grouping atoms by their generated code at each radius:

| relation | child classes | with more than one parent |
|---|---:|---:|
| radius 3 → 2 | 340 | 5 |
| radius 4 → 3 | 428 | 5 |
| radius 5 → 4 | 461 | 2 |

A concrete case: the aromatic CH of phenol (`c1ccccc1O`, atom 1) and that of
salicylic acid (`OC1=CC=CC=C1C(=O)O`, atom 3) share the radius-3 code
`C-3;H*C*C(H,H*C,*C/*CO,H*&/)`, but their independently generated radius-2
codes are `C-3;H*C*C(H,H*C,*C//)` and `C-3;H*C*C(H,H,*C,*C//)` — the same
environment, different comma placement. With `sphere_prefix` the same test
gives **zero** multi-parent classes at every radius.

This is not a defect in the generator. It is what nmrshiftdb2 avoids by
construction: `PredictionTool.java` (SourceForge, `trunk/nmrshiftdb2`, r2665)
builds its shorter codes by re-tokenizing the full code on `()/` and
re-emitting the first *N* tokens with the separators reinserted, then matching
with `HOSE_CODE like '<prefix>%'`. Same idea as `sphere_prefix`, reached from
the other direction.

## 5. The predictor

### Interface

Implement `experiments.predictors.base.Predictor`:

```python
class Predictor(Protocol):
    name: ClassVar[str]
    def fit(self, train: MoleculeSet, val: MoleculeSet, *,
            rng: np.random.Generator) -> None: ...
    def predict(self, test: MoleculeSet) -> Prediction: ...
```

`predict` returns `Prediction(atom_value=...)`, one float per atom, in
`test`'s own flattened atom order (`test.atom_mol_id`-aligned). Do **not**
implement `NormalizablePredictor`; this arm is scored as it predicts, like
every other arm in the manuscript's Table 3.

`MoleculeSet.mols` is a list of RDKit `Mol`s, one per conformer, each atom
carrying the target as a double property. Hydrogens are **explicit** —
`prepare_dash.py` reads the SDF with `removeHs=False` — and atom order is the
file's, so the index you hand `get_Hose_codes` is the index of the atom whose
charge you are predicting. No mapping needed. `train.atom_target` gives the
flattened per-atom targets in the same order as `atom_mol_id`.

Model `GlobalMeanPredictor` in `predictors/global_mean.py` for the shape; it is
the shortest one in the tree.

### Fit

For each training atom: generate its full code at `max_radius`, then for every
*k* from 1 to `max_radius`, accumulate into `tables[k][sphere_prefix(code, k)]`
a running sum and count of the target. Also accumulate the global mean.

Store counts as well as means: the support threshold needs them, and reporting
the distribution of matched radii is worth having.

Keep sums and counts, not lists. The corpus is large and a list per class is
pointless for a mean.

### Predict

For each test atom, generate its full code once, then walk *k* down from
`max_radius` to 1, taking the first key present in `tables[k]` whose count is
at least `n_min`. Fall back to the global mean if no radius matches, which also
covers an element never seen in training.

Record the matched radius per atom. The fraction of atoms answered at each
radius is the single most informative diagnostic this arm produces, and
`TODO.md` in the manuscript repo wants that number for Sieve too.

### Parameters

| name | default | note |
|---|---|---|
| `max_radius` | 5 | matched to the radius Sec. 4.1 selects for Sieve. NMRShiftDB and Kuhn et al. both use 6; run both and report which |
| `n_min` | 1 | the manuscript's own eq. (3) threshold at its least restrictive setting, and what NMRShiftDB does for the estimate itself |

### Registration

Add to `predictors/__init__.py`, following the existing lazy-import pattern —
the generator is an optional dependency, so do not import it eagerly:

```python
if name == "hose" and name not in REGISTRY:
    import experiments.predictors.hose  # noqa: F401
```

and call `register("hose", _build)` at the bottom of the new module, as
`sieve_predictor.py` does.

## 6. Cost

Measured on a 41-atom drug-like molecule (`CC(=O)Nc1ccc(OCC(O)CNC(C)C)cc1`),
single-threaded, pure Python:

| `max_radius` | per atom | throughput |
|---|---:|---:|
| 3 | 162 µs | 6,166 atoms/s |
| 5 | 314 µs | 3,190 atoms/s |

The manuscript reports 12,497,841 carbon atoms in the training set, which at
that rate is about 1.1 h on one core for carbon alone; the count over all
elements is not measured here, but for drug-like molecules with explicit
hydrogens it is several times larger, so budget a few hours. Generating
separately at each radius would be `max_radius` times that — the other reason
§4's construction matters.

It parallelizes cleanly over molecules, and the corpus is already sharded, so
this is minutes on a handful of cores. Featurization is deterministic and
depends only on the molecule and on `max_radius`, so cache codes and reuse
them across every cross-validation fold rather than regenerating per fold.
Key the cache by molecule rather than by conformer: the codes read the 2D
graph, so all conformers of a molecule share them, and the corpus holds
1,029,785 conformers of 348,935 molecules.

**The cache does not survive a change of `max_radius`.** See §7.

## 7. Sweeping the radius

A depth sweep here does **not** have the economy that the manuscript's Sec. 4.1
claims for the other two arms, and the difference is structural rather than
incidental.

Sieve's levels are built bottom-up, so levels 0 through *k* of a model fitted
at the deepest radius are exactly the model fitted at radius *k*; DASH
recovers a shallower depth by truncating walked paths. In both, "depth *k*"
means one thing however deep the sweep went, and one fit serves every setting.

A HOSE code is a *linearization*, and ordering a sphere's branches consults
what lies beyond them, so generating deeper re-renders shallower spheres. The
containment fails. Measured over 780 (atom, *k*) pairs generated at
`max_radius` in {5, 6, 8, 12}, 66 of them -- 8.5% -- give a different *k*-sphere
prefix depending on which radius the code was cut from. For example, at *k*=3
in caffeine:

```
from max_radius 5, 6, 8   N-3;*C*CC(*C*C,H*N,HHH/*N*&,*N=O,*&/
from max_radius 12        N-3;*C*CC(*C*C,H*N,HHH/*N*&,=O*N,*&/
```

So **each point of a sweep regenerates its codes at its own `max_radius`**,
and the point at *k* is the arm you would actually deploy at *k*. Reading
every point out of one deep cache would instead sweep a family of arms none
of which is the deployed one, and would put a different meaning on the axis
than DASH's and Sieve's carry.

The predictor needs no flag for this. It already generates once at its own
`max_radius` and cuts its shallower backoff keys from that; a sweep is simply
one instance per candidate radius. What changes is only the expectation about
caching: the cache is reusable across folds at a fixed `max_radius`, and never
across settings.

The price, measured on a 41-atom drug-like molecule:

| `max_radius` | µs/atom | | `max_radius` | µs/atom |
|---:|---:|---|---:|---:|
| 1 | 94 | | 6 | 418 |
| 2 | 108 | | 7 | 546 |
| 3 | 163 | | 8 | 681 |
| 4 | 232 | | 9 | 856 |
| 5 | 309 | | 10 | 1085 |

One deep pass at radius 10 is 1085 µs/atom; ten native passes over *k*=1…10
total 4491 µs/atom, a factor of **4.1**. Much less than the ten-fold the naive
count suggests, because the shallow settings are cheap. That is the cost of an
axis that means the same thing as the other two arms'.

A sweep that only ever reports one radius does not pay this: fit the arm once
at the radius chosen and cut its backoff keys from that single generation, as
§4 says and as nmrshiftdb2 does.

## 8. Acceptance checks

1. `sphere_prefix(code, k)` is a prefix of `sphere_prefix(code, k+1)` for every
   code and every k. Property test it.
2. Grouping a few hundred atoms by their radius-(k+1) key never yields two
   distinct radius-k keys, at every k. This is §4's table, and it should read
   zero everywhere.
3. A held-out atom whose exact code is in training is answered at
   `max_radius`; one whose element is unseen falls back to the global mean.
4. Fit on two disjoint halves and check the per-class counts add up to a fit on
   the union. HOSE classes are just string keys, so this holds trivially — it
   is a check that the accumulation is correct, not a property of the method.
5. Codes are conformer-invariant: every conformer of one molecule must give
   identical codes and therefore identical predictions. This is true of the
   Sieve arms too, since both read the 2D graph, and it puts a floor under the
   error that neither method can cross. Worth stating once in the write-up.

## 9. Running it

The comparison harness is `experiments/experiments/cv.py` and
`experiments/experiments/compare.py`, which implement the repeated-measures
ANOVA and Tukey HSD protocol of Ash, Wognum & Rodríguez-Pérez (JCIM 2025,
doi:10.1021/acs.jcim.5c01609) that the manuscript's Study B follows. A new
predictor that satisfies the protocol and is registered needs no change to
either.

Run one shard first and report the wall clock and the error before committing
to the full cross-validation. If the throughput above holds, the full run is
cheap; if the corpus has atoms the generator chokes on, better to find out on
one shard.

## 10. Sources

- Bremser, W. *HOSE — a novel substructure code.* Anal. Chim. Acta **103**,
  355–365 (1978). doi:10.1016/S0003-2670(01)83100-7
- Steinbeck, C.; Krause, S.; Kuhn, S. *NMRShiftDB.* J. Chem. Inf. Comput. Sci.
  **43**, 1733–1739 (2003). doi:10.1021/ci0341363
- Kuhn, S.; Egert, B.; Neumann, S.; Steinbeck, C. *Building blocks for
  automated elucidation of metabolites.* BMC Bioinformatics **9**, 400 (2008).
  doi:10.1186/1471-2105-9-400
- Kuhn, S.; Johnson, S. R. *Stereo-Aware Extension of HOSE Codes.* ACS Omega
  **4**, 7323–7329 (2019). doi:10.1021/acsomega.9b00488
- Generator: https://github.com/Ratsemaat/HOSE-code-generator
- nmrshiftdb2 source: https://sourceforge.net/p/nmrshiftdb2/code/HEAD/tree/trunk/nmrshiftdb2/
