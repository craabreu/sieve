# Radius-resolved chirality

Design notes for a mirror-invariant, radius-honest chirality feature in the
refinement chain. Everything here was checked against the sources it cites,
and every number in it was measured unless it says otherwise.

## 1. Status: under consideration

Stereochemistry modelling was scoped out of the manuscript on 2026-09-18 and
is being reconsidered. Nothing here is implemented.

As the manuscript stands, stereo information serves two purposes only:
defining invariances (`collapse_key`, which merges enantiomers and keeps
diastereomers and E/Z apart) and computing floors (the keyed and stereo-blind
floors, both reported in §4.4). Two sentences would have to change if this
design were adopted: `main.tex:490`, which states that no attribute set reads
stereochemistry, and `main.tex:403`, which states that coordinates are never
seen by a predictor.

Two costs are worth weighing before committing to it, and neither is a matter
of taste. It changes what a class means, so it changes `schema_version` and
**invalidates every fit on disk** — Studies A, B and C would all be re-run.
And §3 shows no model here is anywhere near floor-limited by stereochemistry,
so the expected gain is small; §8 lists what to measure before building.

## 2. Why absolute chirality is the wrong feature

MBIS charges are invariant under reflection: enantiomers carry identical
charges atom for atom. So with *n* stereocentres the 2^n absolute
configurations hold only 2^(n-1) distinct charge sets, and **an absolute
chiral tag always doubles the class count for zero information**. The factor
does not improve with *n*.

Measured on the corpus (5,000 molecules sampled):

| | share of molecules |
|---|---|
| 0 stereocentres — stereo irrelevant | 48.0% |
| exactly 1 — an absolute tag is pure loss | 27.4% |
| >= 2 — relative configuration can matter | 24.6% (27.5% of atoms) |

So three quarters of the corpus is unaffected or actively harmed by an
absolute tag. Only *relative* configuration carries information.

`_chirality` in `src/sieve/io/rdkit_adapter.py` is the absolute variant: it
reads RDKit's CIP `_CIPCode`. Its own docstring calls that "a whole-molecule
computation". That is a second defect, independent of the invariance
question — see §4.

## 3. How much is at stake

Both floors, measured on the test split (102,824 conformers, 4.2M atoms):

| | keyed floor | stereo-blind floor | gap |
|---|---|---|---|
| whole test split | 0.009168 | 0.009348 | 0.000180 |
| stereo-sensitive subset | 0.009039 | **0.011161** | **0.002123** |

The "stereo-sensitive subset" is the set of conformers whose stereo-blind key
covers more than one `collapse_key` — molecules with a diastereomeric partner
actually present, which is where a stereo-blind model is forced to merge
things with different targets. It is 7.81% of conformers and 7.77% of atoms,
and it carries **99.8% of the global blind-vs-keyed SSE gap**. The corpus-wide
number is a large effect diluted thirteenfold; on the subset the gap is 11.8x
larger.

Even so, the best model sits at 0.019862 e, roughly twice the subset's blind
floor of 0.011161. **Nothing here is floor-limited**, which is the empirical
case for leaving the feature out.

## 4. The definition: four distinguishable neighbours

An atom is a stereocentre, as far as radius *k* can tell, when it has exactly
four neighbours whose labels at radius *k*-1 are all distinct. No chiral tag,
no CIP, no `FindMolChiralCenters`, and no element restriction — the rule never
names carbon.

Measured on 2,000 molecules (84,557 atoms), element-only refinement:

| radius *k* | atoms with 4 distinct neighbour labels at *k*-1 | % of atoms |
|---|---|---|
| 1 | 45 | 0.05% |
| 2 | 1,511 | 1.79% |
| 3 | 1,887 | 2.23% |
| 4 | 1,954 | 2.31% |
| 5 | 1,963 | 2.32% |

Two results:

- **Zero false positives.** All 1,963 resolved atoms are atoms RDKit
  independently calls chiral centres. Given only elements and connectivity,
  the refinement rediscovers exactly the chemical notion.
- **Incomplete at finite radius.** 158 of RDKit's 2,121 centres (7.4%) are
  still unresolved at *k*=5: their branches first differ more than five bonds
  out.

That incompleteness is **not a defect of this construction** — it is Sieve's
premise. A radius-5 model misses every distinction needing more than five
bonds, constitutional ones included. The anomaly is the other way round: CIP
reaches to unbounded depth and injects the answer into a label that claims to
summarise *k* bonds. A radius-resolved sign would be the first stereo
treatment in this codebase that obeys the method's own rule.

It also dissolves chirality as a special case. It becomes one more distinction
appearing at the radius that earns it, covered by the sentence already in §4
of the manuscript: *refinement does not proceed at a uniform rate, in that it
separates only what it is given reason to separate.*

## 5. The encoding

Three independent pieces. The confusion to avoid is treating them as one.

### 5.1 A sign that depends on nothing arbitrary

At featurisation, compute one **reference parity** per atom from the batch's
own edge order:

    p_ref(v) = sign[ (r0 - r3) . ((r1 - r3) x (r2 - r3)) ]

where 0,1,2,3 are `v`'s neighbours *in whatever order the batch stores them*.
That order is arbitrary, deliberately.

At radius *k*, inside `refine()`, sort those four neighbours by `h_{k-1}` and
let `pi` be the permutation from batch-edge-order to sorted order:

    s_k(v) = p_ref(v) * sgn(pi)

**The arbitrariness cancels exactly.** Permuting the stored edge order by
`tau` sends `p_ref -> p_ref*sgn(tau)` and `sgn(pi) -> sgn(pi)*sgn(tau)`, so the
product picks up `sgn(tau)^2 = 1`. This is what lets a meaningless reference
order yield a canonical sign, and it splits the work across the right
boundary: geometry is touched only at featurisation, and `refine()` sees one
integer per atom.

### 5.2 One column in the signature

A signature row is already integers (parent id in column 0, sorted
(neighbour label, bond) codes after). Add **one column** holding
`{0,1,2}` for `{unresolved, +, -}`. `dense_rows` dedups on it for free, and
`merge._translate` passes it through untouched — it is a value like a bond
code, not an id needing remapping.

The column is 0 wherever the atom is not four-coordinate or any two neighbour
labels tie, which is 97.7% of atoms, so the feature is inert where it does not
apply.

### 5.3 Mirror invariance, without mirroring anything

Reflection negates every signed volume and changes nothing else — same
connectivity, elements and bonds. So the mirrored chain is the identical
computation with one input flipped:

    g-chain = h-chain with p_ref -> -p_ref

No `mirror_mol`, no coordinate reflection, no second `from_rdkit`. One extra
refinement pass over the same batch with one sign array negated.

The invariant class at radius *k* is the **sorted pair** `(h_k, g_k)`, as a
two-column signature following the existing `LEVEL_WL_PAIR` pattern. Nesting
survives: the pair at level *k* determines the pair at *k*-1, because each
component does. Where the sign column is 0 the two chains coincide, the pair
is `(a,a)`, and nothing splits.

## 6. Rejected alternatives

**`min(h_k, h_k^mir)` on class ids.** Mathematically faithful — `h_k`
determines its mirror, so either representative identifies the orbit — but it
breaks the merge monoid. `dense_rows` ids are fit-local and `merge.py`
reconciles them by remapping level by level, so two shards can canonicalise
the same environment to opposite branches and the merge silently unions
unrelated classes. Hence the sorted pair, not the minimum.

**Relative signs on neighbour entries.** Storing products `s(v)s(u)` is
mirror-invariant and needs no second chain, but reaches only *adjacent*
centres. Recovering `s(v)s(u)` for centres several bonds apart means chaining
products along the path, and the chain breaks at the first intermediate atom
that is not itself a stereocentre — almost all of them. The pair construction
never represents relative signs at all; they fall out.

**Chiral tags read off the `collapse_key` representative.** Sound for
collapse, which is a global operation, but not for featurisation: the
`min(SMILES, mirror SMILES)` sign is chosen by comparing whole-molecule
strings, so two atoms with identical local environments can receive opposite
tags because of atoms far away. Local canonicalisation has to be local.

## 7. Precedent

InChI has done the invariance decomposition since 2005, and verified against
the InChI Trust technical FAQ:

- "InChI does not use CIP rules and deduces parities from its own canonical
  numbers of atoms." (§8.5)
- "If two compounds are declared enantiomers they will have the SAME /t string
  and differ in the /m layer (/m0 or /m1)." (§12.5)

So `/t` is the mirror-invariant relative description and `/m` the single
global sign bit. **Dropping `/m` is exactly what §5.3 does.** InChI hits the
tie problem too and resolves it by minimising over the equivalent canonical
numberings; Sieve defers instead, which is better — it has a radius, and an
unresolved sign simply appears at the radius that resolves it.

One caveat from the same documentation: under reflection "not all parities are
necessarily inverted; for example, parities of the stereogenic atoms involved
in a cis or trans arrangement do not change." The "every local sign flips"
identity holds for tetrahedral centres but not once other stereo types are
present. `collapse_key` already encodes that asymmetry, inverting tetrahedral
tags while leaving double-bond geometry untouched.

The precedent covers the decomposition, not a **local, radius-resolved**
version of it — InChI's canonicalisation is whole-molecule. No published
analogue was found for that part. The chirality-aware GNN literature
(Tetra-DMPNN, ChiENN) solves the opposite problem, making models
chirality-*sensitive*; ChiENN discusses neither WL expressivity nor reflection
invariance. ECFP/Morgan with chirality uses absolute R/S parity codes, i.e.
the over-splitting variant.

## 8. Open questions

1. **Conformer robustness — the one that matters.** Is the signed volume
   stable across conformers of one molecule? 7,452 `dash_id`s in the store
   already have conformers disagreeing on perceived E/Z, because RDKit reads
   geometry and floppy amidine-type C=N groups land differently. Four-
   coordinate sp3 centres *should* be safe, since inverting one needs bond
   breaking whereas the E/Z cases were three-coordinate and near-planar, but
   that is an argument, not a measurement. Check by comparing signs across
   conformers of each `dash_id` before building anything.
2. **Cost.** Two refinement passes. §5.3 makes the second one cheap (no
   re-featurisation) but it is still 2x the refinement.
3. **The 158 unresolved centres** at *k*=5 (7.4%). Accepted under §4, but
   worth knowing whether they concentrate in any chemistry.
4. **`main.tex:403`** would need rewording: signed volumes are arguably still
   perception rather than modelling, since they yield a discrete sign, but the
   sentence as written would no longer be true.

## 9. Provenance

Measurements in §2 and §4 are from 5,000 and 2,000 molecule samples of
`dash-molecules` respectively; §3 is the full test split. Scripts were
scratch, not committed. The InChI quotations are from
<https://www.inchi-trust.org/technical-faq/>, read 2026-09-18.
