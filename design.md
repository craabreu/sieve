# WLLR — Core Data Structure and Estimator Design

**Status:** working design note, actively edited
**Date:** 2026-08-17
**Scope:** decisions reached in discussion, with the reasoning that produced them
**Relationship to `wllr.md`:** `wllr.md` is brainstorming output and is treated as a source of
ideas only. This document records what is actually settled. Where the two disagree, this one wins.

---

## 0. What this document covers

Everything below concerns how a fitted WLLR model is **represented, built, combined, and queried**.
It does not attempt to settle the method's statistical questions (choice of $K$, shrinkage strength,
evaluation protocol); those are listed as open in §9.

| Settled | Section |
|---|---|
| Per-level arrays, not `(depth, hash)` keys | §3 |
| Depth folded into the hash input | §3.3 |
| Graded attribute levels below depth 0 | §3.5 |
| Descriptive moments only; shrinkage derived | §4 |
| Immutable models combined by a merge monoid | §5 |
| Bottom-up inference with early termination | §6 |
| Vectorised fit: void-view dedupe + two-pass `bincount` | §7 |

§3.6 records a design option that has been **measured but not adopted** — coarsening the neighbour
attribute schema. Figures quoted elsewhere in this document as "measured on cosmobase" come from
that experiment.

---

## 1. Notation

Throughout, $k$ indexes a **level** of the refinement chain, $0\le k\le L$. In the base method a
level is a WL depth and $L=K$; with graded features (§3.5) the first levels refine the node's own
attributes before any WL round begins, and $L>K$. Nothing else in this document depends on which
kind of level $k$ refers to — that is the point of §3.5.

For level $k$ and identifier $c$:

$$
C_{k,c}=\{v:h_k(v)=c\},\qquad
N_{k,c}=|C_{k,c}|,\qquad
\bar y_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}y_v
$$

$C_{k,c}$ ranges over **labeled training nodes**. $N_{k,c}$ is that class's support — the quantity
that governs both level selection (§6) and shrinkage weight (§4.2).

$\sigma^2_{k,c}$ is the **population variance of the class** — the mean squared deviation, divisor
$N$:

$$
\sigma^2_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}\bigl(y_v-\bar y_{k,c}\bigr)^{2}
$$

Writing $d_v=y_v-\bar y_{k,c}$ makes the symmetry explicit: the stored triple is a count and **two
means**, $\bigl(N,\;\bar y,\;\overline{d^2}\bigr)$. That is the consistency the layout is chosen for
(§4.1).

The quantity actually **reported** is the unbiased sample variance,

$$
s^2_{k,c}=\frac{N_{k,c}}{N_{k,c}-1}\,\sigma^2_{k,c},
$$

which is undefined at $N_{k,c}=1$. Throughout, $\sigma^2$ is *stored* (divisor $N$, always defined)
and $s^2$ is *derived on access* (divisor $N-1$, undefined for singletons). The two are never
conflated.

$p(c)$ is the depth-$(k-1)$ parent of class $c$. $\mu_{\text{global}}$ is the training-set mean.

---

## 2. Three structural facts the design rests on

These are properties of WL refinement, not design choices. Everything downstream follows from them.

They depend on exactly one thing: that each level's identifier is computed **from the previous
level's identifier plus strictly more information**. Any construction with that property extends the
chain for free, which is why §3.5 costs nothing structurally.

**2.1 Nesting.** $h_k(v)$ is computed from $h_{k-1}(v)$, so the level-$k$ identifier determines the
level-$(k-1)$ identifier. Partitions are nested, $\Pi_0\succeq\Pi_1\succeq\cdots\succeq\Pi_L$, and each
node's classes form a chain $C_0(v)\supseteq C_1(v)\supseteq\cdots\supseteq C_L(v)$ [Kriege2016WLOA].

Consequently every class has **exactly one** parent. Whether this needs asserting depends on how ids
are minted: it is *structural* when they come from deduplicating signatures that contain the parent
(§7.2), and an invariant to check when they come from truncated digests, where a violation signals a
collision.

**2.2 Matched depths form a prefix.** If $h_k(v)\notin D_k$ then $h_{k+1}(v)\notin D_{k+1}$.

*Proof.* If $h_{k+1}(v)\in D_{k+1}$, some training node $u$ has $h_{k+1}(u)=h_{k+1}(v)$. By 2.1 the
depth-$(k+1)$ identifier determines the depth-$k$ one, so $h_k(u)=h_k(v)$ and hence $h_k(v)\in D_k$. ∎

So the set of supported depths along a query node's chain is $\{0,\ldots,k^\star\}$ or empty — never
scattered. This is what licenses early termination in §6.

**2.3 Support is monotone non-increasing.** Since $C_{k,c}\subseteq C_{k-1,p(c)}$, refinement can only
split a class, never merge one. Hence $N_{k,c}\le N_{k-1,p(c)}$ along any chain.

Two corollaries:

- A minimum-support cutoff also holds on a prefix, so it can use the same early termination.
- $N_{k,c}=N_{k-1,p(c)}$ implies the class did not split, so $C_{k,c}=C_{k-1,p(c)}$ **as sets** and the
  two records carry identical statistics. See §8.

---

## 3. The refinement chain and vocabulary layout

### 3.1 Decision

**One structure per depth, arrays rather than per-class objects.**

```python
# for each depth k
vocab[k]  : dict[bytes, int]      # WL identifier -> local class id
count[k]  : np.ndarray[int64]     # N
mean[k]   : np.ndarray[float64]   # ybar
msd[k]    : np.ndarray[float64]   # sigma^2, population variance, divisor N (§1)
parent[k] : np.ndarray[int32]     # index into depth k-1 arrays; -1 at depth 0
```

Class ids are local to a depth and dense from $0$. `parent[k][cid]` indexes directly into the
depth-$(k-1)$ arrays, so ancestor traversal involves no hashing at all.

### 3.2 Why not the alternatives

**Rejected — a single dictionary keyed by `(depth, hash)`.** Semantically correct but the weakest
layout: tuple keys pay a tuple-hash on every lookup, allocate more, don't serialize to JSON (tuple
keys aren't valid JSON keys), and give no cheap way to iterate a single depth — which both the
shrinkage pass (§4.2) and per-depth diagnostics need. It pays overhead for the cross-depth-collision
guarantee that per-depth structures provide for free.

**Considered — a single flat dictionary keyed by hash alone.** Sound, but it is a *semantic* change
rather than a re-layout: it asserts the hash already determines the depth. That holds only because
$h_0=H(x_v)$ and $h_k=H(h_{k-1},\text{multiset})$ serialize to differently-shaped byte strings — an
implicit invariant of the serializer. Change `φ` later and cross-depth collisions can become
systematic, silently merging a depth-2 environment with a depth-4 one.

The separate concern is per-class objects. A `WLClassStats` dataclass per class costs 100+ bytes of
Python object overhead, and there will be millions. Columnar arrays matter more than the key choice.

### 3.3 Depth is folded into the hash anyway

$$
h_k(v)=H\!\left(k,\;h_{k-1}(v),\;\operatorname{MULTISET}\{\phi(h_{k-1}(u),e_{uv}):u\in N(v)\}\right)
$$

This costs nothing and makes cross-depth collisions impossible **by construction** rather than by
accident. It also keeps the layout decision reversible: with depth in the hash, collapsing to a flat
dictionary later is a pure refactor.

### 3.4 Hash width

At least 128 bits. With $n$ distinct environments and a $b$-bit digest the birthday bound gives
$P_{\text{collision}}\approx n^2/2^{b+1}$: at $n=10^8$, 64 bits yields $\approx0.3$, 128 bits
$\approx10^{-20}$. A collision both merges unrelated environments and breaks the single-parent
invariant of §2.1.

### 3.5 Graded features below depth 0

**The problem.** With a single depth-0 identifier over the full attribute vector, global fallback
fires precisely when that vector is unseen — and it is a cliff. A carbon with an unusual
charge/hybridization combination goes straight from a fully specified atom to the mean of *every
labeled node in the dataset*. For atom-level regression that is the difference between a usable
prediction and a worthless one.

**The construction.** Replace the single $h_0$ with a sub-chain that introduces attributes one group
at a time, in a declared order $f_1,\ldots,f_m$:

$$
h_{0}=H(f_1),\qquad
h_{j}=H\bigl(h_{j-1},\,f_{j+1}\bigr)\quad\text{for } j=1,\ldots,m-1
$$

WL refinement then begins from $h_{m-1}$, so the full chain has $L=m-1+K$ levels: the first $m$ are
attribute levels, the rest WL depths. Backoff now degrades an unseen atom through
*element+hybridization* to *element* before ever reaching the global mean.

**Why it is free.** Each level is built from the previous one plus strictly more information, which
is the only premise §2 needs. Nesting, the prefix property, support monotonicity, the single-parent
invariant, the merge (§5), and bottom-up early termination (§6) all carry over with no modification —
these are simply more levels on the same chain. The vocabulary cost is negligible: attribute levels
are tiny (elements $\sim10$ classes, $+$hybridization $\sim40$).

**The order is declared, never learned.** It is a hyperparameter with $m!$ settings and must be
serialized with the model. Order by expected effect size, so backoff discards the least informative
attribute first. A reasonable default for scalar atomic properties:

$$
\text{element}\rightarrow\text{aromaticity/hybridization}\rightarrow\text{formal charge}
\rightarrow\text{H count}\rightarrow\text{chirality}
$$

Learning the order would move the method toward DASH's attention-derived expansion hierarchy and
forfeit the "no learned representation" position; declaring it a priori is what keeps the
distinction.

**What it does not fix.** This smooths the tail, not the dominant path. Once refinement has reached
WL level $\ge1$, backing off drops to the full-attribute level $m-1$ and loses *all* neighbor
information in one step; graded attributes do not soften that transition. The gain is confined to
nodes whose own attribute vector is unseen.

**How often that actually happens — measured.** On cosmobase (§3.6), under a molecule-level 80/20
split, global fallback fires for **0.03%** of held-out atoms at $n_{\min}=1$ and **0.11%** at
$n_{\min}=5$. The depth-0 cliff essentially never occurs on this chemistry, so graded attributes
help almost nobody here.

That is an in-distribution random split, and the feature is insurance against *shift* — unseen
elements, unusual charge or protonation states, a corpus that does not resemble the training set —
which such a split cannot exercise. The construction is cheap and structurally free, so it is kept.
But it should not be presented as a significant accuracy contribution on cosmobase-like data, and
any claim to that effect needs an out-of-distribution split behind it.

**Precedent.** Backing off over an ordered set of factors rather than a single context is exactly
the structure of factored language models with generalized parallel backoff
[Bilmes2003FactoredLM], including the problem of choosing the order.

### 3.6 Neighbour attribute resolution — evaluated, not adopted

**The idea.** Let neighbours contribute a *coarser* WL state than the centre. The centre keeps the
full attribute vector; neighbours contribute an element-only chain $g$:

$$
g_k(u)=H\bigl(g_{k-1}(u),\ \operatorname{MULTISET}\{g_{k-1}(w)\}\bigr),
\qquad
h_k(v)=H\bigl(h_{k-1}(v),\ \operatorname{MULTISET}\{\phi(g_{k-1}(u),e_{uv})\}\bigr)
$$

Note $h_k$ still takes $h_{k-1}$ as an explicit argument, so §2 holds unchanged and this stays a
single chain — nothing in §4–§7 is affected.

**A degenerate variant to avoid.** If neighbours contribute a *static* coarse attribute (their
element, not their element-only WL state) the multiset is identical at every level, so
$h_k=H(h_{k-1},\text{same multiset})$ and the partition stabilises at $k=1$. Nothing propagates.
The neighbour descriptor must itself be a refining chain.

**Measurement.** cosmobase, all 13,092 parseable molecules (147,412 heavy atoms, 11.3 per molecule),
levels 0–5, molecule-level 80/20 split, 2026-08-17. Full attribute schema: element, degree, formal
charge, aromaticity, hybridization, H count. Coarse schema: element only. Edge attributes stay at
full resolution in both arms.

| | full | coarse |
|---|---:|---:|
| mean matched level, $n_{\min}=5$ | 2.48 | **3.06** |
| median support at matched class | 28 | 28 |
| held-out atoms matching at level $\ge3$ | 42.4% | **61.7%** |
| singleton-class atoms at level 3 | 24.0% | 13.5% |
| distinct classes at level 2 | 25,382 | 9,895 |

Coarsening buys roughly **0.6 levels of extra reach at identical support**.

**Why it works here.** The non-element attributes are nearly redundant given element and bonding:

$$
H\bigl(\text{charge},\text{aromatic},\text{hybridization},n_H \;\big|\; \text{element},\text{bond multiset}\bigr)
= 0.133 \text{ bits of } 3.095
$$

so 95.7% of that information is already implied, and element-only neighbours discard almost nothing
while collapsing the neighbour alphabet about fourfold. This is a property of *this* corpus. A
dataset rich in charged species, tautomers, or unusual protonation states would score higher here
and benefit less — the entropy is the diagnostic to run before assuming the result transfers.

**Why this is not yet a decision.** The comparison is not like-for-like. Since the fine partition
refines the coarse one at every level (§2.1), coarse level $k$ is strictly *less* informative than
full level $k$. Coarsening therefore trades attribute resolution for topological reach at constant
support; it does not add information. Whether the trade pays depends on whether the target responds
more to nearby detail or to longer-range topology, and cosmobase carries no atom-level targets, so
this measurement cannot settle it. **The decisive experiment is MAE at the matched class with real
targets attached, not coverage.**

**A warning about synthetic validation.** An earlier synthetic motif corpus put the benefit at about
one percentage point — an order of magnitude too small. Template-built molecules make attributes
near-deterministic given element, which drives the conditional entropy above toward zero and makes
the two arms nearly identical *by construction*. Do not use synthetic graphs to size this effect.

---

## 4. What is stored

### 4.1 A count and two means

The model stores, per class, the triple $\bigl(N,\;\bar y,\;\sigma^2\bigr)$ plus `parent`, and the
same triple globally. Nothing else. The reported variance $s^2$ is **derived on access**.

**Why these three, and not a mixture.** The layout is chosen for dimensional consistency: $N$ is
extensive (it counts), while $\bar y$ and $\sigma^2$ are both intensive — per-observation means, of
$y$ and of $d^2$ respectively. A count and two means.

The tempting alternatives each break that consistency somewhere:

| Triple | Problem |
|---|---|
| $(N,\bar y,M_2)$ | mixes intensive $\bar y$ with extensive $M_2$ — no coherent reading |
| $(N,S,M_2)$ | consistent (all extensive) but loses branch-freeness; see below |
| $(N,\sum y,\sum y^2)$ | fully additive, numerically unusable |
| $(N,\bar y,s^2)$ | $s^2$ is undefined at $N=1$, the *dominant* case at large $K$ |

**Why not raw power sums.** $(N,\sum y,\sum y^2)$ makes the entire merge elementwise vector addition
with no correction term at all. It is the most additive choice and it is unusable: recovering the
variance needs $\sum y^2-S^2/N$, which cancels catastrophically. On targets with mean $10^6$ and
spread $3$ it errs by $2.2\times10^{-5}$ relative, against $2.8\times10^{-14}$ for the centered
forms — nine orders of magnitude. One non-additive term in the merge is the price of numerical
stability. Pay it.

**Why not the extensive triple.** $(N,S,M_2)$ is equally consistent and numerically identical, but
$S_A/N_A$ is $0/0$ for a class present only in one model, so the merge needs a guarded division. The
intensive form has no such case: weights are $N_A/N$ with $N=N_A+N_B\ge1$, never $0/0$ (§5.2).

**Why not the reported variance.** Storing $s^2$ directly fails hardest. It is undefined at $N=1$, so
`empty ⊕ singleton` — the first merge of any fold — has no value. The merge coefficients become
$(N_A-1)/(N-1)$ and $(N_B-1)/(N-1)$, which sum to $(N-2)/(N-1)$ rather than $1$, destroying the
weighted-average structure of §5.2. And $s^2$ is normalized by $N-1$, which counts nothing, so it is
neither cleanly intensive nor extensive.

The principle underneath: **store descriptive moments, apply inferential corrections at reporting.**
Bessel's correction is an estimator adjustment, not a property of the data held. Applying and
un-applying it on every merge is doing inference in the storage layer — the same reason shrunk means
are excluded from stored state (§4.2).

**Accumulation.** Never $\sum y^2-(\sum y)^2/n$ — see §7.4. Which stable algorithm to use depends on
how the data arrives:

- **Batch** (the normal case): the two-pass reduction of §7.3. Faster and simpler than a recurrence,
  because the whole shard is in memory at once.
- **Streaming**, one observation at a time: Welford's recurrence [Welford1962Variance], against
  running $(n,\bar y,M_2)$:

$$
n'=n+1,\qquad
\delta=y-\bar y,\qquad
\bar y'=\bar y+\frac{\delta}{n'},\qquad
M_2'=M_2+\delta\,(y-\bar y')
$$

The last step uses **both** the old and the new mean; using either one twice is a common and silently
wrong variant. In both cases $M_2$ is *scratch*; store $\sigma^2=M_2/n$ at the end.

**Variance access.** $s^2=\dfrac{N}{N-1}\,\sigma^2$, **undefined at $N=1$** — return `None`, never
`0.0`. A stored zero is indistinguishable from a genuinely homogeneous class and silently corrupts
any diagnostic reading variance as a confidence proxy. $\sigma^2$ itself is legitimately $0$ at
$N=1$; only the Bessel factor is undefined, so the guard lives in **one accessor** rather than in
every merge.

**Naming.** The array is `msd`, not `var` or `sigma2`. Nothing in the codebase called "variance"
should be capable of being mistaken for the reported $s^2$; the only thing bearing that name is the
accessor that applies the correction.

### 4.2 Shrinkage is derived, never stored

The hierarchically shrunk estimate

$$
\tilde\mu_{k,c}=\frac{N_{k,c}\,\bar y_{k,c}+\alpha_k\,\tilde\mu_{k-1,p(c)}}{N_{k,c}+\alpha_k},
\qquad
\tilde\mu_{0,c}=\frac{N_{0,c}\,\bar y_{0,c}+\alpha_0\,\mu_{\text{global}}}{N_{0,c}+\alpha_0}
$$

is computed by a top-down pass over the stored statistics — depth 0 first, since each level consumes
the **already-shrunk** parent, not the raw parent mean.

**It must not be stored as model state.** Any added data changes $\mu_{\text{global}}$, and every
$\tilde\mu$ depends on its full ancestor chain, so a single new node invalidates essentially every
shrunk value in the model. There is no incremental patch. Materialize it after fitting, or on demand.

This is also precisely why §5 works: sufficient statistics form a monoid under merging, and shrunk
means do not.

---

## 5. Immutable models with a merge monoid

### 5.1 Decision

A fitted model is **immutable and exactly sized**. Combining models is a pure function:

```python
model_c = model_a.merge(model_b)      # __add__ as an alias, for sum()
```

There is no mutable accumulator, no capacity doubling, no `_size`/`_cap`, no `freeze()`, and no stale
array views. Incremental training is not a separate mechanism — it is `model + fit(batch)`. Parallel
and distributed fitting are a fold over independently fitted shards.

### 5.2 Merging statistics

Counts add. With weights $w_A=N_A/N$, $w_B=N_B/N$ and $\delta=\bar y_B-\bar y_A$, both moments are
weighted averages:

$$
N=N_A+N_B,
\qquad
\bar y = w_A\bar y_A + w_B\bar y_B
$$
$$
\sigma^{2}
=
\underbrace{w_A\sigma^{2}_A + w_B\sigma^{2}_B}_{\text{within-group}}
\;+\;
\underbrace{w_A w_B\,\delta^{2}}_{\text{between-group}}
$$

This is the **law of total variance**, not an ad hoc correction factor — it is Chan, Golub and
LeVeque's parallel form [Chan1983ParallelVariance] expressed in intensive variables. Results are
bit-comparable to a single pass and independent of merge order.

Commutativity is manifest: $w_A\leftrightarrow w_B$ swaps the averaged terms, and $\delta\mapsto
-\delta$ leaves $\delta^2$ unchanged while $w_Aw_B$ is symmetric. Associativity is not visible by
inspection and remains a property test (§5.4).

**No edge cases.** Because $N=N_A+N_B\ge1$ whenever either side is non-empty, the weights are never
$0/0$. A class present only in B gives $w_A=0$, $w_B=1$, hence $\bar y=\bar y_B$ and
$\sigma^2=\sigma^2_B$ exactly — the between-group term vanishes with $w_A$. No guard, no branch, and
the empty model is a true identity. This is the concrete advantage of the intensive triple over
$(N,S,M_2)$, where $S_A/N_A$ would be $0/0$ in precisely this case.

### 5.3 Id remapping

The one genuine cost. A and B have independent id spaces, so the merge builds a new one. Keep the
work down by **pinning A's ids and remapping only B** — then A's `parent` array needs no translation,
and B's remap is carried forward across depths to fix B's parents.

```python
def merge_level(A, B, remap_prev):
    """Merge one depth. A's ids are preserved; only B's are remapped."""
    vocab = dict(A.vocab)
    remap = np.empty(len(B.vocab), np.int32)
    for h, bid in B.vocab.items():
        cid = vocab.get(h)
        if cid is None:
            cid = len(vocab)
            vocab[h] = cid
        remap[bid] = cid

    m, n_new = len(A.vocab), len(vocab)
    count  = np.zeros(n_new, np.int64);    count[:m]  = A.count
    mean   = np.zeros(n_new, np.float64);  mean[:m]   = A.mean
    msd    = np.zeros(n_new, np.float64);  msd[:m]    = A.msd
    parent = np.full(n_new, -1, np.int32); parent[:m] = A.parent

    i  = remap                     # a bijection, so no duplicate scatter writes
    nA = count[i].astype(np.float64)
    nB = B.count.astype(np.float64)
    n  = nA + nB
    wA, wB = nA / n, nB / n        # n >= 1 always: no 0/0, no guard needed

    delta = B.mean - mean[i]       # must precede the mean update below

    msd[i]    = wA * msd[i] + wB * B.msd + wA * wB * delta**2
    mean[i]   = wA * mean[i] + wB * B.mean
    count[i]  = n.astype(np.int64)
    parent[i] = -1 if remap_prev is None else remap_prev[B.parent]

    return FrozenLevel(vocab, count, mean, msd, parent), remap
```

Three things worth noting:

- **Order matters within the merge.** `delta` is read from `mean[i]` *before* `mean[i]` is updated,
  and `msd[i]` is likewise consumed before being overwritten. Reordering those lines silently
  corrupts every $\sigma^2$ for classes present in both models — measured at $1.0\times10^{-2}$
  relative error. Large enough that a property test catches it instantly, small enough that eyeballing
  a few predictions would not.
- **B-only classes need no branch at all.** A class absent from A has $w_A=0$, so it contributes
  nothing to either weighted average and the between-group term vanishes: `mean → B.mean` and
  `msd → B.msd` exactly.
- **Parent reassignment is an integrity check.** For classes in both, `remap_prev[B.parent]` must
  equal the parent already stored, by §2.1. Assert rather than overwrite and the hash-collision alarm
  costs nothing.

### 5.4 Watch-outs

**Fold as a balanced tree, not a chain.** Each merge is $O(|A|+|B|)$, so a sequential
`reduce` over $N$ shards is $O(N^2 s)$ — the accumulator grows and is re-copied every step. Pairwise
tree reduction is $O(Ns\log N)$.

**Reject incompatible configs loudly.** Differing $K$, feature schema, or hash version must raise.
This is a safety gain over mutation: config drift that in-place `partial_fit` would silently absorb
becomes an explicit error at a natural boundary. Do not truncate to $\min(K_A,K_B)$.

**Name it `merge()`.** `a + b` reads ambiguously as prediction ensembling; keep `__add__` only as an
ergonomic alias for `sum()`.

**Provide an empty-model identity** so `sum(models)` works and associativity/commutativity can be
property-tested directly. Those tests are cheap and catch remapping bugs immediately.

---

## 6. Inference

### 6.1 Bottom-up with early termination

Computing $h_L$ requires every $h_{k<L}$ anyway, so retaining the whole chain costs nothing. But the
search runs **upward**, not downward:

```text
best = global_mean
for k in 0..L:                                 # attribute levels then WL depths (§3.5)
    h = refine(k)                              # incremental
    if h not in vocab[k]:      break           # §2.2: all deeper levels also miss
    if count[k][id] < n_min:   break           # §2.3: support only decreases
    best = estimate(k, id)
return best
```

Identical answer to a top-down $L\to0$ scan, but never refines past $k^\star+1$. Both `break`
conditions are valid only because of the prefix properties in §2. The loop is indifferent to whether
a level refines attributes or neighborhoods, which is what makes §3.5 a pure extension.

### 6.2 The honest caveat

WL refinement is graph-wide, not per-node: computing depth $k$ for one node needs depth $k-1$ across
its whole neighborhood. So per-node early exit saves little on its own.

The version that pays is a **graph-level stop** — refine level by level and halt entirely once no node
in the graph still has a hit. That helps on graphs dominated by novel environments and does nothing
when most nodes match at the deepest level.

### 6.3 Depth selection

$k^\star$ is the deepest supported class on the chain, optionally subject to $N\ge n_{\min}$. Pure
deepest-match ($n_{\min}=1$) is available but should not be the default: at large $K$ classes fragment
toward singletons, so it recalls a single training label with undefined variance — maximum estimator
variance exactly where the model looks most confident.

**Grounded on cosmobase** (§3.6, full-attribute arm): singleton-class atoms rise from 9.6% at level 2
to 24.0% at level 3 and 42.7% at level 5. On held-out atoms, $n_{\min}=1$ matches at mean level 3.17
with median support **6**, while $n_{\min}=5$ matches at mean level 2.48 with median support **28** —
half a level shallower for four-and-a-half times the support. That is the trade $n_{\min}$ controls,
and it is why the default should not be 1.

---

## 7. Fit path

Fitting is fully vectorised: one pass per level over the entire corpus, with no per-molecule and no
per-atom Python loop. Timings below are cosmobase — 147,412 atoms, 289,774 directed edges, max
degree 6, levels 0–5, measured 2026-08-17.

```text
concatenate corpus block-diagonally, build CSR            (§7.1)
for k = 1..L:  refine all atoms at once                   (§7.2)
for k = 0..L:  two-pass bincount -> (N, mean, sigma^2)    (§7.3)
               parent falls out of the deduped signatures
    -> an immutable shard model
reduce shard models pairwise as a balanced tree           (§5.4)
optionally materialise shrunk estimates top-down          (§4.2)
```

### 7.1 Corpus layout

Concatenate every molecule into one block-diagonal graph with offset-adjusted indices, so a level is
a single array operation rather than a loop over molecules. Store edges in CSR order and precompute
each edge's position within its source atom's adjacency block:

```python
order  = np.argsort(src, kind="stable")
deg    = np.bincount(src, minlength=n_atoms)
indptr = np.concatenate([[0], np.cumsum(deg)])
slot   = np.arange(n_edges) - indptr[src]     # position within the node's block
```

`slot` is computed once and reused at every level.

### 7.2 Refinement

```python
pair = labels[dst] * n_bond + bond            # encode (neighbour label, bond)
pad  = np.full((n_atoms, max_deg), -1, np.int64)
pad[src, slot] = pair
pad.sort(axis=1)                              # canonical multiset; -1 pads sort first
sig  = np.concatenate([labels[:, None], pad], axis=1)
labels, uniq_rows = dense_rows(sig)           # dense ids
parent = uniq_rows[:, 0]                      # column 0 is the parent id
```

Padding with $-1$ makes the row length uniform while still encoding degree, since a node's pad count
is fixed. Two consequences worth noting:

- **`parent` is free, and single-parenthood becomes structural.** The parent id is column 0 of the
  signature, so all members of a class necessarily share it. The assertion §2.1 recommends is
  redundant on this path — it is only needed when identifiers come from truncated digests.
- **No cryptographic hashing is required to fit.** Dense ids come from deduplication. Hashing
  matters only for cross-run stable identifiers at serialisation (§3.3, §3.4).

**Dedupe rows through a void view, not `axis=0`.** This is the whole performance story:

| 6 levels over 147k atoms | time |
|---|---:|
| void-view `np.unique` | **0.27 s** |
| `np.unique(..., axis=0)` | 0.70 s |
| naive Python dict loop | 0.71 s |

`np.unique(axis=0)` is *no faster than the dict loop*. Viewing each row as a `np.void` scalar and
deduplicating in 1-D is what buys the 2.6×:

```python
def dense_rows(mat):
    m = np.ascontiguousarray(mat)
    v = m.view(np.dtype((np.void, m.dtype.itemsize * m.shape[1]))).ravel()
    # numpy returns (unique, index, inverse) in a FIXED order, whatever
    # order the keywords are passed in
    uniq, idx, inv = np.unique(v, return_index=True, return_inverse=True)
    return inv.ravel().astype(np.int64), m[idx]
```

Per-level breakdown: gather+encode 1.0 ms, scatter 1.5 ms, row-sort 2.8 ms, dedupe 51 ms. Dedupe
dominates by an order of magnitude; optimising anything else is wasted effort.

### 7.3 Statistics

```python
N    = np.bincount(labels, minlength=nc)
S    = np.bincount(labels, weights=y, minlength=nc)
mean = np.divide(S, N, out=np.zeros(nc), where=N > 0)
d    = y - mean[labels]                       # centre first, then reduce
M2   = np.bincount(labels, weights=d * d, minlength=nc)
sigma2 = np.divide(M2, N, out=np.zeros(nc), where=N > 0)
```

| level-2 aggregation, 147k atoms → 25,382 classes | time |
|---|---:|
| two-pass `bincount` | **1.0 ms** |
| `reduceat` (including the sort it requires) | 11.8 ms |
| Welford loop | 109.5 ms |

### 7.4 Two traps

**`reduceat` instead of `bincount`.** It requires the data sorted by group — an $O(n\log n)$ argsort
`bincount` avoids — and it corrupts empty groups silently. When `starts[i] >= starts[i+1]` it returns
`y[starts[i]]` rather than $0$:

```text
labels [0 0 3 3]   y [10 20 30 40]      # classes 1 and 2 empty
bincount : [30.  0.  0. 70.]
reduceat : [30. 30. 30. 70.]            <- classes 1,2 silently wrong
```

Empty groups cannot occur while ids are interned per shard, but they appear the moment a shard is
indexed against a merged global vocabulary — which is exactly what §5 does. Use `bincount`.

**Power sums instead of centring.** The one-pass form $Q/N-\bar y^2$ with $Q=\sum y^2$ is the obvious
vectorisation and is unusable, as §4.1 argues on principle and this measures in practice. On targets
with mean $10^6$ and spread $3$:

| | max rel. error vs Welford | negative variances |
|---|---:|---:|
| two-pass `bincount` | 5.4e-08 | 0 |
| `reduceat` | 5.4e-08 | 0 |
| power sums | **1.3e+02** | **2** |

Centring costs one extra pass and one gather. The fast-looking formula and the correct formula are
not the same formula.

### 7.5 Input alignment is the highest-severity failure mode

When targets live in a separate array indexed by atom position, a misalignment between the parsed
molecule and its target rows corrupts every label with **no error raised**. `MolFromSmiles` does
preserve the input SMILES heavy-atom order, but `AddHs` appends hydrogens at the end, and any
canonicalisation round-trip reorders.

Guard it on load: assert per-molecule atom counts match the row slice, and store element symbols
alongside the targets so atomic numbers can be verified against the parsed molecule. The check costs
milliseconds and rules out the one bug that otherwise surfaces only as unexplained inaccuracy.

---

## 8. Optional vocabulary pruning

By §2.3, $N_{k,c}=N_{k-1,p(c)}$ means the class did not split, so its mean and variance are identical
to its parent's. Under **raw** means those records are pure redundancy: dropping them leaves every
prediction unchanged, since backoff resolves to the parent and returns the same value.

Because a class that becomes a singleton stays a singleton at every deeper level, this collapses each
singleton chain to its first appearance — most of the tail at large $K$.

Two caveats:

1. `matched_depth` in prediction metadata would report the surviving shallower depth.
2. It is **not** prediction-preserving under shrinkage, where $\alpha_k$ applies again at each level.

Treat as an opt-in compaction of a finished raw-mean model, not a default.

---

## 9. Open questions

1. **Default $n_{\min}$.** Argued above that 1 is a poor default; the value should be set empirically
   on the first real dataset rather than by fiat.
2. **Raw vs shrunk as the headline model.** Shrinkage handles low support continuously and is the
   more defensible default; raw deepest-match is the configuration most exposed to criticism.
3. **Choice of $K$**, and whether to auto-detect WL partition stabilization on the training fold.
4. **Serialization format.** The columnar layout maps onto `.npz` plus a small JSON sidecar for
   `vocab` and config, but this is not yet decided.
5. **Whether `vocab` should stay `dict[bytes, int]`** or move to a sorted array with binary search,
   which would serialize more compactly at some lookup cost.
6. **The attribute order in §3.5**, and its granularity — one level per attribute, or grouped levels
   (e.g. aromaticity and hybridization together). The proposed chemical default is a starting point,
   not a measured one.
7. **Whether neighbours should carry a coarser attribute schema than the centre (§3.6).** Measured on
   cosmobase: coarsening buys ~0.6 levels of extra reach at identical support. What remains open is
   whether that reach is *worth* the attribute resolution it costs, which needs targets. Run the
   comparison again with real σ-profiles or partial charges and decide on MAE at the matched class.
   Implement as a configurable neighbour schema — an ablation flag, not an architecture.

---

## 10. Deliberately not settled here

Statistical and experimental questions — evaluation protocol, baselines, splitting strategy, leakage
controls, the relationship to message-passing GNNs, and the novelty argument — are untouched by this
document. `wllr.md` holds draft material on all of them, none of it confirmed.
