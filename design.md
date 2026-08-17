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
| Per-depth arrays, not `(depth, hash)` keys | §3 |
| Depth folded into the hash input | §3.3 |
| Descriptive moments only; shrinkage derived | §4 |
| Immutable models combined by a merge monoid | §5 |
| Bottom-up inference with early termination | §6 |

---

## 1. Notation

For WL depth $k$ and identifier $c$:

$$
C_{k,c}=\{v:h_k(v)=c\},\qquad
N_{k,c}=|C_{k,c}|,\qquad
\bar y_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}y_v
$$

$C_{k,c}$ ranges over **labeled training nodes**. $N_{k,c}$ is that class's support — the quantity
that governs both depth selection (§6) and shrinkage weight (§4.2).

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

**2.1 Nesting.** $h_k(v)$ is computed from $h_{k-1}(v)$, so the depth-$k$ identifier determines the
depth-$(k-1)$ identifier. Partitions are nested, $\Pi_0\succeq\Pi_1\succeq\cdots\succeq\Pi_K$, and each
node's classes form a chain $C_0(v)\supseteq C_1(v)\supseteq\cdots\supseteq C_K(v)$ [Kriege2016WLOA].

Consequently every class has **exactly one** parent. This is an invariant worth asserting, not
assuming — a violation means a hash collision.

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

## 3. Vocabulary layout

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

**Accumulation.** Use Welford's recurrence [Welford1962Variance], not $\sum y^2-(\sum y)^2/n$. For a
new observation $y$ against running $(n,\bar y,M_2)$:

$$
n'=n+1,\qquad
\delta=y-\bar y,\qquad
\bar y'=\bar y+\frac{\delta}{n'},\qquad
M_2'=M_2+\delta\,(y-\bar y')
$$

The last step uses **both** the old and the new mean; using either one twice is a common and silently
wrong variant. $M_2$ here is *scratch state during accumulation*; store $\sigma^2=M_2/n$ at the end.

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

Computing $h_K$ requires every $h_{k<K}$ anyway, so retaining the whole chain costs nothing. But the
search runs **upward**, not downward:

```text
best = global_mean
for k in 0..K:
    h = refine(k)                              # incremental
    if h not in vocab[k]:      break           # §2.2: all deeper levels also miss
    if count[k][id] < n_min:   break           # §2.3: support only decreases
    best = estimate(k, id)
return best
```

Identical answer to a top-down $K\to0$ scan, but never refines past $k^\star+1$. Both `break`
conditions are valid only because of the prefix properties in §2.

### 6.2 The honest caveat

WL refinement is graph-wide, not per-node: computing depth $k$ for one node needs depth $k-1$ across
its whole neighborhood. So per-node early exit saves little on its own.

The version that pays is a **graph-level stop** — refine level by level and halt entirely once no node
in the graph still has a hit. That helps on graphs dominated by novel environments and does nothing
when most nodes match at depth $K$.

### 6.3 Depth selection

$k^\star$ is the deepest supported class on the chain, optionally subject to $N\ge n_{\min}$. Pure
deepest-match ($n_{\min}=1$) is available but should not be the default: at large $K$ classes fragment
toward singletons, so it recalls a single training label with undefined variance — maximum estimator
variance exactly where the model looks most confident.

---

## 7. Fit path

```text
for each shard / batch:
    refine all graphs to depth K, retaining h_0..h_K per node
    for k = 0..K ascending:
        intern h_k into vocab[k]
        Welford-update (count, mean, M2); store sigma^2 = M2/N at the end
        record parent id from depth k-1, asserting uniqueness
    -> an immutable shard model

reduce shard models pairwise as a balanced tree   (§5.4)

optionally materialize shrunk estimates top-down  (§4.2)
```

Fitting is two passes if you want exact allocation without any growth logic: one to build `vocab[k]`,
one to accumulate. With the merge design, growth logic is unnecessary in either case.

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

---

## 10. Deliberately not settled here

Statistical and experimental questions — evaluation protocol, baselines, splitting strategy, leakage
controls, the relationship to message-passing GNNs, and the novelty argument — are untouched by this
document. `wllr.md` holds draft material on all of them, none of it confirmed.
