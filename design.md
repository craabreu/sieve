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
| Sufficient statistics only; shrinkage derived | §4 |
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

$M_{2}$ is the **sum of squared deviations from the class mean**:

$$
M_{2,k,c}=\sum_{v\in C_{k,c}}\bigl(y_v-\bar y_{k,c}\bigr)^{2}
$$

so that the unbiased sample variance is $s^2_{k,c}=M_{2,k,c}/(N_{k,c}-1)$. In code this is the `m2`
array. It is a plain sum of squares — *not* a variance, and not divided by anything — which is
exactly why it is the stored form; see §4.1.

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
count[k]  : np.ndarray[int64]
mean[k]   : np.ndarray[float64]
m2[k]     : np.ndarray[float64]   # sum of squared deviations from the mean (§1)
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

### 4.1 Sufficient statistics only

The model stores `count`, `mean`, `m2`, `parent`, plus global `(N, mean, m2)`. Nothing else.

**Why $M_2$ rather than the variance.** Variance does not compose: you cannot combine two classes'
variances without recovering their sums of squares first. $M_2$ does compose, via the merge formula
in §5.2. Storing $M_2$ is therefore what makes §5's merge monoid possible at all — the same reason
shrunk means are excluded from stored state (§4.2). Variance is a *presentation* concern, computed
only when someone asks for it.

Accumulate $M_2$ with Welford's recurrence [Welford1962Variance] rather than the textbook
$\sum y^2-(\sum y)^2/n$, which loses catastrophic precision when the mean is large relative to the
spread. For a new observation $y$ against running $(n,\bar y,M_2)$:

$$
n'=n+1,\qquad
\delta=y-\bar y,\qquad
\bar y'=\bar y+\frac{\delta}{n'},\qquad
M_2'=M_2+\delta\,(y-\bar y')
$$

Note the last step uses **both** the old and the new mean; using either one twice is a common and
silently wrong variant.

Variance is derived as $M_2/(N-1)$ and is **undefined at $N=1$** — surface it as `None`, never `0.0`.
A stored zero is indistinguishable from a genuinely homogeneous class and silently corrupts any
diagnostic that reads variance as a confidence proxy. $M_2$ itself is legitimately $0$ at $N=1$; it is
the division by $N-1$ that is undefined, so the guard belongs in the accessor, not the accumulator.

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

Chan, Golub and LeVeque's parallel form [Chan1983ParallelVariance] merges two accumulators exactly:

$$
n=n_A+n_B,\qquad \delta=\bar y_B-\bar y_A
$$
$$
\bar y=\bar y_A+\delta\frac{n_B}{n},\qquad
M_2=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n}
$$

Results are bit-comparable to a single pass and independent of merge order.

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
    m2     = np.zeros(n_new, np.float64);  m2[:m]     = A.m2
    parent = np.full(n_new, -1, np.int32); parent[:m] = A.parent

    i  = remap                     # a bijection, so no duplicate scatter writes
    nA = count[i].astype(np.float64)
    nB = B.count.astype(np.float64)
    n  = nA + nB
    delta = B.mean - mean[i]

    mean[i]   = mean[i] + delta * nB / n
    m2[i]     = m2[i] + B.m2 + delta**2 * nA * nB / n
    count[i]  = n.astype(np.int64)
    parent[i] = -1 if remap_prev is None else remap_prev[B.parent]

    return FrozenLevel(vocab, count, mean, m2, parent), remap
```

Two things fall out for free:

- **No branch for B-only classes.** Where a class exists only in B, the zero-initialized target gives
  $n_A=0$ and $\delta=\bar y_B$, so `mean → B.mean` and `m2 → B.m2` exactly.
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
        Welford-update (count, mean, m2)
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
