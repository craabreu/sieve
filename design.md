# Sieve — Core Data Structure and Estimator Design

**Status:** working design note, actively edited
**Date:** 2026-08-17
**Scope:** decisions reached in discussion, with the reasoning that produced them
**Relationship to other documents:** `drafts/wllr.md` is superseded brainstorming output, treated as
a source of ideas only; where the two disagree, this one wins. `literature.md` holds the literature
review and novelty assessment and makes no implementation claims.

---

## 0. What this document covers

Everything below concerns how a fitted Sieve model is **represented, built, combined, and queried**.
It does not attempt to settle the method's statistical questions (choice of $K$, shrinkage strength,
evaluation protocol); those are listed as open in §13.

| Settled | Section |
|---|---|
| Per-level arrays, not `(depth, hash)` keys | §3 |
| Depth folded into the hash input | §3.3 |
| Graded attribute levels below depth 0 | §3.5 |
| Descriptive moments only; shrinkage derived | §4 |
| Immutable models combined by a merge monoid | §5 |
| Bottom-up inference with early termination | §6 |
| Vector targets, per-dimension variance | §1, §5.2 |
| Vectorised fit: void-view dedupe + sparse two-pass reduction | §7 |
| Shared target grid; unnormalised areas; closure over non-negative vectors | §11.4 |

§3.6 records a design option that has been **measured but not adopted** — coarsening the neighbour
attribute schema. Figures quoted elsewhere in this document as "measured on cosmobase" come from
that experiment.

---

## 1. Notation

Throughout, $k$ indexes a **level** of the refinement chain, $0\le k\le L$. In the base method a
level is a WL depth and $L=K$; with graded features (§3.5) the first levels refine the node's own
attributes before any WL round begins, and $L>K$. Nothing else in this document depends on which
kind of level $k$ refers to — that is the point of §3.5.

**Targets are vectors.** $y_v\in\mathbb{R}^{d}$, with $d=1$ the scalar case rather than a special
case — nothing branches on it. For the motivating application $d$ is the width of a σ-profile,
typically a few tens of bins.

For level $k$ and identifier $c$:

$$
C_{k,c}=\{v:h_k(v)=c\},\qquad
N_{k,c}=|C_{k,c}|\in\mathbb{N},\qquad
\bar y_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}y_v\ \in\mathbb{R}^{d}
$$

$C_{k,c}$ ranges over **labeled training nodes**. $N_{k,c}$ is that class's support — a scalar, even
for vector targets, since a node is present or absent as a whole. It governs both level selection
(§6) and shrinkage weight (§4.2).

$\sigma^2_{k,c}\in\mathbb{R}^{d}$ is the **per-dimension population variance** of the class — the mean
squared deviation, divisor $N$, taken elementwise:

$$
\sigma^2_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}\bigl(y_v-\bar y_{k,c}\bigr)^{\odot 2}
$$

Writing $r_v=y_v-\bar y_{k,c}$ for the residual makes the symmetry explicit: the stored triple is a
count and **two means**, $\bigl(N,\;\bar y,\;\overline{r^{\odot2}}\bigr)$. That is the consistency the
layout is chosen for (§4.1).

This stores only the diagonal of the class covariance. The full matrix is a supported future
extension at the cost of one term in the merge and $O(d^2)$ storage — see §5.2.

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
count[k]  : np.ndarray[int64]     # (n_classes,)      N -- scalar even for vector targets
mean[k]   : np.ndarray[float64]   # (n_classes, d)    ybar
msd[k]    : np.ndarray[float64]   # (n_classes, d)    sigma^2, divisor N, per dimension (§1)
parent[k] : np.ndarray[int32]     # (n_classes,)      index into level k-1; -1 at level 0
```

Class ids are local to a depth and dense from $0$. `parent[k][cid]` indexes directly into the
depth-$(k-1)$ arrays, so ancestor traversal involves no hashing at all.

Only the moment arrays carry the target dimension. Keeping `count` and `parent` one-dimensional is
what lets the merge weights stay scalar (§5.2) and the level-selection logic stay untouched by $d$.

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
same triple globally. Nothing else. The reported variance $s^2$ is **derived on access**. $N$ is a
scalar; $\bar y$ and $\sigma^2$ are $d$-vectors (§1).

**Why these three, and not a mixture.** The layout is chosen for dimensional consistency: $N$ is
extensive (it counts), while $\bar y$ and $\sigma^2$ are both intensive — per-observation means, of
$y$ and of $r^{\odot2}$ respectively. A count and two means.

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

**Accumulation is one mechanism, not a choice between two.** Never
$\sum y^2-(\sum y)^2/n$ (§7.4). Otherwise there is a single rule:

> Reduce a chunk with the two-pass form (§7.3); combine chunks with the merge of §5.2.

Chunk size is a free parameter, fixed by how the data reaches the trainer rather than by the
statistics. The endpoints happen to have names, which is why this is easily mistaken for two
algorithms:

| chunk size | this is called | $\sigma^2$ rel. error |
|---|---|---:|
| whole corpus | the two-pass algorithm | **0** (it is the definition) |
| a shard or batch | chunked / parallel reduction | $\sim10^{-13}$ |
| 1 observation | Welford's recurrence | $7.6\times10^{-12}$ |

**Welford is the $n_B=1$ case of the merge, not an alternative to it.** Substituting $n_B=1$,
$\bar y_B=y$, $\sigma^2_B=0$ into §5.2 gives

$$
\sigma^{2\prime}=\frac{n_A}{n}\sigma^2_A+\frac{n_A}{n^{2}}\delta^{2},
$$

which is exactly what Welford's $M_2'=M_2+\delta(y-\bar y')$ yields after dividing by $n$. Stepping
both through 500 observations agrees to $1.7\times10^{-9}$ absolute on values of magnitude $10^6$ —
rounding noise, not a different algorithm. So the recurrence needs no separate implementation
[Welford1962Variance]; it falls out of the merge already required by §5.

**Practical consequence: chunk as large as memory allows.** Accuracy is flat across chunk sizes and
*best* at the large end, so there is no statistical argument for smaller chunks — only a memory one.
For a corpus that fits in memory as a dataframe plus an atom-indexed array, that means a single
chunk, or a handful of shards purely to parallelise, and the scalar recurrence never appears at all.
It becomes relevant only when data is streamed from disk or arrives as online updates, and even then
the right response is a smaller chunk, not a chunk of one.

**What bounds the chunk.** Two pressures, not one. The obvious is the corpus itself. The less obvious
is that the centring step of §7.3 materialises an $n_{\text{chunk}}\times d$ residual array: 59 MB at
cosmobase scale with $d=50$, but **4 GB at $10^7$ atoms**. Since chunking is already the mechanism
here, this needs no separate machinery — but it does mean the chunk size must be chosen against $d$,
not against atom count alone.

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

It does double duty: besides combining fitted models, this is also the *accumulation* mechanism
(§4.1). Reducing a chunk and merging chunks are the same operation at different granularities, so
there is no separate streaming path to implement.

**Vector targets change nothing structurally.** The weights $w_A,w_B$ stay scalar — support is a
count of nodes, not of components — so the two moment updates are the same expressions broadcast
across $d$, with $\delta^2$ read as the elementwise $\delta^{\odot2}$. Verified against exact
moments: mean to $1.5\times10^{-12}$, $\sigma^2$ to $2.5\times10^{-14}$.

**Upgrading to full covariance is a one-term change.** Replacing the elementwise square with an outer
product,

$$
\Sigma = w_A\Sigma_A + w_B\Sigma_B + w_A w_B\,\delta\delta^{\!\top},
$$

is the same law-of-total-variance identity in matrix form, and nothing else in the merge, the fold,
or the identity element moves. Verified: chunked merging reproduces the exact covariance to
$1.2\times10^{-13}$, and its diagonal agrees with the per-dimension $\sigma^2$ to $8.9\times10^{-16}$.
The cost is $O(d^2)$ storage per class instead of $O(d)$, which is why the diagonal is the default —
but the option is open at any time, and that is a direct consequence of §5.2 being an identity rather
than an ad hoc correction.

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

    m, n_new, d = len(A.vocab), len(vocab), A.mean.shape[1]
    count  = np.zeros(n_new, np.int64);       count[:m]  = A.count
    mean   = np.zeros((n_new, d), np.float64); mean[:m]  = A.mean
    msd    = np.zeros((n_new, d), np.float64); msd[:m]   = A.msd
    parent = np.full(n_new, -1, np.int32);    parent[:m] = A.parent

    i  = remap                     # a bijection, so no duplicate scatter writes
    nA = count[i].astype(np.float64)
    nB = B.count.astype(np.float64)
    n  = nA + nB
    wA, wB = (nA / n)[:, None], (nB / n)[:, None]   # scalar weights, broadcast over d
                                                    # n >= 1 always: no 0/0, no guard

    delta = B.mean - mean[i]       # must precede the mean update below

    msd[i]    = wA * msd[i] + wB * B.msd + wA * wB * delta**2   # elementwise square
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
for k = 0..L:  sparse two-pass -> (N, mean, sigma^2)      (§7.3)
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

Reduce with a **sparse class-membership operator** $P\in\{0,1\}^{n_c\times n}$, built once per level
and reused across both passes and all $d$ dimensions:

```python
P = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(nc, n))

N      = np.bincount(labels, minlength=nc)
S      = P @ Y                                # (nc, d)
mean   = S / N[:, None]
R      = Y - mean[labels]                     # centre first, then reduce
msd    = (P @ (R * R)) / N[:, None]
```

`np.bincount(weights=...)` is scalar-only, so it cannot carry vector targets without a loop over
dimensions. The sparse operator handles both, and is not a compromise at $d=1$ — fair comparison,
identical work:

| $d$ | sparse $P^\top Y$ | `bincount` × $d$ |
|---:|---:|---:|
| 1 | **0.7 ms** | 0.8 ms |
| 4 | 3.7 ms | **3.3 ms** |
| 16 | **13.4 ms** | 17.0 ms |
| 50 | **33.6 ms** | 59.0 ms |
| 128 | **79.5 ms** | 140.1 ms |

Tied at $d=1$, 1.8× faster at σ-profile widths, and one code path rather than a scalar special case.
Building $P$ costs 1.0 ms and is amortised over both passes.

This reduces one chunk. Chunks combine through §5.2 — see §4.1: chunk size is a memory decision, and
the scalar recurrence is simply the chunk-of-one endpoint, not a different code path. For scale, at
$d=1$ the chunk-of-one extreme (a scalar Welford loop) costs 109.5 ms against 0.7 ms here: identical
statistics, 150× the time.

### 7.4 Two traps

**`reduceat` as the segment reducer.** It handles `axis=0` and so looks like the natural fit for
vector targets, but it requires the data sorted by group — an $O(n\log n)$ argsort the sparse
operator avoids — and it corrupts empty groups silently. When `starts[i] >= starts[i+1]` it returns
`y[starts[i]]` rather than $0$:

```text
labels [0 0 3 3]   y [10 20 30 40]      # classes 1 and 2 empty
correct  : [30.  0.  0. 70.]
reduceat : [30. 30. 30. 70.]            <- classes 1,2 silently wrong
```

Empty groups cannot occur while ids are interned per shard, but they appear the moment a shard is
indexed against a merged global vocabulary — which is exactly what §5 does. It is also slower than
$P$ at every width measured (52.4 ms against 11.4 ms for the segment sum at $d=50$). No reason to
reach for it.

**Power sums instead of centring.** The one-pass form $Q/N-\bar y^{\odot2}$ with $Q=\sum y^{\odot2}$
is the obvious vectorisation and is unusable, as §4.1 argues on principle and this measures in
practice. On targets with mean $10^6$ and spread $3$:

| | max rel. error | negative variances |
|---|---:|---:|
| two-pass, centred | 5.4e-08 | 0 |
| power sums | **1.3e+02** | **2** |

Centring costs one extra pass and one $n\times d$ temporary (§4.1). The fast-looking formula and the
correct formula are not the same formula.

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

## 9. Serialisation

**Everything the model holds is already an array**, including the vocabulary. §7.2 mints class ids by
deduplicating signature rows, and the deduped rows *are* the vocabulary: row $i$ of `vocab[k]` is the
signature of class $i$ at level $k$. There is no dict to encode, and no digest to store.

This settles what was an open question: **`vocab` is stored as an `(n_classes, width)` integer array**,
not `dict[bytes, int]`. Storage is compact, serialisation is trivial, and at load time either form can
be rebuilt — a dict for $O(1)$ lookup, or a lexsorted view for binary search at lower memory.

### 9.1 Format

A single `.npz`, since every member is an array:

```text
config              JSON bytes: see below
global              (2 + 2d,)  N, then mean and sigma^2 of the whole training set
level_{k}_vocab     (n_k, w_k)  int64   signature rows; row i is class i
level_{k}_count     (n_k,)      int64
level_{k}_mean      (n_k, d)    float64
level_{k}_msd       (n_k, d)    float64
level_{k}_parent    (n_k,)      int32   index into level k-1; -1 at level 0
```

Shrunk estimates are **not** stored — they are derived (§4.2), and storing them would both bloat the
file and let $\alpha$ drift out of sync with the values it produced.

### 9.2 Config

```json
{
  "format_version": 1,
  "schema_version": "<hash of the fields below>",
  "target_dim": 50,
  "attribute_levels": [["element"], ["aromatic","hybridization"], ["formal_charge"]],
  "edge_attributes": ["bond_type"],
  "neighbour_schema": null,
  "max_wl_depth": 3,
  "n_levels": 6,
  "n_min": 5,
  "alpha": 2.0
}
```

Two version fields, doing different jobs:

- **`format_version`** describes the file layout. A reader that does not recognise it must refuse to
  load, not guess.
- **`schema_version`** is a digest over everything that affects what a class *means* — the attribute
  levels and their order, edge attributes, neighbour schema, depth. Two models may be merged (§5.4)
  only if their `schema_version` matches. This is the mechanism behind "reject incompatible configs
  loudly"; without it, config drift silently produces a model whose classes mean two different things.

`n_min` and `alpha` are inference-time parameters, not part of `schema_version` — changing them does
not invalidate the fitted statistics, and two models differing only in $\alpha$ are still mergeable.

### 9.3 Round-trip guarantee

Save/load must reproduce predictions **bit-exactly**. Nothing here is lossy: ids are `int64`, moments
are `float64`, and no floating-point recomputation happens on load. Derived quantities ($s^2$,
shrunk means) are recomputed identically from identical inputs. This is a testable property, not an
aspiration — see the round-trip requirement in §10.4.

---

## 10. Public API

### 10.1 The core is functional

A fitted model is immutable (§5.1), so the core does not use the fit-mutates-self convention:

```python
model = sieve.fit(batch, config)             # -> SieveModel, immutable
values = model.predict(batch)                # -> (n_atoms, d)
detail = model.predict_detailed(batch)       # -> Predictions (§12)
merged = model_a.merge(model_b)              # or model_a + model_b
model.save(path);  SieveModel.load(path)
```

```python
@dataclass(frozen=True)
class SieveConfig:
    target_dim: int
    attribute_levels: tuple[tuple[str, ...], ...]   # graded order, §3.5
    max_wl_depth: int
    edge_attributes: tuple[str, ...] = ("bond_type",)
    neighbour_schema: tuple[str, ...] | None = None # §3.6; None = same as centre
    n_min: int = 1
    alpha: float | None = None                      # None = raw means, §4.2
    chunk_size: int | None = None                   # §4.1; None = whole corpus
```

`alpha` and `n_min` are read at prediction time from the model's config, so sweeping them does not
require refitting — a `with_params()` returning a new model sharing the same arrays is the cheap way
to expose that.

### 10.2 A scikit-learn wrapper, not a scikit-learn core

Contorting the immutable core into `get_params`/`set_params`/`fit(self)` would forfeit the merge
monoid for the sake of an interface. Provide instead a thin adapter that wraps the functional core, so
`GridSearchCV` and friends work for the $\alpha$ / $n_{\min}$ / $K$ sweeps without the core inheriting
mutable-estimator semantics.

The adapter must default to **graph-level** splitting. Node-level random splitting puts WL-identical
atoms from one molecule on both sides and inflates scores badly; this is the single easiest way to
produce a misleading number with this method, so the safe behaviour belongs in the default rather than
in the documentation.

### 10.3 Leave-one-out prediction

```python
model.predict_loo(batch) -> Predictions
```

A training node contributes its own label to its class mean, so any in-sample score is meaningless —
at $n_{\min}=1$ and large $L$ it approaches perfect recall. Leave-one-out removes the node's own
contribution before predicting:

$$
\bar y^{(-v)}_{k,c}=\frac{N_{k,c}\,\bar y_{k,c}-y_v}{N_{k,c}-1},
$$

with the class treated as **unsupported** when $N_{k,c}=1$, so backoff proceeds to the parent rather
than dividing by zero. This is the standard remedy from the target-encoding literature
[MicciBarreca2001HighCardinality], and it is also the cheapest test that the implementation is not
leaking — which is why it is a first-class method rather than a notebook recipe.

### 10.4 Behaviours that must hold

These are the properties an implementation has to satisfy; they follow from §2 and §5 and are the
natural test suite.

| Property | Source |
|---|---|
| Same partition regardless of node ordering within a graph | §7.2 |
| Isomorphic graphs give matching class multisets at every level | §2.1 |
| Every class has exactly one parent; $N_{k,c}\le N_{k-1,p(c)}$ | §2.1, §2.3 |
| Matched levels form a prefix — no gaps are constructible | §2.2 |
| Deepest supported level wins; $n_{\min}$ shifts it shallower, never deeper | §6.3 |
| Global fallback iff level 0 is unsupported | §2.2 |
| `merge` is associative, commutative, with the empty model as identity | §5.4 |
| `merge` of disjoint shards equals fitting their union | §5.2 |
| $\alpha=0$ reproduces raw means; $\alpha\to\infty$ reproduces the global mean | §4.2 |
| Save/load reproduces predictions bit-exactly | §9.3 |
| Changing only target values leaves every class id unchanged | §7.2 |
| Batched and per-node prediction agree | §6.1 |
| `predict_loo` on a class of size 2 returns the other member's value | §10.3 |
| Two 1-WL-indistinguishable graphs *do* collide | accepted limit |

The last is a negative control: it pins the known 1-WL expressiveness bound as intended behaviour
rather than an undetected bug.

---

## 11. Input contract

### 11.1 The batch

The core is graph-library agnostic. Everything upstream reduces to one columnar structure, which is
also exactly what §7.1 consumes:

```python
@dataclass(frozen=True)
class AtomBatch:
    node_attrs: np.ndarray     # (n_atoms, n_attr)  int64, encoded categoricals
    edge_src:   np.ndarray     # (n_edges,)   int64, CSR-sorted
    edge_dst:   np.ndarray     # (n_edges,)   int64
    edge_attrs: np.ndarray     # (n_edges, n_edge_attr) int64
    graph_id:   np.ndarray     # (n_atoms,)   int64
    y:          np.ndarray | None   # (n_atoms, d) float64
```

Edges are stored **both directions** for undirected graphs; §7.1's CSR construction assumes it.

`graph_id` is not optional. It is what makes graph-level splitting possible (§10.2), and a batch that
loses it cannot be validated correctly no matter what the splitter does.

### 11.2 Adapters

```python
sieve.io.from_rdkit(mols, y=None, *, config)      -> AtomBatch
sieve.io.from_smiles(smiles, y=None, *, config)   -> AtomBatch
```

The adapter owns attribute encoding: it maps each configured attribute name to a dense integer code
and stores the mapping, so that an unseen category at inference produces a *reserved unknown code*
rather than a silent collision with a seen one. An unknown code then simply fails to match at level 0
and backs off, which is the correct behaviour.

### 11.3 Alignment is checked, not assumed

For the dataframe-plus-atom-array layout this is the highest-severity failure mode (§7.5): a
misalignment corrupts every label and raises nothing. The adapter must verify, not trust:

1. the number of atoms parsed from each molecule equals the length of its target slice;
2. atomic numbers stored alongside the targets match the parsed molecule, element by element.

The second check is what actually catches reordering — counts alone will not, since a permutation
preserves them. It costs one integer comparison per atom and rules out the only bug in this system
that presents purely as unexplained inaccuracy.

`MolFromSmiles` preserves the input SMILES heavy-atom order, but `AddHs` appends hydrogens at the end
and any canonicalisation round-trip reorders, so the guard must run after whatever preprocessing the
pipeline applies, not before.

### 11.4 The target contract

`y` is `(n_atoms, d)`. The core requires only that **component $j$ means the same thing in every
row** — $d$ is fixed across the corpus and the components are aligned. Everything below follows from
that, and the adapter, not the core, is responsible for establishing it.

For the σ-profile application the contract is specific:

- the σ grid is **fixed and shared by every molecule** — same bin edges, same $d$, same order;
- values are **areas**, in the profile's native area units;
- they are **not normalised** — no division by total area, no conversion to a probability density.

Two consequences are worth stating, because they are what make the estimator well behaved here rather
than merely well typed.

**A shared grid is what makes componentwise averaging meaningful.** Component $j$ of a class mean is
the mean of $N$ quantities that denote the same thing, so §7.3's elementwise reduction is a statement
about a physical quantity rather than about array positions. Were grids to differ per molecule, the
elementwise mean would be meaningless, and resampling onto a common grid would be required *upstream*
of `AtomBatch`. The core will not detect a violation: mismatched grids produce plausible numbers and
no error, which puts this in the same hazard class as §11.3's misalignment.

**The estimator is closed over non-negative vectors.** Areas are $\ge 0$; backoff returns one stored
class mean, and shrinkage (§4.2) returns a convex combination of a class mean and its already-shrunk
parent. Every prediction is therefore a convex combination of training rows, so:

- predictions are non-negative automatically — no clipping, no constrained solve, no post-hoc
  correction, and any clipping that does appear in the code is evidence of a bug elsewhere;
- each component lies within the range of the training values for its class chain, so a prediction
  cannot exceed the largest observed area in any bin;
- the predicted total area, being a linear functional of the profile, is the same convex combination
  of the training totals and so is likewise bounded by them.

The last point is a bound, not a constraint: atoms are predicted independently, so summing predicted
per-atom profiles gives an estimate of the molecular surface area, not a guaranteed match to it. If a
hard per-molecule total is ever required it must be imposed downstream by rescaling, which preserves
non-negativity (a positive scale factor) but voids the convex-combination bounds above.

**Targets must not be centred or standardised.** Subtracting a per-component mean would destroy both
the non-negativity closure and the additivity of areas, and would buy nothing: the estimator is a
conditional mean, which is equivariant under such a shift anyway.

---

## 12. Prediction metadata

`predict_detailed` returns a columnar struct, matching the rest of the design rather than a list of
per-node dicts:

```python
@dataclass(frozen=True)
class Predictions:
    value:            np.ndarray   # (n, d)  the prediction
    matched_level:    np.ndarray   # (n,)    k*, or -1 for global fallback
    class_id:         np.ndarray   # (n,)    id at the matched level, -1 if none
    support:          np.ndarray   # (n,)    N at the matched class
    variance:         np.ndarray   # (n, d)  s^2, NaN where support == 1
    threshold_bound:  np.ndarray   # (n,)    bool: stopped by n_min, not by OOV
    # present only when alpha is not None
    raw_value:        np.ndarray   # (n, d)  before shrinkage
    shrinkage_weight: np.ndarray   # (n,)    N / (N + alpha)
```

`threshold_bound` exists because a shallow match caused by $n_{\min}$ and one caused by genuine OOV
are different situations with different remedies — more data versus better coverage — and without the
flag they are indistinguishable in every downstream analysis.

`variance` carries `NaN` rather than `0.0` at $N=1$, for the reason given in §4.1: a zero is
indistinguishable from a genuinely homogeneous class. `NaN` propagates instead of silently reading as
confidence.

**These are diagnostics, not calibrated uncertainties.** The triple
$(k^\star,\,N,\,s^2)$ says how specific the matched environment was, how much training support it had,
and how heterogeneous its labels were. That is genuinely informative for triage, and it is not a
predictive interval — anything claiming to be one needs conformal prediction or the machinery of
[JonasKuhn2019Uncertainty], neither of which is in scope. Any write-up must say so explicitly.

A high global-fallback rate is a *featurisation* alarm, not a prediction: it means atoms are arriving
whose level-0 attributes were never seen. Surface it prominently rather than burying it in a column.

---

## 13. Open questions

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
7. **Whether to store the full class covariance** rather than its diagonal (§5.2). The merge upgrades
   by one term and is already verified; the cost is $O(d^2)$ per class instead of $O(d)$. Deferred
   until there is a use for the off-diagonal structure — correlated σ-profile bins would be the
   obvious one. Whether $\alpha$ (§4.2) should then become per-dimension is a separate question, and
   should stay scalar without evidence.
8. **Whether neighbours should carry a coarser attribute schema than the centre (§3.6).** Measured on
   cosmobase: coarsening buys ~0.6 levels of extra reach at identical support. What remains open is
   whether that reach is *worth* the attribute resolution it costs, which needs targets. Run the
   comparison again with real σ-profiles or partial charges and decide on MAE at the matched class.
   Implement as a configurable neighbour schema — an ablation flag, not an architecture. This
   comparison is part of the planned ablation suite rather than a design decision to be made in
   advance; the entry stays open until that suite runs.

---

## 14. Deliberately not settled here

Statistical and experimental questions — evaluation protocol, baselines, splitting strategy, leakage
controls, the relationship to message-passing GNNs, and the novelty argument — are untouched by this
document. `wllr.md` holds draft material on all of them, none of it confirmed.
