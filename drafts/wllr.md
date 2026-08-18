# Weisfeiler–Lehman Lookup Regression (WLLR)

## Superseded draft — method definition and implementation notes

> ## ⚠ SUPERSEDED — DO NOT IMPLEMENT FROM THIS FILE
>
> This is brainstorming output from another LLM, retained only as a record of where the
> method started.
>
> **The implementation specification is `design.md`.** This document contradicts it in
> several places — the rejected `(depth, hash)` layout, scalar-only targets, a Welford
> instruction that no longer applies, and a test list written against the old design.
> Where the two disagree, `design.md` is correct.
>
> **The literature review and novelty assessment are `literature.md`.** They were moved
> out of this file so there is exactly one copy to maintain. Nothing bibliographic
> remains here.

**Status:** superseded draft, retained for provenance
**Date:** 2026-08-17
**Supersedes:** `wllr_implementation_spec.md`, `wllr_extended_literature_search.md`

---

# Part 0 — Reading guide

| Part | Contents |
|---|---|
| I (§1–§6) | Problem statement, WL environment construction, estimator, backoff, regularization |
| II (§7–§14) | Object model, APIs, validation rules, tests, performance, GNN relationship |
| III (§15) | Review notes and open decisions carried over from consolidation |

Literature, precedents, novelty boundary, related-work structure, and the certified
references were moved to `literature.md`.

Passages marked **▸ Review note** were added or changed during consolidation and record a substantive technical judgment rather than a restatement of the original drafts. §15 lists them in one place.

---

# Part I — Method

# 1. Scope

This document specifies a node-level regression method based on Weisfeiler–Lehman (WL) refinement, together with an object-oriented Python implementation.

The core estimator:

1. assigns a WL identifier to every node at depths $0,\ldots,K$;
2. groups labeled training nodes by `(depth, WL identifier)`;
3. stores the empirical target mean and supporting statistics for each class;
4. predicts from the most specific *adequately supported* WL class observed during training;
5. recursively backs off to lower WL depths when the requested environment is out of vocabulary (OOV) or under-supported;
6. ultimately falls back to the global training-set mean.

A regularized variant shrinks low-support class means toward their parent WL environment while preserving the same OOV logic.

The working name is **Weisfeiler–Lehman Lookup Regression (WLLR)**. The regularized variant is **hierarchically shrunk WLLR**. These are descriptive names, not established terminology.

---

# 2. Problem definition

Let the training dataset be

$$
\mathcal{D}=\{G_i\}_{i=1}^{M},\qquad G_i=(V_i,E_i).
$$

Each labeled node $v\in V_i$ has a scalar target $y_v\in\mathbb{R}$.

For each node, define a sequence of WL identifiers

$$
h_0(v),h_1(v),\ldots,h_K(v),
$$

where $h_k(v)$ is the identifier after WL refinement depth $k$.

The fitted model stores target statistics indexed by `(depth, identifier)`.

---

# 3. WL environment construction

## 3.1 Initial node representation

At depth zero,

$$
h_0(v)=H(x_v),
$$

where $x_v$ is a deterministic serialization of the selected initial node attributes.

For molecular graphs, candidate features include atomic number or element, formal charge, aromaticity, hybridization, chirality, hydrogen count, and other discrete atom attributes.

The exact feature schema is part of the model definition and must be serialized with the fitted estimator.

## 3.2 Recursive refinement

For $k\ge1$,

$$
h_k(v)
=
H\!\left(
h_{k-1}(v),\;
\operatorname{MULTISET}
\left\{
\phi\!\left(h_{k-1}(u),e_{uv}\right):
u\in N(v)
\right\}
\right),
$$

where $N(v)$ is the neighborhood of $v$; $e_{uv}$ contains optional edge attributes; $\phi$ deterministically serializes neighbor and edge state; `MULTISET` is represented canonically (sorted with multiplicity); and $H$ maps the canonical serialized state to a reproducible identifier.

## 3.3 Nesting property and the parent relation

Because $h_k(v)$ is computed *from* $h_{k-1}(v)$, the identifier at depth $k$ determines the identifier at depth $k-1$ (up to hash collisions, §13.4). Consequently the induced vertex partitions are nested,

$$
\Pi_0\succeq\Pi_1\succeq\cdots\succeq\Pi_K,
$$

and for any node the associated classes form a chain

$$
C_0(v)\supseteq C_1(v)\supseteq\cdots\supseteq C_K(v).
$$

This gives the parent map $p(c)$ used in §6 and is the formal basis for calling the backoff sequence an *ancestor* chain rather than an ad hoc list of fallback hashes [Kriege2016WLOA].

**▸ Review note (parent map).** The parent must be *recorded at fit time* from the node that produced the class, not recovered by inverting $H$. The implementation stores `parent_hash` on each class record precisely so no inversion is required, and must assert that all nodes in a depth-$k$ class agree on their depth-$(k-1)$ parent — a disagreement is a hash-collision alarm.

## 3.4 Determinism requirements

The implementation must not rely on Python's process-randomized built-in `hash()`.

WL identifiers must remain identical across repeated calls, across Python processes, across machines, after model serialization/deserialization, and between training and inference.

Acceptable approaches:

- a stable cryptographic hash (BLAKE2b or SHA-256) of canonical bytes, optionally truncated (§13.4);
- a deterministic vocabulary mapping constructed from canonical strings.

All configuration affecting hashing must be stored in the fitted object, including a schema-version tag so that identifiers minted by different library versions are never silently mixed.

## 3.5 Dataset-wide consistency

Two nodes must receive the same identifier at depth $k$ if and only if their serialized WL states are identical under the configured feature schema and refinement rules.

Configuration must explicitly cover: maximum WL depth; node feature keys; edge feature keys; sorting/canonicalization rules; directed vs undirected graphs; self-loop handling; missing-value handling; stereochemistry if relevant; and the hash function or vocabulary mapping.

## 3.6 Choice of $K$ and refinement stabilization

WL color refinement reaches a stable partition after at most $|V|-1$ rounds, and in practice far sooner. Once $\Pi_k=\Pi_{k+1}$ as a partition, further depth adds no discriminative information — but the *identifiers* still change, because $H$ is applied again to different inputs.

**▸ Review note (stabilization).** The original drafts did not distinguish these. Two consequences:

1. Depths beyond stabilization inflate the vocabulary and the model size without adding information. The encoder should optionally detect partition stabilization on the training set and record the effective depth $K_{\text{eff}}$.
2. Stabilization is dataset-dependent. It must be computed on the training fold only, or it becomes a (mild) leakage channel.

In practice, for molecular graphs, useful $K$ is small (typically 1–4); beyond that, classes become singletons long before the partition stabilizes globally.

---

# 4. The estimator over WL classes

For each WL depth $k$ and identifier $c$, define

$$
C_{k,c}=\{v:h_k(v)=c\},\qquad
N_{k,c}=|C_{k,c}|,\qquad
\bar y_{k,c}=\frac{1}{N_{k,c}}\sum_{v\in C_{k,c}}y_v,
$$

together with the sample variance $s^2_{k,c}$.

The unregularized stored estimate is $\hat\mu_{k,c}=\bar y_{k,c}$.

Under squared-error loss,

$$
\bar y_{k,c}=\arg\min_a\sum_{v\in C_{k,c}}(y_v-a)^2,
$$

so at fixed depth the estimator is the squared-error-optimal piecewise-constant predictor over WL equivalence classes. Across depths it is best described as a **hierarchical regressogram over the nested vertex partitions induced by WL color refinement** (`literature.md` §3).

**Variance convention.** $s^2_{k,c}$ is the unbiased sample variance with $\mathrm{ddof}=1$ and is **undefined for $N_{k,c}=1$**. Store `None` in that case rather than `0.0`; a stored `0.0` is indistinguishable from a genuinely homogeneous class and silently corrupts any downstream diagnostic that treats variance as a confidence proxy.

---

# 5. Hierarchical OOV backoff

The backoff rule is core behavior and applies to both the raw and the regularized model.

For a query node $v$, compute $h_0(v),\ldots,h_K(v)$ and search from the most specific environment downward:

$$
h_K(v)\rightarrow h_{K-1}(v)\rightarrow\cdots\rightarrow h_0(v)\rightarrow \mu_{\mathrm{global}}.
$$

## 5.1 Matched depths form a prefix

**Lemma.** Let $D_k$ be the set of depth-$k$ identifiers seen during fitting. If $h_k(v)\notin D_k$ then $h_{k+1}(v)\notin D_{k+1}$.

*Proof.* If $h_{k+1}(v)\in D_{k+1}$, some training node $u$ has $h_{k+1}(u)=h_{k+1}(v)$. Since the depth-$(k+1)$ identifier determines the depth-$k$ identifier (§3.3), $h_k(u)=h_k(v)$, hence $h_k(v)\in D_k$. ∎

**▸ Review note (why this matters).** The set of supported depths along a query node's chain is therefore *downward closed*: it is $\{0,1,\ldots,k^\star\}$ or empty. Three consequences the original spec missed:

- $k^\star(v)=\max\{k:h_k(v)\in D_k\}$ is not a maximum over a scattered set; it is the last index before the first miss. The linear scan in §11.2 is correct but the search can be a **binary search over depth** ($O(\log K)$ lookups) when $K$ is large.
- Test 12.6 ("multi-level fallback") cannot be constructed by making depth $K$ and $K-2$ present while $K-1$ is absent — that state is unreachable. The test must instead exercise a query whose first miss occurs at depth $j<K$.
- Global fallback is genuinely reachable only when $h_0(v)\notin D_0$, i.e. an unseen depth-0 feature vector (a new element, an unseen charge state). This is the correct and only trigger, and it should be surfaced loudly in metadata, because it usually indicates a featurization or domain-coverage problem rather than a routine prediction.

## 5.2 Prediction rule

If a match exists,

$$
\hat y(v)=\hat\mu_{k^\star,\,h_{k^\star}(v)};
\qquad\text{otherwise}\qquad
\hat y(v)=\mu_{\mathrm{global}}=\frac{1}{N}\sum_v y_v .
$$

## 5.3 Support-thresholded depth selection

**▸ Review note (this is the main methodological concern).** Pure deepest-match-first is a poor default and should not be the library's out-of-the-box behavior.

As $k$ grows, WL classes fragment toward singletons. For a node whose deepest class has $N_{K,c}=1$, "prediction" is the recall of a single training label, with undefined variance and no averaging. Deepest-match-first therefore *maximizes* estimator variance exactly where it looks most confident, and the reported $(k^\star,N,s)$ diagnostic will show $N=1,\,s=\text{None}$ on a large fraction of predictions. This is the classic sparse-context failure that language-model backoff addresses with discounting rather than with raw most-specific counts [Katz1987Backoff].

Two remedies, both of which should be available:

**(a) Minimum-support depth selection.** Replace $k^\star$ with

$$
k^\star_{n_{\min}}(v)=\max\{k: h_k(v)\in D_k \ \text{and}\ N_{k,h_k(v)}\ge n_{\min}\}.
$$

By the Lemma, $h_k(v)\in D_k$ holds on a prefix; and since $C_k(v)\subseteq C_{k-1}(v)$, the counts $N_{k,h_k(v)}$ are **non-increasing in $k$**, so the support condition also holds on a prefix. Hence $k^\star_{n_{\min}}$ is again the last index before the first failure, and the same scan or binary search applies. $n_{\min}=1$ recovers the original rule.

**(b) Shrinkage (§6),** which handles low support continuously rather than by a hard cut, and which for this reason is the recommended default for reported results.

**Backoff and shrinkage remain separate mechanisms.** Backoff handles **zero** observations for an environment; shrinkage handles **few**. Support thresholding is a third, intermediate device that reuses the backoff machinery to address the "few" case discretely.

---

# 6. Hierarchically shrunk WLLR

Shrink each WL-class estimate toward its parent environment:

$$
\tilde\mu_{k,c}
=
\frac{N_{k,c}\,\bar y_{k,c}+\alpha_k\,\tilde\mu_{k-1,p(c)}}{N_{k,c}+\alpha_k},
\qquad
\tilde\mu_{0,c}
=
\frac{N_{0,c}\,\bar y_{0,c}+\alpha_0\,\mu_{\mathrm{global}}}{N_{0,c}+\alpha_0}.
$$

The inference-time backoff procedure is unchanged; only the value stored for a known class differs.

## 6.1 Properties and required implementation details

- The recursion uses the **already-shrunk** parent $\tilde\mu_{k-1,p(c)}$, not the raw parent mean $\bar y_{k-1,p(c)}$. This matches the hierarchical-shrinkage formulation of Agarwal et al. [Agarwal2022HierarchicalShrinkage] and makes the estimate a support-weighted average along the whole ancestor chain down to $\mu_{\mathrm{global}}$.
- Therefore **shrunk estimates must be computed top-down**, depth $0$ first. §11.1's pseudocode makes this explicit.
- $\alpha_k=0$ recovers raw class means; $\alpha_k\to\infty$ collapses depth $k$ onto its parent. A single shared $\alpha$ is the sensible starting point; per-depth $\alpha_k$ is a refinement to justify empirically, not by default.
- The effective shrinkage weight on the class's own data, $N_{k,c}/(N_{k,c}+\alpha_k)$, should be stored and returned in metadata — it is the single most informative regularization diagnostic.

**▸ Review note (tuning).** Neither draft specified how $\alpha$ is chosen. It must be tuned by **graph-level** cross-validation inside the training fold (§12), never on the evaluation split, and the selected value reported. Without this the regularized variant is not reproducible.

## 6.2 Swappable estimator

The regularization strategy must be a strategy object, not a subclass of the regressor, so that the backoff policy stays independent of how known-class values are estimated (§9.4).

---

# Part II — Implementation

# 7. Object model overview

Separate three concerns:

1. graph hashing / representation (`WLEncoder`);
2. target-statistics estimation (`WLStatistics` + a `MeanEstimator` strategy);
3. lookup and fallback policy (`WLLookupRegressor`).

## 7.1 `WLEncoder`

Responsibilities: validate graph inputs; extract configured node/edge features; generate $h_0,\ldots,h_K$; guarantee deterministic canonicalization; expose configuration; serialize and deserialize exactly.

```python
fit(self, graphs=None)
transform(self, graphs)
fit_transform(self, graphs)
encode_node(self, graph, node)
get_config(self)
```

`fit()` is unnecessary if the encoder uses only deterministic hashing. If a vocabulary mapping is learned, `fit()` becomes required — and, being learned from the training fold, becomes subject to the leakage rules of §12.

## 7.2 `WLStatistics`

Responsibilities: aggregate training targets by `(depth, identifier)`; store count, mean, variance; store global target statistics; retain parent relationships; expose lookup; serialize.

```python
WLClassStats(
    depth: int,
    hash: str,
    count: int,
    mean: float,
    variance: float | None,   # None when count == 1 (see §4)
    parent_hash: str | None,  # None only at depth 0
)
```

## 7.3 `WLLookupRegressor`

Responsibilities: orchestrate encoder, statistics, and estimator; `fit(graphs, y)`; `predict(graphs)`; `predict_with_metadata(graphs)`; implement depth selection (§5.2/§5.3); implement recursive backoff; expose fitted dictionaries.

A scikit-learn-compatible interface (`get_params`/`set_params`, `fit`/`predict`) is preferable where practical, since it makes the cross-validation machinery of §12 available for free — including the $\alpha$ search.

## 7.4 Regularization strategy

```python
class MeanEstimator(Protocol):
    def fit_level(...): ...
    def estimate(...): ...
```

Implementations: `RawMeanEstimator`, `HierarchicalShrinkageEstimator`.

---

# 8. Prediction metadata

`predict_with_metadata` returns a structured record per node:

```text
prediction
requested_depth
matched_depth              # k*
hash
count                      # N at matched depth
mean
variance                   # None when count == 1
used_global_fallback
was_oov_at_requested_depth
support_threshold_binding  # matched depth limited by n_min, not by OOV
```

For regularized models also:

```text
raw_mean
regularized_mean
parent_hash
shrinkage_weight           # N / (N + alpha_k)
```

The tuple $(k^\star,\,N_{k^\star,c},\,s_{k^\star,c})$ is the primary diagnostic: matched specificity, training support, and label heterogeneity within the matched environment. These are **diagnostics, not calibrated predictive uncertainties**, and must be described as such in any write-up. Calibrated intervals would require the machinery of [JonasKuhn2019Uncertainty] or conformal prediction, neither of which is in scope here.

`support_threshold_binding` is added so that a shallow match caused by §5.3's threshold is distinguishable from a shallow match caused by genuine OOV. Without it the two are conflated in every downstream analysis.

---

# 9. Data API considerations

Do not couple the estimator to one graph package.

Possible adapters: NetworkX; RDKit molecular graphs; PyTorch Geometric; DGL; a small internal graph protocol.

Minimal internal protocol:

```python
nodes(graph)
neighbors(graph, node)
node_features(graph, node)
edge_features(graph, u, v)
```

For a chemistry-focused first implementation an RDKit adapter is the most practical, but core statistics and backoff logic must remain graph-library agnostic. (Note that with a standard atom-invariant schema the RDKit path reproduces Morgan/ECFP atom-environment identifiers almost exactly — see `literature.md` §4.6.)

---

# 10. Fit/predict behavior

## 10.1 Training pseudocode

```text
input:
    labeled training graphs
    node-level targets
    maximum WL depth K
    optional: alpha (shrinkage), n_min (support threshold)

compute global target statistics (mu_global, N, variance)

for each training node:
    compute h_0, ..., h_K

for depth k = 0, ..., K:                 # ascending order is required
    group nodes by h_k
    for every identifier:
        store count, mean, variance
        store parent hash when k > 0
        assert parent is unique within the class   # collision alarm

if shrinkage is enabled:
    for depth k = 0, ..., K:             # top-down; depth k needs shrunk depth k-1
        mu_tilde[k, c] = (N*mean + alpha_k * parent_mu_tilde) / (N + alpha_k)
        store shrinkage_weight = N / (N + alpha_k)

return fitted estimator
```

## 10.2 Inference pseudocode

```text
for every query node:
    compute h_0, ..., h_K

    for k = K down to 0:                 # or binary search; matched depths form a prefix
        if h_k in fitted statistics and count[k, h_k] >= n_min:
            return stored estimate at h_k, with metadata

    return global training mean, flagged used_global_fallback
```

## 10.3 Leave-one-out in-sample evaluation

**▸ Review note (added).** A training node contributes its own label to its class mean, so any in-sample score is meaninglessly optimistic — at $n_{\min}=1$ and large $K$ it approaches perfect recall. Any in-sample diagnostic must therefore use leave-one-out class means,

$$
\bar y^{(-v)}_{k,c}=\frac{N_{k,c}\bar y_{k,c}-y_v}{N_{k,c}-1},
$$

with the class treated as unsupported when $N_{k,c}=1$ (so backoff proceeds to the parent). This is the standard remedy in the target-encoding literature [MicciBarreca2001HighCardinality] and should be a first-class method, `predict_loo(...)`, not a notebook workaround. It is also the cheapest sanity check that the implementation is not leaking.

---

# 11. Validation and leakage rules

Because the dictionaries contain target statistics, they are learned model parameters.

## 11.1 No target leakage

For every cross-validation fold: compute WL target statistics on the training fold only; predict held-out graphs using only those fitted statistics. Never build target dictionaries from the complete dataset before cross-validation. The same applies to any learned vocabulary mapping (§7.1), to stabilization-depth detection (§3.6), and to the $\alpha$ search (§6.1), which requires a nested split.

## 11.2 Split unit

When nodes from one graph are statistically dependent, validation must split at the **graph level**, not randomly at the node level. Node-level random splitting places WL-identical atoms from the same molecule on both sides of the split and inflates scores severely — this is the single most likely way to produce a misleading result with this method.

For molecular applications, scaffold-aware or chemistry-aware splitting may also be required depending on the intended generalization regime.

## 11.3 Duplicates

Duplicate graphs or repeated local environments change class counts directly. The implementation must expose support counts so the deduplication policy can be inspected and documented.

## 11.4 Evaluation protocol

**▸ Review note (added).** Neither draft specified an evaluation protocol; a manuscript needs one fixed in advance:

- **Metrics:** MAE and RMSE overall, plus both broken down by $k^\star$ and by $\log_{10} N_{k^\star}$ — the depth/support breakdown is the informative result for a method of this kind, and is what distinguishes it from a black-box baseline.
- **Coverage statistics:** fraction of test nodes matched at each depth, and fraction hitting global fallback.
- **Baselines:** (i) global mean; (ii) depth-0 lookup (element-type mean); (iii) a fixed-depth lookup with no backoff, to isolate the contribution of backoff; (iv) a GNN with message-passing depth matched to $K$ (§14).
- **Ablations:** $n_{\min}$ sweep; $\alpha$ sweep; $K$ sweep; raw vs shrunk.

---

# 12. Required tests

**12.1 Determinism.** Identical inputs and configuration produce identical identifiers across runs, processes, and after serialization round-trip.

**12.2 Permutation invariance.** Relabeling node indices without changing structure or attributes does not alter the identifier of the corresponding node.

**12.3 Isomorphism invariance.** Two isomorphic graphs yield matching identifier multisets at every depth.

**12.4 1-WL limitation is documented, not accidental.** A negative-control test on a known 1-WL-indistinguishable pair (e.g. two non-isomorphic regular graphs of the same degree) asserts that the identifiers *do* collide. This pins the known expressiveness bound [Shervashidze2011WL; Morris2019WLGoNeural] as accepted behavior rather than an undetected bug.

**12.5 Correct statistics.** On a hand-constructed example, counts, means, and variances match manual calculation; variance is `None` at $N=1$.

**12.6 Nesting.** Every depth-$k$ class has exactly one depth-$(k-1)$ parent, and $N_{k,c}\le N_{k-1,p(c)}$ for all classes.

**12.7 Exact-match precedence.** A supported hit at depth $K$ takes precedence over all shallower levels.

**12.8 Single-level fallback.** If depth $K$ is OOV and $K-1$ is present, prediction uses $K-1$.

**12.9 Multi-level fallback.** Lookup recurses to the deepest supported level. Per §5.1, construct this with a first miss at depth $j<K$; do **not** attempt to construct a "gap" (supported at $K$, unsupported at $K-1$) — it is unreachable, and a test that appears to produce one is evidence of a collision or a parent-tracking bug.

**12.10 Global fallback.** If depth 0 is OOV, prediction equals the fitted global mean and `used_global_fallback` is set.

**12.11 Support threshold.** With $n_{\min}>1$, a class of count $<n_{\min}$ is skipped and `support_threshold_binding` is set.

**12.12 Shrinkage limits.** $\alpha=0$ reproduces `RawMeanEstimator` exactly; $\alpha\to\infty$ at every depth reproduces $\mu_{\mathrm{global}}$ for every node.

**12.13 Serialization.** Save/restore preserves predictions, metadata, identifiers, configuration, and statistics.

**12.14 No target dependence in hashing.** Changing only target values does not alter any WL identifier.

**12.15 Feature-schema sensitivity.** Changing a configured node or edge feature changes identifiers where it should, and does not where it should not.

**12.16 Batched prediction equivalence.** Node-by-node and batched prediction produce identical results.

**12.17 Leakage guard.** `predict_loo` on a class of count 2 returns the *other* member's label, and a full-dataset in-sample score is strictly worse than the naive in-sample score — a cheap regression test against accidental leakage.

---

# 13. Performance considerations

Cache WL states so that computing depth $k$ does not recompute shallower levels. Training complexity is dominated by WL refinement over all edges for $K$ iterations plus aggregation of node statistics: $O(K\cdot(|V|+|E|))$ per graph, plus sorting cost for the neighbor multisets.

Optimizations:

- canonical serialized tuples rather than large strings;
- incremental depth computation;
- preallocated per-depth dictionaries;
- **integer interning of identifiers** — store a `dict[bytes, int]` vocabulary per depth and keep only the small integer on the class records. At $K$ depths and millions of nodes, full hex digests dominate memory;
- numerically stable online (Welford) mean/variance updates;
- optional parallel graph processing (deterministic reduction order required, or the result depends on scheduling).

## 13.1 Hash width and collision budget

**▸ Review note (added).** If identifiers are truncated digests, the collision probability follows the birthday bound: with $n$ distinct environments and a $b$-bit digest, $P_{\text{collision}}\approx n^2/2^{b+1}$. For $n=10^8$ environments, 64 bits gives $\approx 0.3$ — unacceptable; 128 bits gives $\approx 10^{-20}$. **Use at least 128 bits.** A collision silently merges two unrelated environments *and* breaks the unique-parent invariant, which is why §10.1 asserts on it.

Correctness and determinism take priority over premature optimization.

---

# 14. Relationship to message passing

A depth-$k$ WL identifier summarizes a rooted neighborhood after $k$ rounds of refinement, giving a natural comparison with message-passing graph neural networks (MPNNs), whose node states also aggregate information over a finite neighborhood. The correspondence is exact in the standard sense: a $k$-layer MPNN is at most as discriminative as $k$ rounds of 1-WL, with equality for injective aggregators [Morris2019WLGoNeural].

WLLR differs in that:

- WL forms explicit discrete equivalence classes;
- regression is a lookup over empirical target statistics;
- OOV behavior is deterministic and hierarchical;
- every prediction exposes its own support statistics;
- no continuous representation or learned mapping is required.

This motivates **depth-matched** GNN baselines: comparing WLLR at depth $K$ against a $K$-layer MPNN isolates the value of learned continuous interpolation over the WL partition, since both see exactly the same information. A WLLR result close to a depth-matched MPNN is the informative outcome; a large gap localizes the benefit to interpolation across, rather than within, WL classes — which is precisely the "graded similarity" direction discussed in `literature.md` §4.9.

---

# Part III — Review record

# 15. Review notes, changes, and open decisions

## 15.1 Substantive method changes made during consolidation

| # | Section | Change | Rationale |
|---|---|---|---|
| 1 | §5.1 | Added the prefix lemma: matched depths are downward closed | Makes $k^\star$ well defined, permits binary search, and invalidates the original multi-level-fallback test construction |
| 2 | §5.3 | Added support-thresholded depth selection; demoted pure deepest-match from default | Deepest-match on singleton classes maximizes variance while appearing maximally confident |
| 3 | §6.1 | Made top-down computation order and the use of the *shrunk* parent explicit; added $\alpha$ tuning protocol | The recursion is otherwise ambiguous and irreproducible |
| 4 | §3.6 | Distinguished partition stabilization from identifier change; added $K_{\text{eff}}$ | Depths past stabilization cost memory and buy nothing |
| 5 | §4 | Variance is `None`, never `0.0`, at $N=1$ | A stored zero is indistinguishable from a homogeneous class |
| 6 | §10.3 | Added leave-one-out class means and `predict_loo` | Any in-sample number is otherwise meaningless; also the cheapest leakage guard |
| 7 | §11.4 | Added an evaluation protocol (metrics, coverage, baselines, ablations) | Neither draft specified one |
| 8 | §13.1 | Added a hash-collision budget; ≥128 bits required | 64-bit truncation collides at realistic dataset sizes and silently breaks the parent invariant |
| 9 | §8 | Added `support_threshold_binding` to metadata | Otherwise a threshold-limited match is indistinguishable from OOV |
| 10 | §12 | Restructured tests: added isomorphism invariance, the 1-WL negative control, nesting, shrinkage limits, and the leakage guard; corrected the multi-level fallback construction | Coverage gaps plus one test that specified an unreachable state |

## 15.2 Open decisions for the author

1. **Default $n_{\min}$.** §5.3 argues against $n_{\min}=1$ as the library default but does not pick a value; this should be set empirically on the first dataset, not by fiat.
2. **Raw vs shrunk as the headline model.** The review's position is that the shrunk variant should be the headline and the raw variant the ablation, since raw deepest-match is the configuration most likely to be criticized. Both drafts treat shrinkage as an optional extension.
3. **Whether to claim the regressogram framing (`literature.md` §3) in the abstract.** It is the most defensible one-line description, but it invites direct comparison to the nonparametric-regression literature, which the current evaluation plan does not yet address.
4. **Repository layout.** The proposed structure remains:

```text
project/
├── src/wllr/
│   ├── __init__.py
│   ├── encoding.py
│   ├── statistics.py
│   ├── estimators.py
│   ├── regression.py
│   └── adapters/
├── tests/
├── examples/
├── docs/wllr.md              # this document
├── references/
│   ├── README.md             # the bibliography workflow
│   ├── doi_list.txt          # citekey -> DOI registry (source of truth)
│   ├── manual.bib            # DOI-less entries, quarantined and dated
│   ├── doi2bib.sh               # vendored resolver
│   ├── generate_bibtex.sh    # build + validate
│   └── references.bib        # GENERATED, gitignored
├── pyproject.toml
├── README.md
└── LICENSE
```

This document is the canonical behavior specification for the implementation.
