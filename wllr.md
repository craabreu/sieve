# Weisfeiler–Lehman Lookup Regression (WLLR)

## Method definition, implementation specification, and literature assessment

**Status:** consolidated working document
**Date:** 2026-08-17
**Supersedes:** `wllr_implementation_spec.md`, `wllr_extended_literature_search.md`
**Audience:** developers and coding agents implementing the library; authors drafting the manuscript
**Reference status:** every reference in Part IV has been resolved against its DOI (or publisher record where no DOI exists) — see §21

---

# Part 0 — Reading guide

| Part | Contents |
|---|---|
| I (§1–§6) | Problem statement, WL environment construction, estimator, backoff, regularization |
| II (§7–§14) | Object model, APIs, validation rules, tests, performance, GNN relationship |
| III (§15–§20) | Literature, precedents, novelty boundary, related-work structure |
| IV (§21) | Certified references and BibTeX |
| V (§22) | Review notes and open decisions carried over from consolidation |

Passages marked **▸ Review note** were added or changed during consolidation and record a substantive technical judgment rather than a restatement of the original drafts. §22 lists them in one place.

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

so at fixed depth the estimator is the squared-error-optimal piecewise-constant predictor over WL equivalence classes. Across depths it is best described as a **hierarchical regressogram over the nested vertex partitions induced by WL color refinement** (§17).

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

For a chemistry-focused first implementation an RDKit adapter is the most practical, but core statistics and backoff logic must remain graph-library agnostic. (Note that with a standard atom-invariant schema the RDKit path reproduces Morgan/ECFP atom-environment identifiers almost exactly — see §18.6.)

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

This motivates **depth-matched** GNN baselines: comparing WLLR at depth $K$ against a $K$-layer MPNN isolates the value of learned continuous interpolation over the WL partition, since both see exactly the same information. A WLLR result close to a depth-matched MPNN is the informative outcome; a large gap localizes the benefit to interpolation across, rather than within, WL classes — which is precisely the "graded similarity" direction of §18.8.

---

# Part III — Literature and novelty

# 15. Search strategy

A narrow search for "Weisfeiler–Lehman regression," "WL target encoding," or "WL mean regression" misses the relevant precedents, because WLLR combines ideas developed in partly separate literatures. The search was therefore decomposed into:

1. discrete rooted-environment representations;
2. hierarchical graph partitions;
3. empirical target averaging within structural classes;
4. lookup-based property prediction;
5. recursive fallback to less-specific environments;
6. hierarchical shrinkage of low-support estimates;
7. chemistry-specific atom-property tables;
8. WL methods that relax exact equality into graded similarity.

The question was broadened from *"has someone published WL regression?"* to *"has prior work used nested local structural classes, attached empirical target statistics to those classes, and predicted with hierarchical fallback or smoothing?"*

---

# 16. Map of closest precedents

| Work | Main overlap with WLLR | Main difference | Relevance |
|---|---|---|---|
| **Kuhn et al. 2008, HOSE NMR prediction** | Nested atom environments; average of matching targets; sphere-by-sphere fallback | HOSE rather than WL; NMR-specific | Closest precedent for the exact inference rule |
| **Katz 1987, backoff smoothing** | Most-specific context → recursive backoff to shorter context under sparse counts | Language modeling; discrete distributions, not conditional means | Canonical precedent for the backoff *principle* |
| **Lehner et al. 2023, DASH** | Hierarchical atom-centered substructures; property distributions at hierarchy nodes | Hierarchy derived from GNN attention; chemistry-specific | Very close architectural precedent |
| **Lehner et al. 2024, DASH Properties** | Hierarchical structural classes populated with atomic target values; no refit per property | DASH hierarchy and medians rather than WL classes and means | Closest general atomic-property precedent |
| **Kriege, Giscard & Wilson 2016, WL-OA** | Successive WL refinements induce a hierarchy of vertex classes | No target regression | Key theoretical precedent for WL ancestry |
| **Rogers & Hahn 2010, ECFP** | Iterative atom-environment identifiers at increasing radius | Fingerprint features for downstream models, not a lookup estimator | WL identifiers on molecules ≈ ECFP atom-environment identifiers |
| **Dalke, Hert & Kramer 2018, mmpdb** | Radius-specific environments; empirical property-change statistics per environment | Transformation/property-change setting, not node regression | Strong data-structure analogy |
| **Kammeraad et al. 2020** | Graph-derived atom types mapped to average partial charges | Shallow manual classes; no hierarchy or fallback | Direct mean-per-structural-class precedent |
| **Agarwal et al. 2022, Hierarchical Shrinkage** | Specific predictions shrunk toward ancestor sample means by support | Decision-tree hierarchy rather than WL | Direct precedent for the regularized variant |
| **Schulz et al. 2022, generalized WL kernel** | Graded similarity between neighborhood trees instead of strict WL equality | Kernel similarity, not regression | Relevant future extension |

---

# 17. WLLR as a hierarchical regressogram

At every depth $k$, WL defines a partition $\Pi_k=\{C_{k,1},C_{k,2},\ldots\}$. WLLR associates each cell with $\hat\mu_{k,c}$, and at prediction time selects the finest supported cell on the query node's ancestry chain. Thus WLLR is

> **a hierarchical regressogram over the nested vertex partitions induced by WL color refinement**,

which is a more precise framing than "a dictionary of hashes" and connects the method to the nonparametric-regression literature rather than only to cheminformatics lookup tables. Kriege et al. supply the formal statement that WL refinement furnishes the hierarchy [Kriege2016WLOA].

---

# 18. Precedent detail

## 18.1 HOSE codes: the closest inference rule

Bremser introduced **HOSE** (Hierarchically Ordered Spherical Environment) codes to encode progressively larger atom-centered chemical environments [Bremser1978HOSE]. NMRShiftDB used HOSE-based lookup tables for chemical-shift prediction [Steinbeck2003NMRShiftDB].

Kuhn et al. describe an NMR prediction procedure extremely close to WLLR [Kuhn2008NMR]: construct a multi-sphere HOSE code; search training data for atoms with the same environment; if matches exist, use the **average** of their target values; if not, reduce the number of spheres until a match is obtained.

Conceptually,

$$
\text{specific environment}\rightarrow\text{mean of matching labels}\rightarrow\text{lower-radius fallback},
$$

which parallels $h_K\rightarrow h_{K-1}\rightarrow\cdots\rightarrow h_0\rightarrow\mu_{\mathrm{global}}$.

The following must therefore **not** be claimed as independently novel: nested local atom environments; lookup of matching structural environments; averaging matched target values; recursive fallback to smaller environments.

## 18.2 Backoff smoothing

**▸ Review note (added; absent from both drafts).** The backoff rule is not merely HOSE-specific practice — it is the standard sparse-data device from statistical language modeling. Katz backoff predicts from the most specific $n$-gram context with sufficient count and otherwise recurses to the $(n-1)$-gram, with discounting to reserve mass for unseen contexts [Katz1987Backoff].

WLLR's $h_K\to h_{K-1}\to\cdots$ is the same recursion with WL depth playing the role of context length. This matters for the write-up in two ways:

1. It further weakens any claim to novelty for the backoff mechanism itself, and should be acknowledged rather than discovered by a reviewer.
2. It supplies the *correct diagnosis* of the §5.3 concern: the language-modeling literature established decades ago that most-specific-match on raw counts is unstable and requires either a count threshold (Katz's cutoff) or interpolation (Jelinek–Mercer, which is structurally what §6's shrinkage is). Framing §5.3 and §6 as the WL analogues of established smoothing practice is stronger than presenting them as ad hoc regularization.

## 18.3 DASH (2023)

Lehner et al. introduced the **Dynamic Attention-Based Substructure Hierarchy** for atomic partial-charge assignment [Lehner2023DASH]. DASH builds a tree of increasingly detailed atom-centered substructures, with expansion order guided by attention values from a GNN trained for partial-charge prediction.

Shared with WLLR: atom-centered hierarchical environments; increasingly specific local descriptions; interpretable structural matching; empirical property information attached to matched classes. Differences: DASH's hierarchy is derived from a trained GNN and is chemistry-specific; WLLR's is deterministic WL refinement on arbitrary attributed graphs.

## 18.4 DASH Properties (2024): the closest general precedent

The follow-up reuses an existing DASH tree and populates its nodes with additional atomic properties, computing the **median and variance** of each property at each hierarchy node [Lehner2024DASHProperties]. The authors explicitly note that this requires no new fit per property. The same hierarchy serves alternative partial-charge models, atomic dispersion, atomic polarizability, and electrophilicity/nucleophilicity-related quantities.

DASH Properties is therefore already a model of the form *hierarchical local structural class → empirical atomic-property statistic*.

| DASH Properties | WLLR |
|---|---|
| hierarchy extracted from a GNN-attention model | hierarchy defined directly by WL refinement |
| chemistry-specific atom substructures | arbitrary attributed graphs |
| variable substructure expansion order | canonical refinement rounds |
| median reported as class property | conditional mean under squared-error loss |
| learned representation needed to build the tree | no learned representation |
| matching/stopping determined by DASH traversal | explicit deepest-supported-class policy |
| ancestor backoff is not the defining rule | ancestor backoff is core behavior |

**Implication.** WLLR must not be described as "generalizing HOSE lookup to arbitrary atomic properties" — DASH Properties already demonstrates transferable hierarchical lookup across multiple atomic properties. The defensible distinction is that WLLR uses the *canonical* nested partition induced by WL refinement itself as both the regression hierarchy and the OOV hierarchy, without first learning a representation or constructing a domain-specific tree.

## 18.5 WL refinement already defines the hierarchy

Kriege, Giscard, and Wilson exploit the hierarchical structure induced by WL refinement in the WL optimal-assignment kernel [Kriege2016WLOA]. This is the citation to use when defining WLLR's parent/ancestor relation (§3.3), so that the backoff chain is presented as a known property of WL rather than a construction of this work.

## 18.6 ECFP / Morgan identifiers

**▸ Review note (added; absent from both drafts).** Extended-connectivity fingerprints assign each atom an identifier that is iteratively updated from its own identifier and those of its neighbors, at increasing radius [Rogers2010ECFP], following Morgan's connectivity-relaxation algorithm [Morgan1965]. This *is* 1-WL applied to molecular graphs with a specific atom-invariant schema.

Two consequences:

1. On molecules, $h_k(v)$ is essentially the ECFP atom-environment identifier at radius $k$. Any claim that WL identifiers offer a new molecular representation would be incorrect, and a cheminformatics reviewer will say so immediately. The framing must be that WLLR is a new *estimator over* a very familiar representation.
2. It is a practical opportunity: an RDKit-backed implementation can be validated against `GetMorganFingerprint` atom-environment identifiers as an independent correctness check on the encoder (§9).

## 18.7 mmpdb: radius-specific environment statistics

Dalke, Hert, and Kramer's `mmpdb` defines **rule environments** at multiple fingerprint radii: one environment per radius, larger radii giving greater specificity, with property-change statistics stored per rule/environment/property combination [Dalke2018MMPDB].

$$
\text{environment}_r\rightarrow\{\Delta y_1,\ldots,\Delta y_n\}\rightarrow\text{empirical statistics}.
$$

This is not node regression — the target is a property *change* associated with a transformation — but it is a strong precedent for the data structure $D_k[\text{environment}]=\{N,\text{mean},\text{variance},\ldots\}$ and for the specificity/support tradeoff that §5.3 addresses.

## 18.8 Mean property assignment over graph-derived atom classes

Kammeraad et al. aggregated atoms with equivalent graphical connectivity, averaged their partial charges, and reused the means for atoms of the corresponding graph-derived type [Kammeraad2020Representations]:

$$
c(v)\mapsto\frac{1}{|C_c|}\sum_{u\in C_c}q_u .
$$

No WL refinement, hierarchy, or fallback — but a direct precedent for mean-per-structural-class atomic-property assignment.

## 18.9 Generalized WL kernels: a principled future extension

Exact WLLR treats two different identifiers at the same depth as unrelated. Schulz et al. identify the analogous rigidity in standard WL kernels and compare neighborhood trees by graded similarity instead of binary equality [Schulz2022GeneralizedWL].

Current backoff is vertical, $h_k\rightarrow h_{k-1}$. A future model could also smooth horizontally,

$$
\text{exact }h_k\rightarrow\text{similar }h_k\rightarrow h_{k-1}\rightarrow\cdots,
$$

This is future work, not part of the base WLLR definition.

## 18.10 Node regression and WL equivalence

D'Inverno et al. analyze the approximation capability of GNNs for node classification and regression in relation to 1-WL equivalence, showing GNNs are universal approximators in probability for functions satisfying 1-WL node equivalence [DInverno2024NodeRegression]. This supports interpreting WLLR as a model that is constant within finite-depth WL equivalence classes, and bounds what any depth-matched GNN baseline can do that WLLR cannot.

## 18.11 Target/mean encoding

Once a WL identifier is treated as a categorical variable, mapping each observed class to its target average is target (mean) encoding. Micci-Barreca discussed target-dependent encoding for high-cardinality categorical variables and its regularization [MicciBarreca2001HighCardinality]. This literature motivates the shrinkage of §6, the leave-one-out requirement of §10.3, and the strict anti-leakage protocol of §11.

## 18.12 Uncertainty

Jonas and Kuhn developed NMR prediction with quantified uncertainty [JonasKuhn2019Uncertainty]. Not the same estimator, but the relevant background if calibrated uncertainty is added to WLLR's $(k^\star,N,s)$ diagnostics.

---

# 19. Novelty boundary

## 19.1 Strongly established components

Prior work supports each of these individually:

- nested local chemical environments — HOSE, DASH;
- WL identifiers as molecular atom-environment descriptors — ECFP/Morgan;
- empirical property statistics attached to structural classes — HOSE, DASH Properties, mmpdb, atom-type averaging;
- hierarchical atom-property tables — DASH / DASH Properties;
- radius-specific environment statistics — HOSE, mmpdb;
- recursive fallback to less specific contexts — HOSE, Katz backoff;
- nested WL vertex partitions — WL refinement, WL-OA;
- ancestor-based statistical shrinkage — Hierarchical Shrinkage, mean encoding;
- relaxed similarity between nonidentical WL environments — generalized WL kernels.

## 19.2 Combination that still appears distinctive

$$
\boxed{
\begin{array}{c}
\text{canonical finite-depth 1-WL vertex partitions}\\
+\\
\text{empirical node-target conditional means}\\
+\\
\text{deepest-supported-class prediction}\\
+\\
\text{deterministic recursive backoff through WL ancestors}
\end{array}
}
$$

used as a **domain-agnostic node-regression model**.

## 19.3 Recommended novelty statement

> **To our knowledge, prior work has not directly used the nested vertex partitions induced by finite-depth 1-WL refinement as a domain-agnostic nonparametric node-regression model, assigning each observed WL class an empirical conditional target statistic and performing deterministic ancestor backoff through WL refinement levels for unsupported classes.**

Avoid: *"We introduce hierarchical structural lookup regression"* (overstates novelty given HOSE and DASH); *"We are the first to use structural classes for atomic-property prediction"* (contradicted by several chemistry precedents); and any claim that the WL identifiers themselves are a new representation (contradicted by ECFP).

**▸ Review note (implementation consequence).** The implementation must not encode assumptions premised on lookup, averaging, or radius fallback being individually novel. WLLR should remain a configurable, general graph method rather than being tied to one molecular property. The distinctive contribution is the *hierarchy the machinery is applied to*, not the machinery.

---

# 20. Recommended related-work structure

**20.1 WL representations and nested partitions** — Shervashidze et al. 2011; Kriege, Giscard & Wilson 2016; Morris et al. 2019; D'Inverno et al. 2024; Schulz et al. 2022.

**20.2 Hierarchical atom-environment lookup** — Bremser 1978; Morgan 1965 and Rogers & Hahn 2010 (identifier construction); NMRShiftDB 2003; Kuhn et al. 2008; DASH 2023; DASH Properties 2024.

**20.3 Environment-conditioned empirical property statistics** — Kammeraad et al. 2020; mmpdb 2018; target/mean-encoding literature.

**20.4 Hierarchical statistical smoothing and backoff** — Katz 1987; Micci-Barreca 2001; Agarwal et al. 2022.

**Relevance ranking.**

1. **Kuhn et al. 2008 / HOSE** — closest precedent for exact-match averaging plus recursive radius fallback.
2. **Lehner et al. 2024 / DASH Properties** — closest precedent for a hierarchical local-environment classifier populated with arbitrary atomic-property statistics.
3. **Kriege et al. 2016 / WL-OA** — strongest formal precedent that WL refinement supplies a hierarchy.
4. **Agarwal et al. 2022 / Hierarchical Shrinkage** — closest statistical precedent for the regularized variant.
5. **Katz 1987** — canonical precedent for the backoff principle and its sparse-count failure modes.
6. **Rogers & Hahn 2010 / ECFP** — establishes that WL identifiers on molecules are a familiar representation.
7. **Dalke et al. 2018 / mmpdb** — radius-specific environment/property statistics.
8. **Kammeraad et al. 2020** — mean atomic properties over graph-derived structural classes.
9. **Schulz et al. 2022 / generalized WL** — basis for future similarity-based smoothing.

> **WLLR should be presented as a particular use of WL's nested vertex partitions as the complete regression and OOV-backoff hierarchy, not as the invention of hierarchical environment-based property lookup.**

---

# Part IV — References

# 21. Certified references

18 references: 15 resolved through `doi2bib.sh` against `dx.doi.org`, and three with no registered DOI (Shervashidze, Kriege, Agarwal) verified against the publisher's own record — JMLR and PMLR metadata tags and the NeurIPS proceedings page respectively. Corrections applied during certification are listed in §22.3.

**This list is not the source of truth.** The bibliography is generated: DOI-bearing entries live in `references/doi_list.txt` as `citekey → DOI` pairs, DOI-less entries in `references/manual.bib`, and `references/generate_bibtex.sh` builds `references/references.bib` from both. The generator refuses to emit output if any DOI fails to resolve, if a returned record does not match the requested DOI, or if this document cites a key that is not registered. Run it after any change to the reference list; see `references/README.md`.

**[Bremser1978HOSE]** Bremser, W. *HOSE — A Novel Substructure Code.* **Analytica Chimica Acta** **103**(4), 355–365 (1978). DOI: `10.1016/S0003-2670(01)83100-7`

**[Morgan1965]** Morgan, H. L. *The Generation of a Unique Machine Description for Chemical Structures — A Technique Developed at Chemical Abstracts Service.* **Journal of Chemical Documentation** **5**(2), 107–113 (1965). DOI: `10.1021/c160017a018`

**[Katz1987Backoff]** Katz, S. M. *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer.* **IEEE Transactions on Acoustics, Speech, and Signal Processing** **35**(3), 400–401 (1987). DOI: `10.1109/TASSP.1987.1165125`

**[MicciBarreca2001HighCardinality]** Micci-Barreca, D. *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* **ACM SIGKDD Explorations Newsletter** **3**(1), 27–32 (2001). DOI: `10.1145/507533.507538`

**[Steinbeck2003NMRShiftDB]** Steinbeck, C.; Krause, S.; Kuhn, S. *NMRShiftDB — Constructing a Free Chemical Information System with Open-Source Components.* **Journal of Chemical Information and Computer Sciences** **43**(6), 1733–1739 (2003). DOI: `10.1021/ci0341363`

**[Kuhn2008NMR]** Kuhn, S.; Egert, B.; Neumann, S.; Steinbeck, C. *Building Blocks for Automated Elucidation of Metabolites: Machine Learning Methods for NMR Prediction.* **BMC Bioinformatics** **9**, 400 (2008). DOI: `10.1186/1471-2105-9-400`

**[Rogers2010ECFP]** Rogers, D.; Hahn, M. *Extended-Connectivity Fingerprints.* **Journal of Chemical Information and Modeling** **50**(5), 742–754 (2010). DOI: `10.1021/ci100050t`

**[Shervashidze2011WL]** Shervashidze, N.; Schweitzer, P.; van Leeuwen, E. J.; Mehlhorn, K.; Borgwardt, K. M. *Weisfeiler–Lehman Graph Kernels.* **Journal of Machine Learning Research** **12**, 2539–2561 (2011). No DOI; https://www.jmlr.org/papers/v12/shervashidze11a.html

**[Kriege2016WLOA]** Kriege, N. M.; Giscard, P.-L.; Wilson, R. C. *On Valid Optimal Assignment Kernels and Applications to Graph Classification.* **Advances in Neural Information Processing Systems 29** (2016). No DOI; NeurIPS proceedings record; preprint arXiv:1606.01141

**[Dalke2018MMPDB]** Dalke, A.; Hert, J.; Kramer, C. *mmpdb: An Open-Source Matched Molecular Pair Platform for Large Multiproperty Data Sets.* **Journal of Chemical Information and Modeling** **58**(5), 902–910 (2018). DOI: `10.1021/acs.jcim.8b00173`

**[Morris2019WLGoNeural]** Morris, C.; Ritzert, M.; Fey, M.; Hamilton, W. L.; Lenssen, J. E.; Rattan, G.; Grohe, M. *Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks.* **Proceedings of the AAAI Conference on Artificial Intelligence** **33**(01), 4602–4609 (2019). DOI: `10.1609/aaai.v33i01.33014602`

**[JonasKuhn2019Uncertainty]** Jonas, E.; Kuhn, S. *Rapid Prediction of NMR Spectral Properties with Quantified Uncertainty.* **Journal of Cheminformatics** **11**, 50 (2019). DOI: `10.1186/s13321-019-0374-3`

**[Kammeraad2020Representations]** Kammeraad, J. A.; Goetz, J.; Walker, E. A.; Tewari, A.; Zimmerman, P. M. *What Does the Machine Learn? Knowledge Representations of Chemical Reactivity.* **Journal of Chemical Information and Modeling** **60**(3), 1290–1301 (2020). DOI: `10.1021/acs.jcim.9b00721`

**[Agarwal2022HierarchicalShrinkage]** Agarwal, A.; Tan, Y. S.; Ronen, O.; Singh, C.; Yu, B. *Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Models.* **Proceedings of the 39th International Conference on Machine Learning**, PMLR **162**, 111–135 (2022). No DOI; https://proceedings.mlr.press/v162/agarwal22b.html; preprint arXiv:2202.00858

**[Schulz2022GeneralizedWL]** Schulz, T. H.; Horváth, T.; Welke, P.; Wrobel, S. *A Generalized Weisfeiler–Lehman Graph Kernel.* **Machine Learning** **111**(7), 2601–2629 (2022). DOI: `10.1007/s10994-022-06131-w`

**[Lehner2023DASH]** Lehner, M. T.; Katzberger, P.; Maeder, N.; Schiebroek, C. C. G.; Teetz, J.; Landrum, G. A.; Riniker, S. *DASH: Dynamic Attention-Based Substructure Hierarchy for Partial Charge Assignment.* **Journal of Chemical Information and Modeling** **63**(19), 6014–6028 (2023). DOI: `10.1021/acs.jcim.3c00800`

**[DInverno2024NodeRegression]** D'Inverno, G. A.; Bianchini, M.; Sampoli, M. L.; Scarselli, F. *On the Approximation Capability of GNNs in Node Classification/Regression Tasks.* **Soft Computing** **28**(13–14), 8527–8547 (2024). DOI: `10.1007/s00500-024-09676-1`; preprint arXiv:2106.08992

**[Lehner2024DASHProperties]** Lehner, M. T.; Katzberger, P.; Maeder, N.; Landrum, G. A.; Riniker, S. *DASH Properties: Estimating Atomic and Molecular Properties from a Dynamic Attention-Based Substructure Hierarchy.* **The Journal of Chemical Physics** **161**(7), 074103 (2024). DOI: `10.1063/5.0218154`

BibTeX for all of the above is generated into `references/references.bib` — a build artifact, not a tracked file. Do not hand-edit it.

---

# Part V — Review record

# 22. Review notes, changes, and open decisions

## 22.1 Substantive method changes made during consolidation

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

## 22.2 Literature additions

| Reference | Why added |
|---|---|
| Katz 1987 | The backoff rule's canonical precedent, and the correct framing for §5.3/§6 as smoothing rather than ad hoc regularization. Absent from both drafts |
| Rogers & Hahn 2010 (+ Morgan 1965) | ECFP atom-environment identifiers *are* 1-WL on molecules. Its absence was the most exposed gap: a cheminformatics reviewer would raise it immediately. Also enables an independent encoder validation against RDKit |
| Morris et al. 2019 | The standard citation for the MPNN↔1-WL correspondence underpinning §14's depth-matched baseline argument |

## 22.3 Reference corrections applied

| Reference | Correction |
|---|---|
| D'Inverno et al. | No longer preprint-only — published as **Soft Computing 28(13–14), 8527–8547 (2024)**, DOI `10.1007/s00500-024-09676-1`. Both drafts cited only arXiv:2106.08992 |
| Dalke et al. | Second author is **Jérôme** Hert, not "Jérémy" as in the draft BibTeX |
| Agarwal et al. | Published PMLR title ends "tree-based **models**", not "methods" (the arXiv preprint uses "methods") |
| Bremser; Micci-Barreca; Steinbeck; Lehner 2023/2024; Schulz; Rogers | Issue numbers added from the Crossref records |
| Kuhn et al. 2008 | Confirmed as article number 400 with no page range — correct as drafted |
| Shervashidze; Kriege; Agarwal | Confirmed to have no registered DOI; verified against JMLR, NeurIPS, and PMLR records, with arXiv preprint IDs recorded where available |

## 22.4 Open decisions for the author

1. **Default $n_{\min}$.** §5.3 argues against $n_{\min}=1$ as the library default but does not pick a value; this should be set empirically on the first dataset, not by fiat.
2. **Raw vs shrunk as the headline model.** The review's position is that the shrunk variant should be the headline and the raw variant the ablation, since raw deepest-match is the configuration most likely to be criticized. Both drafts treat shrinkage as an optional extension.
3. **Whether to claim the regressogram framing (§17) in the abstract.** It is the most defensible one-line description, but it invites direct comparison to the nonparametric-regression literature, which the current evaluation plan does not yet address.
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
