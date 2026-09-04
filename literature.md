# Sieve — Literature Review and Novelty Assessment

**Status:** reference material for the manuscript
**Scope:** what has been published that overlaps with Sieve, and what is defensibly new
**Relationship to `design.md`:** that document specifies the method and implementation. This one makes
no implementation claims and should not be built from. Method details quoted here for comparison may
have evolved; `design.md` is authoritative on all of them.
**Provenance:** extracted from `drafts/wllr.md`, which is superseded. This is the live copy.

---

## 1. Search strategy

A narrow search for "Weisfeiler–Lehman regression," "WL target encoding," or "WL mean regression"
misses the relevant precedents, because Sieve combines ideas developed in partly separate literatures.
The search was therefore decomposed into:

1. discrete rooted-environment representations;
2. hierarchical graph partitions;
3. empirical target averaging within structural classes;
4. lookup-based property prediction;
5. recursive fallback to less-specific environments;
6. hierarchical shrinkage of low-support estimates;
7. chemistry-specific atom-property tables;
8. WL methods that relax exact equality into graded similarity.

The question was broadened from *"has someone published WL regression?"* to *"has prior work used
nested local structural classes, attached empirical target statistics to those classes, and predicted
with hierarchical fallback or smoothing?"*

---

## 2. Map of closest precedents

| Work | Main overlap with Sieve | Main difference | Relevance |
|---|---|---|---|
| **Kuhn et al. 2008, HOSE NMR prediction** | Nested atom environments; average of matching targets; sphere-by-sphere fallback | HOSE rather than WL; NMR-specific | Closest precedent for the exact inference rule |
| **Katz 1987, backoff smoothing** | Most-specific context → recursive backoff to shorter context under sparse counts | Language modeling; discrete distributions, not conditional means | Canonical precedent for the backoff *principle* |
| **Kneser & Ney 1995, continuation counts** | The distribution backed off *to* is estimated from distinct-context counts, not raw frequency | Language modeling; discrete distributions | The fallback estimator must be calibrated for the population that actually reaches it |
| **Chen & Goodman 1999, smoothing study** | Controlled comparison of the whole smoothing family; introduces modified Kneser–Ney | Language modeling; perplexity rather than regression error | Empirical authority for which smoothing choices matter, and for interpolation over pure backoff |
| **Lehner et al. 2023, DASH** | Hierarchical atom-centered substructures; property distributions at hierarchy nodes | Hierarchy derived from GNN attention; chemistry-specific | Very close architectural precedent |
| **Lehner et al. 2024, DASH Properties** | Hierarchical structural classes populated with atomic target values; no refit per property | DASH hierarchy and medians rather than WL classes and means | Closest general atomic-property precedent |
| **Kriege, Giscard & Wilson 2016, WL-OA** | Successive WL refinements induce a hierarchy of vertex classes | No target regression | Key theoretical precedent for WL ancestry |
| **Rogers & Hahn 2010, ECFP** | Iterative atom-environment identifiers at increasing radius | Fingerprint features for downstream models, not a lookup estimator | WL identifiers on molecules ≈ ECFP atom-environment identifiers |
| **Dalke, Hert & Kramer 2018, mmpdb** | Radius-specific environments; empirical property-change statistics per environment | Transformation/property-change setting, not node regression | Strong data-structure analogy |
| **Kammeraad et al. 2020** | Graph-derived atom types mapped to average partial charges | Shallow manual classes; no hierarchy or fallback | Direct mean-per-structural-class precedent |
| **Agarwal et al. 2022, Hierarchical Shrinkage** | Specific predictions shrunk toward ancestor sample means by support | Decision-tree hierarchy rather than WL | Direct precedent for the regularized variant |
| **Schulz et al. 2022, generalized WL kernel** | Graded similarity between neighborhood trees instead of strict WL equality | Kernel similarity, not regression | Relevant future extension |
| **Bilmes & Kirchhoff 2003, factored LMs** | Backoff over an ordered set of factors, and the problem of choosing the order | Language modeling | Precedent for graded attribute levels |

---

## 3. Sieve as a hierarchical regressogram

At every level $k$, WL defines a partition $\Pi_k=\{C_{k,1},C_{k,2},\ldots\}$. Sieve associates each
cell with its empirical target mean, and at prediction time selects the finest supported cell on the
query node's ancestry chain. Thus Sieve is

> **a hierarchical regressogram over the nested vertex partitions induced by WL color refinement**,

which is a more precise framing than "a dictionary of hashes" and connects the method to the
nonparametric-regression literature rather than only to cheminformatics lookup tables. Kriege et al.
supply the formal statement that WL refinement furnishes the hierarchy [Kriege2016WLOA].

This is the most defensible one-line description of the method. It does invite direct comparison with
the nonparametric-regression literature, which the evaluation plan should be ready for.

---

## 4. Precedent detail

### 4.1 HOSE codes: the closest inference rule

Bremser introduced **HOSE** (Hierarchically Ordered Spherical Environment) codes to encode
progressively larger atom-centered chemical environments [Bremser1978HOSE]. NMRShiftDB used
HOSE-based lookup tables for chemical-shift prediction [Steinbeck2003NMRShiftDB].

Kuhn et al. describe an NMR prediction procedure extremely close to Sieve [Kuhn2008NMR]: construct a
multi-sphere HOSE code; search training data for atoms with the same environment; if matches exist,
use the **average** of their target values; if not, reduce the number of spheres until a match is
obtained.

Conceptually,

$$
\text{specific environment}\rightarrow\text{mean of matching labels}\rightarrow\text{lower-radius fallback},
$$

which parallels $h_L\rightarrow h_{L-1}\rightarrow\cdots\rightarrow h_0\rightarrow\mu_{\mathrm{global}}$.

The following must therefore **not** be claimed as independently novel: nested local atom
environments; lookup of matching structural environments; averaging matched target values; recursive
fallback to smaller environments.

### 4.2 Backoff smoothing

The backoff rule is not merely HOSE-specific practice — it is the standard sparse-data device from
statistical language modeling. Katz backoff predicts from the most specific $n$-gram context with
sufficient count and otherwise recurses to the $(n-1)$-gram, with discounting to reserve mass for
unseen contexts [Katz1987Backoff].

Sieve's $h_L\to h_{L-1}\to\cdots$ is the same recursion with refinement level playing the role of
context length. This matters for the write-up in two ways:

1. It further weakens any claim to novelty for the backoff mechanism itself, and should be
   acknowledged rather than discovered by a reviewer.
2. It supplies the correct framing for support thresholding and shrinkage: the language-modeling
   literature established decades ago that most-specific-match on raw counts is unstable and requires
   either a count threshold (Katz's cutoff) or interpolation (Jelinek–Mercer, which is structurally
   what hierarchical shrinkage is). Presenting these as the WL analogs of established smoothing
   practice is stronger than presenting them as ad hoc regularization.

Bilmes and Kirchhoff generalize this to backoff over *multiple factors* rather than a single context,
including the problem of choosing an order among them [Bilmes2003FactoredLM] — the direct precedent
for graded attribute levels.

Katz's rule is not where that literature stops. Kneser and Ney observed that the distribution the
recursion falls back *to* should not be estimated the same way as the one it falls back *from*
[KneserNey1995Improved]. A lower-order estimate is consulted precisely when the more specific context
was **unseen**, so estimating it from raw frequency calibrates it for the wrong population — it is
dominated by the contexts that were never backed off from in the first place. Their correction is to
estimate it from *continuation counts*, the number of **distinct** higher-order contexts a symbol
completes, rather than from occurrence counts.

Chen and Goodman's controlled comparison of the family — additive, Good–Turing, Katz,
Jelinek–Mercer, Witten–Bell, absolute discounting, Kneser–Ney, on common data with parameters
optimized on held-out sets — is the standard empirical reference for which of these choices actually
matter, and contributes modified Kneser–Ney (interpolated, with count-dependent discounts), which
remains the field's default baseline [ChenGoodman1999Smoothing]. Two of its structural findings bear
directly on Sieve:

1. **Continuation-count estimation of the backoff distribution is the single largest effect** they
   isolate. Sieve's analog is exact and needs no new state: a class's stored $\bar y$ is the
   atom-weighted pool of everything beneath it, so it is dominated by its most abundant children —
   yet it is read *only* by queries whose own child class was absent. Aggregating a class's children
   as units, rather than pooling their atoms, is the direct translation, and the children's means
   already exist one level down.
2. **Interpolated models outperform pure backoff**, especially at low counts, because a small but
   nonzero count still benefits from the coarser estimate instead of being replaced by it. Sieve's
   `shrinkage_strength` (§4.12) is the interpolation arm of precisely this comparison — and Chen and
   Goodman's pairing of the two findings suggests the two are not independent, since what
   interpolation blends toward is what continuation counts correct.

The transfer is not automatic. These methods smooth discrete probability distributions, whereas Sieve
estimates conditional means, and the discounting machinery that reserves probability mass has no
direct analog here. What does transfer is the estimator-calibration argument — a claim about which
population a fallback estimate should represent, not about mass.

### 4.3 DASH (2023)

Lehner et al. introduced the **Dynamic Attention-Based Substructure Hierarchy** for atomic
partial-charge assignment [Lehner2023DASH]. DASH builds a tree of increasingly detailed atom-centered
substructures, with expansion order guided by attention values from a GNN trained for partial-charge
prediction.

Shared with Sieve: atom-centered hierarchical environments; increasingly specific local descriptions;
interpretable structural matching; empirical property information attached to matched classes.
Differences: DASH's hierarchy is derived from a trained GNN and is chemistry-specific; Sieve's is
deterministic WL refinement on arbitrary attributed graphs.

### 4.4 DASH Properties (2024): the closest general precedent

The follow-up reuses an existing DASH tree and populates its nodes with additional atomic properties,
computing the **median and variance** of each property at each hierarchy node
[Lehner2024DASHProperties]. The authors explicitly note that this requires no new fit per property.
The same hierarchy serves alternative partial-charge models, atomic dispersion, atomic polarizability,
and electrophilicity/nucleophilicity-related quantities.

DASH Properties is therefore already a model of the form *hierarchical local structural class →
empirical atomic-property statistic*.

| DASH Properties | Sieve |
|---|---|
| hierarchy extracted from a GNN-attention model | hierarchy defined directly by WL refinement |
| chemistry-specific atom substructures | arbitrary attributed graphs |
| variable substructure expansion order | canonical refinement rounds |
| median reported as class property | conditional mean under squared-error loss |
| learned representation needed to build the tree | no learned representation |
| matching/stopping determined by DASH traversal | explicit deepest-supported-class policy |
| ancestor backoff is not the defining rule | ancestor backoff is core behavior |

**Implication.** Sieve must not be described as "generalizing HOSE lookup to arbitrary atomic
properties" — DASH Properties already demonstrates transferable hierarchical lookup across multiple
atomic properties. The defensible distinction is that Sieve uses the *canonical* nested partition
induced by WL refinement itself as both the regression hierarchy and the OOV hierarchy, without first
learning a representation or constructing a domain-specific tree.

**A caution about multi-property claims.** Since Sieve now carries vector targets, it does what DASH
Properties advertises — one hierarchy, many properties, no refit — by storing a vector per class
rather than repeating a scalar fit. That capability is therefore *not* a point of distinction, and
claiming it as one would be answered by this paper directly. The distinction is the hierarchy, not
the multiplicity of properties.

### 4.5 WL refinement already defines the hierarchy

Kriege, Giscard, and Wilson exploit the hierarchical structure induced by WL refinement in the WL
optimal-assignment kernel [Kriege2016WLOA]. This is the citation to use when defining Sieve's
parent/ancestor relation, so that the backoff chain is presented as a known property of WL rather
than a construction of this work.

### 4.6 ECFP / Morgan identifiers

Extended-connectivity fingerprints assign each atom an identifier that is iteratively updated from its
own identifier and those of its neighbors, at increasing radius [Rogers2010ECFP], following Morgan's
connectivity-relaxation algorithm [Morgan1965]. This *is* 1-WL applied to molecular graphs with a
specific atom-invariant schema.

Two consequences:

1. On molecules, the level-$k$ identifier is essentially the ECFP atom-environment identifier at
   radius $k$. Any claim that WL identifiers offer a new molecular representation would be incorrect,
   and a cheminformatics reviewer will say so immediately. The framing must be that Sieve is a new
   *estimator over* a very familiar representation.
2. It is a practical opportunity: an RDKit-backed implementation can be validated against
   `GetMorganFingerprint` atom-environment identifiers as an independent correctness check on the
   encoder.

ECFP versus FCFP is also the resolution knob already in the wild — the same algorithm with coarser
pharmacophoric atom invariants — which is the precedent for varying attribute resolution.

### 4.7 mmpdb: radius-specific environment statistics

Dalke, Hert, and Kramer's `mmpdb` defines **rule environments** at multiple fingerprint radii: one
environment per radius, larger radii giving greater specificity, with property-change statistics
stored per rule/environment/property combination [Dalke2018MMPDB].

$$
\text{environment}_r\rightarrow\{\Delta y_1,\ldots,\Delta y_n\}\rightarrow\text{empirical statistics}.
$$

This is not node regression — the target is a property *change* associated with a transformation —
but it is a strong precedent for the data structure
$D_k[\text{environment}]=\{N,\text{mean},\text{variance},\ldots\}$ and for the specificity/support
tradeoff.

### 4.8 Mean property assignment over graph-derived atom classes

Kammeraad et al. aggregated atoms with equivalent graphical connectivity, averaged their partial
charges, and reused the means for atoms of the corresponding graph-derived type
[Kammeraad2020Representations]:

$$
c(v)\mapsto\frac{1}{|C_c|}\sum_{u\in C_c}q_u .
$$

No WL refinement, hierarchy, or fallback — but a direct precedent for mean-per-structural-class
atomic-property assignment.

### 4.9 Generalized WL kernels: a principled future extension

Exact Sieve treats two different identifiers at the same level as unrelated. Schulz et al. identify the
analogous rigidity in standard WL kernels and compare neighborhood trees by graded similarity instead
of binary equality [Schulz2022GeneralizedWL].

Current backoff is vertical, $h_k\rightarrow h_{k-1}$. A future model could also smooth horizontally,

$$
\text{exact }h_k\rightarrow\text{similar }h_k\rightarrow h_{k-1}\rightarrow\cdots
$$

This is future work, not part of the base Sieve definition.

### 4.10 Node regression and WL equivalence

D'Inverno et al. analyze the approximation capability of GNNs for node classification and regression
in relation to 1-WL equivalence, showing GNNs are universal approximators in probability for functions
satisfying 1-WL node equivalence [DInverno2024NodeRegression]. This supports interpreting Sieve as a
model that is constant within finite-depth WL equivalence classes, and bounds what any depth-matched
GNN baseline can do that Sieve cannot.

The correspondence between message passing and WL is exact in the standard sense: a $k$-layer MPNN is
at most as discriminative as $k$ rounds of 1-WL, with equality for injective aggregators
[Morris2019WLGoNeural]. This motivates **depth-matched** GNN baselines — comparing Sieve at level $K$
against a $K$-layer MPNN isolates the value of learned continuous interpolation over the WL partition,
since both see exactly the same information.

### 4.11 Target/mean encoding

Once a WL identifier is treated as a categorical variable, mapping each observed class to its target
average is target (mean) encoding. Micci-Barreca discussed target-dependent encoding for
high-cardinality categorical variables and its regularization [MicciBarreca2001HighCardinality]. This
literature motivates shrinkage, the leave-one-out requirement for in-sample diagnostics, and the
strict anti-leakage protocol.

### 4.12 Shrinkage as empirical Bayes

Morris reviewed parametric empirical-Bayes shrinkage estimators for the two-level normal hierarchical
model [Morris1983EmpiricalBayes]. Sieve's weight $N/(N+\alpha)$ (design.md §4.2) is exactly that
model's posterior mean, and Morris's own method-of-moments estimator for the model's variance ratio
is what makes $\alpha=\sigma^2/\tau^2$ (design.md §13 item 9) a computable quantity from a fitted
model rather than a free regularization knob chosen by fiat.

Worth checking against how the closest statistical precedent actually sets its own analogous
parameter in practice, not just its published formula. `imodels`
([csinva.io/imodels](https://csinva.io/imodels)) is the reference implementation of
[Agarwal2022HierarchicalShrinkage], by overlapping authors. Its `HSTreeRegressorCV`/
`HSTreeClassifierCV` do not use a closed-form estimator: they grid-search a fixed candidate list
(`[0, 0.1, 1, 10, 50, 100, 500]`) via k-fold CV. Reading its `_shrink_tree` source directly also
shows the formula itself is not the same recursion, despite the shared "shrink toward ancestors"
framing: unrolled one level, its default scheme shrinks the *increment* between child and parent by
a fixed constant $\lambda$, weighted by the **parent's** sample count
($v_{\text{parent}}+\frac{N_{\text{parent}}}{N_{\text{parent}}+\lambda}(v_{\text{child}}-v_{\text{parent}})$),
where Sieve's recursion blends the child's raw mean against the already-shrunk parent directly,
weighted by the **child's** own sample count, with $\alpha$ an estimated variance ratio rather than
a swept constant. So the closed-form route is a genuine departure from precedent's own practice, not
merely the same idea formalized differently.

### 4.13 Uncertainty

Jonas and Kuhn developed NMR prediction with quantified uncertainty [JonasKuhn2019Uncertainty]. Not
the same estimator, but the relevant background if calibrated uncertainty is ever added to Sieve's
$(k^\star,N,s^2)$ diagnostics — which are explicitly *not* calibrated intervals.

---

## 5. Novelty boundary

### 5.1 Strongly established components

Prior work supports each of these individually:

- nested local chemical environments — HOSE, DASH;
- WL identifiers as molecular atom-environment descriptors — ECFP/Morgan;
- empirical property statistics attached to structural classes — HOSE, DASH Properties, mmpdb,
  atom-type averaging;
- hierarchical atom-property tables, one hierarchy serving many properties — DASH Properties;
- radius-specific environment statistics — HOSE, mmpdb;
- recursive fallback to less specific contexts — HOSE, Katz backoff;
- backoff over ordered factor sets — factored language models;
- nested WL vertex partitions — WL refinement, WL-OA;
- ancestor-based statistical shrinkage — Hierarchical Shrinkage, mean encoding;
- relaxed similarity between nonidentical WL environments — generalized WL kernels.

### 5.2 Combination that still appears distinctive

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

### 5.3 Recommended novelty statement

> **To our knowledge, prior work has not directly used the nested vertex partitions induced by
> finite-depth 1-WL refinement as a domain-agnostic nonparametric node-regression model, assigning
> each observed WL class an empirical conditional target statistic and performing deterministic
> ancestor backoff through WL refinement levels for unsupported classes.**

Avoid:

- *"We introduce hierarchical structural lookup regression"* — overstates novelty given HOSE and DASH.
- *"We are the first to use structural classes for atomic-property prediction"* — contradicted by
  several chemistry precedents.
- Any claim that the WL identifiers themselves are a new representation — contradicted by ECFP.
- Any claim that one hierarchy serving many properties is distinctive — DASH Properties does this
  (§4.4).

The distinctive contribution is the *hierarchy the machinery is applied to*, not the machinery.

This statement should not be considered settled until the gaps in §9.1–§9.3 are closed: each of them
is a literature that could bear on it, and none has been searched.

---

## 6. Recommended related-work structure

**6.1 WL representations and nested partitions** — Shervashidze et al. 2011; Kriege, Giscard & Wilson
2016; Morris et al. 2019; D'Inverno et al. 2024; Schulz et al. 2022.

**6.2 Hierarchical atom-environment lookup** — Bremser 1978; Morgan 1965 and Rogers & Hahn 2010
(identifier construction); NMRShiftDB 2003; Kuhn et al. 2008; DASH 2023; DASH Properties 2024.

**6.3 Environment-conditioned empirical property statistics** — Kammeraad et al. 2020; mmpdb 2018;
target/mean-encoding literature.

**6.4 Hierarchical statistical smoothing and backoff** — Katz 1987; Kneser & Ney 1995; Chen &
Goodman 1999; Bilmes & Kirchhoff 2003; Micci-Barreca 2001; Agarwal et al. 2022.

### Relevance ranking

1. **Kuhn et al. 2008 / HOSE** — closest precedent for exact-match averaging plus recursive radius
   fallback.
2. **Lehner et al. 2024 / DASH Properties** — closest precedent for a hierarchical local-environment
   classifier populated with arbitrary atomic-property statistics.
3. **Kriege et al. 2016 / WL-OA** — strongest formal precedent that WL refinement supplies a hierarchy.
4. **Agarwal et al. 2022 / Hierarchical Shrinkage** — closest statistical precedent for the
   regularized variant.
5. **Katz 1987** — canonical precedent for the backoff principle and its sparse-count failure modes.
6. **Kneser & Ney 1995 / Chen & Goodman 1999** — the backoff estimate must be calibrated for the
   population that actually reaches it, and the controlled empirical case for that correction.
7. **Rogers & Hahn 2010 / ECFP** — establishes that WL identifiers on molecules are a familiar
   representation.
8. **Dalke et al. 2018 / mmpdb** — radius-specific environment/property statistics.
9. **Kammeraad et al. 2020** — mean atomic properties over graph-derived structural classes.
10. **Bilmes & Kirchhoff 2003** — backoff over ordered factors, and the ordering problem.
11. **Schulz et al. 2022 / generalized WL** — basis for future similarity-based smoothing.

> **Sieve should be presented as a particular use of WL's nested vertex partitions as the complete
> regression and OOV-backoff hierarchy, not as the invention of hierarchical environment-based
> property lookup.**

---

## 7. Certified references

All entries are resolved against their DOI, or against the publisher's own record where no DOI is
registered. **This list is not the source of truth** — the bibliography is generated from
`references/doi_list.txt` and `references/manual.bib` by `references/generate_bibtex.sh`, which
refuses to emit output if any DOI fails to resolve, if a returned record does not match the requested
DOI, or if any document cites a key that is not registered. See `references/README.md`.

**[Morgan1965]** Morgan, H. L. *The Generation of a Unique Machine Description for Chemical Structures — A Technique Developed at Chemical Abstracts Service.* **Journal of Chemical Documentation** **5**(2), 107–113 (1965). DOI: `10.1021/c160017a018`

**[Bremser1978HOSE]** Bremser, W. *HOSE — A Novel Substructure Code.* **Analytica Chimica Acta** **103**(4), 355–365 (1978). DOI: `10.1016/S0003-2670(01)83100-7`

**[Morris1983EmpiricalBayes]** Morris, C. N. *Parametric Empirical Bayes Inference: Theory and Applications.* **Journal of the American Statistical Association** **78**(381), 47–55 (1983). DOI: `10.1080/01621459.1983.10477920`

**[Katz1987Backoff]** Katz, S. M. *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer.* **IEEE Transactions on Acoustics, Speech, and Signal Processing** **35**(3), 400–401 (1987). DOI: `10.1109/TASSP.1987.1165125`

**[KneserNey1995Improved]** Kneser, R.; Ney, H. *Improved Backing-off for M-gram Language Modeling.* **1995 International Conference on Acoustics, Speech, and Signal Processing (ICASSP-95)** **1**, 181–184 (1995). DOI: `10.1109/ICASSP.1995.479394`

**[ChenGoodman1999Smoothing]** Chen, S. F.; Goodman, J. *An Empirical Study of Smoothing Techniques for Language Modeling.* **Computer Speech & Language** **13**(4), 359–393 (1999). DOI: `10.1006/csla.1999.0128`

**[MicciBarreca2001HighCardinality]** Micci-Barreca, D. *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* **ACM SIGKDD Explorations Newsletter** **3**(1), 27–32 (2001). DOI: `10.1145/507533.507538`

**[Bilmes2003FactoredLM]** Bilmes, J. A.; Kirchhoff, K. *Factored Language Models and Generalized Parallel Backoff.* **NAACL-HLT 2003, companion volume**, 4–6 (2003). DOI: `10.3115/1073483.1073485`

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

**[Mobley2018SMIRNOFF]** Mobley, D. L.; Bannan, C. C.; Rizzi, A.; Bayly, C. I.; Chodera, J. D.; Lim, V. T.; Lim, N. M.; Beauchamp, K. A.; Slochower, D. R.; Shirts, M. R.; Gilson, M. K.; Eastman, P. K. *Escaping Atom Types in Force Fields Using Direct Chemical Perception.* **Journal of Chemical Theory and Computation** **14**(11), 6076–6092 (2018). DOI: `10.1021/acs.jctc.8b00640`

Mobley et al. is registered for §9.2, where it marks an open gap rather than a surveyed precedent; it
is not yet reflected in §2, §4, or §5.

Welford 1962 and Chan, Golub & LeVeque 1983 are also registered, cited by `design.md` for the
accumulation and merge algorithms rather than as Sieve precedents.

---

## 8. Corrections applied during certification

| Reference | Correction |
|---|---|
| D'Inverno et al. | No longer preprint-only — published as **Soft Computing 28(13–14), 8527–8547 (2024)**, DOI `10.1007/s00500-024-09676-1`. The drafts cited only arXiv:2106.08992 |
| Dalke et al. | Second author is **Jérôme** Hert, not "Jérémy" as in the draft BibTeX |
| Agarwal et al. | Published PMLR title ends "tree-based **models**", not "methods" (the arXiv preprint uses "methods") |
| Bremser; Micci-Barreca; Steinbeck; Lehner 2023/2024; Schulz; Rogers | Issue numbers added from the Crossref records |
| Kuhn et al. 2008 | Confirmed as article number 400 with no page range — correct as drafted |
| Shervashidze; Kriege; Agarwal | Confirmed to have no registered DOI; verified against JMLR, NeurIPS, and PMLR records, with arXiv preprint IDs recorded where available |

### References added during review

| Reference | Why added |
|---|---|
| Katz 1987 | The backoff rule's canonical precedent, and the correct framing for support thresholding and shrinkage as smoothing rather than ad hoc regularization. Absent from the original drafts |
| Rogers & Hahn 2010 (+ Morgan 1965) | ECFP atom-environment identifiers *are* 1-WL on molecules. Its absence was the most exposed gap: a cheminformatics reviewer would raise it immediately. Also enables an independent encoder validation against RDKit |
| Morris et al. 2019 | The standard citation for the MPNN↔1-WL correspondence underpinning the depth-matched baseline argument |
| Bilmes & Kirchhoff 2003 | Precedent for backoff over an ordered set of factors, and for the ordering problem it creates |
| Kneser & Ney 1995 | The backoff *estimator* — not just the backoff rule — has an established correction. Sieve's fallback reads a class's atom-weighted pool, which is exactly the raw-frequency estimate Kneser–Ney replaces. Registering Katz without this made §4.2 cite the problem and omit the standard answer |
| Chen & Goodman 1999 | The controlled empirical study behind the whole family, and the source of modified Kneser–Ney. Supplies the evidence for two claims §4.2 now makes — that continuation-count estimation dominates, and that interpolation beats pure backoff — neither of which should rest on the mechanism papers alone |

---

## 9. Gaps to close before the manuscript

The review above is sufficient to *build* from: it fixes what cannot be claimed and names the
baselines the design must support. It is **not** yet sufficient to write a related-work section from.
The following are identified gaps, not survey results — with the single exception noted, no search has
been run against them and no claim is made here about what the prior art contains.

Items 1–3 bear on the novelty claim and should be closed before §5.3 is committed to. Item 4 should be
closed before the evaluation plan is fixed. Item 5 is the only one that could still touch `design.md`.

### 9.1 The regressogram framing is uncited

§3 calls Sieve "a hierarchical regressogram over the nested vertex partitions induced by WL color
refinement" and §5.3 treats that as the most defensible one-line description of the method. It is also
the only section of this document with no supporting reference.

Partitioning (histogram) regression estimators have a developed theory — consistency, convergence
rates, the bias–variance behavior of cell size. Claiming the framing invites the question of which of
those results transfer, given that WL cells are data-independent and do *not* shrink geometrically
with sample size the way the classical analysis assumes. Backoff makes the effective partition
data-dependent, which is a further departure.

**Needed:** the standard references for partitioning/histogram regression and their consistency
conditions, plus a paragraph stating honestly which apply to WL cells and which do not.

### 9.2 Force-field atom typing is missing, and it is the oldest chemistry precedent

SMARTS-based atom typing in classical force fields is an ordered list of structural patterns, first
match wins, with progressively less specific patterns as fallback, and empirically fitted parameters
attached to each type. That is specificity-ordered structural lookup with backoff, in production
decades before HOSE. §4.8 covers mean-per-class assignment but not this tradition. A force-field
reviewer would raise it the way a cheminformatics reviewer raises ECFP.

The Open Force Field initiative is the right entry point, and the citation cuts both ways.
Mobley et al. introduced **direct chemical perception** (SMIRNOFF) explicitly as a way to *escape*
atom types, arguing that discrete typing hierarchies are brittle, hard to extend, and force
chemically-unjustified parameter sharing [Mobley2018SMIRNOFF]. So the same literature supplies both
the strongest precedent for Sieve's machinery and a published argument that discrete structural-class
hierarchies are the thing to move away from.

Sieve needs an answer to that argument, and it has one worth making explicitly: WL classes are not
hand-authored, the hierarchy is canonical rather than curated, and backoff is a principled estimator
under sparse support rather than a pattern-ordering heuristic. That is a stronger position than
ignoring the objection.

**Needed:** a search of the atom-typing and typing-free force-field literature; a decision on whether
this becomes its own related-work subsection (§6.3 is the natural home).

### 9.3 Multilevel models and partial pooling

Shrinkage toward the already-shrunk parent over a nested grouping structure is, statistically, partial
pooling in a nested random-effects model. This document cites Agarwal et al. (trees) and mentions
Jelinek–Mercer interpolation in §4.2 prose without registering it as a reference.

A statistics reviewer may argue that hierarchically shrunk Sieve *is* an approximate varying-intercept
model over WL classes, and ask why a multilevel model is not fitted directly. The answer is presumably
cost and the merge property — the closed-form shrinkage survives model merging, a fitted hierarchical
model would not — but that answer is not currently written down anywhere.

**Needed:** the multilevel-model and empirical-Bayes references; Jelinek–Mercer registered; a stated
reason for the closed-form estimator over a fitted one.

### 9.4 No σ-profile prediction literature

The intended application has zero references in this document. Existing group-contribution and
machine-learned σ-profile predictors are the application baselines, and they determine what counts as
a competitive result.

**Needed:** this search must precede fixing the evaluation plan, not follow it.

### 9.5 Vector targets are treated as d independent scalars

`design.md` stores a per-class mean vector and per-component MSD, so a σ-profile is handled as d
unrelated numbers that happen to share a hierarchy. On a fixed grid a σ-profile is closer to a
function — or, being non-negative, to an unnormalized density — than to a vector of independent
outputs.

Elementwise means and elementwise shrinkage remain defensible: the grid is shared across all
molecules, which is what makes componentwise averaging meaningful, and since both backoff and
shrinkage return convex combinations the estimator is closed over non-negative vectors. But
"why not a distributional loss, or a barycentre under a transport metric" is a question that deserves
a prepared answer rather than an improvised one.

**Needed:** enough of the functional/multi-output regression literature to justify the elementwise
choice. This is the only gap that could still change `design.md`, so it should be resolved early.
