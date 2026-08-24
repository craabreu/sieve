# Sieve — Literature Review Update

**Status:** merge-ready literature update  
**Purpose:** close the five literature gaps identified in `literature.md` §9  
**Relationship to existing documents:** `design.md` remains authoritative for the method and implementation; `literature.md` remains the live literature review. This file records findings that should now be propagated into `literature.md`.  
**Scope:** partitioning regression, force-field atom typing, multilevel/partial-pooling models, σ-profile prediction, and functional/vector-response geometry.

---

# 1. Partitioning regression, regressograms, and local averaging

## 1.1 Fixed-level Sieve is a partitioning regressor

For a fixed refinement level $k$, Sieve predicts from the empirical mean of the response within the partition cell containing the query node:

$$
\hat m_k(x)
=
\frac{
\sum_{i=1}^{n}Y_i\mathbf 1\{X_i\in C_k(x)\}
}{
\sum_{i=1}^{n}\mathbf 1\{X_i\in C_k(x)\}
}.
$$

This is exactly a **partitioning regression estimator**, or regressogram, over the partition induced by level $k$.

The complete Sieve predictor is not a conventional fixed regressogram because the selected resolution depends on the query and observed support:

$$
k_n^\star(x)
=
\max\left\{
k:
N_{k,h_k(x)}
\ge n_{\min}
\right\}.
$$

A precise description is therefore:

> **Sieve is a support-adaptive partitioning regression estimator over a nested hierarchy of vertex environments. At each fixed level it reduces to a regressogram.**

Stone's local-averaging framework and Nobel's work on histogram regression with data-dependent partitions are the most relevant classical references.

## 1.2 Classical consistency theory does not transfer verbatim

Classical partition-regression consistency balances increasing partition resolution against increasing within-cell sample size. In Euclidean histogram regression this is often expressed through cells whose diameter shrinks while occupancy grows.

Sieve does not naturally live in a Euclidean feature space. Its cells are symbolic graph-equivalence classes, and refinement means structural specificity rather than geometric contraction.

Therefore classical results such as Nobel's should be cited as precedent and motivation, but **not claimed as direct proofs of Sieve consistency**.

## 1.3 Nested sigma-field formulation

Let

$$
\mathcal F_k=\sigma(h_k(X)).
$$

Because refinement is nested,

$$
\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots.
$$

The population prediction at level $k$ is

$$
m_k(X)=E[Y\mid\mathcal F_k].
$$

For a fixed maximum level $L$, infinite-data Sieve therefore targets

$$
E[Y\mid h_L(X)],
$$

not necessarily $E[Y\mid X]$. A fixed-depth model may retain a **representation-induced approximation floor**.

Define

$$
\mathcal F_\infty
=
\sigma\left(\bigcup_{k\ge0}\mathcal F_k\right).
$$

Then, under standard square-integrability assumptions,

$$
E[Y\mid\mathcal F_k]
\rightarrow
E[Y\mid\mathcal F_\infty].
$$

This separates:

- **estimation error** from finite support;
- **approximation error** from the refinement hierarchy.

## 1.4 Support gating as adaptive neighborhood selection

A natural asymptotic regime would use

$$
L_n\rightarrow\infty,
\qquad
n_{\min}(n)\rightarrow\infty,
\qquad
\frac{n_{\min}(n)}{n}\rightarrow0.
$$

This expresses the desired tradeoff:

$$
\underbrace{k_n^\star\rightarrow\infty}_{\text{increasing specificity}}
\qquad
\underbrace{N_{k_n^\star}\rightarrow\infty}_{\text{increasing support}}.
$$

Sieve can also be written as a local averaging estimator

$$
\hat m_n(x)=\sum_iW_{ni}(x)Y_i
$$

with uniform weights inside the deepest supported ancestor cell. This gives a useful analogy:

$$
k\text{-NN}:
\text{smallest geometric neighborhood with enough points}
$$

versus

$$
\text{Sieve}:
\text{most specific structural neighborhood with enough points}.
$$

## 1.5 Optional ultrametric interpretation

A nested refinement chain naturally defines an ultrametric. For example,

$$
r(x,x')=\max\{k:h_k(x)=h_k(x')\},
\qquad
d(x,x')=2^{-r(x,x')}.
$$

Deeper agreement means smaller distance, and refinement cells become nested ultrametric balls.

This is a promising route for a future formal consistency argument, but need not appear in the main manuscript unless the theory is developed.

## 1.6 References to register

**[Stone1977NonparametricRegression]**  
Stone, C. J. *Consistent Nonparametric Regression.* **The Annals of Statistics** 5(4), 595–620 (1977).  
DOI: `10.1214/aos/1176343886`

**[Nobel1996HistogramRegression]**  
Nobel, A. B. *Histogram Regression Estimation Using Data-Dependent Partitions.* **The Annals of Statistics** 24(3), 1084–1105 (1996).  
DOI: `10.1214/aos/1032526958`

**[CattaneoFarrell2013Partitioning]**  
Cattaneo, M. D.; Farrell, M. H. *Optimal Convergence Rates, Bahadur Representation, and Asymptotic Normality of Partitioning Estimators.* **Journal of Econometrics** 174(2), 127–143 (2013).  
DOI: `10.1016/j.jeconom.2013.02.002`

---

# 2. Force-field atom typing and direct chemical perception

## 2.1 Narrowing the historical claim

It is too broad to say that classical force-field atom typing universally consists of an ordered list of structural patterns with "first match wins."

A more accurate statement is:

> **Many force-field atom-typing systems implement chemical perception through hand-authored hierarchical or precedence-ordered structural rules, assigning reusable parameters through discrete chemical classes.**

Different implementations include hierarchical atom types, programmable decision trees, SMARTS-based rules with explicit overrides, and direct chemical perception.

## 2.2 Hierarchical atom type definitions

Jin et al. introduced **hierarchical atom type definitions (HAD)**, where increasingly specific local structural descriptions extend broader parent definitions.

This is a close structural precedent to Sieve's nested hierarchy.

The distinction is fundamental:

### HAD / force-field typing

- hierarchy is designed;
- parameters are fitted or curated;
- structural specificity determines selection.

### Sieve

- hierarchy is generated algorithmically;
- classes store empirical target statistics;
- prediction uses the most specific class satisfying

$$
N_{k,c}\ge n_{\min}.
$$

Thus **support gating** is a major distinction.

## 2.3 SMIRNOFF and direct chemical perception

SMIRNOFF replaces the atom-type intermediary with direct SMARTS/SMIRKS matching. Its parameter assignment remains hierarchical in the sense that more specific patterns override more general ones.

This is a precedent for

$$
\text{structural specificity}
\rightarrow
\text{parameter lookup}.
$$

Sieve differs in that:

| SMIRNOFF / HAD | Sieve |
|---|---|
| hierarchy human-authored | hierarchy algorithmically generated |
| parameters curated/fitted | values are empirical class statistics |
| most specific structural match wins | most specific statistically supported class wins |
| chemistry-specific | graph-domain agnostic |

The strongest Sieve distinction is therefore:

> **support-adaptive inference over an automatically generated refinement hierarchy.**

## 2.4 Foyer and CGenFF

Foyer formalizes precedence and override relations among SMARTS-based atom types.

CGenFF uses a programmable decision tree rather than a simple ordered pattern list.

These should be cited to avoid overgeneralizing the atom-typing tradition.

## 2.5 Force-field literature as a critique

Direct-chemical-perception work criticizes discrete atom typing for type proliferation, brittle extension, duplicated parameters, and chemically unjustified splitting or sharing.

The analogous risk in Sieve is:

$$
\text{increasing specificity}
\rightarrow
\text{class proliferation}
\rightarrow
\text{small support}.
$$

Sieve's empirical answer is to make support explicit, back off automatically, and optionally shrink low-support estimates.

This supports reporting:

- class count by level;
- singleton fraction;
- support distributions;
- matched-level distributions;
- error versus support;
- error versus selected level;
- sensitivity to $n_{\min}$.

## 2.6 Graded attribute levels

The WL portion of the hierarchy is deterministic given a declared feature schema.

The pre-WL graded attribute chain, however, uses a user-declared ordering.

The manuscript should distinguish these two facts rather than claim that the entire hierarchy is canonical.

## 2.7 References to register

**[Jin2016HAD]**  
Jin, Z.; Yang, X.; Xiao, X. *Hierarchical Atom Type Definitions and Extensible All-Atom Force Fields.* **Journal of Computational Chemistry** 37, 653–664 (2016).  
DOI: `10.1002/jcc.24244`

**[Vanommeslaeghe2012CGenFF]**  
Vanommeslaeghe, K.; MacKerell, A. D. Jr. *Automation of the CHARMM General Force Field (CGenFF) I: Bond Perception and Atom Typing.* **Journal of Chemical Information and Modeling** 52, 3144–3154 (2012).  
DOI: `10.1021/ci300363c`

**[Klein2019Foyer]**  
Klein, C. et al. *Formalizing Atom-Typing and the Dissemination of Force Fields with Foyer.* **Computational Materials Science** 167, 215–227 (2019).  
DOI: `10.1016/j.commatsci.2019.05.026`

**[Mobley2018SMIRNOFF]** remains central and should move from "open gap" into surveyed related work.

---

# 3. Multilevel models, empirical Bayes, and partial pooling

## 3.1 Local Bayesian interpretation

For a class $c$,

$$
Y_i\mid\theta_c\sim\mathcal N(\theta_c,\sigma^2)
$$

with prior

$$
\theta_c\mid\theta_{p(c)}
\sim
\mathcal N(\theta_{p(c)},\tau_k^2),
$$

the posterior mean is

$$
E[\theta_c\mid Y,\theta_{p(c)}]
=
\frac{
N_c\bar Y_c+\alpha_k\theta_{p(c)}
}{
N_c+\alpha_k
},
\qquad
\alpha_k=\frac{\sigma^2}{\tau_k^2}.
$$

This has exactly the algebra of Sieve's hierarchical shrinkage rule.

Thus $\alpha_k$ can be interpreted as a prior effective sample size or within/between variance ratio.

## 3.2 Not an exact nested random-effects posterior

Because

$$
C_{k,c}\subseteq C_{k-1,p(c)},
$$

the empirical parent mean includes the child's own observations.

Recursively substituting the pooled parent estimate is therefore not generally the exact posterior update of a full hierarchical Gaussian model.

The safest terminology is:

> **hierarchical shrinkage inspired by partial pooling**

or

> **recursive shrinkage of nested empirical class means.**

## 3.3 "Empirical Bayes" should be used carefully

If $\alpha_k$ is chosen by cross-validation, Sieve is a regularized estimator, not automatically an empirical-Bayes model.

An empirical-Bayes interpretation becomes appropriate if variance components or hyperparameters are estimated under an explicit probabilistic model.

## 3.4 Agarwal et al. and Jelinek–Mercer

Agarwal et al.'s hierarchical shrinkage for decision trees is the closest direct statistical algorithmic precedent.

Jelinek–Mercer interpolation provides another useful analogy:

$$
\hat p_k
=
\lambda_k\hat p_k^{\rm raw}
+
(1-\lambda_k)\hat p_{k-1}.
$$

Sieve's shrinkage is

$$
\tilde\mu_k
=
\frac{N_k}{N_k+\alpha_k}\bar y_k
+
\frac{\alpha_k}{N_k+\alpha_k}\tilde\mu_{k-1}.
$$

Hence

$$
\lambda_k=\frac{N_k}{N_k+\alpha_k}.
$$

This gives a useful conceptual split:

$$
\boxed{
\begin{array}{ccc}
\text{Katz-style backoff}
&\leftrightarrow&
\text{hard support gate}
\\
\text{Jelinek--Mercer interpolation}
&\leftrightarrow&
\text{soft ancestor shrinkage}
\end{array}}
$$

## 3.5 Why not fit a full multilevel model?

A full nested model would offer coherent partial pooling, variance-component estimation, and posterior uncertainty.

Sieve instead preserves:

- closed-form descriptive statistics;
- deterministic prediction;
- exact associative model merging;
- no global latent-variable inference;
- cheap evaluation.

This is a legitimate design tradeoff.

## 3.6 References to register

**[LindleySmith1972BayesLinearModel]**  
Lindley, D. V.; Smith, A. F. M. *Bayes Estimates for the Linear Model.* **Journal of the Royal Statistical Society, Series B** 34(1), 1–18 (1972).  
DOI: `10.1111/j.2517-6161.1972.tb00885.x`

**[EfronMorris1973EmpiricalBayes]**  
Efron, B.; Morris, C. *Stein's Estimation Rule and Its Competitors—An Empirical Bayes Approach.* **Journal of the American Statistical Association** 68, 117–130 (1973).  
DOI: `10.1080/01621459.1973.10481350`

**[Morris1983ParametricEB]**  
Morris, C. N. *Parametric Empirical Bayes Inference: Theory and Applications.* **Journal of the American Statistical Association** 78, 47–55 (1983).  
DOI: `10.1080/01621459.1983.10477920`

**[MorrisLysy2012Shrinkage]**  
Morris, C. N.; Lysy, M. *Shrinkage Estimation in Multilevel Normal Models.* **Statistical Science** 27, 115–134 (2012).  
DOI: `10.1214/11-STS363`

**[JelinekMercer1980Interpolation]**  
Jelinek, F.; Mercer, R. L. *Interpolated Estimation of Markov Source Parameters from Sparse Data.* In **Pattern Recognition in Practice**, 381–397 (1980).

**[Agarwal2022HierarchicalShrinkage]** remains the direct algorithmic precedent.

---

# 4. σ-profile prediction literature

## 4.1 Established application area

Fast COSMO/COSMO-SAC σ-profile prediction already includes:

1. database-fragment composition;
2. group-contribution models;
3. atom-contribution neural models;
4. Transformer models;
5. message-passing / graph-convolutional models;
6. substructure-hash/database systems.

Sieve should therefore not be positioned as introducing fast σ-profile prediction or atom-wise decomposition itself.

## 4.2 COSMOfrag

COSMOfrag constructs approximate σ-profiles by selecting structurally suitable fragments from a database of precomputed COSMO molecules and composing their partial σ-profiles.

Conceptually:

$$
\text{local structural fragment}
\rightarrow
\text{database lookup}
\rightarrow
\text{stored partial }\sigma\text{-profile}.
$$

This is an important application-specific structural-lookup precedent.

## 4.3 Group-contribution models

GC-COSMO methods estimate σ-profiles from additive group contributions:

$$
\hat P(\sigma)=\sum_gn_gP_g(\sigma).
$$

These provide the most natural low-complexity baseline for Sieve.

## 4.4 Liu et al. 2021: closest atom-level learned baseline

Liu et al. introduced a machine-learning atom-contribution model in which atom-centered descriptors are mapped to atomic σ-profile contributions and summed into molecular profiles.

This is a direct comparator:

### ML atom contribution

$$
\text{atomic environment}
\rightarrow
\text{learned atomic }\sigma\text{-profile}
$$

### Sieve

$$
\text{supported discrete environment}
\rightarrow
\text{empirical atomic }\sigma\text{-profile}.
$$

The distinction is a continuous learned mapping versus support-adaptive empirical conditional means over discrete structural classes.

## 4.5 Other learned σ-profile surrogates

Published models include:

- Transformer-CNN models;
- Transformer/k-mer models;
- MPNN-FNN models;
- graph-convolutional models.

Abranches et al. provides a particularly relevant graph baseline.

## 4.6 FastSigma SG1

FastSigma SG1 uses substructure hashing and database searching to build approximate σ-profiles from precomputed profiles.

It is important practical prior art and a useful external comparator, even if the underlying method is not represented by a clearly identified peer-reviewed algorithm paper.

## 4.7 Reference-profile protocol matters

σ-profiles depend on the reference quantum-chemical and segmentation protocol.

Therefore benchmark comparisons should use

$$
\boxed{
\text{same molecules}
+
\text{same reference profiles}
+
\text{same split}
+
\text{retrained baselines}
}
$$

rather than comparing headline errors across incompatible datasets.

## 4.8 Scientific positioning

A useful application-specific spectrum is:

$$
\text{group contribution}
\rightarrow
\boxed{\text{Sieve}}
\rightarrow
\text{learned atom/GNN models}.
$$

The scientific question becomes:

> **How much of the predictive value of learned graph representations for atomic σ-profile prediction can be recovered by sufficiently specific, statistically supported discrete local environments?**

## 4.9 Recommended baselines

At minimum:

1. simple group/atom contribution;
2. fixed-depth or ungated Sieve;
3. full Sieve;
4. atom-level learned model analogous to Liu et al.;
5. GNN trained on the same data and split.

## 4.10 References to register

**[HornigKlamt2005COSMOfrag]**  
Hornig, M.; Klamt, A. *COSMOfrag: A Novel Tool for High-Throughput ADME Property Prediction and Similarity Screening Based on Quantum Chemistry.* **Journal of Chemical Information and Modeling** 45, 1169–1177 (2005).  
DOI: `10.1021/ci0501948`

**[Mu2007GCCOSMORS]**  
Mu, T.; Rarey, J.; Gmehling, J. Group-contribution COSMO-RS σ-profile work (2007).  
DOI: `10.1002/aic.11338`

**[Mu2009GCCOSMOSAC]**  
Mu, T.; Rarey, J.; Gmehling, J. Group-contribution COSMO-SAC σ-profile work (2009).  
DOI: `10.1002/aic.11933`

**[Liu2021MLAtomContribution]**  
Liu et al. *Machine Learning-Based Atom Contribution Method for the Prediction of Surface Charge Density Profiles and Solvent Design.* **AIChE Journal** 67, e17110 (2021).  
DOI: `10.1002/aic.17110`

**[Chen2021TransformerSigma]**  
Chen, Song, Qi. Transformer-CNN σ-profile surrogate work (2021).  
DOI: `10.1016/j.ces.2021.117002`

**[Kang2022TransformerSigma]**  
Kang et al. Transformer/k-mer σ-profile surrogate work (2022).  
DOI: `10.1016/j.dche.2022.100016`

**[Zhang2022MPNNFNN]**  
Zhang, Wang, Shen. MPNN-FNN σ-profile prediction work (2022).  
DOI: `10.1016/j.ces.2022.117624`

**[Abranches2023SigmaGCN]**  
Abranches, Maginn, Colón. Graph-convolutional σ-profile prediction work (2023).  
DOI: `10.1021/acs.jctc.3c01003`

**[Salih2025OpenSPGen]**  
Salih et al. σ-profile generation protocol / OpenSPGen work (2025).  
DOI: `10.1039/D5DD00087D`

**[Gond2026CHAOS]**  
Gond et al. CHAOS σ-profile database (2026).  
DOI: `10.1021/acs.jcim.6c00058`

---

# 5. Vector targets, functional responses, and transport geometry

## 5.1 The point estimator does not assume independent bins

For a vector target

$$
Y\in\mathbb R^d,
$$

under squared Euclidean loss

$$
L(a)=E[\|Y-a\|_2^2\mid C],
$$

the optimal class predictor is

$$
a^\star=E[Y\mid C].
$$

The empirical estimator is

$$
\hat a^\star=\frac1{N_C}\sum_{i\in C}Y_i.
$$

Computing this componentwise does **not** assume independence among components.

The correct wording is:

> **Sieve estimates a vector/function conditional mean under an $L^2$ geometry but stores only marginal second moments, not the full cross-component covariance.**

## 5.2 Functional-response interpretation

A σ-profile on a shared grid is naturally a discretized function $P(\sigma)$.

Squared vector error is proportional to an approximation of integrated squared error:

$$
\sum_j[P(\sigma_j)-Q(\sigma_j)]^2
\propto
\int[P(\sigma)-Q(\sigma)]^2d\sigma.
$$

Thus the class-average profile is the empirical $L^2$ Fréchet mean function.

## 5.3 σ-profiles are unnormalized positive histograms

A σ-profile is an unnormalized surface-area histogram.

Its total mass

$$
A=\sum_jP_j
$$

is physically meaningful.

Therefore normalized-density and compositional methods are not directly equivalent; normalization would remove scale unless total area were modeled separately.

## 5.4 Wasserstein geometry defines a different estimand

Classical Wasserstein distance assumes equal-mass probability measures.

For σ-profiles with variable total area, a transport formulation would require either:

- normalization plus a separate area model; or
- unbalanced optimal transport.

Arithmetic and Wasserstein means answer different questions:

### Arithmetic/L² mean

> expected surface area in each σ bin.

### Wasserstein barycenter

> geometrically central profile when differences are interpreted as mass transport along the σ axis.

Sieve currently estimates the first quantity.

## 5.5 Additivity favors the arithmetic mean

Atomic profiles add:

$$
P_{\mathrm{mol}}(\sigma)
=
\sum_vP_v(\sigma).
$$

Expectation is linear:

$$
E[P_{\mathrm{mol}}]
=
\sum_vE[P_v].
$$

The arithmetic mean also preserves:

- nonnegativity;
- average total area;
- exact mergeability;
- atomic additivity;
- linear profile moments.

This is a strong application-specific justification for the current design.

## 5.6 Cross-bin covariance

Not storing the full covariance matrix does not bias the class mean.

It limits:

- multivariate uncertainty;
- covariance-aware diagnostics;
- coordinated shape-fluctuation modeling.

A low-rank covariance or functional-PCA extension could be considered later if needed.

## 5.7 Weighted quadratic losses

For any fixed positive-definite matrix $W$,

$$
L(a)=E[(Y-a)^TW(Y-a)],
$$

the minimizer remains

$$
a^\star=E[Y].
$$

Thus many shape-aware quadratic losses or linear moment penalties can be used without changing the class mean.

## 5.8 Downstream thermodynamic loss

If the downstream property is nonlinear,

$$
\gamma=F(P),
$$

then minimizing profile-space $L^2$ error need not minimize downstream property error.

Therefore evaluation should include:

1. profile error;
2. downstream COSMO-RS/COSMO-SAC property error.

## 5.9 References to register

**[Morris2015FunctionalRegression]**  
Morris, J. S. *Functional Regression.* **Annual Review of Statistics and Its Application** 2, 321–359 (2015).  
DOI: `10.1146/annurev-statistics-010814-020413`

**[PetersenMuller2019FrechetRegression]**  
Petersen, A.; Müller, H.-G. *Fréchet Regression for Random Objects with Euclidean Predictors.* **The Annals of Statistics** 47, 691–719 (2019).  
DOI: `10.1214/17-AOS1624`

**[Talska2018CompositionalFunctional]**  
Talská, R.; Menafoglio, A.; Machalová, J.; Hron, K.; Fišerová, E. *Compositional Regression with Functional Response.* **Computational Statistics & Data Analysis** 123, 66–85 (2018).  
DOI: `10.1016/j.csda.2018.01.018`

**[AguehCarlier2011WassersteinBarycenters]**  
Agueh, M.; Carlier, G. *Barycenters in the Wasserstein Space.* **SIAM Journal on Mathematical Analysis** 43, 904–924 (2011).  
DOI: `10.1137/100805741`

**[Chizat2018UnbalancedOT]**  
Chizat, L.; Peyré, G.; Schmitzer, B.; Vialard, F.-X. *Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations.* **Journal of Functional Analysis** 274, 3090–3123 (2018).  
DOI: `10.1016/j.jfa.2018.03.008`

Cross-reference **[Kang2022TransformerSigma]** here because it uses profile-moment-aware losses.

---

# 6. Revised overall novelty position

After closing items 1–5, the following are established prior art:

- partitioning regression by empirical cell means;
- local averaging and adaptive neighborhoods;
- hierarchical structural classes in chemistry;
- structural parameter lookup by specificity;
- partial pooling and hierarchical shrinkage;
- hard and soft sparse-context backoff;
- atom-wise σ-profile decomposition;
- learned σ-profile surrogates;
- functional/vector conditional means;
- alternative Fréchet geometries.

What still appears distinctive is the combination:

$$
\boxed{
\begin{array}{c}
\text{deterministic nested vertex-environment refinement}
\\ + \\
\text{empirical vector-valued class statistics}
\\ + \\
\text{support-gated selection of resolution}
\\ + \\
\text{deterministic ancestor backoff}
\\ + \\
\text{exactly mergeable fitted state}
\end{array}}
$$

used as a domain-agnostic node-regression framework.

A revised one-line description is:

> **Sieve is a support-adaptive local-averaging estimator over a deterministic nested hierarchy of vertex environments. At each level it is a partitioning regressor; prediction selects the most specific statistically supported class and optionally shrinks its empirical target mean toward ancestors.**

---

# 7. Recommended manuscript consequences

## 7.1 Prefer this method identity

> **support-adaptive partitioning regression over a nested refinement hierarchy**

rather than:

> "lookup table over WL hashes."

## 7.2 Do not claim novelty for

- hierarchical structural lookup;
- atom-environment regression;
- support/backoff as a generic statistical mechanism;
- ancestor shrinkage;
- vector-valued atom-property prediction;
- atom-wise σ-profile decomposition;
- WL/ECFP-like molecular environment identifiers.

## 7.3 Emphasize instead

1. deterministic refinement-generated hierarchy;
2. support-gated resolution selection;
3. empirical conditional statistics rather than learned embeddings;
4. transparent per-class diagnostics;
5. exact compositional merge of fitted state;
6. domain-agnostic graph formulation;
7. natural closure under positive vector targets such as σ-profiles.

---

# 8. Gap status

The following can now be marked **closed as literature-search gaps**:

- §9.1 partitioning / regressogram framing;
- §9.2 force-field atom typing;
- §9.3 multilevel models and partial pooling;
- §9.4 σ-profile prediction;
- §9.5 vector targets / functional response.

What remains is manuscript integration:

1. certify the newly added references;
2. insert the new related-work subsections;
3. decide which application baselines to reproduce;
4. decide whether to develop a formal asymptotic consistency result;
5. keep the current $L^2$/vector estimator unless downstream experiments motivate a different target geometry.
