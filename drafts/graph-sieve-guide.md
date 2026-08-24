# Graph-Sieve — A Practical Guide to Graph-Level Extension

**Status:** design guide / working specification  
**Purpose:** define a principled graph-level extension of Sieve when only graph-level labels are available  
**Core idea:** progressively infer local latent corrections from graph-level residuals, reconcile them under the graph constraint, and summarize the resulting node-level posterior moments with the same mean/MSD machinery used by Sieve

---

# 1. Problem statement

Original Sieve assumes node-level supervision.

For node $v$, at refinement level $k$,

$$
h_k(v)=c
$$

defines a structural class, and the model can directly estimate

$$
E[Y_v\mid h_k(v)=c]
$$

from observed node targets.

For graph-level prediction, only

$$
Y_G
$$

is observed for the entire graph $G$. Individual node contributions are latent.

The goal is to preserve as much of the original Sieve structure as possible:

- deterministic nested refinement;
- class-specific means;
- class-specific dispersion;
- support awareness;
- progressive specificity;
- interpretable local contributions;
- simple fitting and inference.

The proposed solution is to infer **latent local contributions from aggregate graph labels**.

---

# 2. Central modeling assumption

For an extensive or approximately additive graph property, assume

$$
Y_G
=
b+
\sum_{v\in V(G)} Z_{Gv}
+
\epsilon_G,
$$

where $b$ is an optional graph-level intercept, $Z_{Gv}$ is the latent contribution of node $v$, and $\epsilon_G$ captures graph-level noise or genuinely nonlocal effects.

For an exactly additive property,

$$
\epsilon_G=0.
$$

For an approximately additive property,

$$
\epsilon_G\sim\mathcal N(0,\tau^2).
$$

The key problem is therefore:

> infer transferable local contributions $Z_{Gv}$ from observed graph totals $Y_G$.

---

# 3. Why whole-graph hashing is not the preferred extension

One could define increasingly specific whole-graph signatures and run ordinary Sieve on them:

$$
g_k(G)
=
H\left(
g_{k-1}(G),
\operatorname{MULTISET}\{h_k(v):v\in G\}
\right).
$$

Then one could estimate

$$
E[Y_G\mid g_k(G)].
$$

This preserves the lookup logic but is likely to suffer severe support collapse because exact molecular graph signatures become nearly unique very quickly.

The preferred approach is instead to retain **local structural classes** and infer their contributions from aggregate labels.

---

# 4. Progressive graph-level Sieve

The preferred formulation combines two principles:

1. freeze what has already been learned at coarser levels;
2. fit only the residual contribution introduced by the next refinement level.

After levels $0,\ldots,k-1$, let the cumulative local contribution assigned to node $v$ be

$$
A_{Gv}^{(<k)}.
$$

The current graph prediction is

$$
\hat Y_G^{(<k)}
=
b+
\sum_{v\in G}A_{Gv}^{(<k)}.
$$

Define the remaining graph residual

$$
R_G^{(k)}
=
Y_G-\hat Y_G^{(<k)}.
$$

At refinement level $k$, introduce a latent node correction

$$
D_{Gv}^{(k)}.
$$

The residual model is

$$
\boxed{
R_G^{(k)}
=
\sum_{v\in G}D_{Gv}^{(k)}
+
\epsilon_G^{(k)}.
}
$$

Each correction is associated with the level-$k$ Sieve class

$$
c=h_k(v).
$$

The local prior model is

$$
\boxed{
D_{Gv}^{(k)}
\mid h_k(v)=c
\sim
\mathcal N(
\delta_{k,c},
s_{k,c}^2
).
}
$$

Thus each level learns only what that structural refinement explains beyond all previous levels.

---

# 5. Interpretation of the correction hierarchy

The final local contribution becomes

$$
\boxed{
A_{Gv}^{(K)}
=
\beta_0[h_0(v)]
+
\sum_{k=1}^{K}
\delta_{k,h_k(v)}.
}
$$

This gives every level a precise meaning:

$$
\delta_{k,c}
=
\text{expected correction associated with the additional structural information at level }k.
$$

The final graph prediction is

$$
\boxed{
\hat Y_G
=
b+
\sum_{v\in G}
\left[
\beta_0[h_0(v)]
+
\sum_{k=1}^{K}
\delta_{k,h_k(v)}
\right].
}
$$

This is the graph-level analogue of the Sieve philosophy:

> start with the broadest structural explanation and introduce finer corrections only when the data support them.

---

# 6. Pseudo-target generation as data reconciliation

At a fixed level $k$, suppose the current prior correction for node $i$ is

$$
D_i
\sim
\mathcal N(\delta_i,v_i),
$$

where

$$
\delta_i=\delta_{k,h_k(i)},
\qquad
v_i=s_{k,h_k(i)}^2.
$$

For graph $G$, the observed residual is

$$
R_G.
$$

The latent corrections must collectively explain that residual:

$$
R_G
=
\sum_{i\in G}D_i+\epsilon_G,
\qquad
\epsilon_G\sim\mathcal N(0,\tau_k^2).
$$

Conditioning on $R_G$ gives the reconciled posterior mean

$$
\boxed{
m_i
=
E[D_i\mid R_G]
=
\delta_i
+
\frac{v_i}{
\tau_k^2+\sum_{j\in G}v_j
}
\left(
R_G-\sum_{j\in G}\delta_j
\right).
}
$$

The posterior variance is

$$
\boxed{
q_i
=
\operatorname{Var}(D_i\mid R_G)
=
v_i-
\frac{v_i^2}{
\tau_k^2+\sum_{j\in G}v_j
}.
}
$$

For two nodes $i\neq j$ in the same graph,

$$
\operatorname{Cov}(D_i,D_j\mid R_G)
=
-
\frac{v_i v_j}{
\tau_k^2+\sum_{\ell\in G}v_\ell
}.
$$

The graph constraint therefore induces negative posterior correlations among local contributions.

---

# 7. Hard versus soft graph reconciliation

## 7.1 Exact additive constraint

If the property is exactly additive, set

$$
\tau_k^2=0.
$$

Then

$$
\sum_{i\in G}m_i=R_G.
$$

The graph residual is redistributed completely across the nodes.

The correction is

$$
m_i-\delta_i
=
\frac{v_i}{\sum_jv_j}
\left(
R_G-\sum_j\delta_j
\right).
$$

Nodes with larger uncertainty absorb more of the discrepancy.

## 7.2 Approximate additive constraint

If nonlocal effects or observation noise are expected, use

$$
\tau_k^2>0.
$$

Then only part of the discrepancy is allocated locally:

$$
\sum_i(m_i-\delta_i)
=
\frac{
\sum_i v_i
}{
\tau_k^2+\sum_i v_i
}
\left(
R_G-\sum_i\delta_i
\right).
$$

The remaining discrepancy is attributed to graph-level noise/nonlocal structure.

---

# 8. MaxEnt / minimum-information interpretation

The same reconciliation step can be interpreted through maximum relative entropy.

Let the current independent class beliefs define

$$
q(\mathbf D)
=
\prod_i
\mathcal N(D_i;\delta_i,v_i).
$$

Seek a reconciled distribution $p$ that satisfies the graph information while deviating minimally from $q$:

$$
p^\star
=
\arg\min_p
D_{\mathrm{KL}}(p\|q)
$$

subject to the graph-level constraint.

For Gaussian priors and a linear sum constraint, the solution is the same conditional Gaussian update given above.

Thus the pseudo-targets are the **minimum-information update of the current class-level beliefs required to reconcile them with the observed graph label**.

For the hard constraint and equal variances,

$$
v_i=v,
$$

the update reduces to equal redistribution:

$$
m_i
=
\delta_i+
\frac{1}{|G|}
\left(
R_G-\sum_j\delta_j
\right).
$$

---

# 9. EM interpretation

The graph-level problem is a latent-variable model.

The latent variables are node corrections $D_i$.  
The observed data are graph residuals $R_G$.

This gives a natural EM algorithm.

## E-step

For every graph, compute posterior moments

$$
m_i
=
E[D_i\mid R_G]
$$

and

$$
q_i
=
\operatorname{Var}(D_i\mid R_G).
$$

These are the reconciled node-level pseudo-target moments.

## M-step

For every supported class $c$, aggregate the node posterior moments.

The updated class mean is

$$
\boxed{
\delta_c^{\text{new}}
=
\frac{1}{N_c}
\sum_{i:c_i=c}m_i.
}
$$

The updated class second central moment is

$$
\boxed{
(s_c^2)^{\text{new}}
=
\frac{1}{N_c}
\sum_{i:c_i=c}
\left[
q_i+
(m_i-\delta_c^{\text{new}})^2
\right].
}
$$

This is the latent-variable analogue of the Sieve mean/MSD update.

The first term captures uncertainty remaining within the inferred node correction.  
The second captures variation of reconciled corrections across occurrences of the class.

---

# 10. Why the posterior-variance term matters

If one computes only

$$
\frac1{N_c}
\sum_i
(m_i-\bar m_c)^2,
$$

the result measures variation among posterior means but ignores uncertainty remaining in latent contributions.

The full EM update uses

$$
E[D_i^2\mid R_G]
=
q_i+m_i^2.
$$

Therefore

$$
s_c^2
=
E[D^2\mid c]
-
E[D\mid c]^2
$$

is estimated using both between-occurrence variability and posterior uncertainty.

---

# 11. Level-by-level training

The recommended training strategy is progressive.

## Level 0

Fit the coarsest local contribution model.

Depending on the application, this may be element-level contributions, graded atomic attributes, or another deliberately coarse base partition.

Fit the base model to convergence, then freeze it.

## Level $k>0$

1. Compute graph residuals from all frozen lower levels:

$$
R_G^{(k)}
=
Y_G-
\hat Y_G^{(<k)}.
$$

2. Define level-$k$ classes:

$$
c=h_k(v).
$$

3. Initialize new corrections:

$$
\delta_{k,c}=0.
$$

4. Initialize class correction variances from a pooled depth-level value, parent-derived value, or another stable estimate.

5. Run the E-step/M-step reconciliation loop at level $k$.

6. Freeze the converged correction statistics.

7. Advance to level $k+1$.

The lower-level contributions never need to be reoptimized.

---

# 12. Why progressive fitting is attractive

Progressive fitting avoids one enormous regression over all hierarchy levels.

It also gives a direct interpretation:

$$
\boxed{
\text{level }k
=
\text{predictive information newly available at refinement }k.
}
$$

Useful diagnostics include:

$$
\text{validation error versus }k,
$$

$$
\|\delta_k\|
\text{ versus }k,
$$

and

$$
s_{k,c}^2
\text{ versus support/depth}.
$$

---

# 13. Parent-centered corrections

Because refinement is nested, children of a parent can collectively reproduce a parent-level effect.

To preserve the interpretation that level-$k$ terms are genuine refinements, one may impose

$$
\boxed{
\sum_{c:p(c)=p}
N_c\delta_{k,c}=0
}
$$

for each parent $p$.

This makes child corrections a support-weighted contrast around their parent.

Two variants should be tested:

### Variant A — unconstrained progressive corrections

Fit residual corrections directly.

Advantages:

- simplest EM formulation;
- lets deeper levels repair systematic lower-level bias.

### Variant B — parent-centered corrections

Impose

$$
\sum_cN_c\delta_c=0
$$

within each parent.

Advantages:

- strongest interpretability;
- prevents deeper levels from globally refitting parent effects;
- produces an ANOVA-like decomposition.

The first implementation should ideally support both.

---

# 14. Support handling

For class $c$,

$$
N_c
=
\text{number of training occurrences of }c.
$$

A natural support policy is

$$
N_c<n_{\min}
\Rightarrow
\delta_{k,c}=0.
$$

An unsupported child therefore contributes no persistent refinement beyond its ancestors.

There is an important distinction between the **persistent class mean** and the **latent per-graph correction**.

Even an unsupported class may temporarily absorb some graph residual during reconciliation if it is assigned nonzero latent variance. Its persistent expected correction can still remain zero.

This gives a coherent random-effects interpretation:

- supported classes can learn transferable nonzero corrections;
- unsupported classes can represent uncertainty but do not acquire persistent class means.

A simpler v1 can instead set unsupported correction variance to zero and let $\tau^2$ absorb unexplained residual.

---

# 15. Regularization

Deeper refinement levels generally have more classes, less support, greater identifiability risk, and greater overfitting risk.

Possible approaches include:

### Hard support gating

$$
N_c<n_{\min}
\Rightarrow
\delta_c=0.
$$

### Ridge shrinkage

Add

$$
\lambda_k
\sum_c\delta_{k,c}^2.
$$

### Hierarchical shrinkage

Shrink child corrections toward zero or toward a parent-defined expectation.

### Variance floors / pooling

Use a variance floor or shrink unstable class variances toward a depth-level pooled variance.

For the first implementation, the recommended order of complexity is:

1. hard support threshold;
2. pooled variance initialization;
3. optional ridge;
4. later, hierarchical shrinkage.

---

# 16. Initialization

## Base level

Fit a simple additive model directly from graph labels:

$$
Y_G
\approx
b+
\sum_c n_{G,0,c}\beta_{0,c}.
$$

Ridge regression is a reasonable initializer.

## Refinement levels

Initialize

$$
\delta_{k,c}=0.
$$

Initialize

$$
s_{k,c}^2=s_{k,\mathrm{pool}}^2
$$

using a pooled residual-based variance.

This encodes the prior assumption that finer structural distinctions initially have no systematic effect but may explain residual variation if the data support them.

---

# 17. Convergence at a level

At each level, iterate E and M steps until one of:

- relative change in graph likelihood is below tolerance;
- change in class means is below tolerance;
- change in class second moments is below tolerance;
- a fixed small number of iterations is reached and validation performance is stable.

Because the Gaussian E-step is analytic, each iteration should be inexpensive.

---

# 18. Stopping refinement depth

After level $k$, evaluate held-out performance.

Stop if deeper refinement no longer improves validation performance.

The selected depth has an interpretable meaning:

> the structural radius beyond which additional local graph information is not supported by predictive gains.

---

# 19. Prediction on a new graph

For a new graph, no reconciliation is possible because its true graph label is unknown.

Prediction uses only learned persistent class means.

For each node,

$$
\hat A_v
=
\beta_{0,h_0(v)}
+
\sum_{k=1}^{K}
\hat\delta_{k,h_k(v)}.
$$

Unsupported or unseen classes contribute zero at that refinement level, so the node automatically inherits its coarser contribution.

Then

$$
\boxed{
\hat Y_G
=
b+\sum_v\hat A_v.
}
$$

Thus reconciliation is a **training-time latent-target inference mechanism**, not an inference-time correction.

---

# 20. Intensive graph properties

For an intensive property, sum pooling may be inappropriate.

A simple alternative is

$$
Y_G
=
b+
\frac{1}{|V(G)|}
\sum_v Z_v
+
\epsilon_G.
$$

More generally,

$$
Y_G
=
b+
\sum_v w_{Gv}Z_v
+
\epsilon_G
$$

for known weights $w_{Gv}$.

The Gaussian reconciliation derivation remains valid for a general linear aggregate.

---

# 21. Vector-valued graph labels

For

$$
Y_G\in\mathbb R^d,
$$

the simplest implementation treats each target dimension with the same structural classes.

Each class stores

$$
\boldsymbol\delta_c\in\mathbb R^d
$$

and per-dimension second moments.

The reconciliation can initially be performed componentwise.

For molecular σ-profiles,

$$
\hat{\mathbf P}_G
=
\sum_v
\left[
\boldsymbol\beta_{0,h_0(v)}
+
\sum_k
\boldsymbol\delta_{k,h_k(v)}
\right].
$$

This makes it possible to infer latent atomic σ-profile contributions using only molecular σ-profile supervision.

---

# 22. Node + edge contributions

The additive model can be extended to both node and edge terms:

$$
Y_G
=
b
+
\sum_v Z_v^{\mathrm{node}}
+
\sum_e Z_e^{\mathrm{edge}}
+
\epsilon_G.
$$

Node and edge contributions can each have their own Sieve refinement hierarchy.

This should be treated as a later extension rather than part of the first implementation.

---

# 23. Identifiability

Graph-level labels do not generally identify a unique local decomposition.

If two local environments always co-occur, their contributions may trade off without changing graph predictions.

In matrix form,

$$
X\beta=Y.
$$

Any perturbation

$$
\beta\rightarrow\beta+\Delta
$$

satisfying

$$
X\Delta=0
$$

is observationally indistinguishable at the graph level.

Graph-Sieve introduces structural biases that help select a useful representative:

- deterministic local equivalence classes;
- progressive refinement;
- zero initialization of deeper corrections;
- support gating;
- optional parent-centering;
- regularization;
- minimum-information reconciliation.

The inferred node contributions should therefore be interpreted as **model-dependent latent attributions**, not uniquely observable physical quantities unless the application provides additional justification.

---

# 24. Relationship to nearby literature

## Learning from aggregate outputs

The generic problem is

$$
Y_G=\sum_{i\in G}Y_i
$$

with only $Y_G$ observed.

This is the direct statistical neighborhood for graph-level Sieve.

## Atomistic machine-learning potentials

Behler–Parrinello-type models use

$$
E_{\mathrm{total}}
=
\sum_iE_i(\text{local environment}_i)
$$

while training only from total energies.

This is a canonical chemical example of latent local contributions learned from global labels.

## Posterior regularization / MaxEnt

Graph-Sieve's reconciliation step is a minimum-KL update of latent local beliefs under a global aggregate constraint.

## Data reconciliation

The Gaussian correction is mathematically equivalent to covariance-weighted adjustment under a linear balance constraint.

## Information-theoretic molecular partitioning

Hirshfeld-like approaches provide a conceptual precedent for minimally deforming transferable atomic references while reproducing a molecular aggregate.

---

# 25. Recommended v1 algorithm

A practical first implementation should be deliberately simple.

## Inputs

- training graphs $G_1,\ldots,G_M$;
- scalar graph labels $Y_G$;
- Sieve refinement levels $h_0,\ldots,h_K$;
- support threshold $n_{\min}$;
- graph noise $\tau^2$, fixed or tuned.

## Step 1 — fit base level

Fit

$$
Y_G
\approx
b+
\sum_v\beta_{0,h_0(v)}.
$$

Store base means and a stable variance estimate.

## Step 2 — for each refinement level $k=1,\ldots,K$

Compute

$$
R_G^{(k)}
=
Y_G-
\hat Y_G^{(<k)}.
$$

Initialize

$$
\delta_{k,c}=0.
$$

Initialize class variances from a pooled depth-level value.

Repeat until convergence:

### E-step

For every graph,

$$
m_i
=
\delta_i+
\frac{v_i}{
\tau^2+\sum_jv_j
}
\left(
R_G-\sum_j\delta_j
\right),
$$

$$
q_i
=
v_i-
\frac{v_i^2}{
\tau^2+\sum_jv_j
}.
$$

### M-step

For every supported class,

$$
\delta_c
=
\frac1{N_c}\sum_{i\in c}m_i,
$$

$$
s_c^2
=
\frac1{N_c}
\sum_{i\in c}
\left[
q_i+(m_i-\delta_c)^2
\right].
$$

Apply variance floors/shrinkage as needed.

Optionally apply the parent-centering constraint.

Freeze the level when converged.

## Step 3 — validate depth

Measure held-out graph-level error after every level.

Stop when refinement ceases to improve validation performance.

---

# 26. Pseudocode

```text
fit_graph_sieve(graphs, labels, levels, n_min, tau2):

    base = fit_base_additive_model(graphs, labels, level=0)
    frozen_levels = [base]

    for k in 1..K:

        residual[G] = label[G] - predict(graph[G], frozen_levels)

        classes = refinement_classes(graphs, level=k)

        delta[c] = 0
        variance[c] = pooled_initial_variance(k)

        repeat until convergence:

            # E-step: graph-wise reconciliation
            for G in graphs:

                prior_mean[i] = delta[class_k(i)]
                prior_var[i]  = effective_variance(class_k(i))

                denom = tau2 + sum_i prior_var[i]

                discrepancy =
                    residual[G] - sum_i prior_mean[i]

                post_mean[i] =
                    prior_mean[i]
                    + prior_var[i] / denom * discrepancy

                post_var[i] =
                    prior_var[i]
                    - prior_var[i]^2 / denom

            # M-step: Sieve aggregation
            for class c:

                if support[c] < n_min:
                    delta[c] = 0
                    continue

                delta[c] =
                    mean(post_mean[i] for i in c)

                variance[c] =
                    mean(
                        post_var[i]
                        + (post_mean[i] - delta[c])^2
                        for i in c
                    )

            optional_parent_center(delta)
            stabilize_variances(variance)

        freeze(delta, variance)
        frozen_levels.append(level_k_model)

        if validation_no_longer_improves:
            break

    return frozen_levels
```

---

# 27. Diagnostics to implement from the beginning

## Per level

- number of classes;
- fraction of supported classes;
- number of active correction parameters;
- correction mean distribution;
- correction MSD distribution;
- train/validation graph error;
- improvement relative to previous level.

## Per class

- support $N_c$;
- correction mean $\delta_c$;
- correction MSD $s_c^2$;
- parent class;
- depth;
- fraction of graphs containing the class.

## Per graph

- prediction before reconciliation;
- residual before reconciliation;
- fraction of residual assigned locally;
- posterior correction magnitude;
- graph uncertainty proxy.

## Global

- error versus refinement depth;
- error versus minimum support;
- error versus class uncertainty;
- performance under random versus OOD splits.

---

# 28. Critical experiments

## Synthetic recoverability

Generate graphs with known latent class contributions and test recovery of graph labels, local contributions, class means, and class variances.

Vary co-occurrence structure to probe identifiability.

## Exact additive case

Set

$$
\tau^2=0
$$

and verify exact residual reconciliation.

## Approximate additive case

Inject graph-level noise/nonlocal effects and verify that nonzero $\tau^2$ prevents local contributions from absorbing all model error.

## Hierarchy recovery

Create a synthetic target where only a known refinement depth matters and verify that improvement appears at that depth.

## Support stress test

Hold out rare classes and confirm that unsupported refinements revert to zero correction.

## Correction-scheme comparison

Compare equal redistribution, variance-weighted reconciliation, full Gaussian EM, direct graph-level ridge, and parent-centered versus unconstrained corrections.

---

# 29. Baselines

At minimum compare against:

1. constant graph baseline;
2. fixed atom/group contributions;
3. one-shot linear regression on environment counts;
4. ridge regression on WL/environment-count features;
5. progressively fitted corrections without reconciliation;
6. Graph-Sieve EM/reconciliation;
7. a GNN with sum pooling.

---

# 30. Open theoretical questions

The most important unresolved questions are:

1. Under what graph-collection conditions are class contributions identifiable?
2. Does minimum-information reconciliation select a useful canonical representative among equivalent decompositions?
3. When does progressive freezing outperform or approximate a joint optimum?
4. Should parent-centered corrections be imposed?
5. Should support count node occurrences, graphs containing the class, or an effective independent count?
6. How should posterior uncertainty, context variability, and class heterogeneity be separated?
7. How should $\tau^2$ be estimated?
8. Which non-Gaussian latent families are useful for positive or count-valued properties?
9. Which parts of the fitted state remain exactly mergeable?
10. Can support, depth, and class MSD provide calibrated OOD/applicability-domain signals?

---

# 31. Literature anchors

The current conceptual neighborhood includes:

- **Musicant, Christensen, Olson** — supervised learning from aggregate outputs;
- **Law et al. (NeurIPS 2018)** — variational learning from aggregate outputs;
- **Zhang et al.** — learning from aggregate observations and identifiability;
- **Ganchev et al. (JMLR 2010)** — posterior regularization;
- **Behler & Parrinello (PRL 2007)** — latent atomic energy contributions from total energies;
- **Yoo et al.** — non-identifiability / arbitrariness of learned atomic energy decompositions;
- classical **data reconciliation** literature;
- information-theoretic and variational **Hirshfeld partitioning**.

---

# 32. Working conceptual definition

> **Graph-Sieve is a progressive latent-contribution model for graph-level supervision. Each refinement level introduces class-specific local corrections to a frozen coarser prediction. Graph labels reconcile the latent node corrections through a minimum-information Gaussian update, and the resulting posterior node moments are aggregated into Sieve-style class means and MSDs.**

A compact algorithmic summary is

$$
\boxed{
\text{refine}
\rightarrow
\text{reconcile}
\rightarrow
\text{aggregate}
\rightarrow
\text{freeze}
\rightarrow
\text{refine again}.
}
$$

---

# 33. Recommended next implementation milestone

Before adding edge terms, vector covariances, non-Gaussian distributions, or sophisticated regularization, implement the scalar Gaussian extensive-property case.

The first prototype should answer:

> **Can progressive structural refinement plus graph-wise latent reconciliation recover transferable local contributions from graph totals?**

If the answer is yes, the framework has a strong foundation for a broader Graph-Sieve family.
