# Sieve for Graph-Level Prediction — Converged Design So Far

**Status:** brainstorming / design note  
**Purpose:** capture the graph-level extension of Sieve that has emerged so far, before considering additional ideas  
**Scope:** graph-level labels only, with latent local contributions learned progressively over the Sieve refinement hierarchy

---

# 1. Motivation

The original Sieve setting is naturally supervised at the node level:

$$
\text{local environment}
\rightarrow
\text{node target}.
$$

For a node $v$, Sieve can directly estimate

$$
E[y_v \mid h_k(v)=c].
$$

The graph-level setting is different because only one label $y_G$ is observed for the whole graph. There is no directly observed node target $y_v$.

A literal extension based on hashing the entire graph at progressively deeper levels is possible, but unattractive: exact whole-graph signatures become unique very quickly, causing severe support collapse.

The more compelling extension is therefore to treat local contributions as **latent variables** whose sum or average explains the graph-level target.

---

# 2. Core graph-level model

Assume the graph-level target can be represented approximately as a sum of local contributions:

$$
y_G \approx b+\sum_{v\in V(G)} f(v).
$$

The local contribution is not observed directly.

Instead, it is parameterized through the same nested local-environment hierarchy used by Sieve.

At a fixed refinement level $k$, let

$$
h_k(v)=c
$$

denote the class of node $v$.

A simple fixed-level model is

$$
\hat y_G
=
b+\sum_{v\in V(G)}\beta_{k,h_k(v)}.
$$

Equivalently, defining

$$
n_{G,k,c}
=
\left|
\{v\in V(G):h_k(v)=c\}
\right|,
$$

we obtain

$$
\hat y_G
=
b+\sum_c n_{G,k,c}\beta_{k,c}.
$$

Thus graph-level labels provide linear constraints on latent local contributions.

---

# 3. Why not fit all refinement levels jointly?

A joint model over all hierarchy levels would be possible, but it would have several drawbacks:

- strong multicollinearity across nested levels;
- reduced interpretability;
- deeper classes could implicitly refit effects already represented at coarse levels;
- the meaning of each level-specific coefficient would become ambiguous;
- optimization would become one large coupled regression problem.

The preferred design is therefore **progressive optimization over refinement levels**.

---

# 4. Progressive residual fitting

Start with the coarsest model:

$$
\hat y_G^{(0)}
=
b+\sum_c n_{G,0,c}\beta_{0,c}.
$$

Fit $b,\beta_0$ from the graph-level labels, then freeze them.

At level $k$, define the residual left by all previously fitted levels:

$$
r_G^{(k-1)}
=
y_G-\hat y_G^{(k-1)}.
$$

Fit only the new corrections $\Delta_{k,c}$:

$$
\Delta_k^\star
=
\arg\min_{\Delta_k}
\sum_G
\left[
r_G^{(k-1)}
-
\sum_c n_{G,k,c}\Delta_{k,c}
\right]^2
+
\lambda_k R(\Delta_k).
$$

Then update

$$
\hat y_G^{(k)}
=
\hat y_G^{(k-1)}
+
\sum_c n_{G,k,c}\Delta_{k,c}.
$$

The previous parameters remain fixed.

This gives each level a clean interpretation:

$$
\boxed{
\Delta_k
=
\text{what refinement level }k
\text{ explains beyond levels }0,\ldots,k-1.
}
$$

---

# 5. Final local contribution model

After $K$ levels, the latent contribution of node $v$ is

$$
f(v)
=
\beta_0[h_0(v)]
+
\sum_{k=1}^{K}
\Delta_k[h_k(v)].
$$

The graph prediction becomes

$$
\boxed{
\hat y_G
=
b+
\sum_{v\in V(G)}
\left[
\beta_0(h_0(v))
+
\sum_{k=1}^{K}
\Delta_k(h_k(v))
\right].
}
$$

Equivalently,

$$
\hat y_G
=
b+
\sum_c n_{G,0,c}\beta_{0,c}
+
\sum_{k=1}^{K}
\sum_c n_{G,k,c}\Delta_{k,c}.
$$

The hierarchy therefore becomes a sequence of increasingly specific **correction terms**.

---

# 6. Identifiability across nested levels

Because the hierarchy is nested,

$$
n_{G,k-1,p}
=
\sum_{c:p(c)=p}
n_{G,k,c}.
$$

This means that if all children of a parent receive the same correction $d$,

$$
\sum_{c:p(c)=p}
n_{G,k,c}d
=
n_{G,k-1,p}d.
$$

Therefore a level-$k$ correction could simply reproduce or modify a parent-level effect, even though the parent parameter is frozen.

That would weaken the interpretation that deeper levels capture only newly resolved information.

---

# 7. Parent-centered correction constraint

To make each refinement level a true refinement rather than a refit, impose a zero-mean contrast constraint within each parent:

$$
\boxed{
\sum_{c:p(c)=p}
w_c\Delta_{k,c}
=
0.
}
$$

The most natural weights are training support counts:

$$
w_c=N_c.
$$

Thus

$$
\boxed{
\sum_{c:p(c)=p}
N_c\Delta_{k,c}
=
0.
}
$$

Under this constraint, child corrections cannot change the average contribution of the parent class.

If the parent contribution is $\beta_p$ and its children have

$$
\beta_c=\beta_p+\Delta_c,
$$

then

$$
\frac{
\sum_{c:p(c)=p}
N_c\beta_c
}{
\sum_{c:p(c)=p}N_c
}
=
\beta_p.
$$

This makes each refinement level an interpretable **within-parent contrast**.

---

# 8. Interpretation as hierarchical decomposition

The contribution of a node can be viewed as

$$
f(v)
=
\underbrace{\beta_0}_{\text{coarse contribution}}
+
\underbrace{\Delta_1}_{\text{first refinement}}
+
\underbrace{\Delta_2}_{\text{second refinement}}
+\cdots.
$$

Each correction answers:

> What additional predictive information is gained when this structural distinction becomes available?

This gives the model an ANOVA-like decomposition over graph refinement levels.

Possible analyses include:

$$
\text{validation error vs. refinement level}
$$

and

$$
\|\Delta_k\|
\text{ vs. refinement level}.
$$

Plateauing corrections would indicate that deeper structural information adds little predictive value.

---

# 9. Support gating

Support remains central.

For a class $c$ at level $k$, if

$$
N_{k,c}<n_{\min},
$$

do not fit a refinement correction:

$$
\boxed{
\Delta_{k,c}=0.
}
$$

Then the class inherits the complete contribution of its ancestors.

This gives a particularly clean interpretation of backoff:

> The finer environment is recognized structurally, but there is insufficient evidence that the extra structural detail changes the graph-level target.

Thus, in this formulation,

$$
\text{backoff}
\equiv
\text{zero deeper correction}.
$$

No special inference-time fallback is required for an unsupported child.

---

# 10. Regularization across depth

Deeper levels are more specific, have more classes, and typically have lower support.

It is therefore natural to use stronger regularization at deeper levels:

$$
\lambda_1
\le
\lambda_2
\le
\cdots
\le
\lambda_K.
$$

A first implementation should probably remain simple:

- one support threshold $n_{\min}$;
- one regularization strength $\lambda_k$ per level;
- no class-specific regularization initially.

More elaborate support-aware penalties can be explored later.

---

# 11. Relationship to boosting

The progressive residual update

$$
r^{(k)}
=
r^{(k-1)}
-
X_k\Delta_k
$$

resembles stagewise boosting.

However, the sequence of learners is not chosen adaptively. The order is fixed by the structural hierarchy:

$$
\text{coarse attributes}
\rightarrow
\text{refinement level 1}
\rightarrow
\text{refinement level 2}
\rightarrow
\cdots.
$$

A useful conceptual description is therefore:

> **structurally constrained stagewise residual fitting**

rather than ordinary boosting.

---

# 12. Natural stopping criterion

Because levels are added progressively, maximum depth can be selected empirically.

After fitting level $k$, evaluate

$$
\mathcal L_{\mathrm{val}}^{(k)}.
$$

If deeper levels no longer improve validation performance, stop.

Thus the model can reveal the structural radius required by a graph-level property.

---

# 13. Extensive versus intensive graph properties

The sum-pooling formulation is most natural for extensive or approximately additive targets:

$$
\hat y_G
=
b+
\sum_v f(v).
$$

For an intensive target, a normalized version may be more appropriate:

$$
\hat y_G
=
b+
\frac{1}{|V(G)|}
\sum_v f(v).
$$

The pooling operator should reflect the expected physical scaling of the target.

The preferred first version should use only simple sum or mean pooling in order to preserve interpretability.

---

# 14. Possible node + edge decomposition

The graph-level extension can naturally incorporate both node and edge contributions:

$$
\boxed{
\hat y_G
=
b
+
\sum_v f_{\mathrm{node}}(v)
+
\sum_e f_{\mathrm{edge}}(e).
}
$$

Each could have its own nested Sieve hierarchy and progressive correction structure.

---

# 15. Vector graph-level targets

Nothing conceptually changes if

$$
y_G\in\mathbb R^d.
$$

Then

$$
\beta_{k,c},
\Delta_{k,c}
\in\mathbb R^d.
$$

For example, molecular σ-profiles could be learned directly from graph-level profile labels:

$$
\hat{\mathbf P}_G
=
\sum_v
\left[
\boldsymbol\beta_0(h_0(v))
+
\sum_k
\boldsymbol\Delta_k(h_k(v))
\right].
$$

This would allow latent atomic σ-profile contributions to be inferred even when only molecular σ-profiles are available.

---

# 16. Remaining identifiability challenge

Graph-level supervision does not uniquely determine local contributions in general.

If two structural environments always co-occur, the data may not reveal how the graph-level target should be divided between them.

Formally,

$$
X\beta=y
$$

may have multiple solutions.

Nested hierarchy features are also intrinsically correlated.

The progressive design helps by:

- freezing previously learned coarse effects;
- fitting only residual corrections;
- constraining child corrections to be centered within each parent;
- suppressing unsupported corrections;
- regularizing deeper corrections increasingly strongly.

These choices favor the **coarsest sufficient explanation** of the graph-level target.

---

# 17. Mergeability

For a fixed linear stage, least-squares or ridge fitting can be expressed through additive sufficient statistics:

$$
X^TX
$$

and

$$
X^Ty.
$$

For independent data shards,

$$
X^TX
=
X_A^TX_A
+
X_B^TX_B,
$$

and

$$
X^Ty
=
X_A^Ty_A
+
X_B^Ty_B.
$$

Thus progressive graph-level Sieve could retain a form of exact mergeability.

However, unlike node-level Sieve, the feature cross-products couple different environment classes, so the merged state may be much larger.

The graph-level extension therefore preserves the algebraic principle of mergeability but not necessarily the exceptional storage simplicity of the original node-level model.

---

# 18. Current preferred formulation

The model currently converged on is

$$
\boxed{
\hat y_G^{(K)}
=
b+
\sum_{v\in G}
\left[
\beta_0(h_0(v))
+
\sum_{k=1}^{K}
\Delta_k(h_k(v))
\right].
}
$$

with the following rules:

1. **Fit the base level first.**
2. **Freeze all previously fitted levels.**
3. **At level $k$, fit only the residual left by levels $0,\ldots,k-1$.**
4. **Set $\Delta_{k,c}=0$ for classes below the support threshold.**
5. **Constrain child corrections to have support-weighted zero mean within each parent:**

$$
\sum_{c:p(c)=p}
N_c\Delta_{k,c}
=
0.
$$

6. **Regularize deeper levels increasingly strongly if needed.**
7. **Stop adding levels when validation error stops improving.**

This preserves the core Sieve philosophy:

> **Start with the broadest supported explanation and introduce finer structural corrections only when the data provide evidence for them.**

---

# 19. Conceptual variants

### Node-Sieve

$$
\text{environment}
\rightarrow
E[y_v\mid\text{environment}].
$$

### Edge-Sieve

$$
\text{edge environment}
\rightarrow
E[y_e\mid\text{environment}].
$$

### Graph-Sieve / Sieve Contributions

$$
\boxed{
y_G
=
\text{sum or mean of progressively refined latent local contributions}.
}
$$

The hierarchy remains Sieve-like, while the estimator changes from direct class-wise averaging to progressive latent contribution fitting.
