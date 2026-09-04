# Sieve — Design Update v2

**Status:** correction note *and* one new decision, superseding specific sections of `design.md`
**Date:** 2026-09-04
**Scope:** §4.3 and §6.4 of `design.md` were written before continuation (§4.4) existed and were
never revisited. This note records how they drifted, records that §4.3 was never implemented at
all, and decides what §6.4 should consume instead.
**Relationship to other documents:** `design.md` remains the primary reference. Where this note and
`design.md` disagree, this note wins for the sections it lists, until they are folded back in.
Unlike `design-update-v1.md`, which was purely a historical record, §3 below **is** a decision.

---

## 1. §4.3 and §6.4 predate continuation and do not account for it (supersedes §4.3, §6.4)

**The chronology is checkable.** `git log -- design.md` puts §4.3 and §6.4 in `e66cbb2`
("docs(design): constrained prediction and variance shrinkage"). Continuation landed in `3ffd670`,
`f300f9d`, and `880aeba`; none of the three touched §4.3 or §6.4. Their `design.md` hunks fall
inside §4.4 and in §5/§10/§11/§12. So both sections describe the pooled-estimator world
exclusively.

**What breaks: the likelihood's two moments now come from different estimators.** §6.4 opens with
$x_i\sim\mathcal N(\mu_i,\sigma^2_{\text{pred},i})$, "with $\mu_i,\sigma^2_{\text{pred},i}$ from
§4.2/§4.3." Under `class_estimator="continuation"` that pairing no longer holds, and the code says
so in as many words (`predict.py`'s `Predictions` docstring):

> `value` at a non-deepest level is the unweighted mean of the matched class's *children* —
> `support`/`variance` still describe the matched class's own pooled population, not the children
> averaged into `value`.

`variance[hit] = lvl.variance[cid[hit]]` is the matched class's atom-level $s^2$. So
$\eta_i=1/\sigma^2_{\text{pred},i}$ and $\chi_i=-\mu_i/\sigma^2_{\text{pred},i}$ are assembled from
a mean over children and a spread over atoms. **§6.4's "exact MLE of an idealized model" holds only
for `class_estimator="pooled"`.** Under continuation it is not the MLE of any model yet written
down. The EEM correspondence itself survives untouched — it follows from the quadratic form
whatever $\mu$ and $\sigma^2$ denote, so $\lambda$ is still the equalized chemical potential and
$J_{ij}=0$ is still the named approximation. What lapses is the claim that the specific moments
Sieve hands the solve are the moments of one coherent predictive distribution.

**The mis-weighting is directional, not merely noisy.** §4.4's "Where $C$ actually belonged" makes
the unit-of-replication argument for means and stops there; the predictive variance is the same
argument's other half and never received it. §4.3's $(1+1/N)$ inflation uses the atom count. A
backed-off class has large $N$ — that is why it was matched — so it inflates essentially not at
all, while the actual uncertainty in a continuation mean is about $\hat\tau^2_k/C$ with $C$
typically 2–3. Deep singletons meanwhile keep the full $2\times$ boost. Net: the σ² scheme routes
residual *away* from backed-off nodes and *toward* deep singletons, the opposite of what
continuation's own reasoning implies. Both $\hat\tau^2_k$ and $C$ are already computed
(`continuation.sibling_variance`, `continuation.child_counts`) for the empirical-Bayes weights.

**One thing that cuts against overstating this.** By the variance decomposition the pooled `msd`
already contains a between-child term; it is atom-weighted rather than unit-weighted. So σ² is
mis-weighted in the *same direction* the mean was, which is self-consistently wrong rather than
incoherent. The magnitude is not absurd; the derivation is what lapsed.

---

## 2. §4.3 was never implemented, so the variance is not total (supersedes §4.3)

There is no `alpha_v` field in `SieveConfig`, no $\tilde\sigma^2$ recursion anywhere in
`src/sieve/`, and no $\sigma^2_{\text{pred}}$. `FrozenLevel.variance` is the plain Bessel-corrected
$s^2$, NaN at $N=1$ by deliberate design (`level.py`, so a stored zero cannot be misread as
observed homogeneity).

**The asymmetry with the mean.** `_search` initializes `value` to `model.global_mean` and
overwrites only on a hit, so *every* atom gets a mean — an atom OOV at every backoff level still
gets a number, reported as `matched_level == -1`. Total by construction. `variance` is initialized
to NaN and written only at `hit`, leaving two holes:

1. **Never matched** (`matched == -1`) → NaN. `model.global_msd` is computed by `global_stats`,
   maintained across merges, and serialized — and never read by `predict.py`. The mean has its
   global fallback wired up; the variance does not.
2. **Matched with $N=1$** → NaN from `level.py`'s own guard. Reachable: `minimum_support=1` is the
   default, and §6.3's cosmobase numbers put singleton classes at 42.7% of atoms at level 5.

§4.3 claims totality for itself — using `msd` (exactly 0 at $N=1$) rather than $s^2$ is "what makes
the recursion **total**", collapsing to $\tilde\sigma^2_{k-1,p(c)}$ at $N=1$ — and §6.4 leans on it
("§4.3's shrinkage makes this vanishingly rare in practice"). Both describe code that was never
written. Note also that totality would require $\alpha^v>0$: at $N=1$ the denominator is
$(N-1)+\alpha^v=\alpha^v$. And the OOV case is not covered by that recursion at all, since it is
indexed by class and there is no class.

**Where the NaN currently goes.** `sieve_predictor.predict_raw` takes `sqrt(variance)` into
`atom_std`, and `normalize.std_weighted_normalize` floors any non-positive-or-NaN entry to
`_DEFAULT_STD_VALUE = 0.1` — DASH's own hardcoded fallback. Faithful as a reproduction of DASH's
scheme, but it means Sieve's missing variances are presently absorbed by a constant borrowed from
a different model rather than estimated.

**A note on §13 item 9.** That item posed $\alpha$ and $\alpha^v$ as parallel problems to be solved
sequentially. $\alpha$ was solved and measured best (`shrinkage_weight="empirical_bayes"`,
+0.000105, 10/10 folds, §4.4). $\alpha^v$ was not, and §3 below is the argument that it should not
be — the problem it solves is harder than §6.4 poses.

---

## 3. Decision: §6.4 consumes a level-pooled variance, not §4.3's recursion

**The solve is scale-invariant in $\sigma^2$.** For $c_i=1$, §6.4 reduces to
$x_i=\mu_i+\text{residual}\cdot\sigma^2_i/\sum_j\sigma^2_j$: multiplying every $\sigma^2$ by a
common constant changes nothing. Normalization never needs a calibrated variance, only correct
*ratios among the atoms of one molecule*. §4.3's apparatus — the $\alpha^v$ recursion, the
top-down pass, the Student-$t$ caveat — solves a strictly harder problem than its only consumer
poses.

**The approximation.** Take

$$
\sigma^2_i=\bar\sigma^2_{k^\star_i},
\qquad
\sigma^2_i=\sigma^2_{\text{global}}\ \text{ where }\ k^\star_i=-1
$$

with $\bar\sigma^2_k$ the count-weighted mean within-class variance at the matched level —
`continuation.atom_variance(model)`, one float per level, already computed there for the
empirical-Bayes weights — and $\sigma^2_{\text{global}}$ the stored `global_msd`. Total by
construction, no NaN at $N=1$, no new stored state, no new knob.

**Why this is defensible rather than merely expedient.** By the variance decomposition
$\bar\sigma^2_k\approx\bar\sigma^2_{k+1}+\hat\tau^2_k$: the level-pooled atom variance already
carries both components a backed-off node's predictive variance needs — within-child spread plus
between-child spread. It gets the between-child term atom-weighted rather than unit-weighted, which
is §1's critique — but that is a bias on one component of a quantity required to be correct only up
to a common factor. That is a far weaker objection than the one against per-class pooled `msd`,
where the miscalibration concentrates in whichever single child dominates the class.

**Drop the $(1+1/N)$ inflation rather than porting it.** It is the term §13 item 8 named as the
risk — "concentrating the residual hardest on high-variance nodes — which, after §4.3's $(1+1/N)$
inflation, are disproportionately the low-support nodes whose predictions are already least
reliable" — and at level-pooled granularity it is the only term that would reintroduce per-node
estimation noise.

**What this gives up.** Any genuine per-class variance signal. Within a match level every atom
weights identically, so the scheme reduces to "shallower matches absorb more residual." Given that
per-class variances carry 1–2 degrees of freedom (§4.4's own argument for pooling $\hat\tau^2$ per
level) and that effects measured in this repo run ~1e-4 in $r^2$, that is the right bet to take
first. It is testable afterwards by adding the per-class arm to the same comparison.

**Caveat for $d>1$.** `atom_variance` sums over target dimensions to match `sibling_variance`'s
scale, while §6.4 applies one scalar constraint per dimension independently. For charges $d=1$ and
it does not bite; the per-dimension variant is the same function without the sum.

**Not implemented in this commit.** This note records the decision; `NORMALIZERS` still holds two
of three arms.

---

## 4. Consequence for §13 item 8, and for equal weighting

Item 8 frames the three schemes as differing "in exponent, not in kind" — residual spread
$\propto\sigma^0$ (`equal_weighted`), $\sigma^1$ (`std_weighted`, DASH's eq 4), $\sigma^2$ (the
§6.4 MLE) — and rests on "only $\sigma^2$ has a derivation behind it."

**That premise is false under continuation** (§1), so running the comparison as written would
measure a σ² arm that no longer has the property it is being tested for. §3 restores it.

**Equal weighting gains, but for a negative reason worth recording accurately.** Continuation's
argument is that a non-deepest class's *pooled statistics* are calibrated for a population that
never reads them. Nothing in that sentence is specific to means; it applies verbatim to `msd`.
$\sigma^1$ and $\sigma^2$ both weight by that uncorrected quantity, and weight by it hardest at
exactly the backed-off nodes continuation exists to fix. `equal_weighted` reads no variance at all,
so continuation cannot touch its justification. But invariance is not correctness: $\sigma^0$ is
$\eta_i$ constant — every node equally soft — which is also a strong false claim, merely one this
particular argument has no leverage on.

**A cheaper diagnostic to run first.** All three schemes differ only in proportion to the residual
$X-\sum_j\mu_j$ they distribute, and coincide at zero residual. If continuation reduces per-atom
bias on rare environments (§4.4's sulfonyl sulfur: true +1.07, pooled backoff +0.36), those biases
add coherently across a molecule while estimator variance partly cancels, so $|X-\sum_j\mu_j|$
should fall. Measuring that under `pooled` vs `continuation` needs no new fit and no normalizer at
all, and it bounds how much the three-way comparison can possibly show.

**Nothing here is measured.** `NORMALIZERS` has no σ² entry; `sieve_predictor`'s own docstring
records that `std_weighted` was deliberately deferred "for a follow-up once this `atom_std` has
been checked against real data"; and the only normalization numbers in the repo are DASH's own
baseline (test MAE 0.0190 unnormalized → 0.0193 under `std_weighted`, buying exact charge
conservation). A theoretical ranking is not a predicted outcome.
