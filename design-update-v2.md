# Sieve — Design Update v2

**Status:** correction note *and* one new decision, superseding specific sections of `design.md`
**Date:** 2026-09-04
**Scope:** §4.3 and §6.4 of `design.md` were written before continuation (§4.4) existed and were
never revisited. This note records how they drifted, records that §4.3 was never implemented at
all, measures what the stored `msd` is actually worth (§3.1), and decides what §6.4 should consume
instead.
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

## 3. Decision: §6.4 consumes the posterior predictive variance, not §4.3's recursion

**The solve is scale-invariant in $\sigma^2$.** For $c_i=1$, §6.4 reduces to
$x_i=\mu_i+\text{residual}\cdot\sigma^2_i/\sum_j\sigma^2_j$: multiplying every $\sigma^2$ by a
common constant changes nothing. Normalization never needs a calibrated variance, only correct
*ratios among the atoms of one molecule*. §4.3's apparatus — the $\alpha^v$ recursion, the
top-down pass, the Student-$t$ caveat — solves a strictly harder problem than its only consumer
poses.

**The estimator.** Take

$$
\sigma^2_i=\underbrace{\frac{(N_c-1)\,s^2_{k^\star_i,c_i}+\alpha^v\,\bar\sigma^2_{k^\star_i}}
                            {(N_c-1)+\alpha^v}}_{\text{shrunk within-class}}
+\underbrace{\hat\tau^2_{\mathrm{pa}(k^\star_i)}\left(1-w_{c_i}\right)}
        _{\text{posterior variance of the class mean}},
\qquad
\sigma^2_i=\sigma^2_{\text{global}}\ \text{ where }\ k^\star_i=-1
$$

with $w_c$ the empirical-Bayes weight `shrinkage.empirical_bayes_weights` already returns,
$\hat\tau^2_k$ the sibling variance from `continuation.sibling_variance` (`root_variance` at level
0), $\bar\sigma^2_k$ the level-pooled `continuation.atom_variance`, and $\sigma^2_{\text{global}}$
the stored `global_msd`. Nothing new is stored and nothing new is estimated.

**$\alpha^v\approx10$**, and it is the one knob here — reintroduced deliberately after §3.1 showed
the knob-free alternatives are both worse. Performance is flat over $\alpha^v\in[5,20]$ (§3.1), so
it is not a delicate choice; §13 item 9's calibration route ($\mathrm{Var}(z)=1$) remains the way to
set it properly.

**This is not an approximation — it is the posterior predictive variance of the model §4.4 already
fits.** Writing the normal–normal model Sieve assumes,
$\bar y_c\mid m_c\sim N(m_c,\sigma^2/N_c)$ and $m_c\sim N(\mu_{\mathrm{pa}},\tau^2_k)$, the
posterior mean is $w\bar y_c+(1-w)\mu_{\mathrm{pa}}$ with $w=N/(N+\alpha)$ — exactly
`empirical_bayes_weights` — and the posterior variance falls out of the same fit as
$\tau^2(1-w)=\sigma^2/(N+\alpha)$. A new atom's predictive variance is that plus the within-class
spread. Sieve was already computing the first moment of this posterior and discarding the second.

**Note the sign, because it corrects the reasoning in §4.3 and in an earlier draft here.**
$\sigma^2/(N+\alpha)<\sigma^2/N$: shrinkage makes the class mean *more* certain, not less. So the
mean's standard error cannot be the source of the low-support miscalibration §3.1 measures — the
correct EB term is *smaller* than §4.3's $(1+1/N)$. The entire deficit lives in $\hat\sigma^2$
itself, i.e. in `msd`, which at $N=2$ carries a relative standard error of $\sqrt{2/(N-1)}=141\%$.
$(1+1/N)$ stays dropped, superseded by a derived term rather than by a hand-set one.

**§4.3's mechanism was right; its target and its consumer were not.** Shrinking the *variance* is
necessary — §3.1 measures raw $s^2$ as unusable, not merely miscalibrated. What changes is that the
pooling target is the level-pooled $\bar\sigma^2_k$ rather than the ancestor chain (one lookup, no
top-down recursion), and that the mean-uncertainty term is the derived EB one rather than $(1+1/N)$.

**Three earlier drafts of this section were wrong, and §3.1 records why.** They used
$\bar\sigma^2_k$ alone, then $\max(s^2,\bar\sigma^2_k)$, then $s^2+\hat\tau^2(1-w)$ with no
shrinkage. The first two destroy ranking by discarding per-class spread; the third leaves the
unbounded-$z$ tail intact. Each was adopted on a metric too weak to see its failure — see §3.1's
opening.

**Caveat for $d>1$, and it now bites harder.** `sibling_variance`/`atom_variance` sum over target
dimensions (deliberately — they feed a scalar weight), while $s^2$ is per-dimension and §6.4 applies
one scalar constraint per dimension independently. The estimator above therefore adds a
per-dimension term to a dimension-summed one, which is only coherent at $d=1$. For charges $d=1$;
for any multi-dimensional target both $\hat\tau^2$ and $\bar\sigma^2$ need their per-dimension
variants, which is the same function without the sum.

**Not implemented in this commit.** This note records the decision; `NORMALIZERS` still holds two
of three arms.

### 3.1 What the stored `msd` is actually worth

Measured on `dash-molecules-10fold-1` (82,358 train / 10,299 test conformers, 427,832 test atoms;
default attributes, `max_wl_depth=3`, `minimum_support=1`, `pooled`, no shrinkage; test MAE
0.01765). Every test atom matched something and only 202 — 0.05% — had NaN-or-zero variance, so
§2's totality hole is structurally real but empirically negligible on this store.

**A warning about the metric used in the first three passes of this section.** The ratio
$\mathbb E|y-\mu|\,/\,0.798\,\mathbb E[\sigma]$ below is a *ratio of means*, not the mean of a
ratio, and tests only a first moment. It is nearly blind to the failure mode that matters:
a class whose $s^2$ is spuriously near zero contributes a negligible amount to $\mathbb E[\sigma]$
while contributing an unbounded $z=(y-\mu)/\sigma$. On $\mathbb E[z^2]$ raw $s$ scores $3.5\times
10^{25}$; on the ratio metric it scores 1.13. Every decision recorded in earlier revisions of this
note rested on the weaker statistic. The tables below are kept because they are still true, but
§3.2 is what the decision now rests on.

**`msd` tracks realized error wherever the class has support.** Binning test atoms by their matched
class's $\sigma=\sqrt{s^2}$, the ratio sits at 1.04–1.10 across deciles 2–9 — a tenfold range of
$\sigma$, from 0.0089 to 0.068.

**It fails in exactly one place, and badly.** Binned by support instead:

| $N$ | atoms | mean $\sigma$ | mean $\lvert$err$\rvert$ | ratio |
|---|---:|---:|---:|---:|
| 2 | 1336 | 0.00766 | 0.02076 | **3.40** |
| 3–4 | 28557 | 0.00836 | 0.01911 | **2.86** |
| 5–9 | 30163 | 0.01513 | 0.01836 | 1.52 |
| 10–19 | 24882 | 0.01844 | 0.01800 | 1.22 |
| 20–49 | 37386 | 0.02080 | 0.01832 | 1.10 |
| 50–99 | 31287 | 0.02243 | 0.01854 | 1.04 |
| 100–299 | 49926 | 0.02307 | 0.01828 | 0.99 |
| 300–999 | 57503 | 0.02242 | 0.01775 | 0.99 |
| 1000+ | 166590 | 0.01975 | 0.01666 | 1.06 |

`msd` becomes trustworthy around $N\approx20$–50 and is essentially exact by $N\ge100$.

**This inverts §13 item 8's stated worry.** That item feared σ² weighting would concentrate
residual on high-variance nodes, "which, after §4.3's $(1+1/N)$ inflation, are disproportionately
the low-support nodes whose predictions are already least reliable." In this data the relationship
runs the other way: low-support classes carry *spuriously small* $\sigma$ (0.0077 at $N=2$ against
0.0198 at $N\ge1000$), so an uninflated σ² scheme **under**-weights them. Note also that mean
$|$err$|$ is nearly flat across support (0.0177–0.0208 from $N=2$ to $N=999$) — low support is not
itself predictive of larger error here, though that flatness pools across levels and is partly
deeper-match-is-better cancelling lower-support-is-worse.

**The variance is real signal, and it survives within a molecule** — the only place the solve can
use it, since a molecule-constant factor cancels in $\sigma^2_i/\sum_j\sigma^2_j$:

Scored on both criteria at once — ranking, which is all §6.4 can read, and calibration by support,
which is what an honest reported uncertainty needs:

| $\sigma$ estimator | within-mol $r$ | Spearman | ratio $N<5$ | ratio $N\ge100$ | overall |
|---|---:|---:|---:|---:|---:|
| raw $s$ (today) | 0.5872 | 0.4657 | 2.89 | 1.03 | 1.13 |
| level-pooled $\bar\sigma_k$ | 0.5835 | 0.4074 | 1.24 | 1.02 | 1.06 |
| $\max(s,\bar\sigma_k)$ | 0.6024 | 0.4910 | 1.15 | 0.84 | 0.88 |
| $\sqrt{\bar\sigma^2_k+\hat\tau^2_{\mathrm{pa}}(1-w)}$ | 0.5678 | 0.3908 | **0.98** | **1.01** | **1.02** |
| **$\sqrt{s^2+\hat\tau^2_{\mathrm{pa}}(1-w)}$** | **0.6081** | **0.5279** | 1.32 | 1.02 | 1.05 |
| $\sqrt{\max(s^2,\bar\sigma^2_k)+\hat\tau^2_{\mathrm{pa}}(1-w)}$ | 0.6071 | 0.4955 | 0.93 | 0.84 | 0.85 |

Four readings. **σ-weighting has something `equal_weighted` throws away**: within-molecule
$r\approx0.6$ is not noise. **The EB term is what fixes low support, and it does it better than any
floor**: adding $\hat\tau^2_{\mathrm{pa}}(1-w)$ to raw $s$ moves the $N<5$ ratio from 2.89 to 1.32
*and* raises Spearman from 0.466 to 0.528, because $1-w$ is largest exactly where support is
thinnest — a class-specific correction where a floor is a blunt one. **Calibration and ranking are
not the same criterion**: the best-calibrated row (level-pooled within, 0.98/1.01/1.02) has the
*worst* ranking in the table, worse than raw $s$; pooling away per-class spread buys calibration by
discarding the signal §6.4 reads. **The floor is dominated**: $\max(s,\bar\sigma_k)$, which the
previous draft adopted, is beaten on both ranking metrics and over-estimates $\sigma$ by ~19% at
$N\ge100$ (ratio 0.84), because it lifts genuinely homogeneous classes to their level average.

On this metric $\sqrt{s^2+\hat\tau^2_{\mathrm{pa}}(1-w)}$ wins ranking outright while staying within
5% of calibrated overall, which is why an earlier revision adopted it. §3.2 shows that reading was
an artifact of the metric.

**Clustering was the obvious suspect and it is not the cause.** A class's $N$ counts conformers,
and conformers of one molecule are near-replicates, so a low-$N$ class measuring only conformational
jitter would explain the collapse with no new mechanism. It does not hold: distinct training
molecules per class give mean $M/N=1.000$ at $N=2$, 0.997 at $N=3$–4, and 0.935 at $N=5$–9 — no
pseudo-replication at all at the low end. Recalibrating by $M$ instead of $N$ moves the bottom bin
only from 3.40 to 3.20. Replication does appear in the large classes ($M/N=0.636$ at $N\ge50$), but
those are the well-calibrated ones. The deficit is small-sample noise in $s^2$, exactly as the
$\sqrt{2/(N-1)}$ relative standard error predicts.

**Scope.** One fold, one config, `class_estimator="pooled"`. It says nothing about §1's
continuation-specific pairing problem, which needs the same measurement under `continuation`.
Reference values for that fit: $\bar\sigma_k=[0.194, 0.066, 0.028, 0.017]$,
$\hat\tau_k=[0.332, 0.110, 0.034, \mathrm{nan}]$, root $\hat\tau=0.573$.

### 3.2 Calibration, tested properly

Same fit, on signed $z=(y-\mu)/\sigma$: $\mathbb E[z^2]$ (1.0 if calibrated) marginally and
conditionally, Gaussian NLL as a proper score, and within-molecule ranking, all at once.

| $\sigma$ estimator | $\mathbb E[z^2]$ | $N<5$ | $N\ge10^3$ | cov@95 | NLL | within-$r$ | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw $s$ (today) | $3.5\times10^{25}$ | $5\times10^{26}$ | 1.18 | 0.887 | $1.7\times10^{25}$ | 0.5872 | 0.4657 |
| $\max(s,\bar\sigma_k)$ + EB | 0.864 | 1.35 | 0.67 | 0.961 | −2.433 | 0.6071 | 0.4955 |
| $s$ + EB (prior revision) | 1.645 | 3.68 | 1.17 | 0.927 | −2.395 | 0.6081 | 0.5279 |
| shrunk, $\alpha^v=1$ | 1.465 | 2.23 | 1.17 | 0.933 | −2.459 | 0.6118 | 0.5359 |
| shrunk, $\alpha^v=3$ | 1.374 | 1.78 | 1.17 | 0.937 | −2.484 | 0.6144 | 0.5362 |
| **shrunk, $\alpha^v=10$** | 1.293 | 1.57 | 1.17 | 0.941 | **−2.499** | 0.6170 | 0.5334 |
| shrunk, $\alpha^v=20$ | 1.261 | 1.54 | 1.16 | 0.942 | −2.501 | **0.6182** | 0.5300 |

**Raw $s$ is not miscalibrated, it is unusable.** $\mathbb E[z^2]=3.5\times10^{25}$, kurtosis
$8\times10^4$: near-zero class variances produce unbounded $z$. For §6.4 that is not a cosmetic
problem — an atom with $\sigma^2\approx0$ is *pinned* by the solve (§6.4 says so explicitly), so
a sampling artifact at $N=2$ would freeze an atom and push its share of the residual onto its
neighbours. This is the strongest argument in the note for changing what §6.4 consumes.

**Shrinkage dominates both knob-free alternatives on every axis at once.** At $\alpha^v=10$ it beats
the hard floor on NLL (−2.499 vs −2.433) *and* ranking (0.6170/0.5334 vs 0.6071/0.4955), and beats
the unshrunk EB form on calibration (1.29 vs 1.65) while also ranking better. The knob is flat:
NLL varies by 0.04 over $\alpha^v\in[1,20]$ and within-$r$ by 0.006. That flatness is why
reintroducing a knob is acceptable here.

**Nothing on this list is actually well calibrated.** The best row still runs $\mathbb E[z^2]=1.57$
at $N<5$ down to 1.17 at $N\ge10^3$, and by matched level from 2.4 at level 0 to 0.8 at level 3 —
a systematic trend, not noise, so marginal $\mathbb E[z^2]\approx1.3$ is an average of over- and
under-dispersion. By element it ranges from 1.23 (H) to 2.06 (N) to 3.03 (S), with sulfur the worst
— the same chemistry §4.4's sulfonyl example names.

**The Gaussian family itself is the bigger approximation.** Standardized residuals have skew 2–3 and
kurtosis 80–140 against a Gaussian's 3, and coverage is too *wide* in the centre and too *narrow* in
the tails (for the floored EB form: 68.3% actual at nominal 50%, but 98.3% at nominal 99%). §4.3
flags Student-$t$ as a "stated approximation" taken to preserve §6.4's closed form; the size of that
approximation is now measured, and it is large. It does not invalidate the σ² weighting — §6.4 needs
relative scale, not a density — but any *interval* reported from these numbers would be wrong, and
that is the use for which conformal prediction, not a variance, is the right tool.

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
particular argument has no leverage on. §3.1 now measures it directly and finds it false: within a
molecule, $\sigma$ orders atoms by realized $|$error$|$ at $r\approx0.6$. Nodes are demonstrably
not equally soft, and `equal_weighted` discards that ordering by construction.

**A cheaper diagnostic to run first.** All three schemes differ only in proportion to the residual
$X-\sum_j\mu_j$ they distribute, and coincide at zero residual. If continuation reduces per-atom
bias on rare environments (§4.4's sulfonyl sulfur: true +1.07, pooled backoff +0.36), those biases
add coherently across a molecule while estimator variance partly cancels, so $|X-\sum_j\mu_j|$
should fall. Measuring that under `pooled` vs `continuation` needs no new fit and no normalizer at
all, and it bounds how much the three-way comparison can possibly show.

**No normalization outcome here is measured** — §3.1 measures the variance the schemes would
consume, not what any of them does to MAE. `NORMALIZERS` has no σ² entry; `sieve_predictor`'s own
docstring
records that `std_weighted` was deliberately deferred "for a follow-up once this `atom_std` has
been checked against real data"; and the only normalization numbers in the repo are DASH's own
baseline (test MAE 0.0190 unnormalized → 0.0193 under `std_weighted`, buying exact charge
conservation). A theoretical ranking is not a predicted outcome.
