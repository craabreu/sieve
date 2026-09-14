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
\sigma^2_i=\underbrace{\frac{(N_c-1)\,s^2_c+\alpha^v\,\bar\sigma^2_k}{(N_c-1)+\alpha^v}}
        _{\text{within-class, shrunk}}
\;+\;a\underbrace{\frac{(C_c-1)\,\tau^2_c+\alpha^t\,\hat\tau^2_k}{(C_c-1)+\alpha^t}}
        _{\text{selection, support-independent}}
\;+\;\underbrace{\hat\tau^2_{\mathrm{pa}(k)}\left(1-w_c\right)}
        _{\text{mean estimation}}
$$

with $\sigma^2_i=\sigma^2_{\text{global}}$ where $k^\star_i=-1$, and the middle term **zero at the
deepest level** — no children, and it is read on an exact match, so there is no selection to correct.
$w_c$ is the empirical-Bayes weight `shrinkage.empirical_bayes_weights` already returns,
$\hat\tau^2_k$ the level-pooled `continuation.sibling_variance` (`root_variance` at level 0),
$\bar\sigma^2_k$ the level-pooled `continuation.atom_variance`, $\sigma^2_{\text{global}}$ the
stored `global_msd`, and $\tau^2_c$ a class's **own** children's mean-variance — the only new
quantity, one `bincount` from the stored arrays.

**$\alpha^v=30$, $\alpha^t=1$, $a=0.5$**, selected on the val split by Gaussian NLL over a 200-point
grid (§3.3), never on test. Every parameter is flat: the top five val configs differ by 0.0012 in
NLL, and the val-selected point achieves the *best achievable* test NLL to four decimals.

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

`msd` becomes trustworthy around $N\approx20$–50 and is essentially exact by $N\ge100$ — but $N$
counts *conformers*, roughly 3 per independent unit on this store, so read those thresholds as
$\approx7$–17 and $\approx33$ independent units. See the clustering correction below.

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

**Clustering *is* a cause, and an earlier revision of this section said otherwise on a broken
statistic.** That revision reported "mean $M/N=1.000$ at $N=2$ — no pseudo-replication at all" and
concluded the deficit was ordinary small-sample noise. The $M$ there came from `atom_mol_id`, which
indexes **conformers**, not molecules. Distinct conformers per class is ≈1 by construction — an atom
appears once per conformer — so that measurement could not test the hypothesis it was cited against.
The conclusion drawn from it was wrong and is withdrawn.

Measured properly, with the molecule key coalesced from `chembl_id`/`dash_id` (the store carries two
complementary record schemas), on `dash-molecules-10fold-1`: **95% of molecules have exactly 3
conformers and 5% have 2**, splits are by molecule, and $N/M$ per matched class has median 3.38.
So a class with $N=3$ holds **one molecule**, and 7.8% of test atoms match a single-molecule class.
Calibration re-binned by $M$:

| $M$ (molecules) | 1 | 2 | 3–4 | 5–9 | 10–19 | 20–49 |
|---|---:|---:|---:|---:|---:|---:|
| ratio | **2.71** | 1.58 | 1.29 | 1.16 | 1.06 | 0.99 |

The collapse sits exactly where a class holds one or two molecules, so $s^2$ there is measuring the
*conformational* spread of a single molecule's atom while the test atom comes from a different
molecule entirely. §3.1's "trustworthy around $N\approx20$–50" is better read as $M\approx20$
molecules.

**But the unit is $N/3$, not $M$.** Conformers cap at 3, so $N/M>3.5$ cannot mean "more conformers"
— it means one molecule contributing several *distinct* atoms (the six aromatic CH's of a ring). The
cross-tab separates the two: at fixed $M$, moving right along $N/M$ improves calibration (3.40 →
1.51 at $M=1$), so extra atom-orbits carry real information while extra conformers do not. Rank
correlations against $|z|$ bear it out — $M$ −0.2184, $N$ −0.2201, $N/M$ −0.1219 — with $N$
fractionally *better* than $M$, because dividing all the way to molecules over-corrects. Divide by
conformers-per-molecule and stop.

**Two consequences, one of which is not an improvement.** `minimum_support` counts conformers, so
`minimum_support=1` cannot gate anything — the smallest possible class is 3 — and `design.md` open
item 1 asks for an empirical default in a unit that cannot express the intended quantity. The
tempting follow-on, dividing $N$ by 3 in the shrinkage weight, is **not** supported: for the swept
count rule $\frac{N/3}{N/3+\alpha}=\frac{N}{N+3\alpha}$ exactly, so a swept $\alpha$ already absorbs
it, and the `shrinkage-d6` sweep degrades monotonically with $\alpha$ (0.016046 at 0.5 → 0.024110 at
100), i.e. the optimum is at the *weakest* shrinkage tried. Under `empirical_bayes` $\alpha$ is
estimated rather than swept so the absorption does not happen automatically — yet EB still beats the
best swept count rule on the same batch (0.015799 vs 0.015816). If the deepest-level weight is
inflated, it is not costing measurable accuracy, and an "unbiased" version that shrank three times
harder would plausibly be worse.

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

### 3.3 The search that fixed §3's form, and how it was selected

200 configurations of the three-term form, scored on the val split by Gaussian NLL and reported on
test. Val picks $\alpha^v=30,\ \alpha^t=1,\ a=0.5,\ b=1$; the test-selected optimum differs only in
$\alpha^t$ (3 rather than 1) and **both reach test NLL −2.5130, gap +0.0000**. The surface is flat
enough that honest selection costs nothing, which is the only reason the two extra knobs are
tolerable.

Test metrics at the val-selected point: NLL −2.5130, $\mathbb E[z^2]=1.145$, median-scale 0.79,
coverage at $\pm1.96\sigma$ **0.9508**, per-level $\mathbb E[z^2]$ = 1.44 / 1.13 / 1.00 / 1.18,
within-molecule $r=0.6305$, Spearman 0.5302, post-normalization MAE 0.017199.

**Three findings the grid settles.**

*The selection term must be support-independent.* This is the one component that does not vanish as
$N\to\infty$, and it is what the regressogram framing predicts: a wide bin is read **only** by
queries whose narrow bin was empty (§2.2's prefix property), so the population reading a coarse class
is a biased sample of it — selection bias, not approximation bias. The control settles it: multiplying
the term by $(1-w)$ so that it *does* vanish reproduces having no term at all (NLL −2.5106 vs
−2.5099, level trend 2.71/1.87/1.42 vs 2.85/1.88/1.45). No parameter sweep would have found this;
only the structural argument did.

*The per-class $\tau^2_c$ beats the level-pooled $\hat\tau^2_k$.* Every top-12 config has finite
$\alpha^t$; the pooled variant is 0.0025 worse in NLL and 0.006 worse in within-$r$. §4.4's objection
to per-class variances ("at $C=2$–3 a per-class variance carries 1–2 degrees of freedom and is mostly
noise") is answered by *shrinking* them toward the pooled value, not by discarding them — and
$\alpha^t$ is insensitive (1, 3, 10 within 0.0002).

*The criteria still disagree, and the disagreement is now bounded.* NLL wants
$(30,1,0.5,1)$; within-molecule $r$ wants $(3,10,1.5,1)$ at 0.6354; post-normalization MAE wants
$a=b=0$ at 0.017177. The spread is 4% in ranking and 0.13% in MAE. §3 selects on NLL because the
question is which *uncertainty* estimator is best; this is explicitly **not** the best normalizer,
and §4's measurement of the prize (3.3% total) is why that costs nothing worth having.

**And $a$ is settled by atom counts, not by fit.** $a=1$ nails levels 0–1 (1.01, 0.82) and
over-corrects level 2 (0.78); $a=0.5$ nails level 2 (1.00) and under-corrects level 0 (1.44). Level 2
holds 76,117 atoms against level 0's **251**, so $a=0.5$ is right and level 0's calibration is noise.

### 3.4 Higher moments: merge verified, diagnostic negative — do not build it

§3.2's sulfur anomaly ($\mathbb E[z^2]=3.03$ against 1.03 for chlorine) suggested multimodal classes
— thiol vs sulfonyl vs thioether — which no within-class variance can express. §3.3's $\tau^2_c$
cannot test that at the deepest level, where 78.8% of atoms match and there are no children. The
route that reaches it is 3rd and 4th central moments per class. Prototyped; **not adopted.**

**The merge extension is real.** Pébay's pairwise formulas extend §5.2's law-of-total-variance
identity to arbitrary order,

$$
M_3 = M_{3A}+M_{3B}+\delta^3\frac{n_An_B(n_A-n_B)}{n^2}
      +3\delta\frac{n_AM_{2B}-n_BM_{2A}}{n}
$$

plus the four-term $M_4$ analogue. Verified over 400 random splits with adversarial mixed scales and
offsets: max relative error $4.4\times10^{-14}$ (mean), $5.3\times10^{-14}$ ($M_2$),
$1.6\times10^{-11}$ ($M_3$), $1.4\times10^{-13}$ ($M_4$); order-independence to $1.4\times10^{-14}$
over three-way merges in both associativity orders. Per-class $M_2/N$ reproduces the stored `msd`
to **exactly** 0.0 at all four levels. So the "one more term per moment, $O(1)$ storage" claim holds
and this stays available at any time.

**But the classes are leptokurtic, not bimodal.** At the deepest level with $N\ge20$ (271,400 atoms,
80.5% of deepest), median Sarle $BC$ runs 0.18–0.49 — mostly *below* the 0.556 threshold — while
median $b_2$ runs 4.2 to **29.0** (39.9 for hydrogen at $N\ge100$) against a Gaussian's 3.
Bimodality's signature is $b_2<3$; nothing here is close. Depth-3 refinement apparently already
separates the sulfur environments, leaving unimodal classes with outliers.

**$BC$ does not predict miscalibration.** $\mathbb E[z^2]$ by $BC$ quintile: 1.05, 1.06, 1.01,
**1.32**, 1.06 — non-monotone, and the same at $N\ge100$ (1.02, 1.02, 1.04, 1.35, 1.00). It does
track error *magnitude* (mean $|$err$|$ 0.0071 → 0.0128 across quintiles), but §3's σ already absorbs
that, which is the better outcome. Identical pattern to the ICC diagnostic, which saturated near 1
for 84–92% of classes at every element and was undefined for the 78.8% at the deepest level.

**Two things this does explain.** The class-level $b_2\approx6$–40 is the *source* of the kurtosis-182
in standardized residuals, so §3.2's "no single Gaussian scale satisfies both $\mathbb E[z^2]=1$ and
median-$|z|=1$" is a property of the data, not a defect in the estimator. And sulfur relocates: at the
deepest level with $N\ge20$ it is $\mathbb E[z^2]=1.18$, i.e. fine, so its anomaly lives in backed-off
or low-support classes — neither reachable by a 4th moment, which needs support to mean anything.

**A structural constraint worth recording.** The principled response to leptokurtic classes is a
*robust* scale — MAD, trimmed variance — or a Student-$t$ likelihood. But Sieve's architecture
requires statistics that merge exactly, and moments are precisely the family that merges exactly
while being maximally sensitive to the tails that are the problem. Quantiles and MAD do not merge
exactly at all. Heavy tails are therefore hard for Sieve *by construction*, and the only route is an
approximate mergeable sketch (t-digest, KLL) — a far bigger change than one term, and worth naming as
such rather than rediscovering later.

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

### 4.1 Item 8, answered — and the prize is 3.3%

Post-normalization error needs no refit and no `NORMALIZERS` entry to score: with weights
$p_i\propto\sigma_i^\gamma$ normalised within a molecule and $R=X-\sum_j\mu_j$, the error is
$p_iR-r_i$ with $r_i=y_i-\mu_i$, and $R=\sum_j r_j$ identically. Closed form, from arrays a fit
already produces.

| σ source | $\gamma=0$ | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---:|---:|---:|---:|---:|---:|
| raw $s$ | 0.017780 | 0.017474 | 0.017262 | **0.017174** | 0.017196 | 0.017450 |
| shrunk + EB | 0.017780 | 0.017509 | 0.017312 | 0.017208 | 0.017194 | 0.017390 |

Unnormalized MAE 0.017653; best $\gamma=1.85$ at 0.017188. $\gamma=0,1,2$ are `equal_weighted`,
DASH's eq 4, and the §6.4 MLE.

**The whole weighting question is worth 0.00059 MAE — 3.3%. The choice of σ estimator is worth 0.2%
of that**: every candidate at a sensible $\gamma$ lands within 0.00003 of every other, and raw $s$ —
which §3.2 shows is unusable *as an uncertainty* — ties the carefully shrunk version here. The
exponent matters roughly thirty times more than the estimator.

**$\gamma=2$ is empirically optimal, so §6.4's exponent survives even though its derivation does
not.** 0.017194 at $\gamma=2$ against 0.017188 at the fitted optimum. Every assumption behind the
MLE is violated — non-Gaussian (§3.4: class $b_2$ up to 40), non-independent ($J_{ij}\neq0$, §13
item 10), variances conditionally miscalibrated (§3.2) — and the exponent it prescribes is still
right to three decimals. Report that as an empirical finding, not as vindication of the derivation.

**`equal_weighted` is worse than not normalizing at all** (0.017780 vs 0.017653), so enforcing
conservation by equal spreading costs accuracy — matching the direction of DASH's own baseline
(0.0190 → 0.0193 under `std_weighted`). Any σ-weighting with $\gamma\ge1$ reverses that and comes out
ahead of the unnormalized prediction.

**The sequencing lesson, recorded because it cost four revisions of §3.** This calculation was
available from the first fit, needs no normalizer, and bounds everything §3 was optimizing. Measuring
the objective before refining the estimator would have shown immediately that the estimator is not
rate-limiting. §3.2's own warning about a too-weak metric is the same mistake one level down.
