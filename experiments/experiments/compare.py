"""Model comparison via repeated-measures ANOVA + Tukey HSD, following the
protocol of Ash, Wognum & Rodriguez-Perez (JCIM 2025, 65(18), 9398-9411,
doi:10.1021/acs.jcim.5c01609): repeated cross-validation, no aggregation
across folds, no variance correction (their answer to CV fold dependence
is design -- 5x5 repeated k-fold, not a Nadeau-Bengio-style adjustment).

scipy only (``scipy.stats.f``, ``scipy.stats.studentized_range``) -- no
statsmodels dependency. Reads runs written by ``cv.py``'s own
``run_dash_cv``/``run_sieve_cv`` (or anything else that writes a
``manifest["config"]["cv"]`` block shaped the same way), via
``aggregate.read_runs_from_dirs`` -- this module adds no new run format,
it only pivots the existing one.

The repeated-measures "subject" is one (repeat, fold) pair: because every
method's shard partition for a given repeat comes from
``cv.permute_into_folds(shard_ids, k=k, seed=repeat)`` -- deterministic in
(n, k, seed), not in which predictor called it -- the *same* held-out
molecules back every method's own sample at that (repeat, fold), which is
exactly what a paired repeated-measures design requires. ``read_cv_table``
raises rather than silently comparing mismatched samples if that pairing
does not hold (e.g. two methods were run with different ``repeats``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def run_commits(runs_root: Path, experiments: Sequence[str]) -> list[str]:
    """The distinct git commits the compared runs were produced at, shortened.

    The commit that belongs on the figure is the one the *runs* were made at,
    not the one that happens to be checked out when the plot is drawn -- those
    differ whenever a comparison is re-plotted later, which is exactly when
    provenance matters (friction observation 4). More than one commit in the
    list is a finding, not a formatting problem: it means the arms were not
    produced by the same code.
    """
    import json

    seen: set[str] = set()
    for experiment in experiments:
        for manifest in sorted((runs_root / experiment).glob("*__*/manifest.json")):
            try:
                info = json.loads(manifest.read_text()).get("git") or {}
            except (OSError, json.JSONDecodeError):
                continue
            commit = info.get("commit")
            if commit:
                seen.add(commit[:8] + ("-dirty" if info.get("dirty") else ""))
    return sorted(seen)


def read_cv_table(
    runs_root: Path,
    experiments: Sequence[str],
    *,
    depth_by_method: Mapping[str, int] | None = None,
    metric: str = "mae",
) -> tuple[list[str], np.ndarray]:
    """Read every CV run under ``experiments``, keep each method's own row
    at ``depth_by_method[method]`` when given (otherwise every depth found
    is kept, which is only meaningful if each method already has exactly
    one), and pivot into a ``(n_samples, n_methods)`` matrix -- rows are
    ``(repeat, fold)`` pairs in ascending order, columns are ``methods``
    (sorted).

    Raises if any two methods do not share exactly the same set of
    ``(repeat, fold)`` samples, or if fewer than two methods have any
    recorded ``metric`` at all.
    """
    from experiments.aggregate import read_runs_from_dirs

    rows = []
    for experiment in experiments:
        rows.extend(read_runs_from_dirs(runs_root, experiment))

    by_method_sample: dict[str, dict[tuple[int, int], float]] = {}
    for row in rows:
        params = row.params
        method = params.get("cv.method")
        if method is None:
            continue
        depth_str = params.get("cv.depth")
        if depth_str is None:
            continue
        depth = int(depth_str)
        if depth_by_method is not None and depth_by_method.get(method) != depth:
            continue
        # No non-finite check here: aggregate._numeric_metrics already drops
        # NaN before it reaches this function, so a metric that is NaN on
        # every run (sum_constraint/r2 is) arrives as simply absent, and the
        # "fewer than two methods" error below is what reports it.
        value = row.metrics.get(metric)
        if value is None:
            continue
        repeat = int(params["cv.repeat"])
        fold = int(params["cv.fold"])
        by_method_sample.setdefault(method, {})[(repeat, fold)] = value

    methods = sorted(by_method_sample)
    if len(methods) < 2:
        raise ValueError(
            f"need >= 2 methods with a recorded {metric!r}; found {methods}. "
            "Note that a metric which is NaN on every run reads as absent here "
            "(aggregate drops non-finite values), so this also covers an "
            "undefined metric such as sum_constraint/r2"
        )

    sample_keys = set(by_method_sample[methods[0]])
    for method in methods[1:]:
        keys = set(by_method_sample[method])
        if keys != sample_keys:
            raise ValueError(
                f"methods do not share the same (repeat, fold) samples -- "
                f"{method!r} disagrees with {methods[0]!r} on "
                f"{sorted(keys ^ sample_keys)}; repeated-measures ANOVA "
                "needs every method scored on the same samples"
            )

    ordered_keys = sorted(sample_keys)
    table = np.array(
        [[by_method_sample[m][key] for m in methods] for key in ordered_keys],
        dtype=np.float64,
    )
    return methods, table


@dataclass(frozen=True)
class ANOVAResult:
    f_stat: float
    p_value: float
    df_method: int
    df_subject: int
    df_error: int
    ms_error: float
    grand_mean: float
    method_means: dict[str, float]


def repeated_measures_anova(methods: Sequence[str], table: np.ndarray) -> ANOVAResult:
    """One-way repeated-measures ANOVA (subject = row, method = column).

    ``SS_error = SS_total - SS_method - SS_subject`` -- the two-way
    decomposition with no method-by-subject interaction term (there is
    only one observation per subject/method cell, so an interaction term
    cannot be separated from residual error; this is the standard
    repeated-measures ANOVA, not a mixed-design ANOVA)."""
    n, k = table.shape
    if k != len(methods):
        raise ValueError("methods and table column count must match")
    if n < 2:
        raise ValueError("need >= 2 subjects (repeat, fold pairs)")

    grand_mean = float(table.mean())
    method_means = table.mean(axis=0)
    subject_means = table.mean(axis=1)

    ss_total = float(((table - grand_mean) ** 2).sum())
    ss_method = float(n * ((method_means - grand_mean) ** 2).sum())
    ss_subject = float(k * ((subject_means - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_method - ss_subject

    df_method = k - 1
    df_subject = n - 1
    df_error = df_method * df_subject
    if df_error < 1:
        raise ValueError(
            f"df_error={df_error} < 1 -- need more subjects or more methods"
        )

    ms_method = ss_method / df_method
    ms_error = ss_error / df_error
    if ms_error > 0:
        f_stat = ms_method / ms_error
    elif ms_method > 0:
        # A genuine 0/0 -> +inf case: every sample the same up to a fixed
        # per-method shift, so the effect is perfectly resolved (measured
        # directly by test_repeated_measures_anova_matches_known_textbook_
        # values). Distinct from the fully degenerate case just below.
        f_stat = math.inf
    else:
        # Truly degenerate input (every cell equal, e.g. identical columns
        # with zero noise): both ms_method and ms_error are 0, so the ratio
        # is undefined, not "no effect" -- report it as such rather than
        # picking a side.
        f_stat = math.nan

    from scipy.stats import f as f_dist

    p_value = (
        float(f_dist.sf(f_stat, df_method, df_error))
        if not math.isnan(f_stat)
        else math.nan
    )

    return ANOVAResult(
        f_stat=float(f_stat),
        p_value=p_value,
        df_method=df_method,
        df_subject=df_subject,
        df_error=df_error,
        ms_error=float(ms_error),
        grand_mean=grand_mean,
        method_means=dict(zip(methods, method_means.tolist(), strict=True)),
    )


@dataclass(frozen=True)
class PairwiseComparison:
    a: str
    b: str
    diff: float  # mean(a) - mean(b)
    ci_lo: float
    ci_hi: float
    q_stat: float
    p_value: float


def tukey_hsd(
    methods: Sequence[str],
    table: np.ndarray,
    *,
    alpha: float = 0.05,
    anova: ANOVAResult | None = None,
) -> list[PairwiseComparison]:
    """All ``k choose 2`` pairwise mean differences, each with a Tukey-HSD
    studentized-range p-value and a ``1 - alpha`` simultaneous confidence
    interval -- ``scipy.stats.studentized_range``, no statsmodels.

    ``MS_error``/``df_error`` come from ``repeated_measures_anova`` unless
    already computed and passed in (``anova=``), so a caller doing both
    never pays for the decomposition twice."""
    from scipy.stats import studentized_range

    n, k = table.shape
    if anova is None:
        anova = repeated_measures_anova(methods, table)
    ms_error, df_error = anova.ms_error, anova.df_error

    means = table.mean(axis=0)
    se = math.sqrt(ms_error / n)
    q_crit = float(studentized_range.ppf(1 - alpha, k, df_error))
    margin = q_crit * se

    out: list[PairwiseComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(means[i] - means[j])
            q = abs(diff) / se if se > 0 else math.inf
            p = float(studentized_range.sf(q, k, df_error))
            out.append(
                PairwiseComparison(
                    a=methods[i],
                    b=methods[j],
                    diff=diff,
                    ci_lo=diff - margin,
                    ci_hi=diff + margin,
                    q_stat=float(q),
                    p_value=p,
                )
            )
    return out


@dataclass(frozen=True)
class SimultaneousCI:
    """Per-method mean with a Hochberg-Tamhane simultaneous interval.

    ``halfwidths[i]`` is sized so that *any* two methods' intervals overlap
    exactly when their Tukey HSD comparison is non-significant -- so one
    interval per method replaces ``k choose 2`` pairwise intervals, and
    significance is read off the plot by looking for overlap.
    """

    methods: tuple[str, ...]
    means: np.ndarray
    halfwidths: np.ndarray
    q_crit: float
    alpha: float

    @property
    def lo(self) -> np.ndarray:
        return self.means - self.halfwidths

    @property
    def hi(self) -> np.ndarray:
        return self.means + self.halfwidths

    def differs_from(self, reference: str) -> dict[str, bool]:
        """Which methods' intervals fail to overlap ``reference``'s.

        This is the plot's own visual rule, and for the equal-n case here it
        agrees exactly with ``tukey_hsd``'s p < alpha.
        """
        idx = self.methods.index(reference)
        lo, hi = self.lo, self.hi
        return {
            m: bool(min(hi[i], hi[idx]) - max(lo[i], lo[idx]) < 0)
            for i, m in enumerate(self.methods)
            if i != idx
        }


def simultaneous_ci(
    methods: Sequence[str],
    table: np.ndarray,
    *,
    alpha: float = 0.05,
    anova: ANOVAResult | None = None,
) -> SimultaneousCI:
    """Hochberg & Tamhane's eq. 3.32 intervals -- statsmodels'
    ``TukeyHSDResults.plot_simultaneous`` data, computed with scipy alone.

    With ``d_ij = sqrt(var/n_i + var/n_j)``, ``s1 = sum_{i<j} d_ij`` and
    ``s2_i = sum_j d_ij``::

        w_i        = ((k-1) * s2_i - s1) / ((k-1) * (k-2))      for k > 2
        w_i        = s1 / 2                                      for k == 2
        halfwidth_i = q_crit / sqrt(2) * w_i

    One difference from statsmodels worth being explicit about, because it
    changes the numbers rather than the picture: ``var`` here is the
    *repeated-measures* ``MS_error`` on ``(k-1)(n-1)`` df, not the one-way
    ``MS_within`` on ``k(n-1)`` df that ``pairwise_tukeyhsd`` would use.
    The folds are paired by construction (see this module's docstring), so
    the one-way form would charge fold-to-fold difficulty -- shared by every
    method -- to the error term and widen every interval. Feed a one-way
    ``ANOVAResult`` in via ``anova=`` to reproduce statsmodels exactly.
    """
    from scipy.stats import studentized_range

    n, k = table.shape
    if anova is None:
        anova = repeated_measures_anova(methods, table)
    q_crit = float(studentized_range.ppf(1 - alpha, k, anova.df_error))

    # Equal n by construction: read_cv_table rejects a ragged table.
    gvar = np.full(k, anova.ms_error / n, dtype=np.float64)
    iu = np.triu_indices(k, 1)
    d12 = np.sqrt(gvar[iu[0]] + gvar[iu[1]])
    d = np.zeros((k, k))
    d[iu] = d12
    d = d + d.T

    s1 = float(np.sum(d12))
    s2 = np.sum(d, axis=0)
    if k > 2:
        w = ((k - 1.0) * s2 - s1) / ((k - 1.0) * (k - 2.0))
    else:
        # Hochberg's weights are undefined at k=2; statsmodels splits the
        # single pairwise distance evenly, which puts the two intervals
        # exactly in contact at the HSD critical difference.
        w = np.full(k, s1 / 2.0)

    return SimultaneousCI(
        methods=tuple(methods),
        means=table.mean(axis=0),
        halfwidths=(q_crit / math.sqrt(2.0)) * w,
        q_crit=q_crit,
        alpha=alpha,
    )


def write_simultaneous_ci_plot(
    ci: SimultaneousCI,
    path: str | Path,
    *,
    comparison_name: str | None = None,
    title: str = "Multiple Comparisons Between All Pairs (Tukey)",
    xlabel: str = "",
    provenance: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """statsmodels' ``plot_simultaneous`` layout: one interval per method on
    the metric's own scale, rather than one per pair on a difference scale.

    ``comparison_name`` singles out a reference method -- drawn in blue with
    its interval bounds extended as dashed guides, with every other method
    coloured red if its interval clears those guides and grey if it does
    not. Without it every interval is black and any pair can still be
    compared by eye, which is the whole point of the layout: it stays
    readable as methods are added, where the pairwise plot grows as
    ``k choose 2``.
    """
    # Validated before matplotlib is imported, so a bad comparison_name is
    # reported as such even where matplotlib is absent -- otherwise the caller
    # sees ModuleNotFoundError and has to guess which problem they have.
    if comparison_name is not None and comparison_name not in ci.methods:
        raise ValueError(
            f"comparison_name {comparison_name!r} is not one of {list(ci.methods)}"
        )

    import matplotlib.pyplot as plt

    k = len(ci.methods)
    if figsize is None:
        # Wide enough for whichever is longest: the title, or the method
        # labels plus the plotting area they sit beside. A fixed 8in clipped
        # both the "sum_constraint/rmse" title and the provenance caption --
        # tight_layout cannot rescue text that simply does not fit.
        longest_label = max((len(m) for m in ci.methods), default=0)
        figsize = (
            max(8.0, 0.11 * len(title), 0.10 * longest_label + 5.0),
            0.5 * k + 2.0,
        )
    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(k)
    lo, hi = ci.lo, ci.hi

    if comparison_name is None:
        ax.errorbar(
            ci.means, y, xerr=ci.halfwidths, marker="o", linestyle="None", color="k"
        )
    else:
        midx = ci.methods.index(comparison_name)
        differs = ci.differs_from(comparison_name)
        sig = [i for i, m in enumerate(ci.methods) if i != midx and differs[m]]
        nsig = [i for i, m in enumerate(ci.methods) if i != midx and not differs[m]]

        ax.errorbar(
            ci.means[midx],
            midx,
            xerr=ci.halfwidths[midx],
            marker="o",
            linestyle="None",
            color="b",
        )
        for bound in (lo[midx], hi[midx]):
            ax.plot([bound] * 2, [-1, k], linestyle="--", color="0.7")
        for idx, color in ((sig, "r"), (nsig, "0.5")):
            if idx:
                ax.errorbar(
                    ci.means[idx],
                    idx,
                    xerr=ci.halfwidths[idx],
                    marker="o",
                    linestyle="None",
                    color=color,
                )

    ax.set_title(title)
    span = float(np.max(hi) - np.min(lo))
    ax.set_ylim((-1.0, float(k)))
    ax.set_xlim((float(np.min(lo)) - span / 10.0, float(np.max(hi)) + span / 10.0))
    ax.set_yticks(y)
    ax.set_yticklabels(list(ci.methods))
    ax.set_xlabel(xlabel)

    if provenance:
        # Wrapped to the figure width rather than run off the edge: the
        # caption grows with the number of commits the runs span, and a
        # provenance line that is cut in half records nothing.
        import textwrap

        wrapped = "\n".join(textwrap.wrap(provenance, width=int(figsize[0] * 15)))
        n_lines = wrapped.count("\n") + 1
        fig.text(0.01, 0.01, wrapped, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.035 * n_lines, 1, 1) if provenance else None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_tukey_plot(
    comparisons: Sequence[PairwiseComparison],
    path: str | Path,
    *,
    title: str = "Tukey HSD",
    metric_label: str = "MAE difference",
    provenance: str | None = None,
) -> None:
    """One horizontal CI per pairwise comparison, a vertical zero line, and
    an optional ``provenance`` string (store name, git commit, run count --
    friction observation 4: "figures carry no provenance") printed as a
    caption. Matplotlib imported lazily, mirroring ``plots.py``'s own
    convention -- its absence should only skip the plot, not fail the run.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(comparisons) + 1.5))
    labels = [f"{c.a} - {c.b}" for c in comparisons]
    diffs = [c.diff for c in comparisons]
    los = [c.diff - c.ci_lo for c in comparisons]
    his = [c.ci_hi - c.diff for c in comparisons]

    y = np.arange(len(comparisons))
    ax.errorbar(diffs, y, xerr=[los, his], fmt="o", color="C0", ecolor="C0", capsize=3)
    ax.axvline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    if provenance:
        fig.text(0.01, 0.01, provenance, fontsize=6, color="gray")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
