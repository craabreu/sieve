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
        value = row.metrics.get(metric)
        if value is None:
            continue
        repeat = int(params["cv.repeat"])
        fold = int(params["cv.fold"])
        by_method_sample.setdefault(method, {})[(repeat, fold)] = value

    methods = sorted(by_method_sample)
    if len(methods) < 2:
        raise ValueError(
            f"need >= 2 methods with a recorded {metric!r}; found {methods}"
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
