"""The Study A depth-curve figure: one panel per method, publication-grade.

Study A answers one question -- at what depth does each method stop
improving -- and answers it with k folds of a *single* repeat. Those folds
are not independent replicates: any two of them share k-2 of their k-1
training groups, so the naive standard error of their mean, s/sqrt(k),
understates the real uncertainty. Nadeau and Bengio's correction (Machine
Learning 52:239-281, 2003, sec. 3) is the standard remedy and is what this
module's error bars carry; ``nadeau_bengio_se`` is the whole of it.

Why a separate module from ``plots.py``: that one is imported by ``runner``
on every run and deliberately stays free of heavy imports, and its
``curve_panel`` is a diagnostic grid (150 dpi, 6-9 pt type) rather than a
manuscript figure. Styling here comes from the shared sheets under
``plotting/styles``, never from per-figure rcParams.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.aggregate import read_runs_from_dirs

# Metric names as a reader should see them. Partial charges are in units of
# the elementary charge, which is what makes the axis label a quantity
# rather than a column name.
_METRIC_LABELS = {
    "rmse": "RMSE ($e$)",
    "mae": "MAE ($e$)",
    "r2": "$R^2$",
    "sum_constraint/rmse": "Total-charge RMSE ($e$)",
    "sum_constraint/mae": "Total-charge MAE ($e$)",
}


def metric_label(metric: str) -> str:
    """Axis label for ``metric``, falling back to the bare name."""
    return _METRIC_LABELS.get(metric, metric)


def nadeau_bengio_se(
    values: Sequence[float], *, n_test: float, n_train: float
) -> float:
    """Corrected standard error of the mean of ``values``.

    ``values`` are one score per fold. In k-fold cross-validation the folds
    reuse each other's training molecules, so their scores are positively
    correlated and the usual ``s/sqrt(k)`` is optimistic. Nadeau and Bengio
    estimate the variance of the mean as::

        (1/k + n_test/n_train) * s^2

    which is the naive ``s^2/k`` plus a term for that reuse. The inflation
    over the naive standard error is ``sqrt(1 + k * n_test/n_train)``: at
    this study's own geometry -- k=5, each fold holding out 10 of 50 shards,
    so ``n_test/n_train = 1/4`` -- exactly 1.5.

    ``n_test``/``n_train`` are sizes in the same unit (conformers here); only
    their ratio matters.
    """
    k = len(values)
    if k < 2:
        raise ValueError(
            f"a standard error needs at least two folds, got {k} -- one "
            "observation carries no spread"
        )
    if n_train <= 0:
        raise ValueError(f"n_train must be positive, got {n_train}")
    if n_test < 0:
        raise ValueError(f"n_test must be non-negative, got {n_test}")

    mean = sum(values) / k
    variance = sum((v - mean) ** 2 for v in values) / (k - 1)
    return math.sqrt((1.0 / k + n_test / n_train) * variance)


@dataclass(frozen=True)
class DepthPoint:
    """One depth's k folds, and the mean they support."""

    depth: int
    fold_values: tuple[float, ...]
    fold_n_test: tuple[float, ...]
    fold_n_train: tuple[float, ...]
    mean: float
    se: float
    """Nadeau-Bengio corrected standard error of ``mean``."""


@dataclass(frozen=True)
class ArmCurve:
    """One method's depth curve for one metric."""

    method: str
    metric: str
    points: tuple[DepthPoint, ...]
    """The held-out (validation) curve: one point per depth."""
    train_points: tuple[DepthPoint, ...]
    """The same for the training shards, empty unless the runs were made
    with ``--score-train``."""
    n_runs: int
    x_label: str = "depth"
    normalization: str = ""
    omitted_depths: tuple[int, ...] = ()
    """Depths that were scored but kept off the figure, via ``min_depth``.
    Recorded so the caption can report how wide the sweep actually ran."""


def _points_for_metric(
    by_depth: dict[int, list[tuple[int, dict[str, float]]]], metric: str
) -> tuple[DepthPoint, ...]:
    """One point per depth whose folds all recorded ``metric``.

    A depth missing the metric contributes nothing, matching
    ``aggregate.build_curve``: a gap is honest, a zero or an interpolation
    across it is not.
    """
    points: list[DepthPoint] = []
    for depth in sorted(by_depth):
        folds = sorted(by_depth[depth])
        values = [m[metric] for _, m in folds if metric in m]
        if len(values) != len(folds) or not values:
            continue
        n_tests = [float(m.get("n_test_conformers", 0.0)) for _, m in folds]
        total = sum(n_tests)
        # The folds partition the train split, so a fold trains on
        # everything the others held out -- read off the runs rather than
        # assumed to be (k-1)/k of some nominal total.
        n_trains = [total - n for n in n_tests]
        mean = sum(values) / len(values)
        se = (
            nadeau_bengio_se(
                values,
                n_test=sum(n_tests) / len(n_tests),
                n_train=sum(n_trains) / len(n_trains),
            )
            if len(values) >= 2 and total > 0 and min(n_trains) > 0
            else 0.0
        )
        points.append(
            DepthPoint(
                depth=depth,
                fold_values=tuple(values),
                fold_n_test=tuple(n_tests),
                fold_n_train=tuple(n_trains),
                mean=mean,
                se=se,
            )
        )
    return tuple(points)


def read_depth_curve(
    runs_root: Path,
    experiment: str,
    *,
    method: str,
    metric: str,
    x_label: str = "depth",
    min_depth: int | None = None,
) -> ArmCurve:
    """Read ``method``'s depth curve out of ``experiment``'s run directories.

    Reads through ``aggregate.read_runs_from_dirs`` -- the same reader
    ``sweep`` and ``compare`` use -- so a figure can never show a run
    neither of them would.

    ``min_depth`` keeps shallower points off the curve. A dominated shallow
    point can take over a linear axis -- Sieve at WL depth 0 is element-wise
    pooled means, whose $R^2$ of 0.46 compresses the 0.99 band where every
    difference between the real candidates lives. Those depths are still
    scored and still recorded; they are returned in ``omitted_depths`` so
    the caption can report the range the sweep actually covered.
    """
    rows = read_runs_from_dirs(Path(runs_root), experiment)
    by_depth: dict[int, list[tuple[int, dict[str, float]]]] = {}
    normalization = ""
    n_runs = 0
    for row in rows:
        if row.params.get("cv.method") != method:
            continue
        if metric not in row.metrics:
            continue  # a run that never scored this metric contributes nothing
        depth = int(row.params["cv.depth"])
        fold = int(row.params.get("cv.fold", 0))
        by_depth.setdefault(depth, []).append((fold, row.metrics))
        normalization = normalization or row.params.get("cv.normalization", "")
        n_runs += 1

    if not by_depth:
        raise ValueError(
            f"no runs in {experiment!r} for method {method!r} recording "
            f"metric {metric!r}"
        )

    points = _points_for_metric(by_depth, metric)
    train_points = _points_for_metric(by_depth, f"train/{metric}")
    omitted: tuple[int, ...] = ()
    if min_depth is not None:
        omitted = tuple(p.depth for p in points if p.depth < min_depth)
        points = tuple(p for p in points if p.depth >= min_depth)
        train_points = tuple(p for p in train_points if p.depth >= min_depth)
        if not points:
            raise ValueError(f"min_depth={min_depth} removes every point of {method!r}")

    return ArmCurve(
        method=method,
        metric=metric,
        points=points,
        train_points=train_points,
        n_runs=n_runs,
        x_label=x_label,
        normalization=normalization,
        omitted_depths=omitted,
    )


# Metrics where a larger number is better, so "best" is a maximum. Getting
# this backwards would put the reference line at the other method's WORST
# depth while still calling it a floor.
_HIGHER_IS_BETTER = frozenset({"r2", "norm/r2", "sum_constraint/r2"})


def best_value(arm: ArmCurve) -> float:
    """The best value ``arm`` reached over its depths -- a minimum for an
    error metric, a maximum for a score."""
    if not arm.points:
        raise ValueError(f"{arm.method!r} has no points to take a best of")
    means = [p.mean for p in arm.points]
    return max(means) if arm.metric in _HIGHER_IS_BETTER else min(means)


def _provenance(curves: Sequence[Sequence[ArmCurve]], store: str) -> str:
    """What the figure was drawn from, and what its marks mean.

    Run counts come from the first row only: every row reads the same run
    directories for a different metric, so summing across rows would report
    each run once per metric.
    """
    arm_bits = ", ".join(f"{a.method} ({a.n_runs} runs)" for a in curves[0])
    folds = {len(p.fold_values) for row in curves for a in row for p in a.points}
    k = folds.pop() if len(folds) == 1 else 0
    return (
        f"Store {store}; Study A, one repeat (seed 0); "
        f"arms: {arm_bits}. Points are individual folds; the line is their "
        f"mean and the bars are the Nadeau-Bengio corrected standard error "
        f"(Mach. Learn. 52:239, 2003), which accounts for the training "
        f"molecules the {k or 'k'} folds share -- they are not independent "
        f"replicates. The dashed horizontal line in each panel marks the best "
        f"value the other method reached for that metric."
        + (
            " Dotted lines are the same metric on each fold's own training "
            "shards, scored unnormalized."
            if any(a.train_points for row in curves for a in row)
            else ""
        )
        + _omission_note(curves)
    )


def _omission_note(curves: Sequence[Sequence[ArmCurve]]) -> str:
    """Name every depth that was scored but kept off the figure.

    A note on the plotted range, not a confession: the depth axis is a grid
    of candidate hyperparameter values rather than a sample, so omitting a
    dominated candidate moves no plotted value and cannot move the argmin.
    It is stated only so a reader knows the sweep ran wider than the axis.
    """
    omitted = {a.method: a.omitted_depths for row in curves for a in row}
    bits = [
        f"{method} {', '.join(str(d) for d in depths)}"
        for method, depths in sorted(omitted.items())
        if depths
    ]
    if not bits:
        return ""
    return " The sweep also covered " + "; ".join(bits) + ", not shown here."


def plot_depth_grid(
    curves: Sequence[Sequence[ArmCurve]],
    output_stem: str | Path,
    *,
    store: str,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """A metric-by-method grid: one row per metric, one column per method.

    Rows share a y axis (the metric is the same across a row) and columns
    share an x axis (the depth axis is the same down a column). Each panel
    also carries a horizontal reference line at the best value the *other*
    method reached for that metric -- the two methods count depth in
    different units, so they cannot share an x axis, and the reference line
    is what still lets a reader see one against the other.

    Linear y throughout: the flat tail is where a depth is chosen, and a log
    axis spreads the early, uninteresting decade at its expense.
    """
    if not curves or not all(curves):
        raise ValueError("plot_depth_grid needs at least one row with one arm")

    methods = [arm.method for arm in curves[0]]
    if any([arm.method for arm in row] != methods for row in curves):
        raise ValueError(
            "every row must carry the same methods in the same order, got "
            + "; ".join(str([a.method for a in row]) for row in curves)
        )

    import matplotlib.pyplot as plt

    from experiments.plotting.publication_plotting import (
        STYLE_DIR,
    )

    n_rows, n_cols = len(curves), len(methods)
    # The style is applied through a CONTEXT, not through
    # use_publication_style, which calls plt.style.use and so mutates
    # rcParams for the whole process. The publication sheet turns
    # constrained layout on; leaking that into a later figure that calls
    # tight_layout on a colorbar (plots.parity_panel does) raises
    # "Colorbar layout of new layout engine not compatible with old engine".
    # Still the shared sheets, never per-figure rcParams -- only scoped.
    style = [
        STYLE_DIR / "publication.mplstyle",
        STYLE_DIR / "double-column.mplstyle",
        {"figure.figsize": (7.0, 2.6 * n_rows)},
    ]
    with plt.style.context(style):
        return _build_and_save(
            curves, output_stem, store=store, formats=formats, colors=_colors(n_cols)
        )


def _build_and_save(
    curves: Sequence[Sequence[ArmCurve]],
    output_stem: str | Path,
    *,
    store: str,
    formats: Sequence[str],
    colors: Sequence[str],
) -> list[Path]:
    """The figure itself, built under an already-applied style context."""
    import matplotlib.pyplot as plt

    from experiments.plotting.publication_plotting import (
        label_panels,
        save_publication_figure,
    )

    n_rows, n_cols = len(curves), len(curves[0])

    fig, axes = plt.subplots(n_rows, n_cols, sharex="col", sharey="row", squeeze=False)

    for row_index, row in enumerate(curves):
        for col_index, arm in enumerate(row):
            ax = axes[row_index][col_index]
            _draw_arm(ax, arm, colors[col_index])
            _draw_reference(ax, arm, row, colors)
            if row_index == 0:
                ax.set_title(arm.method)
            if row_index == n_rows - 1:
                ax.set_xlabel(arm.x_label)
            if col_index == 0:
                ax.set_ylabel(metric_label(arm.metric))
            # The panel's own series leads; errorbar containers otherwise
            # sort behind the plain Line2D of the reference line.
            handles, labels = ax.get_legend_handles_labels()
            order = sorted(
                range(len(labels)), key=lambda i: not labels[i].startswith(arm.method)
            )
            ax.legend([handles[i] for i in order], [labels[i] for i in order])

    label_panels([ax for row in axes for ax in row])

    # Through supxlabel so constrained layout (on, from publication.mplstyle)
    # measures the caption like any other artist; reserving its space by hand
    # pushes the axes past the canvas and silently costs titles and legends.
    fig.supxlabel(
        "\n".join(textwrap.wrap(_provenance(curves, store), width=125)),
        fontsize=6,
        color="0.4",
        x=0.01,
        ha="left",
    )

    return save_publication_figure(fig, output_stem, formats=formats, close=True)


def _colors(n: int) -> list[str]:
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [cycle[i % len(cycle)] for i in range(n)]


def _draw_arm(ax: Any, arm: ArmCurve, color: str) -> None:
    """Every fold, then the mean and its corrected error bar."""
    # Five points per depth is well within what can be shown, and showing
    # them keeps the error bar from being the only record of what was
    # measured.
    for point in arm.points:
        ax.plot(
            [point.depth] * len(point.fold_values),
            point.fold_values,
            marker="o",
            markersize=2,
            linestyle="none",
            color=color,
            alpha=0.35,
            zorder=2,
        )

    if arm.points:
        ax.errorbar(
            [p.depth for p in arm.points],
            [p.mean for p in arm.points],
            yerr=[p.se for p in arm.points],
            marker="o",
            markersize=3.5,
            linestyle="-",
            color=color,
            capsize=2,
            zorder=3,
            label=f"{arm.method} (validation)",
        )

    # Train is distinguished by linestyle AND marker fill, not by colour
    # alone: the panel's colour already means "this method", and the gap
    # between the two lines is the thing being read.
    if arm.train_points:
        ax.errorbar(
            [p.depth for p in arm.train_points],
            [p.mean for p in arm.train_points],
            yerr=[p.se for p in arm.train_points],
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            linestyle=":",
            color=color,
            capsize=2,
            zorder=3,
            label=f"{arm.method} (train)",
        )


def _draw_reference(
    ax: Any, arm: ArmCurve, row: Sequence[ArmCurve], colors: Sequence[str]
) -> None:
    """A horizontal line per *other* method at the best value it reached.

    Dashed and in that method's own colour, so the line is readable as
    "where the other column got to" rather than as anything this panel's
    method did.
    """
    for index, other in enumerate(row):
        if other.method == arm.method or not other.points:
            continue
        ax.axhline(
            best_value(other),
            color=colors[index],
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
            label=f"{other.method} best",
        )
