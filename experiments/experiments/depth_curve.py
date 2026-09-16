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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase
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


def ci_half_width(se: float, n_folds: int, level: float = 0.95) -> float:
    """Half-width of the ``level`` confidence interval on a fold mean.

    Student's t on ``n_folds - 1`` degrees of freedom applied to the
    Nadeau-Bengio corrected standard error, which together are the corrected
    resampled t-test of Ref.~2003. Reporting the interval rather than the
    bare standard error keeps the bars and any later claim that two depths
    are indistinguishable resting on exactly the same quantity.
    """
    if n_folds < 2:
        raise ValueError(
            f"a confidence interval needs at least two folds, got {n_folds}"
        )

    from scipy.stats import t as student_t

    return float(student_t.ppf(0.5 + level / 2.0, n_folds - 1)) * se


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
    """The validation curve: one point per depth."""
    train_points: tuple[DepthPoint, ...]
    """The same for the training shards, empty unless the runs were made
    with ``--score-train``."""
    n_runs: int
    panel: str = ""
    """The panel this arm is drawn in, when several share one; the first
    arm's value titles the panel. Empty means the arm's own label."""
    label: str = ""
    """What a reader sees: "Sieve" rather than "sieve-element-pooled". The
    method key identifies an arm unambiguously among run directories and is
    noise on an axis, so the two are kept apart."""
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
    label: str | None = None,
    panel: str | None = None,
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
        label=label or method,
        panel=panel or "",
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


def _provenance(curves: Sequence[Sequence[Sequence[ArmCurve]]], store: str) -> str:
    """What the figure was drawn from, and what its marks mean.

    Run counts come from the first row only: every row reads the same run
    directories for a different metric, so summing across rows would report
    each run once per metric.
    """
    arm_bits = ", ".join(
        f"{a.label} ({a.n_runs} runs)" for panel in curves[0] for a in panel
    )
    folds = {
        len(p.fold_values)
        for row in curves
        for panel in row
        for a in panel
        for p in a.points
    }
    k = folds.pop() if len(folds) == 1 else 0
    return (
        f"Store {store}; Study A, one repeat (seed 0); "
        f"arms: {arm_bits}. Points are individual folds; the line is their "
        f"mean and the bars are 95% confidence intervals from the "
        f"Nadeau-Bengio corrected standard error with Student's t on k-1 "
        f"degrees of freedom (Mach. Learn. 52:239, 2003), which accounts "
        f"for the training "
        f"molecules the {k or 'k'} folds share -- they are not independent "
        f"replicates. The dashed horizontal line in each panel marks the best "
        f"value the other method reached for that metric."
        + (
            " Dotted lines are the same metric on each fold's own training "
            "shards, scored unnormalized; where a panel shows one such line "
            "for several arms, those arms produce identical training numbers, "
            "since a training atom matches the class it was fitted into and "
            "so never backs off."
            if any(a.train_points for row in curves for panel in row for a in panel)
            else ""
        )
        + _omission_note(curves)
    )


def _omission_note(curves: Sequence[Sequence[Sequence[ArmCurve]]]) -> str:
    """Name every depth that was scored but kept off the figure.

    A note on the plotted range, not a confession: the depth axis is a grid
    of candidate hyperparameter values rather than a sample, so omitting a
    dominated candidate moves no plotted value and cannot move the argmin.
    It is stated only so a reader knows the sweep ran wider than the axis.
    """
    omitted = {
        a.label: a.omitted_depths for row in curves for panel in row for a in panel
    }
    bits = [
        f"{label} {', '.join(str(d) for d in depths)}"
        for label, depths in sorted(omitted.items())
        if depths
    ]
    if not bits:
        return ""
    return " The sweep also covered " + "; ".join(bits) + ", not shown here."


def train_groups(
    panel: Sequence[ArmCurve],
) -> list[tuple[ArmCurve, list[str]]]:
    """Group a panel's arms by their train curve, keeping input order.

    On training atoms the class a query matches is the one it was fitted
    into, so no backoff occurs, and estimators that differ only in how a
    *backoff* class is read produce identical training numbers. Drawing that
    curve once per arm would promise the reader a line sitting exactly
    underneath another.

    Detected rather than assumed: an arm whose support threshold exceeds one
    does back off on its own training atoms, and keeps a train curve of its
    own.
    """
    groups: list[tuple[ArmCurve, list[str]]] = []
    for arm in panel:
        if not arm.train_points:
            continue
        key = [(p.depth, p.mean) for p in arm.train_points]
        for representative, labels in groups:
            same = [(p.depth, p.mean) for p in representative.train_points]
            if same == key:
                labels.append(arm.label)
                break
        else:
            groups.append((arm, [arm.label]))
    return groups


# Colour alone cannot separate two arms whose curves nearly coincide, and it
# separates nothing at all for a reader who cannot distinguish the hues. Each
# arm therefore carries a marker and a dash pattern as well. The patterns stay
# clear of the dotted training curves and of the dashed reference lines.
_ARM_STYLES: tuple[tuple[str, Any], ...] = (
    ("o", "-"),
    ("s", (0, (6, 1.5))),
    ("^", (0, (4, 1, 1, 1))),
    ("D", (0, (2, 1))),
)


def _style_by_label(
    curves: Sequence[Sequence[Sequence[ArmCurve]]],
) -> dict[str, tuple[str, Any]]:
    """One marker and dash pattern per arm, assigned across the whole figure."""
    return {
        label: _ARM_STYLES[i % len(_ARM_STYLES)]
        for i, label in enumerate(_arm_labels(curves))
    }


def _arm_labels(curves: Sequence[Sequence[Sequence[ArmCurve]]]) -> list[str]:
    """Every arm label once, in the order the arms were given."""
    labels: list[str] = []
    for row in curves:
        for panel in row:
            for arm in panel:
                if arm.label not in labels:
                    labels.append(arm.label)
    return labels


def _colour_by_label(
    curves: Sequence[Sequence[Sequence[ArmCurve]]],
) -> dict[str, str]:
    """One colour per arm, assigned across the whole figure.

    Colour identifies the arm, not the panel it sits in: an arm that changed
    colour between the rows would make each row's legend contradict the
    other's.
    """
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {label: cycle[i % len(cycle)] for i, label in enumerate(_arm_labels(curves))}


def other_panel_reference(
    row: Sequence[Sequence[ArmCurve]], panel_index: int
) -> tuple[str, float] | None:
    """The best value reached outside ``panel_index``, and whose arm it is.

    With more than one arm elsewhere in the row, the reference is the best
    of them rather than whichever is listed first, since the line exists to
    say how far the competing method got.
    """
    candidates = [
        (arm.label, best_value(arm))
        for index, panel in enumerate(row)
        if index != panel_index
        for arm in panel
        if arm.points
    ]
    if not candidates:
        return None
    higher_is_better = row[panel_index][0].metric in _HIGHER_IS_BETTER
    return (max if higher_is_better else min)(candidates, key=lambda pair: pair[1])


def plot_depth_grid(
    curves: Sequence[Sequence[Sequence[ArmCurve]]],
    output_stem: str | Path,
    *,
    store: str,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """A metric-by-panel grid: one row per metric, one column per panel.

    A panel holds one or more arms sharing a depth axis, so estimators of the
    same method are read against each other in place rather than across the
    figure. Rows share a y axis and columns share an x axis, and each panel
    carries a horizontal line at the best value reached outside it.

    Linear y throughout: the flat tail is where a depth is chosen, and a log
    axis spreads the early, uninteresting decade at its expense.
    """
    if not curves or not all(curves):
        raise ValueError("plot_depth_grid needs at least one row with one panel")
    for row in curves:
        if not all(row):
            raise ValueError("plot_depth_grid was given an empty panel to draw")

    shape = [len(row) for row in curves]
    if len(set(shape)) != 1:
        raise ValueError(f"every row must carry the same panels, got {shape}")

    import matplotlib.pyplot as plt

    from experiments.plotting.publication_plotting import (
        STYLE_DIR,
    )

    n_rows = len(curves)
    # Applied through a CONTEXT, not use_publication_style, which calls
    # plt.style.use and mutates rcParams for the whole process. The sheet
    # turns constrained layout on, and leaking that into a later figure whose
    # tight_layout meets a colorbar (plots.parity_panel) raises "Colorbar
    # layout of new layout engine not compatible with old engine".
    style = [
        STYLE_DIR / "publication.mplstyle",
        STYLE_DIR / "double-column.mplstyle",
        {"figure.figsize": (7.0, 2.6 * n_rows)},
    ]
    with plt.style.context(style):
        return _build_and_save(curves, output_stem, store=store, formats=formats)


def _build_and_save(
    curves: Sequence[Sequence[Sequence[ArmCurve]]],
    output_stem: str | Path,
    *,
    store: str,
    formats: Sequence[str],
) -> list[Path]:
    """The figure itself, built under an already-applied style context."""
    import matplotlib.pyplot as plt

    from experiments.plotting.publication_plotting import (
        label_panels,
        save_publication_figure,
    )

    n_rows, n_cols = len(curves), len(curves[0])
    colours = _colour_by_label(curves)
    styles = _style_by_label(curves)

    fig, axes = plt.subplots(n_rows, n_cols, sharex="col", sharey="row", squeeze=False)

    for row_index, row in enumerate(curves):
        for col_index, panel in enumerate(row):
            ax = axes[row_index][col_index]
            for arm in panel:
                _draw_validation(ax, arm, colours[arm.label], styles[arm.label])
            title = panel[0].panel or panel[0].label
            for representative, labels in train_groups(panel):
                _draw_train(
                    ax,
                    representative,
                    colours[representative.label],
                    styles[representative.label],
                    label=(
                        f"{title} (train)"
                        if len(labels) == len(panel) and len(panel) > 1
                        else f"{labels[0]} (train)"
                    ),
                )
            _draw_reference(ax, row, col_index, colours)
            if row_index == 0:
                ax.set_title(panel[0].panel or panel[0].label)
            if row_index == n_rows - 1:
                ax.set_xlabel(panel[0].x_label)
            if col_index == 0:
                ax.set_ylabel(metric_label(panel[0].metric))
            _order_legend(ax, panel)

    for row_index, row in enumerate(axes):
        for col_index, ax in enumerate(row):
            letter = ascii_lowercase[row_index * len(row) + col_index]
            # The first column has to clear its tick labels; the rest do not,
            # sharing the row's y axis.
            label_panels(
                [ax],
                labels=[letter],
                x=-0.085 if col_index == 0 else -0.045,
                y=1.01,
            )

    outputs = save_publication_figure(fig, output_stem, formats=formats, close=True)

    # The provenance is written beside the figure rather than burned into it:
    # this text is the manuscript's own caption, and a caption belongs in the
    # document, where it can be edited and typeset without regenerating the
    # image.
    caption_path = Path(output_stem).with_suffix(".txt")
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(_provenance(curves, store) + "\n")
    outputs.append(caption_path)
    return outputs


def _order_legend(ax: Any, panel: Sequence[ArmCurve]) -> None:
    """Validation curves first, in the order the arms were given."""
    handles, labels = ax.get_legend_handles_labels()
    wanted = [f"{arm.label} (validation)" for arm in panel] + [
        f"{arm.label} (train)" for arm in panel
    ]
    rank = {label: i for i, label in enumerate(wanted)}
    order = sorted(range(len(labels)), key=lambda i: rank.get(labels[i], len(rank)))
    ax.legend([handles[i] for i in order], [labels[i] for i in order])


def _draw_validation(
    ax: Any, arm: ArmCurve, color: str, style: tuple[str, Any]
) -> None:
    """Every fold, then the validation mean and its confidence interval."""
    marker, dashes = style
    # Five points per depth is well within what can be shown, and showing
    # them keeps the interval from being the only record of what was
    # measured.
    for point in arm.points:
        ax.plot(
            [point.depth] * len(point.fold_values),
            point.fold_values,
            marker=marker,
            markersize=1.2,
            linestyle="none",
            color=color,
            alpha=0.35,
            zorder=2,
        )

    if arm.points:
        ax.errorbar(
            [p.depth for p in arm.points],
            [p.mean for p in arm.points],
            yerr=[ci_half_width(p.se, len(p.fold_values)) for p in arm.points],
            marker=marker,
            markersize=2.0,
            linestyle=dashes,
            color=color,
            capsize=1.5,
            zorder=3,
            label=f"{arm.label} (validation)",
        )


def _draw_train(
    ax: Any, arm: ArmCurve, color: str, style: tuple[str, Any], *, label: str
) -> None:
    """The training curve, distinguished by a dotted line and a hollow marker
    rather than by colour, which already identifies the arm."""
    ax.errorbar(
        [p.depth for p in arm.train_points],
        [p.mean for p in arm.train_points],
        yerr=[ci_half_width(p.se, len(p.fold_values)) for p in arm.train_points],
        marker=style[0],
        markersize=2.0,
        markerfacecolor="white",
        linestyle=":",
        color=color,
        capsize=1.5,
        zorder=3,
        label=label,
    )


def _draw_reference(
    ax: Any,
    row: Sequence[Sequence[ArmCurve]],
    panel_index: int,
    colours: dict[str, str],
) -> None:
    """A horizontal line at the best value reached outside this panel.

    Dashed and in that arm's own colour, so it reads as "where the other
    column got to" rather than as anything this panel's arms did.
    """
    reference = other_panel_reference(row, panel_index)
    if reference is None:
        return
    label, value = reference
    ax.axhline(
        value,
        color=colours[label],
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        zorder=1,
        label=f"{label} best",
    )
