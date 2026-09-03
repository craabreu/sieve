"""Diagnostic plots for a charges-series run.

Ported from cosmo_experiments/sieve_experiments/plots.py's own
``parity_panel``/``_hexbin_subplot`` -- the hexbin-parity-grid core is
fully domain-agnostic (a list of ``{y_true, y_pred, quantity, title,
metrics}`` dicts), so it's reused here near-verbatim. cosmo's
``profile_panel``/``_display_smiles`` are deliberately not ported: this
series has no profile concept (a scalar target, not a 51-bin curve) and
no SMILES anywhere (see data.py's own module docstring).

matplotlib is imported lazily inside ``parity_panel``: a fast, CI-safe
test never needs it, and a run without it in its environment still gets
metrics.json/manifest.json/predictions.npz -- runner.py catches
ImportError around this call and logs a skip instead of failing the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _hexbin_subplot(
    ax: Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    *,
    quantity: str,
    title: str,
) -> None:
    """Draw one hexbin parity plot onto ``ax`` -- the shared core of every
    cell in ``parity_panel``."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    hb = ax.hexbin(y_true, y_pred, gridsize=40, bins="log", mincnt=1, cmap="YlOrRd")
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
    ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(f"true {quantity}", fontsize=8)
    ax.set_ylabel(f"predicted {quantity}", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    lines = [f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, float)]
    if lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )


def _histogram_subplot(
    ax: Axes,
    values: np.ndarray,
    metrics: dict[str, float],
    *,
    xlabel: str,
    title: str,
) -> None:
    """Draw one 1-D histogram onto ``ax`` -- the shared core of a
    ``"histogram"``-kind cell in ``parity_panel`` (e.g. molecule charge
    conservation: the distribution of molecule charge residual across
    conformers, rather than a true-vs-predicted parity scatter -- there is
    only one axis' worth of information in a residual)."""
    values = np.asarray(values).ravel()
    bins = 1 if np.max(np.abs(values)) < 1e-6 else 40
    ax.hist(values, bins=bins, color="#d62728", alpha=0.8)
    ax.axvline(0, color="0.3", lw=1, ls="--")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    lines = [f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, float)]
    if lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )


def parity_panel(
    panels: list[dict[str, Any]],
    out_path: str | Path,
    *,
    suptitle: str,
    n_cols: int = 2,
) -> None:
    """Save a grid of plots, one per entry in ``panels``.

    Each entry is a dict with ``metrics``, ``title``, plus either
    ``y_true``/``y_pred``/``quantity`` for a hexbin parity cell (``kind``
    omitted or ``"hexbin"``) or ``values``/``xlabel`` for a 1-D histogram
    cell (``kind="histogram"``) -- see ``runner._build_parity_panels``,
    which decides which panels a given run has data for (atom charge
    always, as a hexbin parity plot; molecule charge conservation when the
    test split is non-empty, as a histogram of the per-conformer residual,
    not a parity scatter -- there's only one axis'
    worth of information in a residual).
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        return
    n_cols = min(n_cols, n)
    n_rows = -(-n // n_cols)  # ceil division, no math.ceil import for one use

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes_flat, panels, strict=False):
        if panel.get("kind", "hexbin") == "histogram":
            _histogram_subplot(
                ax,
                panel["values"],
                panel["metrics"],
                xlabel=panel["xlabel"],
                title=panel["title"],
            )
        else:
            _hexbin_subplot(
                ax,
                panel["y_true"],
                panel["y_pred"],
                panel["metrics"],
                quantity=panel["quantity"],
                title=panel["title"],
            )
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def curve_panel(
    table: Any,
    out_path: str | Path,
    *,
    suptitle: str,
    n_cols: int = 3,
    band: str = "fill",
) -> None:
    """One subplot per metric, one line per series, mean plus a dispersion
    band (``point.std``, pooled across the runs sharing one x value).

    ``band`` picks how that dispersion renders: ``"fill"`` (default) shades
    mean +/- std; ``"errorbar"`` draws it as error bars instead of a line +
    shading; ``"none"`` draws the mean line alone. Any of the three still
    leaves ``point.lo``/``point.hi`` (min/max) and ``point.std`` in
    ``aggregate.csv`` regardless -- this only changes what the PNG shows.

    ``table`` is an ``aggregate.CurveTable``. Typed as ``Any`` to keep this
    module free of a runtime import from ``aggregate`` -- ``plots`` is
    imported by ``runner`` on every run, and this is the only function that
    needs it.

    matplotlib is imported lazily, as in ``parity_panel``: a caller without
    it still gets the CSV.
    """
    if band not in ("none", "errorbar", "fill"):
        raise ValueError(f"band must be 'none', 'errorbar', or 'fill', got {band!r}")

    import matplotlib.pyplot as plt

    metrics = sorted({metric for _, metric in table.series})
    if not metrics:
        return
    n_cols = min(n_cols, len(metrics))
    n_rows = -(-len(metrics) // n_cols)

    # Tick labels come from the union of every series' x positions, not from
    # one arbitrary series: series can cover different subsets of x (a run
    # missing one metric contributes no point for it), and taking the first
    # would silently drop ticks the other lines actually occupy.
    ticks = {
        point.x_pos: point.x_label
        for points in table.series.values()
        for point in points
    }
    tick_pos = sorted(ticks)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, metrics, strict=False):
        for (name, series_metric), points in sorted(table.series.items()):
            if series_metric != metric or not points:
                continue
            xs = [p.x_pos for p in points]
            means = [p.mean for p in points]
            stds = [p.std for p in points]
            if band == "errorbar":
                ax.errorbar(
                    xs, means, yerr=stds, marker="o", ms=4, capsize=3, label=name
                )
            else:
                ax.plot(xs, means, marker="o", ms=4, label=name)
                if band == "fill" and any(s > 0 for s in stds):
                    lo = [m - s for m, s in zip(means, stds, strict=True)]
                    hi = [m + s for m, s in zip(means, stds, strict=True)]
                    ax.fill_between(xs, lo, hi, alpha=0.2)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([ticks[p] for p in tick_pos])
        ax.set_xlabel(table.x, fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in axes_flat[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
