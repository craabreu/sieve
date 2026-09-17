"""Small helpers for consistent publication-quality Matplotlib figures.

Locates the bundled ``.mplstyle`` files relative to this file, so the module
works whether it lives beside a sibling ``styles/`` directory (the skill
layout) or is copied into a project package alongside its own ``styles/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


def _find_style_dir() -> Path:
    """Return the directory holding the bundled ``.mplstyle`` files.

    Searches, in order, ``styles/`` next to this file and ``styles/`` one level
    up. This keeps the helper working under both the skill layout
    (``helpers/publication_plotting.py`` + ``styles/``) and a flattened project
    layout (``plotting/publication_plotting.py`` + ``plotting/styles/``).
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "styles", here.parent / "styles"):
        if (candidate / "publication.mplstyle").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate 'styles/publication.mplstyle' relative to "
        f"{here}. Keep the styles/ directory beside this module or one level up."
    )


STYLE_DIR = _find_style_dir()

_PRESET_STYLES = {
    "single": "single-column.mplstyle",  # one-column manuscript figure
    "double": "double-column.mplstyle",  # two-column / multi-panel manuscript figure
    "poster": "poster.mplstyle",         # large canvas + type for poster panels
    "slide": "slide.mplstyle",           # 16:9 figure sized for projection
}


def use_publication_style(
    preset: str | None = "double",
    *,
    box: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    extra_styles: Sequence[str | Path] = (),
    column: str | None = None,
) -> None:
    """Load the bundled publication style and an optional layout preset.

    Parameters
    ----------
    preset
        One of ``"single"``, ``"double"``, ``"poster"``, ``"slide"``, or
        ``None`` to apply only the base style (for example when the exact
        ``figsize`` is dictated by a journal and passed explicitly). The
        ``single``/``double`` presets set manuscript column widths; the
        ``poster``/``slide`` presets also enlarge type and strokes for viewing
        at a distance.
    box
        Frame the axes on all four sides with mirrored ticks (the default).
        Pass ``box=False`` for an open, L-shaped frame (top/right spines and
        ticks removed).
    grid
        Draw a light major grid behind the data. Off by default.
    figsize
        Optional ``(width, height)`` in inches applied after the preset. Use
        this for journal-specific or slide-region dimensions instead of editing
        the style sheets.
    extra_styles
        Additional Matplotlib styles applied after the bundled styles, so they
        take precedence over the base, preset, and box/grid overlays.
    column
        Backward-compatible alias for ``preset``; when given, it overrides
        ``preset``.
    """
    if column is not None:
        preset = column

    styles: list[str | Path] = [STYLE_DIR / "publication.mplstyle"]

    if preset is not None:
        try:
            preset_name = _PRESET_STYLES[preset]
        except KeyError as exc:
            choices = ", ".join(sorted(_PRESET_STYLES))
            raise ValueError(f"preset must be None or one of: {choices}") from exc
        styles.append(STYLE_DIR / preset_name)

    # The base style is boxed with no grid; layer overlays only to change that.
    if not box:
        styles.append(STYLE_DIR / "unboxed.mplstyle")
    if grid:
        styles.append(STYLE_DIR / "grid.mplstyle")

    styles.extend(extra_styles)

    missing = [p for p in styles if isinstance(p, Path) and not p.is_file()]
    if missing:
        missing_text = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing Matplotlib style file(s): {missing_text}")

    plt.style.use(styles)

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize


def _flatten_axes(axes: Axes | Iterable[Axes]) -> list[Axes]:
    """Return a flat list from one axis or an array/iterable of axes."""
    if isinstance(axes, Axes):
        return [axes]
    return [ax for ax in np.asarray(axes, dtype=object).ravel() if isinstance(ax, Axes)]


def label_panels(
    axes: Axes | Iterable[Axes],
    labels: Sequence[str] | str = "lower",
    *,
    x: float = -0.12,
    y: float = 1.04,
    fontweight: str = "bold",
) -> None:
    """Add consistently positioned labels to figure panels.

    ``labels`` may be an explicit sequence, ``"lower"`` (a, b, c, ...), or
    ``"upper"`` (A, B, C, ...). Lowercase is the more common journal default.
    """
    flat_axes = _flatten_axes(axes)
    if isinstance(labels, str):
        alph = ascii_lowercase if labels == "lower" else ascii_uppercase if labels == "upper" else None
        if alph is None:
            raise ValueError('labels must be a sequence, "lower", or "upper"')
        if len(flat_axes) > len(alph):
            raise ValueError("Too many panels for single-letter labels; pass an explicit sequence")
        panel_labels = list(alph[: len(flat_axes)])
    else:
        panel_labels = list(labels)

    if len(panel_labels) != len(flat_axes):
        raise ValueError("The number of panel labels must match the number of axes")

    for ax, label in zip(flat_axes, panel_labels, strict=True):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight=fontweight,
            clip_on=False,
        )


def save_publication_figure(
    fig: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    png_dpi: int = 600,
    close: bool = False,
) -> list[Path]:
    """Save a figure in vector and review formats.

    Parameters
    ----------
    fig
        Figure to save.
    output_stem
        Output path without an extension. An existing extension is removed.
    formats
        Output extensions, without leading dots. Defaults to a vector format
        (``pdf``) for the manuscript and a raster (``png``) for quick review;
        add ``"svg"`` when a figure needs manual vector editing.
    png_dpi
        Resolution for PNG output.
    close
        Close the figure after all formats are written (useful in loops).
    """
    stem = Path(output_stem).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"pdf", "svg", "png", "tif", "tiff", "eps"}:
            raise ValueError(f"Unsupported output format: {extension}")
        output = stem.with_suffix(f".{normalized}")
        kwargs = {"dpi": png_dpi} if normalized == "png" else {}
        fig.savefig(output, **kwargs)
        outputs.append(output)

    if close:
        plt.close(fig)
    return outputs
