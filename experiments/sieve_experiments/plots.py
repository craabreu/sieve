"""Diagnostic plots for a run.

Promoted from scripts/train_chaos_sigma_profile.py (parity_hexbin,
profile_panel), writing into a caller-supplied directory instead of next to
the script. matplotlib is imported lazily inside each function: a fast,
CI-safe test never needs it, and a run without it in its environment still
gets metrics.json/manifest.json/predictions.npz -- runner.py catches
ImportError around these calls and logs a skip instead of failing the run.

Both plots show *normalized* profiles only (each row divided by its own
sum -- pure shape, no total-area scale): the harness's own metrics never
report anything about unnormalized profiles either (see metrics.py's
module docstring), so a plot of raw, area-scaled profile values would show
a quantity nothing else in a run's output describes or explains.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _display_smiles(smiles: str) -> str:
    """Strip atom-map numbers (the store's SMILES carry them, design.md
    11.3) so a truncated title reads as chemistry, not as a cut-off
    ``[C:1](...`` token. Falls back to the input unchanged if it doesn't
    parse (e.g. a synthetic test fixture's placeholder SMILES)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, Chem.SmilesParserParams())
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


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
    cell in ``parity_panel``. All points pooled (profile bins, or scalars
    for an area/charge parity cell)."""
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


def parity_panel(
    panels: list[dict[str, Any]],
    out_path: str | Path,
    *,
    suptitle: str,
    n_cols: int = 3,
) -> None:
    """Save a grid of hexbin parity plots, one per entry in ``panels``.

    Each entry is a dict with ``y_true``, ``y_pred``, ``quantity``,
    ``title``, ``metrics`` -- see ``runner._build_parity_panels``, which
    decides which panels a given run actually has data for (molecule
    profile always; molecule area/charge and atom profile/area/charge when
    the predictor and run supply them).
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


def profile_panel(
    sigma_values: np.ndarray,
    mol_true: np.ndarray,
    mol_pred: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    *,
    n_rows: int = 4,
    n_cols: int = 4,
    seed: int = 0,
) -> None:
    """Save an n_rows x n_cols panel: true vs. predicted profile curve, one
    randomly sampled test molecule per cell. ``labels`` are SMILES; shown
    with atom-map numbers stripped (see ``_display_smiles``) so a truncated
    title reads as chemistry, not a cut-off ``[C:1](...`` token."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_sample = min(n_rows * n_cols, len(labels))
    idx = rng.choice(len(labels), size=n_sample, replace=False)
    display_labels = [_display_smiles(s) for s in labels]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes_flat, idx, strict=False):
        ax.plot(sigma_values, mol_true[i], color="0.3", lw=1.5, label="true")
        ax.plot(
            sigma_values, mol_pred[i], color="#d62728", lw=1.5, ls="--", label="pred"
        )
        ax.set_title(display_labels[i][:24], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    for ax in axes_flat[n_sample:]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("sigma (e/A²)", fontsize=9)
    fig.supylabel("normalized profile (per bin)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
