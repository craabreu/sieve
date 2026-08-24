"""Diagnostic plots for a run.

Promoted from scripts/train_chaos_sigma_profile.py (parity_hexbin,
profile_panel), writing into a caller-supplied directory instead of next to
the script. matplotlib is imported lazily inside each function: a fast,
CI-safe test never needs it, and a run without it in its environment still
gets metrics.json/manifest.json/predictions.npz -- runner.py catches
ImportError around these calls and logs a skip instead of failing the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parity_hexbin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    metrics: dict[str, float],
    *,
    level: str,
    title: str,
) -> None:
    """Save a hexbin parity plot, all profile bins (or scalars) pooled."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=60, bins="log", mincnt=1, cmap="YlOrRd")
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--")
    fig.colorbar(hb, ax=ax, label="log10(count)", fraction=0.046, pad=0.04)
    ax.set_xlabel(f"true value per {level}")
    ax.set_ylabel(f"predicted value per {level}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    lines = [f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)]
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
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
    randomly sampled test molecule per cell."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_sample = min(n_rows * n_cols, len(labels))
    idx = rng.choice(len(labels), size=n_sample, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes_flat, idx, strict=False):
        ax.plot(sigma_values, mol_true[i], color="0.3", lw=1.5, label="true")
        ax.plot(
            sigma_values, mol_pred[i], color="#d62728", lw=1.5, ls="--", label="pred"
        )
        ax.set_title(str(labels[i])[:24], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    for ax in axes_flat[n_sample:]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("sigma (e/A²)", fontsize=9)
    fig.supylabel("profile bin value (A²)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
