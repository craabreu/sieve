"""Make characterization plots for a store, from its SMILES strings.

Usage:
    python stores/characterize.py <store-name>

Reads ``stores/<store-name>/molecules.parquet`` (a "smiles" column is
required), computes a handful of RDKit descriptors for every molecule, and
writes one PNG per plot to ``stores/<store-name>/<store-name>-<plot>.png``.

Needs the "viz" extra: ``uv sync --extra viz``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

STORES_DIR = Path(__file__).resolve().parent


def _load_molecules(store_dir: Path):
    import pandas as pd

    df = pd.read_parquet(
        store_dir / "molecules.parquet", columns=["smiles", "cluster_id"]
    )
    return df["smiles"].tolist(), df["cluster_id"].to_numpy()


def _descriptors(smiles: list[str], cluster_ids):
    """Compute per-molecule descriptors, element counts, and the cluster id
    of each molecule, skipping unparseable SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    rows = []
    valid_cluster_ids = []
    elements: Counter[str] = Counter()
    n_bad = 0
    for s, cid in zip(smiles, cluster_ids, strict=True):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_bad += 1
            continue
        rows.append(
            {
                "molecular_weight": Descriptors.MolWt(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            }
        )
        valid_cluster_ids.append(cid)
        for atom in mol.GetAtoms():
            elements[atom.GetSymbol()] += 1
    if n_bad:
        print(f"warning: skipped {n_bad} unparseable SMILES", file=sys.stderr)
    return rows, elements, np.array(valid_cluster_ids)


def _weighted_quantiles(values, quantiles, weights=None):
    """Quantiles of `values`, optionally weighted (e.g. by cluster size)."""
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    # Midpoint of each sample's cumulative weight, normalized to [0, 1].
    cum_weight = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, cum_weight, values)


def _annotate_quantiles(ax, values, weights=None):
    qs = [0.2, 0.5, 0.8]
    qvals = _weighted_quantiles(values, qs, weights)
    pairs = zip(qs, qvals, strict=True)
    text = "\n".join(f"p{int(q * 100)} = {v:.3g}" for q, v in pairs)
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


def _histogram(ax, values, *, title, xlabel, integers=False, weights=None, log=False):
    values = np.asarray(values)
    if log:
        # Highly right-skewed counts (e.g. cluster size): log-spaced bins
        # keep the crowded small-value end visible instead of squeezing it
        # into a sliver next to one huge outlier bin.
        lo, hi = max(values.min(), 1), values.max()
        bins = np.geomspace(lo, hi, 40)
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif integers:
        # One bin per integer value, so bars line up with real counts
        # instead of leaving gaps between sparsely populated bin edges.
        lo, hi = np.floor(values.min()) - 0.5, np.ceil(values.max()) + 0.5
        bins = np.arange(lo, hi + 1)
    else:
        bins = 40
    ax.hist(values, bins=bins, weights=weights, color="#4C72B0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count" if weights is None else "weighted count")
    _annotate_quantiles(ax, values, weights)


def make_plots(store_name: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store_dir = STORES_DIR / store_name
    if not store_dir.is_dir():
        raise FileNotFoundError(f"no such store: {store_dir}")

    smiles, cluster_ids = _load_molecules(store_dir)
    rows, elements, cluster_ids = _descriptors(smiles, cluster_ids)
    if not rows:
        raise ValueError(f"no parseable SMILES in {store_dir / 'molecules.parquet'}")

    written = []

    # One histogram per basic descriptor.
    descriptor_specs = [
        ("molecular_weight", "molecular weight", "Da", False),
        ("heavy_atoms", "heavy-atom count", "heavy atoms", True),
        ("ring_count", "ring count", "rings", True),
        ("rotatable_bonds", "rotatable-bond count", "rotatable bonds", True),
    ]
    for key, title, xlabel, integers in descriptor_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        _histogram(
            ax,
            [r[key] for r in rows],
            title=f"{store_name}: {title}",
            xlabel=xlabel,
            integers=integers,
        )
        fig.tight_layout()
        out = store_dir / f"{store_name}-{key.replace('_', '-')}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    # Element composition bar chart.
    fig, ax = plt.subplots(figsize=(6, 4))
    items = sorted(elements.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    ax.bar(labels, counts, color="#DD8452", log=True)
    ax.set_title(f"{store_name}: element composition")
    ax.set_xlabel("element")
    ax.set_ylabel("atom count")
    fig.tight_layout()
    out = store_dir / f"{store_name}-elements.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    # Per-cluster stats: size, and mean heavy-atom count within the cluster.
    heavy_atoms = np.array([r["heavy_atoms"] for r in rows])
    _, inverse, cluster_sizes = np.unique(
        cluster_ids, return_inverse=True, return_counts=True
    )
    sums = np.zeros(len(cluster_sizes))
    np.add.at(sums, inverse, heavy_atoms)
    cluster_mean_heavy_atoms = sums / cluster_sizes

    fig, ax = plt.subplots(figsize=(6, 4))
    _histogram(
        ax,
        cluster_sizes,
        title=f"{store_name}: cluster size",
        xlabel="molecules per cluster",
        log=True,
    )
    fig.tight_layout()
    out = store_dir / f"{store_name}-cluster-size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    fig, ax = plt.subplots(figsize=(6, 4))
    _histogram(
        ax,
        cluster_mean_heavy_atoms,
        title=f"{store_name}: cluster mean heavy-atom count (size-weighted)",
        xlabel="mean heavy atoms",
        weights=cluster_sizes,
    )
    fig.tight_layout()
    out = store_dir / f"{store_name}-cluster-mean-heavy-atoms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="store directory name, e.g. chaos-store")
    args = parser.parse_args()

    written = make_plots(args.name)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
