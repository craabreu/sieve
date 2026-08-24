import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from cosmolayer import store
from cosmolayer.store.clustering import (
    ClusteringSpecs,
    FingerprintGenerator,
    butina_cluster,
)
from rdkit import Chem

STORES_DIR = Path(__file__).resolve().parent

# Half-width of the coarse-coding band appended to the Morgan fingerprint
# for the biased split's clustering: bits {n-SIZE_BAND_HALF_WIDTH, ...,
# n+SIZE_BAND_HALF_WIDTH} turn on for a molecule with n heavy atoms. Picked
# by comparing size-weighted mean cluster (max-min) heavy-atom range across
# widths on a chaos-store sample: 1 -> 6.11, 2 -> 5.69 (best), 4 -> 6.47
# (wider bands start over-merging clusters, same failure mode as an
# unbounded thermometer code, just weaker).
SIZE_BAND_HALF_WIDTH = 2
SIZE_BAND_BITS = 32


def download_chaos_store():
    url = "https://zenodo.org/records/22050672/files/chaos-store.zip?download=1"
    response = requests.get(url)
    with open("chaos-store.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("chaos-store.zip", "r") as zip_ref:
        zip_ref.extractall("chaos-store")

    os.remove("chaos-store.zip")


def assign_clusters_by_mean_size(
    cluster_ids: np.ndarray,
    cluster_sizes: np.ndarray,
    n_molecules: int,
    train: float,
    val: float,
) -> dict[int, str]:
    """Assign clusters in the given order to train, then val, then test.

    Clusters are assumed already sorted (smallest molecules first). A split
    stops before the next cluster that would push it past its target count.
    Whatever is left after train and val goes to test.
    """
    assignment = {}
    train_target = train * n_molecules
    val_target = val * n_molecules
    train_count = 0
    val_count = 0
    phase = "train"

    for cluster_id, size in zip(cluster_ids, cluster_sizes, strict=True):
        if phase == "train" and train_count + size > train_target:
            phase = "val"
        if phase == "val" and val_count + size > val_target:
            phase = "test"
        assignment[int(cluster_id)] = phase
        if phase == "train":
            train_count += size
        elif phase == "val":
            val_count += size

    return assignment


def _coarse_size_block(
    heavy_atom_counts: np.ndarray, half_width: int, n_bits: int
) -> np.ndarray:
    """One coarse-coding block per molecule: bits {n-half_width, ...,
    n+half_width} on for a molecule with n heavy atoms (clipped to stay
    inside [0, n_bits)). Lets molecules of nearby size share a few bits
    without flooding the fingerprint the way an unbounded thermometer
    code does.
    """
    capped = np.clip(heavy_atom_counts, 0, n_bits - 1)
    block = np.zeros((len(heavy_atom_counts), n_bits), dtype=np.int8)
    idx = np.arange(len(heavy_atom_counts))
    for offset in range(-half_width, half_width + 1):
        neighbor = np.clip(capped + offset, 0, n_bits - 1)
        block[idx, neighbor] = 1
    return block


def _biased_cluster_ids(mols: list, heavy_atom_counts: np.ndarray) -> np.ndarray:
    """Cluster ids for the biased split: Butina clustering (same cutoff as
    the store's own clustering) on Morgan fingerprints with a coarse-coded
    size block appended, so clusters are more size-homogeneous than the
    store's plain-Morgan ``cluster_id`` -- see ``SIZE_BAND_HALF_WIDTH``.
    """
    specs = ClusteringSpecs()
    generator = FingerprintGenerator(specs)
    morgan_fps = np.stack([generator.generate(mol) for mol in mols])
    size_block = _coarse_size_block(
        heavy_atom_counts, SIZE_BAND_HALF_WIDTH, SIZE_BAND_BITS
    )
    fps = np.concatenate([morgan_fps, size_block], axis=1)
    return butina_cluster(fps, cutoff=specs.cutoff, progress=True)


def split_chaos_store(train: float, val: float, test: float):
    assert abs(train + val + test - 1) < 1e-6, "The sum of the fractions must be 1"
    chaos_store = store.SegmentStore.load(STORES_DIR / "chaos-store")

    chaos_store.assign_splits(fractions={"train": train, "val": val, "test": test})
    molecules_df = chaos_store.molecules_df

    df = molecules_df.copy()
    mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"].values]
    if any(mol is None for mol in mols):
        raise ValueError("unparseable SMILES in chaos-store")
    df["num_heavy_atoms"] = np.array([mol.GetNumHeavyAtoms() for mol in mols])

    # Re-cluster with a size-aware fingerprint (rather than reusing the
    # store's plain-Morgan cluster_id) so the biased split's clusters are
    # more size-homogeneous -- see _biased_cluster_ids.
    df["biased_cluster_id"] = _biased_cluster_ids(mols, df["num_heavy_atoms"].values)

    cluster_stats = (
        df.groupby("biased_cluster_id")
        .agg(
            median_heavy_atoms=("num_heavy_atoms", "median"),
            size=("num_heavy_atoms", "size"),
        )
        .sort_values("median_heavy_atoms", kind="mergesort")
    )

    assignment = assign_clusters_by_mean_size(
        cluster_stats.index.to_numpy(),
        cluster_stats["size"].to_numpy(),
        n_molecules=len(df),
        train=train,
        val=val,
    )
    df["split"] = df["biased_cluster_id"].map(assignment)

    summary = (
        df.groupby("split")
        .agg(
            n_molecules=("smiles", "size"),
            n_clusters=("biased_cluster_id", "nunique"),
            mean_heavy_atoms=("num_heavy_atoms", "mean"),
        )
        .reindex(["train", "val", "test"])
    )
    summary["fraction"] = summary["n_molecules"] / len(df)
    print(summary.to_string())

    molecules_df["biased_split"] = df["split"]
    molecules_df.to_parquet(STORES_DIR / "chaos-store" / "molecules.parquet")


def main():
    if not (STORES_DIR / "chaos-store").exists():
        download_chaos_store()
    split_chaos_store(train=0.8, val=0.1, test=0.1)


if __name__ == "__main__":
    main()
