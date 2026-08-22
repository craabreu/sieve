import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from cosmolayer import store
from rdkit import Chem

STORES_DIR = Path(__file__).resolve().parent


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


def split_chaos_store(train: float, val: float, test: float):
    assert abs(train + val + test - 1) < 1e-6, "The sum of the fractions must be 1"
    chaos_store = store.SegmentStore.load(STORES_DIR / "chaos-store")

    chaos_store.assign_splits(fractions={"train": train, "val": val, "test": test})
    molecules_df = chaos_store.molecules_df

    df = molecules_df.copy()
    mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"].values]
    if any(mol is None for mol in mols):
        raise ValueError("unparseable SMILES in chaos-store")
    df["num_heavy_atoms"] = [mol.GetNumHeavyAtoms() for mol in mols]

    cluster_stats = (
        df.groupby("cluster_id")
        .agg(
            mean_heavy_atoms=("num_heavy_atoms", "mean"),
            size=("num_heavy_atoms", "size"),
        )
        .sort_values("mean_heavy_atoms", kind="mergesort")
    )

    assignment = assign_clusters_by_mean_size(
        cluster_stats.index.to_numpy(),
        cluster_stats["size"].to_numpy(),
        n_molecules=len(df),
        train=train,
        val=val,
    )
    df["split"] = df["cluster_id"].map(assignment)

    summary = (
        df.groupby("split")
        .agg(
            n_molecules=("smiles", "size"),
            n_clusters=("cluster_id", "nunique"),
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
