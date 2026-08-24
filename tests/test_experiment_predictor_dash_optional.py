"""End-to-end DASHBackoffPredictor test against the real chaos-store and the
real DASH-tree clone (pins.toml's ``[dash_tree]``). Skipped unless rdkit,
cosmolayer, the store, and the clone are all present -- same pattern as
tests/test_experiment_store.py.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")

STORE_NAME = "chaos-store"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STORES_ROOT = REPO_ROOT / "stores"
STORE = STORES_ROOT / STORE_NAME
DASH_TREE_ROOT = REPO_ROOT / "experiments" / "external" / "DASH-tree"

pytestmark = [
    pytest.mark.skipif(not STORE.exists(), reason="chaos-store absent"),
    pytest.mark.skipif(not DASH_TREE_ROOT.exists(), reason="DASH-tree clone absent"),
]

# Small enough to run fast: a handful of training molecules is enough to
# exercise fit -> predict end to end without a multi-minute tree match.
TRAIN_LIMIT = 40
TEST_LIMIT = 10


def test_atom_map_order_matches_the_stores_own_element_column():
    """The highest-stakes check in this module. predictors/dash.py maps flat
    store position ``j`` to RDKit index ``order[j]``; get that transposed and
    every atom is matched against the wrong tree node -- which still yields
    perfectly finite metrics, so no other assertion here would catch it.

    This is the same guard ``src/sieve/io/cosmolayer_adapter.py`` gets from
    ``check_alignment``: compare the element RDKit reports at ``order[j]``
    against the store's own ``atoms_df['element']`` at the same flat
    position. The inverse convention mismatches on ~36% of chaos-store
    atoms, so this discriminates rather than passing vacuously.
    """
    from cosmolayer.store import SegmentStore
    from rdkit import Chem

    store = SegmentStore.load(STORE)
    df = store.molecules_df.iloc[:TRAIN_LIMIT].reset_index(drop=True)
    store_elements = store.atoms_df["element"].to_numpy()

    params = Chem.SmilesParserParams()
    params.removeHs = False

    flat = 0
    checked = 0
    non_identity = 0
    for smi, n_atoms in zip(df.smiles, df.num_atoms, strict=True):
        mol = Chem.MolFromSmiles(smi, params)
        assert mol is not None
        order = np.argsort([a.GetAtomMapNum() for a in mol.GetAtoms()])
        if not np.array_equal(order, np.arange(len(order))):
            non_identity += 1
        for j in range(n_atoms):
            expected = store_elements[flat + j]
            got = mol.GetAtomWithIdx(int(order[j])).GetSymbol()
            assert got == expected, (
                f"atom map order wrong for {smi[:50]} position {j}: "
                f"store says {expected}, order[j] gives {got}"
            )
            checked += 1
        flat += n_atoms

    assert checked > 0
    # guard the guard: if every order were the identity permutation the
    # assertion above would pass no matter which convention we used.
    assert non_identity > 0, "no reordered molecule in sample; check is vacuous"


def test_dash_backoff_end_to_end_on_real_store(tmp_path):
    from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from sieve_experiments.data import load_molecule_set
    from sieve_experiments.runner import execute

    mset, masks = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=TRAIN_LIMIT,
        stores_root=STORES_ROOT,
    )
    # Force small train/test slices out of the same loaded set, so both the
    # predictor and the ground truth stay consistent regardless of where the
    # real biased_split boundaries fall within the first TRAIN_LIMIT rows.
    train_mask = np.zeros(mset.n_molecules, dtype=bool)
    train_mask[: TRAIN_LIMIT - TEST_LIMIT] = True
    test_mask = ~train_mask
    masks = {"train": train_mask, "val": train_mask, "test": test_mask}

    cfg = ExperimentCfg(
        run=RunCfg(experiment="dash-smoke", seed=0),
        data=DataCfg(
            store=STORE_NAME, scheme="cosmo-sac-2010", split_column="biased_split"
        ),
        predictor=PredictorCfg(
            name="dash_backoff",
            params={
                "store": STORE_NAME,
                "scheme": "cosmo-sac-2010",
                "stores_root": str(STORES_ROOT),
                "tree_folder_path": None,
                "max_depth": 8,
                "attention_threshold": 10,
                "minimum_support": 1,
                "charge_reconciliation": "std_weighted",
            },
        ),
    )
    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.run_dir.is_dir()
    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert result.metrics["n_test"] > 0

    # How much of each split DASH actually covered is on the record: some
    # chaos-store molecules contain atoms outside DASH's published feature
    # vocabulary and fall back to the global mean wholesale.
    import json

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    stats = manifest["match_stats"]
    for split in ("train", "test"):
        assert stats[split]["n_atoms"] > 0
        assert 0 <= stats[split]["n_unmatched_atoms"] <= stats[split]["n_atoms"]
        assert (
            stats[split]["n_unmatched_molecules"] <= stats[split]["n_molecules"]
        )
