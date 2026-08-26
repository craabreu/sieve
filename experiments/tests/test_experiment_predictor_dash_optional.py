"""End-to-end DASHPredictor tests against the real chaos-store and the real
DASH-tree clone (pins.toml's ``[dash_tree]``). Skipped unless rdkit,
cosmolayer, the store, and the clone are all present -- same pattern as
test_experiment_store.py (this directory).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")

from sieve_experiments.data import DEFAULT_STORES_ROOT, REPO_ROOT

STORE_NAME = "chaos-store"
STORES_ROOT = DEFAULT_STORES_ROOT
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


# --- DASHPredictor: load-bearing checks on the real DASH-tree --------------


def test_data_storage_row_count_matches_topology_for_every_branch():
    """The invariant populate_tree_with_sigma_properties/predict_via_
    data_storage_walk both depend on: tree.data_storage[b] must have
    exactly one row per node tree.tree_storage[b] defines, at row position
    == node id -- otherwise writing/reading a new property column by node
    id silently corrupts data. Checked here as a real regression test, not
    just a one-off manual check: if this ever stops holding for any real
    branch, this predictor's whole design needs revisiting.
    """
    import sys

    sys.path.insert(0, str(DASH_TREE_ROOT))
    from serenityff.charge.tree.dash_tree import DASHTree

    tree = DASHTree(preload=True, verbose=False)
    mismatches = [
        b
        for b in tree.tree_storage
        if len(tree.tree_storage[b]) != len(tree.data_storage[b])
    ]
    assert mismatches == [], (
        f"{len(mismatches)} branch(es) have data_storage/tree_storage row "
        f"count mismatches: {mismatches[:5]}"
    )


def test_predict_via_data_storage_walk_matches_get_property_noNAN_on_real_data():
    """predict_via_data_storage_walk is a reimplementation of
    DASHTree.get_property_noNAN's own fallback semantics (deepest ->
    shallowest, first populated node wins), traded for speed instead of
    calling it directly -- so its correctness rests entirely on that
    equivalence actually holding. Cross-checked here against the real
    function, on real matched paths, not just the hand-built fixtures in
    the fast suite: for a sample of real test atoms, every profile bin and
    charge_std predict_via_data_storage_walk returns must equal what
    tree.get_property_noNAN itself returns for the same node path and
    property name.
    """
    import sys

    sys.path.insert(0, str(DASH_TREE_ROOT))
    from sieve_experiments.data import load_molecule_set
    from sieve_experiments.predictors.dash import DASHPredictor

    mset, _ = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=TRAIN_LIMIT,
        stores_root=STORES_ROOT,
    )
    train_mask = np.zeros(mset.n_molecules, dtype=bool)
    train_mask[: TRAIN_LIMIT - TEST_LIMIT] = True
    train = mset.select(train_mask)
    test = mset.select(~train_mask)

    predictor = DASHPredictor(
        store=STORE_NAME, scheme="cosmo-sac-2010", stores_root=str(STORES_ROOT)
    )
    predictor.fit_atoms(train, train, rng=np.random.default_rng(0))
    tree = predictor._load_tree()
    paths = predictor._paths_for(test, split="test")
    props = predictor._props
    assert props is not None

    pred = predictor.predict_atoms(test)
    assert pred.atom_charge_std is not None

    checked = 0
    for i, path in enumerate(paths):
        if not path:
            continue
        raw_path = [path[0][0], *(node_id for _, node_id in path)]
        for b, col in enumerate(props.profile_columns):
            expected = tree.get_property_noNAN(
                matched_node_path=raw_path, property_name=col
            )
            expected = expected if not np.isnan(expected) else props.fallback_profile[b]
            assert pred.atom_profile[i, b] == expected
        expected_std = tree.get_property_noNAN(
            matched_node_path=raw_path, property_name=props.charge_std_column
        )
        expected_std = (
            expected_std if not np.isnan(expected_std) else props.fallback_charge_std
        )
        assert pred.atom_charge_std[i] == expected_std
        checked += 1

    assert checked > 0, "no matched atom in this sample -- test proves nothing"


def test_dash_end_to_end_on_real_store(tmp_path):
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
            name="dash",
            params={
                "store": STORE_NAME,
                "scheme": "cosmo-sac-2010",
                "stores_root": str(STORES_ROOT),
                "max_depth": 8,
                "attention_threshold": 10,
                "charge_reconciliation": "std_weighted",
                "charge_std_floor": 0.1,
            },
        ),
    )
    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.run_dir.is_dir()
    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert result.metrics["n_test"] > 0

    predictions = np.load(result.run_dir / "predictions.npz")
    profile_pred = predictions["mol_profile_pred"]
    n_negative = (profile_pred < 0).sum()
    assert n_negative == 0, (
        f"{n_negative}/{profile_pred.size} rolled-up bins are negative -- "
        "the mean of non-negative training profiles must stay non-negative"
    )
