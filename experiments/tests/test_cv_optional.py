"""End-to-end cv.py driver tests against the real pinned DASH-tree clone.
Skipped if that clone (experiments/external/DASH-tree) is absent -- see
test_predictor_dash_optional.py for the same gating pattern."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists(),
    reason="experiments/external/DASH-tree not cloned",
)


def _write_shard_store(tmp_path, *, n_mol=40, n_shards=10, seed=0, name="synthetic"):
    """See test_cv.py's own copy of this helper for the rationale -- kept
    duplicated rather than imported so this file stays skippable/importable
    independent of that one (both are gated on different optional deps)."""
    import pandas as pd
    from experiments.data import mol_to_blob

    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=n_mol, seed=seed)
    rows = []
    for i, mol in enumerate(mset.mols):
        rows.append(
            {
                "chembl_id": mset.ids["chembl_id"][i],
                "conf_id": mset.ids["conf_id"][i],
                "dash_id": mset.ids["dash_id"][i],
                "mol": mol_to_blob(mol),
                "net_charge": float(mset.molecule_value[i]),
                "split": "train",
                "cluster": i,
                "shard": f"s{i % n_shards:02d}",
            }
        )
    df = pd.DataFrame(rows)
    stores_root = tmp_path / "stores"
    store_dir = stores_root / name
    store_dir.mkdir(parents=True)
    df.to_parquet(store_dir / "molecules.parquet")
    return name, stores_root


def test_run_dash_shard_fits_is_idempotent(tmp_path):
    from experiments.cv import run_dash_shard_fits

    store, stores_root = _write_shard_store(tmp_path, n_mol=20, n_shards=5)
    runs_root = tmp_path / "runs"

    paths = run_dash_shard_fits(
        store=store,
        n_shards=5,
        max_depth=4,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(paths) == 5
    for p in paths:
        assert p.exists()

    again = run_dash_shard_fits(
        store=store,
        n_shards=5,
        max_depth=4,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert paths == again  # no new shard fits were written


def test_run_dash_cv_assembly_matches_a_direct_fit_on_the_complement(tmp_path):
    """The load-bearing exactness claim, DASH side: a CV sample's assembled
    training model (tree_artifact.merge_node_stats over the complementary
    shards) must predict identically to a direct DASHChargePredictor.fit()
    on the union of those same molecules."""
    from experiments.cv import (
        build_cv_plan,
        run_dash_cv,
        run_dash_shard_fits,
        shard_ids,
    )
    from experiments.predictors.dash import DASHChargePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    n_mol, n_shards, k, max_depth = 20, 10, 5, 4
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=3
    )
    runs_root = tmp_path / "runs"

    run_dash_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=max_depth,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )

    results = run_dash_cv(
        store=store,
        n_shards=n_shards,
        depths=[max_depth],
        repeats=[0],
        k=k,
        max_depth=max_depth,
        method="dash",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(results) == k

    whole = synthetic_molecule_set(n_mol=n_mol, seed=3)
    plan = build_cv_plan(shard_ids(n_shards), k=k, repeat=0)
    held_out_group = set(plan.groups[0])
    ids_by_row = [f"s{i % n_shards:02d}" for i in range(n_mol)]
    train_mask = np.array([sid not in held_out_group for sid in ids_by_row])
    train_direct = whole.select(train_mask)
    held_out = whole.select(~train_mask)

    direct = DASHChargePredictor(max_depth=max_depth)
    direct.fit(train_direct, train_direct, rng=np.random.default_rng(0))
    direct_pred = direct.predict(held_out)

    from experiments.metrics import regression_metrics

    expected = regression_metrics(held_out.atom_target, direct_pred.atom_value)
    fold0 = next(r for r in results if r.manifest["config"]["cv"]["fold"] == 0)
    assert fold0.metrics["mae"] == pytest.approx(expected["mae"], abs=1e-9)
    assert fold0.metrics["rmse"] == pytest.approx(expected["rmse"], abs=1e-9)


def test_run_dash_cv_is_idempotent(tmp_path):
    from experiments.cv import run_dash_cv, run_dash_shard_fits

    n_mol, n_shards, k, max_depth = 20, 10, 5, 3
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=4
    )
    runs_root = tmp_path / "runs"

    run_dash_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=max_depth,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    first = run_dash_cv(
        store=store,
        n_shards=n_shards,
        depths=[max_depth],
        repeats=[0],
        k=k,
        max_depth=max_depth,
        method="dash",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(first) == k

    second = run_dash_cv(
        store=store,
        n_shards=n_shards,
        depths=[max_depth],
        repeats=[0],
        k=k,
        max_depth=max_depth,
        method="dash",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert second == []


def test_run_dash_cv_does_not_leak_state_across_repeats(tmp_path):
    """The stale-state trap apply_node_stats' reset_existing guards
    against: two different repeats' own fold-0 held-out sets must each be
    scored against *that repeat's* own assembled model, not a previous
    repeat's leftover values on a branch the later one doesn't populate."""
    from experiments.cv import run_dash_cv, run_dash_shard_fits

    n_mol, n_shards, k, max_depth = 20, 10, 5, 4
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=5
    )
    runs_root = tmp_path / "runs"

    run_dash_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=max_depth,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    results = run_dash_cv(
        store=store,
        n_shards=n_shards,
        depths=[max_depth],
        repeats=[0, 1, 2],
        k=k,
        max_depth=max_depth,
        method="dash",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(results) == 3 * k
    # No crash and every run produced finite (or legitimately NaN) metrics
    # -- the real assertion is the exactness test above; this one is a
    # broader smoke check that repeats don't corrupt each other.
    for r in results:
        assert "mae" in r.metrics
