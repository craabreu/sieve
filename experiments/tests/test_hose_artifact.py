"""HOSE state: the merge is exact, and it refuses what it cannot combine.

The sharding scheme rests entirely on the first claim -- a fold's training
model is 40 shards merged, never a 40-shard fit -- so it is pinned against a
direct fit rather than against itself.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from experiments.tests.helpers import synthetic_molecule_set

hosegen = pytest.importorskip("hosegen", reason="the optional `hose` extra")


def _fit(mset, radius):
    from experiments.predictors.hose import HoseLookupPredictor

    p = HoseLookupPredictor(max_radius=radius)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    return p


def test_merged_shards_equal_a_direct_fit_on_the_union():
    """The claim the CV driver depends on: merging per-shard states gives the
    model a single fit over the union would have produced -- same keys, same
    means, same predictions."""
    from experiments.data import concat_molecule_sets
    from experiments.hose_artifact import fold_hose_states

    a = synthetic_molecule_set(n_mol=8, seed=0)
    b = synthetic_molecule_set(n_mol=8, seed=1)
    union = concat_molecule_sets([a, b])

    direct = _fit(union, 3)
    merged = _fit(a, 3)
    merged.set_model_state(
        fold_hose_states([_fit(a, 3).model_state(), _fit(b, 3).model_state()])
    )

    assert merged._tables is not None and direct._tables is not None
    for k in range(1, 4):
        assert set(merged._tables[k]) == set(direct._tables[k]), k
        for key, (mean, count) in direct._tables[k].items():
            got_mean, got_count = merged._tables[k][key]
            assert got_count == count, (k, key)
            assert got_mean == pytest.approx(mean, rel=1e-12, abs=1e-12), (k, key)
    assert merged._global_mean == pytest.approx(direct._global_mean, rel=1e-12)

    test = synthetic_molecule_set(n_mol=4, seed=7)
    np.testing.assert_allclose(
        merged.predict(test).atom_value,
        direct.predict(test).atom_value,
        rtol=1e-12,
        atol=1e-12,
    )


def test_merge_refuses_two_different_radii():
    """A radius-5 state's 3-sphere table is not a radius-3 state's: the
    generator re-renders shallower spheres when it goes deeper (spec s7), so
    pooling them would mix two linearizations."""
    from experiments.hose_artifact import merge_hose_states

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    with pytest.raises(ValueError, match="different radii"):
        merge_hose_states(_fit(mset, 2).model_state(), _fit(mset, 3).model_state())


def test_state_round_trips_through_disk(tmp_path):
    from experiments.hose_artifact import load_hose_state, save_hose_state

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    state = _fit(mset, 3).model_state()
    path = tmp_path / "hose.npz"
    save_hose_state(state, path)
    back = load_hose_state(path)

    assert back.max_radius == state.max_radius
    assert back.global_count == state.global_count
    assert back.global_sum == pytest.approx(state.global_sum, rel=1e-12)
    for k in range(1, 4):
        assert back.tables[k] == pytest.approx(state.tables[k])


def test_predict_raw_reports_a_unit_std():
    """This arm has no variance to report; the std exists for signature
    parity and equal_weighted discards it."""
    mset = synthetic_molecule_set(n_mol=6, seed=0)
    p = _fit(mset, 2)
    raw = p.predict_raw(mset)
    np.testing.assert_array_equal(raw.atom_std, np.ones_like(raw.atom_value))
    np.testing.assert_array_equal(raw.atom_value, p.predict(mset).atom_value)


def test_run_hose_cv_end_to_end_on_a_sharded_store(tmp_path):
    """The driver, against a tiny sharded store: every (repeat, fold, radius)
    lands one run, the fold's model is the merge of its training shards, and
    a second call is a no-op."""
    from experiments.cv import run_hose_cv, run_hose_shard_fits

    from experiments.tests.test_cv import _write_shard_store

    n_mol, n_shards, k = 20, 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=2
    )
    runs_root = tmp_path / "runs"
    # dict[str, Any] for the reason test_cv.py records at its own `common`:
    # `**common` would otherwise be checked against every keyword of
    # run_hose_shard_fits/run_hose_cv as one inferred value union.
    common: dict[str, Any] = {
        "store": store,
        "n_shards": n_shards,
        "runs_root": runs_root,
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    for radius in (1, 2):
        run_hose_shard_fits(radius=radius, **common)

    results = run_hose_cv(radii=[1, 2], repeats=[0], k=k, **common)
    assert len(results) == k * 2

    for r in results:
        manifest = json.loads((r.run_dir / "manifest.json").read_text())
        cv = manifest["config"]["cv"]
        assert cv["method"] == "hose"
        assert cv["depth"] in (1, 2)
        assert "rmse" in r.metrics

    # idempotent: the guard the workflow relies on to resume
    assert run_hose_cv(radii=[1, 2], repeats=[0], k=k, **common) == []


def test_run_hose_cv_names_the_radius_whose_shards_are_missing(tmp_path):
    """A radius with no shard fits must fail loudly, not silently score a
    different radius -- the one mistake spec section 7 exists to prevent."""
    from experiments.cv import run_hose_cv, run_hose_shard_fits

    from experiments.tests.test_cv import _write_shard_store

    store, stores_root = _write_shard_store(tmp_path, n_mol=12, n_shards=4, seed=0)
    runs_root = tmp_path / "runs"
    common: dict[str, Any] = {
        "store": store,
        "n_shards": 4,
        "runs_root": runs_root,
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    run_hose_shard_fits(radius=2, **common)
    with pytest.raises(FileNotFoundError, match="radius 3"):
        run_hose_cv(radii=[3], repeats=[0], k=2, **common)


def test_merge_states_entry_point_folds_shards_and_refuses_mixed_radii(tmp_path):
    """The seam `experiments merge-states` discovers by name. It must agree
    with a direct fit on the union, and must refuse shards of different radii
    rather than pooling two linearizations (spec s7)."""
    from experiments.data import concat_molecule_sets
    from experiments.hose_artifact import load_hose_state
    from experiments.predictors.hose import HoseLookupPredictor

    a = synthetic_molecule_set(n_mol=8, seed=0)
    b = synthetic_molecule_set(n_mol=8, seed=1)

    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    _fit(a, 3).save_model_state(pa)
    _fit(b, 3).save_model_state(pb)

    out = tmp_path / "merged.npz"
    HoseLookupPredictor.merge_states([pa, pb], out)

    merged = HoseLookupPredictor(max_radius=3)
    merged.set_model_state(load_hose_state(out))
    direct = _fit(concat_molecule_sets([a, b]), 3)

    test = synthetic_molecule_set(n_mol=4, seed=7)
    np.testing.assert_allclose(
        merged.predict(test).atom_value,
        direct.predict(test).atom_value,
        rtol=1e-12,
        atol=1e-12,
    )

    shallow = tmp_path / "r2.npz"
    _fit(a, 2).save_model_state(shallow)
    with pytest.raises(ValueError, match="different radii"):
        HoseLookupPredictor.merge_states([pa, shallow], tmp_path / "bad.npz")

    with pytest.raises(ValueError, match="at least one shard"):
        HoseLookupPredictor.merge_states([], tmp_path / "empty.npz")
