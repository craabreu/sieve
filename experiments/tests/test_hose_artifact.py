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


def test_state_carries_a_second_moment():
    from experiments.hose_artifact import HoseState

    state = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    assert state.has_second_moment
    assert state.tables[1]["C"] == (3.0, 5.0, 2)


def test_merge_sums_all_three_columns():
    from experiments.hose_artifact import HoseState, merge_hose_states

    a = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    b = HoseState(
        1, ({}, {"C": (1.0, 1.0, 1), "N": (2.0, 4.0, 1)}), 3.0, 2, global_sumsq=5.0
    )
    merged = merge_hose_states(a, b)
    assert merged.tables[1]["C"] == (4.0, 6.0, 3)
    assert merged.tables[1]["N"] == (2.0, 4.0, 1)
    assert merged.has_second_moment
    assert merged.global_sumsq == pytest.approx(10.0)


def test_a_legacy_state_still_loads_and_merges(tmp_path):
    """Pre-existing artifacts must keep working; only the analytic call may
    refuse them."""
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        max_radius=np.asarray(1),
        global_sum=np.asarray(3.0),
        global_count=np.asarray(2),
        r1_keys=np.asarray(["C"], dtype=np.str_),
        r1_sum=np.asarray([3.0]),
        r1_count=np.asarray([2]),
    )
    from experiments.hose_artifact import load_hose_state, merge_hose_states

    state = load_hose_state(path)
    assert not state.has_second_moment
    assert state.tables[1]["C"][0] == 3.0
    assert state.tables[1]["C"][2] == 2
    merged = merge_hose_states(state, state)
    assert not merged.has_second_moment
    assert merged.tables[1]["C"][2] == 4


def test_a_new_state_round_trips_its_second_moment(tmp_path):
    from experiments.hose_artifact import HoseState, load_hose_state, save_hose_state

    state = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    path = tmp_path / "hose.npz"
    save_hose_state(state, path)
    back = load_hose_state(path)
    assert back.has_second_moment
    assert back.tables[1]["C"] == pytest.approx((3.0, 5.0, 2))
    assert back.global_sumsq == pytest.approx(5.0)


def _fit_numbered(mset, radius, numbering):
    from experiments.predictors.hose import HoseLookupPredictor

    p = HoseLookupPredictor(max_radius=radius, numbering=numbering)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    return p


def test_a_state_carries_its_numbering_through_disk(tmp_path):
    from experiments.hose_artifact import load_hose_state, save_hose_state

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    assert _fit(mset, 2).model_state().numbering == "stored"
    state = _fit_numbered(mset, 2, "canonical").model_state()
    assert state.numbering == "canonical"
    path = tmp_path / "hose.npz"
    save_hose_state(state, path)
    assert load_hose_state(path).numbering == "canonical"


def test_a_state_saved_before_numbering_existed_is_stored(tmp_path):
    from experiments.hose_artifact import load_hose_state

    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        max_radius=np.asarray(1),
        global_sum=np.asarray(3.0),
        global_count=np.asarray(2),
        r1_keys=np.asarray(["C"], dtype=np.str_),
        r1_sum=np.asarray([3.0]),
        r1_count=np.asarray([2]),
    )
    assert load_hose_state(path).numbering == "stored"


def test_merge_refuses_two_different_numberings():
    """Codes written under two numberings are two vocabularies: the same
    environment can be spelled differently in each."""
    from experiments.hose_artifact import merge_hose_states

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    with pytest.raises(ValueError, match="numbering"):
        merge_hose_states(
            _fit(mset, 2).model_state(),
            _fit_numbered(mset, 2, "canonical").model_state(),
        )


def test_a_predictor_adopts_the_numbering_of_the_state_it_is_given():
    """As with the radius, the state's own numbering wins: a table of
    canonical codes looked up with stored-order codes would miss."""
    from experiments.predictors.hose import HoseLookupPredictor

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    fitted = _fit_numbered(mset, 3, "canonical")
    adopted = HoseLookupPredictor(max_radius=3)
    adopted.set_model_state(fitted.model_state())
    assert adopted.numbering == "canonical"
    np.testing.assert_array_equal(
        adopted.predict(mset).atom_value, fitted.predict(mset).atom_value
    )


def test_run_hose_cv_with_canonical_numbering_uses_its_own_shard_fits(tmp_path):
    """The canonical arm has its own shard set, named apart from the stored
    one, and the driver refuses to score it from the other set."""
    from experiments.cv import run_hose_cv, run_hose_shard_fits
    from experiments.hose_artifact import load_hose_state

    from experiments.tests.test_cv import _write_shard_store

    n_shards, k = 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=20, n_shards=n_shards, seed=2
    )
    runs_root = tmp_path / "runs"
    common: dict[str, Any] = {
        "store": store,
        "n_shards": n_shards,
        "runs_root": runs_root,
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    run_hose_shard_fits(radius=2, **common)
    with pytest.raises(FileNotFoundError, match="canonical"):
        run_hose_cv(radii=[2], repeats=[0], k=k, numbering="canonical", **common)

    paths = run_hose_shard_fits(radius=2, numbering="canonical", **common)
    assert len(paths) == n_shards
    for path in paths:
        assert path.parent.name.startswith("fit-hose-canonical-r2-s")
        assert load_hose_state(path).numbering == "canonical"
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["numbering"] == "canonical"

    results = run_hose_cv(
        radii=[2],
        repeats=[0],
        k=k,
        numbering="canonical",
        method="hose-canonical",
        **common,
    )
    assert len(results) == k
    for r in results:
        cv = json.loads((r.run_dir / "manifest.json").read_text())["config"]["cv"]
        assert cv["method"] == "hose-canonical"


def _fit_keyed(mset, radius, keys):
    from experiments.predictors.hose import HoseLookupPredictor

    p = HoseLookupPredictor(max_radius=radius, keys=keys)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    return p


def test_a_state_carries_its_keys_through_disk(tmp_path):
    from experiments.hose_artifact import load_hose_state, save_hose_state

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    assert _fit(mset, 2).model_state().keys == "code"
    state = _fit_keyed(mset, 2, "closure-tree").model_state()
    assert state.keys == "closure-tree"
    path = tmp_path / "hose.npz"
    save_hose_state(state, path)
    assert load_hose_state(path).keys == "closure-tree"
    save_hose_state(_fit(mset, 2).model_state(), path)
    with np.load(path) as z:
        assert "keys" not in z.files


def test_merge_refuses_two_different_key_kinds():
    from experiments.hose_artifact import merge_hose_states

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    with pytest.raises(ValueError, match="keys"):
        merge_hose_states(
            _fit_keyed(mset, 2, "tree").model_state(),
            _fit_keyed(mset, 2, "closure-tree").model_state(),
        )


def test_a_predictor_adopts_the_keys_of_the_state_it_is_given():
    from experiments.predictors.hose import HoseLookupPredictor

    mset = synthetic_molecule_set(n_mol=6, seed=0)
    fitted = _fit_keyed(mset, 3, "tree")
    adopted = HoseLookupPredictor(max_radius=3)
    adopted.set_model_state(fitted.model_state())
    assert adopted.keys == "tree"
    np.testing.assert_array_equal(
        adopted.predict(mset).atom_value, fitted.predict(mset).atom_value
    )


def test_shard_fits_of_each_key_kind_are_named_apart():
    from experiments.cv import hose_shard_batch_id

    assert hose_shard_batch_id(5, "s01") == "fit-hose-r5-s01"
    assert hose_shard_batch_id(5, "s01", "canonical") == "fit-hose-canonical-r5-s01"
    # tree fits carry roots, and a name of their own apart from older fits
    assert hose_shard_batch_id(5, "s01", keys="tree") == "fit-hose-tree-rooted-r5-s01"
    assert (
        hose_shard_batch_id(5, "s01", keys="closure-tree")
        == "fit-hose-closure-tree-rooted-r5-s01"
    )


def test_run_hose_cv_with_tree_keys_uses_its_own_shard_fits(tmp_path):
    from experiments.cv import run_hose_cv, run_hose_shard_fits
    from experiments.hose_artifact import load_hose_state

    from experiments.tests.test_cv import _write_shard_store

    n_shards, k = 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=20, n_shards=n_shards, seed=2
    )
    common: dict[str, Any] = {
        "store": store,
        "n_shards": n_shards,
        "runs_root": tmp_path / "runs",
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    run_hose_shard_fits(radius=2, **common)
    with pytest.raises(FileNotFoundError, match="closure-tree"):
        run_hose_cv(radii=[2], repeats=[0], k=k, keys="closure-tree", **common)

    paths = run_hose_shard_fits(radius=2, keys="closure-tree", **common)
    assert len(paths) == n_shards
    for path in paths:
        assert path.parent.name.startswith("fit-hose-closure-tree-rooted-r2-s")
        assert load_hose_state(path).keys == "closure-tree"
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["keys"] == "closure-tree"
    results = run_hose_cv(
        radii=[2],
        repeats=[0],
        k=k,
        keys="closure-tree",
        method="hose-ctree",
        **common,
    )
    assert len(results) == k


def test_run_hose_cv_reads_tree_shard_fits_with_sieves_estimator(tmp_path):
    """Tree shard fits carry their roots, so one shard set serves the raw
    lookup and Sieve's estimators alike."""
    from experiments.cv import run_hose_cv, run_hose_shard_fits
    from experiments.hose_artifact import load_hose_state

    from experiments.tests.test_cv import _write_shard_store

    n_shards, k = 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=20, n_shards=n_shards, seed=2
    )
    common: dict[str, Any] = {
        "store": store,
        "n_shards": n_shards,
        "runs_root": tmp_path / "runs",
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    paths = run_hose_shard_fits(radius=2, keys="closure-tree", **common)
    assert all(load_hose_state(p).roots for p in paths)
    by_estimator = {}
    for estimator in ("raw", "continuation", "continuation-eb"):
        results = run_hose_cv(
            radii=[2],
            repeats=[0],
            k=k,
            keys="closure-tree",
            estimator=estimator,
            method=f"hose-ctree-{estimator}",
            **common,
        )
        assert len(results) == k
        by_estimator[estimator] = [r.metrics["rmse"] for r in results]
    assert by_estimator["raw"] != by_estimator["continuation"]
