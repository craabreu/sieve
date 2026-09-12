"""Fast-suite tests for cv.py's pure algorithmic core (permute_into_folds,
leave_one_group_out) plus the shard-fit/CV-assembly drivers against tiny
synthetic stores -- no real 8.3GB SDF, no DASH-tree clone required for the
pure-core tests; the driver tests need rdkit/pandas/pyarrow (and the real
DASH-tree clone only for the DASH driver tests, gated accordingly)."""

from __future__ import annotations

import numpy as np
import pytest


def test_shard_ids_widths_are_zero_padded():
    from experiments.cv import shard_ids

    assert shard_ids(3) == ["s00", "s01", "s02"]
    assert shard_ids(11)[-1] == "s10"


def test_permute_into_folds_covers_every_item_exactly_once():
    from experiments.cv import permute_into_folds

    items = list(range(25))
    groups = permute_into_folds(items, k=5, seed=0)

    assert len(groups) == 5
    assert all(len(g) == 5 for g in groups)
    covered = sorted(v for g in groups for v in g)
    assert covered == items


def test_permute_into_folds_rejects_a_non_divisible_count():
    from experiments.cv import permute_into_folds

    with pytest.raises(ValueError, match="divisible"):
        permute_into_folds(list(range(7)), k=5, seed=0)


def test_permute_into_folds_differs_across_seeds():
    from experiments.cv import permute_into_folds

    items = list(range(25))
    a = permute_into_folds(items, k=5, seed=0)
    b = permute_into_folds(items, k=5, seed=1)
    assert a != b


def test_permute_into_folds_is_deterministic_for_one_seed():
    from experiments.cv import permute_into_folds

    items = list(range(25))
    a = permute_into_folds(items, k=5, seed=7)
    b = permute_into_folds(items, k=5, seed=7)
    assert a == b


def test_leave_one_group_out_matches_brute_force_sum():
    from experiments.cv import leave_one_group_out

    groups = [1, 2, 3, 4, 5]
    out = leave_one_group_out(groups, merge=lambda a, b: a + b)
    total = sum(groups)
    assert out == [total - g for g in groups]


def test_leave_one_group_out_requires_at_least_two_groups():
    from experiments.cv import leave_one_group_out

    with pytest.raises(ValueError, match="at least 2"):
        leave_one_group_out([1], merge=lambda a, b: a + b)


def test_leave_one_group_out_merge_never_sees_the_none_sentinel():
    from experiments.cv import leave_one_group_out

    def _merge(a, b):
        assert a is not None and b is not None
        return a + b

    out = leave_one_group_out([10, 20, 30], merge=_merge)
    assert out == [50, 40, 30]


def test_leave_one_group_out_with_string_concat_and_five_groups():
    """A less trivial merge op, and the k=5 shape the CV design actually
    uses -- confirms the prefix/suffix scheme generalizes past addition
    and past k=3."""
    from experiments.cv import leave_one_group_out

    groups = ["a", "b", "c", "d", "e"]
    out = leave_one_group_out(groups, merge=lambda a, b: a + b)
    assert out == ["bcde", "acde", "abde", "abce", "abcd"]


def test_dash_shard_batch_id_and_sieve_shard_batch_id_disambiguate():
    from experiments.cv import cv_batch_id, dash_shard_batch_id, sieve_shard_batch_id

    assert dash_shard_batch_id("s00") == "fit-dash-s00"
    assert sieve_shard_batch_id("element-eb", 6, "s03") == "fit-sieve-element-eb-w6-s03"
    assert cv_batch_id(repeat=2, fold=3, method="dash", depth=16) == "r2-f3-dash-w16"


pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _write_shard_store(tmp_path, *, n_mol=40, n_shards=10, seed=0, name="synthetic"):
    """A tiny on-disk store carrying a ``shard`` column directly (bypassing
    prepare_dash's own SDF parse/cluster path -- this is about cv.py's own
    plumbing, not clustering), built from ``synthetic_molecule_set``'s own
    known-good molecules. Every row is treated as its own shard-assignable
    unit (no multi-conformer grouping needed for these tests)."""
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


def test_run_sieve_shard_fits_is_idempotent(tmp_path):
    from experiments.cv import run_sieve_shard_fits
    from experiments.predictors.sieve_predictor import (
        _build_config,
        save_codes,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    store, stores_root = _write_shard_store(tmp_path, n_mol=20, n_shards=5)
    runs_root = tmp_path / "runs"

    whole = synthetic_molecule_set(n_mol=20, seed=0)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=2,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    paths = run_sieve_shard_fits(
        store=store,
        n_shards=5,
        depths=[2],
        codes_path=codes_path,
        config_label="test-config",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(paths[2]) == 5
    for p in paths[2]:
        assert p.exists()

    before = {p for group in paths.values() for p in group}
    again = run_sieve_shard_fits(
        store=store,
        n_shards=5,
        depths=[2],
        codes_path=codes_path,
        config_label="test-config",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    after = {p for group in again.values() for p in group}
    assert before == after  # no new shard fits were written


def test_run_sieve_cv_assembly_matches_a_direct_fit_on_the_complement(tmp_path):
    """The load-bearing exactness claim: a CV sample's assembled training
    model (merge of the complementary shards) must predict identically to
    a single Sieve fit on the union of those same molecules, done
    directly -- confirming the shard/merge scheme changes nothing about
    what gets learned."""
    from experiments.cv import (
        build_cv_plan,
        run_sieve_cv,
        run_sieve_shard_fits,
        shard_ids,
    )
    from experiments.predictors.sieve_predictor import (
        SievePredictor,
        _build_config,
        save_codes,
    )

    from experiments.tests.helpers import synthetic_molecule_set

    n_mol, n_shards, k = 20, 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=1
    )
    runs_root = tmp_path / "runs"

    whole = synthetic_molecule_set(n_mol=n_mol, seed=1)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=2,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)

    params = {
        "attributes": ("element",),
        "edge_attributes": (),
        "minimum_support": 1,
    }
    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        depths=[2],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )

    results = run_sieve_cv(
        store=store,
        n_shards=n_shards,
        depths=[2],
        repeats=[0],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        k=k,
        method="sieve-cfg",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(results) == k  # one run per fold, at the one requested depth

    # Independently re-derive fold 0's training set and fit it directly.
    plan = build_cv_plan(shard_ids(n_shards), k=k, repeat=0)
    held_out_group = set(plan.groups[0])
    ids_by_row = [f"s{i % n_shards:02d}" for i in range(n_mol)]
    train_mask = np.array([sid not in held_out_group for sid in ids_by_row])
    train_direct = whole.select(train_mask)

    direct = SievePredictor(codes_path=codes_path, max_wl_depth=2, **params)
    direct.fit(train_direct, train_direct, rng=np.random.default_rng(0))

    held_out_mask = ~train_mask
    held_out = whole.select(held_out_mask)
    direct_pred = direct.predict(held_out)

    # The CV run's own recorded metrics must match scoring that same
    # direct prediction -- i.e. the assembled model is exactly the direct
    # fit, not merely close to it.
    from experiments.metrics import regression_metrics

    expected = regression_metrics(held_out.atom_target, direct_pred.atom_value)
    fold0 = next(r for r in results if r.manifest["config"]["cv"]["fold"] == 0)
    assert fold0.metrics["mae"] == pytest.approx(expected["mae"], abs=1e-9)
    assert fold0.metrics["rmse"] == pytest.approx(expected["rmse"], abs=1e-9)


def test_run_sieve_cv_is_idempotent(tmp_path):
    from experiments.cv import run_sieve_cv, run_sieve_shard_fits
    from experiments.predictors.sieve_predictor import _build_config, save_codes

    from experiments.tests.helpers import synthetic_molecule_set

    n_mol, n_shards, k = 20, 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=2
    )
    runs_root = tmp_path / "runs"

    whole = synthetic_molecule_set(n_mol=n_mol, seed=2)
    config = _build_config(
        whole.mols,
        attributes=("element",),
        edge_attributes=(),
        target_dim=1,
        max_wl_depth=1,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)
    params = {"attributes": ("element",), "edge_attributes": ()}

    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        depths=[1],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    first = run_sieve_cv(
        store=store,
        n_shards=n_shards,
        depths=[1],
        repeats=[0],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        k=k,
        method="sieve-cfg",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(first) == k

    second = run_sieve_cv(
        store=store,
        n_shards=n_shards,
        depths=[1],
        repeats=[0],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        k=k,
        method="sieve-cfg",
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert second == []  # every (repeat, fold, depth) already done
