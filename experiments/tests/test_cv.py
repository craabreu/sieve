"""Fast-suite tests for cv.py's pure algorithmic core (permute_into_folds,
leave_one_group_out) plus the shard-fit/CV-assembly drivers against tiny
synthetic stores -- no real 8.3GB SDF, no DASH-tree clone required for the
pure-core tests; the driver tests need rdkit/pandas/pyarrow (and the real
DASH-tree clone only for the DASH driver tests, gated accordingly)."""

from __future__ import annotations

from typing import Any

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
        max_depth=2,
        codes_path=codes_path,
        config_label="test-config",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(paths) == 5
    for p in paths:
        assert p.exists()

    before = set(paths)
    again = run_sieve_shard_fits(
        store=store,
        n_shards=5,
        max_depth=2,
        codes_path=codes_path,
        config_label="test-config",
        predictor_params={"attributes": ("element",), "edge_attributes": ()},
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    after = set(again)
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

    # Explicitly `dict[str, Any]`: a bare literal's inferred value type
    # (here `tuple[str] | tuple[()] | int`) makes `**params` below type-
    # check every SievePredictor keyword against that one union, since a
    # type checker can't statically know which key lands on which
    # parameter through a `**` unpack of a heterogeneous dict.
    params: dict[str, Any] = {
        "attributes": ("element",),
        "edge_attributes": (),
        "minimum_support": 1,
    }
    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=2,
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
        max_depth=1,
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


def test_truncate_model_matches_a_native_fit_at_every_depth():
    """The claim the whole one-fit-per-shard scheme rests on, and which an
    earlier revision of this module wrongly denied: truncating a deep fit
    reproduces a native shallow fit exactly -- same schema_version, same
    predictions, bit for bit -- under continuation + empirical Bayes, the
    very estimator the old comment said made it impossible."""
    from experiments.cv import truncate_model
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=24, seed=0)
    test = synthetic_molecule_set(n_mol=8, seed=7)
    # dict[str, Any], not a bare literal: `**params` below would otherwise
    # be type-checked against every SievePredictor keyword as one narrow
    # union (see test_run_sieve_cv_... for the same annotation).
    params: dict[str, Any] = {
        "attributes": ("element",),
        "edge_attributes": (),
        "class_estimator": "continuation",
        "shrinkage_weight": "empirical_bayes",
        "minimum_support": 1,
    }

    def fit(depth):
        p = SievePredictor(max_wl_depth=depth, **params)
        p.fit(train, train, rng=np.random.default_rng(0))
        return p

    deep = fit(6)
    for depth in range(0, 7):
        native = fit(depth)
        truncated = SievePredictor(max_wl_depth=depth, **params)
        truncated.set_model(truncate_model(deep._model, depth))

        assert (
            truncated._model.config.schema_version
            == native._model.config.schema_version
        ), depth
        np.testing.assert_array_equal(
            truncated.predict(test).atom_value,
            native.predict(test).atom_value,
            err_msg=f"depth {depth}",
        )


def test_truncate_model_refuses_to_deepen_or_to_touch_neighbor_depth():
    import dataclasses

    from experiments.cv import truncate_model
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=12, seed=0)
    p = SievePredictor(
        attributes=("element",), edge_attributes=(), max_wl_depth=3, minimum_support=1
    )
    p.fit(train, train, rng=np.random.default_rng(0))

    with pytest.raises(ValueError, match="up to depth"):
        truncate_model(p._model, 5)

    # A neighbor_depth model's main WL chain is the *last* level block, so a
    # prefix slice would cut the coarse chain instead -- refused, not risked.
    faked = dataclasses.replace(
        p._model,
        config=dataclasses.replace(
            p._model.config,
            attribute_levels=(("element",), ("degree",)),
            attribute_codes={
                "element": dict(p._model.config.attribute_codes["element"]),
                "degree": {"1": 0, "2": 1},
            },
            neighbor_depth=1,
        ),
    )
    with pytest.raises(ValueError, match="neighbor_depth"):
        truncate_model(faked, 1)


def test_run_dash_cv_refuses_a_depth_truncation_cannot_derive():
    """Depth 1 is the one DASH depth a truncated walk gets measurably
    wrong (the H-atom redirect consumes a depth unit before max_depth is
    checked). dash_depth_sweep has always routed it around truncation;
    run_dash_cv cannot, so it must refuse rather than score it wrongly."""
    from experiments.cv import run_dash_cv

    with pytest.raises(ValueError, match=r"cannot be derived"):
        run_dash_cv(
            store="unused",
            n_shards=10,
            depths=[1, 4],
            repeats=[0],
            max_depth=16,
            allow_dirty=True,
        )


def _sieve_cv_kwargs(tmp_path, *, n_mol=20, n_shards=10, seed=9):
    """Shared setup for the save_predictions tests: a tiny sharded store,
    frozen codes, and one shard set fit at depth 1."""
    from experiments.cv import run_sieve_shard_fits
    from experiments.predictors.sieve_predictor import _build_config, save_codes

    from experiments.tests.helpers import synthetic_molecule_set

    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=seed
    )
    runs_root = tmp_path / "runs"
    whole = synthetic_molecule_set(n_mol=n_mol, seed=seed)
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
    params: dict[str, Any] = {"attributes": ("element",), "edge_attributes": ()}

    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=1,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    return {
        "store": store,
        "n_shards": n_shards,
        "depths": [1],
        "repeats": [0],
        "codes_path": codes_path,
        "config_label": "cfg",
        "predictor_params": params,
        "k": 5,
        "method": "sieve-cfg",
        "runs_root": runs_root,
        "stores_root": stores_root,
        "allow_dirty": True,
    }


def test_cv_runs_write_no_predictions_by_default(tmp_path):
    """A depth sweep repeats everything but atom_target_pred at every
    depth of a given (repeat, fold) -- ~14GB of duplication on the real
    corpus -- so predictions are opt-in, not the default."""
    from experiments.cv import run_sieve_cv

    results = run_sieve_cv(**_sieve_cv_kwargs(tmp_path))

    assert results
    for r in results:
        assert not (r.run_dir / "predictions.npz").exists()
        assert (r.run_dir / "metrics.json").exists()  # metrics still written


def test_cv_runs_write_predictions_when_asked(tmp_path):
    from experiments.cv import run_sieve_cv

    results = run_sieve_cv(save_predictions=True, **_sieve_cv_kwargs(tmp_path, seed=11))

    assert results
    for r in results:
        assert (r.run_dir / "predictions.npz").exists()


def test_run_sieve_cv_reuses_shard_fits_deeper_than_the_requested_depth(tmp_path):
    """Study B asks for one depth shallower than Study A swept, so the shard
    fits on disk are deeper than ``max(depths)``. An earlier revision derived
    the lookup depth from ``max(depths)`` alone and so failed to find them --
    which is what stopped Study B's Sieve arm after its DASH arm had already
    finished. ``fit_depth`` names the depth on disk; truncation supplies the
    rest.
    """
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
        max_wl_depth=2,
        minimum_support=1,
        shrinkage_strength=None,
    )
    codes_path = tmp_path / "codes.json"
    save_codes(config.attribute_codes, config.edge_codes, codes_path)
    params = {"attributes": ("element",), "edge_attributes": ()}

    # Fit at depth 2, the way a sweep would.
    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=2,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )

    kwargs: dict[str, Any] = {
        "store": store,
        "n_shards": n_shards,
        "repeats": [0],
        "codes_path": codes_path,
        "config_label": "cfg",
        "predictor_params": params,
        "k": k,
        "method": "sieve-cfg",
        "stores_root": stores_root,
        "allow_dirty": True,
    }

    # Without fit_depth the lookup goes to depth 1, where nothing was fit.
    with pytest.raises(FileNotFoundError, match="no shard fit"):
        run_sieve_cv(depths=[1], runs_root=tmp_path / "runs-miss", **kwargs)

    results = run_sieve_cv(depths=[1], fit_depth=2, runs_root=runs_root, **kwargs)
    assert len(results) == k

    # And it must not silently accept fits shallower than what is asked for.
    with pytest.raises(ValueError, match="shallower"):
        run_sieve_cv(
            depths=[2],
            fit_depth=1,
            runs_root=tmp_path / "runs-shallow",
            **kwargs,
        )


_VARIANT_SPECS = [
    ("pooled", None),
    ("pooled", "empirical_bayes"),
    ("continuation", None),
    ("continuation", "empirical_bayes"),
]


@pytest.mark.parametrize(("class_estimator", "shrinkage_weight"), _VARIANT_SPECS)
def test_respecify_model_matches_a_native_fit_with_that_estimator(
    class_estimator, shrinkage_weight
):
    """The claim that lets one shard set serve a whole estimator family:
    rewriting ``class_estimator``/``shrinkage_weight`` on a fitted model gives
    exactly the model that was fitted with them in the first place.

    True because those fields are excluded from ``schema_version`` -- they are
    read at predict time and change only how stored numbers are combined, not
    what is stored. Checked against a native fit rather than assumed.
    """
    import numpy as np
    from experiments.cv import EstimatorVariant, respecify_model
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    train = synthetic_molecule_set(n_mol=24, seed=5)
    test = synthetic_molecule_set(n_mol=8, seed=6)
    fitted_as = SievePredictor(
        attributes=("element",),
        edge_attributes=(),
        max_wl_depth=2,
        minimum_support=1,
        class_estimator=class_estimator,
        shrinkage_weight=shrinkage_weight,
    )
    fitted_as.fit(train, train, rng=np.random.default_rng(0))
    native = fitted_as.predict_raw(test)

    # A model fitted under a *different* reading, then respecified.
    other = "continuation" if class_estimator == "pooled" else "pooled"
    fitted_other = SievePredictor(
        attributes=("element",),
        edge_attributes=(),
        max_wl_depth=2,
        minimum_support=1,
        class_estimator=other,
        shrinkage_weight=None,
    )
    fitted_other.fit(train, train, rng=np.random.default_rng(0))
    respecified = respecify_model(
        fitted_other._model,
        EstimatorVariant(
            method="v",
            class_estimator=class_estimator,
            shrinkage_weight=shrinkage_weight,
        ),
    )
    fitted_other.set_model(respecified)
    got = fitted_other.predict_raw(test)

    np.testing.assert_array_equal(got.atom_value, native.atom_value)
    np.testing.assert_array_equal(got.atom_std, native.atom_std)


def test_respecify_model_preserves_schema_version_and_statistics():
    from experiments.cv import EstimatorVariant, respecify_model
    from experiments.predictors.sieve_predictor import SievePredictor

    from experiments.tests.helpers import synthetic_molecule_set

    p = SievePredictor(
        attributes=("element",), edge_attributes=(), max_wl_depth=2, minimum_support=1
    )
    mset = synthetic_molecule_set(n_mol=16, seed=8)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    model = p._model

    out = respecify_model(
        model,
        EstimatorVariant(
            method="v",
            class_estimator="continuation",
            shrinkage_weight="empirical_bayes",
        ),
    )
    assert out.config.schema_version == model.config.schema_version
    assert out.levels is model.levels  # statistics shared, not copied
    assert out.config.class_estimator == "continuation"
    assert out.config.shrinkage_weight == "empirical_bayes"
    assert model.config.class_estimator == "pooled"  # original untouched


def test_run_sieve_cv_variants_share_one_shard_set(tmp_path):
    """Four estimator readings off a single set of shard fits: distinct runs,
    distinct predictions, and no second fit anywhere.
    """
    import json as _json

    from experiments.cv import EstimatorVariant, run_sieve_cv, run_sieve_shard_fits
    from experiments.predictors.sieve_predictor import _build_config, save_codes

    from experiments.tests.helpers import synthetic_molecule_set

    n_mol, n_shards, k = 20, 10, 5
    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=2
    )
    runs_root = tmp_path / "runs"

    config = _build_config(
        synthetic_molecule_set(n_mol=n_mol, seed=2).mols,
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
        max_depth=1,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    from experiments.cv import SHARD_FIT_EXPERIMENT

    def _n_shard_fits() -> int:
        return len(list((runs_root / SHARD_FIT_EXPERIMENT).glob("*__*/tree_stats.npz")))

    n_fits = _n_shard_fits()
    assert n_fits == n_shards, "the count below is only a check if it counts"

    variants = [
        EstimatorVariant(
            method=f"sieve-{ce}{'-eb' if sw else ''}",
            class_estimator=ce,
            shrinkage_weight=sw,
        )
        for ce, sw in _VARIANT_SPECS
    ]
    results = run_sieve_cv(
        store=store,
        n_shards=n_shards,
        depths=[1],
        repeats=[0],
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        variants=variants,
        k=k,
        runs_root=runs_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    assert len(results) == k * len(variants)

    # No shard was refitted to produce three more model families.
    assert _n_shard_fits() == n_fits

    methods = set()
    for r in results:
        cv = _json.loads((r.run_dir / "manifest.json").read_text())["config"]["cv"]
        methods.add(cv["method"])
        assert cv["class_estimator"] in {"pooled", "continuation"}
    assert methods == {v.method for v in variants}

    # Resuming does nothing, per variant as well as per (repeat, fold, depth).
    assert (
        run_sieve_cv(
            store=store,
            n_shards=n_shards,
            depths=[1],
            repeats=[0],
            codes_path=codes_path,
            config_label="cfg",
            predictor_params=params,
            variants=variants,
            k=k,
            runs_root=runs_root,
            stores_root=stores_root,
            allow_dirty=True,
        )
        == []
    )


def test_run_sieve_cv_rejects_variants_with_duplicate_method_names(tmp_path):
    from experiments.cv import EstimatorVariant, run_sieve_cv

    dup = [
        EstimatorVariant(method="same", class_estimator="pooled"),
        EstimatorVariant(method="same", class_estimator="continuation"),
    ]
    with pytest.raises(ValueError, match="distinct method names"):
        run_sieve_cv(
            store="s",
            n_shards=10,
            depths=[1],
            repeats=[0],
            codes_path=tmp_path / "codes.json",
            config_label="cfg",
            variants=dup,
            runs_root=tmp_path / "runs",
            allow_dirty=True,
        )


def _sieve_cv_cache_setup(tmp_path, *, n_mol=20, n_shards=10):
    from experiments.cv import run_sieve_shard_fits
    from experiments.predictors.sieve_predictor import _build_config, save_codes

    from experiments.tests.helpers import synthetic_molecule_set

    store, stores_root = _write_shard_store(
        tmp_path, n_mol=n_mol, n_shards=n_shards, seed=2
    )
    config = _build_config(
        synthetic_molecule_set(n_mol=n_mol, seed=2).mols,
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
    fits_root = tmp_path / "fits"
    run_sieve_shard_fits(
        store=store,
        n_shards=n_shards,
        max_depth=1,
        codes_path=codes_path,
        config_label="cfg",
        predictor_params=params,
        runs_root=fits_root,
        stores_root=stores_root,
        allow_dirty=True,
    )
    return {
        "store": store,
        "n_shards": n_shards,
        "depths": [1],
        "repeats": [0],
        "codes_path": codes_path,
        "config_label": "cfg",
        "predictor_params": params,
        "k": 5,
        "stores_root": stores_root,
        "allow_dirty": True,
    }, fits_root


def _metrics_by_batch(results):
    """Scores only -- the ``time/*`` entries differ between a cache hit and a
    miss by design, and comparing them would test the clock."""
    import json as _json

    out = {}
    for r in results:
        manifest = _json.loads((r.run_dir / "manifest.json").read_text())
        out[manifest["config"]["run"]["batch_id"]] = {
            k: v for k, v in r.metrics.items() if not k.startswith("time/")
        }
    return out


def test_sieve_model_cache_hit_scores_identically_to_a_miss(tmp_path):
    """A cache is only worth having if a hit is indistinguishable from a
    miss. Same store, same partition, same metrics -- exactly."""
    import shutil

    from experiments.cv import run_sieve_cv

    kwargs, fits_root = _sieve_cv_cache_setup(tmp_path)
    cache = tmp_path / "cache"

    # Cold: merges, and populates the cache.
    cold = run_sieve_cv(
        **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "cold"), model_cache=cache
    )
    assert len(cold) == kwargs["k"]
    cached = sorted(p.name for p in cache.rglob("*.npz"))
    assert len(cached) == kwargs["k"], cached

    # Warm: same answer, without opening a single shard fit. Proven by
    # hiding them -- a cache miss here would raise FileNotFoundError.
    warm_runs = _copy_runs(fits_root, tmp_path / "warm")
    shutil.rmtree(warm_runs / "cv-shard-fits")
    with pytest.raises(FileNotFoundError):
        run_sieve_cv(**kwargs, runs_root=warm_runs, model_cache=cache)

    # ... and with the fits present, the hit path is exercised and agrees.
    warm = run_sieve_cv(
        **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "warm2"), model_cache=cache
    )
    assert _metrics_by_batch(cold) == _metrics_by_batch(warm)


def test_model_cache_refuses_an_entry_for_a_different_partition(tmp_path):
    """Silently rebuilding would destroy whatever wrote the mismatched entry;
    the sidecar makes the disagreement visible instead."""
    import json as _json

    from experiments.cv import run_sieve_cv

    kwargs, fits_root = _sieve_cv_cache_setup(tmp_path)
    cache = tmp_path / "cache"
    run_sieve_cv(
        **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "a"), model_cache=cache
    )

    sidecar = next(cache.rglob("*.json"))
    payload = _json.loads(sidecar.read_text())
    payload["train_shards"] = ["s99"]
    sidecar.write_text(_json.dumps(payload))

    with pytest.raises(RuntimeError, match="does not match what was asked for"):
        run_sieve_cv(
            **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "b"), model_cache=cache
        )


def test_model_cache_treats_a_missing_sidecar_as_a_miss(tmp_path):
    """The sidecar is written last, so an interrupted save must read as a
    miss rather than as a truncated hit."""
    from experiments.cv import run_sieve_cv

    kwargs, fits_root = _sieve_cv_cache_setup(tmp_path)
    cache = tmp_path / "cache"
    first = run_sieve_cv(
        **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "a"), model_cache=cache
    )

    next(cache.rglob("*.json")).unlink()
    second = run_sieve_cv(
        **kwargs, runs_root=_copy_runs(fits_root, tmp_path / "b"), model_cache=cache
    )
    assert _metrics_by_batch(first) == _metrics_by_batch(second)
    assert len(list(cache.rglob("*.json"))) == kwargs["k"]  # rewritten


def _copy_runs(fits_root, dest):
    """A fresh runs_root carrying the shard fits, so each call writes its own
    CV runs instead of tripping the already-done guard."""
    import shutil

    shutil.copytree(fits_root, dest)
    return dest
