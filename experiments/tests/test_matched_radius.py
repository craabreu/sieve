"""The radius that answered each held-out atom of a saved CV run."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# --- the two readings of a radius -------------------------------------------


def test_sieve_positions_are_mapped_through_the_backoff_path():
    from experiments.matched_radius import sieve_matched_radius

    model = SimpleNamespace(config=SimpleNamespace(backoff_path=(0, 2, 3)))
    prediction = SimpleNamespace(matched_level=np.array([-1, 0, 1, 2, 1]))
    radius = sieve_matched_radius(model, prediction)
    np.testing.assert_array_equal(radius, [-1, 0, 2, 3, 2])
    assert radius.dtype == np.int8


def test_the_hose_global_mean_is_no_match():
    from experiments.matched_radius import UNMATCHED, hose_matched_radius

    radius = hose_matched_radius(np.array([0, 1, 5, 0, 3]))
    np.testing.assert_array_equal(radius, [UNMATCHED, 1, 5, UNMATCHED, 3])
    assert radius.dtype == np.int8


def test_a_run_s_radii_round_trip_beside_it(tmp_path):
    from experiments.matched_radius import MATCHED_FILE, read_matched, write_matched

    assert read_matched(tmp_path) is None
    radius = np.array([-1, 0, 4, 5], np.int8)
    write_matched(tmp_path, radius)
    assert (tmp_path / MATCHED_FILE).exists()
    back = read_matched(tmp_path)
    assert back is not None
    np.testing.assert_array_equal(back, radius)


# --- recording the radii of saved runs ----------------------------------------

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

VARIANTS = (
    {"method": "plain", "class_estimator": "pooled", "shrinkage_weight": None},
    {
        "method": "strict",
        "class_estimator": "pooled",
        "shrinkage_weight": None,
        "minimum_support": 10**6,
    },
)


@pytest.fixture
def sieve_cv(tmp_path):
    """A tiny CV of two readings of one fit, with saved predictions."""
    from experiments.cv import ModelVariant, run_sieve_cv

    from experiments.tests.test_cv import _sieve_cv_kwargs

    kw = _sieve_cv_kwargs(tmp_path)
    run_sieve_cv(
        save_predictions=True,
        experiment="sieve-cv",
        variants=[ModelVariant.from_dict(v) for v in VARIANTS],
        **kw,
    )
    return kw


def _sieve_arms():
    from experiments.breakdown import Arm

    return [Arm("sieve-cv", "plain", 1), Arm("sieve-cv", "strict", 1)]


def _record_sieve(kw, **extra):
    from experiments.cv import ModelVariant
    from experiments.matched_radius import record_sieve_runs

    return record_sieve_runs(
        kw["runs_root"],
        _sieve_arms(),
        store=kw["store"],
        n_shards=kw["n_shards"],
        k=kw["k"],
        config_label=kw["config_label"],
        fit_depth=1,
        variants=[ModelVariant.from_dict(v) for v in VARIANTS],
        stores_root=kw["stores_root"],
        **extra,
    )


def _by_method(runs):
    import json

    out: dict[str, list] = {}
    for run in runs:
        cv = json.loads((run / "manifest.json").read_text())["config"]["cv"]
        out.setdefault(cv["method"], []).append(run)
    return out


def test_every_sieve_run_gets_one_radius_per_held_out_atom(sieve_cv):
    from experiments.matched_radius import read_matched

    written = _record_sieve(sieve_cv)
    assert len(written) == 2 * sieve_cv["k"]
    for run in written:
        radius = read_matched(run)
        z = np.load(run / "predictions.npz", allow_pickle=True)
        assert radius is not None
        assert radius.shape == (int(z["num_atoms"].sum()),)
        assert set(np.unique(radius)) <= {-1, 0, 1}


def test_a_variant_is_read_under_its_own_parameters(sieve_cv):
    """No class has a million members, so the strict reading matches nothing,
    while the plain reading of the same fit matches most atoms."""
    from experiments.matched_radius import read_matched

    runs = _by_method(_record_sieve(sieve_cv))
    plain = np.concatenate([read_matched(r) for r in runs["plain"]])
    strict = np.concatenate([read_matched(r) for r in runs["strict"]])
    assert (strict == -1).all()
    assert (plain >= 0).mean() > 0.5


def test_recorded_runs_are_skipped_until_stale_or_forced(sieve_cv):
    from experiments.matched_radius import pending

    arms = _sieve_arms()
    assert len(pending(sieve_cv["runs_root"], arms)) == 2 * sieve_cv["k"]
    written = _record_sieve(sieve_cv)
    assert pending(sieve_cv["runs_root"], arms) == []
    assert _record_sieve(sieve_cv) == []
    assert len(_record_sieve(sieve_cv, force=True)) == len(written)
    # stale: predictions newer than the record
    later = (written[0] / "matched_radius.npz").stat().st_mtime + 10
    os.utime(written[0] / "predictions.npz", (later, later))
    assert pending(sieve_cv["runs_root"], arms) == [written[0]]


def test_repeats_restrict_the_selection(sieve_cv):
    from experiments.matched_radius import pending

    assert pending(sieve_cv["runs_root"], _sieve_arms(), repeats=[3]) == []
    assert _record_sieve(sieve_cv, repeats=[3]) == []


def test_a_sieve_run_whose_predictions_disagree_is_refused(sieve_cv):
    from experiments.matched_radius import pending

    run = pending(sieve_cv["runs_root"], _sieve_arms())[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    z["atom_target_pred"] = z["atom_target_pred"] + 1e-6
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="does not reproduce"):
        _record_sieve(sieve_cv)


def test_a_run_whose_ids_disagree_is_refused(sieve_cv):
    from experiments.matched_radius import pending

    run = pending(sieve_cv["runs_root"], _sieve_arms())[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    # chembl_id is the one identifier the synthetic store keeps distinct
    before = z["chembl_id"].astype(str)
    z["chembl_id"] = z["chembl_id"][::-1].copy()
    assert (z["chembl_id"].astype(str) != before).any()
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="order differs"):
        _record_sieve(sieve_cv)


def test_a_variant_named_twice_is_refused(sieve_cv):
    from experiments.cv import ModelVariant
    from experiments.matched_radius import record_sieve_runs

    twice = [ModelVariant.from_dict(VARIANTS[0])] * 2
    with pytest.raises(ValueError, match="twice"):
        record_sieve_runs(
            sieve_cv["runs_root"],
            _sieve_arms(),
            store=sieve_cv["store"],
            n_shards=sieve_cv["n_shards"],
            k=sieve_cv["k"],
            config_label=sieve_cv["config_label"],
            fit_depth=1,
            variants=twice,
            stores_root=sieve_cv["stores_root"],
        )


# --- the HOSE arm --------------------------------------------------------------


@pytest.fixture
def hose_cv(tmp_path):
    pytest.importorskip("hosegen")
    from experiments.cv import run_hose_cv, run_hose_shard_fits

    from experiments.tests.test_cv import _write_shard_store

    store, stores_root = _write_shard_store(tmp_path, n_mol=20, n_shards=10, seed=2)
    common: dict[str, Any] = {
        "store": store,
        "n_shards": 10,
        "runs_root": tmp_path / "runs",
        "stores_root": stores_root,
        "allow_dirty": True,
    }
    run_hose_shard_fits(radius=2, **common)
    run_hose_cv(
        radii=[2],
        repeats=[0],
        k=5,
        save_predictions=True,
        experiment="hose-cv",
        **common,
    )
    return common


def _record_hose(kw, **extra):
    from experiments.breakdown import Arm
    from experiments.matched_radius import record_hose_runs

    return record_hose_runs(
        kw["runs_root"],
        Arm("hose-cv", "hose", 2),
        store=kw["store"],
        n_shards=kw["n_shards"],
        k=5,
        stores_root=kw["stores_root"],
        **extra,
    )


def test_every_hose_run_gets_its_radii_and_the_global_mean_is_no_match(hose_cv):
    from experiments.matched_radius import read_matched

    written = _record_hose(hose_cv)
    assert len(written) == 5
    seen = set()
    for run in written:
        radius = read_matched(run)
        z = np.load(run / "predictions.npz", allow_pickle=True)
        pred = np.asarray(z["atom_target_pred"]).ravel()
        assert radius is not None
        assert radius.shape == pred.shape
        seen |= set(np.unique(radius).tolist())
        # every atom without a match was answered by one number, the global mean
        assert np.unique(pred[radius == -1]).size <= 1
    assert seen <= {-1, 1, 2}
    assert 0 not in seen
    assert _record_hose(hose_cv) == []


def test_folds_restrict_the_hose_selection(hose_cv):
    written = _record_hose(hose_cv, folds=[1, 3])
    assert len(written) == 2
    assert len(_record_hose(hose_cv)) == 3


def test_a_hose_run_whose_predictions_disagree_is_refused(hose_cv):
    from experiments.breakdown import Arm
    from experiments.matched_radius import pending

    run = pending(hose_cv["runs_root"], [Arm("hose-cv", "hose", 2)])[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    z["atom_target_pred"] = z["atom_target_pred"] + 1e-6
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="does not reproduce"):
        _record_hose(hose_cv)


# --- the command ----------------------------------------------------------------


def test_the_command_records_and_checks(sieve_cv, monkeypatch):
    import json

    from experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", sieve_cv["runs_root"])
    monkeypatch.setattr(cli, "DEFAULT_STORES_ROOT", sieve_cv["stores_root"])
    base = [
        "matched-radius",
        sieve_cv["store"],
        "--arm",
        "sieve-cv:plain:1",
        "--arm",
        "sieve-cv:strict:1",
        "--variants",
        json.dumps(VARIANTS),
        "--k",
        str(sieve_cv["k"]),
        "--n-shards",
        str(sieve_cv["n_shards"]),
        "--config-label",
        sieve_cv["config_label"],
        "--fit-depth",
        "1",
    ]
    assert cli.main([*base, "--check"]) == 1
    assert cli.main([*base, "--check", "--repeats", "3"]) == 0
    assert cli.main(base) == 0
    assert cli.main([*base, "--check"]) == 0
