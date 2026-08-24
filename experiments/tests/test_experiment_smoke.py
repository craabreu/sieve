"""End-to-end smoke test on a synthetic store -- no cosmolayer, no rdkit,
no network, no mlflow required (tracking=None). Should run in well under a
second."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
from sieve_experiments.runner import execute

from experiments.tests.helpers import synthetic_molecule_set


def _tiny_cfg(**run_overrides) -> ExperimentCfg:
    return ExperimentCfg(
        run=RunCfg(
            experiment="smoke-test", seed=0, tags={"stage": "smoke"}, **run_overrides
        ),
        data=DataCfg(
            store="synthetic",
            scheme="cosmo-sac-2010",
            split_column="split",
            train_split="train",
            val_split="val",
            eval_split="test",
        ),
        predictor=PredictorCfg(name="global_mean", params={}),
    )


def _synthetic_masks(n_mol: int, seed: int = 0):
    """train/val/test masks over a synthetic_molecule_set, deterministic."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "val", "test"], size=n_mol, p=[0.6, 0.2, 0.2])
    return {name: labels == name for name in ("train", "val", "test")}


def test_smoke_pipeline_writes_every_artifact(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    run_dir = result.run_dir
    assert run_dir.is_dir()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "stdout.log").exists()


def test_smoke_metrics_are_finite_and_charge_has_no_r2(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert "charge/r2" not in result.metrics
    assert "area/r2" in result.metrics  # global_mean supplies area


def test_smoke_manifest_records_git_and_seed(tmp_path):
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["seed"] == 0
    assert "commit" in manifest["git"]
    assert manifest["data"]["n_train_molecules"] > 0
    assert manifest["data"]["n_test_molecules"] > 0


def test_smoke_rejects_dirty_tree_by_default(tmp_path, monkeypatch):
    import sieve_experiments.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_git_info",
        lambda repo_root: {
            "commit": "deadbeef",
            "branch": "main",
            "dirty": True,
            "describe": "deadbeef-dirty",
        },
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()

    import pytest

    with pytest.raises(RuntimeError, match="dirty"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=False, tracking=None)


def test_smoke_handles_an_empty_test_split(tmp_path):
    """A tiny --limit run can land zero molecules in the eval split (e.g. the
    real chaos-store's biased_split with --limit 50, where the first 50
    rows are all train). This must not crash -- pyproject.toml promotes
    RuntimeWarning to an error, so an unguarded np.mean(empty) would."""
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    all_train = np.ones(10, dtype=bool)
    none = np.zeros(10, dtype=bool)
    masks = {"train": all_train, "val": none, "test": none}
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.metrics["n_test"] == 0
    assert np.isnan(result.metrics["profile/w1_norm_mean"])
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["data"]["test_mean_num_atoms"] is None


def test_smoke_metrics_json_matches_returned_metrics(tmp_path):
    mset = synthetic_molecule_set(n_mol=15, seed=2)
    masks = _synthetic_masks(15, seed=3)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    on_disk = json.loads((result.run_dir / "metrics.json").read_text())
    assert on_disk.keys() == result.metrics.keys()
    for key in on_disk:
        if isinstance(on_disk[key], float) and np.isnan(on_disk[key]):
            assert np.isnan(result.metrics[key])
        else:
            assert on_disk[key] == result.metrics[key]


# --- atom-level metrics (flat, atom/-prefixed keys) -------------------------
#
# DASH's own atom-metrics coverage (real store, real predictions) lives in
# test_experiment_predictor_dash_optional.py. These stay in the fast suite
# by faking the predictor and load_atom_truth: what's under test here is
# execute()'s own wiring (gating, prefixing, graceful failure), not DASH.
# Molecule-level keys stay unprefixed and unchanged; only the new atom
# metrics get an "atom/" prefix -- flat, not nested, so they're plain floats
# MLflow's log_metrics can take directly, same as everything else already.


class _FakeAtomPredictor:
    """An AtomPredictor stand-in that just echoes ground truth back as its
    own prediction -- perfect atom-level accuracy, so the resulting metrics
    have simple, known values (~0 error) without needing real numerics."""

    name = "fake_atom"

    def fit(self, train, val, *, rng):
        del train, val, rng

    def predict(self, test):
        from sieve_experiments.predictors.base import Prediction

        return Prediction(
            mol_profile=test.mol_profile,
            atom_profile=test.atom_profile,
            atom_area=test.atom_area,
            atom_charge=test.atom_charge,
        )


def test_smoke_metrics_have_no_atom_keys_for_a_molecule_level_predictor(tmp_path):
    """global_mean is a MoleculePredictor -- no atom_profile output, so
    there is nothing to compute atom-level metrics from."""
    mset = synthetic_molecule_set(n_mol=15, seed=2)
    masks = _synthetic_masks(15, seed=3)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert not any(k.startswith("atom/") for k in result.metrics)


def test_smoke_computes_atom_prefixed_metrics_for_an_atom_level_predictor(
    tmp_path, monkeypatch
):
    import sieve_experiments.runner as runner_mod

    mset = synthetic_molecule_set(n_mol=15, seed=2)
    masks = _synthetic_masks(15, seed=3)
    cfg = _tiny_cfg()

    monkeypatch.setattr(runner_mod, "build", lambda name, params: _FakeAtomPredictor())

    test_split = mset.select(masks["test"])

    def fake_load_atom_truth(store, *, scheme, smiles, num_atoms, **kwargs):
        del store, scheme, smiles, num_atoms, kwargs
        return test_split.atom_profile, test_split.atom_area, test_split.atom_charge

    monkeypatch.setattr(runner_mod, "load_atom_truth", fake_load_atom_truth)

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    assert result.metrics["atom/n_test"] == test_split.n_atoms
    # ground truth echoed back as the prediction -> ~perfect atom accuracy
    assert result.metrics["atom/profile/w1_norm_area_weighted"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert result.metrics["atom/area/mae"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["atom/charge/mae"] == pytest.approx(0.0, abs=1e-9)
    # molecule-level keys stay unprefixed, unaffected
    assert "profile/w1_norm_mean" in result.metrics

    on_disk = json.loads((result.run_dir / "metrics.json").read_text())
    assert on_disk == result.metrics


def test_smoke_atom_metrics_skip_gracefully_when_truth_load_fails(
    tmp_path, monkeypatch
):
    """A predictor supplying atom_profile shouldn't crash the whole run if
    atom-level ground truth can't be loaded (store absent, bad scheme,
    whatever) -- this is a metrics-only concern, not a fit/predict one."""
    import sieve_experiments.runner as runner_mod

    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()

    monkeypatch.setattr(runner_mod, "build", lambda name, params: _FakeAtomPredictor())

    def boom(*args, **kwargs):
        raise RuntimeError("no store here")

    monkeypatch.setattr(runner_mod, "load_atom_truth", boom)

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert not any(k.startswith("atom/") for k in result.metrics)
    assert np.isfinite(
        result.metrics["profile/w1_norm_mean"]
    )  # rest of the run is fine
