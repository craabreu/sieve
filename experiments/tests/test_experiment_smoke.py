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


def test_smoke_computes_train_and_val_prefixed_metrics_alongside_test(tmp_path):
    """execute() scores train and val the same way it scores test, so all
    three can be compared directly (e.g. to check whether an extrapolation
    split makes val a worse proxy for test than a representative split
    does) -- see runner.py's _score_extra_split docstring."""
    mset = synthetic_molecule_set(n_mol=20, seed=0)
    masks = _synthetic_masks(20, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )

    train_split = mset.select(masks["train"])
    val_split = mset.select(masks["val"])
    assert train_split.n_molecules > 0, "test fixture must exercise a non-empty train"
    assert val_split.n_molecules > 0, "test fixture must exercise a non-empty val"
    assert result.metrics["train/n_test"] == train_split.n_molecules
    assert result.metrics["val/n_test"] == val_split.n_molecules
    assert np.isfinite(result.metrics["train/profile/w1_norm_mean"])
    assert np.isfinite(result.metrics["val/profile/w1_norm_mean"])
    # test's own keys stay unprefixed and unaffected by train/val scoring
    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert "time/train_predict_s" in result.metrics
    assert "time/val_predict_s" in result.metrics

    on_disk = json.loads((result.run_dir / "metrics.json").read_text())
    assert on_disk == result.metrics


def test_smoke_omits_val_metrics_when_val_split_is_empty(tmp_path):
    """_score_extra_split's empty-split skip is one shared code path for
    both train and val (see its docstring) -- exercised here via val, since
    global_mean.fit itself requires a non-empty train and so can't be used
    to exercise the empty-train case the same way."""
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    all_train = np.ones(10, dtype=bool)
    none = np.zeros(10, dtype=bool)
    masks = {"train": all_train, "val": none, "test": none}
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert not any(k.startswith("val/") for k in result.metrics)
    assert "time/val_predict_s" not in result.metrics
    # train is non-empty here, so train/* metrics are present
    assert any(k.startswith("train/") for k in result.metrics)


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

    # Real load_atom_truth joins by SMILES, so it correctly returns whatever
    # split (val or test) actually asked -- mirror that here using mset's own
    # full atom-level truth (synthetic_molecule_set populates it for every
    # molecule), rather than hardcoding test's truth regardless of which
    # split's smiles come in. execute() now scores val too, so a fake that
    # ignores its own smiles/num_atoms args and always returns test-shaped
    # truth breaks the moment val and test have different atom counts.
    offsets = np.concatenate([[0], np.cumsum(mset.num_atoms)])
    smiles_to_idx = {s: i for i, s in enumerate(mset.smiles)}

    def fake_load_atom_truth(store, *, scheme, smiles, num_atoms, **kwargs):
        del store, scheme, num_atoms, kwargs
        idx = [smiles_to_idx[s] for s in smiles]
        slices = [slice(offsets[i], offsets[i + 1]) for i in idx]
        return (
            np.concatenate([mset.atom_profile[s] for s in slices]),
            np.concatenate([mset.atom_area[s] for s in slices]),
            np.concatenate([mset.atom_charge[s] for s in slices]),
        )

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


# --- _build_parity_panels ---------------------------------------------------
#
# Which hexbin panels a run's parity_panel.png gets, and what data/metrics
# feed each one. Pure numpy -- no matplotlib, no rdkit -- so this is fast-
# suite tested independently of plots.py actually rendering anything (that
# needs matplotlib/rdkit: see test_experiment_plots.py).


def _panel_titles(panels):
    return [p["title"] for p in panels]


def test_build_parity_panels_molecule_profile_only():
    """No area/charge/atom output at all (e.g. a bare MoleculePredictor
    supplying only mol_profile) -> exactly one panel."""
    import sieve_experiments.runner as runner_mod
    from sieve_experiments.predictors.base import Prediction

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(mol_profile=ms.mol_profile)

    panels = runner_mod._build_parity_panels(
        ms,
        pred,
        {"profile/w1_norm_mean": 0.01},
        area_true=None,
        area_pred=None,
        charge_true=None,
        charge_pred=None,
        atom_truth=None,
    )
    assert _panel_titles(panels) == ["molecule profile"]
    assert panels[0]["metrics"] == {"w1_norm_mean": 0.01}


def test_build_parity_panels_reuses_precomputed_molecule_normalization():
    """_write_plots already normalizes test.mol_profile/pred.mol_profile
    for profile_panel's needs -- _build_parity_panels must reuse that
    result for its own "molecule profile" panel rather than recomputing
    normalize_rows on the same arrays a second time."""
    import sieve_experiments.runner as runner_mod
    from sieve_experiments.predictors.base import Prediction

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(mol_profile=ms.mol_profile)
    precomputed = runner_mod._normalized_profile_rows(ms.mol_profile, pred.mol_profile)

    panels = runner_mod._build_parity_panels(
        ms,
        pred,
        {},
        area_true=None,
        area_pred=None,
        charge_true=None,
        charge_pred=None,
        atom_truth=None,
        molecule_profile_norm=precomputed,
    )
    norm_true, norm_pred, keep = precomputed
    np.testing.assert_array_equal(panels[0]["y_true"], norm_true[keep])
    np.testing.assert_array_equal(panels[0]["y_pred"], norm_pred[keep])


def test_build_parity_panels_includes_area_and_charge_when_supplied():
    import sieve_experiments.runner as runner_mod
    from sieve_experiments.predictors.base import Prediction

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(
        mol_profile=ms.mol_profile, mol_area=ms.mol_area, mol_charge_raw=ms.mol_charge
    )

    panels = runner_mod._build_parity_panels(
        ms,
        pred,
        {"area/mae": 1.0, "charge/mae": 2.0},
        area_true=ms.mol_area,
        area_pred=pred.mol_area,
        charge_true=ms.net_charge,
        charge_pred=pred.mol_charge_raw,
        atom_truth=None,
    )
    assert _panel_titles(panels) == [
        "molecule profile",
        "molecule area",
        "molecule charge",
    ]
    assert panels[1]["metrics"] == {"mae": 1.0}
    assert panels[2]["metrics"] == {"mae": 2.0}


def test_build_parity_panels_includes_atom_panels_when_atom_truth_and_pred_supplied():
    import sieve_experiments.runner as runner_mod
    from sieve_experiments.predictors.base import Prediction

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(
        mol_profile=ms.mol_profile,
        atom_profile=ms.atom_profile,
        atom_area=ms.atom_area,
        atom_charge=ms.atom_charge,
    )
    atom_truth = (ms.atom_profile, ms.atom_area, ms.atom_charge)

    panels = runner_mod._build_parity_panels(
        ms,
        pred,
        {
            "atom/profile/w1_norm_mean": 0.02,
            "atom/area/mae": 0.3,
            "atom/charge/mae": 0.4,
        },
        area_true=None,
        area_pred=None,
        charge_true=None,
        charge_pred=None,
        atom_truth=atom_truth,
    )
    assert _panel_titles(panels) == [
        "molecule profile",
        "atom profile",
        "atom area",
        "atom charge",
    ]
    assert panels[1]["metrics"] == {"w1_norm_mean": 0.02}
    assert panels[2]["metrics"] == {"mae": 0.3}
    assert panels[3]["metrics"] == {"mae": 0.4}


def test_build_parity_panels_excludes_atom_panels_when_atom_truth_load_failed():
    """pred.atom_profile is set (the predictor is an AtomPredictor), but
    atom_truth is None (the store load failed) -- no atom panels, not a
    crash. Mirrors _execute_inner's own graceful-degradation path."""
    import sieve_experiments.runner as runner_mod
    from sieve_experiments.predictors.base import Prediction

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(mol_profile=ms.mol_profile, atom_profile=ms.atom_profile)

    panels = runner_mod._build_parity_panels(
        ms,
        pred,
        {},
        area_true=None,
        area_pred=None,
        charge_true=None,
        charge_pred=None,
        atom_truth=None,
    )
    assert _panel_titles(panels) == ["molecule profile"]


def test_smoke_handles_an_empty_train_split_without_crashing_manifest(
    tmp_path, monkeypatch
):
    """train_mean_num_atoms must be guarded against an empty train split the
    same way val_mean_num_atoms/test_mean_num_atoms already are: not every
    predictor rejects an empty train set the way GlobalMeanPredictor does
    (it raises explicitly) -- a future/custom predictor that tolerates it
    would otherwise hit np.mean(empty) while building the manifest, after
    fit/predict have already succeeded. pyproject.toml promotes
    RuntimeWarning to an error."""
    import sieve_experiments.runner as runner_mod

    class _NoOpPredictor:
        name = "noop"

        def fit(self, train, val, *, rng):
            del train, val, rng

        def predict(self, test):
            from sieve_experiments.predictors.base import Prediction

            return Prediction(mol_profile=np.zeros_like(test.mol_profile))

    monkeypatch.setattr(runner_mod, "build", lambda name, params: _NoOpPredictor())

    mset = synthetic_molecule_set(n_mol=10, seed=0)
    none = np.zeros(10, dtype=bool)
    test_mask = ~none
    masks = {"train": none, "val": none, "test": test_mask}
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["data"]["train_mean_num_atoms"] is None


# --- predictor store/scheme cross-check -------------------------------------
#
# DASHPredictor carries its own store/scheme (predictor.params),
# duplicating data.store/data.scheme, because it has to load atom-level
# truth independently (see dash.py's own docstring). Nothing enforced the
# two copies actually agree -- a config edit to data.store with the
# predictor.params.store copy left stale would silently fit/evaluate DASH
# against two different stores. execute() checks this, duck-typed off
# whatever attributes a predictor happens to expose (like match_stats).


class _StoreSchemePredictor:
    """A minimal predictor exposing store/scheme attributes, the same duck
    type DASHPredictor has -- without needing DASH's real machinery
    (rdkit, the tree clone) just to test the cross-check."""

    name = "fake_store_scheme"

    def __init__(self, store, scheme):
        self.store = store
        self.scheme = scheme

    def fit(self, train, val, *, rng):
        del train, val, rng

    def predict(self, test):
        from sieve_experiments.predictors.base import Prediction

        return Prediction(mol_profile=np.zeros_like(test.mol_profile))


def test_execute_rejects_a_predictor_store_that_disagrees_with_data_store(
    tmp_path, monkeypatch
):
    import sieve_experiments.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build",
        lambda name, params: _StoreSchemePredictor(
            store="a-different-store", scheme="cosmo-sac-2010"
        ),
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()  # data.store == "synthetic"

    with pytest.raises(ValueError, match="store"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)


def test_execute_rejects_a_predictor_scheme_that_disagrees_with_data_scheme(
    tmp_path, monkeypatch
):
    import sieve_experiments.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "build",
        lambda name, params: _StoreSchemePredictor(
            store="synthetic", scheme="a-different-scheme"
        ),
    )
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()  # data.scheme == "cosmo-sac-2010"

    with pytest.raises(ValueError, match="scheme"):
        execute(cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None)


def test_execute_allows_a_predictor_with_no_store_scheme_attributes(tmp_path):
    """global_mean (and any plain Predictor) has no .store/.scheme -- the
    check is a no-op for it, duck-typed, not a required interface field."""
    mset = synthetic_molecule_set(n_mol=10, seed=0)
    masks = _synthetic_masks(10, seed=1)
    cfg = _tiny_cfg()

    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.run_dir.is_dir()
