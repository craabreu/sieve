"""Study F: calibration of Sieve's predictive variance (spec 2026-09-25)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from experiments.calibration import (
    ARMS,
    arm_variances,
    calibration_metrics,
    group_percentile_rank,
)


def _chain(n=24, graphs=4, seed=0):
    """Disjoint alternating C/H paths with random targets -- built here, not
    imported: `tests.helpers` resolves to this directory's own module or the
    root suite's depending on import order."""
    from sieve.batch import NodeBatch

    per = n // graphs
    src, dst, gid = [], [], []
    for g in range(graphs):
        off = g * per
        for i in range(per - 1):
            src += [off + i, off + i + 1]
            dst += [off + i + 1, off + i]
        gid += [g] * per
    return NodeBatch(
        node_attrs=(np.arange(n) % 2).reshape(-1, 1).astype(np.int64),
        edge_src=np.array(src, np.int64),
        edge_dst=np.array(dst, np.int64),
        edge_attrs=np.ones((len(src), 1), np.int64),
        graph_id=np.array(gid, np.int64),
        y=np.random.default_rng(seed).normal(size=(n, 1)),
    )


def _config(**kw):
    from sieve.config import SieveConfig

    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes={"element": {"C": 0, "H": 1}},
        edge_codes={"bond_type": {"SINGLE": 0}},
        max_wl_depth=2,
        predictive_variance=True,
        **kw,
    )


def _model_and_prediction(**kw):
    import sieve

    b = _chain()
    m = sieve.fit(b, _config(**kw)).with_within_structure(2.0e-3, 20)
    # one unseen element code, so the unmatched fallback is exercised
    oov = dataclasses.replace(
        b, node_attrs=np.where(np.arange(b.n_nodes)[:, None] == 0, 7, b.node_attrs)
    )
    return m, sieve.predict_detailed(m, oov)


# ------------------------------------------------------------------ arms --


def test_the_arm_table_is_the_spec_s():
    assert [a.name for a in ARMS] == [
        "form_b",
        "no_sigma2_w",
        "alpha_v_30",
        "no_selection",
        "no_estimation",
    ]
    by = {a.name: a for a in ARMS}
    assert (by["form_b"].alpha_v, by["form_b"].selection_weight) == (10.0, 0.5)
    assert by["alpha_v_30"].alpha_v == 30.0
    assert by["no_selection"].selection_weight == 0.0
    assert not by["no_estimation"].estimation
    assert not by["no_sigma2_w"].within_structure


def test_form_b_is_the_library_s_predictive_variance_bit_for_bit():
    m, p = _model_and_prediction()
    assert p.matched_level[0] == -1  # the unmatched atom is there
    assert p.predictive_variance is not None
    np.testing.assert_array_equal(
        arm_variances(m, p)["form_b"], p.predictive_variance[:, 0]
    )


def test_no_sigma2_w_drops_it_everywhere():
    m, p = _model_and_prediction()
    v = arm_variances(m, p)
    s2w = float(m.within_variance[0])
    matched = p.matched_level >= 0
    np.testing.assert_allclose(
        v["no_sigma2_w"][matched], v["form_b"][matched] - s2w, rtol=1e-12
    )
    np.testing.assert_allclose(v["no_sigma2_w"][~matched], m.global_msd[0])


def test_alpha_v_30_is_the_library_at_alpha_v_30():
    from sieve.uncertainty import predictive_variance

    m, p = _model_and_prediction()
    table = predictive_variance(m, alpha_v=30.0)
    got = arm_variances(m, p)["alpha_v_30"]
    backoff = np.asarray(m.config.backoff_path)
    for i in np.flatnonzero(p.matched_level >= 0):
        assert got[i] == table[backoff[p.matched_level[i]]][p.class_id[i], 0]


def test_form_b_equals_the_library_under_a_stereo_track():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    import sieve
    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes, from_rdkit

    smiles = ["C/C=C/C", r"C/C=C\C", "C/C=C/CC", r"C/C=C\CC", "CCCC", "CC(C)C"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    codes, edges = build_codes(mols, ["element"])
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=3,
        stereo=("cis_trans",),
        predictive_variance=True,
    )
    b = from_rdkit(mols, config=cfg)
    b = dataclasses.replace(b, y=np.random.default_rng(1).normal(size=(b.n_nodes, 1)))
    m = sieve.fit(b, cfg).with_within_structure(1.0e-3, 10)
    p = sieve.predict_detailed(m, b)
    assert p.predictive_variance is not None
    np.testing.assert_array_equal(
        arm_variances(m, p)["form_b"], p.predictive_variance[:, 0]
    )


# --------------------------------------------------------------- metrics --


def _synthetic(n_conf=400, per=10, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    n = n_conf * per
    conf = np.repeat(np.arange(n_conf), per)
    sigma2 = rng.uniform(0.5, 2.0, size=n) * 1e-3
    target = rng.normal(size=n) * 0.1
    raw = target + rng.normal(size=n) * np.sqrt(sigma2) * scale
    k_star = rng.integers(0, 3, size=n)
    k_star[:5] = -1
    return {
        "raw": raw,
        "target": target,
        "sigma2": sigma2,
        "k_star": k_star,
        "conf": conf,
        "molecule_value": np.zeros(n_conf),
        "depth": 3,
    }


def test_calibrated_errors_score_as_calibrated():
    m = calibration_metrics(**_synthetic())
    assert m["ez2"] == pytest.approx(1.0, abs=0.05)
    for c in (50, 90, 95, 99):
        assert m[f"cov{c}"] == pytest.approx(c / 100, abs=0.02)


def test_doubling_sigma_quarters_ez2():
    s = _synthetic()
    a = calibration_metrics(**s)
    b = calibration_metrics(**{**s, "sigma2": 4 * s["sigma2"]})
    assert b["ez2"] == pytest.approx(a["ez2"] / 4, rel=1e-12)


def test_nll_is_the_gaussian_mean():
    s = _synthetic(n_conf=3, per=2)
    e = s["raw"] - s["target"]
    want = np.mean(0.5 * (np.log(2 * np.pi * s["sigma2"]) + e**2 / s["sigma2"]))
    assert calibration_metrics(**s)["nll"] == pytest.approx(want, rel=1e-12)


def test_ez2_by_radius_only_for_radii_that_occur():
    s = _synthetic()
    m = calibration_metrics(**s)
    assert {k for k in m if k.startswith("ez2_k")} == {"ez2_k0", "ez2_k1", "ez2_k2"}
    e = s["raw"] - s["target"]
    sel = s["k_star"] == 1
    assert m["ez2_k1"] == pytest.approx(
        np.mean(e[sel] ** 2 / s["sigma2"][sel]), rel=1e-12
    )


def test_normalised_errors_use_the_variance_weighted_normaliser():
    from experiments.normalize import variance_weighted_normalize

    s = _synthetic(n_conf=20)
    x = variance_weighted_normalize(
        s["raw"], np.sqrt(s["sigma2"]), s["molecule_value"], s["conf"], 20
    )
    m = calibration_metrics(**s)
    assert m["norm_rmse"] == pytest.approx(
        np.sqrt(np.mean((x - s["target"]) ** 2)), rel=1e-12
    )
    assert m["norm_mae"] == pytest.approx(np.mean(np.abs(x - s["target"])), rel=1e-12)


def test_group_percentile_rank_matches_a_brute_force_average_rank():
    from scipy.stats import rankdata

    rng = np.random.default_rng(4)
    groups = rng.integers(0, 6, size=60)
    values = rng.integers(0, 5, size=60).astype(float)  # ties on purpose
    got = group_percentile_rank(values, groups)
    for g in np.unique(groups):
        sel = groups == g
        np.testing.assert_allclose(got[sel], rankdata(values[sel]) / sel.sum())


def test_rho_is_one_when_sigma_orders_the_errors_exactly():
    s = _synthetic()
    e = np.abs(s["raw"] - s["target"])
    m = calibration_metrics(**{**s, "sigma2": e**2 + 1e-9})
    assert m["rho"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------- scoring runs --


@pytest.fixture
def scored_cv(tmp_path):
    """A tiny Study-B-like CV experiment with saved predictions."""
    pytest.importorskip("pandas")
    from experiments.cv import run_sieve_cv

    from experiments.tests.test_cv import _sieve_cv_kwargs

    kw = _sieve_cv_kwargs(tmp_path)
    run_sieve_cv(save_predictions=True, experiment="sieve-cv", **kw)
    return kw


def _score(kw, **extra):
    from experiments.calibration import score_runs

    return score_runs(
        kw["runs_root"],
        store=kw["store"],
        experiment="sieve-cv",
        method=kw["method"],
        depth=1,
        n_shards=kw["n_shards"],
        k=kw["k"],
        config_label=kw["config_label"],
        fit_depth=1,
        stores_root=kw["stores_root"],
        **extra,
    )


def test_every_run_gets_every_arm_s_keys(scored_cv):
    import json

    from experiments.calibration import ARMS, METRICS_FILE

    written = _score(scored_cv)
    assert len(written) == scored_cv["k"]
    for run in written:
        side = json.loads((run / METRICS_FILE).read_text())
        for arm in ARMS:
            for m in ("nll", "ez2", "cov95", "rho", "norm_rmse", "norm_mae"):
                assert f"calibration/{arm.name}/{m}" in side
        assert "calibration/equal/norm_rmse" in side
        assert side["calibration/n_atoms"] > 0
        assert sum(v for k, v in side.items() if "/share_k" in k) == pytest.approx(1.0)


def test_missing_scores_reports_absent_stale_and_incomplete(scored_cv):
    import json
    import os

    from experiments.calibration import METRICS_FILE, missing_scores

    sel = {"experiment": "sieve-cv", "method": scored_cv["method"], "depth": 1}
    root = scored_cv["runs_root"]
    assert len(missing_scores(root, **sel)) == scored_cv["k"]  # absent
    runs = _score(scored_cv)
    assert missing_scores(root, **sel) == []
    assert _score(scored_cv) == []  # nothing left to do
    # stale: predictions newer than the sidecar
    pred = runs[0] / "predictions.npz"
    side = runs[0] / METRICS_FILE
    os.utime(pred, (side.stat().st_mtime + 10, side.stat().st_mtime + 10))
    # incomplete: an arm's keys missing
    d = json.loads((runs[1] / METRICS_FILE).read_text())
    (runs[1] / METRICS_FILE).write_text(
        json.dumps({k: v for k, v in d.items() if "/no_estimation/" not in k})
    )
    assert set(missing_scores(root, **sel)) == {runs[0], runs[1]}


def test_repeats_restrict_the_selection(scored_cv):
    from experiments.calibration import selected_runs

    sel = {"experiment": "sieve-cv", "method": scored_cv["method"], "depth": 1}
    assert (
        len(selected_runs(scored_cv["runs_root"], repeats=[0], **sel)) == scored_cv["k"]
    )
    assert selected_runs(scored_cv["runs_root"], repeats=[3], **sel) == []


def test_a_run_whose_predictions_disagree_is_refused(scored_cv):
    from experiments.calibration import selected_runs

    run = selected_runs(
        scored_cv["runs_root"],
        experiment="sieve-cv",
        method=scored_cv["method"],
        depth=1,
    )[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    z["atom_target_pred"] = z["atom_target_pred"] + 1e-6
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="does not reproduce"):
        _score(scored_cv)


def test_a_run_whose_ids_disagree_is_refused(scored_cv):
    from experiments.calibration import selected_runs

    run = selected_runs(
        scored_cv["runs_root"],
        experiment="sieve-cv",
        method=scored_cv["method"],
        depth=1,
    )[0]
    z = dict(np.load(run / "predictions.npz", allow_pickle=True))
    # the synthetic store has no dash_id and few distinct conf_ids; its
    # chembl_ids are distinct per molecule, so a reversal misaligns them
    z["chembl_id"] = z["chembl_id"][::-1].copy()
    before = np.load(run / "predictions.npz", allow_pickle=True)["chembl_id"]
    assert (z["chembl_id"].astype(str) != before.astype(str)).any()
    np.savez(run / "predictions.npz", **z)
    with pytest.raises(ValueError, match="held-out order"):
        _score(scored_cv)


def test_aggregate_merges_the_calibration_sidecar_and_refuses_a_clash(scored_cv):
    import json

    from experiments.aggregate import read_runs_from_dirs
    from experiments.calibration import METRICS_FILE

    runs = _score(scored_cv)
    rows = read_runs_from_dirs(scored_cv["runs_root"], "sieve-cv")
    assert all("calibration/form_b/nll" in r.metrics for r in rows)
    d = json.loads((runs[0] / METRICS_FILE).read_text())
    d["rmse"] = 0.0  # clashes with metrics.json
    (runs[0] / METRICS_FILE).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="repeats"):
        read_runs_from_dirs(scored_cv["runs_root"], "sieve-cv")


# ---------------------------------------------------------------- report --


def _fake_runs(tmp_path, n_repeats=2, k=5, seed=0):
    """Hand-written runs: a manifest, metrics.json and a calibration sidecar."""
    import json

    from experiments.calibration import ARMS, METRICS_FILE

    rng = np.random.default_rng(seed)
    root = tmp_path / "runs"
    for r in range(n_repeats):
        for f in range(k):
            run = root / "exp" / f"r{r}-f{f}"
            run.mkdir(parents=True)
            cv = {"method": "m", "depth": 2, "repeat": r, "fold": f}
            (run / "manifest.json").write_text(json.dumps({"config": {"cv": cv}}))
            (run / "metrics.json").write_text(json.dumps({"rmse": 0.02, "mae": 0.01}))
            side = {
                "calibration/n_atoms": 100.0,
                "calibration/sigma2_w": 1e-4,
                "calibration/share_k1": 0.4,
                "calibration/share_k2": 0.6,
                "calibration/equal/norm_rmse": 0.0196 + 1e-5 * rng.normal(),
                "calibration/equal/norm_mae": 0.0109,
            }
            for i, arm in enumerate(ARMS):
                for key in (
                    "nll",
                    "ez2",
                    "cov50",
                    "cov90",
                    "cov95",
                    "cov99",
                    "rho",
                    "norm_rmse",
                    "norm_mae",
                    "ez2_k1",
                    "ez2_k2",
                ):
                    side[f"calibration/{arm.name}/{key}"] = (
                        1.0 + 0.01 * i + 1e-3 * rng.normal()
                    )
            (run / METRICS_FILE).write_text(json.dumps(side))
    return root


def test_the_report_has_its_blocks_and_is_deterministic(tmp_path):
    from experiments.calibration import calibration_report

    root = _fake_runs(tmp_path)
    kw = {"experiment": "exp", "method": "m", "depth": 2, "k": 5}
    a = calibration_report(root, **kw)
    assert a == calibration_report(root, **kw)
    for heading in (
        "means over all 10 samples",
        "E[z^2] by matched radius",
        "paired differences, all 10 samples",
        "paired differences, repeats 1-",
    ):
        assert heading in a
    for arm in ("form_b", "no_sigma2_w", "alpha_v_30", "no_selection", "no_estimation"):
        assert arm in a
    assert "no_sigma2_w at alpha_v = 30" in a


def test_the_report_s_differences_are_paired_difference_s(tmp_path):
    import json

    from experiments.calibration import METRICS_FILE, calibration_report, selected_runs
    from experiments.stereo_subsets import paired_difference

    root = _fake_runs(tmp_path)
    runs = selected_runs(root, experiment="exp", method="m", depth=2)
    sides = [json.loads((r / METRICS_FILE).read_text()) for r in runs]
    a = np.array([s["calibration/form_b/nll"] for s in sides])
    b = np.array([s["calibration/alpha_v_30/nll"] for s in sides])
    d = paired_difference(a, b, k=5)
    text = calibration_report(root, experiment="exp", method="m", depth=2, k=5)
    assert f"{d.mean:+.2e}" in text and f"[{d.lo:+.2e}, {d.hi:+.2e}]" in text


def test_score_calibration_check_is_scoped_by_repeats(scored_cv, monkeypatch):
    from experiments import cli

    monkeypatch.setattr(cli, "DEFAULT_RUNS_ROOT", scored_cv["runs_root"])
    monkeypatch.setattr(cli, "DEFAULT_STORES_ROOT", scored_cv["stores_root"])
    base = [
        "score-calibration",
        scored_cv["store"],
        "--experiment",
        "sieve-cv",
        "--method",
        scored_cv["method"],
        "--depth",
        "1",
        "--k",
        str(scored_cv["k"]),
        "--n-shards",
        str(scored_cv["n_shards"]),
        "--config-label",
        scored_cv["config_label"],
        "--fit-depth",
        "1",
    ]
    assert cli.main([*base, "--check"]) == 1  # unscored, and nothing written
    assert cli.main([*base, "--check", "--repeats", "3"]) == 0  # no runs in repeat 3
    assert cli.main(base) == 0
    assert cli.main([*base, "--check"]) == 0


def _set_manifest_collapse(run, value):
    import json

    path = run / "manifest.json"
    m = json.loads(path.read_text())
    m["config"]["cv"]["collapse_train_scoring"] = value
    path.write_text(json.dumps(m))


def test_a_collapse_flag_that_disagrees_with_the_manifest_is_refused(scored_cv):
    """Collapse is the CV's own (spec 2.1): a flag contradicting the manifest
    would attach -- or omit -- sigma2_w wrongly (final review, Important 1)."""
    with pytest.raises(ValueError, match="collapse"):
        _score(scored_cv, collapse_override=True)


def test_a_collapsed_run_without_a_training_floor_is_refused(scored_cv):
    """A collapsed run must carry sigma2_w; without a floor-cache entry the
    sidecar would silently score form_b as no_sigma2_w (final review,
    Important 1)."""
    from experiments.calibration import selected_runs

    for run in selected_runs(
        scored_cv["runs_root"],
        experiment="sieve-cv",
        method=scored_cv["method"],
        depth=1,
    ):
        _set_manifest_collapse(run, True)
    with pytest.raises(ValueError, match="sigma2_w"):
        _score(scored_cv)
