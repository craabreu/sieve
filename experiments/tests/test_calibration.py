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
