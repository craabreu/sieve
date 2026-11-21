"""Error breakdown of saved CV predictions: the atoms a reference arm covers,
elements and charge states, error quantiles, and the HOSE register's
global-mean fallback with its element-mean control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _mol(smiles):
    from rdkit import Chem

    return Chem.AddHs(Chem.MolFromSmiles(smiles))


# Ethanol (9 atoms), methylammonium (8), acetate (7), fluoromethane (5).
SMILES = ["CCO", "C[NH3+]", "CC(=O)[O-]", "CF"]
NET = [0.0, 1.0, -1.0, 0.0]


def _store(tmp_path: Path, smiles=SMILES, conf_ids=None) -> Path:
    pd = pytest.importorskip("pandas")
    from experiments.data import mol_to_blob

    root = tmp_path / "stores"
    (root / "s").mkdir(parents=True)
    pd.DataFrame(
        {
            "dash_id": [f"m{i}" for i in range(len(smiles))],
            "conf_id": conf_ids or ["c0"] * len(smiles),
            "mol": [mol_to_blob(_mol(s)) for s in smiles],
            "net_charge": NET[: len(smiles)],
        }
    ).to_parquet(root / "s" / "molecules.parquet")
    return root


def _expected(true, pred, sel):
    err = (pred - true)[sel]
    t = true[sel]
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - float(np.sum(err**2)) / float(np.sum((t - t.mean()) ** 2)),
    }


# --- per-atom columns and their table ---------------------------------------


def test_atom_columns_give_atomic_number_and_formal_charge():
    from experiments.breakdown import atom_columns

    element, charge = atom_columns(_mol("C[NH3+]"))
    np.testing.assert_array_equal(element, [6, 7, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(charge, [0, 1, 0, 0, 0, 0, 0, 0])


def test_atom_table_returns_rows_in_the_order_asked(tmp_path):
    from experiments.breakdown import atom_columns, build_atom_table, load_atom_table

    root = _store(tmp_path)
    table = load_atom_table(build_atom_table("s", stores_root=root, n_jobs=1))
    mols = [_mol(s) for s in SMILES]
    rows = [2, 0, 3]
    ids = {
        "dash_id": np.array([f"m{i}" for i in rows]),
        "conf_id": np.array(["c0"] * len(rows)),
    }
    num_atoms = np.array([mols[i].GetNumAtoms() for i in rows])
    element, charge = table.rows(ids, num_atoms)
    np.testing.assert_array_equal(
        element, np.concatenate([atom_columns(mols[i])[0] for i in rows])
    )
    np.testing.assert_array_equal(
        charge, np.concatenate([atom_columns(mols[i])[1] for i in rows])
    )


def test_atom_table_refuses_a_row_whose_atom_count_disagrees(tmp_path):
    from experiments.breakdown import build_atom_table, load_atom_table

    table = load_atom_table(
        build_atom_table("s", stores_root=_store(tmp_path), n_jobs=1)
    )
    ids = {"dash_id": np.array(["m0"]), "conf_id": np.array(["c0"])}
    with pytest.raises(ValueError, match="atoms"):
        table.rows(ids, np.array([8]))


def test_atom_table_refuses_a_store_whose_id_columns_repeat(tmp_path):
    from experiments.breakdown import build_atom_table

    pd = pytest.importorskip("pandas")
    root = _store(tmp_path)
    path = root / "s" / "molecules.parquet"
    df = pd.read_parquet(path)
    df.loc[1, "dash_id"] = "m0"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        build_atom_table("s", stores_root=root, n_jobs=1)


# --- the atoms a reference arm covers ---------------------------------------


def test_coverage_scopes_follow_the_reference_predictions():
    from experiments.breakdown import coverage_scopes

    nan = np.nan
    # Three records of 2, 3 and 2 atoms: complete, partly covered, empty.
    reference = np.array([0.1, 0.2, 0.3, nan, 0.5, nan, nan])
    scopes = coverage_scopes(reference, np.array([2, 3, 2]))
    assert scopes["all"].all()
    np.testing.assert_array_equal(scopes["covered"], [1, 1, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(scopes["uncovered"], [0, 0, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(scopes["complete"], [1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(scopes["incomplete"], [0, 0, 1, 1, 1, 1, 1])


def test_coverage_outcomes_count_atoms_and_records():
    from experiments.breakdown import coverage_outcomes

    nan = np.nan
    reference = np.array([0.1, 0.2, 0.3, nan, 0.5, nan, nan])
    out = coverage_outcomes(reference, np.array([2, 3, 2]))
    assert out["coverage/n_atoms"] == 7
    assert out["coverage/n_records"] == 3
    assert out["coverage/atom_fraction_uncovered"] == pytest.approx(3 / 7)
    assert out["coverage/atom_fraction_incomplete"] == pytest.approx(5 / 7)
    assert out["coverage/record_fraction_incomplete"] == pytest.approx(2 / 3)
    assert out["coverage/record_fraction_empty"] == pytest.approx(1 / 3)


# --- the breakdown itself ---------------------------------------------------


def _arrays(n=40, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=n)
    pred = true + rng.normal(scale=0.1, size=n)
    return true, pred


def test_breakdown_scores_each_scope_on_its_own_atoms():
    from experiments.breakdown import breakdown

    true, pred = _arrays()
    sel = np.arange(true.size) % 3 == 0
    got = breakdown(true, pred, {"all": np.ones(true.size, bool), "third": sel})
    for name, mask in (("all", np.ones(true.size, bool)), ("third", sel)):
        want = _expected(true, pred, mask)
        assert got[f"{name}/n_atoms"] == int(mask.sum())
        assert got[f"{name}/n_scored"] == int(mask.sum())
        for metric, value in want.items():
            assert got[f"{name}/{metric}"] == pytest.approx(value)


def test_breakdown_leaves_atoms_without_a_prediction_out_of_the_scores():
    from experiments.breakdown import breakdown

    true, pred = _arrays()
    pred[:5] = np.nan
    got = breakdown(true, pred, {"all": np.ones(true.size, bool)})
    finite = np.isfinite(pred)
    assert got["all/n_atoms"] == true.size
    assert got["all/n_scored"] == true.size - 5
    assert got["all/rmse"] == pytest.approx(_expected(true, pred, finite)["rmse"])


def test_breakdown_reports_quantiles_of_the_absolute_error():
    from experiments.breakdown import breakdown

    true, pred = _arrays(n=1000)
    got = breakdown(true, pred, {"all": np.ones(true.size, bool)})
    err = np.abs(pred - true)
    assert got["all/abs_error/q50"] == pytest.approx(float(np.quantile(err, 0.5)))
    assert got["all/abs_error/q99"] == pytest.approx(float(np.quantile(err, 0.99)))
    assert got["all/abs_error/q999"] == pytest.approx(float(np.quantile(err, 0.999)))
    assert got["all/abs_error/max"] == pytest.approx(float(err.max()))


def test_breakdown_scores_each_group_within_a_scope():
    from experiments.breakdown import breakdown

    true, pred = _arrays()
    element = np.where(np.arange(true.size) % 2 == 0, 6, 8)
    sel = np.arange(true.size) < 30
    got = breakdown(
        true,
        pred,
        {"first": sel},
        groups={"element": (element, {6: "C", 8: "O"})},
    )
    for z, symbol in ((6, "C"), (8, "O")):
        mask = sel & (element == z)
        want = _expected(true, pred, mask)
        assert got[f"first/element/{symbol}/n_atoms"] == int(mask.sum())
        for metric, value in want.items():
            assert got[f"first/element/{symbol}/{metric}"] == pytest.approx(value)


def test_an_empty_scope_is_reported_without_scores():
    from experiments.breakdown import breakdown

    true, pred = _arrays()
    got = breakdown(true, pred, {"none": np.zeros(true.size, bool)})
    assert got["none/n_atoms"] == 0
    assert np.isnan(got["none/rmse"])
    assert np.isnan(got["none/abs_error/q50"])


# --- the HOSE register's last resort ----------------------------------------


def test_fallback_atoms_are_those_predicted_at_the_global_mean():
    from experiments.breakdown import fallback_atoms

    mean = 0.1 + 0.2  # not exactly 0.3 in floating point
    pred = np.array([0.3, mean, 0.3 + 1e-9, -0.3, np.nan])
    np.testing.assert_array_equal(fallback_atoms(pred, 0.3), [1, 1, 0, 0, 0])


def test_element_fallback_replaces_only_the_fallback_atoms():
    from experiments.breakdown import with_element_fallback

    pred = np.array([0.5, 0.0, 0.0, 0.7, 0.0])
    fallback = np.array([False, True, True, False, True])
    element = np.array([6, 6, 8, 8, 9])
    out = with_element_fallback(pred, fallback, element, {6: -0.1, 8: -0.4})
    # Fluorine has no training atom, so its atom keeps the global mean.
    np.testing.assert_allclose(out, [0.5, -0.1, -0.4, 0.7, 0.0])
    assert pred[1] == 0.0  # the input is not modified


def _hose_state(table: dict[str, tuple[float, int]]):
    from experiments.hose_artifact import HoseState

    one = {k: (s, float("nan"), n) for k, (s, n) in table.items()}
    return HoseState(
        max_radius=1,
        tables=({}, one),
        global_sum=sum(s for s, _ in table.values()),
        global_count=sum(n for _, n in table.values()),
    )


def test_hose_fold_statistics_pool_the_training_shards(tmp_path):
    from experiments.breakdown import hose_fold_statistics
    from experiments.hose_artifact import save_hose_state

    a = _hose_state({"C-4;HHHC(": (0.6, 3), "O-2;HC(": (-1.2, 2)})
    b = _hose_state({"C-4;HHHC(": (0.2, 1), "C-3;=OOC(": (1.5, 2), "H-1;C(": (0.4, 4)})
    save_hose_state(a, tmp_path / "a.npz")
    save_hose_state(b, tmp_path / "b.npz")
    stats = hose_fold_statistics([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert stats.global_mean == pytest.approx((0.6 - 1.2 + 0.2 + 1.5 + 0.4) / 12)
    assert stats.element_mean[6] == pytest.approx((0.6 + 0.2 + 1.5) / 6)
    assert stats.element_mean[8] == pytest.approx(-1.2 / 2)
    assert stats.element_mean[1] == pytest.approx(0.4 / 4)
    assert set(stats.element_mean) == {1, 6, 8}


# --- scoring saved runs -----------------------------------------------------


def _cv_run(
    runs_root,
    experiment,
    *,
    method,
    depth,
    normalization,
    true,
    pred,
    rows,
    num_atoms,
    repeat=0,
    fold=0,
    metrics=None,
):
    run = runs_root / experiment / f"r{repeat}-f{fold}-{method}-w{depth}__x"
    run.mkdir(parents=True)
    cv = {
        "method": method,
        "depth": depth,
        "repeat": repeat,
        "fold": fold,
        "normalization": normalization,
        "held_out_shards": "s01",
    }
    (run / "manifest.json").write_text(json.dumps({"config": {"cv": cv}}))
    (run / "metrics.json").write_text(json.dumps(metrics or {}))
    np.savez(
        run / "predictions.npz",
        dash_id=np.array([f"m{i}" for i in rows]),
        conf_id=np.array(["c0"] * len(rows)),
        num_atoms=num_atoms,
        atom_target_true=true,
        atom_target_pred=pred,
        molecule_value=np.array([NET[i] for i in rows]),
    )
    return run


@pytest.fixture
def scored(tmp_path):
    """One CV sample of three arms on four records, and its training shard."""
    from experiments.breakdown import Arm, build_atom_table, load_atom_table
    from experiments.hose_artifact import save_hose_state

    root = _store(tmp_path)
    table = load_atom_table(build_atom_table("s", stores_root=root, n_jobs=1))
    mols = [_mol(s) for s in SMILES]
    rows = [0, 1, 2, 3]
    num_atoms = np.array([mols[i].GetNumAtoms() for i in rows])
    element = np.concatenate(
        [[a.GetAtomicNum() for a in mols[i].GetAtoms()] for i in rows]
    )
    n = int(num_atoms.sum())
    rng = np.random.default_rng(1)
    true = rng.normal(scale=0.3, size=n)
    start = np.concatenate([[0], np.cumsum(num_atoms)])

    # The reference arm misses one atom of record 1 and all of record 3.
    dash = true + rng.normal(scale=0.05, size=n)
    dash[start[1] + 2] = np.nan
    dash[start[3] : start[4]] = np.nan
    complete = np.ones(n, bool)
    complete[start[1] : start[2]] = False
    complete[start[3] : start[4]] = False

    # The training shard: its 1-sphere table gives the global and element means.
    shard = _hose_state(
        {
            "C-4;HHHC(": (0.4, 4),
            "O-2;HC(": (-1.0, 2),
            "H-1;C(": (0.6, 6),
            "N-4+;HHHC(": (-0.3, 1),
        }
    )
    runs = tmp_path / "runs"
    fit = runs / "cv-shard-fits" / "fit-hose-r1-s00__x"
    fit.mkdir(parents=True)
    save_hose_state(shard, fit / "tree_stats.npz")
    global_mean = shard.global_mean

    # HOSE answers every fluoromethane atom, and one ethanol carbon, with the
    # global mean.
    hose = true + rng.normal(scale=0.08, size=n)
    fallback = np.zeros(n, bool)
    fallback[start[3] : start[4]] = True
    fallback[0] = True
    hose[fallback] = global_mean

    sieve = true + rng.normal(scale=0.04, size=n)

    common = {"true": true, "rows": rows, "num_atoms": num_atoms}
    norm_true = true[complete]
    paths = {
        "dash": _cv_run(
            runs,
            "dash-b",
            method="dash",
            depth=16,
            normalization="std_weighted",
            pred=dash,
            metrics={
                "n_test_atoms": float(n),
                "norm/n_nan": float(n - int(complete.sum())),
                "norm/rmse": 0.011,
                "norm/mae": 0.007,
                "norm/r2": 0.99,
            },
            **common,
        ),
        "hose": _cv_run(
            runs,
            "hose-b",
            method="hose",
            depth=1,
            normalization="equal_weighted",
            pred=hose,
            **common,
        ),
        "sieve": _cv_run(
            runs,
            "sieve-b",
            method="sieve",
            depth=5,
            normalization="equal_weighted",
            pred=sieve,
            **common,
        ),
    }
    arms = {
        "dash": Arm("dash-b", "dash", 16),
        "hose": Arm("hose-b", "hose", 1),
        "sieve": Arm("sieve-b", "sieve", 5),
    }
    return {
        "runs": runs,
        "table": table,
        "paths": paths,
        "arms": arms,
        "true": true,
        "pred": {"dash": dash, "hose": hose, "sieve": sieve},
        "num_atoms": num_atoms,
        "start": start,
        "element": element,
        "complete": complete,
        "fallback": fallback,
        "global_mean": global_mean,
        "norm_true": norm_true,
    }


def _score(s, **kwargs):
    from experiments.breakdown import score_runs

    return score_runs(
        s["runs"],
        list(s["arms"].values()),
        s["table"],
        reference=s["arms"]["dash"],
        hose=s["arms"]["hose"],
        n_shards=2,
        **kwargs,
    )


def _sidecar(s, arm):
    from experiments.breakdown import METRICS_FILE

    return json.loads((s["paths"][arm] / METRICS_FILE).read_text())


def test_every_arm_is_scored_on_the_atoms_the_reference_covers(scored):
    written = _score(scored)
    assert sorted(written) == sorted(scored["paths"].values())
    covered = np.isfinite(scored["pred"]["dash"])
    for arm in ("dash", "hose", "sieve"):
        got = _sidecar(scored, arm)
        want = _expected(scored["true"], scored["pred"][arm], covered)
        assert got["covered/n_atoms"] == int(covered.sum())
        assert got["covered/rmse"] == pytest.approx(want["rmse"])
        assert got["coverage/record_fraction_incomplete"] == pytest.approx(0.5)
    sieve = _sidecar(scored, "sieve")
    want = _expected(scored["true"], scored["pred"]["sieve"], ~covered)
    assert sieve["uncovered/rmse"] == pytest.approx(want["rmse"])
    assert _sidecar(scored, "dash")["uncovered/n_scored"] == 0


def test_groups_are_scored_by_element_and_charge_state(scored):
    _score(scored)
    got = _sidecar(scored, "sieve")
    true, pred = scored["true"], scored["pred"]["sieve"]
    oxygen = scored["element"] == 8
    assert got["all/element/O/rmse"] == pytest.approx(
        _expected(true, pred, oxygen)["rmse"]
    )
    start = scored["start"]
    anion = np.zeros(true.size, bool)
    anion[start[2] : start[3]] = True  # acetate, net charge -1
    assert got["all/net_charge/-1/rmse"] == pytest.approx(
        _expected(true, pred, anion)["rmse"]
    )
    assert got["all/net_charge/+1/n_atoms"] == 8
    assert got["all/net_charge/0/n_atoms"] == 9 + 5
    # One charged nitrogen and one charged oxygen in the four records.
    assert got["all/formal_charge/+1/n_atoms"] == 1
    assert got["all/formal_charge/-1/n_atoms"] == 1


def test_equal_weighted_arms_are_also_scored_after_normalization(scored):
    from experiments.normalize import equal_weighted_normalize

    _score(scored)
    true, pred = scored["true"], scored["pred"]["sieve"]
    num_atoms = scored["num_atoms"]
    mol = np.repeat(np.arange(num_atoms.size), num_atoms)
    norm = equal_weighted_normalize(pred, pred, np.array(NET), mol, num_atoms.size)
    got = _sidecar(scored, "sieve")
    want = _expected(true, norm, scored["complete"])
    assert got["norm/complete/rmse"] == pytest.approx(want["rmse"])
    assert got["norm/complete/n_atoms"] == int(scored["complete"].sum())


def test_the_reference_arm_takes_its_normalized_scores_from_the_run(scored):
    _score(scored)
    got = _sidecar(scored, "dash")
    assert got["norm/complete/rmse"] == 0.011
    assert got["norm/complete/mae"] == 0.007
    assert got["norm/complete/n_atoms"] == int(scored["complete"].sum())


def test_normalized_scores_on_other_atoms_than_the_complete_records_are_refused(
    scored,
):
    path = scored["paths"]["dash"] / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["norm/n_nan"] += 1.0
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="complete"):
        _score(scored)


def test_the_hose_arm_reports_its_fallback_and_the_element_control(scored):
    _score(scored)
    got = _sidecar(scored, "hose")
    true, pred = scored["true"], scored["pred"]["hose"]
    fallback = scored["fallback"]
    assert got["fallback/n_atoms"] == int(fallback.sum())
    assert got["fallback/atom_fraction"] == pytest.approx(fallback.mean())
    assert got["fallback/n_records"] == 2
    assert got["fallback/rmse"] == pytest.approx(
        _expected(true, pred, fallback)["rmse"]
    )
    assert got["fallback/element/F/n_atoms"] == 1
    assert got["fallback/element/C/n_atoms"] == 2

    control = pred.copy()
    means = {6: 0.4 / 4, 1: 0.6 / 6}  # fluorine has no training atom
    for z, mean in means.items():
        control[fallback & (scored["element"] == z)] = mean
    everything = np.ones(true.size, bool)
    assert got["element_fallback/all/rmse"] == pytest.approx(
        _expected(true, control, everything)["rmse"]
    )


def test_every_arm_is_scored_on_the_atoms_where_hose_falls_back(scored):
    _score(scored)
    got = _sidecar(scored, "sieve")
    want = _expected(scored["true"], scored["pred"]["sieve"], scored["fallback"])
    assert got["hose_fallback/n_atoms"] == int(scored["fallback"].sum())
    assert got["hose_fallback/rmse"] == pytest.approx(want["rmse"])


def test_scored_runs_are_skipped_until_forced(scored):
    assert len(_score(scored)) == 3
    assert _score(scored) == []
    assert len(_score(scored, force=True)) == 3


def test_arms_that_held_out_different_atoms_are_refused(scored):
    path = scored["paths"]["sieve"] / "predictions.npz"
    z = dict(np.load(path))
    z["atom_target_true"] = z["atom_target_true"] + 1.0
    np.savez(path, **z)
    with pytest.raises(ValueError, match="held-out"):
        _score(scored)


def test_a_sample_without_its_reference_run_is_refused(scored):
    import shutil

    shutil.rmtree(scored["paths"]["dash"])
    with pytest.raises(FileNotFoundError, match="reference"):
        _score(scored)


# --- the report --------------------------------------------------------------


def test_sample_table_pairs_arms_by_repeat_and_fold(scored):
    from experiments.breakdown import sample_table

    _score(scored)
    samples, values = sample_table(
        scored["runs"], list(scored["arms"].values()), ["covered/rmse", "all/n_atoms"]
    )
    assert samples == [(0, 0)]
    covered = np.isfinite(scored["pred"]["dash"])
    want = _expected(scored["true"], scored["pred"]["sieve"], covered)["rmse"]
    assert values["covered/rmse"]["sieve"] == pytest.approx([want])
    assert values["all/n_atoms"]["hose"] == [scored["true"].size]


def test_report_names_every_arm_and_the_coverage(scored):
    from experiments.breakdown import breakdown_report

    _score(scored)
    text, snapshot = breakdown_report(
        scored["runs"],
        list(scored["arms"].values()),
        incumbent=scored["arms"]["sieve"],
        hose=scored["arms"]["hose"],
        k=5,
    )
    for word in ("dash", "hose", "sieve", "covered", "element"):
        assert word in text
    assert snapshot["samples"] == [[0, 0]]
    assert "covered/rmse" in snapshot["values"]


# --- reading arms and scores from elsewhere ---------------------------------


def test_an_arm_is_read_from_experiment_method_and_depth():
    from experiments.breakdown import Arm, parse_arm

    assert parse_arm("dash-cv-study-b:dash:16") == Arm("dash-cv-study-b", "dash", 16)
    for spec in ("dash-cv-study-b:dash", "a:b:c", "::5"):
        with pytest.raises(ValueError, match="EXPERIMENT:METHOD:DEPTH"):
            parse_arm(spec)


def test_the_run_reader_merges_the_breakdown_into_a_runs_metrics(scored):
    from experiments.aggregate import read_runs_from_dirs

    _score(scored)
    (row,) = read_runs_from_dirs(scored["runs"], "sieve-b")
    covered = np.isfinite(scored["pred"]["dash"])
    want = _expected(scored["true"], scored["pred"]["sieve"], covered)["rmse"]
    assert row.metrics["covered/rmse"] == pytest.approx(want)


# --- total charge of a record -----------------------------------------------


def test_total_charge_is_scored_on_each_set_of_records():
    from experiments.breakdown import total_charge_scores

    num_atoms = np.array([2, 3, 2])
    pred = np.array([0.1, 0.2, -0.5, -0.4, 0.2, 0.6, 0.6])  # sums 0.3, -0.7, 1.2
    net = np.array([0.0, -1.0, 1.0])
    sets = {
        "all": np.array([True, True, True]),
        "complete": np.array([True, False, True]),
    }
    got = total_charge_scores(pred, net, num_atoms, sets)
    assert got["all/total/n_records"] == 3
    assert got["all/total/rmse"] == pytest.approx(
        float(np.sqrt((0.3**2 + 0.3**2 + 0.2**2) / 3))
    )
    assert got["complete/total/n_records"] == 2
    assert got["complete/total/n_scored"] == 2
    assert got["complete/total/rmse"] == pytest.approx(
        float(np.sqrt((0.3**2 + 0.2**2) / 2))
    )
    assert got["complete/total/mae"] == pytest.approx(0.25)


def test_a_record_with_an_atom_without_prediction_has_no_total():
    from experiments.breakdown import total_charge_scores

    pred = np.array([0.1, np.nan, 0.5, 0.4])
    got = total_charge_scores(
        pred, np.array([0.0, 1.0]), np.array([2, 2]), {"all": np.array([True, True])}
    )
    assert got["all/total/n_records"] == 2
    assert got["all/total/n_scored"] == 1
    assert got["all/total/rmse"] == pytest.approx(0.1)


def test_every_arm_reports_its_total_charge_error_on_the_complete_records(scored):
    _score(scored)
    num_atoms, start = scored["num_atoms"], scored["start"]
    complete = np.array([True, False, True, False])
    for arm in ("sieve", "hose"):
        pred = scored["pred"][arm]
        sums = np.array([pred[start[i] : start[i + 1]].sum() for i in range(4)])
        err = (sums - np.array(NET))[complete]
        got = _sidecar(scored, arm)
        assert got["complete/total/rmse"] == pytest.approx(
            float(np.sqrt(np.mean(err**2)))
        )
        assert got["complete/total/n_records"] == 2
        assert got["all/total/n_records"] == num_atoms.size
    dash = _sidecar(scored, "dash")
    assert dash["all/total/n_scored"] == 2
    assert dash["all/total/rmse"] == pytest.approx(dash["complete/total/rmse"])
