"""Study D's stereo-affected subsets: per-atom masks, per-run scores, and the
paired report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _mol(smiles):
    from rdkit import Chem

    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_masks_mark_distance_to_the_nearest_stereogenic_double_bond():
    from experiments.stereo_subsets import HAS_EZ, NEAR_1, NEAR_2, subset_masks

    mol = _mol("C/C=C/CCO")  # heavy atoms 0..5; the double bond is 1=2
    masks = subset_masks(mol)
    assert masks.shape == (6, mol.GetNumAtoms())
    assert masks[HAS_EZ].all()
    np.testing.assert_array_equal(
        masks[NEAR_1, :6], [True, True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        masks[NEAR_2, :6], [True, True, True, True, True, False]
    )
    # A hydrogen on C0 is two bonds from C1.
    h_on_c0 = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(0).GetNeighbors()
        if a.GetAtomicNum() == 1
    ]
    assert masks[NEAR_2, h_on_c0].all() and not masks[NEAR_1, h_on_c0].any()


def test_a_molecule_without_e_z_has_empty_masks():
    from experiments.stereo_subsets import subset_masks

    for smiles in ("CCCC", "CC(C)=CC", "C=CC"):
        assert not subset_masks(_mol(smiles)).any()


def _store(tmp_path: Path, smiles: list[str]) -> Path:
    # The store is parquet, and CI's lean environment has no pandas.
    pd = pytest.importorskip("pandas")
    from experiments.data import mol_to_blob

    root = tmp_path / "stores"
    (root / "s").mkdir(parents=True)
    pd.DataFrame(
        {
            "dash_id": [f"m{i}" for i in range(len(smiles))],
            "conf_id": [0] * len(smiles),
            "mol": [mol_to_blob(_mol(s)) for s in smiles],
        }
    ).to_parquet(root / "s" / "molecules.parquet")
    return root


def _predictions(run_dir: Path, mols, rows, seed=0):
    rng = np.random.default_rng(seed)
    n_atoms = np.array([mols[i].GetNumAtoms() for i in rows])
    true = rng.normal(size=int(n_atoms.sum()))
    pred = true + rng.normal(scale=0.1, size=true.shape)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "predictions.npz",
        dash_id=np.array([f"m{i}" for i in rows]),
        conf_id=np.zeros(len(rows), np.int64),
        num_atoms=n_atoms,
        atom_target_true=true,
        atom_target_pred=pred,
    )
    return true, pred


SMILES = ["C/C=C/CCO", "CCCC", r"C/C=C\C", "OCC=O"]


def test_subset_metrics_match_a_direct_computation(tmp_path):
    from experiments.stereo_subsets import (
        NEAR_2,
        build_mask_table,
        load_mask_table,
        subset_masks,
        subset_metrics,
    )

    root = _store(tmp_path, SMILES)
    path = build_mask_table("s", stores_root=root, n_jobs=1)
    table = load_mask_table(path)
    mols = [_mol(s) for s in SMILES]
    rows = [2, 0, 1]
    true, pred = _predictions(tmp_path / "run", mols, rows)
    got = subset_metrics(tmp_path / "run" / "predictions.npz", table)

    near = np.concatenate([subset_masks(mols[i])[NEAR_2] for i in rows])
    err = pred - true
    assert got["near_ez2/n_atoms"] == int(near.sum())
    assert got["near_ez2/rmse"] == pytest.approx(
        float(np.sqrt(np.mean(err[near] ** 2)))
    )
    assert got["near_ez2/mae"] == pytest.approx(float(np.mean(np.abs(err[near]))))


def test_subset_metrics_refuse_a_row_whose_atom_count_disagrees(tmp_path):
    from experiments.stereo_subsets import (
        build_mask_table,
        load_mask_table,
        subset_metrics,
    )

    root = _store(tmp_path, SMILES)
    table = load_mask_table(build_mask_table("s", stores_root=root, n_jobs=1))
    mols = [_mol(s) for s in SMILES]
    _predictions(tmp_path / "run", mols, [0])
    z = dict(np.load(tmp_path / "run" / "predictions.npz"))
    z["num_atoms"] = z["num_atoms"] - 1
    np.savez(tmp_path / "run" / "predictions.npz", **z)
    with pytest.raises(ValueError, match="atoms"):
        subset_metrics(tmp_path / "run" / "predictions.npz", table)


def _cv_run(runs_root, experiment, *, repeat, fold, method, depth, metrics):
    run_dir = runs_root / experiment / f"r{repeat}-f{fold}-{method}-w{depth}__x"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_name": run_dir.name,
        "data": {"split_column": "shard"},
        "seed": 0,
        "git": {"commit": "deadbeef"},
        "config": {
            "run": {"experiment": experiment},
            "predictor": {"name": method},
            "cv": {"repeat": repeat, "fold": fold, "method": method, "depth": depth},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def test_aggregate_merges_the_subset_sidecar_and_refuses_a_collision(tmp_path):
    from experiments.aggregate import read_runs_from_dirs
    from experiments.stereo_subsets import METRICS_FILE

    run = _cv_run(
        tmp_path, "e", repeat=0, fold=0, method="a", depth=5, metrics={"rmse": 1.0}
    )
    (run / METRICS_FILE).write_text(json.dumps({"near_ez2/rmse": 2.0}))
    (row,) = read_runs_from_dirs(tmp_path, "e")
    assert row.metrics == {"rmse": 1.0, "near_ez2/rmse": 2.0}
    (run / METRICS_FILE).write_text(json.dumps({"rmse": 3.0}))
    with pytest.raises(ValueError, match="rmse"):
        read_runs_from_dirs(tmp_path, "e")


def test_score_runs_writes_sidecars_and_reports_what_is_missing(tmp_path):
    from experiments.stereo_subsets import (
        METRICS_FILE,
        build_mask_table,
        load_mask_table,
        missing_scores,
        score_runs,
    )

    root = _store(tmp_path, SMILES)
    table = load_mask_table(build_mask_table("s", stores_root=root, n_jobs=1))
    mols = [_mol(s) for s in SMILES]
    runs = tmp_path / "runs"
    for fold in range(2):
        d = _cv_run(runs, "e", repeat=0, fold=fold, method="a", depth=5, metrics={})
        _predictions(d, mols, [0, 1, 2], seed=fold)
    _cv_run(
        runs, "e", repeat=0, fold=0, method="b", depth=5, metrics={}
    )  # other method
    selection = {"e": ["a"]}
    assert len(missing_scores(runs, selection, depth=5)) == 2
    written = score_runs(runs, selection, table, depth=5)
    assert len(written) == 2 and all((p / METRICS_FILE).exists() for p in written)
    assert missing_scores(runs, selection, depth=5) == []


def test_a_selected_run_without_predictions_is_reported_missing(tmp_path):
    from experiments.stereo_subsets import missing_scores

    runs = tmp_path / "runs"
    _cv_run(runs, "e", repeat=0, fold=0, method="a", depth=5, metrics={})
    assert len(missing_scores(runs, {"e": ["a"]}, depth=5)) == 1


def test_paired_difference_uses_the_nadeau_bengio_correction():
    from experiments.stereo_subsets import paired_difference
    from scipy.stats import t

    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    b = a - np.array([0.1, 0.12, 0.08, 0.11, 0.09, 0.1])
    got = paired_difference(a, b, k=3)
    d = b - a
    se = np.sqrt((1 / d.size + 1 / 2) * d.var(ddof=1))
    half = t.ppf(0.975, d.size - 1) * se
    assert got.mean == pytest.approx(d.mean())
    assert got.lo == pytest.approx(d.mean() - half)
    assert got.hi == pytest.approx(d.mean() + half)
    assert got.n_better == 6 and got.n == 6


def test_the_report_pairs_each_arm_with_its_incumbent(tmp_path):
    from experiments.stereo_subsets import Pair, stereo_report

    rng = np.random.default_rng(0)
    for repeat in range(2):
        for fold in range(3):
            base = 1.0 + rng.normal(scale=0.01)
            _cv_run(
                tmp_path,
                "old",
                repeat=repeat,
                fold=fold,
                method="inc",
                depth=5,
                metrics={"near_ez2/rmse": base, "rmse": 0.5},
            )
            _cv_run(
                tmp_path,
                "new",
                repeat=repeat,
                fold=fold,
                method="ct",
                depth=5,
                metrics={"near_ez2/rmse": base - 0.1, "rmse": 0.5},
            )
    rows = stereo_report(
        tmp_path,
        [Pair("old", "inc", "new", "ct", "cont")],
        depth=5,
        metrics=["near_ez2/rmse", "rmse"],
        k=3,
    )
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["near_ez2/rmse"]["diff"] == pytest.approx(-0.1)
    assert by_metric["near_ez2/rmse"]["n_better"] == 6
    assert by_metric["rmse"]["diff"] == pytest.approx(0.0)
    assert all(r["label"] == "cont" and r["n"] == 6 for r in rows)


def test_tetrahedral_subsets_mark_distance_to_a_tagged_centre():
    from experiments.stereo_subsets import HAS_TET, NEAR_TET1, subset_masks

    mol = _mol("N[C@@H](C)CCO")  # heavy atoms 0..5, centre 1
    m = subset_masks(mol)
    assert m.shape == (6, mol.GetNumAtoms()) and m[HAS_TET].all()
    np.testing.assert_array_equal(
        m[NEAR_TET1, :6], [True, True, True, True, False, False]
    )


def test_a_mask_table_from_an_older_subset_list_is_refused(tmp_path):
    from experiments.stereo_subsets import build_mask_table, load_mask_table

    root = _store(tmp_path, SMILES)
    path = build_mask_table("s", stores_root=root, n_jobs=1)
    z = dict(np.load(path))
    z["subsets"] = z["subsets"][:3]
    np.savez(path, **z)
    with pytest.raises(ValueError, match="rebuild"):
        load_mask_table(path)


def test_a_sidecar_missing_a_subset_is_stale(tmp_path):
    from experiments.stereo_subsets import METRICS_FILE, missing_scores

    runs = tmp_path / "runs"
    d = _cv_run(runs, "e", repeat=0, fold=0, method="a", depth=5, metrics={})
    (d / "predictions.npz").write_bytes(b"x")
    (d / METRICS_FILE).write_text(json.dumps({"has_ez/rmse": 1.0}))
    assert missing_scores(runs, {"e": ["a"]}, depth=5) == [d]
