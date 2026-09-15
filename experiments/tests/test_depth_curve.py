"""Tests for depth_curve.py -- the Study A depth-curve figure and the
Nadeau-Bengio variance correction its error bars carry.

The correction is the load-bearing part and is tested as arithmetic, not
through the figure: k-fold scores share training molecules, so the naive
standard error of their mean is optimistic, and how optimistic is exactly
what the reader of a depth curve needs to know before calling two depths
different.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def _write_cv_run(
    runs_root: Path,
    experiment: str,
    *,
    method: str,
    depth: int,
    fold: int,
    metrics: dict[str, float],
    repeat: int = 0,
) -> Path:
    """One CV run directory, shaped like cv.py's own output."""
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
            "data": {"store": "test-store"},
            "cv": {
                "repeat": repeat,
                "fold": fold,
                "method": method,
                "depth": depth,
                "normalization": "equal_weighted",
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def _arm(runs_root: Path, experiment: str, method: str, *, depths, folds=5) -> None:
    """A full depth curve: one run per (depth, fold), equal fold sizes."""
    for depth in depths:
        for fold in range(folds):
            _write_cv_run(
                runs_root,
                experiment,
                method=method,
                depth=depth,
                fold=fold,
                metrics={
                    "rmse": 0.1 / (depth + 1) + 0.001 * fold,
                    "r2": 1.0 - 0.1 / (depth + 1) - 0.001 * fold,
                    "n_test_conformers": 1000.0,
                },
            )


# --- the correction itself --------------------------------------------------


def test_nadeau_bengio_se_matches_its_closed_form():
    from experiments.depth_curve import nadeau_bengio_se

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    # S^2 with ddof=1 over 1..5 is 2.5.
    se = nadeau_bengio_se(values, n_test=200.0, n_train=800.0)

    expected = math.sqrt((1.0 / 5.0 + 200.0 / 800.0) * 2.5)
    assert se == pytest.approx(expected)


def test_nadeau_bengio_se_is_naive_se_times_sqrt_one_plus_k_rho():
    """The whole point: it inflates the naive SE by sqrt(1 + k*n2/n1).

    At this study's own geometry -- k=5 folds, each holding out 10 of 50
    shards, so n2/n1 = 1/4 -- that factor is exactly 1.5.
    """
    from experiments.depth_curve import nadeau_bengio_se

    values = [0.9, 1.0, 1.1, 1.2, 0.8]
    k = len(values)
    variance = sum((v - sum(values) / k) ** 2 for v in values) / (k - 1)
    naive_se = math.sqrt(variance / k)

    se = nadeau_bengio_se(values, n_test=1000.0, n_train=4000.0)

    assert se == pytest.approx(1.5 * naive_se)
    assert se > naive_se


def test_nadeau_bengio_se_reduces_to_naive_when_test_set_is_negligible():
    """n2/n1 -> 0 is the independent-replicate limit, where no correction
    is due -- a sanity check that the correction is an addition to the
    naive term rather than a replacement for it."""
    from experiments.depth_curve import nadeau_bengio_se

    values = [0.9, 1.0, 1.1, 1.2, 0.8]
    k = len(values)
    variance = sum((v - sum(values) / k) ** 2 for v in values) / (k - 1)

    se = nadeau_bengio_se(values, n_test=1e-9, n_train=1.0)

    assert se == pytest.approx(math.sqrt(variance / k))


def test_nadeau_bengio_se_of_identical_folds_is_zero():
    from experiments.depth_curve import nadeau_bengio_se

    assert nadeau_bengio_se([2.0] * 5, n_test=1.0, n_train=4.0) == 0.0


def test_nadeau_bengio_se_refuses_a_single_fold():
    """One observation cannot carry a spread. Drawing a zero-width bar
    there would assert a precision that was never measured."""
    from experiments.depth_curve import nadeau_bengio_se

    with pytest.raises(ValueError, match="at least two"):
        nadeau_bengio_se([1.0], n_test=1.0, n_train=4.0)


def test_nadeau_bengio_se_refuses_an_empty_training_set():
    from experiments.depth_curve import nadeau_bengio_se

    with pytest.raises(ValueError, match="n_train"):
        nadeau_bengio_se([1.0, 2.0], n_test=1.0, n_train=0.0)


# --- reading the curve off real run directories -----------------------------


def test_read_depth_curve_groups_folds_by_depth(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "study-a", "sieve-pooled", depths=[0, 2, 4])

    curve = read_depth_curve(runs_root, "study-a", method="sieve-pooled", metric="rmse")

    assert curve.method == "sieve-pooled"
    assert [p.depth for p in curve.points] == [0, 2, 4]
    assert all(len(p.fold_values) == 5 for p in curve.points)
    assert curve.n_runs == 15


def test_read_depth_curve_derives_n_train_from_the_other_folds(tmp_path):
    """Folds partition the train split, so the training size for a fold is
    the total minus its own held-out part -- read from the runs themselves
    rather than assumed to be (k-1)/k."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    for fold, n_test in enumerate([100.0, 200.0, 300.0, 400.0]):
        _write_cv_run(
            runs_root,
            "study-a",
            method="m",
            depth=3,
            fold=fold,
            metrics={"rmse": 1.0 + fold, "n_test_conformers": n_test},
        )

    curve = read_depth_curve(runs_root, "study-a", method="m", metric="rmse")
    point = curve.points[0]

    assert point.fold_n_test == (100.0, 200.0, 300.0, 400.0)
    # total 1000, so each fold trains on 1000 - its own held-out count
    assert point.fold_n_train == (900.0, 800.0, 700.0, 600.0)


def test_read_depth_curve_raises_for_a_method_with_no_runs(tmp_path):
    """An arm nobody ran must not become an empty panel -- that reads as a
    method that failed rather than one that was never scored."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "study-a", "sieve-pooled", depths=[0, 2])

    with pytest.raises(ValueError, match="no runs"):
        read_depth_curve(runs_root, "study-a", method="absent", metric="rmse")


def test_read_depth_curve_raises_for_a_metric_no_run_recorded(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "study-a", "sieve-pooled", depths=[0, 2])

    with pytest.raises(ValueError, match="no runs"):
        read_depth_curve(
            runs_root, "study-a", method="sieve-pooled", metric="not_a_metric"
        )


def test_read_depth_curve_skips_a_depth_whose_runs_lack_the_metric(tmp_path):
    """A depth missing the metric contributes no point, exactly as
    build_curve treats it -- it must not become a zero or a gap silently
    filled by its neighbours."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "study-a", "m", depths=[0, 1])
    for fold in range(5):
        _write_cv_run(
            runs_root,
            "study-a",
            method="m",
            depth=2,
            fold=fold,
            metrics={"n_test_conformers": 1000.0},
        )

    curve = read_depth_curve(runs_root, "study-a", method="m", metric="rmse")

    assert [p.depth for p in curve.points] == [0, 1]


# --- the reference line -----------------------------------------------------


def test_best_value_minimises_an_error_metric(tmp_path):
    from experiments.depth_curve import best_value, read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "m", depths=[0, 1, 2])

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse")

    # rmse falls with depth in the fixture, so the best is the deepest.
    assert best_value(curve) == pytest.approx(min(p.mean for p in curve.points))


def test_best_value_maximises_a_score_metric(tmp_path):
    """r2 is a score, not an error: its floor is the largest value, and
    drawing the smallest would put the reference line at the WORST depth
    the other method reached."""
    from experiments.depth_curve import best_value, read_depth_curve

    runs_root = tmp_path / "runs"
    for depth, r2 in [(0, 0.5), (1, 0.9), (2, 0.99)]:
        for fold in range(5):
            _write_cv_run(
                runs_root,
                "a",
                method="m",
                depth=depth,
                fold=fold,
                metrics={"r2": r2 + 0.001 * fold, "n_test_conformers": 1000.0},
            )

    curve = read_depth_curve(runs_root, "a", method="m", metric="r2")

    assert best_value(curve) == pytest.approx(max(p.mean for p in curve.points))


# --- the figure -------------------------------------------------------------


def _grid(runs_root: Path):
    from experiments.depth_curve import read_depth_curve

    _arm(runs_root, "sieve-a", "sieve-pooled", depths=[0, 2, 4, 6])
    _arm(runs_root, "dash-a", "dash", depths=[2, 8, 16])
    return [
        [
            read_depth_curve(
                runs_root, "sieve-a", method="sieve-pooled", metric=metric
            ),
            read_depth_curve(runs_root, "dash-a", method="dash", metric=metric),
        ]
        for metric in ("rmse", "r2")
    ]


def test_plot_depth_grid_writes_vector_and_raster(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.depth_curve import plot_depth_grid

    runs_root = tmp_path / "runs"
    outputs = plot_depth_grid(
        _grid(runs_root), tmp_path / "figures" / "depth-curve", store="test-store"
    )

    assert [p.suffix for p in outputs] == [".pdf", ".png", ".txt"]
    assert all(p.exists() and p.stat().st_size > 0 for p in outputs)


def test_plot_depth_grid_refuses_no_rows(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.depth_curve import plot_depth_grid

    with pytest.raises(ValueError, match="at least one"):
        plot_depth_grid([], tmp_path / "fig", store="s")


def test_plot_depth_grid_refuses_ragged_rows(tmp_path):
    """Every row must carry the same methods in the same order, or a column
    would show one method's rmse above another method's r2."""
    pytest.importorskip("matplotlib")
    from experiments.depth_curve import plot_depth_grid, read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "sieve-a", "sieve-pooled", depths=[0, 2])
    _arm(runs_root, "dash-a", "dash", depths=[2, 8])
    row_a = [
        read_depth_curve(runs_root, "sieve-a", method="sieve-pooled", metric="rmse"),
        read_depth_curve(runs_root, "dash-a", method="dash", metric="rmse"),
    ]
    row_b = [read_depth_curve(runs_root, "sieve-a", method="sieve-pooled", metric="r2")]

    with pytest.raises(ValueError, match="same methods"):
        plot_depth_grid([row_a, row_b], tmp_path / "fig", store="s")


# --- the train curve --------------------------------------------------------


def test_read_depth_curve_picks_up_the_train_family(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    for depth in (0, 1):
        for fold in range(5):
            _write_cv_run(
                runs_root,
                "a",
                method="m",
                depth=depth,
                fold=fold,
                metrics={
                    "rmse": 0.1 / (depth + 1) + 0.001 * fold,
                    # train error is lower than held-out -- that gap is the
                    # whole point of the curve
                    "train/rmse": 0.05 / (depth + 1) + 0.001 * fold,
                    "n_test_conformers": 1000.0,
                },
            )

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse")

    assert [p.depth for p in curve.train_points] == [0, 1]
    assert all(
        t.mean < v.mean for t, v in zip(curve.train_points, curve.points, strict=True)
    )


def test_read_depth_curve_leaves_train_empty_when_runs_lack_it(tmp_path):
    """Runs made without --score-train must yield no train curve, rather
    than a curve that silently duplicates the validation one."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "m", depths=[0, 1])

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse")

    assert curve.points
    assert curve.train_points == ()


# --- omitting shallow depths ------------------------------------------------


def test_read_depth_curve_min_depth_drops_shallow_points_and_records_them(tmp_path):
    """A trimmed depth axis still reports how wide the sweep ran, so the
    caption can name the candidates that were scored but not plotted."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "m", depths=[0, 1, 2, 3])

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse", min_depth=2)

    assert [p.depth for p in curve.points] == [2, 3]
    assert curve.omitted_depths == (0, 1)


def test_read_depth_curve_min_depth_applies_to_the_train_curve_too(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    for depth in (0, 1, 2):
        for fold in range(5):
            _write_cv_run(
                runs_root,
                "a",
                method="m",
                depth=depth,
                fold=fold,
                metrics={
                    "rmse": 0.1 / (depth + 1),
                    "train/rmse": 0.05 / (depth + 1),
                    "n_test_conformers": 1000.0,
                },
            )

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse", min_depth=1)

    assert [p.depth for p in curve.points] == [1, 2]
    assert [p.depth for p in curve.train_points] == [1, 2]


def test_read_depth_curve_without_min_depth_omits_nothing(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "m", depths=[0, 1])

    curve = read_depth_curve(runs_root, "a", method="m", metric="rmse")

    assert [p.depth for p in curve.points] == [0, 1]
    assert curve.omitted_depths == ()


# --- confidence intervals ---------------------------------------------------


def test_ci_half_width_is_the_corrected_resampled_t_interval():
    """Nadeau-Bengio's variance with Student's t on k-1 degrees of freedom
    is the corrected resampled t-test's own interval, so the bars and any
    later significance claim rest on the same quantity."""
    pytest.importorskip("scipy")
    from experiments.depth_curve import ci_half_width
    from scipy.stats import t as student_t

    se, k = 0.00022, 5
    assert ci_half_width(se, k) == pytest.approx(student_t.ppf(0.975, k - 1) * se)
    # k=5 -> t(0.975, 4) = 2.7764
    assert ci_half_width(se, k) == pytest.approx(2.7764 * se, rel=1e-4)


def test_ci_half_width_widens_as_folds_shrink():
    """Fewer folds means a heavier tail, so the interval must widen even at
    identical standard error."""
    pytest.importorskip("scipy")
    from experiments.depth_curve import ci_half_width

    assert ci_half_width(1.0, 3) > ci_half_width(1.0, 5) > ci_half_width(1.0, 25)


def test_ci_half_width_refuses_fewer_than_two_folds():
    pytest.importorskip("scipy")
    from experiments.depth_curve import ci_half_width

    with pytest.raises(ValueError, match="at least two"):
        ci_half_width(1.0, 1)


# --- display labels and the caption file ------------------------------------


def test_read_depth_curve_carries_a_display_label(tmp_path):
    """The method key names the runs on disk; the label is what a reader
    sees. They differ deliberately: 'sieve-element-pooled' identifies an arm
    unambiguously in a run directory and is noise on an axis."""
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "sieve-element-pooled", depths=[0, 1])

    curve = read_depth_curve(
        runs_root, "a", method="sieve-element-pooled", metric="rmse", label="Sieve"
    )

    assert curve.method == "sieve-element-pooled"
    assert curve.label == "Sieve"


def test_read_depth_curve_label_defaults_to_the_method(tmp_path):
    from experiments.depth_curve import read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "m", depths=[0, 1])

    assert read_depth_curve(runs_root, "a", method="m", metric="rmse").label == "m"


def test_plot_depth_grid_writes_the_caption_beside_the_figure(tmp_path):
    """The provenance belongs in the manuscript's own caption, so it is
    written as text rather than burned into the image."""
    pytest.importorskip("matplotlib")
    from experiments.depth_curve import plot_depth_grid

    runs_root = tmp_path / "runs"
    outputs = plot_depth_grid(
        _grid(runs_root), tmp_path / "figures" / "depth-curve", store="test-store"
    )

    assert [p.suffix for p in outputs] == [".pdf", ".png", ".txt"]
    caption = outputs[-1].read_text()
    assert "test-store" in caption
    assert caption.endswith("\n")


def test_caption_file_uses_display_labels(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.depth_curve import plot_depth_grid, read_depth_curve

    runs_root = tmp_path / "runs"
    _arm(runs_root, "a", "sieve-element-pooled", depths=[0, 1, 2])
    row = [
        read_depth_curve(
            runs_root, "a", method="sieve-element-pooled", metric="rmse", label="Sieve"
        )
    ]
    outputs = plot_depth_grid([row], tmp_path / "fig", store="s")
    caption = outputs[-1].read_text()

    assert "Sieve" in caption
    assert "sieve-element-pooled" not in caption
