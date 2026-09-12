"""Fast-suite tests for compare.py's ANOVA/Tukey HSD statistics -- pure
numpy/scipy, plus a small on-disk manifest fixture for read_cv_table."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")


def _write_cv_manifest(
    runs_root: Path,
    experiment: str,
    *,
    repeat: int,
    fold: int,
    method: str,
    depth: int,
    mae: float,
) -> None:
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
    (run_dir / "metrics.json").write_text(json.dumps({"mae": mae}))


def test_read_cv_table_pivots_by_repeat_fold_and_method(tmp_path):
    from experiments.compare import read_cv_table

    runs_root = tmp_path / "runs"
    for repeat in range(2):
        for fold in range(3):
            _write_cv_manifest(
                runs_root,
                "exp",
                repeat=repeat,
                fold=fold,
                method="a",
                depth=5,
                mae=float(repeat * 10 + fold),
            )
            _write_cv_manifest(
                runs_root,
                "exp",
                repeat=repeat,
                fold=fold,
                method="b",
                depth=5,
                mae=float(repeat * 10 + fold) + 0.5,
            )

    methods, table = read_cv_table(runs_root, ["exp"], metric="mae")

    assert methods == ["a", "b"]
    assert table.shape == (6, 2)
    # b is always exactly 0.5 higher than a, sample for sample.
    np.testing.assert_allclose(table[:, 1] - table[:, 0], 0.5)


def test_read_cv_table_filters_by_depth_per_method(tmp_path):
    from experiments.compare import read_cv_table

    runs_root = tmp_path / "runs"
    for repeat in range(2):
        for fold in range(3):
            for depth in (4, 5, 6):
                _write_cv_manifest(
                    runs_root,
                    "exp",
                    repeat=repeat,
                    fold=fold,
                    method="a",
                    depth=depth,
                    mae=float(depth),
                )
            _write_cv_manifest(
                runs_root,
                "exp",
                repeat=repeat,
                fold=fold,
                method="b",
                depth=7,
                mae=1.0,
            )

    methods, table = read_cv_table(
        runs_root, ["exp"], depth_by_method={"a": 5, "b": 7}, metric="mae"
    )
    assert methods == ["a", "b"]
    np.testing.assert_allclose(table[:, 0], 5.0)
    np.testing.assert_allclose(table[:, 1], 1.0)


def test_read_cv_table_raises_on_mismatched_samples(tmp_path):
    from experiments.compare import read_cv_table

    runs_root = tmp_path / "runs"
    for fold in range(3):
        _write_cv_manifest(
            runs_root, "exp", repeat=0, fold=fold, method="a", depth=5, mae=1.0
        )
    for fold in range(2):  # method b is missing fold 2 -- unpaired
        _write_cv_manifest(
            runs_root, "exp", repeat=0, fold=fold, method="b", depth=5, mae=1.0
        )

    with pytest.raises(ValueError, match="do not share the same"):
        read_cv_table(runs_root, ["exp"], metric="mae")


def test_read_cv_table_requires_at_least_two_methods(tmp_path):
    from experiments.compare import read_cv_table

    runs_root = tmp_path / "runs"
    _write_cv_manifest(runs_root, "exp", repeat=0, fold=0, method="a", depth=5, mae=1.0)

    with pytest.raises(ValueError, match=">= 2 methods"):
        read_cv_table(runs_root, ["exp"], metric="mae")


def test_repeated_measures_anova_matches_known_textbook_values():
    """A hand-checkable case: two methods, a constant per-sample gap and a
    small amount of sample-to-sample noise shared by both -- SS_method
    should equal n * (mean diff / 2)^2 * 2 exactly (method effect only,
    subject variation removed by the repeated-measures design)."""
    from experiments.compare import repeated_measures_anova

    rng = np.random.default_rng(0)
    n = 30
    subject_noise = rng.normal(scale=1.0, size=n)
    table = np.stack([subject_noise, subject_noise + 2.0], axis=1)

    result = repeated_measures_anova(["a", "b"], table)

    assert result.method_means["b"] - result.method_means["a"] == pytest.approx(2.0)
    # No subject-by-method interaction possible with a fixed additive
    # shift -- SS_error must be ~0 up to floating point, so F is huge and
    # p is ~0.
    assert result.ms_error == pytest.approx(0.0, abs=1e-10)
    assert result.p_value < 1e-6


def test_repeated_measures_anova_null_case_gives_a_large_p_value():
    """No true difference between methods -> a small, noisy method effect
    against real sample-to-sample noise should not look significant."""
    from experiments.compare import repeated_measures_anova

    rng = np.random.default_rng(42)
    n = 20
    subject = rng.normal(size=n)
    table = np.stack([subject, subject + rng.normal(scale=1.0, size=n)], axis=1)

    result = repeated_measures_anova(["a", "b"], table)
    assert result.p_value > 0.05


def test_repeated_measures_anova_fully_degenerate_input_gives_nan():
    """Identical columns with literally zero variation make MS_method and
    MS_error both exactly 0 -- the ratio is undefined (0/0), not "no
    effect": reported as NaN rather than a value that looks like a
    confident null result."""
    from experiments.compare import repeated_measures_anova

    values = np.arange(10, dtype=np.float64)
    table = np.stack([values, values], axis=1)  # identical columns

    result = repeated_measures_anova(["a", "b"], table)
    assert math.isnan(result.f_stat)
    assert math.isnan(result.p_value)


def test_tukey_hsd_two_methods_matches_anova_p_value():
    """With exactly two methods, Tukey HSD's own pairwise p-value must
    equal the ANOVA's own p-value (both are testing the same single
    contrast; F = t^2 = (q/sqrt(2))^2 relationship for k=2)."""
    from experiments.compare import repeated_measures_anova, tukey_hsd

    rng = np.random.default_rng(1)
    n = 20
    subject = rng.normal(size=n)
    table = np.stack([subject, subject + rng.normal(scale=0.3, size=n) + 1.0], axis=1)

    anova = repeated_measures_anova(["a", "b"], table)
    comparisons = tukey_hsd(["a", "b"], table, anova=anova)

    assert len(comparisons) == 1
    assert comparisons[0].p_value == pytest.approx(anova.p_value, rel=1e-6)


def test_tukey_hsd_confidence_interval_contains_the_true_diff_typically():
    """A generous sanity check, not a formal coverage proof: with a large
    true effect and low noise, the 95% CI should not straddle zero."""
    from experiments.compare import tukey_hsd

    rng = np.random.default_rng(2)
    n = 40
    subject = rng.normal(size=n)
    table = np.stack([subject, subject + 5.0 + rng.normal(scale=0.1, size=n)], axis=1)

    comparisons = tukey_hsd(["a", "b"], table)
    c = comparisons[0]
    assert c.ci_lo < c.ci_hi
    assert c.diff < 0  # a - b, and b is much larger
    assert c.ci_hi < 0  # CI does not straddle zero
    assert c.p_value < 0.001


def test_tukey_hsd_all_pairs_for_three_methods():
    from experiments.compare import tukey_hsd

    rng = np.random.default_rng(3)
    n = 15
    subject = rng.normal(size=n)
    table = np.stack([subject, subject + 1.0, subject + 2.0], axis=1)

    comparisons = tukey_hsd(["a", "b", "c"], table)
    pairs = {(c.a, c.b) for c in comparisons}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_write_tukey_plot_writes_a_file(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.compare import PairwiseComparison, write_tukey_plot

    comparisons = [
        PairwiseComparison(
            a="a", b="b", diff=-1.0, ci_lo=-1.5, ci_hi=-0.5, q_stat=3.0, p_value=0.01
        )
    ]
    out = tmp_path / "tukey.png"
    write_tukey_plot(comparisons, out, provenance="store=test git=deadbeef n=1")
    assert out.exists()
    assert out.stat().st_size > 0
