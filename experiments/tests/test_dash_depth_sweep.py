"""Non-optional tests for dash_depth_sweep -- the parts that need neither
the real DASH-tree clone nor a real store (merge_fold_shards is pure
tree_artifact I/O over saved shards)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")


def _write_shard(runs_root, experiment, depth, fold, stats):
    from experiments.tree_artifact import save_node_stats

    d = runs_root / experiment / f"d{depth}-f{fold}__dash-x-s0__20260101T000000Z__abc"
    d.mkdir(parents=True)
    save_node_stats(stats, d / "tree_stats.npz")


def _stats(paths, values):
    from experiments.tree_artifact import compute_node_stats

    return compute_node_stats(paths, np.asarray(values, dtype=np.float64))


def test_merge_fold_shards_equals_fold_node_stats_of_the_shards(tmp_path):
    from experiments.dash_depth_sweep import merge_fold_shards
    from experiments.tree_artifact import fold_node_stats, load_node_stats

    a = _stats([[(0, 1)], [(0, 1), (0, 2)]], [0.2, 0.6])
    b = _stats([[(0, 1)], [(0, 3)]], [0.4, 0.9])
    _write_shard(tmp_path, "sweep", 16, 1, a)
    _write_shard(tmp_path, "sweep", 16, 2, b)

    out = merge_fold_shards(
        from_experiment="sweep",
        depth=16,
        n_folds=2,
        out_path=tmp_path / "merged.npz",
        runs_root=tmp_path,
    )

    merged = load_node_stats(out)
    expected = fold_node_stats([a, b])
    order_m = np.lexsort((merged.node_id, merged.branch_idx))
    order_e = np.lexsort((expected.node_id, expected.branch_idx))
    assert np.array_equal(merged.branch_idx[order_m], expected.branch_idx[order_e])
    assert np.array_equal(merged.node_id[order_m], expected.node_id[order_e])
    assert np.array_equal(merged.count[order_m], expected.count[order_e])
    assert merged.mean[order_m] == pytest.approx(expected.mean[order_e])


def test_merge_fold_shards_is_idempotent(tmp_path):
    from experiments.dash_depth_sweep import merge_fold_shards

    _write_shard(tmp_path, "sweep", 16, 1, _stats([[(0, 1)]], [0.2]))
    _write_shard(tmp_path, "sweep", 16, 2, _stats([[(0, 1)]], [0.4]))

    kwargs = {
        "from_experiment": "sweep",
        "depth": 16,
        "n_folds": 2,
        "out_path": tmp_path / "merged.npz",
        "runs_root": tmp_path,
    }
    first = merge_fold_shards(**kwargs)
    mtime = first.stat().st_mtime_ns
    second = merge_fold_shards(**kwargs)
    assert second == first
    assert second.stat().st_mtime_ns == mtime  # not rewritten


def test_merge_fold_shards_raises_on_a_missing_fold(tmp_path):
    from experiments.dash_depth_sweep import merge_fold_shards

    _write_shard(tmp_path, "sweep", 16, 1, _stats([[(0, 1)]], [0.2]))
    # fold 2's shard deliberately absent

    with pytest.raises(FileNotFoundError, match="fold 2"):
        merge_fold_shards(
            from_experiment="sweep",
            depth=16,
            n_folds=2,
            out_path=tmp_path / "merged.npz",
            runs_root=tmp_path,
        )


def _touch_run(runs_root, experiment, batch_id):
    d = runs_root / experiment / f"{batch_id}__dash-store-s0__stamp__uuid"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text("{}")


def test_batch_id_labels_a_fold_and_an_arbitrary_store_the_same_way():
    from experiments.dash_depth_sweep import _batch_id, _fold_label

    assert _batch_id(16, _fold_label(7)) == "d16-f7"
    assert _batch_id(10, "full") == "d10-full"


def test_sweep_done_is_per_label_and_needs_every_depth(tmp_path):
    """The unit of idempotency is one label's whole sweep: a partially
    written sweep is not done, and two labels do not satisfy each other."""
    from experiments.dash_depth_sweep import sweep_done

    depths = [2, 4]
    assert not sweep_done(tmp_path, "exp", depths, "full")

    _touch_run(tmp_path, "exp", "d2-full")
    assert not sweep_done(tmp_path, "exp", depths, "full")

    _touch_run(tmp_path, "exp", "d4-full")
    assert sweep_done(tmp_path, "exp", depths, "full")
    assert not sweep_done(tmp_path, "exp", depths, "f1")


def test_fold_done_still_delegates_to_the_fold_label(tmp_path):
    from experiments.dash_depth_sweep import fold_done

    depths = [2]
    assert not fold_done(tmp_path, "exp", depths, 3)
    _touch_run(tmp_path, "exp", "d2-f3")
    assert fold_done(tmp_path, "exp", depths, 3)
