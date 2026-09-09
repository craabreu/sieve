"""End-to-end tests for the efficient DASH depth sweep, against the real
pinned DASH-tree clone and the real, already-partitioned
dash-molecules-10fold-1 store. Skipped if either is absent.

Every test uses ``--limit``-equivalent slicing (via ``run_fold``'s own
``limit``) and writes to ``tmp_path``, never the real ``experiments/runs/``
tree -- these are correctness/idempotency checks, not real sweep runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.data import DEFAULT_STORES_ROOT

_DASH_TREE_ROOT = Path(__file__).resolve().parents[1] / "external" / "DASH-tree"
_STORE_DIR = DEFAULT_STORES_ROOT / "dash-molecules-10fold-1"
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "dash-charge-example.yaml"
)

pytestmark = pytest.mark.skipif(
    not _DASH_TREE_ROOT.exists() or not (_STORE_DIR / "molecules.parquet").exists(),
    reason="experiments/external/DASH-tree not cloned, or "
    "dash-molecules-10fold-1 not prepared locally",
)


def _baseline_mae(tmp_path, *, depth: int, limit: int) -> float:
    """An ordinary, independent run() at a single depth -- the ground
    truth every derived depth's own metrics must match exactly."""
    from experiments.config import load_config
    from experiments.runner import run

    cfg = load_config(
        _CONFIG_PATH,
        overrides=[
            "data.store=dash-molecules-10fold-1",
            f"predictor.params.max_depth={depth}",
            "run.experiment=baseline",
        ],
    )
    result = run(cfg, runs_root=tmp_path, allow_dirty=True, limit=limit)
    return result.metrics["mae"]


def test_derived_depths_match_independent_runs_exactly(tmp_path):
    """The whole point of dash_depth_sweep: a depth derived by truncating
    an already-walked path must be bit-for-bit identical to an
    independent run at that same depth -- not merely close."""
    from experiments.dash_depth_sweep import run_fold

    limit = 300
    baseline = {
        d: _baseline_mae(tmp_path / "baseline", depth=d, limit=limit) for d in (1, 2, 4)
    }

    results = run_fold(
        config_path=_CONFIG_PATH,
        store="dash-molecules-10fold-1",
        depths=[1, 2, 4],
        experiment="dash-depth-sweep-test",
        fold=1,
        runs_root=tmp_path / "sweep",
        allow_dirty=True,
        limit=limit,
    )

    by_depth = {}
    for result in results:
        depth = result.manifest["config"]["predictor"]["params"]["max_depth"]
        by_depth[depth] = result.metrics["mae"]

    assert by_depth.keys() == baseline.keys()
    for depth, expected in baseline.items():
        assert by_depth[depth] == pytest.approx(expected, abs=0.0), (
            f"depth {depth}: derived mae {by_depth[depth]} != "
            f"independent mae {expected}"
        )


def test_depth_one_specifically_matches_despite_the_h_atom_redirect(tmp_path):
    """Regression test for the exact bug found in development: DASH-tree's
    own match_new_atom redirects a hydrogen atom to its heavy neighbor and
    pre-consumes one depth unit, making a *derived* (truncated) depth-1
    path one entry short. dash_depth_sweep must route depth 1 through a
    real run instead of truncation -- this fails loudly (a mismatched mae)
    if that routing regresses."""
    from experiments.dash_depth_sweep import run_fold

    limit = 300
    expected = _baseline_mae(tmp_path / "baseline", depth=1, limit=limit)

    results = run_fold(
        config_path=_CONFIG_PATH,
        store="dash-molecules-10fold-1",
        depths=[1],
        experiment="dash-depth-sweep-test",
        fold=1,
        runs_root=tmp_path / "sweep",
        allow_dirty=True,
        limit=limit,
    )

    assert len(results) == 1
    assert results[0].metrics["mae"] == pytest.approx(expected, abs=0.0)


def test_run_fold_is_idempotent(tmp_path):
    from experiments.dash_depth_sweep import run_fold

    kwargs = {
        "config_path": _CONFIG_PATH,
        "store": "dash-molecules-10fold-1",
        "depths": [1, 2],
        "experiment": "dash-depth-sweep-test",
        "fold": 1,
        "runs_root": tmp_path / "sweep",
        "allow_dirty": True,
        "limit": 200,
    }

    first = run_fold(**kwargs)
    assert len(first) == 2

    second = run_fold(**kwargs)
    assert second == []
