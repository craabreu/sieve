"""Efficient DASH-tree depth sweep: one fit + one tree-matching walk per
fold, at the deepest depth requested, with every shallower depth's own
metrics derived by truncating the already-walked paths
(``predictors.dash.predict_raw_from_paths``) instead of re-walking from
scratch. See that function's own docstring for why this is exactly
equivalent to a fresh ``max_depth=k`` walk, not an approximation --
**for k >= 2**. Depth 1 is a genuine, verified exception (see
``_MIN_DERIVABLE_DEPTH`` below) and is never derived by truncation.

Ordinary per-depth ``run`` invocations are still exactly right for a
single depth, or for any other predictor -- this module exists only
because ``match_new_atom`` is expensive enough (~7 minutes/fold at
depth 8, measured directly against a full fold) that redoing it once per
depth in a 9-point sweep would waste ~9x the necessary compute, unlike
sieve's own sweep (cheap enough that N separate runs need no code at all
-- see docs/superpowers/specs/2026-09-02-charge-sweep-and-loo-design.md).

Deliberately narrower than ``runner.execute``: no LOO (``dash`` doesn't
implement ``predict_loo_raw``), no MLflow tracking (the whole point of
this sweep is untracked), no ``tree_stats_load_path`` (there is exactly
one fit per fold here). ``save_tree_stats`` *is* honored, but only for
the one fit that actually happens: the fold's shared fit at
``max(derived_depths)``, saved once into that deepest depth's own run
directory. The depth-1 real sub-run is forced *not* to save its own --
that shallow tree is neither reusable nor mergeable with the deep ones
(``tree_artifact.merge_node_stats`` needs every shard at the same
depth). ``normalization`` is honored the same way ``runner._normalize``
applies it. Every other artifact a run directory carries --
``config.resolved.yaml``, ``metrics.json``, ``manifest.json``,
``predictions.npz``, the parity plot -- is written the same way, via the
same runner helpers, so ``summarize``/``sweep`` read a depth-sweep run
exactly like an ordinary one.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from experiments.config import ExperimentCfg, load_config, to_dict
from experiments.normalize import NORMALIZERS
from experiments.predictors.base import Prediction, RawPrediction
from experiments.predictors.dash import DASHChargePredictor
from experiments.runner import (
    DEFAULT_RUNS_ROOT,
    RunResult,
    _git_info,
    _package_versions,
    _run_name,
    _savez_run,
    _score,
    _write_plots,
    load_molecule_set,
)
from experiments.runner import run as runner_run

logger = logging.getLogger("experiments")

# DASH-tree's own match_new_atom (_get_init_layer, dash_tree.py) redirects a
# hydrogen atom to its heavy neighbor and pre-consumes one depth unit before
# the caller's own max_depth is even checked -- confirmed by reading that
# function directly, and by a real, reproducible discrepancy: an
# independent max_depth=1 run and a truncate-to-1-entry derived prediction
# disagreed (mae 0.0937 vs 0.1315 on the same 300-conformer slice), while
# every depth >= 2 matched the independent run bit-for-bit. The mechanism:
# for an H atom, requested max_depth=1 and max_depth=2 both resolve to the
# *same* 2-entry path (the special case triggers "max_depth <= 1" only
# *after* decrementing once for the redirect), so the true depth-1 path is
# one entry longer than truncation alone would ever produce -- and nothing
# in the path itself says which entries came from that redirect, so this
# cannot be fixed by adjusting the truncation length inside
# predict_raw_from_paths. Depth 1 is therefore never derived here; see
# run_fold, which routes it through the ordinary, already-correct
# ``runner.run`` instead (cheap regardless -- max_depth=1 is the fastest
# case to walk for real).
_MIN_DERIVABLE_DEPTH = 2


def _cfg_for(
    config_path: str | Path, *, store: str, depth: int, experiment: str, batch_id: str
) -> ExperimentCfg:
    return load_config(
        config_path,
        overrides=[
            f"data.store={store}",
            f"predictor.params.max_depth={depth}",
            f"run.experiment={experiment}",
            f"run.batch_id={batch_id}",
        ],
    )


def _batch_id(depth: int, label: str) -> str:
    return f"d{depth}-{label}"


def _fold_label(fold: int) -> str:
    return f"f{fold}"


def sweep_done(runs_root: Path, experiment: str, depths: list[int], label: str) -> bool:
    """True when every depth's own run directory for this sweep already has
    a ``metrics.json``. The unit of idempotency is the whole sweep over one
    store, not one depth within it: fit + walk is shared work across every
    depth, so a partially-written sweep isn't meaningfully resumable at a
    finer grain than "redo the store"."""
    for depth in depths:
        matches = list(
            (runs_root / experiment).glob(f"{_batch_id(depth, label)}__*/metrics.json")
        )
        if not matches:
            return False
    return True


def fold_done(runs_root: Path, experiment: str, depths: list[int], fold: int) -> bool:
    """``sweep_done`` for one fold of a partitioned store."""
    return sweep_done(runs_root, experiment, depths, _fold_label(fold))


def _raw_to_prediction(
    raw: RawPrediction, mset, *, normalization: str | None
) -> Prediction:
    """Mirrors ``runner._predict``'s own normalize-or-not branch, applied
    to an already-computed ``RawPrediction`` instead of calling
    ``predictor.predict_raw`` itself."""
    if normalization is None:
        return Prediction(atom_value=raw.atom_value)
    assert mset.molecule_value is not None
    atom_value = NORMALIZERS[normalization](
        raw.atom_value,
        raw.atom_std,
        mset.molecule_value,
        mset.atom_mol_id,
        mset.n_conformers,
    )
    return Prediction(atom_value=atom_value)


def _write_depth_run(
    *,
    cfg: ExperimentCfg,
    predictor: DASHChargePredictor,
    train,
    val,
    test,
    train_paths,
    val_paths,
    test_paths,
    depth: int,
    fit_s: float,
    walk_s: float,
    data_s: float,
    git_info: dict,
    runs_root: Path,
    save_tree_stats: bool = False,
) -> RunResult:
    started = datetime.now(UTC)

    def _predict_at(mset, paths):
        raw = predictor.predict_raw_at_depth(paths, max_depth=depth)
        return _raw_to_prediction(raw, mset, normalization=cfg.normalization)

    t0 = time.perf_counter()
    test_pred = _predict_at(test, test_paths)
    predict_s = time.perf_counter() - t0

    run_metrics = _score(test, test_pred)
    if train.n_conformers:
        train_pred = _predict_at(train, train_paths)
        run_metrics.update(
            {f"train/{k}": v for k, v in _score(train, train_pred).items()}
        )
    if val.n_conformers:
        val_pred = _predict_at(val, val_paths)
        run_metrics.update({f"val/{k}": v for k, v in _score(val, val_pred).items()})

    run_metrics["time/fit_s"] = fit_s
    run_metrics["time/walk_s"] = walk_s
    run_metrics["time/predict_s"] = predict_s
    run_metrics["time/data_s"] = data_s

    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / cfg.run.experiment / f"{_run_name(cfg)}__{stamp}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "run_name": _run_name(cfg),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": {
            "fit": fit_s,
            "walk": walk_s,
            "predict": predict_s,
            "data": data_s,
        },
        "tree_stats_source": "fit",
        "git": git_info,
        "seed": cfg.run.seed,
        "packages": _package_versions(),
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "n_train_conformers": train.n_conformers,
            "n_val_conformers": val.n_conformers,
            "n_test_conformers": test.n_conformers,
            "n_train_atoms": train.n_atoms,
            "n_test_atoms": test.n_atoms,
        },
        "config": to_dict(cfg),
        "match_stats": predictor.match_stats,
    }

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(to_dict(cfg)))
    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    _savez_run(run_dir / "predictions.npz", test, test_pred)
    _write_plots(run_dir, test, test_pred, run_metrics, cfg)
    if save_tree_stats:
        # The fold's one shared fit, at this (deepest) depth -- the run
        # directory that produced it is its own provenance record, exactly
        # as in runner._execute_inner. Depth-invariant node stats: a shard
        # reusable via tree_stats_load_path at any depth, and mergeable
        # across folds (tree_artifact.merge_node_stats) since every fold
        # saves at this same depth.
        predictor.save_model_state(run_dir / "tree_stats.npz")

    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


def run_fold(
    *,
    config_path: str | Path,
    store: str,
    depths: list[int],
    experiment: str,
    fold: int,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    limit: int | None = None,
) -> list[RunResult]:
    """``run_store`` for one fold of a partitioned store, labelled ``f<n>``."""
    return run_store(
        config_path=config_path,
        store=store,
        depths=depths,
        experiment=experiment,
        label=_fold_label(fold),
        runs_root=runs_root,
        allow_dirty=allow_dirty,
        limit=limit,
    )


def run_store(
    *,
    config_path: str | Path,
    store: str,
    depths: list[int],
    experiment: str,
    label: str,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    limit: int | None = None,
) -> list[RunResult]:
    """One store's worth of the depth sweep, tagged ``d<depth>-<label>``.

    ``store`` is any store -- one fold of a partition (via ``run_fold``,
    label ``f<n>``) or the whole corpus (label ``full``). The saving is
    what makes this worth using on the full corpus too: one fit + one
    walk at ``max(depths)`` serves every derivable depth, against one fit
    + one walk *per depth* if the same sweep were run as independent
    ``run`` invocations.

    Depths ``>= _MIN_DERIVABLE_DEPTH`` share one fit + one tree-matching
    walk (at ``max(depths)``), with each depth's own metrics derived by
    truncating the already-walked paths -- see the module docstring for
    why depth 1 is excluded from this and, when present in ``depths``, is
    instead run for real via the ordinary ``runner.run`` (cheap regardless
    -- max_depth=1 is the fastest case to walk).

    When the config sets ``save_tree_stats``, the fold's one shared fit
    is saved once, as ``tree_stats.npz`` in the deepest derived depth's
    own run directory -- see the module docstring.

    Skips entirely (returns ``[]``) when ``fold_done`` already holds for
    this fold. ``limit`` mirrors ``run``'s own ``--limit`` (a literal
    row-prefix slice of the store), for a quick, cheap sanity check
    against a real store rather than a full, ~7-minute fold."""
    from experiments.data import REPO_ROOT

    if sweep_done(runs_root, experiment, depths, label):
        logger.info(
            "%r of %r already done for every depth; skipping", label, experiment
        )
        return []

    derived_depths = [d for d in depths if d >= _MIN_DERIVABLE_DEPTH]
    real_depths = [d for d in depths if d < _MIN_DERIVABLE_DEPTH]

    results: list[RunResult] = []
    for depth in real_depths:
        # Forced save_tree_stats=False regardless of config: a max_depth<2
        # fit's node stats are a small subset, neither worth reusing nor
        # mergeable with the deep shards -- see the module docstring.
        cfg = replace(
            _cfg_for(
                config_path,
                store=store,
                depth=depth,
                experiment=experiment,
                batch_id=_batch_id(depth, label),
            ),
            save_tree_stats=False,
        )
        results.append(
            runner_run(cfg, runs_root=runs_root, allow_dirty=allow_dirty, limit=limit)
        )

    if not derived_depths:
        return results

    max_depth = max(derived_depths)
    base_cfg = _cfg_for(
        config_path,
        store=store,
        depth=max_depth,
        experiment=experiment,
        batch_id=_batch_id(max_depth, label),
    )

    git_info = _git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
        )

    t0 = time.perf_counter()
    mset, masks = load_molecule_set(
        store,
        target=base_cfg.target,
        split_column=base_cfg.data.split_column,
        splits=(
            base_cfg.data.train_split,
            base_cfg.data.val_split,
            base_cfg.data.eval_split,
        ),
        limit=limit,
    )
    data_s = time.perf_counter() - t0

    train = mset.select(masks[base_cfg.data.train_split])
    val = mset.select(masks[base_cfg.data.val_split])
    test = mset.select(masks[base_cfg.data.eval_split])

    params = {k: v for k, v in base_cfg.predictor.params.items() if k != "max_depth"}
    predictor = DASHChargePredictor(max_depth=max_depth, **params)

    t0 = time.perf_counter()
    predictor.fit(train, val, rng=np.random.default_rng(base_cfg.run.seed))
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_paths = predictor.match_paths(train, split="train")
    val_paths = predictor.match_paths(val, split="val") if val.n_conformers else []
    test_paths = predictor.match_paths(test, split="test")
    walk_s = time.perf_counter() - t0

    for depth in derived_depths:
        cfg = _cfg_for(
            config_path,
            store=store,
            depth=depth,
            experiment=experiment,
            batch_id=_batch_id(depth, label),
        )
        results.append(
            _write_depth_run(
                cfg=cfg,
                predictor=predictor,
                train=train,
                val=val,
                test=test,
                train_paths=train_paths,
                val_paths=val_paths,
                test_paths=test_paths,
                depth=depth,
                fit_s=fit_s,
                walk_s=walk_s,
                data_s=data_s,
                git_info=git_info,
                runs_root=runs_root,
                save_tree_stats=base_cfg.save_tree_stats and depth == max_depth,
            )
        )
    return results


def run_sweep(
    *,
    config_path: str | Path,
    store_prefix: str,
    n_folds: int,
    depths: list[int],
    experiment: str,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    limit: int | None = None,
) -> list[RunResult]:
    """``run_fold`` for every fold ``1..n_folds`` of ``<store_prefix>-N``."""
    results: list[RunResult] = []
    for fold in range(1, n_folds + 1):
        results.extend(
            run_fold(
                config_path=config_path,
                store=f"{store_prefix}-{fold}",
                depths=depths,
                experiment=experiment,
                fold=fold,
                runs_root=runs_root,
                allow_dirty=allow_dirty,
                limit=limit,
            )
        )
    return results


def _fold_shard_path(
    runs_root: Path, from_experiment: str, depth: int, fold: int
) -> Path:
    """The ``tree_stats.npz`` a depth sweep saved for one fold at ``depth``
    -- newest match if that fold was ever re-run. Raises if absent."""
    matches = sorted(
        (runs_root / from_experiment).glob(
            f"{_batch_id(depth, _fold_label(fold))}__*/tree_stats.npz"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"no tree_stats.npz for fold {fold} at depth {depth} under "
            f"{runs_root / from_experiment} -- run the sweep with "
            "save_tree_stats set first"
        )
    return matches[-1]


def merge_fold_shards(
    *,
    from_experiment: str,
    depth: int,
    n_folds: int,
    out_path: str | Path,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> Path:
    """Merge a depth sweep's per-fold ``tree_stats.npz`` shards (all at
    ``depth``) into one node-stats artifact at ``out_path``, via
    ``tree_artifact.fold_node_stats`` -- exact, no re-fit. A
    ``partition-store`` split partitions the corpus by molecule and each
    shard was fit train-only, so the merged result is exactly what one
    fit on the union of every fold's train split would produce.

    Idempotent: returns ``out_path`` unchanged if it already exists."""
    from experiments.tree_artifact import (
        fold_node_stats,
        load_node_stats,
        save_node_stats,
    )

    out_path = Path(out_path)
    if out_path.exists():
        logger.info("merged shard %s already exists; skipping", out_path)
        return out_path

    shard_paths = [
        _fold_shard_path(runs_root, from_experiment, depth, fold)
        for fold in range(1, n_folds + 1)
    ]
    merged = fold_node_stats(load_node_stats(p) for p in shard_paths)
    save_node_stats(merged, out_path)
    logger.info("merged %d shards -> %s", len(shard_paths), out_path)
    return out_path
