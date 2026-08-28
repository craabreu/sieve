"""Nested-run orchestration: one predictor's fit()+save+raw-predict as an
MLflow parent run, one child run per normalization scheme in
NestedExperimentCfg.children, each re-normalizing the same already-computed
raw predictions (no re-matching, no re-fit). See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.

Reuses charge_experiments.runner's own private helpers directly (``_score``,
``_build_parity_panels``, ``_savez_run``, ``_git_info``,
``_package_versions``, ``_log_mlflow_run``) -- this codebase already has
precedent for one module importing another's underscore-prefixed helpers
(predictors/dash_pretrained.py imports predictors/dash.py's own
``_atom_paths``), so this continues an established pattern rather than
starting a new one.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from charge_experiments import runner as _runner
from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.nested_config import NestedExperimentCfg
from charge_experiments.normalize import NORMALIZERS
from charge_experiments.predictors import build
from charge_experiments.predictors.base import Prediction, RawPrediction

logger = logging.getLogger("charge_experiments")

DEFAULT_RUNS_ROOT = _runner.DEFAULT_RUNS_ROOT
DEFAULT_TRACKING_URI = _runner.DEFAULT_TRACKING_URI


@dataclass(frozen=True)
class NestedRunResult:
    parent: _runner.RunResult
    children: dict[str, _runner.RunResult]


def _tags_for(
    cfg: NestedExperimentCfg, *, normalization: str, tree_stats_source: str,
    run_dir: Path,
) -> dict[str, str]:
    return {
        "predictor": cfg.predictor.name,
        "split_column": cfg.data.split_column,
        "store": cfg.data.store,
        "seed": str(cfg.run.seed),
        "normalization": normalization,
        "tree_stats_source": tree_stats_source,
        "run_dir": str(run_dir),
        **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
    }


def _params_for(cfg: NestedExperimentCfg) -> dict[str, str]:
    return {
        "run.experiment": cfg.run.experiment,
        "run.seed": str(cfg.run.seed),
        "data.store": cfg.data.store,
        "data.split_column": cfg.data.split_column,
        "predictor.name": cfg.predictor.name,
        **{f"predictor.params.{k}": str(v) for k, v in cfg.predictor.params.items()},
    }


def _write_one_run(
    *,
    run_name: str,
    cfg: NestedExperimentCfg,
    test: MoleculeSet,
    pred: Prediction,
    run_metrics: dict[str, float],
    runs_root: Path,
    started: datetime,
    git_info: dict[str, Any],
    extra_manifest: dict[str, Any],
) -> _runner.RunResult:
    """Write one run directory's full artifact set (metrics/manifest/
    predictions/plots), mirroring runner._execute_inner's own file writing
    -- but for a run whose fit()/predict() timing doesn't fit that
    function's flat shape (a child never calls either)."""
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / cfg.run.experiment / f"{run_name}__{stamp}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "git": git_info,
        "seed": cfg.run.seed,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _runner._package_versions(),
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "n_test_conformers": test.n_conformers,
            "n_test_atoms": test.n_atoms,
        },
        "config": {
            "run": {
                "experiment": cfg.run.experiment, "seed": cfg.run.seed,
                "tags": dict(cfg.run.tags),
            },
            "data": {
                "store": cfg.data.store, "split_column": cfg.data.split_column,
                "train_split": cfg.data.train_split, "val_split": cfg.data.val_split,
                "eval_split": cfg.data.eval_split,
            },
            "predictor": {
                "name": cfg.predictor.name, "params": dict(cfg.predictor.params),
            },
        },
        **extra_manifest,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    _runner._savez_run(run_dir / "predictions.npz", test, pred)
    if test.n_conformers:
        try:
            from charge_experiments import plots

            panels = _runner._build_parity_panels(test, pred, run_metrics)
            plots.parity_panel(
                panels, run_dir / "plots" / "parity_panel.png", suptitle=run_name
            )
        except ImportError:
            logger.warning("matplotlib not installed; skipping plots for %s", run_name)

    return _runner.RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


def execute_nested(
    cfg: NestedExperimentCfg,
    mset: MoleculeSet,
    masks: dict[str, NDArray[np.bool_]],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    data_seconds: float = 0.0,
) -> NestedRunResult:
    git_info = _runner._git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
        )

    random.seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)

    splits = {
        "train": mset.select(masks[cfg.data.train_split]),
        "val": mset.select(masks[cfg.data.val_split]),
        "test": mset.select(masks[cfg.data.eval_split]),
    }

    # build()'s declared return type is the base Predictor protocol; this
    # function needs the wider surface (predict_raw, and conditionally
    # save_model_state/load_model_state) that only some predictors implement
    # -- checked at runtime via hasattr, not statically, so Any here.
    predictor: Any = build(cfg.predictor.name, cfg.predictor.params)

    t0 = time.perf_counter()
    if cfg.tree_stats.load_path is not None:
        if not hasattr(predictor, "load_model_state"):
            raise AttributeError(
                f"predictor {cfg.predictor.name!r} has no load_model_state "
                "method; tree_stats.load_path is not usable with it"
            )
        predictor.load_model_state(cfg.tree_stats.load_path)
        tree_stats_source = "loaded"
    else:
        predictor.fit(splits["train"], splits["val"], rng=rng)
        tree_stats_source = "fit"
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_splits: dict[str, RawPrediction] = {}
    for name, split_mset in splits.items():
        if split_mset.n_conformers == 0:
            raw_splits[name] = RawPrediction(
                atom_charge=np.zeros(0), atom_std=np.zeros(0)
            )
        else:
            raw_splits[name] = predictor.predict_raw(split_mset)
    predict_s = time.perf_counter() - t0

    match_stats = getattr(predictor, "match_stats", None)

    def _run_metrics_for(
        atom_charge_by_split: dict[str, NDArray[np.float64]],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in ("test", "train", "val"):
            split_mset = splits[name]
            if split_mset.n_conformers == 0:
                continue
            score = _runner._score(
                split_mset, Prediction(atom_charge=atom_charge_by_split[name])
            )
            if name == "test":
                out.update(score)
            else:
                out.update({f"{name}/{k}": v for k, v in score.items()})
        out["time/fit_s"] = fit_s
        out["time/predict_s"] = predict_s
        out["time/data_s"] = data_seconds
        return out

    tracking_ok = tracking is not None
    mlflow: Any = None
    if tracking_ok:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_tracking_uri(tracking)
            _runner._ensure_experiment(cfg.run.experiment)
        except ImportError:
            logger.warning("mlflow not installed; skipping tracking for this run")
            tracking_ok = False

    parent_charges = {name: raw.atom_charge for name, raw in raw_splits.items()}
    parent_metrics = _run_metrics_for(parent_charges)
    parent_extra_manifest: dict[str, Any] = {
        "elapsed_s": {"fit": fit_s, "predict": predict_s, "data": data_seconds},
        "tree_stats_source": tree_stats_source,
    }
    if match_stats:
        parent_extra_manifest["match_stats"] = match_stats

    parent_run_name = f"{cfg.predictor.name}-raw"
    started = datetime.now(UTC)
    children_results: dict[str, _runner.RunResult] = {}

    parent_ctx = (
        mlflow.start_run(run_name=parent_run_name) if tracking_ok else nullcontext()
    )
    with parent_ctx:
        parent_result = _write_one_run(
            run_name=parent_run_name, cfg=cfg, test=splits["test"],
            pred=Prediction(atom_charge=parent_charges["test"]),
            run_metrics=parent_metrics, runs_root=runs_root, started=started,
            git_info=git_info, extra_manifest=parent_extra_manifest,
        )
        # Automatic and unconditional whenever fit() actually ran: the
        # saved state lives inside the run it came from, so the run
        # directory itself is the provenance record -- no separately
        # configurable path to silently collide across runs. Skipped for a
        # predictor without save_model_state (e.g. dash_pretrained, whose
        # fit() is a no-op with nothing to save).
        if tree_stats_source == "fit" and hasattr(predictor, "save_model_state"):
            predictor.save_model_state(parent_result.run_dir / "tree_stats.npz")
        if tracking_ok:
            _runner._log_mlflow_run(
                _tags_for(
                    cfg, normalization="raw", tree_stats_source=tree_stats_source,
                    run_dir=parent_result.run_dir,
                ),
                _params_for(cfg), parent_metrics, parent_result.run_dir,
            )

        for name in cfg.children:
            normalize_fn = NORMALIZERS[name]
            child_charges = {
                split_name: (
                    normalize_fn(
                        raw.atom_charge, raw.atom_std, splits[split_name].net_charge,
                        splits[split_name].atom_mol_id, splits[split_name].n_conformers,
                    )
                    if splits[split_name].n_conformers
                    else np.zeros(0)
                )
                for split_name, raw in raw_splits.items()
            }
            child_metrics = _run_metrics_for(child_charges)
            child_run_name = f"{cfg.predictor.name}-{name}"
            child_extra_manifest: dict[str, Any] = {
                "elapsed_s": {"fit": 0.0, "predict": 0.0, "data": 0.0},
                "normalization": name,
                "tree_stats_source": tree_stats_source,
            }
            if match_stats:
                child_extra_manifest["match_stats"] = match_stats

            child_ctx = (
                mlflow.start_run(run_name=child_run_name, nested=True)
                if tracking_ok
                else nullcontext()
            )
            with child_ctx:
                child_result = _write_one_run(
                    run_name=child_run_name, cfg=cfg, test=splits["test"],
                    pred=Prediction(atom_charge=child_charges["test"]),
                    run_metrics=child_metrics, runs_root=runs_root,
                    started=datetime.now(UTC), git_info=git_info,
                    extra_manifest=child_extra_manifest,
                )
                if tracking_ok:
                    child_tags = _tags_for(
                        cfg, normalization=name, tree_stats_source=tree_stats_source,
                        run_dir=child_result.run_dir,
                    )
                    _runner._log_mlflow_run(
                        child_tags, _params_for(cfg), child_metrics,
                        child_result.run_dir,
                    )
                children_results[name] = child_result

    return NestedRunResult(parent=parent_result, children=children_results)


def run_nested(
    cfg: NestedExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    limit: int | None = None,
) -> NestedRunResult:
    """Load the real store, then run the nested pipeline (see
    execute_nested). Mirrors runner.run's own shape."""
    t0 = time.perf_counter()
    mset, masks = _runner.load_molecule_set(
        cfg.data.store,
        split_column=cfg.data.split_column,
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.eval_split),
        limit=limit,
    )
    data_seconds = time.perf_counter() - t0

    return execute_nested(
        cfg, mset, masks, runs_root=runs_root, allow_dirty=allow_dirty,
        tracking=tracking, data_seconds=data_seconds,
    )
