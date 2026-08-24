"""Run one experiment: config + data in, a run directory (and optionally an
MLflow record) out.

``execute`` is the core, testable pipeline: it takes an already-built
``MoleculeSet`` and split masks, so the fast smoke test can drive it with a
synthetic set and never import cosmolayer. ``run`` is the real entry point:
it loads the store via ``data.load_molecule_set`` and then calls ``execute``.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from sieve_experiments import metrics as metrics_mod
from sieve_experiments import plots
from sieve_experiments.config import ExperimentCfg, to_dict, to_flat_params
from sieve_experiments.data import REPO_ROOT, MoleculeSet, load_atom_truth, molecule_sum
from sieve_experiments.predictors import build
from sieve_experiments.predictors.base import Prediction

DEFAULT_TRACKING_URI = f"file:{REPO_ROOT / 'experiments' / 'mlruns'}"
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"

logger = logging.getLogger("sieve_experiments")


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, float]
    manifest: dict[str, Any]


def _git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": dirty,
        "describe": _run(["git", "describe", "--always", "--dirty"]),
    }


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in (
        "sieve",
        "numpy",
        "scipy",
        "pyyaml",
        "rdkit",
        "cosmolayer",
        "mlflow",
        "pandas",
        "matplotlib",
    ):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _run_name(cfg: ExperimentCfg) -> str:
    d = cfg.data
    return f"{cfg.predictor.name}-{d.split_column}-{d.scheme}-s{cfg.run.seed}"


def _savez_run(path: Path, test: MoleculeSet, pred: Prediction, /) -> None:
    payload: dict[str, np.ndarray] = {
        "smiles": np.array(test.smiles),
        "num_atoms": test.num_atoms,
        "net_charge": test.net_charge,
        "mol_profile_pred": pred.mol_profile,
    }
    if test.mol_profile is not None:
        payload["mol_profile_true"] = test.mol_profile
    if test.mol_area is not None:
        payload["mol_area_true"] = test.mol_area
    if pred.mol_area is not None:
        payload["mol_area_pred"] = pred.mol_area
    if pred.mol_charge_raw is not None:
        payload["mol_charge_raw_pred"] = pred.mol_charge_raw
    np.savez(path, **payload)


def execute(
    cfg: ExperimentCfg,
    mset: MoleculeSet,
    masks: dict[str, NDArray[np.bool_]],
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    data_seconds: float = 0.0,
) -> RunResult:
    """Run the pipeline against an already-loaded ``mset``/``masks``."""
    git_info = _git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True "
            "(a run's reproducibility depends on the code that produced it being "
            "recorded exactly)"
        )

    random.seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)

    train = mset.select(masks[cfg.data.train_split])
    val = mset.select(masks[cfg.data.val_split])
    test = mset.select(masks[cfg.data.eval_split])

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / cfg.run.experiment / f"{_run_name(cfg)}__{stamp}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    file_handler = logging.FileHandler(run_dir / "stdout.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    try:
        return _execute_inner(
            cfg,
            train,
            val,
            test,
            rng=rng,
            run_dir=run_dir,
            started=started,
            git_info=git_info,
            tracking=tracking,
            data_seconds=data_seconds,
        )
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def _compute_atom_metrics(
    cfg: ExperimentCfg, test: MoleculeSet, pred: Prediction
) -> dict[str, float] | None:
    """Atom-level accuracy metrics, merged into ``run_metrics`` under an
    ``atom/`` key prefix, for a predictor that supplies atom-level output
    (an ``AtomPredictor``: DASH, later Sieve). ``molecule_metrics`` is
    granularity-agnostic (it just operates on rows), so it's reused
    directly here rather than duplicated for atoms. Returns its keys
    unprefixed -- the caller adds ``atom/`` when merging, so this stays
    reusable independent of that naming choice.

    ``load_molecule_set`` never populates atom-level truth for the test
    split (see data.py's module docstring), so it's loaded here the same
    way DASH's own ``fit_atoms`` already loads it for train. Returns
    ``None`` (and logs a warning) rather than raising if that load fails --
    a predictor's atom_profile output is a metrics-only concern, not a
    reason to fail the whole run.
    """
    try:
        atom_profile_true, atom_area_true, atom_charge_true = load_atom_truth(
            cfg.data.store,
            scheme=cfg.data.scheme,
            smiles=test.smiles,
            num_atoms=test.num_atoms,
        )
    except Exception:
        logger.warning(
            "could not load atom-level truth for atom metrics; skipping",
            exc_info=True,
        )
        return None

    return metrics_mod.molecule_metrics(
        profile_true=atom_profile_true,
        profile_pred=pred.atom_profile,
        area_true=atom_area_true if pred.atom_area is not None else None,
        area_pred=pred.atom_area,
        charge_true=atom_charge_true if pred.atom_charge is not None else None,
        charge_pred=pred.atom_charge,
    )


def _execute_inner(
    cfg: ExperimentCfg,
    train: MoleculeSet,
    val: MoleculeSet,
    test: MoleculeSet,
    *,
    rng: np.random.Generator,
    run_dir: Path,
    started: datetime,
    git_info: dict[str, Any],
    tracking: str | None,
    data_seconds: float,
) -> RunResult:
    predictor = build(cfg.predictor.name, cfg.predictor.params)

    t0 = time.perf_counter()
    predictor.fit(train, val, rng=rng)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = predictor.predict(test)
    predict_s = time.perf_counter() - t0

    if test.mol_profile is None:
        raise ValueError("test split has no mol_profile ground truth")

    area_true = (
        test.mol_area
        if (test.mol_area is not None and pred.mol_area is not None)
        else None
    )
    area_pred = pred.mol_area if area_true is not None else None
    charge_true = test.net_charge if pred.mol_charge_raw is not None else None
    charge_pred = pred.mol_charge_raw

    run_metrics = metrics_mod.molecule_metrics(
        profile_true=test.mol_profile,
        profile_pred=pred.mol_profile,
        area_true=area_true,
        area_pred=area_pred,
        charge_true=charge_true,
        charge_pred=charge_pred,
    )
    # A free correctness signal on any predictor that supplies both
    # per-atom and per-molecule area: they should already agree.
    if pred.atom_area is not None and pred.mol_area is not None:
        atom_sum = molecule_sum(pred.atom_area, test.atom_mol_id, test.n_molecules)
        run_metrics["area/self_consistency_mae"] = float(
            np.mean(np.abs(atom_sum - pred.mol_area))
        )

    if pred.atom_profile is not None:
        atom_metrics = _compute_atom_metrics(cfg, test, pred)
        if atom_metrics is not None:
            # atom/-prefixed, flat: molecule-level keys stay exactly as
            # they are (unprefixed, unchanged -- widely referenced already)
            # and MLflow's own log_metrics needs flat float values anyway,
            # not a nested dict.
            run_metrics.update({f"atom/{k}": v for k, v in atom_metrics.items()})

    run_metrics["time/fit_s"] = fit_s
    run_metrics["time/predict_s"] = predict_s
    run_metrics["time/data_s"] = data_seconds

    manifest = {
        "schema_version": 1,
        "run_name": _run_name(cfg),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": {"fit": fit_s, "predict": predict_s, "data": data_seconds},
        "git": git_info,
        "seed": cfg.run.seed,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "data": {
            "store": cfg.data.store,
            "scheme": cfg.data.scheme,
            "split_column": cfg.data.split_column,
            "n_train_molecules": train.n_molecules,
            "n_val_molecules": val.n_molecules,
            "n_test_molecules": test.n_molecules,
            "n_train_atoms": train.n_atoms,
            "n_test_atoms": test.n_atoms,
            "train_mean_num_atoms": float(np.mean(train.num_atoms)),
            "val_mean_num_atoms": float(np.mean(val.num_atoms))
            if val.n_molecules
            else None,
            "test_mean_num_atoms": float(np.mean(test.num_atoms))
            if test.n_molecules
            else None,
            "grid": {
                "max_abs_sigma": test.grid.max_abs_sigma,
                "num_points": test.grid.num_points,
                "bin_width": test.grid.bin_width,
            },
        },
        "config": to_dict(cfg),
    }

    # Optional, duck-typed: a predictor that can only partially cover a split
    # (DASH cannot match atoms outside its published feature vocabulary, and
    # falls back to a global mean for those) reports the counts here, so how
    # much of a split the model actually covered is always on the record and
    # never has to be inferred from the metrics.
    match_stats = getattr(predictor, "match_stats", None)
    if match_stats:
        manifest["match_stats"] = match_stats

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(to_dict(cfg)))
    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    _savez_run(run_dir / "predictions.npz", test, pred)

    _write_plots(run_dir, test, pred, run_metrics, cfg)

    if tracking is not None:
        _log_mlflow(cfg, run_metrics, run_dir, tracking)

    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


def _write_plots(
    run_dir: Path,
    test: MoleculeSet,
    pred: Prediction,
    run_metrics: dict[str, float],
    cfg: ExperimentCfg,
) -> None:
    if test.mol_profile is None:
        return  # nothing to plot without ground truth; execute() already
        # rejects this case before calling us, this is just for the type
        # checker's benefit.
    if test.n_molecules == 0:
        return  # nothing to plot on an empty eval split (e.g. a small
        # --limit run whose split happens to land entirely in train)
    try:
        plots.parity_hexbin(
            test.mol_profile,
            pred.mol_profile,
            run_dir / "plots" / "parity_molecule.png",
            {
                k.split("/", 1)[1]: v
                for k, v in run_metrics.items()
                if k.startswith("profile/")
            },
            level="molecule",
            title=f"{cfg.predictor.name} ({cfg.data.split_column}, {cfg.data.scheme})",
        )
        profile_index = np.argsort(-pred.mol_profile.sum(axis=1))[:16]
        plots.profile_panel(
            test.grid.values,
            test.mol_profile[profile_index],
            pred.mol_profile[profile_index],
            [test.smiles[i] for i in profile_index],
            run_dir / "plots" / "profile_panel.png",
        )
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots for this run")


def _log_mlflow(
    cfg: ExperimentCfg, run_metrics: dict[str, float], run_dir: Path, tracking: str
) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping tracking for this run")
        return

    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(cfg.run.experiment)
    with mlflow.start_run(run_name=_run_name(cfg)):
        tags = {
            "predictor": cfg.predictor.name,
            "split_column": cfg.data.split_column,
            "scheme": cfg.data.scheme,
            "store": cfg.data.store,
            "seed": str(cfg.run.seed),
            "run_dir": str(run_dir),
            **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
        }
        mlflow.set_tags(tags)
        mlflow.log_params(to_flat_params(cfg))
        clean_metrics = {
            k: v
            for k, v in run_metrics.items()
            if isinstance(v, float) and not np.isnan(v)
        }
        mlflow.log_metrics({f"test/{k}": v for k, v in clean_metrics.items()})
        mlflow.log_artifacts(str(run_dir))


def run(
    cfg: ExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    limit: int | None = None,
) -> RunResult:
    """Load the real store, then run the pipeline (see ``execute``)."""
    from sieve_experiments.data import load_molecule_set

    t0 = time.perf_counter()
    mset, masks = load_molecule_set(
        cfg.data.store,
        scheme=cfg.data.scheme,
        split_column=cfg.data.split_column,
        splits=(cfg.data.train_split, cfg.data.val_split, cfg.data.eval_split),
        limit=limit,
    )
    data_seconds = time.perf_counter() - t0

    return execute(
        cfg,
        mset,
        masks,
        runs_root=runs_root,
        allow_dirty=allow_dirty,
        tracking=tracking,
        data_seconds=data_seconds,
    )
