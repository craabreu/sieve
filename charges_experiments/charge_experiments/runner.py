"""Run one charges experiment: config + data in, a run directory (and
optionally an MLflow record) out.

``execute`` is the core, testable pipeline (an already-built ``MoleculeSet``
and split masks in, so the smoke test never touches the real store).
``run`` is the real entry point: loads the store via
``data.load_molecule_set`` then calls ``execute``. Mirrors
cosmo_experiments/sieve_experiments/runner.py's shape, much smaller: no
plots, no profile/area rollup, no train/val parity-plot bookkeeping.
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

from charge_experiments import metrics as metrics_mod
from charge_experiments.config import ExperimentCfg, to_dict, to_flat_params
from charge_experiments.data import REPO_ROOT, MoleculeSet
from charge_experiments.predictors import build
from charge_experiments.predictors.base import Prediction

DEFAULT_TRACKING_URI = f"file:{REPO_ROOT / 'charges_experiments' / 'mlruns'}"
DEFAULT_RUNS_ROOT = REPO_ROOT / "charges_experiments" / "runs"

logger = logging.getLogger("charge_experiments")


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
    package_names = (
        "sieve", "numpy", "scipy", "pyyaml", "rdkit", "pandas", "pyarrow", "mlflow",
    )
    for name in package_names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _run_name(cfg: ExperimentCfg) -> str:
    d = cfg.data
    return f"{cfg.predictor.name}-{d.split_column}-s{cfg.run.seed}"


def _savez_run(path: Path, test: MoleculeSet, pred: Prediction, /) -> None:
    np.savez(
        path,
        chembl_id=np.array(test.chembl_id),
        conf_id=np.array(test.conf_id),
        num_atoms=test.num_atoms,
        net_charge=test.net_charge,
        atom_charge_true=test.atom_charge,
        atom_charge_pred=pred.atom_charge,
    )


def _score(test: MoleculeSet, pred: Prediction) -> dict[str, float]:
    out = metrics_mod.regression_metrics(test.atom_charge, pred.atom_charge)
    out["n_test_atoms"] = float(test.n_atoms)
    out["n_test_conformers"] = float(test.n_conformers)
    conservation = metrics_mod.charge_conservation_metrics(
        pred.atom_charge, test.atom_mol_id, test.net_charge, test.n_conformers
    )
    out.update({f"charge_conservation/{k}": v for k, v in conservation.items()})
    return out


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
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
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


def _score_extra_split(
    predictor: Any, mset: MoleculeSet, *, split: str
) -> dict[str, float]:
    """Predict + score train/val the same way test is scored, mirroring
    cosmo_experiments' own train/val-alongside-test convention. Empty split
    -> no keys, not a crash."""
    if mset.n_conformers == 0:
        return {}
    pred = predictor.predict(mset)
    score = _score(mset, pred)
    return {f"{split}/{k}": v for k, v in score.items()}


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
    train_metrics = _score_extra_split(predictor, train, split="train")
    train_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_metrics = _score_extra_split(predictor, val, split="val")
    val_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = predictor.predict(test)
    predict_s = time.perf_counter() - t0

    run_metrics = _score(test, pred)
    run_metrics.update(train_metrics)
    if train_metrics:
        run_metrics["time/train_predict_s"] = train_predict_s
    run_metrics.update(val_metrics)
    if val_metrics:
        run_metrics["time/val_predict_s"] = val_predict_s
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
            "split_column": cfg.data.split_column,
            "n_train_conformers": train.n_conformers,
            "n_val_conformers": val.n_conformers,
            "n_test_conformers": test.n_conformers,
            "n_train_atoms": train.n_atoms,
            "n_test_atoms": test.n_atoms,
        },
        "config": to_dict(cfg),
    }
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

    if tracking is not None:
        _log_mlflow(cfg, run_metrics, run_dir, tracking)

    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


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


def load_molecule_set(
    store_name: str,
    *,
    split_column: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    stores_root: Path | None = None,
) -> tuple[MoleculeSet, dict[str, NDArray[np.bool_]]]:
    """Load ``molecules.parquet`` for ``store_name`` into a ``MoleculeSet``
    plus split masks."""
    import pandas as pd

    from charge_experiments.data import DEFAULT_STORES_ROOT, blob_to_mol

    root = stores_root if stores_root is not None else DEFAULT_STORES_ROOT
    store_dir = root / store_name
    df = pd.read_parquet(store_dir / "molecules.parquet")
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    mols = [blob_to_mol(b) for b in df["mol"]]
    mset = MoleculeSet(
        chembl_id=list(df["chembl_id"]),
        conf_id=list(df["conf_id"]),
        mols=mols,
        net_charge=df["net_charge"].to_numpy(dtype=np.float64),
        split=list(df[split_column]),
    )
    masks = {name: (df[split_column] == name).to_numpy() for name in splits}
    return mset, masks


def run(
    cfg: ExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = DEFAULT_TRACKING_URI,
    limit: int | None = None,
) -> RunResult:
    """Load the real store, then run the pipeline (see ``execute``)."""
    t0 = time.perf_counter()
    mset, masks = load_molecule_set(
        cfg.data.store,
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
