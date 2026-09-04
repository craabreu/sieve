"""Run one charges experiment: config + data in, a run directory (and
optionally an MLflow record) out.

``execute`` is the core, testable pipeline (an already-built ``MoleculeSet``
and split masks in, so the smoke test never touches the real store).
``run`` is the real entry point: loads the store via
``data.load_molecule_set`` then calls ``execute``. Mirrors
cosmo_experiments/sieve_experiments/runner.py's shape, still much smaller:
no profile/area rollup, no train/val parity-plot bookkeeping -- plots.py's
own ``parity_panel`` is reused as-is (it's fully domain-agnostic), just
fed a smaller, scalar-charge-shaped set of panels here.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from charge_experiments import metrics as metrics_mod
from charge_experiments import plots
from charge_experiments.config import ExperimentCfg, to_dict, to_flat_params
from charge_experiments.data import REPO_ROOT, MoleculeSet, molecule_sum
from charge_experiments.normalize import NORMALIZERS
from charge_experiments.predictors import build
from charge_experiments.predictors.base import Prediction

# mlflow's plain filesystem tracking backend ("file:...") is in maintenance
# mode as of mlflow 3.x and refuses to open without an explicit env-var
# opt-out -- sqlite is the backend mlflow itself points users toward.
_MLFLOW_RUNS_DB = REPO_ROOT / "charge_experiments" / "mlflow_runs.db"
DEFAULT_TRACKING_URI = f"sqlite:///{_MLFLOW_RUNS_DB}"
# A sqlite/db tracking backend's own default artifact root is "./mlruns"
# relative to the CWD the process happens to run from -- not tied to the
# tracking URI's own location at all, so it silently writes an unignored
# ./mlruns/ wherever the CLI was invoked from unless every experiment is
# created with an explicit artifact_location. See _ensure_experiment.
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "charge_experiments" / "mlflow_artifacts"
DEFAULT_RUNS_ROOT = REPO_ROOT / "charge_experiments" / "runs"

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
        "sieve",
        "numpy",
        "scipy",
        "pyyaml",
        "rdkit",
        "pandas",
        "pyarrow",
        "mlflow",
    )
    for name in package_names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _run_name(cfg: ExperimentCfg) -> str:
    d = cfg.data
    name = f"{cfg.predictor.name}-{d.store}-s{cfg.run.seed}"
    if cfg.run.batch_id is not None:
        # Prefixed, not appended: every run in a batch then shares one
        # sorts-and-greps-together prefix regardless of what predictor/
        # store/seed varies between them (e.g. one predictor run per fold
        # of a partition-store split) -- feeds the run directory name,
        # MLflow's own run_name, and manifest.json's "run_name" all at
        # once, since all three call this function.
        name = f"{cfg.run.batch_id}__{name}"
    return name


def _savez_run(path: Path, test: MoleculeSet, pred: Prediction, /) -> None:
    np.savez(
        path,
        chembl_id=np.array(test.chembl_id),
        conf_id=np.array(test.conf_id),
        dash_id=np.array(test.dash_id),
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
    tracking: str | None = None,
    data_seconds: float = 0.0,
) -> RunResult:
    """Run the pipeline against an already-loaded ``mset``/``masks``.

    ``tracking`` defaults to ``None`` (no MLflow record) -- every real
    analysis this series has produced came from ``sweep``/``summarize``
    reading ``manifest.json``/``metrics.json`` directly off disk, never
    from MLflow, while MLflow's own artifact duplication
    (``mlflow.log_artifacts`` copies every file a run writes) has twice
    caused real disk-usage incidents on this shared machine. Pass
    ``DEFAULT_TRACKING_URI`` explicitly (or ``--track`` on the CLI) to opt
    back in for a given run; ``promote_run`` is unaffected -- logging to
    MLflow is its entire purpose, so its own default still points at the
    real tracking URI."""
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


def _normalize(raw: Any, mset: MoleculeSet, *, normalization: str) -> Prediction:
    """Apply ``normalize.NORMALIZERS[normalization]`` to a predictor's raw
    walk output (``predictors.base.RawPrediction``, from ``predict_raw``/
    ``predict_loo_raw``) -- the same call shape the old nested_runner.py
    used, now reachable from a flat run's ``normalization`` config key."""
    atom_charge = NORMALIZERS[normalization](
        raw.atom_charge,
        raw.atom_std,
        mset.net_charge,
        mset.atom_mol_id,
        mset.n_conformers,
    )
    return Prediction(atom_charge=atom_charge)


def _predict(predictor: Any, mset: MoleculeSet, *, cfg: ExperimentCfg) -> Prediction:
    """``predictor.predict(mset)``, unless ``cfg.normalization`` is set, in
    which case the predictor's own raw output is fetched via
    ``predict_raw`` and that scheme applied instead. Raises if a
    normalization is requested from a predictor that has no ``predict_raw``
    (e.g. ``global_mean``) -- there is no raw output to normalize."""
    if cfg.normalization is None:
        return predictor.predict(mset)
    if not hasattr(predictor, "predict_raw"):
        raise AttributeError(
            f"predictor {cfg.predictor.name!r} has no predict_raw method; "
            f"normalization={cfg.normalization!r} is not usable with it"
        )
    return _normalize(
        predictor.predict_raw(mset), mset, normalization=cfg.normalization
    )


def _score_extra_split(
    predictor: Any, mset: MoleculeSet, *, split: str, cfg: ExperimentCfg
) -> dict[str, float]:
    """Predict + score train/val the same way test is scored, mirroring
    cosmo_experiments' own train/val-alongside-test convention. Empty split
    -> no keys, not a crash."""
    if mset.n_conformers == 0:
        return {}
    pred = _predict(predictor, mset, cfg=cfg)
    score = _score(mset, pred)
    return {f"{split}/{k}": v for k, v in score.items()}


def _score_loo(
    predictor: Any, train: MoleculeSet, *, cfg: ExperimentCfg
) -> dict[str, float]:
    """Leave-one-out scoring of the *training* split, when the predictor
    opts in via ``report_loo``. Train-only by construction: LOO subtracts a
    node's own contribution from its class mean, which for a val or test
    node was never there (see SievePredictor.predict_loo_raw)."""
    if not getattr(predictor, "report_loo", False):
        return {}
    if not hasattr(predictor, "predict_loo_raw"):
        return {}
    if train.n_conformers == 0:
        return {}
    raw = predictor.predict_loo_raw(train)
    pred = (
        _normalize(raw, train, normalization=cfg.normalization)
        if cfg.normalization is not None
        else Prediction(atom_charge=raw.atom_charge)
    )
    score = _score(train, pred)
    return {f"train_loo/{k}": v for k, v in score.items()}


def _finite_pair(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Drop any (y_true, y_pred) pair where either side is NaN -- a
    predictor faithful to its own source's real missing-value behavior
    (e.g. predictors/dash_pretrained.py) can genuinely return NaN for an
    atom it declines to guess at, and a hexbin plot's own ``.min()``/
    ``.max()`` axis limits would otherwise go NaN from a single such
    point."""
    finite = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[finite], y_pred[finite]


def _build_parity_panels(
    test: MoleculeSet, pred: Prediction, run_metrics: dict[str, float]
) -> list[dict[str, Any]]:
    """Which panels a run's ``parity_panel.png`` gets: atom charge always
    (the primary prediction target, a hexbin parity plot -- ``test`` is
    assumed non-empty, ``_write_plots`` checks that before calling this);
    molecule charge conservation as a secondary diagnostic -- a 1-D
    histogram of the per-conformer residual (predicted
    atom charges summed, minus the conformer's own real ``net_charge``),
    not a parity scatter, since a residual is one number per conformer, not
    a true/predicted pair. The same residual ``metrics.
    charge_conservation_metrics`` already scores (its own ``err = y_pred -
    y_true`` is this exact quantity). Pure numpy -- no matplotlib -- so
    this is testable independent of ``plots.py`` actually rendering
    anything."""
    panels: list[dict[str, Any]] = []

    atom_true, atom_pred = _finite_pair(test.atom_charge, pred.atom_charge)
    if atom_true.size:  # every atom NaN (e.g. a pretrained baseline that
        # matched nothing in this split) -- an empty hexbin panel would
        # crash on its own .min()/.max() axis limits, so skip it, not fake it
        panels.append(
            {
                "y_true": atom_true,
                "y_pred": atom_pred,
                "quantity": "charge (e)",
                "title": "atom charge",
                "metrics": {
                    k: v for k, v in run_metrics.items() if k in ("mae", "rmse", "r2")
                },
            }
        )

    pred_net_charge = molecule_sum(
        pred.atom_charge, test.atom_mol_id, test.n_conformers
    )
    residual = pred_net_charge - test.net_charge
    residual = residual[~np.isnan(residual)]
    # A predictor whose own normalization already conserves charge exactly
    # (e.g. std_weighted/equal_weighted -- residuals at float round-off,
    # ~1e-16) has nothing worth plotting here: a histogram of that is a
    # single spike carrying no information, not a diagnostic. Same
    # threshold plots._histogram_subplot's own degenerate-bin-count guard
    # uses, kept in sync deliberately.
    if residual.size and np.max(np.abs(residual)) >= 1e-6:
        panels.append(
            {
                "kind": "histogram",
                "values": residual,
                "xlabel": "molecule charge residual (e)",
                "title": "molecule charge conservation",
                "metrics": {
                    k.removeprefix("charge_conservation/"): v
                    for k, v in run_metrics.items()
                    if k.startswith("charge_conservation/")
                },
            }
        )
    return panels


def _write_plots(
    run_dir: Path,
    test: MoleculeSet,
    pred: Prediction,
    run_metrics: dict[str, float],
    cfg: ExperimentCfg,
) -> None:
    if test.n_conformers == 0:
        return  # nothing to plot on an empty eval split (e.g. a small
        # --limit run whose split happens to land entirely in train)
    try:
        panels = _build_parity_panels(test, pred, run_metrics)
        plots.parity_panel(
            panels,
            run_dir / "parity_panel.png",
            suptitle=cfg.predictor.name,
        )
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots for this run")


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
    if cfg.tree_stats_load_path is not None:
        load_model_state = getattr(predictor, "load_model_state", None)
        if load_model_state is None:
            raise AttributeError(
                f"predictor {cfg.predictor.name!r} has no load_model_state "
                f"method; tree_stats_load_path is not usable with it"
            )
        load_model_state(cfg.tree_stats_load_path)
        tree_stats_source = "loaded"
    else:
        predictor.fit(train, val, rng=rng)
        tree_stats_source = "fit"
    fit_s = time.perf_counter() - t0

    # Opt-in via cfg.save_tree_stats, and only when fit() actually ran.
    # This was briefly automatic for any predictor exposing
    # save_model_state, which wrote ~21GB across ~380 `sieve` runs (57MB
    # each) to avoid ~15-second refits -- a bad trade that only pays when
    # fit() is genuinely expensive (as for `dash`). Written straight into
    # this run's own directory, so save_model_state(run_dir /
    # "tree_stats.npz") is itself the provenance record of which run
    # produced it. Still skipped when tree_stats_load_path loaded state
    # instead: re-saving would only write a byte-identical duplicate of the
    # file already at that path.
    if cfg.save_tree_stats and tree_stats_source == "fit":
        save_model_state = getattr(predictor, "save_model_state", None)
        if save_model_state is not None:
            save_model_state(run_dir / "tree_stats.npz")

    t0 = time.perf_counter()
    train_metrics = _score_extra_split(predictor, train, split="train", cfg=cfg)
    train_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_metrics = _score_extra_split(predictor, val, split="val", cfg=cfg)
    val_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    loo_metrics = _score_loo(predictor, train, cfg=cfg)
    loo_predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = _predict(predictor, test, cfg=cfg)
    predict_s = time.perf_counter() - t0

    run_metrics = _score(test, pred)
    run_metrics.update(train_metrics)
    if train_metrics:
        run_metrics["time/train_predict_s"] = train_predict_s
    run_metrics.update(val_metrics)
    if val_metrics:
        run_metrics["time/val_predict_s"] = val_predict_s
    run_metrics.update(loo_metrics)
    if loo_metrics:
        run_metrics["time/train_loo_predict_s"] = loo_predict_s
    run_metrics["time/fit_s"] = fit_s
    run_metrics["time/predict_s"] = predict_s
    run_metrics["time/data_s"] = data_seconds
    # Optional, predictor-specific: today only SievePredictor tracks this
    # (build_codes/from_rdkit, accumulated across fit's own featurization and
    # every predict_raw call this run made -- test, plus train/val above).
    # time/fit_s and time/predict_s each already include their own share of
    # it but do not separate it out, so without this the featurization/core
    # split (~96%/~4% on real data) is invisible in every recorded run.
    featurize_s = getattr(predictor, "last_featurize_s", None)
    if featurize_s is not None:
        run_metrics["time/featurize_s"] = featurize_s

    manifest = {
        "schema_version": 1,
        "run_name": _run_name(cfg),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": {"fit": fit_s, "predict": predict_s, "data": data_seconds},
        "tree_stats_source": tree_stats_source,
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
    _write_plots(run_dir, test, pred, run_metrics, cfg)

    if tracking is not None:
        _log_mlflow(cfg, run_metrics, run_dir, tracking)

    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


_ENSURE_EXPERIMENT_RETRIES = 5


def _ensure_experiment(
    experiment_name: str, *, artifact_root: Path = DEFAULT_ARTIFACT_ROOT
) -> None:
    """``mlflow.set_experiment`` alone auto-creates a missing experiment
    with whatever artifact root the backend defaults to -- for the sqlite
    backend, that's an unignored ``./mlruns/`` relative to the process's
    CWD, not anything tied to ``DEFAULT_TRACKING_URI``. Create the
    experiment explicitly first, with an artifact location under this
    series' own tree, so ``set_experiment`` only ever has to look it up.

    Retried against ``sqlalchemy.exc.OperationalError``: the sqlite
    backend's very first connection against a not-yet-existing (or
    not-yet-migrated) ``mlflow_runs.db`` runs Alembic's one-time schema
    migration. Several run processes launched in parallel against a fresh
    db (e.g. right after deleting it) all race to be that first connection
    -- one wins, the rest hit a genuine ``OperationalError`` (observed:
    ``table _alembic_tmp_experiments already exists``), not a real data
    problem. A short retry loop rides that out: once the winner finishes
    the migration, every loser's retry proceeds normally."""
    import time

    import mlflow
    from sqlalchemy.exc import OperationalError

    last_error: OperationalError | None = None
    for attempt in range(_ENSURE_EXPERIMENT_RETRIES):
        try:
            if mlflow.get_experiment_by_name(experiment_name) is None:
                artifact_root.mkdir(parents=True, exist_ok=True)
                artifact_location = (artifact_root / experiment_name).as_uri()
                mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
            mlflow.set_experiment(experiment_name)
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(
        f"could not initialize MLflow experiment {experiment_name!r} after "
        f"{_ENSURE_EXPERIMENT_RETRIES} attempts (sqlite migration race?)"
    ) from last_error


def _log_mlflow_run(
    tags: dict[str, str],
    params: dict[str, str],
    run_metrics: dict[str, float],
    run_dir: Path,
) -> None:
    """Log tags/params/metrics/artifacts onto whatever MLflow run is
    currently open (inside ``mlflow.start_run``'s own context) -- split out
    of ``_log_mlflow`` (below) so the actual logging calls have one place to
    live regardless of what opened the run."""
    import mlflow

    mlflow.set_tags(tags)
    mlflow.log_params(params)
    clean_metrics = {
        k: v for k, v in run_metrics.items() if isinstance(v, float) and not np.isnan(v)
    }
    mlflow.log_metrics({f"test/{k}": v for k, v in clean_metrics.items()})
    mlflow.log_artifacts(str(run_dir))


def _log_mlflow(
    cfg: ExperimentCfg, run_metrics: dict[str, float], run_dir: Path, tracking: str
) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping tracking for this run")
        return

    mlflow.set_tracking_uri(tracking)
    _ensure_experiment(cfg.run.experiment)
    tags = {
        "predictor": cfg.predictor.name,
        "split_column": cfg.data.split_column,
        "store": cfg.data.store,
        "seed": str(cfg.run.seed),
        "run_dir": str(run_dir),
        **({"batch_id": cfg.run.batch_id} if cfg.run.batch_id is not None else {}),
        **{f"tag.{k}": v for k, v in cfg.run.tags.items()},
    }
    with mlflow.start_run(run_name=_run_name(cfg)):
        _log_mlflow_run(tags, to_flat_params(cfg), run_metrics, run_dir)


@dataclass(frozen=True)
class PromoteResult:
    run_dir: Path
    experiment: str
    moved: bool
    logged: bool
    """False when an MLflow run already carried this run_dir's own tag --
    nothing was re-logged (and, if ``target_experiment`` was given but the
    run_dir already lived there, nothing was re-moved either)."""


def promote_run(
    run_dir: str | Path,
    *,
    target_experiment: str | None = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    tracking: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> PromoteResult:
    """Give an already-completed run an MLflow record, without re-running
    fit()/predict() -- reads exactly the ``config.resolved.yaml`` and
    ``metrics.json`` bytes a real run already wrote.

    Two situations this covers, both real:

    - ``target_experiment=None`` (the back-fill case): a run executed
      without ``--track`` (the default, untracked), or one whose own
      MLflow logging step crashed *after* local artifacts were already
      written -- e.g. the sqlite-migration race ``_ensure_experiment`` now
      retries against. Logs under the run's own ``config.run.experiment``,
      in place.
    - ``target_experiment`` set (the exploration -> baseline case): moves
      ``run_dir`` from ``runs_root/<old>/<name>`` to
      ``runs_root/<target_experiment>/<name>`` first, then logs there
      instead -- picking one result out of an exploratory sweep (its own
      ``run.experiment``, per the "experiments stay untracked" convention)
      to become part of the tracked baseline.

    Idempotent either way: an MLflow run already carrying this run_dir's
    own final path as its ``"run_dir"`` tag (the same tag ``_log_mlflow``
    always sets) is left alone -- ``PromoteResult.logged`` is ``False``,
    nothing is re-logged or moved again. Resolved to an absolute path
    first specifically to make that comparison reliable regardless of
    what form the caller passed: ``execute()`` always logs the tag as an
    absolute path (``run_dir`` is built from ``DEFAULT_RUNS_ROOT``, itself
    absolute), so a relative ``run_dir`` argument here would otherwise
    never match an already-tracked run's own tag and silently create a
    duplicate MLflow record on every call instead of no-op'ing.
    """
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.resolved.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no config.resolved.yaml/metrics.json -- only a "
            "completed run (one _execute_inner finished) can be promoted"
        )

    from charge_experiments.config import _build

    raw = yaml.safe_load(config_path.read_text())
    cfg = _build(raw)
    run_metrics = json.loads(metrics_path.read_text())

    moved = False
    if target_experiment is not None and target_experiment != cfg.run.experiment:
        new_dir = runs_root / target_experiment / run_dir.name
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(new_dir))
        run_dir = new_dir
        cfg = replace(cfg, run=replace(cfg.run, experiment=target_experiment))
        moved = True

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "mlflow is not installed; promote_run has nothing to log to"
        ) from exc

    mlflow.set_tracking_uri(tracking)
    _ensure_experiment(cfg.run.experiment, artifact_root=artifact_root)
    client = MlflowClient(tracking_uri=tracking)
    experiment = client.get_experiment_by_name(cfg.run.experiment)
    assert experiment is not None  # _ensure_experiment just created/found it
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.run_dir = '{run_dir}'",
        max_results=1,
    )
    if existing:
        return PromoteResult(
            run_dir=run_dir, experiment=cfg.run.experiment, moved=moved, logged=False
        )

    _log_mlflow(cfg, run_metrics, run_dir, tracking)
    return PromoteResult(
        run_dir=run_dir, experiment=cfg.run.experiment, moved=moved, logged=True
    )


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
        dash_id=list(df["dash_id"]),
        split=list(df[split_column]),
    )
    masks = {name: (df[split_column] == name).to_numpy() for name in splits}
    return mset, masks


def run(
    cfg: ExperimentCfg,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    allow_dirty: bool = False,
    tracking: str | None = None,
    limit: int | None = None,
) -> RunResult:
    """Load the real store, then run the pipeline (see ``execute``, whose
    own docstring covers why ``tracking`` defaults to ``None``)."""
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
