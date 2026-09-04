"""Run configuration: YAML in, frozen dataclasses out. Mirrors
cosmo_experiments/sieve_experiments/config.py's shape, with two series-
specific additions: an optional top-level `normalization` key (a name from
``normalize.NORMALIZERS``) applied to a predictor's raw output -- see
``ExperimentCfg.normalization``'s own docstring and runner.py's `_predict`
-- and an optional `tree_stats_load_path` key to skip `fit()` in favor of
an earlier run's own saved model state -- see
``ExperimentCfg.tree_stats_load_path``'s own docstring. `run.batch_id`
groups several runs launched together as one sweep into a shared,
greppable run-directory prefix -- see ``RunCfg.batch_id``'s own
docstring. There is exactly one valid split column (`split` -- this
series builds no size-biased second split, per the design spec's
"Out of scope" list)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from charge_experiments.normalize import NORMALIZERS

VALID_SPLIT_COLUMNS = ("split",)

_RUN_KEYS = {"experiment", "seed", "tags", "batch_id"}
_DATA_KEYS = {"store", "split_column", "train_split", "val_split", "eval_split"}
_PREDICTOR_KEYS = {"name", "params"}
_TOP_KEYS = {
    "run",
    "data",
    "predictor",
    "normalization",
    "tree_stats_load_path",
    "save_tree_stats",
}


@dataclass(frozen=True)
class RunCfg:
    experiment: str
    seed: int
    tags: Mapping[str, str] = field(default_factory=dict)
    batch_id: str | None = None
    """Groups several independent runs (e.g. one per fold of a
    ``partition-store`` split) that were launched together as one logical
    sweep -- distinct from ``experiment``, which is a broad, reused
    category (``dash-charges``), not a specific sweep instance: the same
    experiment accumulates many unrelated batches over time. When set,
    prefixes the run directory name (``runner._run_name``) so every run in
    a batch sorts and greps together regardless of when each one finished,
    and is also logged as an MLflow tag. ``None`` (the default) leaves
    run-directory naming exactly as before."""


@dataclass(frozen=True)
class DataCfg:
    store: str
    split_column: str
    train_split: str = "train"
    val_split: str = "val"
    eval_split: str = "test"

    def __post_init__(self) -> None:
        if self.split_column not in VALID_SPLIT_COLUMNS:
            raise ValueError(
                f"data.split_column must be one of {VALID_SPLIT_COLUMNS}, "
                f"got {self.split_column!r}"
            )


@dataclass(frozen=True)
class PredictorCfg:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentCfg:
    run: RunCfg
    data: DataCfg
    predictor: PredictorCfg
    normalization: str | None = None
    """A key into ``normalize.NORMALIZERS`` -- when set, the run predicts via
    the predictor's own ``predict_raw`` and applies this scheme to every
    split (test/train/val/LOO) instead of calling ``predict()`` directly.
    ``None`` (the default) is today's behavior: each predictor's own
    internal handling, unchanged."""
    save_tree_stats: bool = False
    """Write the fitted predictor's own state to ``<run_dir>/tree_stats.npz``
    after ``fit()``, for later reuse via ``tree_stats_load_path``. Opt-in,
    and deliberately so: this was briefly automatic for any predictor
    exposing ``save_model_state``, which wrote ~21GB across ~380 `sieve`
    runs (57MB each) to avoid ~15-second refits -- worth it only when
    ``fit()`` is genuinely expensive, as it is for ``dash`` (a real
    tree-matching walk). A predictor without ``save_model_state`` ignores
    this silently; a run that *loaded* its state never re-saves regardless
    (see ``runner._execute_inner``)."""
    tree_stats_load_path: str | None = None
    """Path to an earlier run's own ``tree_stats.npz`` (written by a run
    that set ``save_tree_stats`` -- see its docstring above). When set,
    ``fit()`` is skipped in favor of
    ``predictor.load_model_state(tree_stats_load_path)``, and this run does
    *not* write its own ``tree_stats.npz`` -- that would only be a
    byte-identical duplicate of the file already at this path (real DASH
    trees are O(100MB)); ``tree_stats_load_path`` itself is the provenance
    record. Raises if the predictor has no ``load_model_state``. ``None``
    (the default) always calls ``fit()``."""


def _check_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"unknown key(s) in {where}: {sorted(extra)}")


def _parse_scalar(text: str) -> Any:
    """Best-effort str -> int/float/bool for ``--set`` overrides."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a raw (pre-validation) config dict."""
    out = copy.deepcopy(dict(raw))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be 'key.path=value', got {override!r}")
        path, value = override.split("=", 1)
        keys = path.split(".")
        node = out
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _parse_scalar(value)
    return out


def _build(raw: Mapping[str, Any]) -> ExperimentCfg:
    _check_keys(raw, _TOP_KEYS, "config")
    for section in ("run", "data", "predictor"):
        if section not in raw:
            raise ValueError(f"config is missing required section {section!r}")

    run_raw = raw["run"]
    _check_keys(run_raw, _RUN_KEYS, "run")
    run = RunCfg(
        experiment=run_raw["experiment"],
        seed=run_raw["seed"],
        tags=dict(run_raw.get("tags", {})),
        batch_id=run_raw.get("batch_id"),
    )

    data_raw = raw["data"]
    _check_keys(data_raw, _DATA_KEYS, "data")
    data = DataCfg(
        store=data_raw["store"],
        split_column=data_raw["split_column"],
        **{
            k: data_raw[k]
            for k in ("train_split", "val_split", "eval_split")
            if k in data_raw
        },
    )

    predictor_raw = raw["predictor"]
    _check_keys(predictor_raw, _PREDICTOR_KEYS, "predictor")
    predictor = PredictorCfg(
        name=predictor_raw["name"], params=dict(predictor_raw.get("params", {}))
    )

    normalization = raw.get("normalization")
    if normalization is not None and normalization not in NORMALIZERS:
        raise ValueError(
            f"normalization must be one of {sorted(NORMALIZERS)} or omitted, "
            f"got {normalization!r}"
        )

    tree_stats_load_path = raw.get("tree_stats_load_path")
    save_tree_stats = bool(raw.get("save_tree_stats", False))

    return ExperimentCfg(
        run=run,
        data=data,
        predictor=predictor,
        normalization=normalization,
        tree_stats_load_path=tree_stats_load_path,
        save_tree_stats=save_tree_stats,
    )


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentCfg:
    """Load and validate a YAML config file, applying ``--set`` overrides."""
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(raw)


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = str(value)


def to_dict(cfg: ExperimentCfg) -> dict[str, Any]:
    """The resolved config as a plain nested dict -- what gets written to
    ``config.resolved.yaml`` in a run directory."""
    return {
        "run": {
            "experiment": cfg.run.experiment,
            "seed": cfg.run.seed,
            "tags": dict(cfg.run.tags),
            "batch_id": cfg.run.batch_id,
        },
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "eval_split": cfg.data.eval_split,
        },
        "predictor": {"name": cfg.predictor.name, "params": dict(cfg.predictor.params)},
        "normalization": cfg.normalization,
        "tree_stats_load_path": cfg.tree_stats_load_path,
        "save_tree_stats": cfg.save_tree_stats,
    }


def to_flat_params(cfg: ExperimentCfg) -> dict[str, str]:
    """Flatten a config to dot-separated string params, for MLflow logging."""
    out: dict[str, str] = {}
    _flatten("run", {"experiment": cfg.run.experiment, "seed": cfg.run.seed}, out)
    _flatten("run.batch_id", cfg.run.batch_id, out)
    for k, v in cfg.run.tags.items():
        out[f"run.tags.{k}"] = str(v)
    _flatten(
        "data",
        {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "eval_split": cfg.data.eval_split,
        },
        out,
    )
    _flatten("predictor.name", cfg.predictor.name, out)
    _flatten("predictor.params", dict(cfg.predictor.params), out)
    _flatten("normalization", cfg.normalization, out)
    _flatten("tree_stats_load_path", cfg.tree_stats_load_path, out)
    _flatten("save_tree_stats", cfg.save_tree_stats, out)
    return out
