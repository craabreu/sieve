"""Run configuration: YAML in, frozen dataclasses out. Mirrors
cosmo_experiments/sieve_experiments/config.py's shape; this series drops the
`scheme` field (no sigma-averaging-scheme concept here) and has exactly one
valid split column (`split` -- this series builds no size-biased second
split, per the design spec's "Out of scope" list)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_SPLIT_COLUMNS = ("split",)

_RUN_KEYS = {"experiment", "seed", "tags"}
_DATA_KEYS = {"store", "split_column", "train_split", "val_split", "eval_split"}
_PREDICTOR_KEYS = {"name", "params"}
_TOP_KEYS = {"run", "data", "predictor"}


@dataclass(frozen=True)
class RunCfg:
    experiment: str
    seed: int
    tags: Mapping[str, str] = field(default_factory=dict)


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

    return ExperimentCfg(run=run, data=data, predictor=predictor)


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
        },
        "data": {
            "store": cfg.data.store,
            "split_column": cfg.data.split_column,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "eval_split": cfg.data.eval_split,
        },
        "predictor": {"name": cfg.predictor.name, "params": dict(cfg.predictor.params)},
    }


def to_flat_params(cfg: ExperimentCfg) -> dict[str, str]:
    """Flatten a config to dot-separated string params, for MLflow logging."""
    out: dict[str, str] = {}
    _flatten("run", {"experiment": cfg.run.experiment, "seed": cfg.run.seed}, out)
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
    return out
