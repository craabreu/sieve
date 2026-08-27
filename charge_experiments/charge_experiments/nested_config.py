"""Config for a nested run: one predictor's fit()+save+raw-predict (the
"parent"), plus one child run per normalization scheme applied to the same
already-computed raw predictions. See docs/superpowers/specs/
2026-08-27-dash-charges-nested-runs-design.md.

``tree_stats`` doesn't hardcode which predictor names support it -- that's a
duck-typed runtime concern for nested_runner.execute_nested (checked via
hasattr), not a parse-time one here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from charge_experiments.config import (
    DataCfg,
    PredictorCfg,
    RunCfg,
    _check_keys,
    apply_overrides,
)
from charge_experiments.normalize import NORMALIZERS

_TOP_KEYS = {"run", "data", "predictor", "tree_stats", "children"}
_TREE_STATS_KEYS = {"save_path", "load_path"}


@dataclass(frozen=True)
class TreeStatsCfg:
    save_path: str | None = None
    load_path: str | None = None


@dataclass(frozen=True)
class NestedExperimentCfg:
    run: RunCfg
    data: DataCfg
    predictor: PredictorCfg
    tree_stats: TreeStatsCfg
    children: tuple[str, ...]


def _build(raw: Mapping[str, Any]) -> NestedExperimentCfg:
    _check_keys(raw, _TOP_KEYS, "config")
    for section in ("run", "data", "predictor", "children"):
        if section not in raw:
            raise ValueError(f"config is missing required section {section!r}")

    run_raw = raw["run"]
    run = RunCfg(
        experiment=run_raw["experiment"],
        seed=run_raw["seed"],
        tags=dict(run_raw.get("tags", {})),
    )

    data_raw = raw["data"]
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
    predictor = PredictorCfg(
        name=predictor_raw["name"], params=dict(predictor_raw.get("params", {}))
    )

    tree_stats_raw = raw.get("tree_stats")
    if tree_stats_raw is not None:
        _check_keys(tree_stats_raw, _TREE_STATS_KEYS, "tree_stats")
        tree_stats = TreeStatsCfg(
            save_path=tree_stats_raw.get("save_path"),
            load_path=tree_stats_raw.get("load_path"),
        )
    else:
        tree_stats = TreeStatsCfg()

    children = tuple(raw["children"])
    if not children:
        raise ValueError("children must list at least one normalization scheme")
    unknown = [c for c in children if c not in NORMALIZERS]
    if unknown:
        raise ValueError(
            f"unknown normalization scheme(s) in children: {unknown}; "
            f"known: {sorted(NORMALIZERS)}"
        )

    return NestedExperimentCfg(
        run=run, data=data, predictor=predictor, tree_stats=tree_stats,
        children=children,
    )


def load_nested_config(
    path: str | Path, overrides: Sequence[str] = ()
) -> NestedExperimentCfg:
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(raw)
