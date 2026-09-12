"""Cluster-clean-shard cross-validation: fit N shards once, assemble every
CV sample's training model by merging, evaluate on the held-out shards --
see docs/superpowers/specs/2026-09-12-cv-shard-redesign-design.md.

Why this module exists rather than N independent ``runner.run`` calls per
sample: partial fits are shards (design.md 5 for Sieve; ``tree_artifact``
for DASH), so a k-fold training model is a merge of k-1 shards, not a
fresh fit. That turns "more CV samples" from a compute question into a
statistical one -- fitting N shards once serves any number of repeats.

Three layers, in the order below:

- pure algorithmic core (``permute_into_folds``, ``leave_one_group_out``)
  -- no I/O, fully unit-testable;
- shard fitting (``fit_dash_shard``/``fit_sieve_shard``) -- one run
  directory per shard under the ``cv-shard-fits`` experiment, predicting
  nothing (a shard is only ever used merged), idempotent via its own
  ``tree_stats.npz``;
- CV assembly + evaluation (``run_dash_cv``/``run_sieve_cv``) -- one run
  directory per (repeat, fold, method, depth), written in the same
  ``manifest["config"]`` shape ``runner.py`` writes, so ``summarize``/
  ``sweep``/``compare`` (this package's own, plus this module's) read a CV
  run exactly like an ordinary one.

Deliberately narrower than ``runner.execute``: no MLflow tracking, no LOO,
no ``val`` split (the CV redesign has none). ``normalization`` is always
applied *in addition to* the raw score, not instead of it -- both metric
families land in one ``metrics.json``, the normalized one under a
``norm/`` prefix, since normalization is a pure post-hoc transform of the
same raw prediction and costs nothing extra to also report unnormalized
(design decision, not a default carried over from ``runner``).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from experiments.data import REPO_ROOT, MoleculeSet, concat_molecule_sets
from experiments.normalize import NORMALIZERS
from experiments.predictors.base import Prediction, RawPrediction
from experiments.runner import (
    DEFAULT_RUNS_ROOT,
    RunResult,
    _git_info,
    _package_versions,
    _savez_run,
    _score,
    load_molecule_set,
)

logger = logging.getLogger("experiments")

T = TypeVar("T")

SHARD_FIT_EXPERIMENT = "cv-shard-fits"


# ---------------------------------------------------------------------------
# Pure algorithmic core
# ---------------------------------------------------------------------------


def shard_ids(n_shards: int) -> list[str]:
    """``s00`` .. ``s{n_shards-1}`` -- ``prepare_dash.assign_splits``'s own
    shard naming, repeated here so nothing else has to hardcode the width."""
    return [f"s{i:02d}" for i in range(n_shards)]


def permute_into_folds(items: Sequence[T], *, k: int, seed: int) -> list[list[T]]:
    """One random permutation of ``items``, cut into ``k`` equal-size
    groups (``len(items)`` must be divisible by ``k``).

    ``seed`` is what actually re-randomizes a repeat. The shards themselves
    were assigned by ``greedy_cluster_split``, which is deterministic --
    re-seeding *it* would barely move anything (the large clusters land in
    a near-fixed pattern; only small ones move). Permuting the already-built
    shards instead moves every one of them, uniformly, at no refit cost.
    """
    n = len(items)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n % k != 0:
        raise ValueError(f"{n} items not evenly divisible into {k} groups")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    size = n // k
    return [[items[i] for i in order[g * size : (g + 1) * size]] for g in range(k)]


def leave_one_group_out(groups: Sequence[T], *, merge: Callable[[T, T], T]) -> list[T]:
    """``k`` already-merged group-level values in, ``k`` values out -- each
    the merge of every group except its own index, via a prefix/suffix scan
    so this costs ``O(k)`` merges total, not ``O(k)`` merges each redone
    over the other ``k-1`` groups from scratch.

    A ``None`` sentinel stands in for "the merge of zero groups": neither
    artifact this module drives has a natural identity element worth
    building just for this (``sieve.merge.SieveModel.empty()`` exists but
    needs a config to hand it; DASH's own ``TreeNodeStats`` has none at all,
    per ``fold_node_stats``'s own docstring) -- ``merge`` itself is only
    ever called on two real group values, never on the sentinel.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("leave_one_group_out needs at least 2 groups")

    def _merge_opt(a: T | None, b: T | None) -> T | None:
        if a is None:
            return b
        if b is None:
            return a
        return merge(a, b)

    prefix: list[T | None] = [None] * (k + 1)
    for i in range(k):
        prefix[i + 1] = _merge_opt(prefix[i], groups[i])
    suffix: list[T | None] = [None] * (k + 1)
    for i in reversed(range(k)):
        suffix[i] = _merge_opt(groups[i], suffix[i + 1])
    # Never actually None: prefix[i] and suffix[i+1] together cover every
    # group except i, and k >= 2 guarantees at least one of them is real.
    return [cast(T, _merge_opt(prefix[i], suffix[i + 1])) for i in range(k)]


# ---------------------------------------------------------------------------
# Batch-id / experiment-name conventions
# ---------------------------------------------------------------------------


def dash_shard_batch_id(shard: str) -> str:
    return f"fit-dash-{shard}"


def sieve_shard_batch_id(config_label: str, depth: int, shard: str) -> str:
    return f"fit-sieve-{config_label}-w{depth}-{shard}"


def cv_batch_id(*, repeat: int, fold: int, method: str, depth: int) -> str:
    return f"r{repeat}-f{fold}-{method}-w{depth}"


# ---------------------------------------------------------------------------
# Shared store-loading / run-writing helpers
# ---------------------------------------------------------------------------


def load_shards(
    store: str,
    ids: Sequence[str],
    *,
    atom_property: str = "MBIScharge",
    molecule_property: str | None = "net_charge",
    stores_root: Path | None = None,
) -> dict[str, MoleculeSet]:
    """Load ``store``'s ``shard`` column once and split it into one
    ``MoleculeSet`` per requested shard id -- a single parquet read shared
    across every repeat/fold/depth that follows, rather than one read per
    call (the store is re-read only when this function is called again)."""
    from experiments.config import TargetCfg

    target = TargetCfg(atom_property=atom_property, molecule_property=molecule_property)
    mset, masks = load_molecule_set(
        store,
        target=target,
        split_column="shard",
        splits=tuple(ids),
        stores_root=stores_root,
    )
    return {s: mset.select(masks[s]) for s in ids}


def _new_run_dir(
    runs_root: Path,
    experiment: str,
    batch_id: str,
    *,
    predictor: str,
    store: str,
    seed: int,
) -> tuple[Path, str, datetime]:
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    name = f"{batch_id}__{predictor}-{store}-s{seed}__{stamp}__{run_id}"
    run_dir = runs_root / experiment / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, name, started


def _check_clean(allow_dirty: bool) -> dict[str, Any]:
    git_info = _git_info(REPO_ROOT)
    if git_info["dirty"] and not allow_dirty:
        raise RuntimeError(
            "git working tree is dirty; commit your changes or pass allow_dirty=True"
        )
    return git_info


# ---------------------------------------------------------------------------
# Shard fitting
# ---------------------------------------------------------------------------


def _shard_fit_done(runs_root: Path, batch_id: str) -> Path | None:
    matches = sorted(
        (runs_root / SHARD_FIT_EXPERIMENT).glob(f"{batch_id}__*/tree_stats.npz")
    )
    return matches[-1] if matches else None


def fit_dash_shard(
    *,
    store: str,
    shard: str,
    max_depth: int,
    atom_property: str = "MBIScharge",
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> Path:
    """Fit DASH on one shard's own rows (``split_column="shard"``) at
    ``max_depth``, save its ``tree_stats.npz``, predict nothing -- a shard
    is only ever used merged (see the module docstring). Idempotent: skips
    straight to the existing artifact if this shard was already fit.

    ``max_depth`` should be the deepest depth the CV sweep will ever need:
    DASH node stats are depth-invariant (one fit at the max depth serves
    every shallower one via ``predict_raw_at_depth``, exactly as
    ``dash_depth_sweep`` already exploits), so one shard fit per shard
    suffices for the whole depth sweep, unlike Sieve (see
    ``fit_sieve_shard``)."""
    from experiments.config import TargetCfg
    from experiments.predictors.dash import DASHChargePredictor

    batch_id = dash_shard_batch_id(shard)
    existing = _shard_fit_done(runs_root, batch_id)
    if existing is not None:
        logger.info("dash shard %r already fit; skipping", shard)
        return existing

    git_info = _check_clean(allow_dirty)

    mset, masks = load_molecule_set(
        store,
        target=TargetCfg(atom_property=atom_property),
        split_column="shard",
        splits=(shard,),
        stores_root=stores_root,
    )
    train = mset.select(masks[shard])

    predictor = DASHChargePredictor(max_depth=max_depth)
    t0 = time.perf_counter()
    predictor.fit(train, train, rng=np.random.default_rng(seed))
    fit_s = time.perf_counter() - t0

    run_dir, _name, started = _new_run_dir(
        runs_root,
        SHARD_FIT_EXPERIMENT,
        batch_id,
        predictor="dash",
        store=store,
        seed=seed,
    )
    out_path = run_dir / "tree_stats.npz"
    predictor.save_model_state(out_path)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(UTC).isoformat(),
                "git": git_info,
                "seed": seed,
                "packages": _package_versions(),
                "shard": shard,
                "max_depth": max_depth,
                "n_train_conformers": train.n_conformers,
                "elapsed_s": {"fit": fit_s},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return out_path


def fit_sieve_shard(
    *,
    store: str,
    shard: str,
    depth: int,
    codes_path: str | Path,
    config_label: str,
    predictor_params: dict[str, Any] | None = None,
    atom_property: str = "MBIScharge",
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> Path:
    """Fit Sieve on one shard's own rows at ``max_wl_depth=depth``, using a
    vocabulary frozen over the whole train split (``codes_path`` --
    without it, this shard's own ``build_codes`` pass would discover a
    vocabulary local to its own ~1/N of train, and two shards' configs
    could then disagree on what integer code means what value, refusing to
    merge -- see ``predictors.sieve_predictor.SievePredictor``'s own
    docstring on ``codes_path``).

    Unlike DASH, Sieve fits one shard set **per depth**: under
    ``class_estimator="continuation"`` a class's estimate depends on
    whether its own level is the model's *deepest* one, so a shallow
    config is not a truncation of a deep one and the two are not
    mergeable either (different ``max_wl_depth`` means different
    ``schema_version``) -- see ``workflows/sieve_charges.sh``'s own Stage 2
    note, which established this for the original fold sweep.

    ``config_label`` distinguishes one named Sieve configuration (e.g.
    ``"element-eb"``) from another sharing the same store/shard/depth, so
    several configurations' shards can coexist under one experiment."""
    from experiments.config import TargetCfg
    from experiments.predictors.sieve_predictor import SievePredictor

    batch_id = sieve_shard_batch_id(config_label, depth, shard)
    existing = _shard_fit_done(runs_root, batch_id)
    if existing is not None:
        logger.info(
            "sieve shard %r (%s, w%d) already fit; skipping", shard, config_label, depth
        )
        return existing

    git_info = _check_clean(allow_dirty)

    mset, masks = load_molecule_set(
        store,
        target=TargetCfg(atom_property=atom_property),
        split_column="shard",
        splits=(shard,),
        stores_root=stores_root,
    )
    train = mset.select(masks[shard])

    params = dict(predictor_params or {})
    params["max_wl_depth"] = depth
    params["codes_path"] = str(codes_path)
    predictor = SievePredictor(**params)
    t0 = time.perf_counter()
    predictor.fit(train, train, rng=np.random.default_rng(seed))
    fit_s = time.perf_counter() - t0

    run_dir, _name, started = _new_run_dir(
        runs_root,
        SHARD_FIT_EXPERIMENT,
        batch_id,
        predictor="sieve",
        store=store,
        seed=seed,
    )
    out_path = run_dir / "tree_stats.npz"
    predictor.save_model_state(out_path)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(UTC).isoformat(),
                "git": git_info,
                "seed": seed,
                "packages": _package_versions(),
                "shard": shard,
                "config_label": config_label,
                "depth": depth,
                "predictor_params": params,
                "n_train_conformers": train.n_conformers,
                "elapsed_s": {"fit": fit_s, "featurize": predictor.last_featurize_s},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return out_path


def run_dash_shard_fits(
    *,
    store: str,
    n_shards: int,
    max_depth: int,
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> list[Path]:
    return [
        fit_dash_shard(
            store=store,
            shard=s,
            max_depth=max_depth,
            seed=seed,
            runs_root=runs_root,
            stores_root=stores_root,
            allow_dirty=allow_dirty,
        )
        for s in shard_ids(n_shards)
    ]


def run_sieve_shard_fits(
    *,
    store: str,
    n_shards: int,
    depths: Sequence[int],
    codes_path: str | Path,
    config_label: str,
    predictor_params: dict[str, Any] | None = None,
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> dict[int, list[Path]]:
    return {
        depth: [
            fit_sieve_shard(
                store=store,
                shard=s,
                depth=depth,
                codes_path=codes_path,
                config_label=config_label,
                predictor_params=predictor_params,
                seed=seed,
                runs_root=runs_root,
                stores_root=stores_root,
                allow_dirty=allow_dirty,
            )
            for s in shard_ids(n_shards)
        ]
        for depth in depths
    }


# ---------------------------------------------------------------------------
# CV assembly + evaluation
# ---------------------------------------------------------------------------


def _score_raw_and_normalized(
    raw: RawPrediction, mset: MoleculeSet, *, normalization: str
) -> dict[str, float]:
    """Score a single raw prediction twice: unnormalized, and (when the
    store carries a per-molecule total) with ``normalization`` applied --
    the second costs no extra fit or walk, only ``NORMALIZERS``' own
    post-hoc redistribution, so both are always worth recording rather
    than choosing one in advance."""
    metrics = _score(mset, Prediction(atom_value=raw.atom_value))
    if mset.molecule_value is not None:
        normalized_value = NORMALIZERS[normalization](
            raw.atom_value,
            raw.atom_std,
            mset.molecule_value,
            mset.atom_mol_id,
            mset.n_conformers,
        )
        norm_metrics = _score(mset, Prediction(atom_value=normalized_value))
        metrics.update({f"norm/{k}": v for k, v in norm_metrics.items()})
    return metrics


def _cv_run_done(runs_root: Path, experiment: str, batch_id: str) -> Path | None:
    matches = sorted((runs_root / experiment).glob(f"{batch_id}__*/metrics.json"))
    return matches[-1].parent if matches else None


def _write_cv_run(
    *,
    experiment: str,
    batch_id: str,
    method: str,
    store: str,
    seed: int,
    held_out: MoleculeSet,
    raw: RawPrediction,
    normalization: str,
    repeat: int,
    fold: int,
    depth: int,
    held_out_shards: list[str],
    elapsed_s: dict[str, float],
    git_info: dict[str, Any],
    runs_root: Path,
    save_predictions: bool = True,
) -> RunResult:
    """Write one CV sample's run directory. The ``manifest["config"]`` shape
    mirrors what ``config.to_dict``/``runner`` write (``predictor.name``,
    ``run.batch_id``/``run.tags``, plus a ``cv`` block of this driver's own
    fields) purely so ``aggregate.read_runs_from_dirs``'s existing
    ``flatten_params`` -- and therefore ``summarize``/``sweep`` -- read a CV
    run exactly like an ordinary one, with no changes to either."""
    started = datetime.now(UTC)
    run_metrics = _score_raw_and_normalized(raw, held_out, normalization=normalization)
    for k, v in elapsed_s.items():
        run_metrics[f"time/{k}_s"] = v

    run_dir, name, _started = _new_run_dir(
        runs_root, experiment, batch_id, predictor=method, store=store, seed=seed
    )
    config = {
        "run": {
            "experiment": experiment,
            "batch_id": batch_id,
            "seed": seed,
            "tags": {
                "repeat": str(repeat),
                "fold": str(fold),
                "method": method,
                "depth": str(depth),
            },
        },
        "data": {"store": store, "split_column": "shard"},
        "predictor": {"name": method},
        "cv": {
            "repeat": repeat,
            "fold": fold,
            "method": method,
            "depth": depth,
            "normalization": normalization,
            "held_out_shards": ",".join(held_out_shards),
        },
    }
    manifest = {
        "schema_version": 1,
        "run_name": name,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed_s,
        "git": git_info,
        "seed": seed,
        "packages": _package_versions(),
        "data": {
            "store": store,
            "split_column": "shard",
            "n_test_conformers": held_out.n_conformers,
            "n_test_atoms": held_out.n_atoms,
        },
        "config": config,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    if save_predictions:
        _savez_run(
            run_dir / "predictions.npz", held_out, Prediction(atom_value=raw.atom_value)
        )
    return RunResult(run_dir=run_dir, metrics=run_metrics, manifest=manifest)


@dataclass(frozen=True)
class CVPlan:
    """One repeat's own fold assignment: ``k`` groups of shard ids, and the
    complementary "everything but this group" grouping ``leave_one_group_out``
    needs no explicit list for (it works on group-level *values*, not shard
    lists) -- kept here purely for provenance/logging."""

    repeat: int
    groups: list[list[str]]


def build_cv_plan(ids: Sequence[str], *, k: int, repeat: int) -> CVPlan:
    return CVPlan(repeat=repeat, groups=permute_into_folds(ids, k=k, seed=repeat))


def run_dash_cv(
    *,
    store: str,
    n_shards: int,
    depths: Sequence[int],
    repeats: Sequence[int],
    k: int = 5,
    max_depth: int,
    normalization: str = "std_weighted",
    method: str = "dash",
    experiment: str = "dash-cv",
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> list[RunResult]:
    """Run DASH's own CV sweep: for each ``repeat``, permute the ``n_shards``
    shards into ``k`` groups, assemble each group's leave-one-out training
    model by merging (``tree_artifact.merge_node_stats``, exact), and score
    every requested depth against that fold's held-out group.

    Requires every shard already fit (``run_dash_shard_fits`` at
    ``max_depth``); raises naming any that are missing. One tree load, one
    ``match_paths`` walk per (repeat, fold) -- shared across every depth,
    the same saving ``dash_depth_sweep`` already relies on for a single
    fold sweep.
    """
    from experiments.predictors.dash import DASHChargePredictor
    from experiments.tree_artifact import (
        apply_node_stats,
        fold_node_stats,
        load_node_stats,
        merge_node_stats,
    )

    ids = shard_ids(n_shards)
    shard_paths_or_none = {
        s: _shard_fit_done(runs_root, dash_shard_batch_id(s)) for s in ids
    }
    missing = [s for s, p in shard_paths_or_none.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"no shard fit for {missing}; run run_dash_shard_fits first"
        )
    shard_paths: dict[str, Path] = {
        s: p for s, p in shard_paths_or_none.items() if p is not None
    }
    stats_by_shard = {s: load_node_stats(p) for s, p in shard_paths.items()}

    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    git_info = _check_clean(allow_dirty)

    predictor = DASHChargePredictor(max_depth=max_depth)
    predictor._load_tree()  # loaded once, reused across every repeat/fold

    results: list[RunResult] = []
    for repeat in repeats:
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        group_stats = [
            fold_node_stats(stats_by_shard[s] for s in g) for g in plan.groups
        ]
        train_stats = leave_one_group_out(group_stats, merge=merge_node_stats)

        for fold, group in enumerate(plan.groups):
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])

            t0 = time.perf_counter()
            paths = predictor.match_paths(held_out, split="cv")
            walk_s = time.perf_counter() - t0

            predictor._stats = train_stats[fold]
            predictor._mean_props, predictor._std_props = apply_node_stats(
                predictor._tree, predictor._stats, reset_existing=True
            )

            for depth in depths:
                batch_id = cv_batch_id(
                    repeat=repeat, fold=fold, method=method, depth=depth
                )
                if _cv_run_done(runs_root, experiment, batch_id) is not None:
                    logger.info("%s already done; skipping", batch_id)
                    continue
                t0 = time.perf_counter()
                raw = predictor.predict_raw_at_depth(paths, max_depth=depth)
                predict_s = time.perf_counter() - t0
                results.append(
                    _write_cv_run(
                        experiment=experiment,
                        batch_id=batch_id,
                        method=method,
                        store=store,
                        seed=seed,
                        held_out=held_out,
                        raw=raw,
                        normalization=normalization,
                        repeat=repeat,
                        fold=fold,
                        depth=depth,
                        held_out_shards=group,
                        elapsed_s={"walk": walk_s, "predict": predict_s},
                        git_info=git_info,
                        runs_root=runs_root,
                    )
                )
    return results


def run_sieve_cv(
    *,
    store: str,
    n_shards: int,
    depths: Sequence[int],
    repeats: Sequence[int],
    codes_path: str | Path,
    config_label: str,
    predictor_params: dict[str, Any] | None = None,
    k: int = 5,
    normalization: str = "equal_weighted",
    method: str | None = None,
    experiment: str = "sieve-cv",
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> list[RunResult]:
    """Sieve's own CV sweep -- the same shape as ``run_dash_cv``, except a
    training model is assembled **per depth** (Sieve shards are fit one set
    per depth; see ``fit_sieve_shard``'s own docstring on why) and
    evaluation reuses one featurized batch across every depth
    (``SievePredictor.build_predict_batch``/``predict_raw_from_batch`` --
    valid because every depth's shards share ``codes_path``'s one frozen
    vocabulary), rather than one tree-matching walk shared across depths
    the way DASH's paths are.

    Requires every shard already fit at every requested depth
    (``run_sieve_shard_fits``); raises naming any that are missing.
    """
    import sieve
    from experiments.predictors.sieve_predictor import SievePredictor
    from sieve.merge import fold as sieve_fold
    from sieve.merge import merge_models

    method = method or f"sieve-{config_label}"
    ids = shard_ids(n_shards)

    shard_paths: dict[int, dict[str, Path]] = {}
    for depth in depths:
        paths_or_none = {
            s: _shard_fit_done(runs_root, sieve_shard_batch_id(config_label, depth, s))
            for s in ids
        }
        missing = [s for s, p in paths_or_none.items() if p is None]
        if missing:
            raise FileNotFoundError(
                f"no shard fit for {config_label!r} depth {depth}, shard(s) "
                f"{missing}; run run_sieve_shard_fits first"
            )
        shard_paths[depth] = {s: p for s, p in paths_or_none.items() if p is not None}

    models_by_shard_by_depth = {
        depth: {s: sieve.SieveModel.load(p) for s, p in paths.items()}
        for depth, paths in shard_paths.items()
    }

    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    git_info = _check_clean(allow_dirty)

    # One predictor per depth, reused across every repeat/fold; a second,
    # bare predictor supplies build_predict_batch (any depth's config has
    # the same attribute_codes/edge_codes, since they were all frozen from
    # one codes_path -- see fit_sieve_shard).
    predictors_by_depth = {
        depth: SievePredictor(**(predictor_params or {})) for depth in depths
    }
    batch_predictor = predictors_by_depth[depths[0]]

    results: list[RunResult] = []
    for repeat in repeats:
        plan = build_cv_plan(ids, k=k, repeat=repeat)

        train_models_by_depth: dict[int, list[Any]] = {}
        for depth, models_by_shard in models_by_shard_by_depth.items():
            group_models = [
                sieve_fold(
                    [models_by_shard[s] for s in g], models_by_shard[g[0]].config
                )
                for g in plan.groups
            ]
            train_models_by_depth[depth] = leave_one_group_out(
                group_models, merge=merge_models
            )

        for fold, group in enumerate(plan.groups):
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])

            # build_predict_batch needs *a* fitted config; set it once from
            # the shallowest depth's own assembled model (codes are shared
            # across every depth by construction).
            batch_predictor._model = train_models_by_depth[depths[0]][fold]
            batch_predictor._config = batch_predictor._model.config
            t0 = time.perf_counter()
            batch = batch_predictor.build_predict_batch(held_out.mols)
            featurize_s = time.perf_counter() - t0

            for depth in depths:
                batch_id = cv_batch_id(
                    repeat=repeat, fold=fold, method=method, depth=depth
                )
                if _cv_run_done(runs_root, experiment, batch_id) is not None:
                    logger.info("%s already done; skipping", batch_id)
                    continue
                predictor = predictors_by_depth[depth]
                predictor._model = train_models_by_depth[depth][fold]
                predictor._config = predictor._model.config
                t0 = time.perf_counter()
                raw = predictor.predict_raw_from_batch(batch)
                predict_s = time.perf_counter() - t0
                results.append(
                    _write_cv_run(
                        experiment=experiment,
                        batch_id=batch_id,
                        method=method,
                        store=store,
                        seed=seed,
                        held_out=held_out,
                        raw=raw,
                        normalization=normalization,
                        repeat=repeat,
                        fold=fold,
                        depth=depth,
                        held_out_shards=group,
                        elapsed_s={"featurize": featurize_s, "predict": predict_s},
                        git_info=git_info,
                        runs_root=runs_root,
                    )
                )
    return results
