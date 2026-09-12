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

    Fit at the deepest depth the sweep needs, once -- not once per depth.
    Level *k*'s stored statistics do not depend on how deep the model goes
    (WL refinement never looks ahead), so every shallower depth is
    ``truncate_model`` applied to the merged result; see that function for
    the proof obligation and the merge-then-truncate ordering.

    This corrects an earlier claim here that a shallow config "is not a
    truncation of a deep one". What is true is narrower: under
    ``class_estimator="continuation"`` a shallow depth's *prediction*
    cannot be read out of a deep model's *output*, because the deepest
    level is read differently from the rest. That says nothing about the
    stored sufficient statistics, which truncation rebuilds the reading of
    -- and which are bit-identical, as the tests now pin.

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
    max_depth: int,
    codes_path: str | Path,
    config_label: str,
    predictor_params: dict[str, Any] | None = None,
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> list[Path]:
    """One fit per shard, at ``max_depth`` -- the deepest the sweep will
    ask for. Every shallower depth comes from ``truncate_model`` applied to
    the *merged* model, so this is N fits, not N per depth (see
    ``truncate_model``'s own docstring for why that is exact)."""
    return [
        fit_sieve_shard(
            store=store,
            shard=s,
            depth=max_depth,
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


def truncate_model(model: Any, depth: int) -> Any:
    """The depth-``depth`` model implied by an already-fitted deeper one.

    WL refinement is bottom-up -- level *k* is built from level *k-1* and
    never looks ahead -- so a depth-*D* fit's levels ``0..d`` hold exactly
    the sufficient statistics a native depth-*d* fit would have stored.
    Only the *predict-time reading* of them depends on the model's own
    depth (under ``class_estimator="continuation"`` the deepest level uses
    its pooled mean while every other level averages its children, and
    empirical-Bayes alpha is estimated per level against that same
    population) -- and rebuilding the config at ``max_wl_depth=depth``
    restores exactly that reading. Verified bit-for-bit against native fits
    at every depth in
    ``test_truncate_model_matches_a_native_fit_at_every_depth``.

    **Truncate after merging, never before.** ``max_wl_depth`` feeds
    ``schema_version``, so a truncated model will not merge with its
    untruncated siblings (``check_mergeable`` refuses). Merging first and
    truncating the result is both legal and cheaper: one merge then serves
    every depth, and it gives the same answer, since ``merge_models`` works
    level by level and level *k*'s merge reads only level *k*.

    Refuses a ``neighbor_depth`` config: there the level tuple is
    ``[attr][coarse WL chain][main WL_PAIR chain]`` (design.md 3.6), so the
    main chain is the *last* block and a prefix slice would cut through the
    coarse one, silently misaligning every level.
    """
    import dataclasses

    import sieve

    if model.config.neighbor_depth is not None:
        raise ValueError(
            "truncate_model does not support neighbor_depth configs: the "
            "main WL chain is the last level block, so a prefix slice would "
            "cut through the coarse chain instead of shortening the main one"
        )
    if depth > model.config.max_wl_depth:
        raise ValueError(
            f"cannot truncate a max_wl_depth={model.config.max_wl_depth} "
            f"model up to depth {depth}"
        )
    cfg = dataclasses.replace(model.config, max_wl_depth=depth)
    return sieve.SieveModel(
        cfg,
        tuple(model.levels[: cfg.n_levels]),
        model.global_count,
        model.global_mean,
        model.global_msd,
    )


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
    save_predictions: bool = False,
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
        # Off by default for CV sweeps. Each of these is ~151MB on the real
        # corpus, and across a depth sweep most of it is duplication:
        # atom_target_true, the id columns, num_atoms and molecule_value are
        # identical for every depth of a given (repeat, fold) -- only
        # atom_target_pred differs. Worth writing where per-atom predictions
        # are actually analysed (one selected depth), not for every point of
        # a curve that is read as aggregate metrics.
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
    save_predictions: bool = False,
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

    **Depths below** ``dash_depth_sweep._MIN_DERIVABLE_DEPTH`` **are
    refused.** Every depth here is derived by truncating one walk, and
    depth 1 is the one case where that is measurably wrong: DASH-tree's
    ``_get_init_layer`` redirects a hydrogen to its heavy neighbour and
    consumes a depth unit *before* ``max_depth`` is checked, so an H atom's
    depth-1 and depth-2 requests resolve to the same 2-entry path, and the
    true depth-1 path is one entry longer than truncation can produce (mae
    0.0937 vs 0.1315 on a real slice -- see that module's own note). The
    path itself does not record which entries came from the redirect, so no
    truncation length fixes it; a real depth-1 point needs its own shard
    set, which this driver does not build. Refused loudly rather than
    silently scored wrong.
    """
    from experiments.dash_depth_sweep import _MIN_DERIVABLE_DEPTH
    from experiments.predictors.dash import DASHChargePredictor

    too_shallow = sorted(d for d in depths if d < _MIN_DERIVABLE_DEPTH)
    if too_shallow:
        raise ValueError(
            f"depth(s) {too_shallow} cannot be derived by truncating a "
            f"deeper walk (see run_dash_cv's own docstring); request "
            f">= {_MIN_DERIVABLE_DEPTH}"
        )
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
                        save_predictions=save_predictions,
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
    save_predictions: bool = False,
    experiment: str = "sieve-cv",
    seed: int = 0,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stores_root: Path | None = None,
    allow_dirty: bool = False,
) -> list[RunResult]:
    """Sieve's own CV sweep, structurally the same as ``run_dash_cv``: one
    shard set, fit once at the deepest depth, serves every depth.

    Per sample the complementary shards are merged **once**, and each
    requested depth is then ``truncate_model`` applied to that one merged
    model -- exact, and the reason this needs N shard fits rather than N
    per depth. Evaluation reuses a single featurized batch across every
    depth as well (``SievePredictor.build_predict_batch``/
    ``predict_raw_from_batch``), which is valid because a ``NodeBatch``
    carries no depth information and every model here shares
    ``codes_path``'s one frozen vocabulary. DASH gets the same saving from
    the other direction: its paths are prefix-nested, so one walk serves
    every depth.

    Requires every shard already fit at ``max(depths)``
    (``run_sieve_shard_fits``); raises naming any that are missing.
    """
    import sieve
    from experiments.predictors.sieve_predictor import SievePredictor
    from sieve.merge import fold as sieve_fold
    from sieve.merge import merge_models

    method = method or f"sieve-{config_label}"
    ids = shard_ids(n_shards)
    fit_depth = max(depths)

    paths_or_none = {
        s: _shard_fit_done(runs_root, sieve_shard_batch_id(config_label, fit_depth, s))
        for s in ids
    }
    missing = [s for s, p in paths_or_none.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"no shard fit for {config_label!r} at depth {fit_depth}, shard(s) "
            f"{missing}; run run_sieve_shard_fits(max_depth={fit_depth}) first"
        )
    models_by_shard = {
        s: sieve.SieveModel.load(p) for s, p in paths_or_none.items() if p is not None
    }

    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    git_info = _check_clean(allow_dirty)

    # One predictor, reused throughout: the model it carries is swapped per
    # (fold, depth), and build_predict_batch only ever reads the config's
    # attribute_codes/edge_codes, which every model here shares by
    # construction (one frozen codes_path -- see fit_sieve_shard).
    predictor = SievePredictor(**(predictor_params or {}))

    results: list[RunResult] = []
    for repeat in repeats:
        plan = build_cv_plan(ids, k=k, repeat=repeat)

        # Merged once, at fit_depth; every requested depth is a truncation
        # of these, not a separate merge of a separate shard set.
        group_models = [
            sieve_fold([models_by_shard[s] for s in g], models_by_shard[g[0]].config)
            for g in plan.groups
        ]
        train_models = leave_one_group_out(group_models, merge=merge_models)

        for fold, group in enumerate(plan.groups):
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])

            predictor.set_model(train_models[fold])
            t0 = time.perf_counter()
            batch = predictor.build_predict_batch(held_out.mols)
            featurize_s = time.perf_counter() - t0

            for depth in depths:
                batch_id = cv_batch_id(
                    repeat=repeat, fold=fold, method=method, depth=depth
                )
                if _cv_run_done(runs_root, experiment, batch_id) is not None:
                    logger.info("%s already done; skipping", batch_id)
                    continue
                predictor.set_model(truncate_model(train_models[fold], depth))
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
                        save_predictions=save_predictions,
                    )
                )
    return results
