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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
# Optional cache of assembled CV training models
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CACHE = REPO_ROOT / "experiments" / "results" / "cv-model-cache"


def cv_model_cache_dir(cache_root: str | Path, predictor: str, key: str) -> Path:
    """Where one (predictor, configuration) family's assembled models live.

    ``key`` must name everything outside ``(repeat, fold)`` that changes the
    model -- shard count, fold count, and for Sieve the config label and fit
    depth -- so two families can never collide in one cache root.
    """
    return Path(cache_root) / predictor / key


def _cache_entry(cache_dir: Path, repeat: int, fold: int) -> tuple[Path, Path]:
    stem = cache_dir / f"r{repeat}-f{fold}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def _write_cache_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _check_cache_sidecar(path: Path, expected: dict[str, Any]) -> None:
    """Refuse a cache entry that does not describe the model being asked for.

    Rebuilding silently would be worse than failing: a mismatch means either
    the partition rule changed (so every cached model is now mislabelled) or
    two incompatible studies are sharing one cache root, and in both cases
    overwriting destroys the other party's work. The message says how to
    recover, which is simply to delete the directory.
    """
    try:
        found = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable cache sidecar {path}: {exc}") from exc
    bad = {k: (v, found.get(k)) for k, v in expected.items() if found.get(k) != v}
    if bad:
        raise RuntimeError(
            f"cached model {path.with_suffix('.npz')} does not match what was "
            f"asked for (expected vs cached: {bad}); the partition rule or the "
            "shard fits have changed. Delete the cache directory to rebuild."
        )


def _load_cached_train_models(
    cache_dir: Path,
    *,
    repeat: int,
    plan: CVPlan,
    load_one: Callable[[Path], T],
    sidecar_extra: dict[str, Any],
) -> list[T] | None:
    """Every fold of ``repeat``, or ``None`` if any one is absent.

    All-or-nothing on purpose: a partially cached repeat still has to load
    the shards and merge, and at that point reusing the few cached folds
    saves a merge each but risks mixing entries written by different code.
    """
    entries = [_cache_entry(cache_dir, repeat, f) for f in range(len(plan.groups))]
    if not all(npz.exists() and side.exists() for npz, side in entries):
        return None
    out: list[T] = []
    for fold, (npz, side) in enumerate(entries):
        _check_cache_sidecar(
            side, {**sidecar_extra, "train_shards": _train_shards(plan, fold)}
        )
        out.append(load_one(npz))
    logger.info(
        "repeat %d: loaded %d training models from %s", repeat, len(out), cache_dir
    )
    return out


def _save_train_models(
    cache_dir: Path,
    *,
    repeat: int,
    plan: CVPlan,
    models: Sequence[T],
    save_one: Callable[[T, Path], None],
    sidecar_extra: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fold, model in enumerate(models):
        npz, side = _cache_entry(cache_dir, repeat, fold)
        save_one(model, npz)
        # Sidecar last: its presence is what marks the entry complete, so an
        # interrupted save leaves a miss rather than a truncated hit.
        _write_cache_sidecar(
            side, {**sidecar_extra, "train_shards": _train_shards(plan, fold)}
        )
    logger.info(
        "repeat %d: cached %d training models in %s", repeat, len(models), cache_dir
    )


def _train_shards(plan: CVPlan, fold: int) -> list[str]:
    return sorted(s for g, group in enumerate(plan.groups) if g != fold for s in group)


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


@dataclass(frozen=True)
class ModelVariant:
    """One reading of an already-fitted model: a name, plus whatever
    ``SieveModel.with_params`` accepts.

    The variant axis is exactly ``with_params``' own allow-list --
    ``class_estimator``, ``shrinkage_weight``, ``shrinkage_strength``,
    ``minimum_support`` and ``chunk_size`` -- because that list is the set of
    fields ``SieveConfig.schema_version`` deliberately excludes: they are
    "read at prediction time and do not invalidate fitted statistics". So any
    model reachable by ``with_params`` is a variant, and a whole family can be
    compared off **one** set of shard fits: no refit, no re-merge, not even a
    re-featurization, since the variants share the eval batch too.

    ``params`` is a *patch* over the fitted config, matching ``with_params``'
    own semantics: a field left out keeps the value it was fitted with. So a
    variant meaning "do not shrink" must say ``shrinkage_weight: None``
    explicitly when the fit itself shrank.

    ``chunk_size`` is accepted but is a performance knob, not a model
    difference -- two variants differing only in it produce identical
    predictions and merely duplicate runs.
    """

    method: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> ModelVariant:
        """``{"method": ..., <with_params kwargs>}`` -- the CLI/JSON shape.

        Flat rather than nested so a variant reads as what it is: a method
        name and the inference params that define it.
        """
        rest = dict(spec)
        try:
            method = rest.pop("method")
        except KeyError:
            raise ValueError(f"variant spec has no 'method': {spec}") from None
        return cls(method=method, params=rest)


def respecify_model(model: Any, variant: ModelVariant) -> Any:
    """``model`` read under ``variant``'s params, sharing its arrays.

    Thin over ``SieveModel.with_params``, which owns the allow-list and
    rejects anything that would need a refit. The one thing added here is the
    ``schema_version`` assertion: ``with_params`` checks field *names*, and
    this checks the consequence those names are chosen for, so a field
    migrating into the digest without the allow-list being updated fails
    loudly here rather than quietly comparing models fitted to different
    vocabularies.
    """
    out = model.with_params(**dict(variant.params))
    if out.config.schema_version != model.config.schema_version:
        raise AssertionError(
            f"variant {variant.method!r} changed schema_version -- "
            f"{sorted(variant.params)} is no longer excluded from the digest, "
            "so these variants can no longer share one set of shard fits"
        )
    return out


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


# Counts inside the train family are renamed off ``_score``'s own
# ``n_test_*`` spelling: "train/n_test_conformers" reads as a contradiction,
# and the number is simply how much was scored in the training pass.
_TRAIN_RENAMES = {"n_test_atoms": "n_atoms", "n_test_conformers": "n_conformers"}


def _score_train(raw: RawPrediction, mset: MoleculeSet) -> dict[str, float]:
    """The training-set score, under a ``train/`` prefix.

    Deliberately unnormalized, unlike the held-out score. A normalizer is a
    deployment-time transform applied to predictions on unseen molecules;
    the train curve exists to show the fit's own optimism, and putting a
    redistribution step in front of it would measure the normalizer instead.
    """
    scored = _score(mset, Prediction(atom_value=raw.atom_value))
    return {f"train/{_TRAIN_RENAMES.get(k, k)}": v for k, v in scored.items()}


def _other_groups(plan: Any, fold: int) -> list[list[str]]:
    """Every group except ``fold``'s -- that fold's training shards."""
    return [g for i, g in enumerate(plan.groups) if i != fold]


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
    train_set: MoleculeSet | None = None,
    train_raw: RawPrediction | None = None,
    repeat: int,
    fold: int,
    depth: int,
    held_out_shards: list[str],
    elapsed_s: dict[str, float],
    git_info: dict[str, Any],
    runs_root: Path,
    save_predictions: bool = False,
    variant: ModelVariant | None = None,
) -> RunResult:
    """Write one CV sample's run directory. The ``manifest["config"]`` shape
    mirrors what ``config.to_dict``/``runner`` write (``predictor.name``,
    ``run.batch_id``/``run.tags``, plus a ``cv`` block of this driver's own
    fields) purely so ``aggregate.read_runs_from_dirs``'s existing
    ``flatten_params`` -- and therefore ``summarize``/``sweep`` -- read a CV
    run exactly like an ordinary one, with no changes to either."""
    started = datetime.now(UTC)
    run_metrics = _score_raw_and_normalized(raw, held_out, normalization=normalization)
    if train_raw is not None and train_set is not None:
        run_metrics.update(_score_train(train_raw, train_set))
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
    if variant is not None:
        # The params are what distinguish these runs, and a method name is only
        # a label for them; record them so a run cannot be misread if a label
        # is ever reused. Stringified because flatten_params wants scalars and
        # None is a meaningful value here, not an absence.
        config["cv"].update(
            {f"param/{k}": str(v) for k, v in sorted(variant.params.items())}
        )
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
    model_cache: str | Path | None = None,
    k: int = 5,
    max_depth: int,
    normalization: str = "std_weighted",
    method: str = "dash",
    save_predictions: bool = False,
    score_train: bool = False,
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
    ``max_depth``); raises naming any that are missing.

    ``model_cache`` persists the assembled training models under
    ``<cache>/dash/n<N>-k<k>/r<repeat>-f<fold>.npz`` and reuses them on a
    later call, skipping both the shard load and the merge. DASH node stats
    are depth-invariant, so the key carries no depth and one cached repeat
    serves every depth. Off by default -- it trades disk for time.

    One tree load, one
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
        save_node_stats,
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
    # Lazy for the same reason as Sieve's: a fully cached repeat need not open
    # a single shard. Node stats are depth-invariant, so the cache key carries
    # no depth -- one cached repeat serves every depth this driver scores.
    _stats_by_shard: dict[str, Any] = {}

    def stats_by_shard() -> dict[str, Any]:
        if not _stats_by_shard:
            _stats_by_shard.update(
                {s: load_node_stats(p) for s, p in shard_paths.items()}
            )
        return _stats_by_shard

    cache_dir = (
        None
        if model_cache is None
        else cv_model_cache_dir(model_cache, "dash", f"n{n_shards}-k{k}")
    )

    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    git_info = _check_clean(allow_dirty)

    predictor = DASHChargePredictor(max_depth=max_depth)
    predictor._load_tree()  # loaded once, reused across every repeat/fold

    results: list[RunResult] = []
    for repeat in repeats:
        plan = build_cv_plan(ids, k=k, repeat=repeat)

        train_stats = None
        if cache_dir is not None:
            train_stats = _load_cached_train_models(
                cache_dir,
                repeat=repeat,
                plan=plan,
                load_one=load_node_stats,
                sidecar_extra={},
            )
        if train_stats is None:
            shards = stats_by_shard()
            group_stats = [fold_node_stats(shards[s] for s in g) for g in plan.groups]
            train_stats = leave_one_group_out(group_stats, merge=merge_node_stats)
            if cache_dir is not None:
                _save_train_models(
                    cache_dir,
                    repeat=repeat,
                    plan=plan,
                    models=train_stats,
                    save_one=save_node_stats,
                    sidecar_extra={},
                )

        for fold, group in enumerate(plan.groups):
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])

            t0 = time.perf_counter()
            paths = predictor.match_paths(held_out, split="cv")
            walk_s = time.perf_counter() - t0

            # The training molecules are already in mset_by_shard, so the
            # train pass costs a walk and a predict, never a reload. Walked
            # once per fold like the held-out set: every depth is a
            # truncation of the same matched paths.
            train_set = train_paths = None
            if score_train:
                train_set = concat_molecule_sets(
                    [mset_by_shard[s] for g in _other_groups(plan, fold) for s in g]
                )
                t0 = time.perf_counter()
                train_paths = predictor.match_paths(train_set, split="cv")
                walk_s += time.perf_counter() - t0

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
                train_raw = (
                    predictor.predict_raw_at_depth(train_paths, max_depth=depth)
                    if train_paths is not None
                    else None
                )
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
                        train_set=train_set,
                        train_raw=train_raw,
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
    fit_depth: int | None = None,
    predictor_params: dict[str, Any] | None = None,
    variants: Sequence[ModelVariant] | None = None,
    model_cache: str | Path | None = None,
    k: int = 5,
    normalization: str = "equal_weighted",
    method: str | None = None,
    save_predictions: bool = False,
    score_train: bool = False,
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

    ``fit_depth`` is the depth the shards on disk were actually fit at,
    which need not equal ``max(depths)``: a Study-A sweep fits once at the
    deepest depth it sweeps, and a later study asking for one shallower
    depth truncates those same fits rather than refitting. It defaults to
    ``max(depths)`` and must not be smaller. Raises naming any shard whose
    fit is missing.

    ``variants`` extends that same reuse sideways. Anything
    ``SieveModel.with_params`` accepts is a *reading* of one fit rather than
    a model to fit -- estimator, shrinkage rule and strength, minimum support
    -- so each variant is a ``respecify_model`` of the already merged, already
    truncated model, scored against the already featurized batch, and written
    under its own ``ModelVariant.method``. Omitted,
    the model is scored as fitted, under ``method``.

    ``model_cache`` persists the assembled training models under
    ``<cache>/sieve/<config>-w<fit_depth>-n<N>-k<k>/r<repeat>-f<fold>.npz``
    and reuses them on a later call. Assembling one repeat's k models costs
    ~123 s and 30 GB of peak RSS at N=50 (load 2.6 s, fold into k groups
    27.3 s, leave-one-group-out 92.9 s), all of which a hit skips -- the
    shard fits are not even opened. Models are cached **untruncated**, at
    ``fit_depth``, so one cached repeat serves every depth and every variant.
    Off by default: this trades a large amount of disk for that time, and
    ``config.py`` records a previous 21 GB incident from persisting more than
    was needed.
    """
    import sieve
    from experiments.predictors.sieve_predictor import SievePredictor
    from sieve.merge import fold as sieve_fold
    from sieve.merge import merge_models

    method = method or f"sieve-{config_label}"
    variant_list: list[ModelVariant | None] = list(variants) if variants else [None]
    names = [v.method for v in variant_list if v is not None]
    if len(set(names)) != len(names):
        raise ValueError(f"variants must have distinct method names, got {names}")
    ids = shard_ids(n_shards)
    if fit_depth is None:
        fit_depth = max(depths)
    elif fit_depth < max(depths):
        raise ValueError(
            f"fit_depth={fit_depth} is shallower than the deepest requested depth "
            f"{max(depths)}; truncation can only remove levels"
        )

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
    shard_paths = {s: p for s, p in paths_or_none.items() if p is not None}
    # Loaded lazily: a fully cached repeat never needs the shards at all, and
    # opening 50 of them is the first 2.6 s of the 123 s this cache exists to
    # avoid. One is still read eagerly below, to pin schema_version.
    _models_by_shard: dict[str, Any] = {}

    def models_by_shard() -> dict[str, Any]:
        if not _models_by_shard:
            _models_by_shard.update(
                {s: sieve.SieveModel.load(p) for s, p in shard_paths.items()}
            )
        return _models_by_shard

    cache_dir = None
    sidecar: dict[str, Any] = {}
    if model_cache is not None:
        reference = sieve.SieveModel.load(shard_paths[ids[0]])
        cache_dir = cv_model_cache_dir(
            model_cache, "sieve", f"{config_label}-w{fit_depth}-n{n_shards}-k{k}"
        )
        # schema_version pins the vocabulary and depth the fits were built
        # with; a cached model that disagrees is not the same model.
        sidecar = {"schema_version": reference.config.schema_version}

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

        train_models = None
        if cache_dir is not None:
            train_models = _load_cached_train_models(
                cache_dir,
                repeat=repeat,
                plan=plan,
                load_one=sieve.SieveModel.load,
                sidecar_extra=sidecar,
            )
        if train_models is None:
            # Merged once, at fit_depth; every requested depth is a truncation
            # of these, not a separate merge of a separate shard set.
            shards = models_by_shard()
            group_models = [
                sieve_fold([shards[s] for s in g], shards[g[0]].config)
                for g in plan.groups
            ]
            train_models = leave_one_group_out(group_models, merge=merge_models)
            if cache_dir is not None:
                _save_train_models(
                    cache_dir,
                    repeat=repeat,
                    plan=plan,
                    models=train_models,
                    save_one=lambda m, path: m.save(path),
                    sidecar_extra=sidecar,
                )

        for fold, group in enumerate(plan.groups):
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])

            predictor.set_model(train_models[fold])
            t0 = time.perf_counter()
            batch = predictor.build_predict_batch(held_out.mols)
            featurize_s = time.perf_counter() - t0

            # Featurized once per fold, like the held-out batch: the batch is
            # independent of depth and of variant, so every depth and every
            # variant reads this one.
            train_set = train_batch = None
            if score_train:
                train_set = concat_molecule_sets(
                    [mset_by_shard[s] for g in _other_groups(plan, fold) for s in g]
                )
                t0 = time.perf_counter()
                train_batch = predictor.build_predict_batch(train_set.mols)
                featurize_s += time.perf_counter() - t0

            for depth in depths:
                # Built once per depth and only if some variant still needs
                # it, so a fully resumed (repeat, fold) costs no truncation.
                truncated: Any = None
                for variant in variant_list:
                    vmethod = method if variant is None else variant.method
                    batch_id = cv_batch_id(
                        repeat=repeat, fold=fold, method=vmethod, depth=depth
                    )
                    if _cv_run_done(runs_root, experiment, batch_id) is not None:
                        logger.info("%s already done; skipping", batch_id)
                        continue
                    if truncated is None:
                        truncated = truncate_model(train_models[fold], depth)
                    predictor.set_model(
                        truncated
                        if variant is None
                        else respecify_model(truncated, variant)
                    )
                    t0 = time.perf_counter()
                    raw = predictor.predict_raw_from_batch(batch)
                    train_raw = (
                        predictor.predict_raw_from_batch(train_batch)
                        if train_batch is not None
                        else None
                    )
                    predict_s = time.perf_counter() - t0
                    results.append(
                        _write_cv_run(
                            experiment=experiment,
                            batch_id=batch_id,
                            method=vmethod,
                            store=store,
                            seed=seed,
                            held_out=held_out,
                            raw=raw,
                            normalization=normalization,
                            train_set=train_set,
                            train_raw=train_raw,
                            repeat=repeat,
                            fold=fold,
                            depth=depth,
                            held_out_shards=group,
                            elapsed_s={
                                "featurize": featurize_s,
                                "predict": predict_s,
                            },
                            git_info=git_info,
                            runs_root=runs_root,
                            save_predictions=save_predictions,
                            variant=variant,
                        )
                    )
    return results
