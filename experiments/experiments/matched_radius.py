"""Matched radius: which sphere answered each held-out atom of a CV run.

Sieve and the HOSE register both back off: an atom whose environment at the
largest radius was not seen in training is answered by a smaller one, and in
the last resort by the global mean. How often that happens, and what the error
is at each radius, is not in a run's record, since ``predictions.npz`` keeps
the predicted value alone.

Nothing is refitted and no run is rewritten. Each run's fold model is
assembled again exactly as its driver assembled it (``SieveTrainModels`` for
``run_sieve_cv``, the merged shard states for ``run_hose_cv``), the held-out
set is predicted again, and the prediction must reproduce the saved one before
the radii are written to ``matched_radius.npz`` beside the run. The error
breakdown (``experiments.breakdown``) then scores every run that has such a
file by matched radius.

One convention for both arms: ``-1`` where no class matched and the global
mean answered, otherwise the radius of the class that did. For Sieve the
radius is the level of the matched class, read through the model's backoff
path, so that level 0 is the atom's own attributes. The HOSE register has no
such level: its smallest key is the one-sphere code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.breakdown import Arm, arm_runs

MATCHED_FILE = "matched_radius.npz"
UNMATCHED = -1
# The saved prediction must be reproduced to this absolute difference.
TOLERANCE = 1e-12


def sieve_matched_radius(model: Any, prediction: Any) -> np.ndarray:
    """The level that answered each atom of a ``sieve.predict_detailed`` result.

    ``prediction.matched_level`` is a position on the backoff path, which is
    mapped to the level itself.
    """
    backoff = np.asarray(model.config.backoff_path, np.int64)
    position = np.asarray(prediction.matched_level, np.int64)
    level = np.where(position >= 0, backoff[np.maximum(position, 0)], UNMATCHED)
    return level.astype(np.int8)


def hose_matched_radius(matched: np.ndarray) -> np.ndarray:
    """``HoseLookupPredictor.matched_radius`` under this module's convention:
    its ``0``, the global mean, becomes ``UNMATCHED``."""
    matched = np.asarray(matched, np.int64)
    return np.where(matched > 0, matched, UNMATCHED).astype(np.int8)


def write_matched(run: Path, radius: np.ndarray) -> None:
    np.savez_compressed(Path(run) / MATCHED_FILE, matched_radius=radius)


def read_matched(run: Path) -> np.ndarray | None:
    """The run's matched radii, or ``None`` when they were never recorded."""
    path = Path(run) / MATCHED_FILE
    if not path.exists():
        return None
    return np.asarray(np.load(path)["matched_radius"], np.int8)


def _cv(run: Path) -> dict[str, Any]:
    return (
        json.loads((run / "manifest.json").read_text()).get("config", {}).get("cv", {})
    )


def _stale(run: Path) -> bool:
    record, pred = run / MATCHED_FILE, run / "predictions.npz"
    return not record.exists() or record.stat().st_mtime < pred.stat().st_mtime


def pending(
    runs_root: Path,
    arms: Sequence[Arm],
    *,
    repeats: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Runs of ``arms`` whose radii are absent or older than their predictions,
    in (repeat, fold) order within each arm; every selected run with ``force``.
    A run without saved predictions is refused: its radii could not be checked.
    """
    out = []
    for arm in arms:
        for (repeat, fold), run in sorted(arm_runs(Path(runs_root), arm).items()):
            if repeats is not None and repeat not in set(repeats):
                continue
            if folds is not None and fold not in set(folds):
                continue
            if not (run / "predictions.npz").exists():
                raise FileNotFoundError(
                    f"{run} has no predictions.npz; rerun it with --save-predictions"
                )
            if force or _stale(run):
                out.append(run)
    return out


def _check(run: Path, held_out: Any, value: np.ndarray) -> None:
    """Refuse a recomputed prediction that is not the one the run saved."""
    z = np.load(run / "predictions.npz", allow_pickle=True)
    shared = [c for c in held_out.ids if c.endswith("_id") and c in z.files]
    same = bool(shared) and all(
        np.array_equal(np.asarray(held_out.ids[c], dtype=str), z[c].astype(str))
        for c in shared
    )
    if not same:
        raise ValueError(f"{run}: the held-out order differs from predictions.npz")
    saved = np.asarray(z["atom_target_pred"], np.float64).ravel()
    if saved.shape != value.shape or np.max(np.abs(saved - value)) > TOLERANCE:
        raise ValueError(f"{run}: the fold model does not reproduce its predictions")


def _held_out(run: Path, plan: Any, mset_by_shard: dict[str, Any]) -> Any:
    from experiments.cv import concat_molecule_sets

    cv = _cv(run)
    group = plan.groups[int(cv["fold"])]
    if cv.get("held_out_shards") != ",".join(group):
        raise ValueError(f"{run}: held_out_shards disagrees with the CV plan")
    return concat_molecule_sets([mset_by_shard[s] for s in group])


def _by_sample(runs: Sequence[Path]) -> dict[int, dict[int, list[Path]]]:
    out: dict[int, dict[int, list[Path]]] = {}
    for run in runs:
        cv = _cv(run)
        out.setdefault(int(cv["repeat"]), {}).setdefault(int(cv["fold"]), []).append(
            run
        )
    return out


def _held_out_shards(
    samples: dict[int, dict[int, list[Path]]], ids: Sequence[str], k: int
) -> list[str]:
    from experiments.cv import build_cv_plan

    needed: set[str] = set()
    for repeat, folds in samples.items():
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        for fold in folds:
            needed.update(plan.groups[fold])
    return sorted(needed)


def record_sieve_runs(
    runs_root: Path,
    arms: Sequence[Arm],
    *,
    store: str,
    n_shards: int,
    k: int,
    config_label: str,
    fit_depth: int,
    variants: Sequence[Any] = (),
    model_cache: str | Path | None = None,
    stores_root: Path | None = None,
    repeats: Sequence[int] | None = None,
    force: bool = False,
    n_jobs: int | None = None,
) -> list[Path]:
    """Write the matched radii of every Sieve run of ``arms`` that lacks them.

    ``variants`` are the ``ModelVariant`` readings ``run_sieve_cv`` scored,
    matched to an arm by method name; an arm that names none is read as
    fitted. The held-out set is featurized once per fold and serves every arm.
    """
    import sieve
    from experiments.cv import (
        SieveTrainModels,
        build_cv_plan,
        load_shards,
        respecify_model,
        shard_ids,
        truncate_model,
    )
    from sieve.io.rdkit_adapter import from_rdkit

    by_method = {v.method: v for v in variants}
    if len(by_method) != len(variants):
        raise ValueError(
            f"a variant is named twice: {sorted(v.method for v in variants)}"
        )
    todo = pending(runs_root, arms, repeats=repeats, force=force)
    if not todo:
        return []
    ids = shard_ids(n_shards)
    samples = _by_sample(todo)
    train_models_for = SieveTrainModels(
        store=store,
        n_shards=n_shards,
        k=k,
        config_label=config_label,
        fit_depth=fit_depth,
        model_cache=model_cache,
        runs_root=Path(runs_root),
        stores_root=stores_root,
    )
    mset_by_shard = load_shards(
        store, _held_out_shards(samples, ids, k), stores_root=stores_root
    )
    written = []
    for repeat, folds in sorted(samples.items()):
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        models = train_models_for(repeat, plan)
        for fold, runs in sorted(folds.items()):
            held_out = _held_out(runs[0], plan, mset_by_shard)
            batch = from_rdkit(held_out.mols, config=models[fold].config, n_jobs=n_jobs)
            for run in runs:
                cv = _cv(run)
                model = truncate_model(models[fold], int(cv["depth"]))
                variant = by_method.get(str(cv["method"]))
                if variant is not None:
                    model = respecify_model(model, variant)
                prediction = sieve.predict_detailed(model, batch)
                _check(run, held_out, np.asarray(prediction.value)[:, 0])
                write_matched(run, sieve_matched_radius(model, prediction))
                written.append(run)
    return written


def record_hose_runs(
    runs_root: Path,
    arm: Arm,
    *,
    store: str,
    n_shards: int,
    k: int,
    numbering: str = "stored",
    keys: str = "code",
    stores_root: Path | None = None,
    repeats: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Write the matched radii of every run of the HOSE ``arm`` that lacks them.

    ``numbering`` is the atom numbering the arm's shard fits were generated
    under (``predictors.hose_keys.NUMBERINGS``); a run does not record it, and
    the fold model of the other numbering would not reproduce its predictions.
    ``keys`` is what their keys were cut from (``predictors.hose_keys.
    KEY_KINDS``), for the same reason.

    The cost is the generation of the held-out codes, about as long as the run
    itself took, and nothing is shared between folds. ``repeats`` and ``folds``
    restrict the call to those samples, so that a dispatch can give each
    (repeat, fold) its own process, as ``run_hose_cv``'s does.
    """
    from experiments.cv import (
        _shard_fit_done,
        build_cv_plan,
        hose_shard_batch_id,
        leave_one_group_out,
        load_shards,
        shard_ids,
    )
    from experiments.hose_artifact import (
        fold_hose_states,
        load_hose_state,
        merge_hose_states,
    )
    from experiments.predictors.hose import HoseLookupPredictor

    runs_root = Path(runs_root)
    todo = pending(runs_root, [arm], repeats=repeats, folds=folds, force=force)
    if not todo:
        return []
    ids = shard_ids(n_shards)
    radius = arm.depth
    paths = {
        s: _shard_fit_done(runs_root, hose_shard_batch_id(radius, s, numbering, keys))
        for s in ids
    }
    missing = [s for s, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"no hose shard fit at radius {radius} under the {numbering} "
            f"numbering with {keys} keys for {missing}"
        )
    states = {s: load_hose_state(p) for s, p in paths.items() if p is not None}
    samples = _by_sample(todo)
    mset_by_shard = load_shards(
        store, _held_out_shards(samples, ids, k), stores_root=stores_root
    )
    written = []
    for repeat, by_fold in sorted(samples.items()):
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        train_states = leave_one_group_out(
            [fold_hose_states(states[s] for s in g) for g in plan.groups],
            merge=merge_hose_states,
        )
        for fold, runs in sorted(by_fold.items()):
            (run,) = runs
            held_out = _held_out(run, plan, mset_by_shard)
            predictor = HoseLookupPredictor(max_radius=radius)
            predictor.set_model_state(train_states[fold])
            value = np.asarray(predictor.predict(held_out).atom_value, np.float64)
            _check(run, held_out, value)
            write_matched(run, hose_matched_radius(predictor.matched_radius))
            written.append(run)
    return written
