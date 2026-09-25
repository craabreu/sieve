"""Study F: calibration of Sieve's predictive variance.

Spec: docs/superpowers/specs/2026-09-25-study-f-calibration-design.md.

The shipped predictive variance (form B, within-structure-variance spec) and
four ablations, each dropping one ingredient, are scored on the Study B
incumbent's runs: calibration (NLL, E[z^2], coverage, E[z^2] by matched
radius, within-conformer ranking) and the variance-weighted normalisation they
drive. Every arm is a per-atom variance from one prediction, so all five are
scored on the same atoms. Scores go to a sidecar beside each run's own
metrics, like Study D's subset scores (``stereo_subsets``), so a run's record
is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

METRICS_FILE = "calibration_metrics.json"
PREFIX = "calibration"
COVERAGE = (0.5, 0.9, 0.95, 0.99)
ID_COLUMNS = ("chembl_id", "dash_id", "conf_id")


@dataclass(frozen=True)
class Arm:
    """One predictive variance: which terms, at which constants."""

    name: str
    alpha_v: float
    selection_weight: float
    estimation: bool
    within_structure: bool


ARMS = (
    Arm("form_b", 10.0, 0.5, True, True),
    Arm("no_sigma2_w", 10.0, 0.5, True, False),
    Arm("alpha_v_30", 30.0, 0.5, True, True),
    Arm("no_selection", 10.0, 0.0, True, True),
    Arm("no_estimation", 10.0, 0.5, False, True),
)


def arm_variances(model: Any, prediction: Any) -> dict[str, np.ndarray]:
    """Per-atom sigma^2 of every arm, ``(n,)`` each, for a ``d == 1`` model.

    ``prediction.matched_level`` is a backoff-path position, while the
    variance tables are indexed by raw level, so positions are mapped through
    ``config.backoff_path`` first. Unmatched atoms take ``global_msd``, plus
    sigma2_w in the arms that carry it -- ``predict``'s own fallback.
    """
    from sieve.uncertainty import variance_terms

    backoff = np.asarray(model.config.backoff_path, dtype=np.int64)
    pos = np.asarray(prediction.matched_level)
    level = np.where(pos >= 0, backoff[np.maximum(pos, 0)], -1)
    cid = np.asarray(prediction.class_id)
    s2w = float(model.within_variance[0])
    gmsd = float(model.global_msd[0])
    terms: dict[float, list[dict[str, np.ndarray]]] = {}
    out: dict[str, np.ndarray] = {}
    for arm in ARMS:
        if arm.alpha_v not in terms:
            terms[arm.alpha_v] = variance_terms(model, alpha_v=arm.alpha_v)
        s2 = np.full(pos.shape[0], gmsd + (s2w if arm.within_structure else 0.0))
        for k in np.unique(level[level >= 0]):
            t = terms[arm.alpha_v][int(k)]
            # the same order as predictive_variance's sum, so form_b is
            # bit-identical to it
            table = t["within"] + arm.selection_weight * t["selection"]
            if arm.estimation:
                table = table + t["estimation"]
            if arm.within_structure:
                table = table + t["within_structure"]
            sel = level == k
            s2[sel] = table[cid[sel], 0]
        out[arm.name] = s2
    return out


def group_percentile_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each value's average rank within its group, divided by the group's size
    (pandas' ``groupby().rank(pct=True)``, without pandas)."""
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    n = values.shape[0]
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    size = np.diff(np.r_[start, n])
    pos = np.arange(n) - np.repeat(start, size)
    new_run = np.r_[True, (np.diff(g) != 0) | (np.diff(v) != 0)]
    run_id = np.cumsum(new_run) - 1
    first = np.flatnonzero(new_run)
    run_len = np.diff(np.r_[first, n])
    avg_rank = pos[first] + (run_len - 1) / 2.0 + 1.0
    out = np.empty(n)
    out[order] = avg_rank[run_id] / np.repeat(size, size)
    return out


def normalised_errors(
    normalised: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    d = np.asarray(normalised) - np.asarray(target)
    return float(np.sqrt(np.mean(d * d))), float(np.mean(np.abs(d)))


def calibration_metrics(
    *,
    raw: np.ndarray,
    target: np.ndarray,
    sigma2: np.ndarray,
    k_star: np.ndarray,
    conf: np.ndarray,
    molecule_value: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """One arm's calibration and normalisation scores (spec section 2.3).

    ``conf`` is each atom's conformer index, ``0..n_conformers-1``, and
    ``molecule_value`` each conformer's constrained total.
    """
    from scipy.stats import norm

    from experiments.normalize import variance_weighted_normalize

    e = np.asarray(raw) - np.asarray(target)
    z2 = e * e / sigma2
    out = {
        "nll": float(np.mean(0.5 * (np.log(2 * np.pi * sigma2) + z2))),
        "ez2": float(np.mean(z2)),
    }
    for c in COVERAGE:
        out[f"cov{round(100 * c)}"] = float(np.mean(z2 < norm.ppf(0.5 + c / 2) ** 2))
    for k in range(depth + 1):
        sel = k_star == k
        if sel.any():
            out[f"ez2_k{k}"] = float(np.mean(z2[sel]))
    ra = group_percentile_rank(np.abs(e), conf)
    rs = group_percentile_rank(sigma2, conf)
    out["rho"] = float(np.corrcoef(ra, rs)[0, 1])
    n_conf = int(np.asarray(molecule_value).shape[0])
    x = variance_weighted_normalize(raw, np.sqrt(sigma2), molecule_value, conf, n_conf)
    out["norm_rmse"], out["norm_mae"] = normalised_errors(x, target)
    return out


# ---------------------------------------------------------------- runs ---


def _cv(run) -> dict[str, Any]:
    import json

    return (
        json.loads((run / "manifest.json").read_text()).get("config", {}).get("cv", {})
    )


def selected_runs(
    runs_root, *, experiment: str, method: str, depth: int, repeats=None
) -> list:
    """Run directories of ``method`` at ``depth`` under ``experiment``, by
    (repeat, fold); only ``repeats`` when given."""
    from pathlib import Path

    found = []
    for manifest in sorted(Path(runs_root).glob(f"{experiment}/*/manifest.json")):
        cv = _cv(manifest.parent)
        if cv.get("method") != method or int(cv.get("depth", -1)) != depth:
            continue
        if repeats is not None and int(cv["repeat"]) not in set(repeats):
            continue
        found.append((int(cv["repeat"]), int(cv["fold"]), manifest.parent))
    return [run for *_, run in sorted(found)]


def _complete(side: dict[str, Any]) -> bool:
    return all(f"{PREFIX}/{arm.name}/nll" in side for arm in ARMS) and (
        f"{PREFIX}/equal/norm_rmse" in side
    )


def missing_scores(runs_root, *, experiment, method, depth, repeats=None) -> list:
    """Selected runs whose sidecar is absent, older than their predictions, or
    missing an arm. A run with no predictions is reported too: it can never
    be scored, and passing over it would silently drop a paired sample."""
    import json

    stale = []
    for run in selected_runs(
        runs_root, experiment=experiment, method=method, depth=depth, repeats=repeats
    ):
        pred, side = run / "predictions.npz", run / METRICS_FILE
        if not pred.exists() or not side.exists():
            stale.append(run)
        elif side.stat().st_mtime < pred.stat().st_mtime:
            stale.append(run)
        elif not _complete(json.loads(side.read_text())):
            stale.append(run)
    return stale


def score_run(
    run, model: Any, held_out: Any, *, n_jobs: int | None = None
) -> dict[str, float]:
    """One run's sidecar: ``model`` is the run's untruncated training model
    with its training floor attached, ``held_out`` the run's held-out set in
    the run's own order."""
    from dataclasses import replace

    import sieve
    from experiments.cv import truncate_model
    from experiments.normalize import equal_weighted_normalize
    from sieve.io.rdkit_adapter import from_rdkit

    cv = _cv(run)
    depth = int(cv["depth"])
    z = np.load(run / "predictions.npz", allow_pickle=True)
    # Every conformer identifier both sides carry must line up, row for row;
    # (dash_id, conf_id) is the key on the real store, and chembl_id is
    # compared too wherever it is saved.
    shared = [c for c in ID_COLUMNS if c in held_out.ids and c in z.files]
    same = bool(shared) and all(
        np.array_equal(np.asarray(held_out.ids[c], dtype=str), z[c].astype(str))
        for c in shared
    )
    if not same:
        raise ValueError(f"{run}: the held-out order differs from predictions.npz")

    m = truncate_model(model, depth)
    params = {k.split("/", 1)[1]: v for k, v in cv.items() if k.startswith("param/")}
    if params:
        m = m.with_params(**params)
    m = replace(m, config=replace(m.config, predictive_variance=True))
    p = sieve.predict_detailed(
        m, from_rdkit(held_out.mols, config=m.config, n_jobs=n_jobs)
    )

    raw = p.value[:, 0]
    if np.max(np.abs(raw - z["atom_target_pred"].ravel())) > 1e-12:
        raise ValueError(f"{run}: the fold model does not reproduce its predictions")

    target = z["atom_target_true"].ravel()
    num_atoms = np.asarray(z["num_atoms"], dtype=np.int64)
    conf = np.repeat(np.arange(num_atoms.size), num_atoms)
    molecule_value = np.asarray(z["molecule_value"], dtype=np.float64)
    k_star = np.asarray(p.matched_level)

    out: dict[str, float] = {
        f"{PREFIX}/n_atoms": float(raw.size),
        f"{PREFIX}/sigma2_w": float(m.within_variance[0]),
    }
    for k in range(-1, depth + 1):
        share = float(np.mean(k_star == k))
        if share:
            out[f"{PREFIX}/share_k{k}"] = share
    for name, s2 in arm_variances(m, p).items():
        scores = calibration_metrics(
            raw=raw,
            target=target,
            sigma2=s2,
            k_star=k_star,
            conf=conf,
            molecule_value=molecule_value,
            depth=depth,
        )
        out.update({f"{PREFIX}/{name}/{key}": v for key, v in scores.items()})
    x = equal_weighted_normalize(
        raw, np.ones_like(raw), molecule_value, conf, num_atoms.size
    )
    out[f"{PREFIX}/equal/norm_rmse"], out[f"{PREFIX}/equal/norm_mae"] = (
        normalised_errors(x, target)
    )
    return out


def score_runs(
    runs_root,
    *,
    store: str,
    experiment: str,
    method: str,
    depth: int,
    n_shards: int,
    k: int,
    config_label: str,
    fit_depth: int,
    collapse: bool,
    model_cache=None,
    stores_root=None,
    repeats=None,
    force: bool = False,
    n_jobs: int | None = None,
) -> list:
    """Write the sidecar of every selected run that needs one, and return
    those runs. Fold models come from ``SieveTrainModels``, the same assembly
    ``run_sieve_cv`` scored the runs with; training floors are attached as
    ``run_sieve_cv`` does (``collapse`` must be the CV's own)."""
    import json

    from experiments.cv import (
        SieveTrainModels,
        _attach_training_floors,
        build_cv_plan,
        concat_molecule_sets,
        load_shards,
        shard_ids,
    )

    sel = {
        "experiment": experiment,
        "method": method,
        "depth": depth,
        "repeats": repeats,
    }
    todo = (
        selected_runs(runs_root, **sel) if force else missing_scores(runs_root, **sel)
    )
    if not todo:
        return []
    for run in todo:
        if not (run / "predictions.npz").exists():
            raise FileNotFoundError(
                f"{run} has no predictions.npz; rerun it with --save-predictions"
            )
    ids = shard_ids(n_shards)
    train_models_for = SieveTrainModels(
        store=store,
        n_shards=n_shards,
        k=k,
        config_label=config_label,
        fit_depth=fit_depth,
        model_cache=model_cache,
        runs_root=runs_root,
        stores_root=stores_root,
    )
    mset_by_shard = load_shards(store, ids, stores_root=stores_root)
    written = []
    by_repeat: dict[int, list] = {}
    for run in todo:
        by_repeat.setdefault(int(_cv(run)["repeat"]), []).append(run)
    for repeat, runs in sorted(by_repeat.items()):
        plan = build_cv_plan(ids, k=k, repeat=repeat)
        models = _attach_training_floors(
            train_models_for(repeat, plan),
            plan,
            store,
            collapse=collapse,
            stores_root=stores_root,
        )
        for run in runs:
            cv = _cv(run)
            fold = int(cv["fold"])
            group = plan.groups[fold]
            if cv.get("held_out_shards") != ",".join(group):
                raise ValueError(f"{run}: held_out_shards disagrees with the CV plan")
            held_out = concat_molecule_sets([mset_by_shard[s] for s in group])
            scores = score_run(run, models[fold], held_out, n_jobs=n_jobs)
            (run / METRICS_FILE).write_text(
                json.dumps(scores, indent=1, sort_keys=True)
            )
            written.append(run)
    return written
