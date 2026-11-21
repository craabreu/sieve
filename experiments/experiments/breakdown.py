"""Where the held-out error of a CV run sits: a breakdown of saved predictions.

A run's own ``metrics.json`` holds one RMSE, MAE and R2 over every held-out
atom the arm answered. That is not enough to compare arms that answer
different atoms, nor to say where an arm's error comes from. This module
rescores the saved ``predictions.npz`` of the arms of one CV sample (a repeat
and a fold) on common sets of atoms, and writes the result beside each run.

**Scopes.** A *reference* arm (the DASH tree, which leaves atoms it cannot
match without a prediction) defines four sets of held-out atoms, and every arm
is scored on each of them as well as on all atoms:

- ``covered``: atoms with a reference prediction, the set on which the
  reference can be compared with anything else;
- ``uncovered``: the others;
- ``complete``: atoms of records in which every atom is covered, the set on
  which a normalized reference prediction exists, since normalizing a record
  with one missing atom leaves the whole record without predictions;
- ``incomplete``: the others.

When a HOSE arm is named, ``hose_fallback`` is a fifth scope: the atoms that
arm answered with the global mean of the training atoms, its last resort.

**Groups.** Within ``all`` and ``covered``, atoms are also grouped by element,
by the net charge of their record, and by their own formal charge. A global R2
is dominated by the differences between elements, which every arm resolves.

**Matched radius.** A run whose matched radii were recorded
(``experiments.matched_radius``) is also grouped by them: the radius of the
class that answered each atom, or ``none`` where the global mean did. The
share of atoms at each radius says how often the arm backs off, and the error
at each radius what backing off costs.

**Quantiles.** Every scope carries quantiles of the absolute error, because a
comparison of MAE with RMSE says nothing certain about medians or tails.

**Total charge.** The predicted charges of a record are also summed and
compared with its net charge, over all records and over the complete ones. A
record with an atom without a prediction has no sum, so an arm that leaves
atoms unanswered is scored on fewer records than the others unless the
comparison is restricted to the complete ones.

**Normalization.** Predictions are saved before normalization. An arm
normalized with equal weights is renormalized here, which needs only the saved
predictions and the net charges. The reference arm's own scheme needs a
per-atom spread that is not saved, so its normalized scores are taken from the
run's ``metrics.json`` instead, after checking that they were computed on
exactly the ``complete`` atoms.

**The HOSE control.** The register goes from the one-sphere code directly to
the global mean, whereas Sieve's coarsest class is the element. The HOSE arm is
therefore scored a second time with each fallback atom given the mean of its
element over the training atoms (``element_fallback``), which separates the
cost of that policy from the cost of the representation. The global mean and
the element means of a fold are read from the one-sphere tables of its
training shards, whose keys begin with the element of the central atom.

The per-atom element and formal charge depend only on the stored molecule, so
they are computed once per store (``build_atom_table``). Scores go to a
sidecar, ``breakdown_metrics.json``, as ``stereo_subsets`` and ``calibration``
do, so a run's own record is never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.metrics import regression_metrics

ATOMS_FILE = "atom-table.npz"
METRICS_FILE = "breakdown_metrics.json"
QUANTILES = {"q50": 0.5, "q90": 0.9, "q95": 0.95, "q99": 0.99, "q999": 0.999}
GROUP_SCOPES = ("all", "covered")
# The group a run is scored by when its matched radii were recorded, and the
# label of the atoms no class matched.
MATCHED_GROUP = "matched_radius"
MATCHED_NONE = "none"
# The global mean of a fold is a ratio of sums over its training shards, and
# the order of those sums here is not the order the fold model was merged in.
FALLBACK_ATOL = 1e-12
NAN = float("nan")


# --- per-atom columns and their table ---------------------------------------


def atom_columns(mol: Any) -> tuple[np.ndarray, np.ndarray]:
    """Atomic number and formal charge of every atom of ``mol``."""
    atoms = list(mol.GetAtoms())
    return (
        np.array([a.GetAtomicNum() for a in atoms], np.uint8),
        np.array([a.GetFormalCharge() for a in atoms], np.int8),
    )


def _columns_of_blob(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    from experiments.data import blob_to_mol

    return atom_columns(blob_to_mol(blob))


def build_atom_table(
    store: str, *, stores_root: Path, n_jobs: int | None = None
) -> Path:
    """Compute every stored record's atom columns and write them beside the
    store.

    Records are keyed by the store's identifier columns, those whose name ends
    in ``_id``. A store in which they do not identify a record is refused,
    since predictions could then not be matched to their atoms.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    store_dir = Path(stores_root) / store
    path = store_dir / "molecules.parquet"
    keys = [c for c in pq.ParquetFile(path).schema_arrow.names if c.endswith("_id")]
    if not keys:
        raise ValueError(f"{path} has no identifier column (a name ending in _id)")
    df = pd.read_parquet(path, columns=[*keys, "mol"])
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{path}: the columns {keys} are not unique per record")
    blobs = df["mol"].tolist()
    if n_jobs == 1:
        columns = [_columns_of_blob(b) for b in blobs]
    else:
        import multiprocessing as mp

        with mp.Pool(n_jobs) as pool:
            columns = pool.map(_columns_of_blob, blobs, chunksize=2000)
    counts = np.array([c[0].shape[0] for c in columns], np.int64)
    out = store_dir / ATOMS_FILE
    np.savez(
        out,
        key_columns=np.array(keys),
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        element=np.concatenate([c[0] for c in columns]),
        formal_charge=np.concatenate([c[1] for c in columns]),
        **{f"key_{k}": df[k].to_numpy().astype(str) for k in keys},
    )
    return out


@dataclass(frozen=True)
class AtomTable:
    key_columns: tuple[str, ...]
    index: dict[tuple[str, ...], int]
    offsets: np.ndarray
    element: np.ndarray
    formal_charge: np.ndarray

    def rows(self, ids: Any, num_atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Element and formal charge of the atoms of these records, in the
        given order. ``ids`` maps each identifier column to one value per
        record (a loaded ``predictions.npz`` does).

        Refuses a record whose stored atom count differs from the one the
        predictions carry: the two would then not describe the same atoms.
        """
        columns = [np.asarray(ids[k]).astype(str).tolist() for k in self.key_columns]
        rows = np.array([self.index[key] for key in zip(*columns, strict=True)])
        num_atoms = np.asarray(num_atoms, np.int64)
        if rows.size == 0:
            return np.zeros(0, np.uint8), np.zeros(0, np.int8)
        lo = self.offsets[rows]
        stored = self.offsets[rows + 1] - lo
        bad = np.flatnonzero(stored != num_atoms)
        if bad.size:
            i = int(bad[0])
            key = "/".join(c[i] for c in columns)
            raise ValueError(
                f"record {key}: predictions carry {int(num_atoms[i])} atoms, "
                f"the store {int(stored[i])}"
            )
        start = np.concatenate([[0], np.cumsum(num_atoms)[:-1]])
        atoms = np.repeat(lo - start, num_atoms) + np.arange(int(num_atoms.sum()))
        return self.element[atoms], self.formal_charge[atoms]


def load_atom_table(path: Path) -> AtomTable:
    z = np.load(path)
    keys = tuple(str(k) for k in z["key_columns"].tolist())
    columns = [z[f"key_{k}"].tolist() for k in keys]
    return AtomTable(
        keys,
        {key: i for i, key in enumerate(zip(*columns, strict=True))},
        z["offsets"],
        z["element"],
        z["formal_charge"],
    )


# --- the atoms a reference arm covers ---------------------------------------


def _record_of_atom(num_atoms: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(num_atoms)), num_atoms)


def _records_missing(covered: np.ndarray, num_atoms: np.ndarray) -> np.ndarray:
    """Number of atoms without a prediction in each record."""
    return np.bincount(
        _record_of_atom(num_atoms), weights=~covered, minlength=len(num_atoms)
    )


def coverage_scopes(
    reference: np.ndarray, num_atoms: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-atom masks of the five scopes the reference predictions define."""
    covered = np.isfinite(np.asarray(reference, np.float64))
    record = _record_of_atom(num_atoms)
    complete = (_records_missing(covered, num_atoms) == 0)[record]
    return {
        "all": np.ones(covered.size, bool),
        "covered": covered,
        "uncovered": ~covered,
        "complete": complete,
        "incomplete": ~complete,
    }


def coverage_outcomes(reference: np.ndarray, num_atoms: np.ndarray) -> dict[str, float]:
    """How much of the held-out set the reference arm leaves unanswered."""
    num_atoms = np.asarray(num_atoms, np.int64)
    covered = np.isfinite(np.asarray(reference, np.float64))
    missing = _records_missing(covered, num_atoms)
    incomplete = missing > 0
    n_atoms, n_records = int(covered.size), int(num_atoms.size)
    return {
        "coverage/n_atoms": n_atoms,
        "coverage/n_records": n_records,
        "coverage/atom_fraction_uncovered": float((~covered).sum()) / n_atoms,
        "coverage/atom_fraction_incomplete": float(num_atoms[incomplete].sum())
        / n_atoms,
        "coverage/record_fraction_incomplete": float(incomplete.sum()) / n_records,
        "coverage/record_fraction_empty": float((missing == num_atoms).sum())
        / n_records,
    }


# --- the breakdown itself ---------------------------------------------------


def _signed(value: int) -> str:
    return "0" if value == 0 else f"{value:+d}"


def _scope_scores(true: np.ndarray, pred: np.ndarray, sel: np.ndarray) -> dict:
    t, p = true[sel], pred[sel]
    scores = regression_metrics(t, p)
    n = int(sel.sum())
    out: dict[str, float] = {
        "n_atoms": n,
        "n_scored": n - int(scores["n_nan"]),
        "rmse": scores["rmse"],
        "mae": scores["mae"],
        "r2": scores["r2"],
    }
    err = np.abs(p - t)
    err = err[np.isfinite(err)]
    if err.size:
        values = np.quantile(err, list(QUANTILES.values()))
        out.update(
            {f"abs_error/{q}": float(v) for q, v in zip(QUANTILES, values, strict=True)}
        )
        out["abs_error/max"] = float(err.max())
    else:
        out.update({f"abs_error/{q}": NAN for q in QUANTILES})
        out["abs_error/max"] = NAN
    return out


def _group_scores(
    true: np.ndarray,
    pred: np.ndarray,
    sel: np.ndarray,
    codes: np.ndarray,
    labels: Mapping[int, str],
) -> dict[str, float]:
    """Scores of every group present in ``sel``, keyed ``<label>/<score>``."""
    out: dict[str, float] = {}
    if not sel.any():
        return out
    codes = np.asarray(codes, np.int64)
    low = int(codes[sel].min())
    size = int(codes[sel].max()) - low + 1
    scored = sel & np.isfinite(pred) & np.isfinite(true)
    c = codes[scored] - low
    t = true[scored]
    err = pred[scored] - t

    def total(weights: np.ndarray) -> np.ndarray:
        return np.bincount(c, weights=weights, minlength=size)

    n_atoms = np.bincount(codes[sel] - low, minlength=size)
    n_scored = np.bincount(c, minlength=size)
    sq, ab, st, stt = total(err**2), total(np.abs(err)), total(t), total(t**2)
    for i in np.flatnonzero(n_atoms):
        label = labels.get(int(i) + low, str(int(i) + low))
        n = int(n_scored[i])
        out[f"{label}/n_atoms"] = int(n_atoms[i])
        out[f"{label}/n_scored"] = n
        if n == 0:
            out.update({f"{label}/{m}": NAN for m in ("rmse", "mae", "r2")})
            continue
        ss_tot = float(stt[i] - st[i] ** 2 / n)
        out[f"{label}/rmse"] = float(np.sqrt(sq[i] / n))
        out[f"{label}/mae"] = float(ab[i] / n)
        out[f"{label}/r2"] = 1.0 - float(sq[i]) / ss_tot if ss_tot > 0 else NAN
    return out


def breakdown(
    true: np.ndarray,
    pred: np.ndarray,
    scopes: Mapping[str, np.ndarray],
    *,
    groups: Mapping[str, tuple[np.ndarray, Mapping[int, str]]] | None = None,
    group_scopes: Collection[str] | None = None,
) -> dict[str, float]:
    """Scores of ``pred`` on every scope, and on every group within a scope.

    ``scopes`` maps a name to a per-atom mask. ``groups`` maps a name to
    per-atom integer codes and the label of each code; ``group_scopes`` names
    the scopes that are broken down by group (all of them when ``None``).
    Atoms without a prediction count in ``n_atoms`` and not in ``n_scored``.
    """
    true = np.asarray(true, np.float64).ravel()
    pred = np.asarray(pred, np.float64).ravel()
    out: dict[str, float] = {}
    for scope, sel in scopes.items():
        for key, value in _scope_scores(true, pred, sel).items():
            out[f"{scope}/{key}"] = value
        if groups is None or (group_scopes is not None and scope not in group_scopes):
            continue
        for group, (codes, labels) in groups.items():
            for key, value in _group_scores(true, pred, sel, codes, labels).items():
                out[f"{scope}/{group}/{key}"] = value
    return out


def total_charge_scores(
    pred: np.ndarray,
    net: np.ndarray,
    num_atoms: np.ndarray,
    record_sets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Error of the summed charges of each record against its net charge.

    ``record_sets`` maps a name to a per-record mask. Records with an atom
    without a prediction count in ``n_records`` and not in ``n_scored``.
    """
    from experiments.data import molecule_sum

    net = np.asarray(net, np.float64)
    sums = molecule_sum(pred, _record_of_atom(num_atoms), len(num_atoms))
    out: dict[str, float] = {}
    for name, sel in record_sets.items():
        scores = regression_metrics(net[sel], sums[sel])
        n = int(sel.sum())
        out[f"{name}/total/n_records"] = n
        out[f"{name}/total/n_scored"] = n - int(scores["n_nan"])
        out[f"{name}/total/rmse"] = scores["rmse"]
        out[f"{name}/total/mae"] = scores["mae"]
    return out


# --- the HOSE register's last resort ----------------------------------------


def _symbol(atomic_number: int) -> str:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number))


def _atomic_number(symbol: str) -> int:
    from rdkit import Chem

    return int(Chem.GetPeriodicTable().GetAtomicNumber(symbol))


@dataclass(frozen=True)
class FallbackStatistics:
    """What a fold's training atoms offer when no HOSE code matches."""

    global_mean: float
    element_mean: dict[int, float]


def hose_fold_statistics(shard_files: Sequence[Path]) -> FallbackStatistics:
    """Global mean and element means of the training atoms of one fold.

    Read from the one-sphere table of each training shard: its keys begin with
    the element symbol of the central atom followed by ``-``, and every
    training atom carries exactly one of them, which is checked.
    """
    global_sum, global_count = 0.0, 0
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path in shard_files:
        with np.load(path, allow_pickle=False) as z:
            keys, key_sum, key_count = z["r1_keys"], z["r1_sum"], z["r1_count"]
            if int(key_count.sum()) != int(z["global_count"]):
                raise ValueError(
                    f"{path}: the one-sphere table holds {int(key_count.sum())} "
                    f"atoms, the shard {int(z['global_count'])}"
                )
            global_sum += float(z["global_sum"])
            global_count += int(z["global_count"])
            for key, s, n in zip(keys.tolist(), key_sum, key_count, strict=True):
                symbol = key.split("-", 1)[0]
                sums[symbol] = sums.get(symbol, 0.0) + float(s)
                counts[symbol] = counts.get(symbol, 0) + int(n)
    if global_count == 0:
        raise ValueError("no training atoms in the shards given")
    return FallbackStatistics(
        global_sum / global_count,
        {_atomic_number(s): sums[s] / counts[s] for s in sums},
    )


def fallback_atoms(pred: np.ndarray, global_mean: float) -> np.ndarray:
    """The atoms whose prediction is the global mean of the training atoms."""
    pred = np.asarray(pred, np.float64)
    with np.errstate(invalid="ignore"):
        return np.abs(pred - global_mean) <= FALLBACK_ATOL


def with_element_fallback(
    pred: np.ndarray,
    fallback: np.ndarray,
    element: np.ndarray,
    element_mean: Mapping[int, float],
) -> np.ndarray:
    """``pred`` with each fallback atom given the training mean of its element.

    An element with no training atom has no such mean, and its atoms keep the
    global mean.
    """
    out = np.array(pred, np.float64)
    for z, mean in element_mean.items():
        out[fallback & (element == z)] = mean
    return out


# --- scoring saved runs -----------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One arm of a study: where its runs are and the setting they are read at."""

    experiment: str
    method: str
    depth: int


def parse_arm(spec: str) -> Arm:
    """An arm from ``EXPERIMENT:METHOD:DEPTH``, the form the command line takes."""
    parts = spec.split(":")
    if len(parts) != 3 or not all(parts) or not parts[2].lstrip("-").isdigit():
        raise ValueError(f"an arm is EXPERIMENT:METHOD:DEPTH, got {spec!r}")
    return Arm(parts[0], parts[1], int(parts[2]))


def _cv(run: Path) -> dict[str, Any]:
    return (
        json.loads((run / "manifest.json").read_text()).get("config", {}).get("cv", {})
    )


def arm_runs(runs_root: Path, arm: Arm) -> dict[tuple[int, int], Path]:
    """The arm's run directory for each (repeat, fold) sample."""
    out: dict[tuple[int, int], Path] = {}
    pattern = f"{arm.experiment}/*/manifest.json"
    for manifest in sorted(Path(runs_root).glob(pattern)):
        cv = _cv(manifest.parent)
        if cv.get("method") != arm.method or int(cv.get("depth", -1)) != arm.depth:
            continue
        sample = (int(cv["repeat"]), int(cv["fold"]))
        if sample in out:
            raise ValueError(
                f"{arm.method} at {arm.depth} has two runs for repeat "
                f"{sample[0]}, fold {sample[1]}: {out[sample]} and {manifest.parent}"
            )
        out[sample] = manifest.parent
    return out


def _hose_shard_files(
    runs_root: Path, radius: int, held_out: Sequence[str], n_shards: int
) -> list[Path]:
    """The fitted one-shard states a fold's HOSE model was merged from."""
    from experiments.cv import SHARD_FIT_EXPERIMENT, hose_shard_batch_id, shard_ids

    files = []
    for shard in shard_ids(n_shards):
        if shard in held_out:
            continue
        batch = hose_shard_batch_id(radius, shard)
        found = sorted(
            (Path(runs_root) / SHARD_FIT_EXPERIMENT).glob(f"{batch}__*/tree_stats.npz")
        )
        if not found:
            raise FileNotFoundError(f"no HOSE shard fit {batch} under {runs_root}")
        files.append(found[-1])
    return files


def _prefixed(prefix: str, scores: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in scores.items()}


def _pred(z: Any) -> np.ndarray:
    pred = np.asarray(z["atom_target_pred"], np.float64)
    return pred.reshape(pred.shape[0], -1)[:, 0] if pred.ndim > 1 else pred


def _reference_norm(
    run: Path, pred: np.ndarray, num_atoms: np.ndarray, complete: np.ndarray
) -> dict[str, float]:
    """The run's own normalized scores, relabeled as scores on ``complete``.

    A scheme that leaves a whole record without predictions when one atom has
    none scores exactly the complete records, which is what is checked.
    """
    metrics = json.loads((run / "metrics.json").read_text())
    if "norm/rmse" not in metrics:
        return {}
    own = coverage_scopes(pred, num_atoms)["complete"]
    n = int(own.sum())
    scored = int(metrics["n_test_atoms"]) - int(metrics["norm/n_nan"])
    if scored != n or not np.array_equal(own, complete):
        raise ValueError(
            f"{run}: its normalized scores cover {scored} atoms, not the "
            f"{int(complete.sum())} atoms of the complete records"
        )
    out = {f"norm/complete/{m}": metrics[f"norm/{m}"] for m in ("rmse", "mae", "r2")}
    return {**out, "norm/complete/n_atoms": n, "norm/complete/n_scored": n}


def _score_sample(job: dict[str, Any], table: AtomTable) -> list[Path]:
    from experiments.normalize import equal_weighted_normalize

    ref = np.load(job["reference"] / "predictions.npz", allow_pickle=True)
    true = np.asarray(ref["atom_target_true"], np.float64).ravel()
    num_atoms = np.asarray(ref["num_atoms"], np.int64)
    record = _record_of_atom(num_atoms)
    scopes = coverage_scopes(_pred(ref), num_atoms)
    outcomes = coverage_outcomes(_pred(ref), num_atoms)
    record_sets = {
        "all": np.ones(num_atoms.size, bool),
        "complete": _records_missing(scopes["covered"], num_atoms) == 0,
    }
    element, formal_charge = table.rows(ref, num_atoms)
    groups: dict[str, tuple[np.ndarray, Mapping[int, str]]] = {
        "element": (element, {int(z): _symbol(z) for z in np.unique(element)}),
        "formal_charge": (
            formal_charge,
            {int(q): _signed(int(q)) for q in np.unique(formal_charge)},
        ),
    }
    net = None
    if "molecule_value" in ref.files:
        net = np.asarray(ref["molecule_value"], np.float64)
        codes = np.rint(net).astype(np.int64)
        groups["net_charge"] = (
            codes[record],
            {int(q): _signed(int(q)) for q in np.unique(codes)},
        )

    hose_run = job.get("hose")
    stats = fallback = None
    if hose_run is not None:
        cv = _cv(hose_run)
        stats = hose_fold_statistics(
            _hose_shard_files(
                job["runs_root"],
                int(cv["depth"]),
                str(cv["held_out_shards"]).split(","),
                job["n_shards"],
            )
        )
        hose = np.load(hose_run / "predictions.npz", allow_pickle=True)
        fallback = fallback_atoms(_pred(hose), stats.global_mean)
        scopes = {**scopes, "hose_fallback": fallback}

    def normalized(pred: np.ndarray) -> np.ndarray:
        assert net is not None
        return equal_weighted_normalize(pred, pred, net, record, num_atoms.size)

    written = []
    for run in job["todo"]:
        z = np.load(run / "predictions.npz", allow_pickle=True)
        if not np.array_equal(z["num_atoms"], num_atoms) or not np.array_equal(
            np.asarray(z["atom_target_true"], np.float64).ravel(), true
        ):
            raise ValueError(
                f"{run} and {job['reference']} did not score the same held-out atoms"
            )
        pred = _pred(z)
        out: dict[str, float] = dict(outcomes)
        run_groups = {**groups, **_matched_radius_group(run, true.size)}
        out.update(
            breakdown(true, pred, scopes, groups=run_groups, group_scopes=GROUP_SCOPES)
        )
        normalization = _cv(run).get("normalization")
        if net is not None:
            out.update(total_charge_scores(pred, net, num_atoms, record_sets))
        if normalization == "equal_weighted" and net is not None:
            out.update(_prefixed("norm", breakdown(true, normalized(pred), scopes)))
        elif normalization is not None:
            out.update(_reference_norm(run, pred, num_atoms, scopes["complete"]))
        if run == hose_run and stats is not None and fallback is not None:
            out.update(_fallback_scores(true, pred, fallback, scopes, element, record))
            control = with_element_fallback(pred, fallback, element, stats.element_mean)
            out.update(_prefixed("element_fallback", breakdown(true, control, scopes)))
            if net is not None:
                out.update(
                    _prefixed(
                        "element_fallback",
                        total_charge_scores(control, net, num_atoms, record_sets),
                    )
                )
                out.update(
                    _prefixed(
                        "element_fallback/norm",
                        breakdown(true, normalized(control), scopes),
                    )
                )
        (run / METRICS_FILE).write_text(json.dumps(out, indent=1, sort_keys=True))
        written.append(run)
    return written


def _matched_radius_group(
    run: Path, n_atoms: int
) -> dict[str, tuple[np.ndarray, Mapping[int, str]]]:
    """The run's own group by matched radius, when its radii were recorded
    (``experiments.matched_radius``); no group otherwise."""
    from experiments.matched_radius import UNMATCHED, read_matched

    radius = read_matched(run)
    if radius is None:
        return {}
    if radius.shape != (n_atoms,):
        raise ValueError(
            f"{run}: {radius.size} matched radii for {n_atoms} held-out atoms"
        )
    labels = {int(r): str(int(r)) for r in np.unique(radius)}
    if UNMATCHED in labels:
        labels[UNMATCHED] = MATCHED_NONE
    return {MATCHED_GROUP: (radius, labels)}


def _fallback_scores(
    true: np.ndarray,
    pred: np.ndarray,
    fallback: np.ndarray,
    scopes: Mapping[str, np.ndarray],
    element: np.ndarray,
    record: np.ndarray,
) -> dict[str, float]:
    """How often the register reached the global mean, and on which atoms."""
    n = int(fallback.sum())
    n_records = int(record[-1]) + 1 if record.size else 0
    out: dict[str, float] = {
        "fallback/n_atoms": n,
        "fallback/atom_fraction": n / fallback.size,
        "fallback/n_records": int(np.unique(record[fallback]).size),
        "fallback/record_fraction": int(np.unique(record[fallback]).size) / n_records,
        "fallback/covered_fraction": float((fallback & scopes["covered"]).sum()) / n
        if n
        else NAN,
    }
    scores = regression_metrics(true[fallback], pred[fallback])
    out["fallback/rmse"], out["fallback/mae"] = scores["rmse"], scores["mae"]
    for z in np.unique(element[fallback]):
        out[f"fallback/element/{_symbol(z)}/n_atoms"] = int(
            (fallback & (element == z)).sum()
        )
    return out


def _stale(run: Path, *sources: Path) -> bool:
    side = run / METRICS_FILE
    if not side.exists():
        return True
    written = side.stat().st_mtime
    if any(written < (s / "predictions.npz").stat().st_mtime for s in sources):
        return True
    from experiments.matched_radius import MATCHED_FILE

    radii = run / MATCHED_FILE
    return radii.exists() and written < radii.stat().st_mtime


_TABLE: AtomTable | None = None


def _init_worker(table: AtomTable) -> None:
    global _TABLE
    _TABLE = table


def _score_job(job: dict[str, Any]) -> list[Path]:
    assert _TABLE is not None
    return _score_sample(job, _TABLE)


def score_runs(
    runs_root: Path,
    arms: Sequence[Arm],
    table: AtomTable | None,
    *,
    reference: Arm,
    hose: Arm | None = None,
    n_shards: int = 50,
    force: bool = False,
    n_jobs: int = 1,
    check: bool = False,
) -> list[Path]:
    """Write ``breakdown_metrics.json`` for every run of ``arms`` that needs one.

    A sidecar is redone when it is older than the predictions it was computed
    from: the run's own, the reference arm's, or the HOSE arm's; or older than
    the run's recorded matched radii. With
    ``check``, nothing is written and the runs that would be are returned,
    which needs no ``table``.
    """
    runs_root = Path(runs_root)
    reference_runs = arm_runs(runs_root, reference)
    hose_runs = arm_runs(runs_root, hose) if hose is not None else {}
    jobs: dict[tuple[int, int], dict[str, Any]] = {}
    for arm in arms:
        for sample, run in arm_runs(runs_root, arm).items():
            if sample not in reference_runs:
                raise FileNotFoundError(
                    f"{run} has no reference run ({reference.method} at "
                    f"{reference.depth}) for repeat {sample[0]}, fold {sample[1]}"
                )
            if hose is not None and sample not in hose_runs:
                raise FileNotFoundError(
                    f"{run} has no HOSE run for repeat {sample[0]}, fold {sample[1]}"
                )
            sources = [run, reference_runs[sample]]
            if hose is not None:
                sources.append(hose_runs[sample])
            for source in sources:
                if not (source / "predictions.npz").exists():
                    raise FileNotFoundError(
                        f"{source} has no predictions.npz; rerun it with "
                        "--save-predictions"
                    )
            if not (force or _stale(run, *sources)):
                continue
            job = jobs.setdefault(
                sample,
                {
                    "runs_root": runs_root,
                    "n_shards": n_shards,
                    "reference": reference_runs[sample],
                    "hose": hose_runs.get(sample),
                    "todo": [],
                },
            )
            job["todo"].append(run)
    ordered = [jobs[s] for s in sorted(jobs)]
    if check:
        return [run for job in ordered for run in job["todo"]]
    if table is None:
        raise ValueError("scoring needs the store's atom table")
    if n_jobs == 1 or len(ordered) <= 1:
        results = [_score_sample(job, table) for job in ordered]
    else:
        import multiprocessing as mp

        # Forked workers share the table instead of each unpickling a copy.
        with mp.get_context("fork").Pool(
            min(n_jobs, len(ordered)), initializer=_init_worker, initargs=(table,)
        ) as pool:
            results = pool.map(_score_job, ordered, chunksize=1)
    return [run for written in results for run in written]


# --- the report --------------------------------------------------------------


def sample_table(
    runs_root: Path, arms: Sequence[Arm], keys: Sequence[str] | None = None
) -> tuple[list[tuple[int, int]], dict[str, dict[str, list[float]]]]:
    """Sidecar values of every arm, paired by (repeat, fold).

    Returns the samples in ascending order and ``values[key][method]``, one
    entry per sample, ``nan`` where a sidecar lacks the key. With ``keys``
    omitted, every key found is returned. Arms that did not score the same
    samples are refused.
    """
    methods = [a.method for a in arms]
    if len(set(methods)) != len(methods):
        raise ValueError(f"an arm is named twice: {methods}")
    scored: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    for arm in arms:
        scored[arm.method] = {}
        for sample, run in arm_runs(runs_root, arm).items():
            side = run / METRICS_FILE
            if not side.exists():
                raise FileNotFoundError(f"{run} has no {METRICS_FILE}; score it first")
            scored[arm.method][sample] = json.loads(side.read_text())
    samples = sorted(scored[methods[0]])
    for method in methods[1:]:
        if sorted(scored[method]) != samples:
            raise ValueError(
                f"{method} and {methods[0]} did not score the same samples"
            )
    if keys is None:
        keys = sorted({k for by in scored.values() for s in by.values() for k in s})
    values = {
        key: {m: [scored[m][s].get(key, NAN) for s in samples] for m in methods}
        for key in keys
    }
    return samples, values


def _mean_ci(values: Sequence[float], k: int) -> tuple[float, float]:
    """Mean and Nadeau-Bengio corrected 95% half-width over the samples."""
    from experiments.depth_curve import ci_half_width, nadeau_bengio_se

    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return NAN, NAN
    if a.size < 2:
        return float(a.mean()), NAN
    se = nadeau_bengio_se(a.tolist(), n_test=1.0, n_train=float(k - 1))
    return float(a.mean()), ci_half_width(se, a.size)


def _cell(values: Sequence[float], k: int, *, scale: float = 1.0) -> str:
    mean, half = _mean_ci(values, k)
    if not np.isfinite(mean):
        return "-"
    if not np.isfinite(half):
        return f"{scale * mean:.5f}"
    return f"{scale * mean:.5f} ± {scale * half:.5f}"


def _table(title: str, head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    lines = [title, ""]
    for r in [head, ["-" * w for w in widths], *rows]:
        first = f"{r[0]:<{widths[0]}}"
        rest = "  ".join(f"{c:>{w}}" for c, w in zip(r[1:], widths[1:], strict=True))
        lines.append(f"  {first}  {rest}".rstrip())
    return [*lines, ""]


def _group_labels(values: Mapping[str, Any], scope: str, group: str) -> list[str]:
    """Labels of ``group`` within ``scope``, most populated first."""
    prefix, suffix = f"{scope}/{group}/", "/n_atoms"
    sizes = {}
    for key, by_method in values.items():
        if key.startswith(prefix) and key.endswith(suffix):
            counts = np.asarray(next(iter(by_method.values())), np.float64)
            sizes[key[len(prefix) : -len(suffix)]] = float(np.nansum(counts))
    return sorted(sizes, key=lambda label: -sizes[label])


def _matched_radius_tables(
    values: Mapping[str, Mapping[str, list[float]]],
    methods: Sequence[str],
    n: int,
    k: int,
    number: int,
) -> list[str]:
    """Share of atoms, RMSE and MAE by matched radius, one column per arm whose
    radii were recorded; nothing when no arm's were."""

    def get(key: str, method: str) -> np.ndarray:
        return np.asarray(values.get(key, {}).get(method, [NAN] * n), np.float64)

    lines: list[str] = []
    for scope in GROUP_SCOPES:
        prefix, suffix = f"{scope}/{MATCHED_GROUP}/", "/n_atoms"
        labels = {
            key[len(prefix) : -len(suffix)]
            for key in values
            if key.startswith(prefix) and key.endswith(suffix)
        }
        if not labels:
            continue
        ordered = sorted(labels, key=lambda s: -1 if s == MATCHED_NONE else int(s))
        counts = {
            m: {r: np.nan_to_num(get(f"{prefix}{r}/n_atoms", m)) for r in ordered}
            for m in methods
        }
        totals = {m: sum(counts[m].values()) for m in methods}
        recorded = [m for m in methods if np.all(totals[m] > 0)]
        rows = [
            [r]
            + [
                _cell((counts[m][r] / totals[m]).tolist(), k, scale=100.0)
                for m in recorded
            ]
            for r in ordered
        ]
        lines += _table(
            f"{number}. Atoms by matched radius, {scope} atoms (%)",
            ["radius", *recorded],
            rows,
        )
        for metric in ("rmse", "mae"):
            rows = [
                [r]
                + [_cell(get(f"{prefix}{r}/{metric}", m).tolist(), k) for m in recorded]
                for r in ordered
            ]
            lines += _table(
                f"{number}. {metric.upper()} by matched radius, {scope} atoms",
                ["radius", *recorded],
                rows,
            )
    return lines


def breakdown_report(
    runs_root: Path,
    arms: Sequence[Arm],
    *,
    incumbent: Arm,
    hose: Arm | None = None,
    k: int = 5,
) -> tuple[str, dict[str, Any]]:
    """The report as plain text, and a snapshot of every per-sample value.

    Every cell is a mean over the samples with its corrected 95% half-width.
    Paired differences are an arm minus ``incumbent``, sample by sample.
    """
    from experiments.stereo_subsets import paired_difference

    samples, values = sample_table(runs_root, arms)
    methods = [a.method for a in arms]
    n = len(samples)

    def get(key: str, method: str) -> list[float]:
        return values.get(key, {}).get(method, [NAN] * n)

    lines = [
        f"Error breakdown of {n} cross-validation samples",
        f"arms: {', '.join(f'{a.method} ({a.experiment}, {a.depth})' for a in arms)}",
        "cells: mean ± corrected 95% half-width over the samples",
        "",
    ]

    first = methods[0]
    rows = []
    for key, label in (
        ("coverage/atom_fraction_uncovered", "atoms without a reference prediction"),
        ("coverage/record_fraction_incomplete", "records with such an atom"),
        ("coverage/record_fraction_empty", "records with no reference prediction"),
        ("coverage/atom_fraction_incomplete", "atoms in incomplete records"),
    ):
        a = np.asarray(get(key, first), np.float64)
        rows.append(
            [
                label,
                f"{100 * a.mean():.2f}",
                f"{100 * float(np.min(a)):.2f}",
                f"{100 * float(np.max(a)):.2f}",
            ]
        )
    lines += _table(
        "1. Coverage of the reference arm (% of the held-out set)",
        ["", "mean", "min", "max"],
        rows,
    )

    scopes = ["all", "covered", "uncovered", "complete", "incomplete"]
    if hose is not None:
        scopes.append("hose_fallback")
    for number, metric in ((2, "rmse"), (3, "mae")):
        rows = [
            [m]
            + [_cell(get(f"{s}/{metric}", m), k) for s in scopes]
            + [_cell(get(f"norm/complete/{metric}", m), k)]
            for m in methods
        ]
        lines += _table(
            f"{number}. {metric.upper()} by scope",
            ["arm", *scopes, "complete, normalized"],
            rows,
        )

    rows = [
        [m, _cell(get("all/total/rmse", m), k), _cell(get("complete/total/rmse", m), k)]
        for m in methods
    ]
    lines += _table(
        "3. RMSE of the total charge of a record, before normalization",
        ["arm", "all records", "complete records"],
        rows,
    )

    rows = []
    for key in ("covered/rmse", "covered/mae", "complete/rmse", "norm/complete/rmse"):
        base = np.asarray(get(key, incumbent.method), np.float64)
        for m in methods:
            arm = np.asarray(get(key, m), np.float64)
            if m == incumbent.method or not (
                np.isfinite(arm).all() and np.isfinite(base).all()
            ):
                continue
            if n < 2:
                rows.append([key, m, f"{float((arm - base).mean()):+.5f}", "-", "-"])
                continue
            d = paired_difference(base, arm, k=k)
            rows.append(
                [
                    key,
                    m,
                    f"{d.mean:+.5f}",
                    f"[{d.lo:+.5f}, {d.hi:+.5f}]",
                    f"{d.n_better}/{d.n}",
                ]
            )
    lines += _table(
        f"4. Paired differences, arm minus {incumbent.method}",
        ["score", "arm", "difference", "95% CI (corrected)", "arm lower"],
        rows,
    )

    names = [*QUANTILES, "max"]
    for scope in ("covered", "all"):
        rows = [
            [m]
            + [f"{_mean_ci(get(f'{scope}/abs_error/{q}', m), k)[0]:.5f}" for q in names]
            for m in methods
        ]
        lines += _table(
            f"5. Quantiles of the absolute error, {scope} atoms (mean over samples)",
            ["arm", *names],
            rows,
        )

    number = 6
    for group in ("element", "net_charge", "formal_charge"):
        for scope in GROUP_SCOPES:
            labels = _group_labels(values, scope, group)
            if not labels:
                continue
            for metric in ("rmse", "r2"):
                rows = []
                for label in labels:
                    count = np.nanmean(get(f"{scope}/{group}/{label}/n_atoms", first))
                    key = f"{scope}/{group}/{label}/{metric}"
                    cells = [f"{_mean_ci(get(key, m), k)[0]:.5f}" for m in methods]
                    rows.append([label, f"{count:.0f}", *cells])
                lines += _table(
                    f"{number}. {metric.upper()} by {group}, {scope} atoms "
                    "(mean over samples)",
                    [group, "atoms per sample", *methods],
                    rows,
                )
        number += 1

    radius_lines = _matched_radius_tables(values, methods, n, k, number)
    if radius_lines:
        lines += radius_lines
        number += 1

    if hose is not None:
        h = hose.method
        rows = [
            [
                "atoms answered by the global mean (%)",
                _cell(get("fallback/atom_fraction", h), k, scale=100.0),
            ],
            [
                "records with such an atom (%)",
                _cell(get("fallback/record_fraction", h), k, scale=100.0),
            ],
            [
                "of those atoms, covered by the reference (%)",
                _cell(get("fallback/covered_fraction", h), k, scale=100.0),
            ],
            [
                "RMSE of the global mean on those atoms",
                _cell(get("fallback/rmse", h), k),
            ],
        ]
        lines += _table(f"{number}. The fallback of {h}", ["", "value"], rows)
        labels = sorted(
            {
                key.split("/")[2]
                for key in values
                if key.startswith("fallback/element/") and key.endswith("/n_atoms")
            },
            key=lambda s: -float(np.nansum(get(f"fallback/element/{s}/n_atoms", h))),
        )
        rows = []
        for s in labels:
            counts = np.nan_to_num(get(f"fallback/element/{s}/n_atoms", h))
            rows.append([s, f"{counts.mean():.1f}"])
        lines += _table(
            f"{number + 1}. Fallback atoms of {h} by element (mean per sample)",
            ["element", "atoms"],
            rows,
        )
        rows = []
        for scope in scopes:
            for prefix, norm in (("", ""), ("norm/", ", normalized")):
                plain = np.asarray(get(f"{prefix}{scope}/rmse", h), np.float64)
                control = np.asarray(
                    get(f"element_fallback/{prefix}{scope}/rmse", h), np.float64
                )
                if not (np.isfinite(plain).all() and np.isfinite(control).all()):
                    continue
                row = [
                    f"{scope}{norm}",
                    _cell(plain.tolist(), k),
                    _cell(control.tolist(), k),
                ]
                if n >= 2:
                    d = paired_difference(plain, control, k=k)
                    row += [f"{d.mean:+.5f}", f"[{d.lo:+.5f}, {d.hi:+.5f}]"]
                else:
                    row += [f"{float((control - plain).mean()):+.5f}", "-"]
                rows.append(row)
        lines += _table(
            f"{number + 2}. RMSE of {h} with the element mean as its last resort",
            [
                "scope",
                "global mean",
                "element mean",
                "difference",
                "95% CI (corrected)",
            ],
            rows,
        )

    snapshot = {
        "arms": [
            {"experiment": a.experiment, "method": a.method, "depth": a.depth}
            for a in arms
        ],
        "incumbent": incumbent.method,
        "k": k,
        "samples": [list(s) for s in samples],
        "values": values,
    }
    return "\n".join(lines) + "\n", snapshot
