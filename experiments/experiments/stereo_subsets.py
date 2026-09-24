"""Study D's stereo-affected subsets of a CV run's held-out atoms.

Study D asks whether the cis/trans track improves the atoms stereo can
actually reach, not the corpus as a whole (docs/superpowers/specs/2026-09-23-
stereo-refines-the-blind-class-design.md, section 11). Three subsets, each a
per-atom mask of a stored conformer:

- ``has_ez``: every atom of a conformer with at least one stereogenic double
  bond;
- ``near_ez2``: atoms within two bonds of such a bond's atoms, the primary
  metric's subset;
- ``near_ez1``: atoms within one bond.

A bond counts when the stored molecule marks it E/Z (``GetStereo`` in cis,
trans, E or Z), which is what the adapter's cis/trans rows are read from.

The masks depend only on the stored molecule, so they are computed once per
store (``build_mask_table``) and every run is scored from its saved
``predictions.npz`` (``score_runs``). Scoring saved predictions rather than
inside the CV loop is what lets Study B's incumbent runs, already on disk and
holding out the same molecules, serve as the paired arm without being redone.
Each run's scores go to a sidecar, ``subset_metrics.json``, which
``aggregate.read_runs_from_dirs`` merges into the run's metrics, so
``compare`` and everything else read them like any other metric.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HAS_EZ, NEAR_2, NEAR_1 = 0, 1, 2
SUBSETS = ("has_ez", "near_ez2", "near_ez1")  # indexed by the constants above
MASKS_FILE = "stereo-subset-masks.npz"
METRICS_FILE = "subset_metrics.json"


def subset_masks(mol: Any) -> np.ndarray:
    """``(3, n_atoms)`` bool: the three subsets for one molecule, rows
    indexed by ``HAS_EZ``, ``NEAR_2``, ``NEAR_1``."""
    from rdkit import Chem

    stereo = {
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOZ,
    }
    ends = sorted(
        {
            a
            for b in mol.GetBonds()
            if b.GetStereo() in stereo
            for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        }
    )
    n = mol.GetNumAtoms()
    out = np.zeros((3, n), bool)
    if not ends:
        return out
    near = Chem.GetDistanceMatrix(mol)[:, ends].min(axis=1)
    out[HAS_EZ] = True
    out[NEAR_2] = near <= 2
    out[NEAR_1] = near <= 1
    return out


def _packed(blob: bytes) -> np.ndarray:
    """One byte per atom, bit ``s`` set when the atom is in subset ``s``."""
    from experiments.data import blob_to_mol

    m = subset_masks(blob_to_mol(blob))
    return (m[HAS_EZ] | (m[NEAR_2] << 1) | (m[NEAR_1] << 2)).astype(np.uint8)


def build_mask_table(
    store: str, *, stores_root: Path, n_jobs: int | None = None
) -> Path:
    """Compute every stored conformer's masks and write them beside the store.

    Rows in store order: ``dash_id``, ``conf_id``, ``offsets`` into ``bits``,
    and ``bits`` (the packed masks, one byte per atom).
    """
    import pandas as pd

    store_dir = Path(stores_root) / store
    df = pd.read_parquet(
        store_dir / "molecules.parquet", columns=["dash_id", "conf_id", "mol"]
    )
    blobs = df["mol"].tolist()
    if n_jobs == 1:
        packed = [_packed(b) for b in blobs]
    else:
        import multiprocessing as mp

        with mp.Pool(n_jobs) as pool:
            packed = pool.map(_packed, blobs, chunksize=2000)
    counts = np.array([p.shape[0] for p in packed], np.int64)
    out = store_dir / MASKS_FILE
    np.savez(
        out,
        dash_id=df["dash_id"].to_numpy().astype(str),
        conf_id=df["conf_id"].to_numpy().astype(str),
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        bits=np.concatenate(packed) if packed else np.zeros(0, np.uint8),
    )
    return out


@dataclass(frozen=True)
class MaskTable:
    index: dict[tuple[str, str], int]
    offsets: np.ndarray
    bits: np.ndarray

    def rows(self, dash_ids, conf_ids, num_atoms) -> np.ndarray:
        """Packed masks for these conformers, concatenated in the given order.

        Refuses a conformer whose stored atom count differs from the one the
        predictions carry: the two would then not describe the same atoms.
        """
        parts = []
        for d, c, n in zip(dash_ids, conf_ids, num_atoms, strict=True):
            r = self.index[(str(d), str(c))]
            lo, hi = self.offsets[r], self.offsets[r + 1]
            if hi - lo != int(n):
                raise ValueError(
                    f"conformer {d}/{c}: predictions carry {int(n)} atoms, "
                    f"the store {hi - lo}"
                )
            parts.append(self.bits[lo:hi])
        return np.concatenate(parts) if parts else np.zeros(0, np.uint8)


def load_mask_table(path: Path) -> MaskTable:
    z = np.load(path)
    keys = zip(z["dash_id"].tolist(), z["conf_id"].tolist(), strict=True)
    return MaskTable(
        {(str(d), str(c)): i for i, (d, c) in enumerate(keys)},
        z["offsets"],
        z["bits"],
    )


def subset_metrics(predictions: Path, table: MaskTable) -> dict[str, float]:
    """RMSE, MAE and atom count on each subset, for one saved run."""
    z = np.load(predictions, allow_pickle=True)
    bits = table.rows(z["dash_id"], z["conf_id"], z["num_atoms"])
    err = np.asarray(z["atom_target_pred"], np.float64) - np.asarray(
        z["atom_target_true"], np.float64
    )
    if err.ndim > 1:
        err = err.reshape(err.shape[0], -1)[:, 0]
    out: dict[str, float] = {}
    for s, name in enumerate(SUBSETS):
        sel = (bits >> s) & 1 == 1
        n = int(sel.sum())
        out[f"{name}/n_atoms"] = n
        nan = float("nan")
        out[f"{name}/rmse"] = float(np.sqrt(np.mean(err[sel] ** 2))) if n else nan
        out[f"{name}/mae"] = float(np.mean(np.abs(err[sel]))) if n else nan
    return out


def _selected_runs(
    runs_root: Path, selection: Mapping[str, Sequence[str]], depth: int
) -> list[Path]:
    """Run directories of ``selection``'s (experiment -> methods) at ``depth``."""
    out = []
    for experiment, methods in selection.items():
        for manifest in sorted(Path(runs_root).glob(f"{experiment}/*/manifest.json")):
            cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
            if cv.get("method") in methods and int(cv.get("depth", -1)) == depth:
                out.append(manifest.parent)
    return out


def missing_scores(
    runs_root: Path, selection: Mapping[str, Sequence[str]], *, depth: int
) -> list[Path]:
    """Selected runs whose sidecar is absent or older than their predictions.

    A selected run with no predictions at all is reported too: it can never
    be scored, and silently passing over it would drop a paired sample.
    """
    stale = []
    for run in _selected_runs(runs_root, selection, depth):
        pred, side = run / "predictions.npz", run / METRICS_FILE
        if not pred.exists() or not side.exists():
            stale.append(run)
        elif side.stat().st_mtime < pred.stat().st_mtime:
            stale.append(run)
    return stale


def score_runs(
    runs_root: Path,
    selection: Mapping[str, Sequence[str]],
    table: MaskTable,
    *,
    depth: int,
    force: bool = False,
) -> list[Path]:
    """Write ``subset_metrics.json`` for every selected run that needs one."""
    todo = (
        _selected_runs(runs_root, selection, depth)
        if force
        else missing_scores(runs_root, selection, depth=depth)
    )
    written = []
    for run in todo:
        pred = run / "predictions.npz"
        if not pred.exists():
            raise FileNotFoundError(
                f"{run} has no predictions.npz; rerun it with --save-predictions"
            )
        (run / METRICS_FILE).write_text(
            json.dumps(subset_metrics(pred, table), indent=1, sort_keys=True)
        )
        written.append(run)
    return written


@dataclass(frozen=True)
class PairedDifference:
    """``b - a`` over paired (repeat, fold) samples."""

    mean: float
    lo: float
    hi: float
    n: int
    n_better: int  # samples where b is lower than a


def paired_difference(
    a: np.ndarray, b: np.ndarray, *, k: int, level: float = 0.95
) -> PairedDifference:
    """Mean paired difference with the Nadeau-Bengio corrected interval.

    Every sample holds out one fold of ``k``, so ``n_test/n_train`` is
    ``1/(k-1)``, and the correction is the repeated k-fold form, with the
    variance of the mean ``(1/n + n_test/n_train) s^2`` over all ``n``
    samples and ``n - 1`` degrees of freedom.
    """
    from experiments.depth_curve import ci_half_width, nadeau_bengio_se

    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    se = nadeau_bengio_se(d.tolist(), n_test=1.0, n_train=float(k - 1))
    half = ci_half_width(se, d.size, level)
    mean = float(d.mean())
    return PairedDifference(
        mean, mean - half, mean + half, int(d.size), int((d < 0).sum())
    )


@dataclass(frozen=True)
class Pair:
    """One arm and the incumbent it is paired with, by experiment and method."""

    incumbent_experiment: str
    incumbent_method: str
    experiment: str
    method: str
    label: str


def stereo_report(
    runs_root: Path,
    pairs: Sequence[Pair],
    *,
    depth: int,
    metrics: Sequence[str],
    k: int,
) -> list[dict[str, Any]]:
    """One row per (pair, metric): both arms' means and their paired
    difference, arm minus incumbent, with the corrected interval.

    Samples are paired by (repeat, fold) through ``compare.read_cv_table``,
    which refuses two arms that did not score the same samples.
    """
    from experiments.compare import read_cv_table

    rows = []
    for pair in pairs:
        for metric in metrics:
            methods, table = read_cv_table(
                Path(runs_root),
                [pair.incumbent_experiment, pair.experiment],
                depth_by_method={pair.incumbent_method: depth, pair.method: depth},
                metric=metric,
            )
            a = table[:, methods.index(pair.incumbent_method)]
            b = table[:, methods.index(pair.method)]
            diff = paired_difference(a, b, k=k)
            rows.append(
                {
                    "label": pair.label,
                    "metric": metric,
                    "incumbent": float(a.mean()),
                    "arm": float(b.mean()),
                    "diff": diff.mean,
                    "lo": diff.lo,
                    "hi": diff.hi,
                    "relative": diff.mean / float(a.mean()),
                    "n": diff.n,
                    "n_better": diff.n_better,
                }
            )
    return rows


def format_report(rows: Sequence[Mapping[str, Any]]) -> str:
    """A plain-text table of ``stereo_report``'s rows."""
    head = (
        f"{'estimator':<10} {'metric':<16} {'incumbent':>10} {'+ cis/trans':>11} "
        f"{'change':>11} {'95% CI (NB-corrected)':>26} {'rel.':>7} {'better':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        ci = "[{:+.2e}, {:+.2e}]".format(r["lo"], r["hi"])
        lines.append(
            f"{r['label']:<10} {r['metric']:<16} {r['incumbent']:>10.6f} "
            f"{r['arm']:>11.6f} {r['diff']:>+11.2e} {ci:>26} "
            f"{100 * r['relative']:>+6.2f}% {r['n_better']:>3}/{r['n']:<3}"
        )
    return "\n".join(lines) + "\n"
