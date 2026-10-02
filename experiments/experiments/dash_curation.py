"""Curation of the two DASH subsets, as described in the manuscript.

Both subsets lose the records whose geometry contradicts their own molecular graph (the
``stretched_bond`` and ``close_contact`` flags of ``experiments.geometry``). DASH/QMugs,
whose records carry the electrostatic potential that QMugs computed at every nucleus, is
curated further:

* **Bounds from triangles of copies.** Every triple of mutual copies is a triangle whose
  sides are charge discrepancies. A single defective copy enlarges only the two sides it
  touches, so the smallest side joins two healthy copies unless two of the three failed;
  its ``AGREE_QUANTILE`` quantile is the bound below which two copies agree, and its
  maximum the bound above which they disagree.
* **ESP score.** The potential at nucleus ``i`` is regressed on the record's own MBIS
  charge at that nucleus and on the potential of its other MBIS charges there, ``phi_i ~
  a_k + b_k q_i + c_k sum_j q_j / r_ij`` (atomic units), with coefficients per atom
  class ``k`` (element, degree, bonded hydrogens, aromaticity, formal charge) for
  classes of at least ``CLASS_MIN_ATOMS`` atoms and per element otherwise, fitted by
  least squares on all DASH/QMugs records with residuals beyond four robust sigmas
  iteratively discarded. Each residual is divided by the robust scale of its element,
  and the score of a record is their root mean square.
* **Copy rule.** In a pair of copies beyond the disagreement bound, a third copy that
  agrees with one member and disagrees with the other is a witness and condemns the
  other; without a single verdict, the member with the higher ESP score goes.
* **ESP cut.** The ``1 - ESP_FPR`` quantile of the score over the healthy copies
  (records in an agreeing pair of copies that no witness condemns); every record above
  it without an agreeing copy goes.

Ported from sieve_paper's ``data_analysis/qmugs_curation.py`` and
``curation2.esp_score``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from experiments.dash_diagnostics import COPY_RMSD

logger = logging.getLogger("experiments")

AGREE_QUANTILE = 0.999
ESP_FPR = 5e-4
CLASS_MIN_ATOMS = 300
ELEMENT_MIN_ATOMS = 100
BOHR = 0.52917721  # angstrom
ELEMENTS = (1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53)
STEPS = ("geometry", "copy rule, witness", "copy rule, ESP", "ESP cut")


# --------------------------------------------------------------------------
# ESP score


def robust_lstsq(
    X: np.ndarray, y: np.ndarray, iters: int = 4, k: float = 4.0
) -> np.ndarray:
    """Least squares refitted after trimming residuals beyond ``k`` robust sigmas."""
    keep = np.ones(len(y), dtype=bool)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < k * max(s, 1e-12)
    return beta


def atom_terms(mol: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per atom of a stored record: atomic number, class key, MBIS charge, and the
    potential (a.u.)
    of the record's other MBIS charges at its nucleus."""
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    key = z.copy()
    for values in (
        [a.GetDegree() for a in mol.GetAtoms()],
        [a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms()],
        [int(a.GetIsAromatic()) for a in mol.GetAtoms()],
        [a.GetFormalCharge() for a in mol.GetAtoms()],
    ):
        key = key * 16 + (np.asarray(values, dtype=np.int64) + 8)
    q = np.array([a.GetDoubleProp("MBIScharge") for a in mol.GetAtoms()])
    x = mol.GetConformer().GetPositions() / BOHR
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return z, key, q, (q[None, :] / d).sum(axis=1)


def _terms_task(blobs: list[bytes]) -> list[tuple[np.ndarray, ...]]:
    from rdkit import rdBase

    from experiments.data import blob_to_mol

    rdBase.DisableLog("rdApp.*")
    return [atom_terms(blob_to_mol(b)) for b in blobs]


def esp_score(
    z: np.ndarray,
    key: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    esp: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Per record, the RMS over its atoms of the standardised residual of eq. (esp
    model).

    All arrays but ``offsets`` are per atom; record ``r`` owns atoms
    ``offsets[r]:offsets[r + 1]``.
    Atoms whose potential is NaN are left out, and a record with none gets NaN."""
    ok = ~np.isnan(esp) & ~np.isnan(v)
    X = np.c_[np.ones(len(q)), q, v]
    resid = np.full(len(q), np.nan)
    for element in ELEMENTS:
        m = ok & (z == element)
        if m.sum() < ELEMENT_MIN_ATOMS:
            continue
        resid[m] = esp[m] - X[m] @ robust_lstsq(X[m], esp[m])
    for k in np.unique(key[ok]):
        m = ok & (key == k)
        if m.sum() < CLASS_MIN_ATOMS:
            continue
        resid[m] = esp[m] - X[m] @ robust_lstsq(X[m], esp[m])
    has = ~np.isnan(resid)
    scale = np.full(128, np.nan)
    for element in ELEMENTS:
        m = has & (z == element)
        if m.sum() >= ELEMENT_MIN_ATOMS:
            scale[element] = 1.4826 * np.median(np.abs(resid[m] - np.median(resid[m])))
    zs = np.where(has, np.abs(resid) / scale[z], 0.0)
    count = np.add.reduceat(has.astype(float), offsets[:-1])
    score = np.full(len(offsets) - 1, np.nan)
    good = count > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        score[good] = np.sqrt(np.add.reduceat(zs * zs, offsets[:-1]) / count)[good]
    return score


def esp_scores(blobs: list[bytes], esp: list[Any], *, workers: int = 16) -> np.ndarray:
    """ESP score of every record of ``blobs``, fitted on all of them; ``esp`` holds
    each record's
    potential at the nuclei (one value per atom)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    chunk = 5000
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        terms = [
            t
            for part in pool.map(
                _terms_task, [blobs[k : k + chunk] for k in range(0, len(blobs), chunk)]
            )
            for t in part
        ]
    sizes = np.array([len(t[0]) for t in terms])
    offsets = np.r_[0, np.cumsum(sizes)]
    z, key, q, v = (np.concatenate([t[i] for t in terms]) for i in range(4))
    phi = np.concatenate([np.asarray(e, dtype=np.float64) for e in esp])
    if len(phi) != len(q):
        raise ValueError(
            "the potential at the nuclei does not match the atoms of the records"
        )
    return esp_score(z, key, q, v, phi, offsets)


# --------------------------------------------------------------------------
# Copies


def triangles(copies: Any) -> Any:
    """Every triangle of mutual copies, with its MBIS and GFN2-xTB sides sorted in
    increasing order."""
    import pandas as pd

    d, x, adj = {}, {}, defaultdict(set)
    for a, b, dm, dx in copies[["row_a", "row_b", "d_mbis", "d_xtb"]].itertuples(
        index=False
    ):
        d[a, b] = d[b, a] = dm
        x[a, b] = x[b, a] = dx
        adj[a].add(b)
        adj[b].add(a)
    out = []
    for a in adj:
        for b in (b for b in adj[a] if b > a):
            for c in (c for c in adj[a] & adj[b] if c > b):
                out.append(
                    (
                        *sorted((d[a, b], d[a, c], d[b, c])),
                        *sorted((x[a, b], x[a, c], x[b, c])),
                    )
                )
    return pd.DataFrame(out, columns=["d1", "d2", "d3", "x1", "x2", "x3"])


def triangle_bounds(tri: Any) -> tuple[float, float]:
    """``(agree, disagree)``: the ``AGREE_QUANTILE`` quantile and the maximum of the
    smallest side."""
    if len(tri) == 0:
        raise ValueError("no triangle of copies to take the bounds from")
    return float(np.quantile(tri["d1"], AGREE_QUANTILE)), float(tri["d1"].max())


def copy_rule(
    copies: Any, score: np.ndarray, agree: float, disagree: float
) -> tuple[dict[int, str], dict]:
    """Records condemned by the copy rule, as ``{row: step}``, and the counts behind
    them.

    ``copies`` holds ``row_a``, ``row_b`` and ``d_mbis`` per pair of copies; ``score``
    is indexed by row."""
    import pandas as pd

    both = pd.concat(
        [
            copies[["row_a", "row_b", "d_mbis"]],
            copies[["row_b", "row_a", "d_mbis"]].set_axis(
                ["row_a", "row_b", "d_mbis"], axis=1
            ),
        ]
    )
    d = {(a, b): x for a, b, x in both.itertuples(index=False)}
    partners = both.groupby("row_a")["row_b"].apply(set).to_dict()
    removed: dict[int, str] = {}
    witnessed = judge_agrees = conflicts = 0
    disagreeing = copies.loc[copies["d_mbis"] > disagree, ["row_a", "row_b"]]
    for a, b in disagreeing.itertuples(index=False):
        verdict = set()
        for w in partners.get(a, set()) & partners.get(b, set()):
            if d[a, w] < agree and d[b, w] > disagree:
                verdict.add(b)
            elif d[b, w] < agree and d[a, w] > disagree:
                verdict.add(a)
        judged = a if score[a] > score[b] else b
        if len(verdict) == 1:
            failed = verdict.pop()
            witnessed += 1
            judge_agrees += failed == judged
            removed[int(failed)] = "copy rule, witness"
        else:
            conflicts += len(verdict) > 1
            removed.setdefault(int(judged), "copy rule, ESP")
    info = {
        "copy_pairs": len(copies),
        "disagreeing_pairs": len(disagreeing),
        "witnessed_pairs": witnessed,
        "conflicting_pairs": conflicts,
        "esp_agrees_with_witness": judge_agrees,
    }
    return removed, info


# --------------------------------------------------------------------------
# The whole curation


def curate(parsed: Any, pairs: Any, esp: Any, *, workers: int = 16) -> tuple[Any, dict]:
    """``(table, info)``: per row of the staging store, ``curation_step`` (empty when
    kept) and
    ``esp_score`` (NaN outside DASH/QMugs), and the diagnostics of every step.

    ``parsed`` holds ``mol``, ``subset`` and the geometry flags; ``pairs`` is
    ``dash_diagnostics``'s
    pair table; ``esp`` the potential at the nuclei per row (None outside
    DASH/QMugs)."""
    import pandas as pd

    n = len(parsed)
    step = np.full(n, "", dtype=object)
    subset = parsed["subset"].to_numpy()
    geometry = (parsed["stretched_bond"] | parsed["close_contact"]).to_numpy()
    step[geometry] = "geometry"
    info: dict = {
        "geometry": {
            s: int(np.sum(geometry & (subset == s))) for s in ("QMugs", "Extra")
        }
    }

    qmugs = np.flatnonzero(subset == "QMugs")
    score = np.full(n, np.nan)
    score[qmugs] = esp_scores(
        [parsed["mol"].iat[r] for r in qmugs],
        [esp.iat[r] for r in qmugs],
        workers=workers,
    )

    alive = step == ""
    qp = pairs[
        (pairs["subset"] == "QMugs") & alive[pairs["row_a"]] & alive[pairs["row_b"]]
    ]
    copies = qp[qp["rmsd"] < COPY_RMSD].reset_index(drop=True)
    tri = triangles(copies)
    agree, disagree = triangle_bounds(tri)
    removed, rule = copy_rule(copies, score, agree, disagree)
    for row, label in removed.items():
        step[row] = label
    witnessed = np.zeros(n, dtype=bool)
    witnessed[[r for r, label in removed.items() if label == "copy rule, witness"]] = (
        True
    )
    in_agreeing = np.zeros(n, dtype=bool)
    ag = copies[copies["d_mbis"] < agree]
    in_agreeing[ag["row_a"]] = in_agreeing[ag["row_b"]] = True
    healthy = in_agreeing & ~witnessed
    cut = float(np.quantile(score[healthy], 1.0 - ESP_FPR))
    in_qmugs = subset == "QMugs"
    esp_cut = in_qmugs & ~in_agreeing & (step == "") & (score > cut)
    step[esp_cut] = "ESP cut"

    in_copy = np.zeros(n, dtype=bool)
    in_copy[np.r_[copies["row_a"], copies["row_b"]]] = True
    unpaired = in_qmugs & ~in_copy
    detect = float(np.mean(score[witnessed] > cut)) if witnessed.any() else float("nan")
    by_rule = int(np.sum(np.char.startswith(step.astype(str), "copy rule")))
    prevalence = by_rule / max(int(in_copy.sum()), 1)
    predicted = (detect * prevalence + ESP_FPR * (1 - prevalence)) * float(
        unpaired.sum()
    )
    distinct = qp[qp["rmsd"] >= COPY_RMSD]
    gone = step != ""
    keep = ~(gone[distinct["row_a"]] | gone[distinct["row_b"]])
    info["QMugs"] = {
        "triangles": len(tri),
        "agree": agree,
        "disagree": disagree,
        "triangle_largest_xtb_side": float(tri["x3"].max()),
        **rule,
        "removed_by_witness": int(np.sum(step == "copy rule, witness")),
        "removed_by_esp_judge": int(np.sum(step == "copy rule, ESP")),
        "records_in_copy_pairs": int(in_copy.sum()),
        "healthy_copies": int(healthy.sum()),
        "esp_cut": cut,
        "esp_fpr": ESP_FPR,
        "witnessed_detected": detect,
        "unpaired_above_cut_observed": int(np.sum(unpaired & (score > cut))),
        "unpaired_above_cut_predicted": predicted,
        "spared_by_agreeing_copy": int(np.sum(in_agreeing & (score > cut) & ~gone)),
        "removed_by_esp_cut": int(esp_cut.sum()),
        "removed_total": int(np.sum(gone & in_qmugs)),
        "records": int(in_qmugs.sum()),
        "distinct_beyond_disagree": [
            float(np.mean(distinct["d_mbis"] > disagree)),
            float(np.mean(distinct["d_mbis"][keep] > disagree)),
        ],
        "distinct_max": [
            float(distinct["d_mbis"].max()),
            float(distinct["d_mbis"][keep].max()),
        ],
    }
    info["Extra"] = {
        "removed_total": int(np.sum(gone & (subset == "Extra"))),
        "records": int(np.sum(subset == "Extra")),
    }
    logger.info(
        "curation removes %d DASH/QMugs and %d DASH/Extra records",
        info["QMugs"]["removed_total"],
        info["Extra"]["removed_total"],
    )
    return pd.DataFrame({"curation_step": step, "esp_score": score}), info
