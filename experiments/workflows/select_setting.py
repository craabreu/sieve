"""The depth or radius Study B fixes an arm at, read off Study A's own curve.

**The rule.** The shallowest setting whose mean held-out RMSE is within a
relative tolerance of the minimum's. A plateau is then read as a plateau
rather than as a peak, and the cheapest point on it wins -- which matters
most for the HOSE arm, whose cost grows steeply with radius and which
cannot amortize a deep fit over shallow ones.

**Why a relative tolerance and not a confidence interval.** The rule this
replaces admitted any setting inside the *minimum's own* 95% Nadeau-Bengio
interval. Every setting is scored on the same five held-out sets, so the
settings are strongly correlated and the variance of a difference is far
smaller than either interval suggests; an unpaired tolerance on paired data
is loose. Measured on the pre-collapse Study A curves it admitted depth 4,
which sits ~1.9% above the minimum and visibly still on the descent, and
the paper had to override it in prose to reach 5. A relative tolerance
states the same judgement as a criterion: we do not buy a larger
environment for a gain below the tolerance.

**Why 0.5%.** On the pre-collapse curves the band of tolerances selecting
the published settings is (0.160%, 1.881%] for Sieve pooled, (0.175%,
2.049%] for Sieve continuation, and (0.000%, 1.032%] for HOSE. Their
geometric centres are ~0.55% and ~0.60% for the two Sieve arms, so 0.5%
sits mid-band rather than on an edge, and it clears HOSE's upper bound by a
factor of two where 1% would have cleared it by 3% and could flip that arm
between radii on a small change in the curve.

Prints one integer, or exits 1 when Study A has not produced a usable
curve, so the caller can fall back to the spec's deployed setting.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TOLERANCE = float(os.environ.get("CV_SELECT_TOLERANCE", "0.005"))


def curve(experiment: str, method: str) -> dict[int, list[float]]:
    rows: dict[int, list[float]] = defaultdict(list)
    for manifest in Path("experiments/runs", experiment).glob("*__*/manifest.json"):
        metrics = manifest.parent / "metrics.json"
        if not metrics.exists():
            continue
        cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
        if cv.get("method") != method:
            continue
        rows[int(cv["depth"])].append(json.loads(metrics.read_text())["rmse"])
    return rows


def main(experiment: str, method: str = "hose") -> int:
    usable = {d: v for d, v in curve(experiment, method).items() if len(v) > 1}
    if not usable:
        return 1
    means = {d: st.mean(v) for d, v in usable.items()}
    ceiling = min(means.values()) * (1.0 + TOLERANCE)
    print(min(d for d in means if means[d] <= ceiling))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
