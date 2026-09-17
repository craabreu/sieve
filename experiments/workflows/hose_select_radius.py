"""The radius Study B fixes the HOSE arm at, read off Study A's own curve.

Same rule that chose Sieve's depth 5: the shallowest radius whose mean RMSE
lies inside the minimum's own 95% Nadeau-Bengio interval. A plateau is then
read as a plateau rather than as a peak, and the cheapest point on it wins --
which matters more for this arm than for the others, since its cost grows
steeply with radius and it cannot amortize a deep fit over shallow ones.

Prints one integer, or exits 1 when Study A has not produced a usable curve,
so the caller can fall back to the spec's deployed setting.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

K_FOLDS_TEST_OVER_TRAIN = 10 / 40  # this study's geometry: 10 held out of 50
T_975_DF4 = 2.776


def main(experiment: str, method: str = "hose") -> int:
    rows: dict[int, list[float]] = defaultdict(list)
    for manifest in Path("experiments/runs", experiment).glob("*__*/manifest.json"):
        metrics = manifest.parent / "metrics.json"
        if not metrics.exists():
            continue
        cv = json.loads(manifest.read_text()).get("config", {}).get("cv", {})
        if cv.get("method") != method:
            continue
        rows[int(cv["depth"])].append(json.loads(metrics.read_text())["rmse"])

    usable = {r: v for r, v in rows.items() if len(v) > 1}
    if not usable:
        return 1

    k = max(len(v) for v in usable.values())
    nb = math.sqrt(1 / k + K_FOLDS_TEST_OVER_TRAIN)
    stats = {r: (st.mean(v), T_975_DF4 * nb * st.stdev(v)) for r, v in usable.items()}
    best = min(stats, key=lambda r: stats[r][0])
    ceiling = stats[best][0] + stats[best][1]
    print(min(r for r in stats if stats[r][0] <= ceiling))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
