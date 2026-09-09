"""experiments: a node-level regression harness for molecular stores. A run
names the atomic property it trains and predicts (``config.TargetCfg``);
the data-preparation script is the only dataset-specific component. Built
for, and still principally used for, DASH atomic partial charges
(``MBIScharge``) -- see
docs/superpowers/specs/2026-08-26-dash-charges-experiment-series-design.md
and docs/superpowers/specs/2026-09-09-experiments-generalization-design.md.
"""

from __future__ import annotations
