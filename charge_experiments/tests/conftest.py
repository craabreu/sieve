"""Ensures the repo root is on ``sys.path``.

``charge_experiments/tests/`` has no ``__init__.py`` (deliberately -- it isn't part
of any installed package), so pytest's own rootdir-insertion for files here
only adds *this* directory to ``sys.path``, not the repo root. Without this,
``from charge_experiments.tests.helpers import synthetic_molecule_set`` (an
implicit-namespace-package import: ``charge_experiments/`` has no ``__init__.py``
either) only resolves by accident, when ``tests/`` (which does have an
``__init__.py``) happens to be collected in the same pytest run and triggers
repo-root insertion as a side effect of importing *that* package -- e.g. it
breaks running ``pytest charge_experiments/tests/`` on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
