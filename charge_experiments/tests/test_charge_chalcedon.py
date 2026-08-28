"""The vendored _chalcedon modules import cleanly under this package's own
module path (the one thing Task 1's copy could get wrong) and reproduce the
doctested behavior from their upstream docstrings."""

from __future__ import annotations

import numpy as np
import pytest


def test_butina_cluster_matches_upstream_doctest_example():
    # butina_cluster.py (vendored, unmodified import behavior) does an
    # unconditional `from tqdm.auto import tqdm` -- tqdm ships in the
    # `charges` extra (used for prepare-store's real progress bars), not in
    # CI's `[dev,chem]` install, matching this repo's own convention of
    # not installing heavy/optional extras in CI (see e.g.
    # test_charge_prepare_store_optional.py's pandas/pyarrow guards).
    pytest.importorskip("tqdm")
    from charge_experiments._chalcedon.butina_cluster import butina_cluster

    fingerprints = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.uint8
    )
    assert butina_cluster(fingerprints, cutoff=0.5).tolist() == [1, 1, 0, 0]


def test_greedy_cluster_split_matches_upstream_doctest_example():
    from charge_experiments._chalcedon.greedy_cluster_split import (
        greedy_cluster_split,
    )

    ids = np.array([0, 0, 0, 1, 1, 2, 3])
    result = greedy_cluster_split(ids, {"train": 0.6, "test": 0.4})
    assert result["train"].tolist() == [0, 1, 2, 5]
    assert result["test"].tolist() == [3, 4, 6]
