"""Within-structure variance (within-structure-variance spec, phase 1).

Collapse removes each structure's conformer-to-conformer scatter from the fit;
these two additive statistics carry it back, so the predictive variance can
add it to every class.
"""

from __future__ import annotations

import numpy as np
import pytest

import sieve
from sieve.model import SieveModel
from tests.helpers import chain_batch, simple_config


def _fitted(**kw):
    return sieve.fit(chain_batch(10, graphs=3), simple_config(**kw))


# ----------------------------------------------------------------- model --


def test_a_model_without_statistics_has_zero_within_variance():
    m = _fitted()
    assert m.within_sse is None
    assert m.within_n == 0.0
    np.testing.assert_array_equal(m.within_variance, np.zeros(1))


def test_with_within_structure_sets_the_sums():
    m = _fitted().with_within_structure(3.0e-3, 30)
    np.testing.assert_array_equal(m.within_sse, [3.0e-3])
    assert m.within_n == 30.0
    np.testing.assert_allclose(m.within_variance, [1.0e-4])


@pytest.mark.parametrize(
    ("sse", "n"), [(-1.0, 3), (np.nan, 3), (1.0, -2), (1.0, np.inf), (1.0, 0)]
)
def test_with_within_structure_refuses_impossible_sums(sse, n):
    with pytest.raises(ValueError):
        _fitted().with_within_structure(sse, n)


def test_with_within_structure_changes_neither_predictions_nor_the_digest():
    cfg = simple_config(max_wl_depth=2)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(1.0, 10)
    np.testing.assert_array_equal(sieve.predict(w, batch), sieve.predict(m, batch))
    assert w.config.schema_version == m.config.schema_version


def test_merge_adds_the_sums():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted().with_within_structure(1.0, 5)
    ab = a.merge(b)
    np.testing.assert_allclose(ab.within_sse, [3.0])
    assert ab.within_n == 15.0


def test_merge_with_a_model_without_statistics_keeps_the_other_side():
    a = _fitted().with_within_structure(2.0, 10)
    b = _fitted()
    for merged in (a.merge(b), b.merge(a)):
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


def test_merge_of_two_models_without_statistics_has_none():
    ab = _fitted().merge(_fitted())
    assert ab.within_sse is None
    assert ab.within_n == 0.0


def test_the_empty_model_is_still_the_merge_identity():
    a = _fitted().with_within_structure(2.0, 10)
    e = SieveModel.empty(a.config)
    for merged in (a.merge(e), e.merge(a)):
        np.testing.assert_allclose(merged.within_sse, [2.0])
        assert merged.within_n == 10.0


# ----------------------------------------------------------- persistence --


def test_save_load_round_trips_the_sums(tmp_path):
    m = _fitted().with_within_structure(2.5e-3, 25)
    path = tmp_path / "m.npz"
    m.save(path)
    back = SieveModel.load(path)
    np.testing.assert_array_equal(back.within_sse, m.within_sse)
    assert back.within_n == m.within_n


def test_a_file_without_statistics_loads_and_saves_back_without_them(tmp_path):
    m = _fitted()
    path = tmp_path / "m.npz"
    m.save(path)
    assert "within" not in np.load(path).files
    back = SieveModel.load(path)
    assert back.within_sse is None
    assert back.within_n == 0.0
    again = tmp_path / "again.npz"
    back.save(again)
    assert sorted(np.load(again).files) == sorted(np.load(path).files)


# ------------------------------------------------------ predictive variance --


def test_alpha_v_is_ten():
    from sieve.uncertainty import ALPHA_T, ALPHA_V, SELECTION_WEIGHT

    assert (ALPHA_V, ALPHA_T, SELECTION_WEIGHT) == (10.0, 1.0, 0.5)


def test_predictive_variance_adds_the_within_variance_exactly():
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    w = m.with_within_structure(4.0e-2, 100)
    for base, form_b in zip(
        predictive_variance(m), predictive_variance(w), strict=True
    ):
        np.testing.assert_array_equal(form_b, base + 4.0e-4)


def test_without_statistics_the_variance_is_the_three_terms():
    """σ²_w = 0 adds nothing: bit-identical to an explicit zero."""
    from sieve.uncertainty import predictive_variance

    m = _fitted(max_wl_depth=3)
    z = m.with_within_structure(0.0, 0)
    for a, b in zip(predictive_variance(m), predictive_variance(z), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unmatched_nodes_fall_back_to_global_msd_plus_the_within_variance():
    from sieve.batch import NodeBatch

    cfg = simple_config(max_wl_depth=1, predictive_variance=True)
    m = sieve.fit(chain_batch(10, graphs=2), cfg).with_within_structure(1.0, 10)
    oov = NodeBatch(
        node_attrs=np.array([[7]], np.int64),  # an element code never fitted
        edge_src=np.zeros(0, np.int64),
        edge_dst=np.zeros(0, np.int64),
        edge_attrs=np.zeros((0, 1), np.int64),
        graph_id=np.zeros(1, np.int64),
    )
    out = sieve.predict_detailed(m, oov)
    assert out.matched_level[0] == -1
    assert out.predictive_variance is not None
    np.testing.assert_allclose(out.predictive_variance[0], m.global_msd + 0.1)


def test_matched_nodes_read_form_b():
    cfg = simple_config(max_wl_depth=2, predictive_variance=True)
    batch = chain_batch(10, graphs=3)
    m = sieve.fit(batch, cfg)
    w = m.with_within_structure(2.0, 10)
    base = sieve.predict_detailed(m, batch).predictive_variance
    form_b = sieve.predict_detailed(w, batch).predictive_variance
    assert base is not None and form_b is not None
    np.testing.assert_allclose(form_b, base + 0.2, rtol=1e-12)
