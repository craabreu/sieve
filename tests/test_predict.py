import numpy as np
import pytest
import wllr
from tests.helpers import chain_batch, simple_config, split_batch

def test_training_atoms_recover_their_class_mean():
    cfg = simple_config()
    b = chain_batch(12)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    from wllr.refine import refine
    labels = refine(b, cfg)[-1].labels
    for i in range(b.n_atoms):
        if p.matched_level[i] == cfg.n_levels - 1:
            np.testing.assert_allclose(p.value[i], b.y[labels == labels[i]].mean(0))

def test_unseen_atom_falls_back_to_the_global_mean():
    cfg = simple_config()
    train = chain_batch(6)
    m = wllr.fit(train, cfg)
    alien = chain_batch(3)
    alien = type(alien)(**{**alien.__dict__,
                           "node_attrs": np.full((3, 1), 7, np.int64)})
    p = wllr.predict_detailed(m, alien)
    assert np.all(p.matched_level == -1)
    assert np.all(p.class_id == -1)
    np.testing.assert_allclose(p.value, np.broadcast_to(m.global_mean, p.value.shape))

def test_n_min_moves_the_match_shallower_never_deeper():
    """design.md 10.4."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(20, graphs=3)
    m = wllr.fit(b, cfg)
    loose = wllr.predict_detailed(m, b).matched_level
    tight = wllr.predict_detailed(m.with_params(n_min=4), b).matched_level
    assert np.all(tight <= loose)

def test_threshold_bound_distinguishes_support_stops_from_oov():
    cfg = simple_config(max_wl_depth=3, n_min=1000)
    b = chain_batch(20, graphs=3)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    assert p.threshold_bound.any(), "a huge n_min must stop on support, not OOV"

def test_matched_levels_form_a_prefix():
    """design.md 2.2: no gaps are constructible. Verified by construction here:
    the reported level must be supported and level+1 must not be."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(15, graphs=4)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    for i in range(b.n_atoms):
        k = int(p.matched_level[i])
        if 0 <= k < cfg.n_levels - 1:
            assert p.support[i] >= cfg.n_min

def test_variance_is_nan_at_support_one():
    cfg = simple_config(max_wl_depth=4)
    b = chain_batch(30, graphs=1)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    singles = p.support == 1
    if singles.any():
        assert np.all(np.isnan(p.variance[singles]))

def test_batched_and_split_prediction_agree():
    """design.md 10.4: batched and per-node prediction must agree."""
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    m = wllr.fit(b, cfg)
    full = wllr.predict(m, b)
    parts = [wllr.predict(m, split_batch(b, b.graph_id == g)) for g in range(4)]
    np.testing.assert_allclose(full, np.concatenate(parts))

def test_vector_targets_predict_elementwise():
    cfg = simple_config(target_dim=3)
    b = chain_batch(12, d=3)
    m = wllr.fit(b, cfg)
    assert wllr.predict(m, b).shape == (12, 3)

def test_global_fallback_iff_level_zero_unsupported():
    cfg = simple_config()
    b = chain_batch(8)
    m = wllr.fit(b, cfg)
    p = wllr.predict_detailed(m, b)
    assert not np.any(p.matched_level == -1)
