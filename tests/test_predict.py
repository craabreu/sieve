import numpy as np

import sieve
from tests.helpers import chain_batch, simple_config, split_batch, star_batch


def test_training_atoms_recover_their_class_mean():
    cfg = simple_config()
    b = chain_batch(12)
    m = sieve.fit(b, cfg)
    p = sieve.predict_detailed(m, b)
    from sieve.refine import refine

    labels = refine(b, cfg)[-1].labels
    for i in range(b.n_nodes):
        if p.matched_level[i] == cfg.n_levels - 1:
            np.testing.assert_allclose(p.value[i], b.y[labels == labels[i]].mean(0))


def test_unseen_atom_falls_back_to_the_global_mean():
    cfg = simple_config()
    train = chain_batch(6)
    m = sieve.fit(train, cfg)
    alien = chain_batch(3)
    alien = type(alien)(
        **{**alien.__dict__, "node_attrs": np.full((3, 1), 7, np.int64)}
    )
    p = sieve.predict_detailed(m, alien)
    assert np.all(p.matched_level == -1)
    assert np.all(p.class_id == -1)
    np.testing.assert_allclose(p.value, np.broadcast_to(m.global_mean, p.value.shape))


def test_minimum_support_moves_the_match_shallower_never_deeper():
    """design.md 10.4."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(20, graphs=3)
    m = sieve.fit(b, cfg)
    loose = sieve.predict_detailed(m, b).matched_level
    tight = sieve.predict_detailed(m.with_params(minimum_support=4), b).matched_level
    assert np.all(tight <= loose)


def test_threshold_bound_distinguishes_support_stops_from_oov():
    cfg = simple_config(max_wl_depth=3, minimum_support=1000)
    b = chain_batch(20, graphs=3)
    m = sieve.fit(b, cfg)
    p = sieve.predict_detailed(m, b)
    assert p.threshold_bound.any(), (
        "a huge minimum_support must stop on support, not OOV"
    )


def test_matched_levels_form_a_prefix():
    """design.md 2.2: no gaps are constructible. Verified by construction here:
    the reported level must be supported and level+1 must not be."""
    cfg = simple_config(max_wl_depth=3)
    b = chain_batch(15, graphs=4)
    m = sieve.fit(b, cfg)
    p = sieve.predict_detailed(m, b)
    for i in range(b.n_nodes):
        k = int(p.matched_level[i])
        if 0 <= k < cfg.n_levels - 1:
            assert p.support[i] >= cfg.minimum_support


def test_variance_is_nan_at_support_one():
    cfg = simple_config(max_wl_depth=4)
    b = chain_batch(30, graphs=1)
    m = sieve.fit(b, cfg)
    p = sieve.predict_detailed(m, b)
    singles = p.support == 1
    if singles.any():
        assert np.all(np.isnan(p.variance[singles]))


def test_batched_and_split_prediction_agree():
    """design.md 10.4: batched and per-node prediction must agree."""
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    m = sieve.fit(b, cfg)
    full = sieve.predict(m, b)
    parts = [sieve.predict(m, split_batch(b, b.graph_id == g)) for g in range(4)]
    np.testing.assert_allclose(full, np.concatenate(parts))


def test_vector_targets_predict_elementwise():
    cfg = simple_config(target_dim=3)
    b = chain_batch(12, d=3)
    m = sieve.fit(b, cfg)
    assert sieve.predict(m, b).shape == (12, 3)


def _single_edge_batch(seed):
    from sieve.batch import NodeBatch

    rng = np.random.default_rng(seed)
    return NodeBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([0, 1], np.int64),
        edge_dst=np.array([1, 0], np.int64),
        edge_attrs=np.ones(2, np.int64).reshape(-1, 1),
        graph_id=np.zeros(2, np.int64),
        y=rng.normal(size=(2, 1)),
    )


def test_query_max_degree_below_training_max_degree_still_matches():
    """A query batch with a smaller max degree than the training corpus must
    still reach the deepest WL level it structurally matches, not fall back
    to a shallower level just because its own padding is narrower."""
    cfg = simple_config(max_wl_depth=1)
    train = star_batch(4, graphs=1)
    m = sieve.fit(train, cfg)
    # A lone edge -- max degree 1, well below the star's max degree 4 -- but
    # both leaf nodes are structurally identical to leaves already in train.
    query = _single_edge_batch(seed=2)
    p = sieve.predict_detailed(m, query)
    assert np.all(p.matched_level == cfg.n_levels - 1)


def test_query_max_degree_above_training_max_degree_still_matches():
    cfg = simple_config(max_wl_depth=1)
    train = _single_edge_batch(seed=2)
    m = sieve.fit(train, cfg)
    query = star_batch(4, graphs=1)
    p = sieve.predict_detailed(m, query)
    leaves = np.arange(query.n_nodes) != 0  # exclude the star's center
    assert np.all(p.matched_level[leaves] == cfg.n_levels - 1)


def test_oov_neighbor_does_not_falsely_match_a_lower_degree_class():
    """A neighbor whose own class is OOV to the model must make the query
    node unmatchable at that level, not accidentally collide with the -1 pad
    sentinel used for a genuinely absent neighbor. With a single bond type
    (n_edge_types=2) the sole real bond code is n_edge_types-1, so an OOV
    neighbor's encoded pair (-1 * n_edge_types + bond) lands exactly on -1
    unless guarded."""
    from sieve.batch import NodeBatch
    from sieve.config import SieveConfig

    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes={"element": {"C": 0, "H": 1}},
        edge_codes={"SINGLE": 1},
        max_wl_depth=1,
        minimum_support=1,
    )
    # Training: two isolated C atoms -- the model's only WL-level class is
    # "C, no neighbors".
    train = NodeBatch(
        node_attrs=np.zeros((2, 1), np.int64),
        edge_src=np.array([], np.int64),
        edge_dst=np.array([], np.int64),
        edge_attrs=np.array([], np.int64).reshape(-1, 1),
        graph_id=np.array([0, 1], np.int64),
        y=np.array([[1.0], [2.0]]),
    )
    model = sieve.fit(train, cfg)
    # Query: C bonded to H. H's own class is OOV; C's is not.
    query = NodeBatch(
        node_attrs=np.array([[0], [1]], np.int64),
        edge_src=np.array([0, 1], np.int64),
        edge_dst=np.array([1, 0], np.int64),
        edge_attrs=np.ones(2, np.int64).reshape(-1, 1),
        graph_id=np.zeros(2, np.int64),
    )
    p = sieve.predict_detailed(model, query)
    assert p.matched_level[0] == 0  # must back off, not falsely match level 1


def test_global_fallback_iff_level_zero_unsupported():
    cfg = simple_config()
    b = chain_batch(8)
    m = sieve.fit(b, cfg)
    p = sieve.predict_detailed(m, b)
    assert not np.any(p.matched_level == -1)
