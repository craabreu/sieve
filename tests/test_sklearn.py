import numpy as np
import pytest
from wllr.sklearn import WLLRRegressor, GraphKFold
from tests.helpers import chain_batch, simple_config

def test_get_set_params_round_trip():
    r = WLLRRegressor(simple_config(), n_min=3, alpha=1.0)
    assert r.get_params()["n_min"] == 3
    r.set_params(n_min=9)
    assert r.get_params()["n_min"] == 9

def test_fit_predict_matches_the_functional_core():
    import wllr
    cfg = simple_config()
    b = chain_batch(15, graphs=3)
    r = WLLRRegressor(cfg).fit(b, b.y)
    np.testing.assert_allclose(r.predict(b), wllr.predict(wllr.fit(b, cfg), b))

def test_graph_kfold_never_splits_a_molecule():
    b = chain_batch(6, graphs=10)
    for train, test in GraphKFold(n_splits=5).split(b):
        assert not (set(b.graph_id[train]) & set(b.graph_id[test]))

def test_graph_kfold_covers_every_atom_exactly_once_as_test():
    b = chain_batch(6, graphs=10)
    seen = np.concatenate([test for _, test in GraphKFold(5).split(b)])
    assert np.array_equal(np.sort(seen), np.arange(b.n_atoms))

def test_score_is_r2():
    cfg = simple_config()
    b = chain_batch(20, graphs=4)
    r = WLLRRegressor(cfg).fit(b, b.y)
    assert r.score(b, b.y) <= 1.0
