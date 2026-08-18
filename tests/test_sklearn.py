import numpy as np

from sieve.sklearn import GraphKFold, SieveRegressor
from tests.helpers import chain_batch, simple_config


def test_get_set_params_round_trip():
    r = SieveRegressor(simple_config(), n_min=3, alpha=1.0)
    assert r.get_params()["n_min"] == 3
    r.set_params(n_min=9)
    assert r.get_params()["n_min"] == 9


def test_fit_predict_matches_the_functional_core():
    import sieve

    cfg = simple_config()
    b = chain_batch(15, graphs=3)
    r = SieveRegressor(cfg).fit(b, b.y)
    np.testing.assert_allclose(r.predict(b), sieve.predict(sieve.fit(b, cfg), b))


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
    r = SieveRegressor(cfg).fit(b, b.y)
    assert r.score(b, b.y) <= 1.0


def test_score_averages_r2_per_output_not_pooled_variance():
    """score() must not let one target column's scale dominate another's
    (design.md 10.2 asks for scikit-learn-standard behaviour, and
    RegressorMixin.score's r2_score(multioutput="uniform_average") is
    exactly that)."""
    from sklearn.metrics import r2_score

    cfg = simple_config(target_dim=2)
    b = chain_batch(20, d=2, graphs=4, seed=7)
    y = b.y.copy()
    y[:, 1] *= 1000.0  # wildly different scale on the second output
    b = type(b)(**{**b.__dict__, "y": y})
    r = SieveRegressor(cfg).fit(b, y)
    pred = r.predict(b)
    np.testing.assert_allclose(r.score(b, y), r2_score(y, pred))


def test_cross_val_score_actually_works():
    """design.md 10.2: the whole point of this module is that GridSearchCV
    and friends work. sklearn's cross-validation indexes X and y with the
    fold's index arrays itself (X[train], y[train]) rather than calling
    r.fit(batch, batch.y) directly, so this exercises AtomBatch's own
    array-like protocol (`.shape`, `__len__`, `__getitem__`), not just the
    estimator interface."""
    from sklearn.model_selection import cross_val_score

    cfg = simple_config()
    b = chain_batch(10, graphs=12, seed=1)
    r = SieveRegressor(cfg)
    scores = cross_val_score(r, b, b.y, cv=GraphKFold(n_splits=3, random_state=1))
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_grid_search_cv_sweeps_n_min_and_alpha():
    from sklearn.model_selection import GridSearchCV

    cfg = simple_config(max_wl_depth=2)
    b = chain_batch(12, graphs=10, seed=2)
    search = GridSearchCV(
        SieveRegressor(cfg),
        param_grid={"n_min": [1, 2], "alpha": [None, 1.0]},
        cv=GraphKFold(n_splits=3, random_state=2),
    )
    search.fit(b, b.y)
    assert search.best_params_["n_min"] in (1, 2)
