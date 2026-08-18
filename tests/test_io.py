import numpy as np
import pytest

import sieve
from tests.helpers import chain_batch, simple_config


def test_round_trip_reproduces_predictions_bit_exactly(tmp_path):
    """design.md 9.3: not an aspiration, a testable property."""
    cfg = simple_config(max_wl_depth=3, alpha=1.5)
    b = chain_batch(20, graphs=3, d=4)
    m = sieve.fit(b, simple_config(max_wl_depth=3, alpha=1.5, target_dim=4))
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    a = sieve.predict(m, b)
    c = sieve.predict(loaded, b)
    assert np.array_equal(a, c), "round trip must be bit-exact, not merely close"


def test_round_trip_preserves_config(tmp_path):
    cfg = simple_config(max_wl_depth=2, n_min=3, alpha=0.5)
    m = sieve.fit(chain_batch(10), cfg)
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    assert loaded.config.schema_version == cfg.schema_version
    assert loaded.config.n_min == 3 and loaded.config.alpha == 0.5


def test_loaded_model_still_merges(tmp_path):
    cfg = simple_config()
    b = chain_batch(10, graphs=4)
    from tests.helpers import split_batch

    a = sieve.fit(split_batch(b, b.graph_id < 2), cfg)
    c = sieve.fit(split_batch(b, b.graph_id >= 2), cfg)
    p = tmp_path / "a.npz"
    a.save(p)
    merged = sieve.SieveModel.load(p).merge(c)
    np.testing.assert_allclose(merged.global_mean, a.merge(c).global_mean)


def test_unknown_format_version_is_refused(tmp_path):
    import json

    m = sieve.fit(chain_batch(6), simple_config())
    p = tmp_path / "m.npz"
    m.save(p)
    data = dict(np.load(p, allow_pickle=False))
    cfg = json.loads(bytes(data["config"]).decode())
    cfg["format_version"] = 999
    data["config"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    np.savez(p, **data)
    with pytest.raises(ValueError, match="format_version"):
        sieve.SieveModel.load(p)


def test_shrunk_means_are_not_stored(tmp_path):
    m = sieve.fit(chain_batch(10), simple_config(alpha=2.0))
    p = tmp_path / "m.npz"
    m.save(p)
    keys = list(np.load(p, allow_pickle=False).keys())
    assert not any("shrunk" in k or "shrink" in k for k in keys)
