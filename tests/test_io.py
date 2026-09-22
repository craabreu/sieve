import numpy as np
import pytest

import sieve
from tests.helpers import chain_batch, simple_config


def test_round_trip_reproduces_predictions_bit_exactly(tmp_path):
    """design.md 9.3: not an aspiration, a testable property."""
    b = chain_batch(20, graphs=3, d=4)
    m = sieve.fit(
        b, simple_config(max_wl_depth=3, shrinkage_strength=1.5, target_dim=4)
    )
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    a = sieve.predict(m, b)
    c = sieve.predict(loaded, b)
    assert np.array_equal(a, c), "round trip must be bit-exact, not merely close"


def test_round_trip_preserves_config(tmp_path):
    cfg = simple_config(max_wl_depth=2, minimum_support=3, shrinkage_strength=0.5)
    m = sieve.fit(chain_batch(10), cfg)
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    assert loaded.config.schema_version == cfg.schema_version
    assert loaded.config.minimum_support == 3
    assert loaded.config.shrinkage_strength == 0.5


def test_round_trip_preserves_class_estimator(tmp_path):
    cfg = simple_config(max_wl_depth=2, class_estimator="continuation")
    m = sieve.fit(chain_batch(10), cfg)
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    assert loaded.config.class_estimator == "continuation"
    a = sieve.predict(m, chain_batch(10))
    c = sieve.predict(loaded, chain_batch(10))
    assert np.array_equal(a, c)


def test_round_trip_preserves_the_new_inference_knobs(tmp_path):
    cfg = simple_config(
        max_wl_depth=2,
        class_estimator="continuation_recursive",
        shrinkage_weight="diversity",
        shrinkage_strength=0.5,
    )
    m = sieve.fit(chain_batch(10), cfg)
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    assert loaded.config.class_estimator == "continuation_recursive"
    assert loaded.config.shrinkage_weight == "diversity"
    assert np.array_equal(
        sieve.predict(m, chain_batch(10)), sieve.predict(loaded, chain_batch(10))
    )


def test_loading_a_format_version_3_model_refuses_rather_than_guessing(tmp_path):
    """format_version 3 files predate class_estimator entirely -- there is no
    default to fall back to that would not silently misdescribe what
    predictions the file's classes actually support. Same clean-break policy
    as the v2 case above."""
    import json

    m = sieve.fit(chain_batch(6), simple_config())
    p = tmp_path / "m.npz"
    m.save(p)
    data = dict(np.load(p, allow_pickle=False))
    cfg = json.loads(bytes(data["config"]).decode())
    cfg["format_version"] = 3
    del cfg["class_estimator"]
    del cfg["shrinkage_weight"]
    data["config"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    np.savez(p, **data)
    with pytest.raises(ValueError, match="format_version 3"):
        sieve.SieveModel.load(p)


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


def test_loading_a_format_version_2_model_refuses_rather_than_guessing(tmp_path):
    """format_version 2 stored edge_codes as a flat {value: code} table. There
    is no migration (spec: clean break), so a v2 file must be refused outright
    rather than guessed at (design.md 9.2)."""
    import json

    m = sieve.fit(chain_batch(6), simple_config())
    p = tmp_path / "m.npz"
    m.save(p)
    data = dict(np.load(p, allow_pickle=False))
    cfg = json.loads(bytes(data["config"]).decode())
    cfg["format_version"] = 2
    data["config"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    np.savez(p, **data)
    with pytest.raises(ValueError, match="format_version 2"):
        sieve.SieveModel.load(p)


def test_shrunk_means_are_not_stored(tmp_path):
    m = sieve.fit(chain_batch(10), simple_config(shrinkage_strength=2.0))
    p = tmp_path / "m.npz"
    m.save(p)
    keys = list(np.load(p, allow_pickle=False).keys())
    assert not any("shrunk" in k or "shrink" in k for k in keys)


def _stereo_model_and_batch():
    import dataclasses

    from rdkit import Chem

    from sieve.io.rdkit_adapter import from_rdkit

    cfg = simple_config(stereo=("cis_trans",), max_wl_depth=3)
    mols = [Chem.MolFromSmiles(s) for s in ["C/C=C/C", r"C/C=C\C", "CCCC"]]
    rng = np.random.default_rng(0)
    b = from_rdkit(mols, y=None, config=cfg)
    b = dataclasses.replace(b, y=rng.normal(size=(b.n_nodes, 1)))
    return sieve.fit(b, cfg), b


def test_round_trip_preserves_a_stereo_track(tmp_path):
    """save()/load() went through an explicit field whitelist on each side,
    and stereo was missing from both: a stereo-enabled model fit correctly
    but raised on load, since the reconstructed config's schema_version
    (stereo defaulting back to ()) no longer matched what was stored."""
    m, b = _stereo_model_and_batch()
    p = tmp_path / "m.npz"
    m.save(p)
    loaded = sieve.SieveModel.load(p)
    assert loaded.config.stereo == ("cis_trans",)
    assert loaded.config.schema_version == m.config.schema_version
    assert np.array_equal(sieve.predict(m, b), sieve.predict(loaded, b))


def test_loading_a_pre_stereo_file_defaults_to_no_stereo_rather_than_refusing(
    tmp_path,
):
    """A real file saved before this field existed has no "stereo" key in its
    JSON blob at all -- not an empty one. Unlike class_estimator (refused,
    since guessing would silently misdescribe how a stored class is read),
    a missing stereo key has an unambiguous historical answer: the feature
    did not exist yet, so the file's classes are stereo-blind by fact, not
    by guess. This must load successfully, not raise -- the schema_version
    payload only ever gained the "stereo" key conditionally (config.py), so
    an old file's stored digest already matches a stereo=() reconstruction."""
    import json

    m = sieve.fit(chain_batch(6), simple_config())
    p = tmp_path / "m.npz"
    m.save(p)
    data = dict(np.load(p, allow_pickle=False))
    cfg = json.loads(bytes(data["config"]).decode())
    del cfg["stereo"]  # simulates a file saved before this field existed
    data["config"] = np.frombuffer(json.dumps(cfg).encode(), np.uint8)
    np.savez(p, **data)

    loaded = sieve.SieveModel.load(p)
    assert loaded.config.stereo == ()
    assert loaded.config.schema_version == m.config.schema_version
