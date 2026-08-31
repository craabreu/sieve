import pickle
from dataclasses import replace

import pytest

from sieve.config import SieveConfig, check_mergeable

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"SINGLE": 1},
    max_wl_depth=3,
)


def base(**kw):
    return replace(_BASE_CONFIG, **kw)


def test_schema_version_is_stable():
    assert base().schema_version == base().schema_version


def test_schema_version_ignores_inference_params():
    assert (
        base(minimum_support=1).schema_version == base(minimum_support=9).schema_version
    )
    assert (
        base(shrinkage_strength=None).schema_version
        == base(shrinkage_strength=2.0).schema_version
    )


def test_schema_version_tracks_meaning():
    assert base().schema_version != base(max_wl_depth=4).schema_version
    assert (
        base().schema_version
        != base(attribute_codes={"element": {"C": 0, "H": 2}}).schema_version
    )
    assert (
        base().schema_version
        != base(attribute_levels=(("element",), ("aromatic",))).schema_version
    )


def test_mergeable_requires_matching_schema():
    check_mergeable(base(), base(minimum_support=7))  # inference params may differ
    with pytest.raises(ValueError, match="schema"):
        check_mergeable(base(), base(max_wl_depth=4))


def test_n_levels_counts_attribute_levels_plus_wl_depths():
    cfg = base(attribute_levels=(("element",), ("aromatic",)), max_wl_depth=3)
    assert cfg.n_levels == 5


def test_neighbor_depth_rejects_out_of_range_values():
    cfg = base(attribute_levels=(("element",), ("aromatic",)))
    with pytest.raises(ValueError, match="neighbor_depth"):
        replace(cfg, neighbor_depth=0)
    with pytest.raises(ValueError, match="neighbor_depth"):
        replace(cfg, neighbor_depth=3)  # only 2 attribute levels declared


def test_neighbor_depth_equal_to_attribute_count_normalizes_to_none():
    """ "No coarsening" has exactly one spelling, so a config that spells it
    as neighbor_depth == len(attribute_levels) hashes identically to one
    that leaves neighbor_depth unset."""
    cfg = base(attribute_levels=(("element",), ("aromatic",)))
    explicit = replace(cfg, neighbor_depth=2)
    assert explicit.neighbor_depth is None
    assert explicit.schema_version == cfg.schema_version


def test_schema_version_distinguishes_neighbor_depth():
    cfg = base(attribute_levels=(("element",), ("aromatic",)))
    coarsened = replace(cfg, neighbor_depth=1)
    assert coarsened.schema_version != cfg.schema_version


def test_empty_attribute_group_is_rejected():
    """A zero-width group makes refine()'s dedupe degrade silently (design.md
    3.5) rather than raising anywhere near the actual mistake."""
    with pytest.raises(ValueError, match="attribute"):
        base(attribute_levels=((),))
    with pytest.raises(ValueError, match="attribute"):
        base(attribute_levels=(("element",), ()))


def test_config_survives_a_pickle_round_trip():
    """MappingProxyType (attribute_codes/edge_codes) has no stdlib pickle
    support and raises by default -- this is what multiprocessing needs to
    ship a config to a worker process for parallel fitting (design.md 5.1)."""
    cfg = base(attribute_codes={"element": {"C": 0, "H": 1}})
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.schema_version == cfg.schema_version
    assert restored.attribute_codes == cfg.attribute_codes
    with pytest.raises(TypeError):
        restored.attribute_codes["element"]["N"] = 2  # still frozen
