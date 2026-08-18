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
    assert base(n_min=1).schema_version == base(n_min=9).schema_version
    assert base(alpha=None).schema_version == base(alpha=2.0).schema_version


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
    check_mergeable(base(), base(n_min=7))  # inference params may differ
    with pytest.raises(ValueError, match="schema"):
        check_mergeable(base(), base(max_wl_depth=4))


def test_n_levels_counts_attribute_levels_plus_wl_depths():
    cfg = base(attribute_levels=(("element",), ("aromatic",)), max_wl_depth=3)
    assert cfg.n_levels == 5


def test_neighbour_schema_is_not_implemented():
    with pytest.raises(NotImplementedError):
        base(neighbour_schema=("element",))


def test_empty_attribute_group_is_rejected():
    """A zero-width group makes refine()'s dedupe degrade silently (design.md
    3.5) rather than raising anywhere near the actual mistake."""
    with pytest.raises(ValueError, match="attribute"):
        base(attribute_levels=((),))
    with pytest.raises(ValueError, match="attribute"):
        base(attribute_levels=(("element",), ()))
