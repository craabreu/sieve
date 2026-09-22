import pickle
from dataclasses import replace

import pytest

from sieve.config import SieveConfig, check_mergeable

_BASE_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"bond_type": {"SINGLE": 0}},
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


def test_class_estimator_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="class_estimator"):
        base(class_estimator="median")


def test_class_estimator_defaults_to_pooled_and_is_excluded_from_schema_version():
    assert base().class_estimator == "pooled"
    assert base().schema_version == base(class_estimator="continuation").schema_version
    assert (
        base().schema_version
        == base(class_estimator="continuation_recursive").schema_version
    )


def test_shrinkage_weight_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="shrinkage_weight"):
        base(shrinkage_weight="entropy")


def test_shrinkage_weight_defaults_to_none_meaning_no_shrinkage():
    """Naming a rule is how shrinkage is requested; the default asks for
    none, which is exactly what the field's absence used to mean."""
    cfg = base()
    assert cfg.shrinkage_weight is None
    assert cfg.applies_shrinkage is False
    assert cfg.effective_shrinkage_weight == "count"


def test_a_bare_shrinkage_strength_still_means_the_count_rule():
    """Backward compatibility: configs predating shrinkage_weight must
    behave exactly as they did."""
    cfg = base(shrinkage_strength=0.5)
    assert cfg.applies_shrinkage is True
    assert cfg.effective_shrinkage_weight == "count"


def test_shrinkage_weight_is_excluded_from_schema_version():
    a = base(shrinkage_strength=0.5)
    b = base(shrinkage_weight="diversity", shrinkage_strength=0.5)
    assert a.schema_version == b.schema_version


def test_edge_radices_and_n_edge_types_are_a_product_over_attributes():
    """n_edge_types is the size of the collapsed edge alphabet. Each attribute
    contributes its vocabulary plus one reserved unknown code, and the
    collapse is mixed-radix, so the alphabet is the product."""
    cfg = base(
        edge_attributes=("bond_type", "conjugated"),
        edge_codes={
            "bond_type": {"SINGLE": 0, "DOUBLE": 1},
            "conjugated": {"False": 0, "True": 1},
        },
    )
    assert cfg.edge_radices == (3, 3)
    assert cfg.n_edge_types == 9


def test_empty_edge_schema_gives_a_single_edge_type():
    """edge_attributes == () is a supported control arm: every edge becomes
    indistinguishable and refinement is pure topology. The empty product is 1,
    so `pair = base[dst] * 1 + 0` degenerates to `base[dst]` with no
    special-casing anywhere."""
    cfg = base(edge_attributes=(), edge_codes={})
    assert cfg.edge_radices == ()
    assert cfg.n_edge_types == 1


def test_edge_attributes_and_edge_codes_must_agree():
    """A named attribute with no code table, or a table for an unnamed
    attribute, is config drift -- it would silently change column count or
    ordering. Both directions raise."""
    with pytest.raises(ValueError, match="edge_attributes"):
        base(
            edge_attributes=("bond_type", "conjugated"),
            edge_codes={"bond_type": {"SINGLE": 0}},
        )
    with pytest.raises(ValueError, match="edge_attributes"):
        base(
            edge_attributes=("bond_type",),
            edge_codes={"bond_type": {"SINGLE": 0}, "conjugated": {"True": 0}},
        )


def test_edge_attributes_enter_schema_version():
    """Two models whose edge columns mean different things must not merge,
    even when every code table is identical (design.md 9.2)."""
    a = base(edge_attributes=("bond_type",), edge_codes={"bond_type": {"SINGLE": 0}})
    b = base(edge_attributes=("conjugated",), edge_codes={"conjugated": {"SINGLE": 0}})
    assert a.schema_version != b.schema_version


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


_REF_CONFIG = SieveConfig(
    target_dim=1,
    attribute_levels=(("element",),),
    attribute_codes={"element": {"C": 0, "H": 1}},
    edge_codes={"bond_type": {"SINGLE": 0, "DOUBLE": 1}},
    max_wl_depth=3,
)


def _ref_config(**kw):
    """``replace`` over a frozen base, as ``tests/helpers.simple_config``
    does. Unpacking a plain ``**base`` dict instead loses each field's own
    type, so every keyword lands as the dict's value union."""
    return replace(_REF_CONFIG, **kw)


def test_stereo_defaults_off_and_leaves_the_digest_untouched():
    # This literal is today's digest for the reference config above. A fitted
    # model on disk carries this digest; changing it silently invalidates
    # every one of them.
    assert (
        _ref_config().schema_version
        == "f54a104c61946eef939179d20cf473cd1a1e42b778fdc115599472d71eb8e4b0"
    )
    assert _ref_config().stereo == ()
    assert _ref_config().stereo_radices == ()


def test_enabling_a_stereo_track_changes_the_digest():
    off, on = _ref_config(), _ref_config(stereo=("cis_trans",))
    assert on.schema_version != off.schema_version


def test_a_stereo_track_widens_the_edge_alphabet_by_four():
    off, on = _ref_config(), _ref_config(stereo=("cis_trans",))
    assert on.n_edge_types == off.n_edge_types * 4
    assert on.stereo_radices == (4,)


def test_an_unknown_stereo_track_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown stereo track"):
        _ref_config(stereo=("helical",))


def test_stereo_with_neighbor_depth_is_refused():
    """The two are not yet designed to compose, and the failure is silent.

    With neighbor_depth set, level_kinds is [ATTR]*a + [WL]*d + [WL_PAIR]*d
    and level_parents sends the WL block to attribute level
    neighbor_depth-1 -- so the LEVEL_WL levels are the *coarse* chain and
    the main chain is LEVEL_WL_PAIR. refine folds the stereo code into
    LEVEL_WL only, so it would land on the coarse chain, whose rounds are
    measured from a shallower base than the k-2 radius rule is derived
    against, while the main chain got no direct code at all. Refuse rather
    than guess which chain it belongs on.
    """
    import pytest

    with pytest.raises(ValueError, match=r"stereo.*neighbor_depth"):
        _ref_config(
            attribute_levels=(("element",), ("aromatic",)),
            attribute_codes={
                "element": {"C": 0, "H": 1},
                "aromatic": {"True": 0, "False": 1},
            },
            neighbor_depth=1,
            stereo=("cis_trans",),
        )


def test_stereo_is_allowed_when_neighbor_depth_normalizes_away():
    """neighbor_depth == len(attribute_levels) means "no coarsening" and is
    normalized to None, so there is no coarse chain to be confused by and
    the combination is legal."""
    cfg = _ref_config(neighbor_depth=1, stereo=("cis_trans",))
    assert cfg.neighbor_depth is None
    assert cfg.stereo == ("cis_trans",)
