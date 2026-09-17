"""Prefix keys for HOSE codes. No optional dependency: these are strings."""

from __future__ import annotations

from experiments.predictors.hose_keys import DELIMITERS, MAX_SPHERES, sphere_prefix

# hosegen.HoseGenerator().get_Hose_codes(ethanol, 0, max_radius=6)
ETHANOL_METHYL = "C-4;HHHC(HHO/H/)//"
PHENOL_CH = "C-3;H*C*C(H,H*C,*C/*CO,H*&/H*&,H)//"


def test_sphere_prefix_cuts_at_each_sphere_boundary():
    assert sphere_prefix(ETHANOL_METHYL, 1) == "C-4;HHHC("
    assert sphere_prefix(ETHANOL_METHYL, 2) == "C-4;HHHC(HHO/"
    assert sphere_prefix(ETHANOL_METHYL, 3) == "C-4;HHHC(HHO/H/"
    assert sphere_prefix(ETHANOL_METHYL, 4) == "C-4;HHHC(HHO/H/)"
    assert sphere_prefix(ETHANOL_METHYL, 5) == "C-4;HHHC(HHO/H/)/"


def test_each_key_is_a_prefix_of_the_next():
    """The property the backoff rests on: a class at radius k+1 sits inside
    exactly one class at radius k."""
    for code in (ETHANOL_METHYL, PHENOL_CH):
        for k in range(1, 8):
            assert sphere_prefix(code, k + 1).startswith(sphere_prefix(code, k))


def test_a_code_shorter_than_k_spheres_is_returned_unchanged():
    assert sphere_prefix("C-4;HHHC(", 5) == "C-4;HHHC("


def test_zero_spheres_is_the_empty_key():
    assert sphere_prefix(ETHANOL_METHYL, 0) == ""


def test_the_delimiter_table_matches_the_generator_cap():
    """12 is hosegen's own ceiling: it indexes sphere_delimiters directly, so
    a thirteenth sphere raises IndexError inside it."""
    assert len(DELIMITERS) == MAX_SPHERES == 12


def test_a_child_key_never_has_two_parents():
    """Spec acceptance check 2, over real codes hardcoded so the fast suite
    can run it without the generator: grouping by the radius-(k+1) key must
    never turn up two distinct radius-k keys."""
    codes = [
        "C-4;HHHC(HHO/H/)//",  # ethanol, methyl C
        "C-4;HHCO(HHH,H//)//",  # ethanol, methylene C
        "C-3;H*C*C(*CO,H*C/H*C,H,H*&/H*&)//",  # phenol, C bearing OH
        "C-3;H*C*C(H,H*C,*C/*CO,H*&/H*&,H)//",  # phenol, ortho CH
        "C-4;HHHC(=OO/,C/HHH)//",  # methyl acetate, acetyl CH3
        "C-3;=OCO(,HHH,C/HHH/)//",  # methyl acetate, carbonyl C
    ]
    for k in range(1, 6):
        parent_of: dict[str, str] = {}
        for code in codes:
            child, parent = sphere_prefix(code, k + 1), sphere_prefix(code, k)
            assert parent_of.setdefault(child, parent) == parent
