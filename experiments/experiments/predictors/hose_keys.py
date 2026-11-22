"""Per-radius keys for HOSE codes, cut as prefixes of one full code.

Sphere ``i`` of a HOSE code is closed by ``DELIMITERS[i - 1]``, the positional
sequence ``hosegen.HoseGenerator.sphere_delimiters`` emits, so the first ``k``
spheres are a literal prefix of the whole string. That is what makes a key at
radius ``k + 1`` determine the key at radius ``k``, and hence every class sit
inside exactly one coarser class.

Generating a code separately at each radius does *not* have that property --
the branch grouping is rendered differently, and classes acquire more than one
parent. See §4 of docs/superpowers/specs/2026-09-16-hose-baseline-design.md,
which measures it. This module exists so that mistake cannot be made by
accident.

The prefixes of one code are mutually consistent, but codes generated at
*different* ``max_radius`` are not: ordering a sphere's branches consults what
lies beyond them, so a deeper generation re-renders shallower spheres. Cut
every key a model uses from a single generation. Spec §7.
"""

from __future__ import annotations

# hosegen.HoseGenerator.sphere_delimiters, copied exactly: "(" closes sphere
# 1, ")" closes sphere 4, "/" closes every other. The generator indexes this
# list directly, so 12 is a hard ceiling -- asking it for a thirteenth sphere
# raises a bare IndexError from inside it, with no message of its own.
DELIMITERS: list[str] = ["(", "/", "/", ")"] + ["/"] * 8
MAX_SPHERES: int = len(DELIMITERS)

# How the atoms of a molecule are numbered when its codes are generated.
#
# The generator orders the nodes of a sphere by a score and leaves nodes of
# equal score in the order of their atom indices, so the code of an atom
# depends on the order in which the molecule's atoms happen to be stored: the
# same atom of the same molecule, stored in another order, can get another
# code, miss its class and back off. The Java generators this one was ported
# from (CDK's and NMRShiftDB's) break those ties by a canonical label instead.
# "canonical" restores that: the atoms are renumbered canonically before the
# generator sees them. "stored" is the generator as it comes, kept because the
# runs made before this was known were generated that way.
#
# The two are different vocabularies, since one environment can be spelled
# differently in each, so a fitted state records its numbering and states of
# different numberings do not merge.
STORED = "stored"
CANONICAL = "canonical"
NUMBERINGS: tuple[str, ...] = (STORED, CANONICAL)


def check_numbering(numbering: str) -> str:
    if numbering not in NUMBERINGS:
        raise ValueError(f"numbering must be one of {NUMBERINGS}, got {numbering!r}")
    return numbering


# What a key is cut from. "code" is a HOSE code's string, cut by
# ``sphere_prefix``. "tree" and "closure-tree" are the tree the code is written
# from, without and with its ring closures, keyed with no order at all
# (``predictors.hose_trees``). They are three vocabularies: a state records its
# kind, and states of different kinds do not merge.
CODE = "code"
TREE = "tree"
CLOSURE_TREE = "closure-tree"
KEY_KINDS: tuple[str, ...] = (CODE, TREE, CLOSURE_TREE)


def check_keys(keys: str) -> str:
    if keys not in KEY_KINDS:
        raise ValueError(f"keys must be one of {KEY_KINDS}, got {keys!r}")
    return keys


def sphere_prefix(code: str, k: int) -> str:
    """The first ``k`` spheres of ``code``, as a prefix of it.

    Returns ``code`` unchanged when it holds fewer than ``k`` spheres, so a
    query deeper than a molecule reaches degrades to its deepest description
    rather than raising."""
    pos = 0
    for i in range(k):
        nxt = code.find(DELIMITERS[i], pos)
        if nxt < 0:
            return code
        pos = nxt + 1
    return code[:pos]
