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
