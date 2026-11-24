"""Keys for the HOSE lookup cut from the tree a HOSE code is written from,
instead of from the code's string.

A HOSE code is the text of a breadth-first tree around the central atom. Each
node carries an element, a formal charge and the bond that leads to it, and a
ring atom reached again is written as a closure (``&``), with nothing under
it. Writing the tree as text needs an order for the branches of each sphere,
and hosegen's order consults the atom numbering and nests the branches the way
CDK did before it was fixed (sieve_paper/HOSE.md, sections 3 and 4), so one
tree can be written as several codes. A key here is the tree itself, with the
children of every node taken as a multiset, so no order is needed and two atoms
share a key exactly when they share a tree.

Two trees are offered.

* ``closure=False``: every walk that does not step straight back is followed,
  rings included. This is the unrolled tree of a Sieve label with element and
  formal charge on the atoms and bond types on the bonds, so its classes are
  that label's classes.
* ``closure=True``: the tree a HOSE code is written from. A walk whose atom
  was already written in an earlier sphere is a closure, as in the code. Where
  two walks reach the same atom in the same sphere, HOSE writes the atom under
  the walk that comes first in its order and a closure under the other; that
  choice is the one place where the order reaches into the tree. Here the atom
  is written under every walk that reaches it, marked as a meeting, and
  expanded under each, so the tree needs no order at all.

The atom labels are the code's own: the element symbol and hosegen's charge
code. Bond symbols are hosegen's too, so the trees have exactly the vocabulary
of the codes.

A key at ``k`` spheres is the element symbol of the central atom and a ``-``,
followed by the digests of the tree cut at ``1, ..., k`` spheres, each closed
by the delimiter that closes that sphere in a code. So ``sphere_prefix`` cuts
a key to fewer spheres exactly as it cuts a code, and everything that walks
the backoff chain reads both alike. The element prefix is what the error
breakdown reads from the one-sphere table, as it does from a code.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from experiments.predictors.hose_keys import DELIMITERS, MAX_SPHERES

# hosegen.HoseGenerator.get_bond_symbol, copied exactly: a single bond and any
# type not listed are written as nothing.
_BOND_SYMBOLS = {"AROMATIC": "*", "ZERO": "<", "DOUBLE": "=", "TRIPLE": "%"}


def _bond_symbol(bond: Any) -> str:
    return _BOND_SYMBOLS.get(str(bond.GetBondType()), "")


def _charge_code(atom: Any) -> str:
    """hosegen.HoseGenerator.create_charge_code, copied exactly."""
    q = atom.GetFormalCharge()
    if q == 0:
        return ""
    if abs(q) == 1:
        return "-" if q < 0 else "+"
    return "'" + ("+" if q > 0 else "") + str(q) + "'"


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


@dataclass
class TreeNode:
    """A node of the closure tree. ``kind`` is ``"atom"``, ``"meet"`` (an
    atom reached by more than one walk in its sphere) or ``"closure"`` (an
    atom written in an earlier sphere, as ``&`` in a code)."""

    atom: int
    symbol: str
    charge: str
    bond: str = ""
    kind: str = "atom"
    parent: int = -1
    children: list[TreeNode] = field(default_factory=list)

    def label(self) -> str:
        # A closure is written as its bond, ``&`` and its charge, as in a code.
        if self.kind == "closure":
            return f"c{self.bond}&{self.charge}"
        return f"{self.kind[0]}{self.bond}{self.symbol}{self.charge}"


def _check_radius(radius: int) -> None:
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if radius > MAX_SPHERES:
        raise ValueError(
            f"radius must be <= {MAX_SPHERES}, the number of sphere "
            f"delimiters, got {radius}"
        )


def _chain(symbol: str, digest_at: Callable[[int], str], radius: int) -> list[str]:
    """The keys at ``1 .. radius`` spheres, each the one before it plus the
    digest of the tree cut one sphere deeper and that sphere's delimiter."""
    keys, key = [], f"{symbol}-"
    for k in range(1, radius + 1):
        key += digest_at(k) + DELIMITERS[k - 1]
        keys.append(key)
    return keys


def closure_tree(mol: Any, center: int, radius: int) -> TreeNode:
    """The tree of ``radius`` spheres around atom ``center``, with ring
    closures and the symmetric meeting rule described in the module
    docstring."""
    _check_radius(radius)
    atoms = list(mol.GetAtoms())
    root = TreeNode(center, atoms[center].GetSymbol(), _charge_code(atoms[center]))
    written = {center: 0}
    layer = [root]
    for sphere in range(1, radius + 1):
        reached: list[TreeNode] = []
        for node in layer:
            if node.kind == "closure":
                continue
            for bond in atoms[node.atom].GetBonds():
                other = bond.GetOtherAtomIdx(node.atom)
                if other == node.parent:
                    continue
                child = TreeNode(
                    other,
                    atoms[other].GetSymbol(),
                    _charge_code(atoms[other]),
                    _bond_symbol(bond),
                    parent=node.atom,
                )
                node.children.append(child)
                reached.append(child)
        walks: dict[int, int] = {}
        for child in reached:
            walks[child.atom] = walks.get(child.atom, 0) + 1
        for child in reached:
            if written.get(child.atom, sphere) < sphere:
                child.kind = "closure"
            elif walks[child.atom] > 1:
                child.kind = "meet"
        for child in reached:
            if child.kind != "closure":
                written.setdefault(child.atom, sphere)
        layer = reached
    return root


def _closure_keys(mol: Any, center: int, radius: int) -> list[str]:
    root = closure_tree(mol, center, radius)
    memo: dict[tuple[int, int], str] = {}

    def digest(node: TreeNode, depth: int) -> str:
        key = (id(node), depth)
        if key not in memo:
            below = sorted(digest(c, depth - 1) for c in node.children) if depth else []
            memo[key] = _digest(node.label() + "(" + ",".join(below) + ")")
        return memo[key]

    return _chain(root.symbol, lambda k: digest(root, k), radius)


def tree_keys(mol: Any, radius: int, *, closure: bool) -> list[list[str]]:
    """``keys[i][k - 1]``: the key of atom ``i`` at ``k`` spheres, for ``k = 1
    .. radius``. A key at ``k`` spheres determines the key at fewer, since the
    tree of fewer spheres is the deeper tree cut short."""
    _check_radius(radius)
    if closure:
        return [_closure_keys(mol, i, radius) for i in range(mol.GetNumAtoms())]
    atoms = list(mol.GetAtoms())
    label = [f"a{a.GetSymbol()}{_charge_code(a)}" for a in atoms]
    steps = [
        [(b.GetOtherAtomIdx(a.GetIdx()), _bond_symbol(b)) for b in a.GetBonds()]
        for a in atoms
    ]
    # Without closures the subtree under a walk that arrived at ``atom`` from
    # ``parent`` depends on nothing else, so it is shared by every center.
    memo: dict[tuple[int, int, int], str] = {}

    def digest(atom: int, parent: int, bond: str, depth: int) -> str:
        key = (atom, parent, depth)
        if key not in memo:
            below = (
                sorted(
                    digest(other, atom, sym, depth - 1)
                    for other, sym in steps[atom]
                    if other != parent
                )
                if depth
                else []
            )
            memo[key] = _digest(f"a{bond}{label[atom][1:]}(" + ",".join(below) + ")")
        return memo[key]

    return [
        _chain(atoms[i].GetSymbol(), lambda k, i=i: digest(i, -1, "", k), radius)
        for i in range(len(atoms))
    ]


def root_keys(mol: Any) -> list[str]:
    """``roots[i]``: the class of atom ``i`` before any sphere, its element
    symbol and hosegen's charge code. This is the attribute level of the Sieve
    chain whose unrolled tree the plain tree is, and the parent of every
    one-sphere tree, which determines it: the root of a tree carries both."""
    return [f"{a.GetSymbol()}{_charge_code(a)}" for a in mol.GetAtoms()]
