"""A plain HOSE-code lookup baseline: the inference rule of Bremser's register
and of NMRShiftDB, scored on this corpus.

Backoff only. There is deliberately no continuation estimate and no shrinkage
here; see docs/superpowers/specs/2026-09-16-hose-baseline-design.md §1. The arm
answers what the incumbent lookup tradition does, and is not an ablation of
Sieve -- HOSE codes carry bond order and aromaticity, which the fitted Sieve
models in this series do not.

Every key a fitted model uses is cut from a single code generated at its own
``max_radius``, never from codes generated separately per radius. A sweep over
``max_radius`` is therefore one instance per setting rather than one deep fit
read at several depths, which is where this arm differs from Sieve and DASH;
spec §7 measures why.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from experiments.data import MoleculeSet
from experiments.predictors import register
from experiments.predictors.base import Prediction, RawPrediction
from experiments.predictors.hose_keys import (
    CANONICAL,
    CLOSURE_TREE,
    CODE,
    MAX_SPHERES,
    STORED,
    check_keys,
    check_numbering,
    sphere_prefix,
)


def canonically_numbered(mol: Any) -> tuple[Any, list[int]]:
    """``mol`` with its atoms renumbered canonically, and ``order``: atom
    ``j`` of the copy is atom ``order[j]`` of ``mol``.

    The ranking reads the constitution only. Stereochemistry, isotopes and
    atom maps are left out because the codes do not carry them: with them in,
    two stereoisomers could be numbered differently and an environment they
    share be spelled in two ways. Atoms the ranking cannot tell apart are
    ordered by their index; where they are equivalent by symmetry, which is
    the usual case, either order gives the same codes.
    """
    from rdkit import Chem

    rank = list(
        Chem.CanonicalRankAtoms(
            mol,
            breakTies=True,
            includeChirality=False,
            includeIsotopes=False,
            includeAtomMaps=False,
        )
    )
    order = sorted(range(mol.GetNumAtoms()), key=rank.__getitem__)
    copy = Chem.RenumberAtoms(mol, order)
    # The generator asks for ring sizes even when it does not write them, and
    # a renumbered molecule does not always keep its ring information.
    Chem.FastFindRings(copy)
    return copy, order


class HoseLookupPredictor:
    name: ClassVar[str] = "hose"

    def __init__(
        self,
        max_radius: int = 5,
        n_min: int = 1,
        numbering: str = STORED,
        keys: str = CODE,
    ) -> None:
        if max_radius < 1:
            raise ValueError(f"max_radius must be >= 1, got {max_radius}")
        if max_radius > MAX_SPHERES:
            raise ValueError(
                f"max_radius must be <= {MAX_SPHERES}, the generator's own "
                f"ceiling, got {max_radius}"
            )
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        self.max_radius = max_radius
        self.n_min = n_min
        self.numbering = check_numbering(numbering)
        self.keys = check_keys(keys)
        if self.keys != CODE and self.numbering != STORED:
            raise ValueError(
                f"keys={self.keys!r} takes no numbering: a tree is the same "
                f"under any numbering of the atoms, got {self.numbering!r}"
            )
        self._tables: list[dict[str, tuple[float, int]]] | None = None
        # Per-key sum of y^2, kept separate from ``_tables`` (mean, count):
        # predict/n_min/the backoff loop read only ``_tables``, so this is
        # pure addition that changes no prediction -- it exists solely to
        # feed the analytic training curve (experiments.analytic).
        self._sumsq: list[dict[str, float]] | None = None
        self._global_mean: float | None = None
        self._global_count: int = 0
        self._global_sumsq: float | None = None
        self._matched_radius: NDArray[np.int64] | None = None

    def _codes(self, mols: list[Any]) -> Iterator[str]:
        """One full code per atom, in the flattened atom order of ``mols``.

        Generated once at ``max_radius``; every shorter key is cut from it by
        ``sphere_prefix``. The generator is imported here, not at module
        scope, so the harness loads without the optional dependency.

        Under the canonical numbering the generator is given a canonically
        numbered copy of each molecule (see ``hose_keys.NUMBERINGS`` for why);
        the codes are still yielded in the stored order of the atoms."""
        from hosegen import HoseGenerator

        generator = HoseGenerator()
        for mol in mols:
            if self.numbering == CANONICAL:
                copy, order = canonically_numbered(mol)
                codes = [""] * len(order)
                for j, i in enumerate(order):
                    codes[i] = generator.get_Hose_codes(
                        copy, j, max_radius=self.max_radius
                    )
                yield from codes
            else:
                for atom in mol.GetAtoms():
                    yield generator.get_Hose_codes(
                        mol, atom.GetIdx(), max_radius=self.max_radius
                    )

    def _keys(self, mols: list[Any]) -> Iterator[list[str]]:
        """Per atom, in the flattened atom order of ``mols``, its keys at
        ``1 .. max_radius`` spheres: prefixes of one code, or trees."""
        if self.keys == CODE:
            for code in self._codes(mols):
                yield [sphere_prefix(code, k) for k in range(1, self.max_radius + 1)]
            return
        from experiments.predictors.hose_trees import tree_keys

        for mol in mols:
            yield from tree_keys(
                mol, self.max_radius, closure=self.keys == CLOSURE_TREE
            )

    def fit(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        del val, rng
        if train.n_conformers == 0:
            raise ValueError("hose requires a non-empty train split")
        target = train.atom_target
        sums: list[defaultdict[str, float]] = [
            defaultdict(float) for _ in range(self.max_radius + 1)
        ]
        sumsqs: list[defaultdict[str, float]] = [
            defaultdict(float) for _ in range(self.max_radius + 1)
        ]
        counts: list[defaultdict[str, int]] = [
            defaultdict(int) for _ in range(self.max_radius + 1)
        ]
        for value, keys in zip(target, self._keys(train.mols), strict=True):
            v = float(value)
            for k in range(1, self.max_radius + 1):
                key = keys[k - 1]
                sums[k][key] += v
                sumsqs[k][key] += v * v
                counts[k][key] += 1
        self._tables = [
            {key: (sums[k][key] / counts[k][key], counts[k][key]) for key in counts[k]}
            for k in range(self.max_radius + 1)
        ]
        self._sumsq = [dict(sumsqs[k]) for k in range(self.max_radius + 1)]
        self._global_mean = float(np.mean(target))
        self._global_count = int(np.size(target))
        self._global_sumsq = float(np.sum(np.square(target)))

    def predict(self, test: MoleculeSet) -> Prediction:
        if self._tables is None or self._global_mean is None:
            raise RuntimeError("fit must be called before predict")
        atom_value: NDArray[np.float64] = np.empty(test.n_atoms, dtype=np.float64)
        matched: NDArray[np.int64] = np.zeros(test.n_atoms, dtype=np.int64)
        for i, keys in enumerate(self._keys(test.mols)):
            value, answered_at = self._global_mean, 0
            for k in range(self.max_radius, 0, -1):
                entry = self._tables[k].get(keys[k - 1])
                if entry is not None and entry[1] >= self.n_min:
                    value, answered_at = entry[0], k
                    break
            atom_value[i] = value
            matched[i] = answered_at
        self._matched_radius = matched
        return Prediction(atom_value=atom_value)

    @property
    def matched_radius(self) -> NDArray[np.int64]:
        """The radius that answered each atom of the most recent ``predict``,
        ``0`` where the global mean did. One entry per atom, in the same order
        as ``Prediction.atom_value``."""
        if self._matched_radius is None:
            raise RuntimeError("predict must be called before matched_radius")
        return self._matched_radius

    def model_state(self) -> Any:
        """This fit's sufficient statistics, as sums, sums of squares and
        counts.

        Sums rather than the means ``_tables`` holds, because sums are what
        add: see ``hose_artifact`` for why a 40-shard merge needs them.
        """
        from experiments.hose_artifact import HoseState

        if self._tables is None or self._global_mean is None or self._sumsq is None:
            raise RuntimeError("fit must be called before model_state")
        tables = [{}] + [
            {
                key: (mean * count, self._sumsq[k][key], count)
                for key, (mean, count) in self._tables[k].items()
            }
            for k in range(1, self.max_radius + 1)
        ]
        return HoseState(
            max_radius=self.max_radius,
            tables=tuple(tables),
            global_sum=self._global_mean * self._global_count,
            global_count=self._global_count,
            global_sumsq=self._global_sumsq,
            numbering=self.numbering,
            keys=self.keys,
        )

    def set_model_state(self, state: Any) -> None:
        """Adopt an already-merged state, skipping ``fit``. The state's own
        radius wins: a state is only valid at the radius it was generated
        at (spec section 7), so silently reading it at another would score a
        model nobody fitted. Its numbering and its kind of keys win for the
        same reason: the queries must be spelled the way the stored keys
        were."""
        self.max_radius = state.max_radius
        self.numbering = state.numbering
        self.keys = state.keys
        self._tables = [{}] + [
            {key: (s / c, c) for key, (s, _qq, c) in state.tables[k].items()}
            for k in range(1, state.max_radius + 1)
        ]
        self._sumsq = [{}] + [
            {key: qq for key, (_s, qq, _c) in state.tables[k].items()}
            for k in range(1, state.max_radius + 1)
        ]
        self._global_mean = state.global_mean
        self._global_count = state.global_count
        self._global_sumsq = state.global_sumsq

    def save_model_state(self, path: str | Any) -> None:
        from experiments.hose_artifact import save_hose_state

        save_hose_state(self.model_state(), path)

    def predict_raw(self, test: MoleculeSet) -> RawPrediction:
        """The lookup output plus a unit std.

        This arm has no per-atom variance: it is a plain mean lookup with no
        continuation estimate and no shrinkage (spec section 1), so there is
        no spread to report. The std exists only for signature parity with
        the other predictors, and the CV driver pairs this arm with
        ``equal_weighted`` normalization, which discards ``raw_std``
        outright. It is never used as a variance.
        """
        pred = self.predict(test)
        return RawPrediction(
            atom_value=pred.atom_value,
            atom_std=np.ones_like(pred.atom_value),
        )

    @staticmethod
    def merge_states(paths: list[str | Any], out: str | Any) -> None:
        """Merge N saved HOSE shard states into one, exact, no re-fit.

        Discovered by name by the generic ``experiments merge-states`` seam,
        the same duck-typed convention ``SievePredictor.merge_states`` follows.

        Every shard must have been fitted at the SAME radius, and
        ``merge_hose_states`` refuses a mismatch rather than reconciling it: a
        radius-6 fit's 3-sphere table is not a radius-3 fit's, because the
        generator re-renders shallower spheres when it goes deeper (spec
        section 7). That is the one way this merge differs from the other two
        predictors', whose shards are radius/depth-agnostic.
        """
        from experiments.hose_artifact import (
            fold_hose_states,
            load_hose_state,
            save_hose_state,
        )

        if not paths:
            raise ValueError("merge_states needs at least one shard path")
        save_hose_state(fold_hose_states(load_hose_state(p) for p in paths), out)


def _build(params: Mapping[str, Any]) -> HoseLookupPredictor:
    return HoseLookupPredictor(**params)


register("hose", _build)
