"""T11: a per-ATOM sigma-profile predictor on Chemprop's ``MolAtomBondMPNN``.

Where T10 (``predictors/chemprop_dmpnn.py``) pools a molecule's atoms into one
embedding and predicts one 51-bin profile per molecule, this predicts a 51-bin
profile for *every atom*, off that atom's own message-passing hidden state.
Molecule-level results are then the harness's ordinary ``roll_up`` of those
atom predictions -- so the molecule numbers here are a pure readout of atom
quality, not a separately-trained head.

Everything else follows T10's default configuration -- raw (unshifted,
unnormalized) profile bins, min-max target scaling fit on the training split
only, a non-negative output activation, plain MSE -- so that T11-vs-T10
isolates a single variable (atom-level vs molecule-level head), and
T11-vs-DASH (``predictors/dash.py``, the only other atom-level predictor)
isolates another (learned vs tree-node-averaged).

THREE DELIBERATE DEPARTURES FROM T10, all forced by predicting per atom:

1. **The atom featurizer.** T10 uses ``PaperAtomFeaturizer``, the COSMO-NET
   paper's own Table 1: an 8-element one-hot over {C,N,O,S,F,P,Cl,Br}. That
   works there only because a molecule-level model runs on the implicit-H
   (heavy-atom-only) graph, where hydrogen is never a node. Here the graph
   must keep explicit hydrogens -- the store has atom-level truth for every
   atom, H included -- and **56.8% of chaos-store atoms are hydrogen**, which
   that vocabulary cannot represent at all (H would one-hot to all zeros,
   indistinguishable from the boron/silicon/etc. that are also outside it).
   So this uses chemprop's own ``MultiHotAtomFeaturizer.v2()`` (72-dim,
   atomic numbers 1-36 plus 53), which covers every element in chaos-store
   except Sb and Te.

2. **Explicit hydrogens.** ``make_mol(smi, keep_h=True, add_h=False,
   reorder_atoms=True)``, not T10's plain ``Chem.MolFromSmiles(smi)``.

3. **The output activation: squared (``x**2``), not softplus.** Not a
   preference -- softplus *structurally cannot train* on atom-level targets,
   and collapses to identically-zero output. Atom profiles are **81.4% exact
   zeros** (each atom occupies only ~9.5 of the 51 bins, whereas a molecule
   profile is dense: 50.2% zeros, 25.4 bins populated), so MSE drives most
   outputs toward exactly zero -- which softplus reaches only as its
   pre-activation goes to -inf, exactly where its own derivative
   ``sigmoid(x)`` vanishes. The head dies and never recovers. Measured on a
   20-molecule overfit (true mean atom area 7.676):

       softplus     w1 6.717   predicted area 0.000   <- dead, cannot overfit
       squared      w1 0.878   predicted area 7.859
       abs          w1 0.947   predicted area 7.939
       plain linear w1 0.825   predicted area 7.786   <- 13,110 negative bins

   ``x**2`` keeps non-negativity exactly as structural as softplus while
   making a target of exactly zero reachable at a finite pre-activation, and
   costs essentially nothing against an unconstrained linear head. Selectable
   via ``output_activation``; ``"softplus"`` remains available so the
   collapse above stays reproducible.

ATOM ORDERING -- the one thing that silently corrupts everything if wrong.
The store's atom-level truth is a flat concatenation in molecule order, and
within a molecule it is COSMO-file order, which the store's atom-mapped SMILES
encode as atom-map numbers (see ``data.py`` and ``dash.py::_atom_paths``).
chemprop's own ``make_mol(..., reorder_atoms=True)`` renumbers atoms by
exactly that atom-map order, so **no manual permutation is needed** -- unlike
``dash.py``, which has to do its own ``np.argsort(GetAtomMapNum())``. Verified
on real chaos-store molecules, but never assumed: ``_datapoints`` asserts both
the atom count and the map-number ordering per molecule, and there is a test
checking the resulting order against the store's own ``element`` column.

Two INDEPENDENT misalignments are possible, and they need different guards:

- *within* a molecule -- handled by the per-molecule asserts above.
- *between* molecules -- ``build_dataloader`` shuffles MOLECULES, and each
  molecule's atoms travel with it as a contiguous block, so a shuffled loader
  returns the exact same ``(n_atoms, 51)`` array shape with molecule blocks
  permuted. **A shape check cannot detect this.** ``predict_atoms`` therefore
  passes ``shuffle=False`` and, as a tripwire, verifies the per-molecule atom
  counts recovered from each batch's own ``bmg.batch`` against
  ``MoleculeSet.num_atoms``, element for element.

Non-negativity is structural here for the same reason as in T10: every one of
the 51 atom-profile bins has a training minimum of exactly 0, so ``y_min >=
0``, and ``activation(x) * scale + y_min >= 0`` always, for any of the
supported activations. The FFN class itself comes from
``chemprop_dmpnn.nonneg_regression_ffn_class`` so the invariant -- activation
applied to the raw output, *before* unscaling -- is defined exactly once and
shared with T10.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sieve_experiments.data import (
    DEFAULT_GRID,
    MoleculeSet,
    SigmaGridSpec,
    load_atom_truth,
)
from sieve_experiments.predictors import register
from sieve_experiments.predictors.base import (
    VALID_CHARGE_RECONCILIATION,
    AtomPrediction,
    AtomPredictor,
)
from sieve_experiments.predictors.chemprop_dmpnn import (
    VALID_OUTPUT_ACTIVATIONS,
    minmax_apply,
    minmax_fit,
    nonneg_regression_ffn_class,
)

# chemprop's own MultiHotAtomFeaturizer.v2() / MultiHotBondFeaturizer(),
# asserted against the real classes in the optional test suite rather than
# trusted -- a silent width change would otherwise mis-size the encoder.
ATOM_FDIM = 72
BOND_FDIM = 14


def atom_prediction_from_profile(
    profile: NDArray[np.floating], sigma_values: NDArray[np.floating]
) -> AtomPrediction:
    """Atom-level counterpart of ``chemprop_dmpnn.prediction_from_profile``:
    area and charge are *derived* from the predicted profile, never predicted
    as their own quantities -- ``area = sum(p)``, ``charge = p @ sigma``, the
    same convention every profile-predicting model in this harness uses (and
    the same one ``dash.py``'s ``profile_mode="raw"`` uses).
    """
    profile = np.asarray(profile, dtype=np.float64)
    sigma_values = np.asarray(sigma_values, dtype=np.float64)
    return AtomPrediction(
        atom_profile=profile,
        atom_area=profile.sum(axis=1),
        atom_charge=profile @ sigma_values,
    )


def _build_model(
    *,
    hidden_size: int,
    depth: int,
    dropout: float,
    ffn_n_layers: int,
    n_tasks: int,
    output_transform,
    output_activation: str = "squared",
):
    """Lazy: only imports chemprop when actually called.

    ``MolAtomBondMPNN`` with only an atom head -- ``agg`` and the molecule/
    bond predictors stay ``None``, which chemprop explicitly allows (it only
    requires ``agg`` when a ``mol_predictor`` is given). ``MABBondMessagePassing``
    is the message-passing class this model requires; T10's
    ``BondMessagePassing`` is a different class and does not fit here, since
    this one must return per-vertex hidden states rather than an aggregated
    molecule vector.
    """
    from chemprop.models import MolAtomBondMPNN
    from chemprop.nn.message_passing import MABBondMessagePassing

    NonNegativeRegressionFFN = nonneg_regression_ffn_class(output_activation)

    message_passing = MABBondMessagePassing(
        d_v=ATOM_FDIM,
        d_e=BOND_FDIM,
        d_h=hidden_size,
        depth=depth,
        dropout=dropout,
        return_vertex_embeddings=True,
        # no bond-level head, so chemprop skips building W_eo entirely
        return_edge_embeddings=False,
    )
    # Read the head's input width off the message-passing block rather than
    # hardcoding hidden_size -- chemprop's own build_MAB_model does exactly
    # this, and it stays correct if per-atom descriptors (d_vd) are ever added.
    atom_input_dim = message_passing.output_dims[0]
    atom_predictor = NonNegativeRegressionFFN(
        n_tasks=n_tasks,
        input_dim=atom_input_dim,
        hidden_dim=hidden_size,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=None,  # chemprop's own MSE, as in T10's default loss_mode
        output_transform=output_transform,
    )
    return MolAtomBondMPNN(
        message_passing,
        agg=None,
        atom_predictor=atom_predictor,
    )


class ChempropAtomPredictor(AtomPredictor):
    """Per-atom D-MPNN sigma-profile predictor -- see the module docstring.

    ``store``/``scheme`` are needed here (not just in the run config's
    ``data`` section) because atom-level truth for the training split has to
    be loaded independently -- ``load_molecule_set`` only ever supplies
    molecule-level truth. Same reason ``DASHBackoffPredictor`` needs them.

    ``charge_reconciliation`` defaults to ``"none"`` so the reported charge
    metrics are the model's raw output, directly comparable to T10's.
    ``"std_weighted"`` (DASH's setting) is accepted but needs an
    ``atom_charge_std`` this model does not predict, so it would raise in
    ``reconcile_charge``; ``"shift"`` works.
    """

    name = "chemprop_atom"

    def __init__(
        self,
        *,
        store: str,
        scheme: str,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        ffn_n_layers: int = 2,
        batch_size: int = 64,
        max_epochs: int = 100,
        charge_reconciliation: str = "none",
        output_activation: str = "squared",
        stores_root: str | None = None,
        grid: SigmaGridSpec = DEFAULT_GRID,
    ) -> None:
        # Validated here, at construction, so a bad config fails immediately
        # and without needing chemprop installed (fast-suite tested).
        if charge_reconciliation not in VALID_CHARGE_RECONCILIATION:
            raise ValueError(
                f"charge_reconciliation must be one of "
                f"{VALID_CHARGE_RECONCILIATION}, got {charge_reconciliation!r}"
            )
        if output_activation not in VALID_OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {VALID_OUTPUT_ACTIVATIONS}, "
                f"got {output_activation!r}"
            )
        self.store = store
        self.scheme = scheme
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.ffn_n_layers = ffn_n_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.charge_reconciliation = charge_reconciliation
        self.output_activation = output_activation
        self.stores_root = stores_root
        self.grid = grid
        self._model: Any = None
        self._y_min: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None

    def _featurizer(self):
        from chemprop.featurizers import MultiHotAtomFeaturizer, MultiHotBondFeaturizer
        from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer

        return SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=MultiHotAtomFeaturizer.v2(),
            bond_featurizer=MultiHotBondFeaturizer(),
        )

    def _datapoints(self, mset: MoleculeSet, atom_y: NDArray[np.float64] | None):
        """One ``MolAtomBondDatapoint`` per molecule, with this molecule's
        slice of the flat atom-target array.

        The two asserts here are the within-molecule half of the ordering
        guard described in the module docstring. They are cheap and they turn
        the single worst failure mode (silently training on atom targets
        paired with the wrong graph nodes) into a loud error.
        """
        from chemprop.data import MolAtomBondDatapoint
        from chemprop.utils import make_mol

        blocks: list[NDArray[np.float64] | None]
        if atom_y is None:
            blocks = [None] * mset.n_molecules
        else:
            if len(atom_y) != mset.n_atoms:
                raise ValueError(
                    f"atom_y has {len(atom_y)} rows but the molecule set has "
                    f"{mset.n_atoms} atoms"
                )
            blocks = list(np.split(atom_y, np.cumsum(mset.num_atoms)[:-1]))

        datapoints = []
        for smi, n_atoms, block in zip(
            mset.smiles, mset.num_atoms, blocks, strict=True
        ):
            mol = make_mol(smi, keep_h=True, add_h=False, reorder_atoms=True)
            if mol is None:
                raise ValueError(f"unparseable SMILES: {smi[:60]}")
            if mol.GetNumAtoms() != int(n_atoms):
                raise ValueError(
                    f"atom count mismatch for {smi[:60]}: rdkit parsed "
                    f"{mol.GetNumAtoms()}, store says {int(n_atoms)}"
                )
            maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
            if maps != sorted(maps):
                raise ValueError(
                    f"make_mol(reorder_atoms=True) did not sort {smi[:60]} by "
                    f"atom map number (got {maps[:8]}...) -- atom-level targets "
                    "would be paired with the wrong graph nodes"
                )
            datapoints.append(MolAtomBondDatapoint(mol, atom_y=block))
        return datapoints

    def _dataset(self, mset: MoleculeSet, atom_y: NDArray[np.float64] | None):
        from chemprop.data import MolAtomBondDataset

        return MolAtomBondDataset(self._datapoints(mset, atom_y), self._featurizer())

    def _atom_truth(self, mset: MoleculeSet):
        kwargs: dict[str, Any] = {}
        if self.stores_root is not None:
            # predictor.params comes straight from YAML, so this arrives as a
            # plain str -- load_atom_truth does `stores_root / store_name`.
            kwargs["stores_root"] = Path(self.stores_root)
        return load_atom_truth(
            self.store,
            scheme=self.scheme,
            smiles=mset.smiles,
            num_atoms=mset.num_atoms,
            **kwargs,
        )

    def fit_atoms(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        import lightning.pytorch as pl
        from chemprop.data import build_dataloader
        from chemprop.nn.transforms import UnscaleTransform

        if not np.allclose(train.grid.values, self.grid.values):
            raise ValueError(
                "the molecule set's sigma grid does not match this "
                "predictor's configured grid"
            )
        pl.seed_everything(int(rng.integers(0, 2**31 - 1)), workers=True)

        atom_profile, _, _ = self._atom_truth(train)
        # Fit the target scaling on the TRAINING split only -- as in T10.
        # Bins that are constant across all training atoms (the extreme sigma
        # bins genuinely are, at atom level) are handled by minmax_fit's own
        # zero-scale guard.
        y_min, scale = minmax_fit(atom_profile)
        self._y_min, self._scale = y_min, scale

        train_loader = build_dataloader(
            self._dataset(train, minmax_apply(atom_profile, y_min, scale)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = None
        if val.n_molecules:
            val_profile, _, _ = self._atom_truth(val)
            val_loader = build_dataloader(
                self._dataset(val, minmax_apply(val_profile, y_min, scale)),
                batch_size=self.batch_size,
                shuffle=False,
            )

        self._model = _build_model(
            hidden_size=self.hidden_size,
            depth=self.depth,
            dropout=self.dropout,
            ffn_n_layers=self.ffn_n_layers,
            n_tasks=self.grid.num_points,
            output_transform=UnscaleTransform(mean=y_min, scale=scale),
            output_activation=self.output_activation,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            # devices=1 is load-bearing: "auto" silently launches multi-GPU
            # DDP on this machine and splits the eval batch across ranks --
            # see chemprop_dmpnn.py, where that was first hit.
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(self._model, train_loader, val_loader)

    def predict_atoms(self, test: MoleculeSet) -> AtomPrediction:
        import lightning.pytorch as pl
        from chemprop.data import build_dataloader

        if self._model is None:
            raise RuntimeError("fit_atoms must be called before predict_atoms")

        loader = build_dataloader(
            self._dataset(test, None),
            batch_size=self.batch_size,
            # Load-bearing: shuffling permutes MOLECULE blocks, which leaves
            # the output shape identical -- see the module docstring.
            shuffle=False,
        )
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        batches = trainer.predict(self._model, loader)
        if not batches:
            raise RuntimeError("chemprop returned no prediction batches")

        # MolAtomBondMPNN returns [mol_preds, atom_preds, bond_preds] per
        # batch; only the atom slot is populated here.
        profile = np.concatenate(
            [np.asarray(batch[1]) for batch in batches], axis=0
        ).astype(np.float64)
        self._check_molecule_order(loader, test)
        if profile.shape != (test.n_atoms, self.grid.num_points):
            raise RuntimeError(
                f"expected atom predictions of shape "
                f"({test.n_atoms}, {self.grid.num_points}), got {profile.shape}"
            )
        return atom_prediction_from_profile(profile, self.grid.values)

    @staticmethod
    def _check_molecule_order(loader: Any, test: MoleculeSet) -> None:
        """Tripwire for a permuted (shuffled) predict loader.

        Recovers each batch's per-molecule atom counts from its own
        ``bmg.batch`` -- the per-atom molecule index -- and checks the
        concatenated sequence against ``test.num_atoms``. A shuffled loader
        yields the same total atom count and the same array shape, so this is
        the only cheap check that actually notices; any pair of
        differently-sized molecules swapped changes the sequence. It is a
        tripwire, not a proof (swapping two equal-sized molecules is
        invisible to it) -- ``shuffle=False`` remains the real mechanism.
        """
        sizes: list[int] = []
        for batch in loader:
            counts = np.bincount(np.asarray(batch.bmg.batch))
            sizes.extend(int(c) for c in counts)
        expected = [int(n) for n in test.num_atoms]
        if sizes == expected:
            return
        if len(sizes) != len(expected):
            detail = f"got {len(sizes)} molecules, expected {len(expected)}"
        else:
            first = next(
                i
                for i, (a, b) in enumerate(zip(sizes, expected, strict=True))
                if a != b
            )
            detail = (
                f"first mismatch at molecule {first}: "
                f"{sizes[first]} atoms vs {expected[first]} expected"
            )
        raise RuntimeError(
            "predicted atom blocks do not line up with the molecule set's own "
            f"atom counts -- the predict dataloader may be shuffling ({detail})"
        )


def _build(params: Mapping[str, Any]) -> ChempropAtomPredictor:
    return ChempropAtomPredictor(**params)


register("chemprop_atom", _build)
