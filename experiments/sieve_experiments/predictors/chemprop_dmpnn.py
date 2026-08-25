"""T10: a from-scratch D-MPNN sigma-profile predictor on Chemprop, reproducing
COSMO-NET's published architecture (Naseri Boroujeni et al., *Mol. Syst. Des.
Eng.*, 2026, DOI 10.1039/d6me00088f) -- not COSMO-NET-Paper's own repo, whose
training script, published checkpoint, and published results are mutually
inconsistent (three independently-confirmed gaps; see pins.toml's
``[chemprop]`` notes and README.md's T10 section for the full account).

Two layers, deliberately split so the featurization is testable without the
optional dependency (mirrors predictors/dash.py's ``fit_backoff``/
``DASHBackoffPredictor`` split):

- ``PaperAtomFeaturizer``/``PaperBondFeaturizer``, ``minmax_*``,
  ``prediction_from_profile`` -- pure numpy + rdkit, no chemprop import at
  module scope. Fast-suite tested
  (experiments/tests/test_experiment_predictor_chemprop.py).
- ``ChempropDMPNNPredictor`` -- wires those onto a real chemprop ``MPNN`` and
  a lightning ``Trainer``. Needs the ``chemprop`` extra
  (``uv sync --extra chemprop``). Optional-data tested only.

Architecture, from the paper's Table 1/2 and Section 2.2.2 (message_passing_
steps=3, hidden_size=51, ffn_num_layers=1, dropout=0.1, softplus output,
MSE loss, batch_size=64, n_epochs=100, per-task min-max target scaling from
training-set statistics only):

- Atom features (35-dim, Table 1): element one-hot {C,N,O,S,F,P,Cl,Br} (8),
  degree one-hot 0-5 (6), formal charge one-hot {-2..2} (5), hybridization
  one-hot {sp,sp2,sp3,sp3d,sp3d2} (5), aromaticity (1), total-H one-hot 0-4
  (5), chirality one-hot 0-3 (4), atomic mass x 0.01 (1). No "unknown" pad
  bit -- the paper's table gives an exact, fixed width, and chemprop's own
  ``MultiHotAtomFeaturizer`` cannot be parameterized down to 35 (it always
  reserves one pad slot per one-hot block, landing at 41). An element outside
  this 8-element vocabulary (chaos-store has B, Si, Ge, Sb, Te too -- the
  same gap DASH's own vocabulary hits) one-hots to all-zero: a documented
  deviation, not a crash.
- Bond features (12-dim, Table 2): bond type one-hot {single,double,triple,
  aromatic} (4), conjugated (1), in-ring (1), stereo one-hot over RDKit's own
  6-value ``BondStereo`` enum (6) -- exhaustive, no pad needed here.
- ``ffn_num_layers=1`` means one FFN layer total (deepchem/paper convention:
  a single ``Linear(hidden_size, n_tasks)``, no hidden layer at all). This is
  ``n_layers=0`` in chemprop's own terminology, NOT ``n_layers=1`` --
  chemprop's ``MLP.build`` treats ``n_layers`` as the count of *additional*
  hidden layers beyond the direct input->output projection (confirmed by
  inspecting ``MLP.build``: ``n_layers=0`` yields exactly one ``Linear``,
  ``n_layers=1`` yields two). Using chemprop's ``n_layers=1`` here would
  silently build a deeper FFN than the paper describes.
- Softplus sits on the FFN's raw (scaled-space) output, *before*
  ``output_transform`` (min-max unscaling) -- see ``SoftplusRegressionFFN``.
  Since min-max scaling's ``y_min >= 0`` for every profile bin (a real
  physical density), ``softplus(x) > 0`` composed with unscaling guarantees
  ``p(sigma) > 0`` for every prediction, structurally -- not a post-hoc clip,
  which is exactly the demonstration COSMO-NET-Paper's own repo could not
  produce (its published results have 0% negative bins; its published script
  has no non-negativity enforcement anywhere in its prediction path).
- chemprop's ``normalize_targets()`` is z-score only; the paper specifies
  min-max (its eqn 17), so this module bypasses it and builds an
  ``UnscaleTransform`` directly from training-set min/max
  (``UnscaleTransform.forward`` is a documented no-op while
  ``self.training``, so MSE loss is computed in scaled space during training,
  matching the paper).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sieve_experiments.data import DEFAULT_GRID, MoleculeSet, SigmaGridSpec
from sieve_experiments.predictors import register
from sieve_experiments.predictors.base import MoleculePredictor, Prediction

ATOM_FDIM = 35
BOND_FDIM = 12

_ELEMENTS = ("C", "N", "O", "S", "F", "P", "Cl", "Br")
_DEGREES = tuple(range(6))
_FORMAL_CHARGES = (-2, -1, 0, 1, 2)
_HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
_TOTAL_HS = tuple(range(5))
_CHIRAL_TAGS = (0, 1, 2, 3)

_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
_STEREO = tuple(range(6))


def _one_hot(value: object, choices: tuple) -> NDArray[np.float32]:
    """A ``len(choices)``-wide one-hot; a value outside ``choices`` one-hots
    to all-zero rather than raising or reserving an "unknown" slot -- see
    the module docstring.
    """
    out = np.zeros(len(choices), dtype=np.float32)
    if value in choices:
        out[choices.index(value)] = 1.0
    return out


class PaperAtomFeaturizer:
    """The 35-dim atom feature vector from COSMO-NET's Table 1, exactly.

    Duck-typed to chemprop's ``VectorFeaturizer`` protocol (``__call__`` +
    ``__len__``) rather than subclassing it, so this class stays importable
    and unit-testable without chemprop installed.
    """

    def __len__(self) -> int:
        return ATOM_FDIM

    def __call__(self, a: Any) -> NDArray[np.float32]:
        if a is None:
            return np.zeros(ATOM_FDIM, dtype=np.float32)
        return np.concatenate(
            [
                _one_hot(a.GetSymbol(), _ELEMENTS),
                _one_hot(a.GetTotalDegree(), _DEGREES),
                _one_hot(a.GetFormalCharge(), _FORMAL_CHARGES),
                _one_hot(a.GetHybridization().name, _HYBRIDIZATIONS),
                np.array([float(a.GetIsAromatic())], dtype=np.float32),
                _one_hot(a.GetTotalNumHs(), _TOTAL_HS),
                _one_hot(int(a.GetChiralTag()), _CHIRAL_TAGS),
                np.array([0.01 * a.GetMass()], dtype=np.float32),
            ]
        )


class PaperBondFeaturizer:
    """The 12-dim bond feature vector from COSMO-NET's Table 2, exactly."""

    def __len__(self) -> int:
        return BOND_FDIM

    def __call__(self, b: Any) -> NDArray[np.float32]:
        if b is None:
            return np.zeros(BOND_FDIM, dtype=np.float32)
        return np.concatenate(
            [
                _one_hot(b.GetBondType().name, _BOND_TYPES),
                np.array([float(b.GetIsConjugated())], dtype=np.float32),
                np.array([float(b.IsInRing())], dtype=np.float32),
                _one_hot(int(b.GetStereo()), _STEREO),
            ]
        )


def minmax_fit(
    y: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-column ``(min, scale)`` from ``y`` -- call on training targets only.

    A constant column (``scale == 0``) is guarded to ``1.0`` rather than
    dividing by zero: ``pyproject.toml`` promotes ``RuntimeWarning`` to an
    error (see ``metrics.py``'s ``normalize_rows`` for the same discipline).
    """
    y = np.asarray(y, dtype=np.float64)
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    scale = y_max - y_min
    scale = np.where(scale == 0, 1.0, scale)
    return y_min, scale


def minmax_apply(
    y: NDArray[np.floating], y_min: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    return (np.asarray(y, dtype=np.float64) - y_min) / scale


def minmax_invert(
    y_scaled: NDArray[np.floating],
    y_min: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(y_scaled, dtype=np.float64) * scale + y_min


def prediction_from_profile(
    profile: NDArray[np.floating], sigma_values: NDArray[np.floating]
) -> Prediction:
    """``mol_profile`` plus its derived area (row sum) and charge (first
    moment) -- the same convention as ``predictors/cosmonet.py``.
    """
    profile = np.asarray(profile, dtype=np.float64)
    mol_area = profile.sum(axis=1)
    mol_charge_raw = profile @ np.asarray(sigma_values, dtype=np.float64)
    return Prediction(
        mol_profile=profile, mol_area=mol_area, mol_charge_raw=mol_charge_raw
    )


def _build_model(
    *, hidden_size: int, depth: int, dropout: float, n_tasks: int, output_transform
):
    """Lazy: only imports chemprop when actually called."""
    import torch.nn.functional as F
    from chemprop.models import MPNN
    from chemprop.nn.agg import SumAggregation
    from chemprop.nn.message_passing import BondMessagePassing
    from chemprop.nn.predictors import RegressionFFN

    class SoftplusRegressionFFN(RegressionFFN):
        """``RegressionFFN`` with softplus forced on the FFN's raw output,
        before unscaling -- see the module docstring for why this makes
        non-negativity structural rather than a post-hoc clip.
        """

        def forward(self, Z):
            return self.output_transform(F.softplus(self.ffn(Z)))

        train_step = forward

    message_passing = BondMessagePassing(
        d_v=ATOM_FDIM, d_e=BOND_FDIM, d_h=hidden_size, depth=depth, dropout=dropout
    )
    agg = SumAggregation()
    # n_layers=0, not 1: chemprop's MLP.build counts *additional* hidden
    # layers beyond the direct projection -- see the module docstring.
    predictor = SoftplusRegressionFFN(
        n_tasks=n_tasks,
        input_dim=hidden_size,
        n_layers=0,
        dropout=dropout,
        output_transform=output_transform,
    )
    return MPNN(message_passing, agg, predictor)


class ChempropDMPNNPredictor(MoleculePredictor):
    """Trains and predicts the paper's D-MPNN architecture in-process."""

    name = "chemprop_dmpnn"

    def __init__(
        self,
        *,
        store: str,
        scheme: str,
        hidden_size: int = 51,
        depth: int = 3,
        dropout: float = 0.1,
        batch_size: int = 64,
        max_epochs: int = 100,
        grid: SigmaGridSpec = DEFAULT_GRID,
    ) -> None:
        self.store = store
        self.scheme = scheme
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.grid = grid
        self._model: Any = None

    def _featurizer(self):
        from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer

        return SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=PaperAtomFeaturizer(), bond_featurizer=PaperBondFeaturizer()
        )

    def _dataset(self, mset: MoleculeSet, y: NDArray[np.float64] | None):
        from chemprop.data import MoleculeDataset
        from chemprop.data.datapoints import MoleculeDatapoint
        from rdkit import Chem

        datapoints = [
            MoleculeDatapoint(
                Chem.MolFromSmiles(smi), y=(y[i] if y is not None else None)
            )
            for i, smi in enumerate(mset.smiles)
        ]
        return MoleculeDataset(datapoints, self._featurizer())

    def fit_molecules(
        self, train: MoleculeSet, val: MoleculeSet, *, rng: np.random.Generator
    ) -> None:
        import lightning.pytorch as pl
        from chemprop.data import build_dataloader
        from chemprop.nn.transforms import UnscaleTransform

        if train.mol_profile is None:
            raise ValueError("chemprop_dmpnn requires train.mol_profile")
        if not np.allclose(train.grid.values, self.grid.values):
            raise ValueError("train.grid does not match this predictor's grid")

        pl.seed_everything(int(rng.integers(0, 2**31 - 1)), workers=True)

        y_min, scale = minmax_fit(train.mol_profile)
        train_y = minmax_apply(train.mol_profile, y_min, scale)
        train_dset = self._dataset(train, train_y)
        train_loader = build_dataloader(
            train_dset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if val.n_molecules and val.mol_profile is not None:
            val_y = minmax_apply(val.mol_profile, y_min, scale)
            val_dset = self._dataset(val, val_y)
            val_loader = build_dataloader(
                val_dset, batch_size=self.batch_size, shuffle=False
            )

        output_transform = UnscaleTransform(mean=y_min, scale=scale)
        self._model = _build_model(
            hidden_size=self.hidden_size,
            depth=self.depth,
            dropout=self.dropout,
            n_tasks=self.grid.num_points,
            output_transform=output_transform,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            # devices=1: this machine has 2 GPUs, and "auto" devices would
            # silently launch multi-GPU DDP, splitting the val/test batch
            # across ranks -- caught in testing as a metrics shape mismatch
            # (profile_true 10 rows vs profile_pred 5, half lost to the
            # other rank). A single-process run is what the harness expects.
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(self._model, train_loader, val_loader)

    def predict_molecules(self, test: MoleculeSet) -> Prediction:
        import lightning.pytorch as pl
        from chemprop.data import build_dataloader

        if self._model is None:
            raise RuntimeError("fit_molecules must be called before predict_molecules")

        test_dset = self._dataset(test, y=None)
        test_loader = build_dataloader(
            test_dset, batch_size=self.batch_size, shuffle=False
        )
        trainer = pl.Trainer(
            accelerator="auto", devices=1, enable_progress_bar=False, logger=False
        )
        batches = trainer.predict(self._model, test_loader)
        if not batches:
            raise RuntimeError("chemprop prediction produced no batches")
        profile = np.concatenate([np.asarray(b) for b in batches], axis=0)
        return prediction_from_profile(profile, self.grid.values)


def _build(params: Mapping[str, Any]) -> ChempropDMPNNPredictor:
    return ChempropDMPNNPredictor(**params)


register("chemprop_dmpnn", _build)
