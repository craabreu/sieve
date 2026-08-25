"""T10: a from-scratch D-MPNN sigma-profile predictor on Chemprop, reproducing
the architecture COSMO-NET-Paper's own script *actually trains* -- not the
architecture the peer-reviewed paper's text claims (Naseri Boroujeni et al.,
*Mol. Syst. Des. Eng.*, 2026, DOI 10.1039/d6me00088f, Tables 1-2 and Section
2.2.2). Those two disagree, and not by a little: see pins.toml's
``[cosmonet]`` CORRECTION note for the full derivation. In short,
``DMPNN-Train-pSigma.py``'s ``MODEL_HPARAMS`` dict passes several keys
(``dropout``, ``message_passing_steps``, ``hidden_size``, ``ffn_num_layers``)
that are not real ``DMPNNModel.__init__`` parameter names on the installed
deepchem version -- they get silently absorbed into ``**kwargs`` and never
reach the encoder, which is built from deepchem's own defaults instead.
``message_passing_steps=3`` is the one of the four whose failure has no
visible symptom: deepchem's real parameter (``depth``) also defaults to 3,
so the "right" value comes out by coincidence, not because the script
successfully passed it through -- it did not, and a different intended
value would have silently trained with ``depth=3`` regardless. Confirmed
directly against real trained checkpoint weights (both the paper
repo's own shipped one and ours, independently): the model that was *really*
trained -- and that generated T9's numbers, and plausibly the paper's own
reported results too -- has `hidden_size=300` (not 51), a 3-layer FFN (not
1), no dropout (not 0.1), and mean-pooling readout (not the sum pooling the
paper's own eqn 11 specifies). This module now reproduces *that* real
architecture, since it is the one thing in this whole picture with direct
empirical evidence (checkpoint weights) behind it -- the paper's own prose
does not.

Two layers, deliberately split so the featurization is testable without the
optional dependency (mirrors predictors/dash.py's ``fit_backoff``/
``DASHBackoffPredictor`` split):

- ``PaperAtomFeaturizer``, ``minmax_*``, ``prediction_from_profile`` -- pure
  numpy + rdkit, no chemprop import at module scope. Fast-suite tested
  (experiments/tests/test_experiment_predictor_chemprop.py).
- ``ChempropDMPNNPredictor`` -- wires those onto a real chemprop ``MPNN`` and
  a lightning ``Trainer``. Needs the ``chemprop`` extra
  (``uv sync --extra chemprop``). Optional-data tested only.

Architecture:

- Atom features (35-dim, the paper's own Table 1): element one-hot
  {C,N,O,S,F,P,Cl,Br} (8), degree one-hot 0-5 (6), formal charge one-hot
  {-2..2} (5), hybridization one-hot {sp,sp2,sp3,sp3d,sp3d2} (5), aromaticity
  (1), total-H one-hot 0-4 (5), chirality one-hot 0-3 (4), atomic mass x 0.01
  (1). This is the one part of the paper's own claimed architecture that
  *was* genuinely realized: the paper repo's own shipped checkpoint has
  `encoder.W_i.weight` of `(300, 49)` = 35 (patched atom_fdim) + 14 (stock
  bond_fdim) -- see pins.toml's "WHY THE AUTHORS' OWN CHECKPOINT IS (300,
  49)" note. chemprop's own built-in ``MultiHotAtomFeaturizer`` cannot be
  parameterized down to exactly 35 (it always reserves one pad slot per
  one-hot block, landing at 41), so this stays a custom, duck-typed class.
  An element outside this 8-element vocabulary (chaos-store has B, Si, Ge,
  Sb, Te too -- the same gap DASH's own vocabulary hits) one-hots to
  all-zero: a documented deviation, not a crash.
- Bond features (14-dim): deepchem's *actual* stock featurization (never
  patched -- confirmed both checkpoints share the same 14-dim bond side),
  which is structurally identical to chemprop's own built-in
  ``MultiHotBondFeaturizer()`` default (null-bit + 4-way bond-type one-hot +
  conjugated + in-ring + 6-category stereo one-hot with an unknown pad).
  Used directly -- no custom bond featurizer needed here, unlike the atom
  side.
- ``hidden_size=300`` (deepchem's `enc_hidden` default, not the paper's
  claimed 51).
- FFN: deepchem's real `ffn_layers=3` means **3 total** Linear layers
  (confirmed by reading `PositionwiseFeedForward.__init__` directly:
  `n_layers` there *is* the total layer count). Chemprop's own
  `MLP.build(n_layers=N)` instead yields **N+1** total layers (an
  *additional*-hidden-layers count) -- confirmed by inspecting `MLP.build`.
  So matching deepchem's real 3-layer FFN needs chemprop's `n_layers=2`, not
  3 and not 1. Getting this wrong in either direction silently builds a
  different-depth FFN than what was actually trained.
- `dropout=0.0` (both `enc_dropout_p` and `ffn_dropout_p` default to 0.0 in
  deepchem; the paper's claimed 0.1 was another silently-ignored
  `MODEL_HPARAMS` key).
- Readout: **mean** pooling, not the sum pooling the paper's own eqn 11
  specifies for the DMPNN -- `aggregation` isn't in `MODEL_HPARAMS` at all
  (grepped, absent), so deepchem's default `aggregation='mean'` was used.
  This is the one deviation that isn't just an ignored kwarg landing back on
  a default that happens to not matter (like `depth=3`, which coincidentally
  matches the paper's own intent) -- it's the real model doing something the
  paper's own text says it doesn't do.
- Softplus sits on the FFN's raw (scaled-space) output, *before*
  ``output_transform`` (min-max unscaling) -- see ``SoftplusRegressionFFN``.
  Since min-max scaling's ``y_min >= 0`` for every profile bin (a real
  physical density), ``softplus(x) > 0`` composed with unscaling guarantees
  ``p(sigma) > 0`` for every prediction, structurally -- not a post-hoc
  clip, which is exactly the demonstration COSMO-NET-Paper's own repo could
  not produce (its published results have 0% negative bins; its published
  script has no non-negativity enforcement anywhere in its prediction
  path). This part of T10's design is unaffected by the architecture
  correction above -- softplus is our own addition either way, not
  something deepchem's real run had.
- chemprop's ``normalize_targets()`` is z-score only; per-task min-max
  scaling (the paper's eqn 17, and something the real deepchem run's own
  ``MinMaxTransformer`` did apply, unaffected by the kwarg-passing bug) is
  built by hand via ``UnscaleTransform`` from training-set statistics only
  (a documented no-op while ``self.training``, so MSE loss is computed in
  scaled space during training).
- Optimizer/LR schedule: deepchem's real run used an explicit
  `ExponentialDecay(initial_rate=0.001, decay_rate=0.95, decay_steps=1000)`
  (set directly in the script, not via the broken `MODEL_HPARAMS` dict, so
  genuinely applied). chemprop's `MPNN` instead defaults to Adam with its
  own Noam-like warmup schedule. Not replicated here -- would need
  subclassing `MPNN.configure_optimizers` -- and left as a known,
  documented remaining deviation.

``loss_mode`` (predictor param, default ``"mse"``) picks the training
objective, independent of the architecture above:

- ``"mse"`` -- chemprop's own default: plain per-bin MSE, each of the 51
  sigma bins as an independent, equally-weighted regression target. What
  the real deepchem run (and this module, before this option existed) uses.
- ``"w1_normalized"`` -- Wasserstein-1 distance between the predicted and
  target profiles, each normalized by its OWN row sum first (a pure shape
  loss, decoupled from magnitude) -- matches `metrics.molecule_metrics`'s
  primary eval metric (`profile/w1_norm_mean`) exactly, used as the
  training objective instead of just an eval metric.
- ``"mse_cumsum"`` -- MSE between the predicted and target profiles'
  cumulative sums ("accumulated area sums"), unnormalized -- a joint
  shape+magnitude loss: matching the true cumsum vector uniquely
  determines the true profile (`profile[i] = cumsum[i] - cumsum[i-1]`), so
  every output bin gets real gradient, unlike an MSE on total area alone
  (which would leave the 50 shape degrees of freedom undetermined if used
  as the sole loss).

Both custom losses need the whole row (all 51 bins together, not
per-bin independently) in REAL, unscaled units to mean anything physically
-- see ``_UnscalingRowLoss``'s docstring for why per-bin min-max scaling
makes a row-sum or cumsum computed directly on the *scaled* tensor
meaningless, and why `UnscaleTransform`'s own train-mode no-op can't be
relied on here (both classes unscale explicitly, with their own copy of
`y_min`/`scale`, regardless of train/eval mode).
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
# chemprop's own built-in MultiHotBondFeaturizer() default length -- see the
# module docstring for why this matches deepchem's real (never-patched)
# bond featurization exactly. Asserted against the real class in
# test_experiment_predictor_chemprop_optional.py, since chemprop isn't
# available to the fast suite.
BOND_FDIM = 14

_ELEMENTS = ("C", "N", "O", "S", "F", "P", "Cl", "Br")
_DEGREES = tuple(range(6))
_FORMAL_CHARGES = (-2, -1, 0, 1, 2)
_HYBRIDIZATIONS = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
_TOTAL_HS = tuple(range(5))
_CHIRAL_TAGS = (0, 1, 2, 3)


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
    """The 35-dim atom feature vector from COSMO-NET's Table 1, exactly --
    the one part of the paper's claimed architecture that was genuinely
    realized (see the module docstring).

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


VALID_OUTPUT_ACTIVATIONS = ("softplus", "squared", "abs")

_NONNEG_FFN_CLASSES: dict[str, Any] = {}


def nonneg_regression_ffn_class(kind: str = "softplus") -> Any:
    """A ``RegressionFFN`` subclass that forces a non-negative activation onto
    the FFN's raw output, *before* unscaling -- see the module docstring for
    why applying it there makes ``p(sigma) > 0`` structural rather than a
    post-hoc clip.

    Built lazily and memoized per kind rather than defined at module scope,
    because it subclasses chemprop and this module must stay importable
    without it (the fast-suite contract). Shared with
    ``predictors/chemprop_atom.py`` (T11), so the invariant lives in one place.

    ``kind`` picks the activation. All three keep non-negativity equally, but
    they differ in whether an output of *exactly* zero is reachable, which
    turns out to decide whether the model trains at all:

    - ``"softplus"`` -- T10's default and what the COSMO-NET paper describes.
      ``softplus(x) = 0`` only as ``x -> -inf``. Fine for molecule-level
      profiles, which are dense (50.2% exact-zero bins, 25.4 of 51 bins
      populated) and never need the output driven to exactly zero.
    - ``"squared"`` -- T11's default. ``x**2`` hits exactly zero at ``x = 0``,
      a finite point, so a target of exactly zero is reachable and the
      gradient does not vanish on the way there.
    - ``"abs"`` -- same reachability property, non-smooth at the origin.
      Measured slightly worse than ``"squared"``; kept for comparison.

    WHY THIS IS PARAMETERIZED (2026-08-25): softplus **collapses entirely** at
    the atom level. Atom profiles are 81.4% exact zeros (each atom occupies
    only ~9.5 of the 51 bins), so MSE relentlessly drives those outputs toward
    exactly zero; softplus can only approach that as its pre-activation goes
    to -inf, where its own derivative ``sigmoid(x)`` vanishes, and the head
    dies. Measured: a softplus atom model predicts identically zero and cannot
    overfit even 20 molecules (w1 6.72, predicted area 0.000 against a true
    7.676), while the same model with ``"squared"`` reaches w1 0.878 and area
    7.859 -- statistically indistinguishable from an unconstrained linear head
    (w1 0.825), which however emits 13,110 negative bins. See README's T11
    section.
    """
    if kind not in VALID_OUTPUT_ACTIVATIONS:
        raise ValueError(
            f"output_activation must be one of {VALID_OUTPUT_ACTIVATIONS}, got {kind!r}"
        )
    if kind in _NONNEG_FFN_CLASSES:
        return _NONNEG_FFN_CLASSES[kind]

    import torch.nn.functional as F
    from chemprop.nn.predictors import RegressionFFN

    activations = {
        "softplus": F.softplus,
        "squared": lambda x: x**2,
        "abs": lambda x: x.abs(),
    }
    activation = activations[kind]

    class NonNegativeRegressionFFN(RegressionFFN):
        """``RegressionFFN`` with a non-negative activation forced onto the
        FFN's raw output, before unscaling -- see
        ``nonneg_regression_ffn_class``.
        """

        def forward(self, Z):
            return self.output_transform(activation(self.ffn(Z)))

        train_step = forward

    NonNegativeRegressionFFN.__name__ = f"{kind.capitalize()}RegressionFFN"
    NonNegativeRegressionFFN.__qualname__ = NonNegativeRegressionFFN.__name__
    _NONNEG_FFN_CLASSES[kind] = NonNegativeRegressionFFN
    return NonNegativeRegressionFFN


def _build_model(
    *,
    hidden_size: int,
    depth: int,
    dropout: float,
    ffn_n_layers: int,
    n_tasks: int,
    output_transform,
    y_min: NDArray[np.float64],
    scale: NDArray[np.float64],
    loss_mode: str,
):
    """Lazy: only imports chemprop when actually called."""
    import torch
    from chemprop.models import MPNN
    from chemprop.nn.agg import MeanAggregation
    from chemprop.nn.message_passing import BondMessagePassing
    from chemprop.nn.metrics import ChempropMetric

    SoftplusRegressionFFN = nonneg_regression_ffn_class("softplus")

    class _UnscalingRowLoss(ChempropMetric):
        """Base for a loss that needs the whole row (all ``n_tasks`` profile
        bins together, not per-bin independently) in REAL, unscaled units.

        ``preds``/``targets`` arrive here in per-bin-independently-scaled
        space (each of the 51 bins has its own min-max transform) -- a
        row-sum or cumsum computed directly on that scaled tensor would not
        correspond to the true physical row-sum/cumsum at all, since scaling
        differs bin to bin. ``UnscaleTransform`` itself is a no-op during
        training (by design, so plain per-element losses stay in scaled
        space) so it can't be relied on here; this class unscales explicitly
        instead, with its own copy of ``y_min``/``scale``, regardless of
        train/eval mode. Subclasses override ``_row_loss`` only.
        """

        def __init__(self, y_min, scale, task_weights=1.0, **kwargs):
            super().__init__(task_weights)
            self.register_buffer("y_min", torch.as_tensor(y_min, dtype=torch.float))
            self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

        def _calc_unreduced_loss(self, *args, **kwargs):
            raise NotImplementedError(
                f"{type(self).__name__} overrides update() directly."
            )

        def _row_loss(
            self, pred_real: torch.Tensor, target_real: torch.Tensor
        ) -> torch.Tensor:
            raise NotImplementedError

        def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor | None = None,
            weights: torch.Tensor | None = None,
            lt_mask: torch.Tensor | None = None,
            gt_mask: torch.Tensor | None = None,
        ) -> None:
            if mask is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            if weights is None:
                weights = torch.ones(targets.shape[0], device=targets.device)
            valid = mask[:, 0]

            pred_real = preds * self.scale + self.y_min
            target_real = targets * self.scale + self.y_min
            loss = self._row_loss(pred_real, target_real) * valid * weights.view(-1)

            self.total_loss += loss.sum()
            self.num_samples += valid.sum()

    class NormalizedWasserstein1Loss(_UnscalingRowLoss):
        """Wasserstein-1 distance between the predicted and target profiles,
        each normalized by its OWN row sum first -- a pure shape loss,
        decoupled from magnitude, matching ``metrics.molecule_metrics``'s
        primary eval metric (``profile/w1_norm_mean``) exactly, but used as
        the training objective instead of just an eval metric.
        """

        def __init__(self, y_min, scale, task_weights=1.0, eps: float = 1e-8, **kwargs):
            super().__init__(y_min, scale, task_weights, **kwargs)
            self.eps = eps

        def _row_loss(
            self, pred_real: torch.Tensor, target_real: torch.Tensor
        ) -> torch.Tensor:
            pred_norm = pred_real / (pred_real.sum(1, keepdim=True) + self.eps)
            target_norm = target_real / (target_real.sum(1, keepdim=True) + self.eps)
            return (target_norm.cumsum(1) - pred_norm.cumsum(1)).abs().sum(1)

        def extra_repr(self) -> str:
            return f"eps={self.eps}"

    class CumulativeSumMSELoss(_UnscalingRowLoss):
        """MSE between the predicted and target profiles' cumulative sums
        ("accumulated area sums"), unnormalized -- a joint shape+magnitude
        loss: matching the true cumsum vector uniquely determines the true
        profile (profile[i] = cumsum[i] - cumsum[i-1]), so every output bin
        gets real gradient, unlike an MSE on total area alone.
        """

        def _row_loss(
            self, pred_real: torch.Tensor, target_real: torch.Tensor
        ) -> torch.Tensor:
            return ((pred_real.cumsum(1) - target_real.cumsum(1)) ** 2).mean(1)

    if loss_mode == "mse":
        criterion = None  # RegressionFFN's own default (plain per-bin MSE)
    elif loss_mode == "w1_normalized":
        criterion = NormalizedWasserstein1Loss(y_min=y_min, scale=scale)
    elif loss_mode == "mse_cumsum":
        criterion = CumulativeSumMSELoss(y_min=y_min, scale=scale)
    else:
        raise ValueError(
            "loss_mode must be one of 'mse', 'w1_normalized', 'mse_cumsum', "
            f"got {loss_mode!r}"
        )

    message_passing = BondMessagePassing(
        d_v=ATOM_FDIM, d_e=BOND_FDIM, d_h=hidden_size, depth=depth, dropout=dropout
    )
    # Mean pooling, not sum: deepchem's real (never-overridden) default --
    # see the module docstring for why this contradicts the paper's own
    # eqn 11.
    agg = MeanAggregation()
    predictor = SoftplusRegressionFFN(
        n_tasks=n_tasks,
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=criterion,
        output_transform=output_transform,
    )
    return MPNN(message_passing, agg, predictor)


class ChempropDMPNNPredictor(MoleculePredictor):
    """Trains and predicts the architecture COSMO-NET-Paper's script
    actually trains (see the module docstring for why that's the target,
    not the paper's own claimed hyperparameters).
    """

    name = "chemprop_dmpnn"

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
        loss_mode: str = "mse",
        grid: SigmaGridSpec = DEFAULT_GRID,
    ) -> None:
        if loss_mode not in ("mse", "w1_normalized", "mse_cumsum"):
            raise ValueError(
                "loss_mode must be one of 'mse', 'w1_normalized', 'mse_cumsum', "
                f"got {loss_mode!r}"
            )
        self.store = store
        self.scheme = scheme
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.ffn_n_layers = ffn_n_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss_mode = loss_mode
        self.grid = grid
        self._model: Any = None

    def _featurizer(self):
        from chemprop.featurizers import MultiHotBondFeaturizer
        from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer

        return SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=PaperAtomFeaturizer(),
            bond_featurizer=MultiHotBondFeaturizer(),
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
            ffn_n_layers=self.ffn_n_layers,
            n_tasks=self.grid.num_points,
            output_transform=output_transform,
            y_min=y_min,
            scale=scale,
            loss_mode=self.loss_mode,
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
