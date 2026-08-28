"""Tests for plots.py and runner.py's own parity-panel wiring.

``_build_parity_panels`` is pure numpy (no matplotlib, no rdkit) so it's
tested independently of ``plots.parity_panel`` actually rendering anything
-- mirrors cosmo_experiments/tests/test_experiment_smoke.py's own split
between panel-selection logic and rendering.
"""

from __future__ import annotations

import numpy as np
import pytest


def _panel_titles(panels):
    return [p["title"] for p in panels]


def test_build_parity_panels_includes_atom_charge():
    from charge_experiments.predictors.base import Prediction
    from charge_experiments.runner import _build_parity_panels

    from charge_experiments.tests.helpers import synthetic_molecule_set

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(atom_charge=ms.atom_charge)  # perfect predictions
    run_metrics = {
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 1.0,
        "charge_conservation/mae": 0.0,
        "charge_conservation/rmse": 0.0,
        "charge_conservation/r2": 1.0,
    }

    panels = _build_parity_panels(ms, pred, run_metrics)

    np.testing.assert_array_equal(panels[0]["y_true"], ms.atom_charge)
    np.testing.assert_array_equal(panels[0]["y_pred"], pred.atom_charge)
    assert panels[0]["metrics"] == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_build_parity_panels_omits_conservation_panel_when_exactly_conserved():
    """A histogram of residuals that are all ~0 (a predictor whose own
    normalization already conserves charge exactly) carries no information
    -- it's a single spike, not a diagnostic -- so it's skipped entirely,
    not plotted degenerate."""
    from charge_experiments.predictors.base import Prediction
    from charge_experiments.runner import _build_parity_panels

    from charge_experiments.tests.helpers import synthetic_molecule_set

    ms = synthetic_molecule_set(n_mol=6, seed=0)
    pred = Prediction(atom_charge=ms.atom_charge)  # perfect predictions

    panels = _build_parity_panels(ms, pred, {})

    assert _panel_titles(panels) == ["atom charge"]


def test_build_parity_panels_conservation_panel_is_the_residual_histogram():
    from charge_experiments.data import molecule_sum
    from charge_experiments.predictors.base import Prediction
    from charge_experiments.runner import _build_parity_panels

    from charge_experiments.tests.helpers import synthetic_molecule_set

    ms = synthetic_molecule_set(n_mol=6, seed=1)
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.05, size=ms.atom_charge.shape)
    atom_charge_pred = ms.atom_charge + noise
    pred = Prediction(atom_charge=atom_charge_pred)

    panels = _build_parity_panels(ms, pred, {})

    pred_net_charge = molecule_sum(atom_charge_pred, ms.atom_mol_id, ms.n_conformers)
    expected_residual = pred_net_charge - ms.net_charge
    np.testing.assert_allclose(panels[1]["values"], expected_residual)


def test_build_parity_panels_conservation_panel_drops_nan_residuals():
    from charge_experiments.predictors.base import Prediction
    from charge_experiments.runner import _build_parity_panels

    from charge_experiments.tests.helpers import synthetic_molecule_set

    ms = synthetic_molecule_set(n_mol=3, seed=0)
    rng = np.random.default_rng(4)
    noise = rng.normal(scale=0.05, size=ms.atom_charge.shape)
    atom_charge_pred = ms.atom_charge + noise  # real residual spread, not exact
    atom_charge_pred[0] = np.nan  # poisons the whole first conformer's sum
    pred = Prediction(atom_charge=atom_charge_pred)

    panels = _build_parity_panels(ms, pred, {})

    assert panels[1]["values"].shape[0] == ms.n_conformers - 1
    assert not np.any(np.isnan(panels[1]["values"]))


def test_parity_panel_writes_a_png_file(tmp_path):
    pytest.importorskip("matplotlib")
    from charge_experiments import plots

    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.05, size=50)
    panels = [
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "quantity": "charge (e)",
            "title": "atom charge",
            "metrics": {"mae": 0.05, "rmse": 0.06, "r2": 0.99},
        }
    ]
    out_path = tmp_path / "parity_panel.png"

    plots.parity_panel(panels, out_path, suptitle="test")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_parity_panel_renders_a_histogram_kind_panel(tmp_path):
    pytest.importorskip("matplotlib")
    from charge_experiments import plots

    rng = np.random.default_rng(0)
    panels = [
        {
            "kind": "histogram",
            "values": rng.normal(scale=0.02, size=50),
            "xlabel": "molecule charge residual (e)",
            "title": "molecule charge conservation",
            "metrics": {"mae": 0.02, "rmse": 0.03, "r2": 0.5},
        }
    ]
    out_path = tmp_path / "parity_panel.png"

    plots.parity_panel(panels, out_path, suptitle="test")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_parity_panel_renders_mixed_hexbin_and_histogram_panels(tmp_path):
    pytest.importorskip("matplotlib")
    from charge_experiments import plots

    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.05, size=50)
    panels = [
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "quantity": "charge (e)",
            "title": "atom charge",
            "metrics": {"mae": 0.05},
        },
        {
            "kind": "histogram",
            "values": rng.normal(scale=0.02, size=50),
            "xlabel": "molecule charge residual (e)",
            "title": "molecule charge conservation",
            "metrics": {"mae": 0.02},
        },
    ]
    out_path = tmp_path / "parity_panel.png"

    plots.parity_panel(panels, out_path, suptitle="test")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_parity_panel_no_panels_writes_nothing(tmp_path):
    pytest.importorskip("matplotlib")
    from charge_experiments import plots

    out_path = tmp_path / "parity_panel.png"
    plots.parity_panel([], out_path, suptitle="test")
    assert not out_path.exists()
