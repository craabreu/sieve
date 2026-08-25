"""Tests for experiments/sieve_experiments/plots.py. Optional-data tests:
matplotlib (and rdkit, for _display_smiles) aren't part of the fast suite's
dependencies (see plots.py's module docstring) -- skipped gracefully when
either is absent, same pattern as tests/test_benchmark.py.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless: no display needed to render to a file
pytest.importorskip("rdkit")

from sieve_experiments import plots

# --- _display_smiles ---------------------------------------------------


def test_display_smiles_strips_atom_map_numbers():
    result = plots._display_smiles("[C:1]([H:2])([H:3])([H:4])[H:5]")
    assert ":1]" not in result
    assert ":2]" not in result


def test_display_smiles_falls_back_to_input_when_unparseable():
    """synthetic_molecule_set's placeholder SMILES ("C0", "C1", ...) are
    deliberately never valid SMILES -- must not raise."""
    assert plots._display_smiles("C0") == "C0"


# --- parity_hexbin / profile_panel (smoke: file gets written, no crash) --


def test_parity_panel_writes_a_file_with_multiple_panels(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.random((5, 51))
    y_pred = y_true + rng.normal(scale=0.01, size=y_true.shape)
    out_path = tmp_path / "parity_panel.png"

    plots.parity_panel(
        [
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "quantity": "normalized sigma-profile (per bin)",
                "title": "molecule profile",
                "metrics": {"w1_norm_mean": 0.01},
            },
            {
                "y_true": rng.random(5),
                "y_pred": rng.random(5),
                "quantity": "area (A²)",
                "title": "molecule area",
                "metrics": {"mae": 0.2},
            },
        ],
        out_path,
        suptitle="test",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_parity_panel_writes_nothing_for_an_empty_panel_list(tmp_path):
    out_path = tmp_path / "parity_panel.png"
    plots.parity_panel([], out_path, suptitle="test")
    assert not out_path.exists()


def test_profile_panel_writes_a_file_and_survives_unparseable_smiles(tmp_path):
    rng = np.random.default_rng(0)
    n_mol = 6
    sigma_values = np.linspace(-0.025, 0.025, 51)
    mol_true = rng.random((n_mol, 51))
    mol_pred = mol_true + rng.normal(scale=0.01, size=mol_true.shape)
    labels = [f"C{i}" for i in range(n_mol)]  # placeholders, never parsed
    out_path = tmp_path / "panel.png"

    plots.profile_panel(sigma_values, mol_true, mol_pred, labels, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_profile_panel_only_parses_smiles_it_actually_plots(tmp_path, monkeypatch):
    """A large test split (thousands of molecules) shouldn't pay an
    RDKit parse per molecule just to render n_rows*n_cols (default 16)
    plot titles."""
    calls = []

    def spy(smiles):
        calls.append(smiles)
        return smiles

    monkeypatch.setattr(plots, "_display_smiles", spy)

    rng = np.random.default_rng(0)
    n_mol = 500
    sigma_values = np.linspace(-0.025, 0.025, 51)
    mol_true = rng.random((n_mol, 51))
    mol_pred = mol_true + rng.normal(scale=0.01, size=mol_true.shape)
    labels = [f"C{i}" for i in range(n_mol)]
    out_path = tmp_path / "panel.png"

    plots.profile_panel(sigma_values, mol_true, mol_pred, labels, out_path, seed=1)

    assert len(calls) == 16  # n_rows*n_cols default, not n_mol
