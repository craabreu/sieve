"""Tests for base.py's RawPrediction/NormalizablePredictor -- the
predict_raw split both DASH predictors implement (Tasks 4/5)."""

from __future__ import annotations

import numpy as np


def test_raw_prediction_holds_atom_charge_and_atom_std():
    from experiments.predictors.base import RawPrediction

    raw = RawPrediction(
        atom_value=np.array([0.1, 0.2]), atom_std=np.array([0.05, 0.06])
    )
    np.testing.assert_array_equal(raw.atom_value, [0.1, 0.2])
    np.testing.assert_array_equal(raw.atom_std, [0.05, 0.06])


def test_normalizable_predictor_protocol_matches_a_conforming_class():
    from experiments.predictors.base import (
        NormalizablePredictor,
        Prediction,
        RawPrediction,
    )

    class _Conforming:
        name = "fake"

        def fit(self, train, val, *, rng):
            pass

        def predict(self, test):
            return Prediction(atom_value=np.zeros(0))

        def predict_raw(self, test):
            return RawPrediction(atom_value=np.zeros(0), atom_std=np.zeros(0))

    class _NonConforming:
        name = "fake2"

        def fit(self, train, val, *, rng):
            pass

        def predict(self, test):
            return Prediction(atom_value=np.zeros(0))

    assert isinstance(_Conforming(), NormalizablePredictor)
    assert not isinstance(_NonConforming(), NormalizablePredictor)


def test_prediction_field_is_atom_value():
    from experiments.predictors.base import Prediction

    pred = Prediction(atom_value=np.zeros(3))
    assert pred.atom_value.shape == (3,)
