"""End-to-end HoseLookupPredictor test through the real run() pipeline, gated
on the real, already-split dash-molecules store and on the optional hosegen
dependency. Mirrors test_predictor_sieve_optional.py."""

from __future__ import annotations

import pytest

from experiments.tests.helpers import real_store_has_columns

pytestmark = pytest.mark.skipif(
    not real_store_has_columns("dash-molecules", "split"),
    reason="real dash-molecules store not prepared and split locally",
)


def test_hose_charge_predictor_runs_end_to_end_via_run(tmp_path):
    pytest.importorskip("hosegen")
    from experiments.config import (
        DataCfg,
        ExperimentCfg,
        PredictorCfg,
        RunCfg,
        TargetCfg,
    )
    from experiments.runner import run

    cfg = ExperimentCfg(
        run=RunCfg(experiment="hose-charge-optional", seed=0),
        data=DataCfg(store="dash-molecules", split_column="split"),
        target=TargetCfg(atom_property="MBIScharge", molecule_property="net_charge"),
        predictor=PredictorCfg(name="hose", params={"max_radius": 5}),
    )
    result = run(cfg, runs_root=tmp_path, allow_dirty=True, tracking=None, limit=200)
    assert result.metrics["n_test_conformers"] >= 0
