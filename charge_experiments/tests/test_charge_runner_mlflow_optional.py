"""MLflow-dependent test for runner._ensure_experiment's retry-on-race
behavior. Skipped when mlflow is absent -- same convention as
test_charge_aggregate_optional.py."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mlflow")


def _operational_error() -> Exception:
    from sqlalchemy.exc import OperationalError

    return OperationalError("CREATE TABLE ...", {}, Exception("already exists"))


def test_ensure_experiment_retries_past_a_transient_operational_error(
    tmp_path, monkeypatch
):
    """The real failure mode: several run processes racing to run
    Alembic's one-time schema migration against a freshly created sqlite
    db all hit a genuine OperationalError except the winner. A transient
    one is retried, not fatal -- once it stops recurring, the call
    succeeds."""
    import mlflow

    from charge_experiments import runner

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(time, "sleep", lambda _: None)

    real_get = mlflow.get_experiment_by_name
    calls = {"n": 0}

    def _flaky_get(name):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _operational_error()
        return real_get(name)

    monkeypatch.setattr(mlflow, "get_experiment_by_name", _flaky_get)

    runner._ensure_experiment("exp", artifact_root=tmp_path / "art")

    assert calls["n"] == 3
    assert mlflow.get_experiment_by_name("exp") is not None


def test_ensure_experiment_raises_after_exhausting_retries(tmp_path, monkeypatch):
    """A persistent OperationalError (not just a one-time migration race)
    still surfaces, with the original error chained as its cause."""
    import mlflow

    from charge_experiments import runner

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        mlflow,
        "get_experiment_by_name",
        lambda name: (_ for _ in ()).throw(_operational_error()),
    )

    with pytest.raises(RuntimeError, match="could not initialize") as excinfo:
        runner._ensure_experiment("exp", artifact_root=tmp_path / "art")
    assert excinfo.value.__cause__ is not None
