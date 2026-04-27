import math

import mlflow
import pytest

from utils.logging import MlflowTrackingError, end_experiment, log_metrics_batch, log_param, start_experiment_run


@pytest.fixture(autouse=True)
def isolated_tracking_uri(tmp_path, monkeypatch):
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    while mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_log_param_allows_same_value_when_resuming_run():
    run_id = start_experiment_run("exp_test_logging", run_name="run_a")
    log_param("players", 2)
    end_experiment()

    start_experiment_run("exp_test_logging", run_id=run_id)
    log_param("players", 2)

    with pytest.raises(MlflowTrackingError):
        log_param("players", 3)

    end_experiment()


def test_log_metrics_batch_rejects_non_finite_values():
    start_experiment_run("exp_test_logging", run_name="run_b")

    with pytest.raises(MlflowTrackingError):
        log_metrics_batch({"good_metric": 1.0, "bad_metric": math.nan}, step=1)

    end_experiment(status="FAILED")
