import json
import math
import os

import mlflow
import numpy as np
import torch

_MLFLOW_TRACKING_URI_DEFAULT = "file:mlruns"
_MLFLOW_STRICT_ENV = "POKER_MLFLOW_STRICT"


class MlflowTrackingError(RuntimeError):
    pass


def _strict_mlflow_enabled():
    raw = os.environ.get(_MLFLOW_STRICT_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _ensure_tracking_uri():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", _MLFLOW_TRACKING_URI_DEFAULT)
    mlflow.set_tracking_uri(tracking_uri)


def _raise_mlflow_error(message, exc=None):
    if _strict_mlflow_enabled():
        if exc is not None:
            raise MlflowTrackingError(message) from exc
        raise MlflowTrackingError(message)
    print(f"[MLFLOW WARNING] {message}")


def _safe_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("expected scalar tensor value")
        return value.detach().cpu().item()

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError("expected scalar ndarray value")
        return float(value.reshape(-1)[0])

    if isinstance(value, np.generic):
        return value.item()

    return value


def _stringify_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _require_active_run(operation: str):
    active_run = mlflow.active_run()
    if active_run is None:
        _raise_mlflow_error(f"cannot {operation}: no active MLflow run")
    return active_run


def _normalize_metric_value(name, value):
    safe = _safe_value(value)
    if safe is None:
        return None
    try:
        numeric = float(safe)
    except (TypeError, ValueError) as exc:
        _raise_mlflow_error(f"metric '{name}' is not numeric: {safe!r}", exc)
    if not math.isfinite(numeric):
        _raise_mlflow_error(f"metric '{name}' is not finite: {numeric!r}")
    return numeric


def _current_param_value(name):
    active_run = _require_active_run(f"read parameter '{name}'")
    run = mlflow.get_run(active_run.info.run_id)
    return run.data.params.get(name)


def log_metric(name, value, step=None):
    numeric = _normalize_metric_value(name, value)
    if numeric is None:
        return

    try:
        _require_active_run(f"log metric '{name}'")
        mlflow.log_metric(name, numeric, step=step)
    except Exception as exc:
        _raise_mlflow_error(f"failed to log metric '{name}'", exc)

    print(f"[MLFLOW] {name}: {numeric}")


def log_metrics_batch(metrics, step=None):
    safe_metrics = {}
    for name, value in metrics.items():
        numeric = _normalize_metric_value(name, value)
        if numeric is None:
            continue
        safe_metrics[name] = numeric

    if not safe_metrics:
        print("[MLFLOW] metrics batch skipped: no finite metrics")
        return

    try:
        _require_active_run("log metrics batch")
        mlflow.log_metrics(safe_metrics, step=step)
    except Exception as exc:
        _raise_mlflow_error("failed to log metrics batch", exc)

    for name, value in safe_metrics.items():
        print(f"[MLFLOW] {name}: {value}")


def log_param(name, value):
    serialized = _stringify_value(_safe_value(value))
    if serialized is None:
        print(f"[MLFLOW PARAM] {name}: <skipped None>")
        return

    try:
        existing = _current_param_value(name)
        if existing is not None:
            if existing != serialized:
                _raise_mlflow_error(
                    f"parameter '{name}' already exists with value {existing!r}; "
                    f"refusing to overwrite with {serialized!r}"
                )
            print(f"[MLFLOW PARAM] {name}: {serialized} (unchanged)")
            return
        mlflow.log_param(name, serialized)
    except MlflowTrackingError:
        raise
    except Exception as exc:
        _raise_mlflow_error(f"failed to log param '{name}'", exc)

    print(f"[MLFLOW PARAM] {name}: {serialized}")


def set_run_tag(name, value):
    serialized = _stringify_value(_safe_value(value))
    if serialized is None:
        return

    try:
        _require_active_run(f"set tag '{name}'")
        mlflow.set_tag(name, serialized)
    except Exception as exc:
        _raise_mlflow_error(f"failed to set tag '{name}'", exc)

    print(f"[MLFLOW TAG] {name}: {serialized}")


def set_run_tags(tags):
    for name, value in tags.items():
        set_run_tag(name, value)


def log_artifact(path, artifact_path=None):
    try:
        _require_active_run(f"log artifact '{path}'")
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:
        _raise_mlflow_error(f"failed to log artifact '{path}'", exc)

    print(f"[MLFLOW ARTIFACT] {path}")


def start_experiment(name="poker_cfr_ai"):
    _ensure_tracking_uri()

    try:
        mlflow.set_experiment(name)
        mlflow.start_run()
    except Exception as exc:
        _raise_mlflow_error(f"failed to start experiment '{name}'", exc)

    print(f"[MLFLOW] experiment started: {name}")


def start_experiment_run(name="poker_cfr_ai", run_name=None, run_id=None):
    _ensure_tracking_uri()

    try:
        mlflow.set_experiment(name)
        if run_id:
            run = mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(run_name=run_name)
    except Exception as exc:
        _raise_mlflow_error(
            f"failed to start run (experiment='{name}', run_name='{run_name}', run_id='{run_id}')",
            exc,
        )

    started_run_id = run.info.run_id
    print(f"[MLFLOW] tracking_uri: {mlflow.get_tracking_uri()}")
    print(f"[MLFLOW] experiment started: {name}")
    if run_id:
        print(f"[MLFLOW] resumed_run_id: {started_run_id}")
    else:
        print(f"[MLFLOW] run_id: {started_run_id}")
    return started_run_id


def end_experiment(status="FINISHED"):
    active_run = mlflow.active_run()
    if active_run is None:
        print("[MLFLOW] no active run to end")
        return

    try:
        mlflow.end_run(status=status)
    except Exception as exc:
        _raise_mlflow_error(f"failed to end run with status '{status}'", exc)

    print(f"[MLFLOW] experiment ended with status={status}")