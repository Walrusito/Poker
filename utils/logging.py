import os
import traceback

import mlflow
import numpy as np
import torch

_MLFLOW_TRACKING_URI_DEFAULT = "file:mlruns"


def _ensure_tracking_uri():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI_DEFAULT)


# -----------------------------
# SAFE VALUE CONVERSION
# -----------------------------
def _safe_value(value):

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()

    if isinstance(value, np.ndarray):
        return float(value)

    return value


# -----------------------------
# METRICS
# -----------------------------
def log_metric(name, value, step=None):

    value = _safe_value(value)

    try:
        if mlflow.active_run() is not None:
            mlflow.log_metric(name, value, step=step)
    except Exception:
        print(f"[MLFLOW WARNING] failed to log metric '{name}': {traceback.format_exc()}")

    print(f"[MLFLOW] {name}: {value}")


def log_metrics_batch(metrics, step=None):
    safe_metrics = {}
    for name, value in metrics.items():
        safe = _safe_value(value)
        if safe is None:
            continue
        try:
            safe_metrics[name] = float(safe)
        except (TypeError, ValueError):
            continue

    try:
        if mlflow.active_run() is not None and safe_metrics:
            mlflow.log_metrics(safe_metrics, step=step)
    except Exception:
        print(f"[MLFLOW WARNING] failed to log metrics batch: {traceback.format_exc()}")

    for name, value in safe_metrics.items():
        print(f"[MLFLOW] {name}: {value}")


# -----------------------------
# PARAMETERS
# -----------------------------
def log_param(name, value):

    value = _safe_value(value)

    try:
        if mlflow.active_run() is not None:
            mlflow.log_param(name, value)
    except Exception:
        print(f"[MLFLOW WARNING] failed to log param '{name}': {traceback.format_exc()}")

    print(f"[MLFLOW PARAM] {name}: {value}")


# -----------------------------
# ARTIFACTS
# -----------------------------
def log_artifact(path):

    try:
        if mlflow.active_run() is not None:
            mlflow.log_artifact(path)
    except Exception:
        print(f"[MLFLOW WARNING] failed to log artifact '{path}': {traceback.format_exc()}")

    print(f"[MLFLOW ARTIFACT] {path}")


# -----------------------------
# EXPERIMENT CONTEXT
# -----------------------------
def start_experiment(name="poker_cfr_ai"):
    _ensure_tracking_uri()

    try:
        mlflow.set_experiment(name)
        mlflow.start_run()
    except Exception:
        print(f"[MLFLOW ERROR] failed to start experiment '{name}': {traceback.format_exc()}")

    print(f"[MLFLOW] experiment started: {name}")


def start_experiment_run(name="poker_cfr_ai", run_name=None, run_id=None):
    _ensure_tracking_uri()

    started_run_id = None
    try:
        mlflow.set_experiment(name)
        if run_id:
            run = mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(run_name=run_name)
        started_run_id = run.info.run_id
    except Exception:
        print(f"[MLFLOW ERROR] failed to start run (experiment='{name}', "
              f"run_name='{run_name}', run_id='{run_id}'): {traceback.format_exc()}")
        started_run_id = None

    tracking_uri = mlflow.get_tracking_uri()
    print(f"[MLFLOW] tracking_uri: {tracking_uri}")
    print(f"[MLFLOW] experiment started: {name}")
    if started_run_id is not None:
        print(f"[MLFLOW] run_id: {started_run_id}")
    else:
        print("[MLFLOW WARNING] no active run — metrics will NOT be recorded")
    return started_run_id


def end_experiment():

    try:
        mlflow.end_run()
    except Exception:
        print(f"[MLFLOW WARNING] failed to end run: {traceback.format_exc()}")

    print("[MLFLOW] experiment ended")
