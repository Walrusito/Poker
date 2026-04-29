import mlflow
import numpy as np
import torch


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
        pass

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
        pass

    print(f"[MLFLOW PARAM] {name}: {value}")


# -----------------------------
# ARTIFACTS
# -----------------------------
def log_artifact(path):

    try:
        if mlflow.active_run() is not None:
            mlflow.log_artifact(path)
    except Exception:
        pass

    print(f"[MLFLOW ARTIFACT] {path}")


# -----------------------------
# EXPERIMENT CONTEXT (NEW)
# -----------------------------
def start_experiment(name="poker_cfr_ai"):

    try:
        mlflow.set_experiment(name)
        mlflow.start_run()
    except Exception:
        pass

    print(f"[MLFLOW] experiment started: {name}")


def end_experiment():

    try:
        mlflow.end_run()
    except Exception:
        pass

    print("[MLFLOW] experiment ended")