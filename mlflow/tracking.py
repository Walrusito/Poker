import mlflow


def log_metric(name, value, step=None):
    """
    MLflow metric logger con fallback seguro.
    """

    # MLflow tracking (si está activo)
    try:
        if mlflow.active_run() is not None:
            mlflow.log_metric(name, value, step=step)
    except Exception:
        pass  # evita crashes en Docker o modo sin tracking

    print(f"[MLFLOW] {name}: {value}")


def log_param(name, value):
    """
    Log de parámetros del experimento
    """

    try:
        if mlflow.active_run() is not None:
            mlflow.log_param(name, value)
    except Exception:
        pass

    print(f"[MLFLOW PARAM] {name}: {value}")


def log_artifact(path):
    """
    Guarda archivos (modelos, checkpoints, etc.)
    """

    try:
        if mlflow.active_run() is not None:
            mlflow.log_artifact(path)
    except Exception:
        pass

    print(f"[MLFLOW ARTIFACT] {path}")