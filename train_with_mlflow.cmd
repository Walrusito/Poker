@echo off
setlocal

REM Starts MLflow UI in background, then runs training.
REM Usage:
REM   train_with_mlflow.cmd [any docker compose run args...]

docker compose up -d --build mlflow-ui
if errorlevel 1 (
  echo Failed to start mlflow-ui
  exit /b 1
)

REM Pass-through args to training container
docker compose run --rm --build poker-ai %*

endlocal
