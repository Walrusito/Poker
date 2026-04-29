# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile optimizado para RTX 3060 (CUDA 12.1) + Ryzen 7 5800X
# Base: PyTorch oficial con CUDA — evita compilar desde cero
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (torch ya viene en la imagen base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente (después de deps para aprovechar cache de Docker)
COPY . .

# Directorio MLflow persistente
RUN mkdir -p /app/mlruns

# Variables de entorno para optimizar PyTorch en CUDA
ENV PYTHONUNBUFFERED=1 \
    TORCH_BACKENDS_CUDNN_BENCHMARK=1 \
    OMP_NUM_THREADS=8

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=2 \
  CMD python -c "from env.poker_env import PokerEnv; from train.train_deep_cfr import DeepCFRTrainer; e=PokerEnv(); t=DeepCFRTrainer(e); print('OK')" || exit 1

ENTRYPOINT ["python", "main.py"]
