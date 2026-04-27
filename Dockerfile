FROM python:3.11-slim

WORKDIR /app

ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (pinned in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url ${TORCH_CUDA_INDEX_URL} torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code (after deps so Docker cache is efficient)
COPY . .

# MLflow local tracking directory
RUN mkdir -p /app/mlruns

# Healthcheck: verify Python + imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=2 \
  CMD python -c "from env.poker_env import PokerEnv; PokerEnv()" || exit 1

ENTRYPOINT ["python", "main.py"]
