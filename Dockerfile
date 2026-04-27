FROM python:3.11-slim AS builder

WORKDIR /app

ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url ${TORCH_CUDA_INDEX_URL} torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# ---- runtime stage (no build-essential, git, etc.) ----
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p /app/mlruns

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=2 \
  CMD python -c "from env.poker_env import PokerEnv; PokerEnv()" || exit 1

ENTRYPOINT ["python", "main.py"]
