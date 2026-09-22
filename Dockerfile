# Week 3, Day 5 — Dockerfile
FROM python:3.10-slim

# OpenCV needs these system libraries at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements FIRST so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Checkpoints are NOT baked into the image — mount them at runtime instead
ENV SAM_CHECKPOINT=/app/checkpoints/sam_vit_b_01ec64.pth
ENV SAM_MODEL_TYPE=vit_b
ENV CLASSIFIER_CHECKPOINT=/app/checkpoints/damage_classifier_best.pth

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
