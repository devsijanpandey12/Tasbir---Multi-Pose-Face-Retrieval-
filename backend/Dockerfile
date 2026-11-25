FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TASBIR_DATA_DIR=/tmp/tasbir_data \
    TASBIR_MODELS_DIR=/app/models/buffalo_l \
    TASBIR_SESSION_TTL_MINUTES=30

WORKDIR /app

# Runtime libs required by opencv-python-headless and onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# Pre-bake official SCRFD-10G and ArcFace WebFace600K ONNX models into image
# so cold starts on Azure Container Apps take ~2s instead of downloading at startup
RUN python -c "from app.vision import VisionPipeline; vp = VisionPipeline(); vp.ensure_models_downloaded()"

RUN mkdir -p /tmp/tasbir_data/sessions

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
