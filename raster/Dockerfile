# syntax=docker/dockerfile:1
#
# The floor-plan service (YOLO + a model for the sheet) that reads raster plans: PNG and scanned PDF pages.
#   docker build -t northline-raster raster
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=... northline-raster
# It has no authentication and no limits of its own: do not publish it, let only the Northline server reach it.
# Without ANTHROPIC_API_KEY it still detects objects and answers rule-based questions; the key adds sheet reading,
# the inventory and free-form questions (and costs money, see raster/README.md).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    FP_SCENES=/data/scenes \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only torch first: the default Linux wheel pulls CUDA libraries (gigabytes) that this service never uses
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# OpenCV loads these at import even in the headless build (found by running the image: libxcb.so.1, then libGL.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 libgl1 && rm -rf /var/lib/apt/lists/*

# the weights are not in git (see WEIGHTS.md): fetched from the release and checked against the published hash
ARG WEIGHTS_URL=https://github.com/Nillo25/HJ/releases/download/weights-v2/floorplan_v2.pt
ARG WEIGHTS_SHA256=7e637dadb1a886f0b14468dc3da74c230d5b7a59cd4d6f2b399df0953f6f77e5
RUN mkdir -p runs/colab \
    && curl -fsSL -o runs/colab/floorplan_v2.pt "$WEIGHTS_URL" \
    && echo "$WEIGHTS_SHA256  runs/colab/floorplan_v2.pt" | sha256sum -c -

COPY fp ./fp
COPY server ./server
COPY web ./web
RUN chmod -R a+rX /app && useradd --create-home app && mkdir -p /data/scenes && chown -R app:app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:'+os.environ['PORT']+'/api/meta', timeout=4)"
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
