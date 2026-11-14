# syntax=docker/dockerfile:1
#
# North Line: ONE image, one service: the API, the built web app, the Python indexer and the floor-plan service that
# reads raster plans (raster/: YOLO + a model for the sheet). The floor-plan service listens on 127.0.0.1:8001 inside
# the container only; start.sh runs both and restarts it if it dies.
#   docker build -t northline .
#   docker run --rm -p 3000:3000 --env-file server/.env --env-file raster/.env northline
# The model key and its settings come from the environment (see server/.env.example), never from the image.

# ---- 1. the web app (Vite writes it to ../server/public)
FROM node:22-bookworm-slim AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- 2. the server: compiled TypeScript and production dependencies
FROM node:22-bookworm-slim AS server
# better-sqlite3 ships prebuilt binaries; the toolchain is only a fallback when none matches
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/server
COPY server/package.json server/package-lock.json ./
RUN npm ci
COPY server/tsconfig.json ./
COPY server/src ./src
RUN npm run build && npm prune --omit=dev

# ---- 3. what runs
FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production \
    PORT=3000 \
    UPLOADS_DIR=/data/uploads \
    CACHE_DB=/data/cache.db \
    PYTHON_BIN=/opt/venv/bin/python \
    RASTER_URL=http://127.0.0.1:8001 \
    FP_SCENES=/data/scenes \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
# Python 3.11 (Debian 12) for the indexer and the floor-plan service; the libraries are what OpenCV loads at import
# even in its headless build (found by running the image: libxcb.so.1, then libGL.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv curl libglib2.0-0 libxcb1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
# Debian's pip is old: it builds some wheels from source on the torch index; a current pip does not
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip
# CPU-only torch first: the default Linux wheel pulls CUDA libraries (gigabytes) that nothing here uses
RUN /opt/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY indexer/requirements.txt /tmp/indexer-requirements.txt
COPY raster/requirements.txt /tmp/raster-requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/indexer-requirements.txt -r /tmp/raster-requirements.txt && rm /tmp/*-requirements.txt

# the weights are not in git (raster/WEIGHTS.md): fetched from the release and checked against the published hash
ARG WEIGHTS_URL=https://github.com/Nillo25/HJ/releases/download/weights-v2/floorplan_v2.pt
ARG WEIGHTS_SHA256=7e637dadb1a886f0b14468dc3da74c230d5b7a59cd4d6f2b399df0953f6f77e5
RUN mkdir -p /app/raster/runs/colab \
    && curl -fsSL -o /app/raster/runs/colab/floorplan_v2.pt "$WEIGHTS_URL" \
    && echo "$WEIGHTS_SHA256  /app/raster/runs/colab/floorplan_v2.pt" | sha256sum -c -

WORKDIR /app
COPY indexer ./indexer
COPY raster/fp ./raster/fp
COPY raster/server ./raster/server
COPY raster/web ./raster/web
COPY start.sh ./start.sh
COPY --from=server /app/server/node_modules ./server/node_modules
COPY --from=server /app/server/dist ./server/dist
COPY --from=server /app/server/package.json ./server/package.json
COPY --from=web /app/server/public ./server/public
# files copied from a laptop can be mode 600: the app runs as "node", so everything must be readable
RUN chmod -R a+rX /app && chmod +x /app/start.sh && /opt/venv/bin/python -m compileall -q /app/indexer \
    && mkdir -p /data && chown -R node:node /data
WORKDIR /app/server
USER node
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD node -e "fetch('http://localhost:'+process.env.PORT+'/api/usage').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["/app/start.sh"]
