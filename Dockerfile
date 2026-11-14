# syntax=docker/dockerfile:1
#
# North Line: one image with the API, the built web app and the Python indexer.
#   docker build -t northline .
#   docker run --rm -p 3000:3000 -v northline-data:/data --env-file server/.env northline
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
    PYTHON_BIN=/opt/venv/bin/python
# the indexer needs Python 3.10+ (Debian 12 ships 3.11) and pdfminer.six
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY indexer/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
WORKDIR /app
COPY indexer ./indexer
COPY --from=server /app/server/node_modules ./server/node_modules
COPY --from=server /app/server/dist ./server/dist
COPY --from=server /app/server/package.json ./server/package.json
COPY --from=web /app/server/public ./server/public
COPY server/accuracy.json ./server/accuracy.json
# files copied from a laptop can be mode 600: the app runs as "node", so everything must be readable
RUN chmod -R a+rX /app && /opt/venv/bin/python -m compileall -q /app/indexer \
    && mkdir -p /data && chown -R node:node /data
WORKDIR /app/server
USER node
VOLUME /data
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD node -e "fetch('http://localhost:'+process.env.PORT+'/api/usage').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "dist/index.js"]
