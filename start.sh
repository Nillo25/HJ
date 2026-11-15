#!/bin/bash
# One container, two processes: the floor-plan service (internal, 127.0.0.1:8001) and the Northline server (PORT).
# The floor-plan service is restarted if it dies (memory, a bad file); if the server exits the container exits and the host restarts it.
(
  cd /app/raster
  while true; do
    /opt/venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
    echo "[start] floor-plan service stopped, restarting in 3 s" >&2
    sleep 3
  done
) &
LOOP=$!

cd /app/server
node dist/index.js &
NODE=$!

stop() { kill "$NODE" "$LOOP" 2>/dev/null; pkill -P "$LOOP" 2>/dev/null; wait; }
trap stop TERM INT
wait "$NODE"
code=$?
stop
exit "$code"
