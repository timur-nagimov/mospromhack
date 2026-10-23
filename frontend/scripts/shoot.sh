#!/usr/bin/env bash
# Поднять dev-сервер в мок-режиме, прогнать сценарий Playwright, остановить сервер.
# Всё в одном процессе-обёртке, чтобы не оставлять фоновых серверов (память ноутбука).
# usage: tools/heavy.sh frontend/scripts/shoot.sh [screenshots.mjs|smoke.mjs]
set -euo pipefail
cd "$(dirname "$0")/.."
NODE_BIN=${NODE_BIN:-/opt/homebrew/opt/node@20/bin}
export PATH="$NODE_BIN:$PATH"
PORT=${PORT:-5199}
SCRIPT=${1:-screenshots.mjs}

npx vite --mode "${VITE_MODE:-mock}" --port "$PORT" --strictPort >/tmp/guard-ui-vite.log 2>&1 &
VITE_PID=$!
trap 'kill $VITE_PID 2>/dev/null || true; wait $VITE_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://localhost:$PORT/" >/dev/null && break
  sleep 0.5
done

UI_URL="http://localhost:$PORT/" node "scripts/$SCRIPT"
