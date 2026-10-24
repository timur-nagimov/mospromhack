#!/usr/bin/env bash
# Ролевая модель на мок-данных: собранный dist на :8013 (статика), сценарий rbac.mjs, стоп.
# Сборка заранее: tools/heavy.sh npm --prefix frontend run build
# usage: tools/heavy.sh frontend/scripts/rbac.sh      (PORT=8013, SHOTS_DIR=…)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${NODE_BIN:-/opt/homebrew/opt/node@20/bin}:$PATH"
PORT=${PORT:-8013}
PY=${GUARD_PYTHON:-/Users/nagimov-timur/Desktop/mosprom/.venv/bin/python}
[ -f dist/index.html ] || { echo "нет frontend/dist — сначала npm run build"; exit 1; }

env -u PYTHONPATH "$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory dist >/tmp/guard-rbac-static.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break
  sleep 0.25
done

MOCK=1 UI_URL="http://127.0.0.1:$PORT/" node scripts/rbac.mjs
