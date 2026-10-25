#!/usr/bin/env bash
# Ролевая модель на живом бэкенде: uvicorn с входом (GUARD_AUTH=1, демо-учётки) на :8013,
# собранный frontend/dist, камеры cam_s1 и cam_s2. Журнал, медиа, auth.db — во временном
# каталоге (общий data/ не трогаем, ролики берём ссылкой). Сценарий rbac.mjs, сервер гасится.
# usage: tools/heavy.sh frontend/scripts/rbac-live.sh      (PORT=8013, WAIT_S=40)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
PORT=${PORT:-8013}
WORK=${WORK:-$(mktemp -d /tmp/rbac-live.XXXXXX)}
PY=${GUARD_PYTHON:-/Users/nagimov-timur/Desktop/mosprom/.venv/bin/python}
SHARED=${GUARD_SHARED_ROOT:-/Users/nagimov-timur/Desktop/mosprom}
export PATH="${NODE_BIN:-/opt/homebrew/opt/node@20/bin}:$PATH"
[ -f frontend/dist/index.html ] || { echo "нет frontend/dist — сначала npm run build"; exit 1; }

mkdir -p "$WORK/data"
ln -sfn "$SHARED/data/videos" "$WORK/data/videos"
env -u PYTHONPATH "$PY" - "$ROOT/configs/cameras.yaml" "$WORK/cameras.yaml" <<'EOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["cameras"] = [c for c in cfg["cameras"] if c["camera_id"] in ("cam_s1", "cam_s2")]
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
EOF

env -u PYTHONPATH PYTHONPATH="$ROOT" \
  GUARD_AUTH=1 GUARD_DEMO_LOGIN=1 GUARD_BACKEND=onnx-cpu GUARD_MODE=local \
  GUARD_DATA_DIR="$WORK/data" GUARD_MODELS_DIR="$SHARED/models" \
  GUARD_CAMERAS_CONFIG="$WORK/cameras.yaml" GUARD_FRONTEND_DIST="$ROOT/frontend/dist" \
  "$PY" -m uvicorn guard.api.app:app --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; echo "сервер :$PORT остановлен, каталог $WORK"' EXIT

for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
echo "auth/config: $(curl -s "http://127.0.0.1:$PORT/api/auth/config")"
echo "events без входа: $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/events")"
# ждём первых событий с роликов
sleep "${WAIT_S:-40}"

cd frontend
MOCK=0 UI_URL="http://127.0.0.1:$PORT/" node scripts/rbac.mjs || { tail -30 "$WORK/server.log"; exit 1; }
