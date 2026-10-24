#!/usr/bin/env bash
# Живая проверка страницы «Данные» на настоящем бэкенде (local, onnx-cpu), около минуты.
# Журнал, медиа и маховик — во временном каталоге: общий data/guard.db и data/flywheel не трогаем.
# Камеры — cam_s1 и cam_s2 из configs/cameras.yaml (ролики data/videos). Сервер гасится в конце.
# usage: tools/heavy.sh frontend/scripts/flywheel-live.sh      (PORT=8011, SHOTS_DIR=…)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
PORT=${PORT:-8011}
WORK=${WORK:-$(mktemp -d /tmp/fwui-live.XXXXXX)}
PY=${GUARD_PYTHON:-/Users/nagimov-timur/Desktop/mosprom/.venv/bin/python}
export PATH="${NODE_BIN:-/opt/homebrew/opt/node@20/bin}:$PATH"

"$PY" - "$ROOT/configs/cameras.yaml" "$WORK/cameras.yaml" <<'EOF'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["cameras"] = [c for c in cfg["cameras"] if c["camera_id"] in ("cam_s1", "cam_s2")]
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
EOF

env -u PYTHONPATH PYTHONPATH="$ROOT" \
  GUARD_BACKEND=onnx-cpu GUARD_MODE=local \
  GUARD_CAMERAS_CONFIG="$WORK/cameras.yaml" GUARD_DB_PATH="$WORK/guard.db" GUARD_MEDIA_DIR="$WORK/media" \
  GUARD_FLYWHEEL_DIR="$WORK/flywheel" GUARD_FLYWHEEL_PERIOD_S="${GUARD_FLYWHEEL_PERIOD_S:-3}" \
  "$PY" -m uvicorn guard.api.app:app --host 127.0.0.1 --port "$PORT" --log-level warning >"$WORK/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; echo "сервер :$PORT остановлен, каталог $WORK"' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/api/flywheel/summary" >/dev/null || { echo "нет /api/flywheel/summary"; tail -20 "$WORK/server.log"; exit 1; }

cd frontend
UI_URL="http://127.0.0.1:$PORT/" node scripts/flywheel-live.mjs
