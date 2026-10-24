#!/usr/bin/env bash
# Dev-сервер: uvicorn guard.api.app:app на порту 8000 (HOST/PORT переопределяются).
# GUARD_* (данные, модели, бэкенд) берутся из окружения или из .env корня проекта.
# Перед стартом — проверка свободной памяти (tools/memguard.sh), ноутбук уже падал по OOM.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${GUARD_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in .venv/bin/python /Users/nagimov-timur/Desktop/mosprom/.venv/bin/python; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
PY="${PY:-python3}"

if [ -x tools/memguard.sh ] && ! tools/memguard.sh "${GUARD_MIN_FREE:-35}"; then
  echo "run_dev: мало свободной памяти, сервер не запускаю" >&2
  exit 1
fi

# Веб-интерфейс раздаётся из frontend/dist (в git его нет). Если сборки нет — собираем: Node 18+, npm.
if [ ! -f frontend/dist/index.html ]; then
  NODE_BIN=""
  for cand in /opt/homebrew/opt/node@20/bin /usr/local/opt/node@20/bin; do
    if [ -x "$cand/node" ]; then NODE_BIN="$cand"; break; fi
  done
  echo "run_dev: интерфейс не собран — собираю frontend (npm ci && npm run build)..." >&2
  if ( cd frontend && PATH="${NODE_BIN:+$NODE_BIN:}$PATH" && npm ci --no-audit --no-fund --loglevel=error && npm run build ) >&2; then
    echo "run_dev: интерфейс собран" >&2
  else
    echo "run_dev: собрать интерфейс не удалось — API работает, UI нет (см. frontend/README.md)" >&2
  fi
fi
# Унаследованный PYTHONPATH не берём: на этом маке там FUSE-монтирование arcadia, оно иногда отвечает
# «Device not configured», и интерпретатор падает на старте.
export PYTHONPATH="$PWD"
exec "$PY" -m uvicorn guard.api.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --log-level info "$@"
