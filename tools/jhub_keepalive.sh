#!/usr/bin/env bash
# Пинг активности JupyterHub стенда, чтобы idle-culler не остановил сервер с конвейером.
# Раз в 5 мин делает лёгкий запрос через tools/jhub.py. usage: tools/jhub_keepalive.sh [HOURS=14]
HOURS=${1:-14}
ROOT=/Users/nagimov-timur/Desktop/mosprom
LOG=$ROOT/.orchestra/keepalive.log
mkdir -p "$(dirname "$LOG")"
end=$(( $(date +%s) + HOURS*3600 ))
unset PYTHONPATH
while [ "$(date +%s)" -lt "$end" ]; do
  if "$ROOT/.venv/bin/python" "$ROOT/tools/jhub.py" --url http://127.0.0.1:8888/ status >/dev/null 2>&1; then
    echo "$(date '+%F %T') ok" >> "$LOG"
  else
    echo "$(date '+%F %T') FAIL status — повторный login" >> "$LOG"
    "$ROOT/.venv/bin/python" "$ROOT/tools/jhub.py" --url http://127.0.0.1:8888/ login >> "$LOG" 2>&1
  fi
  sleep 300
done
