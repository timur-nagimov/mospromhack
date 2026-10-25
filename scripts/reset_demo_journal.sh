#!/usr/bin/env bash
# Сброс журнала демо перед показом. Зацикленные ролики за ночь набирают тысячи событий,
# и счётчики «Мониторинга» становятся неправдоподобными.
#   1) остановить шлюз на PORT (8000);
#   2) перенести журнал (guard.db с -wal и -shm) в data/archive/guard-<дата-время>.db;
#      медиа, auth.db и flywheel.db не трогаются, очереди брокера тоже;
#   3) засеять демо-сотрудников (python -m tools.seed_demo_employees);
#   4) снова запустить шлюз в фоне той же командой демо, лог — GATEWAY_LOG.
#
#   scripts/reset_demo_journal.sh
#   GATEWAY_LOG=/path/to/run_tpu.log scripts/reset_demo_journal.sh
# Команда шлюза: GUARD_PREVIEW_STALE_AFTER_S=30 GUARD_VPN_MODE=external scripts/run_tpu.sh
# (README, «Демо на защите»). Остальные GUARD_* наследуются из окружения.
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH  # FUSE arcadia в PYTHONPATH роняет интерпретатор, см. run_dev.sh

PY="${GUARD_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in .venv/bin/python /Users/nagimov-timur/Desktop/mosprom/.venv/bin/python; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
PY="${PY:-python3}"
PORT="${PORT:-8000}"
DATA_DIR="${GUARD_DATA_DIR:-data}"
DB="${GUARD_DB_PATH:-$DATA_DIR/guard.db}"
LOG="${GATEWAY_LOG:-$DATA_DIR/gateway.log}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# 1. Шлюз
pids="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
if [ -n "$pids" ]; then
  echo "reset_demo_journal: останавливаю шлюз на :$PORT (pid $pids)" >&2
  kill $pids
  for _ in $(seq 1 30); do
    alive=""
    for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ -z "$alive" ] && break
    sleep 0.5
  done
  if [ -n "$alive" ]; then
    echo "reset_demo_journal: шлюз не остановился за 15 с — журнал не трогаю" >&2
    exit 1
  fi
fi

# 2. Архив журнала
if [ -f "$DB" ]; then
  mkdir -p "$DATA_DIR/archive"
  target="$DATA_DIR/archive/guard-$STAMP.db"
  mv "$DB" "$target"
  for suffix in -wal -shm; do
    if [ -f "$DB$suffix" ]; then mv "$DB$suffix" "$target$suffix"; fi
  done
  echo "reset_demo_journal: журнал перенесён в $target" >&2
else
  echo "reset_demo_journal: журнала $DB нет — создаю новый" >&2
fi

# 3. Демо-сотрудники (создаёт пустой журнал со схемой)
"$PY" -m tools.seed_demo_employees --db-path "$DB" >&2

# 4. Шлюз снова в фоне
mkdir -p "$(dirname "$LOG")"
echo "=== reset_demo_journal $(date '+%F %T'): архив $STAMP, перезапуск шлюза ===" >> "$LOG"
(
  GUARD_PREVIEW_STALE_AFTER_S="${GUARD_PREVIEW_STALE_AFTER_S:-30}" GUARD_VPN_MODE="${GUARD_VPN_MODE:-external}" \
    nohup scripts/run_tpu.sh >> "$LOG" 2>&1 &
)
for _ in $(seq 1 120); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "reset_demo_journal: шлюз отвечает на :$PORT, лог $LOG" >&2
    exit 0
  fi
  sleep 1
done
echo "reset_demo_journal: шлюз не ответил за 120 с, см. $LOG" >&2
exit 1
