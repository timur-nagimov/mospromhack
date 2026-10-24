#!/usr/bin/env bash
# Сторож памяти на время работы агентов: ноутбук (36 ГБ) уже падал по OOM,
# а 24.10 ~00:40 процесс Python из worktree агента занимал 18.5 ГБ.
# Каждые 10 с: процесс проекта (python/node из mosprom*) больше LIMIT_MB — SIGTERM, через 5 с SIGKILL;
# если свободной памяти меньше MIN_FREE% — завершается самый крупный процесс проекта.
# usage: tools/memwatch.sh [LIMIT_MB=6144] [MIN_FREE=15] [HOURS=14]
LIMIT_MB=${1:-6144}; MIN_FREE=${2:-15}; HOURS=${3:-14}
LOG=/Users/nagimov-timur/Desktop/mosprom/.orchestra/memwatch.log
mkdir -p "$(dirname "$LOG")"
end=$(( $(date +%s) + HOURS*3600 ))

project_procs() {  # pid rss_kb для python/node, чья команда или cwd внутри mosprom*
  ps -axo pid=,rss=,comm= | awk '$3 ~ /(Python|python|node)$/ {print $1, $2}' | while read -r pid rss; do
    cmdline=$(ps -o command= -p "$pid" 2>/dev/null)
    cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
    case "$cmdline $cwd" in *Desktop/mosprom*) echo "$pid $rss";; esac
  done
}

kill_proc() {  # pid reason
  local cmd; cmd=$(ps -o command= -p "$1" 2>/dev/null | cut -c1-200)
  echo "$(date '+%F %T') KILL pid=$1 ($2): $cmd" >> "$LOG"
  kill -TERM "$1" 2>/dev/null; sleep 5; kill -KILL "$1" 2>/dev/null
}

echo "$(date '+%F %T') memwatch start limit=${LIMIT_MB}MB min_free=${MIN_FREE}%" >> "$LOG"
while [ "$(date +%s)" -lt "$end" ]; do
  procs=$(project_procs)
  echo "$procs" | while read -r pid rss; do
    [ -n "$pid" ] && [ $(( rss / 1024 )) -gt "$LIMIT_MB" ] && kill_proc "$pid" "rss=$(( rss / 1024 ))MB > ${LIMIT_MB}MB"
  done
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/System-wide memory free percentage/{gsub("%","",$2); print $2}')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then
    top=$(echo "$procs" | sort -k2 -rn | head -1)
    [ -n "$top" ] && kill_proc "${top%% *}" "system free=${free}% < ${MIN_FREE}%, rss=$(( ${top##* } / 1024 ))MB"
  fi
  sleep 10
done
echo "$(date '+%F %T') memwatch stop" >> "$LOG"
