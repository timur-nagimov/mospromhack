#!/usr/bin/env bash
# Тяжёлые команды — строго по одной в своей полосе и только при запасе памяти:
# torch (экспорт, обучение, eval), компиляция SDK и сборка образов, headless-браузер, npm install/build.
# Полосы (HEAVY_LANE):
#   default — всё перечисленное выше, кроме обучения и стенда;
#   train   — обучение на MPS: одно за раз, lock считается протухшим через 4 ч;
#   stand   — компиляция и тяжёлые операции на стенде LinQ (контейнер 1.5 CPU), одна за раз.
# Lock общий для main и всех worktree: он лежит в .orchestra основного репозитория.
# usage: [HEAVY_LANE=default|train|stand] tools/heavy.sh <команда> [аргументы...]
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
COMMON=$(git -C "$DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
ROOT=$(dirname "${COMMON:-$DIR/../.git}")
case "${HEAVY_LANE:-default}" in
  default) LOCK="$ROOT/.orchestra/heavy.lock";       STALE=7200 ;;
  train)   LOCK="$ROOT/.orchestra/heavy-train.lock"; STALE=14400 ;;
  stand)   LOCK="$ROOT/.orchestra/heavy-stand.lock"; STALE=3600 ;;
  *) echo "heavy.sh: неизвестная полоса ${HEAVY_LANE}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$LOCK")"
until mkdir "$LOCK" 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || date +%s) ))
  [ "$age" -gt "$STALE" ] && rm -rf "$LOCK" && continue   # протухший lock
  echo "heavy.sh[${HEAVY_LANE:-default}]: жду очередь ($(cat "$LOCK/owner" 2>/dev/null))"; sleep 20
done
echo "$$ $*" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
until "$DIR/memguard.sh" 35 >/dev/null; do echo "heavy.sh: мало памяти, жду"; sleep 30; done
"$@"
