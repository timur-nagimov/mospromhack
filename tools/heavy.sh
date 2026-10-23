#!/usr/bin/env bash
# Тяжёлые команды — строго по одной и только при запасе памяти:
# torch (экспорт, обучение, eval), компиляция SDK и сборка образов, headless-браузер, npm install/build.
# usage: tools/heavy.sh <команда> [аргументы...]
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
LOCK="$DIR/../.orchestra/heavy.lock"
until mkdir "$LOCK" 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || date +%s) ))
  [ "$age" -gt 7200 ] && rmdir "$LOCK" 2>/dev/null && continue   # протухший lock (> 2 ч)
  echo "heavy.sh: жду очередь ($(cat "$LOCK/owner" 2>/dev/null))"; sleep 20
done
echo "$$ $*" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
until "$DIR/memguard.sh" 35 >/dev/null; do echo "heavy.sh: мало памяти, жду"; sleep 30; done
"$@"
