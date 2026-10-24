#!/usr/bin/env bash
# Демо с распознаванием на LinQ TPU (D-005, docs/EDGE.md). Конвейер работает на стенде
# (python -m guard.edge, backend linq-tpu), ноутбук — шлюз: журнал, оповещения и UI на 127.0.0.1:8000
# по сообщениям из очереди zazerkalye.gateway. Брокер стенда и Jupyter доступны через VPN-контейнер
# (infra/vpn/vpnctl.sh): он пробрасывает их на 127.0.0.1:5672 и 127.0.0.1:8888, системный VPN не нужен.
#
#   scripts/run_tpu.sh             VPN → edge на стенде (запускает, если не работает) → шлюз
#   SYNC=1 scripts/run_tpu.sh      сначала залить код на стенд и перезапустить edge
#   KEEP_QUEUE=1 scripts/run_tpu.sh  не очищать очередь шлюза (по умолчанию старые сообщения удаляются,
#                                  иначе в журнал попадут события, накопленные без шлюза)
# Остальные переменные (HOST, PORT, GUARD_*) передаются в scripts/run_dev.sh.
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
STAND=("$PY" tools/stand.py --url "${JUPYTER_LOCAL_URL:-http://127.0.0.1:8888/}")

if ! infra/vpn/vpnctl.sh status >/dev/null 2>&1; then
  echo "run_tpu: VPN-контейнер не отвечает — поднимаю" >&2
  infra/vpn/vpnctl.sh up
fi

if [ "${SYNC:-0}" = 1 ]; then
  echo "run_tpu: заливаю код на стенд и перезапускаю edge" >&2
  "${STAND[@]}" sync >/dev/null
  "${STAND[@]}" restart >/dev/null
elif ! "${STAND[@]}" status >/dev/null 2>&1; then
  echo "run_tpu: edge на стенде не работает — запускаю" >&2
  "${STAND[@]}" start >/dev/null
fi
"${STAND[@]}" status >/dev/null || { echo "run_tpu: edge на стенде не поднялся: ${STAND[*]} logs" >&2; exit 1; }
echo "run_tpu: edge на стенде работает (TPU)" >&2

if [ "${KEEP_QUEUE:-0}" != 1 ]; then
  PYTHONPATH="$PWD" "$PY" -m guard.edge_probe --purge --duration 1 >/dev/null
  echo "run_tpu: очередь шлюза очищена от старых сообщений" >&2
fi

export GUARD_MODE=gateway GUARD_TRANSPORT=rabbit
exec scripts/run_dev.sh "$@"
