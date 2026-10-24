#!/usr/bin/env bash
# Демо с распознаванием на LinQ TPU (D-005, docs/EDGE.md). Конвейер работает на стенде
# (python -m guard.edge, backend linq-tpu), ноутбук — шлюз: журнал, оповещения и UI на 127.0.0.1:8000
# через durable event queue и bounded telemetry.v2. Брокер стенда и Jupyter доступны через VPN-контейнер
# (infra/vpn/vpnctl.sh): он пробрасывает их на 127.0.0.1:5672 и 127.0.0.1:8888, системный VPN не нужен.
#
#   scripts/run_tpu.sh             VPN → edge на стенде (запускает, если не работает) → шлюз
#   SYNC=1 scripts/run_tpu.sh      сначала залить код на стенд и перезапустить edge
#   GUARD_VPN_MODE=external scripts/run_tpu.sh  использовать готовые локальные пробросы без VPN up
# Очереди и накопленные события сохраняются: автоматической очистки нет.
# Остальные переменные (HOST, PORT, GUARD_*) передаются в scripts/run_dev.sh.
# Вход и роли (docs/ROLES.md): GUARD_AUTH=1 и GUARD_DEMO_LOGIN=1 по умолчанию; демо-учётки печатает
# scripts/run_dev.sh перед стартом шлюза (python -m guard.auth banner).
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH  # FUSE arcadia в PYTHONPATH роняет интерпретатор, см. run_dev.sh

PY="${GUARD_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in .venv/bin/python; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
PY="${PY:-python3}"
export JUPYTER_LOCAL_URL="${JUPYTER_LOCAL_URL:-http://127.0.0.1:8888/}"
export GUARD_RABBIT_URL="${GUARD_RABBIT_URL:-amqp://guest:guest@127.0.0.1:5672/}"
STAND=("$PY" tools/stand.py --url "$JUPYTER_LOCAL_URL")

forward_ready() {
  # Only public AMQP protocol negotiation and Jupyter HTTP reachability. No
  # credentials are printed, and this is not an inference acceptance check.
  "$PY" - <<'PY' >/dev/null 2>&1
import os
import socket
import ssl
from urllib.parse import urlsplit
from urllib.request import urlopen

broker = urlsplit(os.environ["GUARD_RABBIT_URL"])
if broker.scheme not in {"amqp", "amqps"} or not broker.hostname:
    raise ValueError("Invalid event-bus URL")
connection = socket.create_connection((broker.hostname, broker.port or (5671 if broker.scheme == "amqps" else 5672)), timeout=3)
if broker.scheme == "amqps":
    connection = ssl.create_default_context().wrap_socket(connection, server_hostname=broker.hostname)
with connection:
    connection.sendall(b"AMQP\x00\x00\x09\x01")
    header = b""
    while len(header) < 3:
        chunk = connection.recv(3 - len(header))
        if not chunk:
            raise ConnectionError("AMQP negotiation unavailable")
        header += chunk
    if header != b"\x01\x00\x00":
        raise ConnectionError("Unexpected AMQP negotiation")
with urlopen(os.environ["JUPYTER_LOCAL_URL"], timeout=3) as response:
    if response.status >= 400:
        raise ConnectionError("Jupyter unavailable")
PY
}

ensure_container() {
  if ! infra/vpn/vpnctl.sh status >/dev/null 2>&1; then
    echo "run_tpu: VPN-контейнер не отвечает — поднимаю" >&2
    infra/vpn/vpnctl.sh up
  fi
  if ! forward_ready; then
    echo "run_tpu: RabbitMQ или Jupyter недоступны после запуска VPN-контейнера" >&2
    exit 1
  fi
}

case "${GUARD_VPN_MODE:-auto}" in
  external)
    if ! forward_ready; then
      echo "run_tpu: внешние пробросы RabbitMQ/Jupyter недоступны; VPN автоматически не запускается" >&2
      exit 1
    fi
    ;;
  auto)
    if forward_ready; then
      echo "run_tpu: использую доступные RabbitMQ/Jupyter без запуска VPN-контейнера" >&2
    else
      ensure_container
    fi
    ;;
  container) ensure_container ;;
  *) echo "run_tpu: GUARD_VPN_MODE должен быть auto, external или container" >&2; exit 2 ;;
esac

if [ "${SYNC:-0}" = 1 ]; then
  echo "run_tpu: заливаю код на стенд и перезапускаю edge" >&2
  "${STAND[@]}" sync >/dev/null
  "${STAND[@]}" restart >/dev/null
elif ! "${STAND[@]}" status >/dev/null 2>&1; then
  echo "run_tpu: edge на стенде не работает — запускаю" >&2
  "${STAND[@]}" start >/dev/null
fi
"${STAND[@]}" status >/dev/null || { echo "run_tpu: edge на стенде не поднялся: ${STAND[*]} logs" >&2; exit 1; }
echo "run_tpu: edge-процесс запущен; работа TPU требует отдельной проверки активных счётчиков моделей" >&2

export GUARD_MODE=gateway GUARD_TRANSPORT=rabbit
export GUARD_EDGE_ID="${GUARD_EDGE_ID:-stand-linq}"
export GUARD_BACKEND_FALLBACK=none
exec scripts/run_dev.sh "$@"
