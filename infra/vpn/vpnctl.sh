#!/usr/bin/env bash
# Доступ к стенду хакатона без системного VPN: OpenVPN в контейнере Podman.
#   build  — собрать образ
#   up     — поднять контейнер (пересоздаёт)
#   down   — остановить
#   status — проверить туннель, Rabbit и Jupyter
#   logs   — последние строки лога OpenVPN
# Наружу на 127.0.0.1: SOCKS5 :1080 (вся сеть 10.0.114.0/24, DNS внутри VPN), RabbitMQ :5672, Jupyter :8888.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
set -a; . "$ROOT/.env"; set +a
NAME="${VPN_CONTAINER:-mosprom-hackvpn}"
IMAGE=localhost/mosprom-vpn:latest
SECRETS="$ROOT/vendor/vpn"

case "${1:-status}" in
  build)
    podman build -q -t "$IMAGE" "$ROOT/infra/vpn"
    ;;
  up)
    mkdir -p "$SECRETS"
    (umask 077; printf '%s\n%s\n' "$VPN_LOGIN" "$VPN_PASSWORD" > "$SECRETS/auth.txt")
    cp "$VPN_PROFILE" "$SECRETS/client.ovpn"
    podman rm -f "$NAME" >/dev/null 2>&1 || true
    podman run -d --name "$NAME" --cap-add=NET_ADMIN --device /dev/net/tun --add-host hackathon.lin-q.ru:10.0.114.30 \
      -v "$SECRETS/client.ovpn:/vpn/client.ovpn:ro" -v "$SECRETS/auth.txt:/vpn/auth.txt:ro" \
      -p 127.0.0.1:1080:1080 -p 127.0.0.1:5672:5672 -p 127.0.0.1:8888:8888 "$IMAGE" >/dev/null
    for _ in $(seq 1 45); do
      podman exec "$NAME" ip link show tun0 >/dev/null 2>&1 && break
      sleep 1
    done
    "$0" status
    ;;
  down)
    podman rm -f "$NAME" >/dev/null && echo "stopped"
    ;;
  status)
    if ! podman container exists "$NAME" || [ "$(podman inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
      echo "VPN: DOWN (контейнер не запущен) — запусти: infra/vpn/vpnctl.sh up"; exit 1
    fi
    tun=$(podman exec "$NAME" sh -c "ip -4 addr show tun0 2>/dev/null | awk '/inet /{print \$2}'")
    [ -n "$tun" ] && echo "VPN: UP tun0=$tun" || { echo "VPN: туннель не поднят"; podman logs --tail 15 "$NAME"; exit 1; }
    podman exec "$NAME" nc -z -w 3 "${RABBIT_HOST:-10.0.114.30}" 5672 && echo "Rabbit 10.0.114.30:5672 (через туннель): OK" || echo "Rabbit: НЕДОСТУПЕН"
    echo "Jupyter DNS в VPN: $(podman exec "$NAME" sh -c 'nslookup hackathon.lin-q.ru 2>/dev/null | awk "/^Address/ && !/#/{print \$2}" | head -1')"
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 --socks5-hostname 127.0.0.1:1080 http://hackathon.lin-q.ru/ || true)
    echo "Jupyter через SOCKS 127.0.0.1:1080: HTTP $code"
    nc -z -G 3 127.0.0.1 5672 && echo "Rabbit проброс 127.0.0.1:5672: OK" || echo "Rabbit проброс: НЕТ"
    ;;
  logs)
    podman logs --tail "${2:-40}" "$NAME"
    ;;
  *)
    echo "usage: $0 build|up|down|status|logs"; exit 2
    ;;
esac
