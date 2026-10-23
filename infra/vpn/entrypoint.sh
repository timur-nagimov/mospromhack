#!/bin/sh
# OpenVPN + прокси под надзором: упавший прокси перезапускается,
# при падении OpenVPN контейнер завершается (vpnctl.sh up поднимет заново).
set -u
RABBIT="${RABBIT_HOST:-10.0.114.30}"
JUPYTER="${JUPYTER_HOST:-10.0.114.30}"

openvpn --config /vpn/client.ovpn --auth-user-pass /vpn/auth.txt \
  --script-security 2 --up /usr/local/bin/vpn-up.sh --up-restart &
OVPN=$!

while kill -0 "$OVPN" 2>/dev/null; do
  pgrep -x microsocks >/dev/null || microsocks -q -i 0.0.0.0 -p 1080 &
  pgrep -f 'TCP-LISTEN:5672' >/dev/null || socat TCP-LISTEN:5672,fork,reuseaddr "TCP:$RABBIT:5672" &
  pgrep -f 'TCP-LISTEN:8888' >/dev/null || socat TCP-LISTEN:8888,fork,reuseaddr "TCP:$JUPYTER:80" &
  sleep 5
done
echo "openvpn завершился — контейнер останавливается"
exit 1
