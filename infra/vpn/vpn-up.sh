#!/bin/sh
# DNS из push-опций сервера. Только VPN-резолвер: musl опрашивает серверы параллельно
# и берёт первый ответ, поэтому быстрый локальный NXDOMAIN ломал резолв *.lin-q.ru.
[ -f /etc/resolv.conf.orig ] || cp /etc/resolv.conf /etc/resolv.conf.orig
vpn_dns=$(env | sed -n 's/^foreign_option_[0-9]*=dhcp-option DNS //p')
if [ -n "$vpn_dns" ]; then
  for d in $vpn_dns; do echo "nameserver $d"; done > /etc/resolv.conf
fi
echo "vpn-up: dev=$dev ip=${ifconfig_local:-?} dns=[${vpn_dns}]"
