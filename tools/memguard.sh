#!/usr/bin/env bash
# Проверка памяти перед запуском агента или тяжёлого процесса.
# usage: tools/memguard.sh [min_free_percent=35]   exit 0 — можно запускать, 1 — нельзя.
# 24.10: ноутбук (36 ГБ) падал по OOM от параллельных агентов, Codex, torch и сборок.
min=${1:-35}
free=$(memory_pressure 2>/dev/null | awk -F': ' '/System-wide memory free percentage/{gsub("%","",$2); print $2}')
swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6}')
echo "memory free=${free:-?}% swap_used=${swap:-?}M (порог ${min}%)"
ps -axo rss=,comm= | sort -rn | head -5 | awk '{printf "  %6.0f MB  %s\n", $1/1024, $2}'
[ "${free:-0}" -ge "$min" ]
