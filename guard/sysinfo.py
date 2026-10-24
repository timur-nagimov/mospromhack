"""Замеры узла без зависимостей: CPU и память процесса и контейнера (cgroup v2), сдвиг часов по NTP.

Нужны edge (метрики в транспорт) и замеряющему подписчику на ноутбуке (guard.edge_probe).
Контейнер Jupyter стенда ограничен через cgroup v2: cpu.max = "150000 100000" (1.5 CPU),
memory.max = 4 ГиБ. nproc и free показывают хост (64 ядра), поэтому загрузку считаем по cgroup.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

CGROUP = Path("/sys/fs/cgroup")
NTP_EPOCH_DELTA = 2208988800  # 1900-01-01 → 1970-01-01, с
DEFAULT_NTP_SERVERS = ("time.google.com", "pool.ntp.org", "time.cloudflare.com")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def cgroup_cpu_usage_s(root: Path = CGROUP) -> float | None:
    """Суммарное процессорное время всех процессов контейнера, с (cpu.stat: usage_usec)."""
    text = _read(root / "cpu.stat")
    if not text:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec":
            try:
                return int(value) / 1e6
            except ValueError:
                return None
    return None


def cgroup_cpu_limit(root: Path = CGROUP) -> float | None:
    """Квота в ядрах из cpu.max ("150000 100000" → 1.5); None — без квоты или не cgroup v2."""
    text = _read(root / "cpu.max")
    if not text:
        return None
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        return int(parts[0]) / int(parts[1])
    except (ValueError, ZeroDivisionError):
        return None


def cgroup_memory(root: Path = CGROUP) -> tuple[int | None, int | None]:
    """(memory.current, memory.max) в байтах; max=None — без ограничения."""
    current = _read(root / "memory.current")
    limit = _read(root / "memory.max")
    try:
        cur = int(current) if current else None
    except ValueError:
        cur = None
    try:
        lim = int(limit) if limit and limit != "max" else None
    except ValueError:
        lim = None
    return cur, lim


def process_rss_bytes() -> int | None:
    """Текущий RSS процесса (Linux /proc/self/status); на macOS — None."""
    text = _read(Path("/proc/self/status"))
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def process_cpu_s() -> float:
    t = os.times()
    return t.user + t.system


class CpuSampler:
    """Загрузка CPU между вызовами sample(): процесс и весь контейнер, в ядрах и в % квоты."""

    def __init__(self, cgroup_root: Path = CGROUP) -> None:
        self.root = cgroup_root
        self.limit_cores = cgroup_cpu_limit(cgroup_root)
        self._last = self._now()

    def _now(self) -> tuple[float, float, float | None]:
        return time.monotonic(), process_cpu_s(), cgroup_cpu_usage_s(self.root)

    def sample(self) -> dict[str, Any]:
        now = self._now()
        (t0, p0, c0), (t1, p1, c1) = self._last, now
        self._last = now
        dt = t1 - t0
        out: dict[str, Any] = {
            "window_s": round(dt, 2),
            "process_cores": None,
            "container_cores": None,
            "limit_cores": self.limit_cores,
            "container_pct_of_limit": None,
        }
        if dt <= 0:
            return out
        out["process_cores"] = round((p1 - p0) / dt, 3)
        if c0 is not None and c1 is not None:
            cores = (c1 - c0) / dt
            out["container_cores"] = round(cores, 3)
            if self.limit_cores:
                out["container_pct_of_limit"] = round(100.0 * cores / self.limit_cores, 1)
        return out


def memory_snapshot(cgroup_root: Path = CGROUP) -> dict[str, Any]:
    rss = process_rss_bytes()
    cur, lim = cgroup_memory(cgroup_root)
    mb = 1024 * 1024
    return {
        "process_rss_mb": round(rss / mb, 1) if rss is not None else None,
        "container_mb": round(cur / mb, 1) if cur is not None else None,
        "limit_mb": round(lim / mb, 1) if lim is not None else None,
    }


# ---------------------------------------------------------------- NTP (SNTP, RFC 4330)


def _ntp_time(data: bytes, offset: int) -> float:
    sec, frac = struct.unpack("!II", data[offset:offset + 8])
    return sec - NTP_EPOCH_DELTA + frac / 2**32


def ntp_query(server: str, timeout: float = 2.0, port: int = 123) -> dict[str, Any]:
    """Один запрос SNTP: offset (сколько прибавить к локальным часам, чтобы получить время сервера)
    и rtt без времени обработки на сервере. Погрешность offset не больше rtt/2."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        t0 = time.time()
        sock.sendto(b"\x1b" + 47 * b"\0", (server, port))
        data, _ = sock.recvfrom(48)
        t3 = time.time()
    if len(data) < 48:
        raise ValueError("short NTP reply")
    t1 = _ntp_time(data, 32)  # сервер получил
    t2 = _ntp_time(data, 40)  # сервер ответил
    return {
        "server": server,
        "offset_ms": round(((t1 - t0) + (t2 - t3)) / 2 * 1000.0, 2),
        "rtt_ms": round(((t3 - t0) - (t2 - t1)) * 1000.0, 2),
        "measured_at": round(t3, 3),
    }


def ntp_best(servers: tuple[str, ...] | list[str] = DEFAULT_NTP_SERVERS, samples: int = 4,
             timeout: float = 2.0) -> dict[str, Any] | None:
    """Лучший из нескольких замеров (минимальный rtt — минимальная погрешность); None — NTP недоступен."""
    best: dict[str, Any] | None = None
    errors: list[str] = []
    for server in servers:
        failures = 0
        for _ in range(max(1, samples)):
            try:
                res = ntp_query(server, timeout=timeout)
            except (OSError, ValueError) as exc:  # UDP теряется: пробуем ещё раз, после 2 отказов — другой сервер
                errors.append(f"{server}: {type(exc).__name__}")
                failures += 1
                if failures >= 2:
                    break
                continue
            if best is None or res["rtt_ms"] < best["rtt_ms"]:
                best = res
    if best is not None:
        best = {**best, "samples": samples, "uncertainty_ms": round(best["rtt_ms"] / 2, 2)}
    elif errors:
        return {"error": "; ".join(errors[:4])}
    return best


class NtpClock:
    """Фоновое уточнение сдвига часов по NTP раз в period_s (первый замер — сразу в потоке)."""

    def __init__(self, servers: tuple[str, ...] | list[str] = DEFAULT_NTP_SERVERS, period_s: float = 60.0) -> None:
        self.servers = tuple(servers)
        self.period_s = period_s
        self.state: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None and self.servers:
            self._thread = threading.Thread(target=self._run, name="ntp-clock", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            res = ntp_best(self.servers)
            if res is not None and ("offset_ms" in res or self.state is None or "offset_ms" not in self.state):
                self.state = res
            if self._stop.wait(self.period_s):
                return

    def stop(self) -> None:
        self._stop.set()
