"""Мягкое ограничение подбора пароля: N неудачных входов подряд на пару «логин, адрес» — пауза.

Дополнительно (ip_max_failures) — общий счётчик неудач на адрес: перебор многих логинов с одного адреса
тоже получает паузу, и журнал действий не раздувается записями auth.login_failed. Удачный вход
сбрасывает только счётчик пары: своя учётка не обнуляет перебор чужих.

Состояние в памяти процесса: после перезапуска счётчики обнуляются, это осознанно (ограничение мягкое).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

Key = tuple[str, str]


@dataclass
class _State:
    failures: int = 0
    blocked_until: float = 0.0


class LoginLimiter:
    def __init__(
        self,
        max_failures: int = 10,
        block_s: float = 60.0,
        clock: Callable[[], float] = time.time,
        max_keys: int = 10_000,
        ip_max_failures: int | None = None,
    ) -> None:
        self.max_failures = max_failures
        self.block_s = block_s
        self.clock = clock
        self.max_keys = max_keys
        self.ip_max_failures = ip_max_failures  # None — без общего счётчика на адрес
        self._lock = threading.Lock()
        self._state: OrderedDict[Key, _State] = OrderedDict()
        self._ip_state: OrderedDict[Key, _State] = OrderedDict()  # отдельно: не пересечётся с парами

    def __len__(self) -> int:
        return len(self._state) + len(self._ip_state)

    @staticmethod
    def _key(username: str, ip: str | None) -> Key:
        return (username or "").strip().lower(), ip or ""

    @staticmethod
    def _ip_key(ip: str | None) -> Key:
        return "", ip or ""

    @staticmethod
    def _current(table: OrderedDict[Key, _State], key: Key, now: float) -> _State | None:
        state = table.get(key)
        if state is not None and state.blocked_until and state.blocked_until <= now:
            del table[key]  # пауза кончилась — счёт с нуля
            return None
        return state

    @classmethod
    def _wait(cls, table: OrderedDict[Key, _State], key: Key, now: float) -> float:
        state = cls._current(table, key, now)
        if state is None or not state.blocked_until:
            return 0
        return state.blocked_until - now

    def retry_after(self, username: str, ip: str | None) -> float:
        """Сколько секунд ещё нельзя входить; 0 — можно."""
        with self._lock:
            now = self.clock()
            wait = self._wait(self._state, self._key(username, ip), now)
            if self.ip_max_failures:
                wait = max(wait, self._wait(self._ip_state, self._ip_key(ip), now))
            return wait

    def _count(self, table: OrderedDict[Key, _State], key: Key, limit: int, now: float) -> float:
        state = self._current(table, key, now)
        if state is None:
            state = table[key] = _State()
        elif state.blocked_until:
            return 0
        table.move_to_end(key)
        state.failures += 1
        if state.failures >= limit:
            state.blocked_until = now + self.block_s
            return self.block_s
        self._evict(table, now)
        return 0

    def failure(self, username: str, ip: str | None) -> float:
        """Учесть неудачу; вернуть длину паузы, если она началась сейчас, иначе 0."""
        with self._lock:
            now = self.clock()
            started = self._count(self._state, self._key(username, ip), self.max_failures, now)
            if self.ip_max_failures:
                started = max(started, self._count(self._ip_state, self._ip_key(ip), self.ip_max_failures, now))
            return started

    def _evict(self, table: OrderedDict[Key, _State], now: float) -> None:
        """Держать не больше max_keys ключей; первыми уходят старые счётчики без активной паузы."""
        while len(table) > self.max_keys:
            victim = next((k for k, s in table.items() if s.blocked_until <= now), None)
            if victim is None:
                table.popitem(last=False)
            else:
                del table[victim]

    def success(self, username: str, ip: str | None) -> None:
        with self._lock:
            self._state.pop(self._key(username, ip), None)
