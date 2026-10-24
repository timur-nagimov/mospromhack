"""Мягкое ограничение подбора пароля: N неудачных входов подряд на пару «логин, адрес» — пауза.

Состояние в памяти процесса: после перезапуска счётчики обнуляются, это осознанно (ограничение мягкое).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


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
    ) -> None:
        self.max_failures = max_failures
        self.block_s = block_s
        self.clock = clock
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._state: OrderedDict[tuple[str, str], _State] = OrderedDict()

    def __len__(self) -> int:
        return len(self._state)

    @staticmethod
    def _key(username: str, ip: str | None) -> tuple[str, str]:
        return (username or "").strip().lower(), ip or ""

    def _current(self, key: tuple[str, str], now: float) -> _State | None:
        state = self._state.get(key)
        if state is not None and state.blocked_until and state.blocked_until <= now:
            del self._state[key]  # пауза кончилась — счёт с нуля
            return None
        return state

    def retry_after(self, username: str, ip: str | None) -> float:
        """Сколько секунд ещё нельзя входить; 0 — можно."""
        with self._lock:
            now = self.clock()
            state = self._current(self._key(username, ip), now)
            if state is None or not state.blocked_until:
                return 0
            return state.blocked_until - now

    def failure(self, username: str, ip: str | None) -> float:
        """Учесть неудачу; вернуть длину паузы, если она началась сейчас, иначе 0."""
        key = self._key(username, ip)
        with self._lock:
            now = self.clock()
            state = self._current(key, now)
            if state is None:
                state = self._state[key] = _State()
            elif state.blocked_until:
                return 0
            self._state.move_to_end(key)
            state.failures += 1
            if state.failures >= self.max_failures:
                state.blocked_until = now + self.block_s
                return self.block_s
            self._evict(now)
            return 0

    def _evict(self, now: float) -> None:
        """Держать не больше max_keys пар; первыми уходят старые счётчики без активной паузы."""
        while len(self._state) > self.max_keys:
            victim = next((k for k, s in self._state.items() if s.blocked_until <= now), None)
            if victim is None:
                self._state.popitem(last=False)
            else:
                del self._state[victim]

    def success(self, username: str, ip: str | None) -> None:
        with self._lock:
            self._state.pop(self._key(username, ip), None)
