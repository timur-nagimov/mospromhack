"""Шина сообщений для WebSocket: публикация из любых потоков, подписчики — asyncio-очереди."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


class Subscription:
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int) -> None:
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _put(self, message: dict[str, Any]) -> None:
        # медленный клиент не тормозит остальных: выбрасываем самое старое
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(message)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()


class EventBus:
    def __init__(self, maxsize: int = 256) -> None:
        self._subs: set[Subscription] = set()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self.published = 0

    def subscribe(self) -> Subscription:
        sub = Subscription(asyncio.get_running_loop(), self._maxsize)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, message: dict[str, Any]) -> None:
        """Потокобезопасно: можно звать из потоков камер и из event loop."""
        with self._lock:
            subs = list(self._subs)
        self.published += 1
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(sub._put, message)
            except RuntimeError:  # цикл закрыт — подписчик мёртв
                self.unsubscribe(sub)
