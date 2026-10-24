"""Deterministic in-process transport for reconnect and uncertain-delivery tests."""

from __future__ import annotations

from collections import deque
from typing import Callable

from guard.transport.protocol import Envelope, decode, encode


class FakeBroker:
    def __init__(self):
        self.online = True
        self.lose_confirm = False
        self.events: deque[bytes] = deque()
        self.telemetry: deque[bytes] = deque(maxlen=32)

    def publish(self, envelope: Envelope, *, durable: bool = True) -> None:
        if not self.online:
            raise ConnectionError("Fake broker disconnected")
        (self.events if durable else self.telemetry).append(encode(envelope))
        if self.lose_confirm:
            self.lose_confirm = False
            raise ConnectionError("Confirmation lost after accepted publication")

    def deliver(self, callback: Callable[[Envelope], None], *, durable: bool = True, duplicate: bool = False, newest: bool = False):
        if not self.online:
            raise ConnectionError("Fake broker disconnected")
        queue = self.events if durable else self.telemetry
        data = queue.pop() if newest else queue.popleft()
        try:
            callback(decode(data))
        except Exception:
            queue.appendleft(data)
            raise
        if duplicate:
            queue.appendleft(data)

    def close(self):
        pass

    def status(self):
        return {"connected": self.online, "events_pending": len(self.events)}
