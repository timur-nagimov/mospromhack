"""In-process queue for local mode and tests.

MemoryTransport implements both interfaces directly. MemorySink/MemorySource
provide separate endpoints over the same transport when desired.

Messages are copied on publication. Consumers compete for queue items; this
transport is not a broadcast bus. close() seals publication and allows queued
messages to drain. An iterator can also drain the queue after close().
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator

from .messages import (
    MessageCallback,
    MessageInput,
    Msg,
    TransportClosed,
    parse_message,
)


class MemoryTransport:
    def __init__(self) -> None:
        self._messages: deque[Msg] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._thread: threading.Thread | None = None
        self.callback_error: Exception | None = None

    def publish(self, message: MessageInput) -> bool:
        owned = parse_message(message).model_copy(deep=True)
        with self._condition:
            if self._closed:
                raise TransportClosed("Memory transport is closed")
            self._messages.append(owned)
            self._condition.notify_all()
        return True

    def iter(self) -> Iterator[Msg]:
        while True:
            with self._condition:
                while not self._messages and not self._closed:
                    self._condition.wait()
                if not self._messages:
                    return
                message = self._messages.popleft()
            yield message

    def on_message(self, callback: MessageCallback) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._closed:
                raise TransportClosed("Memory transport is closed")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A callback consumer is already running")
            self.callback_error = None
            self._thread = threading.Thread(
                target=self._consume,
                args=(callback,),
                name="guard-memory-source",
                daemon=True,
            )
            self._thread.start()

    def _consume(self, callback: MessageCallback) -> None:
        for message in self.iter():
            try:
                callback(message)
            except Exception as exc:
                # Retain the failed item and stop. The caller can inspect the
                # error and restart the callback or drain using iter().
                with self._condition:
                    self._messages.appendleft(message)
                    self.callback_error = exc
                    self._condition.notify_all()
                return

    def close(self, timeout: float | None = 5.0) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Memory consumer has not finished")
        if self.callback_error is not None:
            raise RuntimeError(
                "Memory callback failed; the message remains queued"
            ) from self.callback_error


class MemorySink:
    def __init__(self, transport: MemoryTransport) -> None:
        self.transport = transport

    def publish(self, message: MessageInput) -> bool:
        return self.transport.publish(message)

    def close(self, timeout: float | None = 5.0) -> None:
        self.transport.close(timeout)


class MemorySource:
    def __init__(self, transport: MemoryTransport) -> None:
        self.transport = transport

    def iter(self) -> Iterator[Msg]:
        return self.transport.iter()

    def on_message(self, callback: MessageCallback) -> None:
        self.transport.on_message(callback)

    def close(self, timeout: float | None = 5.0) -> None:
        self.transport.close(timeout)
