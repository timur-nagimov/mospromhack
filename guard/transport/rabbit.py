"""Threaded RabbitMQ transport with bounded buffering and publisher confirms.

Events and updates are never evicted. If only events occupy the bounded
outbox, publish() applies backpressure; an explicit timeout raises instead of
silently dropping an event. The bound includes the in-flight message.

The broker uses reject-publish overflow rather than RabbitMQ's default
drop-head policy, which could discard events. Rejected or unconfirmed messages
remain in the outbox and are retried. Both endpoints declare the topology, so
publishing before the gateway starts does not lose unroutable events.

Delivery is at least once. A lost confirmation or acknowledgement can cause
redelivery. The exchange and queue are deliberately non-durable, as specified;
this is not a disk-backed outbox or a broker-restart durability guarantee.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pika

from .codec import decode, encode
from .messages import (
    MessageCallback,
    MessageInput,
    TransportClosed,
    parse_message,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "amqp://guest:guest@127.0.0.1:5672/"
EXCHANGE = "zazerkalye"
GATEWAY_QUEUE = "zazerkalye.gateway"
BINDINGS = ("edge.event", "edge.frame_meta", "edge.metrics")
FRAME_TTL_SECONDS = 10.0
# Сколько неподтверждённых сообщений брокер держит у потребителя. С prefetch 1 следующее сообщение
# приходит только после ack, то есть одно за RTT: через VPN стенда (225 мс) ~4 сообщения/с при ~20/с
# от edge, и оповещение ждёт за frame_meta ~TTL (замер A10, docs/EDGE.md §7.1: 12.6 с вместо 2.2 с).
DEFAULT_PREFETCH = 200

_ROUTING_KEYS = {
    "event": "edge.event",
    "event_update": "edge.event",
    "frame_meta": "edge.frame_meta",
    "metrics": "edge.metrics",
    "heartbeat": "edge.metrics",
}
_EVENT_KINDS = frozenset(("event", "event_update"))


@dataclass(frozen=True)
class _Packet:
    kind: str
    body: bytes
    headers: dict[str, Any]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enqueued_at: float = field(default_factory=time.monotonic)

    @classmethod
    def make(cls, message: MessageInput) -> _Packet:
        msg = parse_message(message)
        body, headers = encode(msg)
        return cls(kind=msg.kind, body=body, headers=headers)


class _Outbox:
    """Single consumer, multiple producers; retries retain the in-flight slot."""

    def __init__(self, maxsize: int) -> None:
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = maxsize
        self._items: deque[_Packet] = deque()
        self._inflight: _Packet | None = None
        self._accepting = True
        self._condition = threading.Condition()
        self._dropped = {"frame_meta": 0, "metrics": 0, "heartbeat": 0}

    def _count(self) -> int:
        return len(self._items) + int(self._inflight is not None)

    @property
    def pending(self) -> int:
        with self._condition:
            return self._count()

    @property
    def dropped(self) -> dict[str, int]:
        with self._condition:
            return dict(self._dropped)

    @property
    def finished(self) -> bool:
        with self._condition:
            return not self._accepting and self._count() == 0

    def put(self, packet: _Packet, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if not self._accepting:
                    raise TransportClosed("Rabbit sink is closed")
                if self._count() < self.maxsize:
                    self._items.append(packet)
                    self._condition.notify_all()
                    return True

                # First evict the oldest queued frame, including when the new
                # arrival is another frame. An in-flight packet is never evicted.
                victim = next(
                    (
                        i
                        for i, item in enumerate(self._items)
                        if item.kind == "frame_meta"
                    ),
                    None,
                )
                if victim is None and packet.kind != "frame_meta":
                    victim = next(
                        (
                            i
                            for i, item in enumerate(self._items)
                            if item.kind not in _EVENT_KINDS
                        ),
                        None,
                    )
                if victim is not None:
                    self._dropped[self._items[victim].kind] += 1
                    del self._items[victim]
                    continue

                if packet.kind not in _EVENT_KINDS:
                    self._dropped[packet.kind] += 1
                    return False

                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Outbox full; event was not accepted")
                self._condition.wait(remaining)

    def take(self, timeout: float = 0.1) -> _Packet | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._inflight is None and not self._items:
                if not self._accepting:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._inflight is None:
                # Prioritize events while preserving event/update ordering.
                index = next(
                    (
                        i
                        for i, item in enumerate(self._items)
                        if item.kind in _EVENT_KINDS
                    ),
                    0,
                )
                self._inflight = self._items[index]
                del self._items[index]
            return self._inflight

    def complete(self, packet: _Packet, *, dropped: bool = False) -> None:
        with self._condition:
            if self._inflight is not packet:
                raise RuntimeError("Completing a packet that is not in flight")
            if dropped:
                if packet.kind in _EVENT_KINDS:
                    raise RuntimeError("Events cannot be dropped")
                self._dropped[packet.kind] += 1
            self._inflight = None
            self._condition.notify_all()

    def seal(self) -> None:
        with self._condition:
            self._accepting = False
            self._condition.notify_all()


def _parameters(url: str | None) -> pika.URLParameters:
    params = pika.URLParameters(
        url or os.environ.get("GUARD_TRANSPORT_URL") or DEFAULT_URL
    )
    params.connection_attempts = 1
    params.retry_delay = 0
    if params.heartbeat is None:
        params.heartbeat = 30
    if params.blocked_connection_timeout is None:
        params.blocked_connection_timeout = 5.0
    if params.socket_timeout is None:
        params.socket_timeout = 5.0
    if params.stack_timeout is None:
        params.stack_timeout = 10.0
    return params


def _validate_options(
    max_length: int, backoff_initial: float, backoff_max: float
) -> None:
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not (
        math.isfinite(backoff_initial)
        and math.isfinite(backoff_max)
        and 0 < backoff_initial <= backoff_max
    ):
        raise ValueError("Require finite 0 < backoff_initial <= backoff_max")


def _declare(channel: Any, max_length: int) -> None:
    channel.exchange_declare(
        exchange=EXCHANGE,
        exchange_type="direct",
        durable=False,
        auto_delete=False,
    )
    channel.queue_declare(
        queue=GATEWAY_QUEUE,
        durable=False,
        exclusive=False,
        auto_delete=False,
        arguments={
            "x-max-length": max_length,
            "x-overflow": "reject-publish",
        },
    )
    for routing_key in BINDINGS:
        channel.queue_bind(
            queue=GATEWAY_QUEUE,
            exchange=EXCHANGE,
            routing_key=routing_key,
        )


def _safe_close(connection: Any) -> None:
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def _connect(
    params: pika.URLParameters, max_length: int, *, publisher: bool, prefetch: int = DEFAULT_PREFETCH
) -> tuple[Any, Any]:
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        _declare(channel, max_length)
        if publisher:
            channel.confirm_delivery()
        else:
            channel.basic_qos(prefetch_count=prefetch)
        return connection, channel
    except Exception:
        _safe_close(connection)
        raise


class RabbitSink:
    def __init__(
        self,
        url: str | None = None,
        *,
        max_queue_size: int = 256,
        max_length: int = 2000,
        backoff_initial: float = 0.25,
        backoff_max: float = 5.0,
        auto_start: bool = True,
    ) -> None:
        _validate_options(max_length, backoff_initial, backoff_max)
        self._params = _parameters(url)
        self._outbox = _Outbox(max_queue_size)
        self._max_length = max_length
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._retry_wait = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if auto_start:
            self.start()

    @property
    def pending(self) -> int:
        return self._outbox.pending

    @property
    def dropped(self) -> dict[str, int]:
        return self._outbox.dropped

    def start(self) -> None:
        with self._state_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="guard-rabbit-sink",
                    daemon=True,
                )
                self._thread.start()

    def publish(
        self, message: MessageInput, *, timeout: float | None = None
    ) -> bool:
        """Serialize before accepting ownership; only the worker touches pika.

        With the default timeout, an event waits for capacity indefinitely.
        A timeout or TransportClosed means that the caller retains ownership.
        """
        return self._outbox.put(_Packet.make(message), timeout=timeout)

    def close(self, timeout: float | None = 5.0) -> None:
        """Seal and drain accepted messages.

        On timeout, raise and leave the worker retrying the retained messages.
        A subsequent close() can wait again. No implicit discard is performed.
        """
        self._outbox.seal()
        self.start()
        with self._state_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError(
                    f"Rabbit sink is still draining {self.pending} messages"
                )

    @staticmethod
    def _remaining_frame_ttl(packet: _Packet) -> float:
        return FRAME_TTL_SECONDS - (time.monotonic() - packet.enqueued_at)

    def _run(self) -> None:
        connection = None
        channel = None
        delay = self._backoff_initial
        try:
            while not self._outbox.finished:
                packet = self._outbox.take()
                try:
                    if packet is None:
                        if connection is not None:
                            connection.process_data_events(time_limit=0.05)
                        continue

                    if (
                        packet.kind == "frame_meta"
                        and self._remaining_frame_ttl(packet) <= 0
                    ):
                        self._outbox.complete(packet, dropped=True)
                        continue

                    if connection is None:
                        connection, channel = _connect(
                            self._params, self._max_length, publisher=True
                        )

                    expiration = None
                    if packet.kind == "frame_meta":
                        remaining = self._remaining_frame_ttl(packet)
                        if remaining <= 0:
                            self._outbox.complete(packet, dropped=True)
                            continue
                        expiration = str(max(1, math.ceil(remaining * 1000)))

                    properties = pika.BasicProperties(
                        content_type="application/octet-stream",
                        type=packet.kind,
                        headers=packet.headers,
                        message_id=packet.message_id,
                        delivery_mode=1,
                        expiration=expiration,
                    )
                    result = channel.basic_publish(
                        exchange=EXCHANGE,
                        routing_key=_ROUTING_KEYS[packet.kind],
                        body=packet.body,
                        properties=properties,
                        mandatory=True,
                    )
                    # BlockingChannel normally returns None on success and
                    # raises on a nack/return in confirmation mode.
                    if result is False:
                        raise RuntimeError("Broker rejected publication")
                    self._outbox.complete(packet)
                    delay = self._backoff_initial
                except Exception as exc:
                    # Do not include URL/credentials or message contents.
                    log.warning(
                        "Rabbit publisher reconnecting after %s",
                        type(exc).__name__,
                    )
                    _safe_close(connection)
                    connection = None
                    channel = None
                    if not self._outbox.finished:
                        self._retry_wait.wait(delay)
                        delay = min(delay * 2, self._backoff_max)
        finally:
            _safe_close(connection)


class RabbitSource:
    def __init__(
        self,
        callback: MessageCallback | None = None,
        *,
        url: str | None = None,
        max_length: int = 2000,
        backoff_initial: float = 0.25,
        backoff_max: float = 5.0,
        auto_start: bool = True,
        prefetch: int = DEFAULT_PREFETCH,
    ) -> None:
        _validate_options(max_length, backoff_initial, backoff_max)
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError("prefetch must be a positive integer")
        self._params = _parameters(url)
        self._prefetch = prefetch
        self._max_length = max_length
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._callback = callback
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if callback is not None and auto_start:
            self.start()

    def on_message(self, callback: MessageCallback) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._state_lock:
            if self._stop.is_set():
                raise TransportClosed("Rabbit source is closed")
            if self._thread is not None:
                raise RuntimeError("A callback consumer is already registered")
            self._callback = callback
        self.start()

    def start(self) -> None:
        with self._state_lock:
            if self._stop.is_set():
                raise TransportClosed("Rabbit source is closed")
            if self._callback is None:
                raise ValueError("Register a callback before starting")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="guard-rabbit-source",
                    daemon=True,
                )
                self._thread.start()

    def close(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        with self._state_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Rabbit consumer has not finished")

    def _run(self) -> None:
        delay = self._backoff_initial
        while not self._stop.is_set():
            connection = None
            try:
                connection, channel = _connect(
                    self._params, self._max_length, publisher=False, prefetch=self._prefetch
                )
                for method, properties, body in channel.consume(
                    queue=GATEWAY_QUEUE,
                    auto_ack=False,
                    inactivity_timeout=0.2,
                ):
                    if self._stop.is_set():
                        break
                    if method is None:
                        delay = self._backoff_initial
                        continue
                    try:
                        message = decode(body, properties.headers)
                    except (ValueError, TypeError, AttributeError):
                        log.warning("Rejecting malformed transport message")
                        channel.basic_reject(
                            delivery_tag=method.delivery_tag,
                            requeue=False,
                        )
                        continue
                    try:
                        assert self._callback is not None
                        self._callback(message)
                    except Exception:
                        channel.basic_nack(
                            delivery_tag=method.delivery_tag,
                            requeue=True,
                        )
                        # Reconnect with backoff to avoid a hot requeue loop.
                        raise
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                    delay = self._backoff_initial
                if not self._stop.is_set():
                    raise ConnectionError("Rabbit consumer was cancelled")
            except Exception as exc:
                if not self._stop.is_set():
                    log.warning(
                        "Rabbit consumer reconnecting after %s",
                        type(exc).__name__,
                    )
            finally:
                # Closing on the owning thread requeues unacknowledged items.
                _safe_close(connection)
            if not self._stop.is_set():
                self._stop.wait(delay)
                delay = min(delay * 2, self._backoff_max)
