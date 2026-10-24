"""Управляющий канал gateway → edge на той же топологии guard.edge.v1 (docs/EDGE_PROTOCOL.md).

Команды camera_config идут в durable-очередь edge ``guard.edge.<edge_id>.config``, подтверждения
camera_config_ack — в durable-очередь шлюза ``guard.gateway.<edge_id>.config_ack``. Сообщения —
тот же конверт v1, persistent, с publisher confirm. Каждой стороной владеет один поток: входящее
сообщение подтверждается брокеру только после обработчика и публикации его ответа, поэтому сбой
между ними приводит к повторной доставке, а не к потере. Повтор безопасен: edge применяет версию
камеры один раз и на повтор отвечает прежним результатом.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable, Iterable

from guard.transport.protocol import Envelope, decode
from guard.transport.rabbit import RabbitConfig, _connection

log = logging.getLogger(__name__)

Wire = tuple[str, bytes]  # (message_id, закодированный конверт)


def _sequence(message_id: str) -> int:
    try:
        return int(message_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def control_topology(channel: Any, config: RabbitConfig) -> None:
    """Объявляют обе стороны: команда, отправленная до старта edge, ждёт его в durable-очереди."""
    channel.exchange_declare(exchange=config.exchange_name, exchange_type="direct", durable=True)
    for queue, ack in ((config.config_queue, False), (config.config_ack_queue, True)):
        channel.queue_declare(queue=queue, durable=True)
        channel.queue_bind(queue=queue, exchange=config.exchange_name, routing_key=config.config_routing_key(ack))


class RabbitControlLink:
    """Одна сторона канала: edge читает команды и отвечает, шлюз шлёт команды и читает ответы.

    handler(envelope) возвращает ответы для публикации до ACK входящего. on_connect() при каждом
    (пере)подключении возвращает сообщения для повторной отправки (у шлюза — неподтверждённые команды).
    """

    def __init__(
        self, config: RabbitConfig, *, role: str, handler: Callable[[Envelope], Iterable[Wire] | None],
        on_connect: Callable[[], Iterable[Wire]] | None = None, connection_factory: Callable = _connection,
        poll_s: float = 0.1,
    ) -> None:
        if role not in ("edge", "gateway"):
            raise ValueError("role must be edge or gateway")
        self.config = config
        self.role = role
        self.queue = config.config_queue if role == "edge" else config.config_ack_queue
        self.routing_key = config.config_routing_key(ack=role == "edge")
        self.accepts = "camera_config" if role == "edge" else "camera_config_ack"
        self._handler = handler
        self._on_connect = on_connect
        self._factory = connection_factory
        self.poll_s = poll_s
        self._connection = None
        self._channel = None
        self._outgoing: deque[Wire] = deque()
        self._out_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.received = self.published = self.rejected = self.connects = 0
        self.last_error: str | None = None

    # ---- отправка ----

    def send(self, message_id: str, wire: bytes) -> None:
        """Из любого потока: сообщение уйдёт из потока канала (pika не потокобезопасна)."""
        with self._out_lock:
            self._outgoing.append((message_id, wire))

    def _publish(self, channel: Any, message_id: str, wire: bytes) -> None:
        import pika

        confirmed = channel.basic_publish(
            exchange=self.config.exchange_name, routing_key=self.routing_key, body=wire,
            properties=pika.BasicProperties(content_type="application/json", message_id=message_id, delivery_mode=2),
            mandatory=True,
        )
        if confirmed is False:
            raise ConnectionError("Broker did not confirm publication")
        self.published += 1

    def _flush(self, channel: Any) -> None:
        while True:
            with self._out_lock:
                if not self._outgoing:
                    return
                message_id, wire = self._outgoing[0]
            self._publish(channel, message_id, wire)
            with self._out_lock:
                if self._outgoing and self._outgoing[0][0] == message_id:
                    self._outgoing.popleft()

    # ---- приём ----

    def _connect(self) -> Any:
        if self._connection is None or not self._connection.is_open:
            self.close()
            connection = self._factory(self.config)
            try:
                channel = connection.channel()
                control_topology(channel, self.config)
                channel.confirm_delivery()
                channel.basic_qos(prefetch_count=1)
            except BaseException:
                try:
                    connection.close()
                except Exception:  # noqa: BLE001
                    pass
                raise
            self._connection, self._channel = connection, channel
            self.connects += 1
            resend = list(self._on_connect() or ()) if self._on_connect is not None else []
            if resend:
                with self._out_lock:  # без дублей и в порядке выдачи (sequence из message_id)
                    merged = dict(self._outgoing)
                    merged.update(resend)
                    self._outgoing = deque(sorted(merged.items(), key=lambda item: _sequence(item[0])))
                log.info("control link (%s): resending %d unconfirmed message(s)", self.role, len(resend))
        return self._channel

    def poll_once(self) -> int:
        """Отправить накопленное и обработать не больше одного входящего. Ошибка канала — исключение."""
        channel = self._connect()
        self._flush(channel)
        method, _, body = channel.basic_get(queue=self.queue, auto_ack=False)
        if method is None:
            self._connection.process_data_events(time_limit=0)
            return 0
        try:
            envelope = decode(body, expected_edge_id=self.config.edge_id)
            if envelope.kind != self.accepts:
                raise ValueError("Message arrived in the wrong control queue")
        except ValueError:
            self.rejected += 1
            self.last_error = "InvalidMessage"
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return 1
        try:
            for message_id, wire in self._handler(envelope) or ():
                self._publish(channel, message_id, wire)
        except Exception:
            try:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            except Exception:  # noqa: BLE001 — при закрытии соединения брокер вернёт сообщение сам
                pass
            raise
        channel.basic_ack(delivery_tag=method.delivery_tag)
        self.received += 1
        self.last_error = None
        return 1

    def run(self, stop: threading.Event) -> None:
        backoff = 0.5
        while not stop.is_set():
            try:
                received = self.poll_once()
                backoff = 0.5
                if not received:
                    stop.wait(self.poll_s)
            except Exception as exc:  # noqa: BLE001 — переподключение с ограниченной задержкой
                self.last_error = type(exc).__name__
                log.warning("control link (%s) reconnecting after %s", self.role, type(exc).__name__)
                self.close()
                stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
        self.close()

    # ---- жизненный цикл ----

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self.run, args=(self._stop,), name=f"control-{self.role}", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 6.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("Control link did not stop")
        if thread is None:
            self.close()

    def close(self) -> None:
        connection, self._connection = self._connection, None
        self._channel = None
        if connection is not None:
            try:
                if connection.is_open:
                    connection.close()
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> dict[str, Any]:
        with self._out_lock:
            outgoing = len(self._outgoing)
        return {"connected": bool(self._connection and self._connection.is_open), "queue": self.queue,
                "received": self.received, "published": self.published, "rejected": self.rejected,
                "connects": self.connects, "outgoing": outgoing, "last_error": self.last_error}
