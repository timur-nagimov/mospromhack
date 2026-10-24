"""Канал сэмплов маховика edge → шлюз в AMQP topology v2 (docs/EDGE_PROTOCOL.md, docs/FLYWHEEL.md).

Сэмплы — не события: доставка без гарантии, как у телеметрии, но в своей очереди.
- Exchange тот же — guard.edge.v1. Очередь — guard.gateway.<edge_id>.flywheel.v1, ключ — <edge_id>.flywheel.
- Очередь durable только по метаданным (RabbitMQ 4.4 не принимает transient non-exclusive). Сообщения
  непостоянные (delivery_mode=1), TTL 120 с, не больше 256 частей и 4 МБ: без шлюза части истекают, а не копятся.
- В telemetry.v2 сэмплы не идут: там TTL 5 с и 32 сообщения, а шлюз через VPN читает ~2 сообщения/с.
  Часть сэмпла терялась бы почти всегда, и сэмпл не собирался бы.
- Свои соединения на edge и на шлюзе: опрос событий шлюзом не ждёт сэмплов, а publisher событий edge
  не делится потоком с отправителем сэмплов.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable

from guard.transport.protocol import Envelope, decode, encode
from guard.transport.rabbit import RabbitConfig, RabbitPublisher, _connection

log = logging.getLogger(__name__)

KIND = "flywheel_sample"
TTL_MS = 120_000
MAX_LENGTH = 256
MAX_BYTES = 4 * 1024 * 1024


def queue_name(config: RabbitConfig) -> str:
    return f"{config.queue_prefix}.{config.edge_id}.flywheel.v1"


def routing_key(config: RabbitConfig) -> str:
    return f"{config.edge_id}.flywheel"


def declare(channel: Any, config: RabbitConfig) -> None:
    """Обе стороны объявляют одинаково: отправка до старта шлюза не теряет маршрут."""
    channel.exchange_declare(exchange=config.exchange, exchange_type="direct", durable=True)
    channel.queue_declare(
        queue=queue_name(config), durable=True,
        arguments={"x-message-ttl": TTL_MS, "x-max-length": MAX_LENGTH, "x-max-length-bytes": MAX_BYTES},
    )
    channel.queue_bind(queue=queue_name(config), exchange=config.exchange, routing_key=routing_key(config))


class FlywheelPublisher:
    """Edge: публикация частей сэмпла. Один поток-владелец (EdgeSender). Успех — подтверждение брокера."""

    dedicated = True  # своё соединение: закрывает EdgeSender

    def __init__(self, config: RabbitConfig, connection_factory: Callable = _connection) -> None:
        self.config = config
        self._factory = connection_factory
        self._connection = None
        self._channel = None
        self.published = 0
        self.last_error: str | None = None

    def _connect(self) -> Any:
        if self._connection is None or not self._connection.is_open:
            self.close()
            self._connection = self._factory(self.config)
            self._channel = self._connection.channel()
            declare(self._channel, self.config)
            self._channel.confirm_delivery()
        return self._channel

    def publish(self, envelope: Envelope, *, durable: bool = False) -> None:
        import pika

        if durable or envelope.kind != KIND:
            raise ValueError("Only flywheel samples belong in the flywheel channel")
        data = encode(envelope)
        try:
            channel = self._connect()
            confirmed = channel.basic_publish(
                exchange=self.config.exchange,
                routing_key=routing_key(self.config),
                body=data,
                properties=pika.BasicProperties(
                    content_type="application/json", message_id=envelope.message_id,
                    delivery_mode=1, expiration=str(TTL_MS),
                ),
                mandatory=True,
            )
            if confirmed is False:
                raise ConnectionError("Broker did not confirm publication")
            self.published += 1
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.close()
            raise

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
        return {"connected": bool(self._connection and self._connection.is_open),
                "published": self.published, "last_error": self.last_error}


class FlywheelConsumer:
    """Шлюз: по одной части за basic_get. В полёте не больше одной части (~11 КБ) — канал событий не забит."""

    def __init__(self, config: RabbitConfig, connection_factory: Callable = _connection) -> None:
        self.config = config
        self._factory = connection_factory
        self._connection = None
        self._channel = None
        self.received = 0
        self.rejected = 0
        self.errors = 0
        self.last_error: str | None = None

    def _connect(self) -> Any:
        if self._connection is None or not self._connection.is_open:
            self.close()
            self._connection = self._factory(self.config)
            self._channel = self._connection.channel()
            declare(self._channel, self.config)
            self._channel.basic_qos(prefetch_count=1)
        return self._channel

    def poll(self, callback: Callable[[Envelope], Any]) -> int:
        channel = self._connect()
        method, _, body = channel.basic_get(queue=queue_name(self.config), auto_ack=False)
        if method is None:
            self._connection.process_data_events(time_limit=0)
            return 0
        try:
            envelope = decode(body, expected_edge_id=self.config.edge_id)
            if envelope.kind != KIND:
                raise ValueError("Message arrived in the wrong channel")
            callback(envelope)
        except ValueError:
            self.rejected += 1
            self.last_error = "InvalidMessage"
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as exc:  # noqa: BLE001 — сэмпл не стоит повторной доставки по кругу
            self.errors += 1
            self.last_error = type(exc).__name__
            if self.errors in (1, 10, 100):
                log.exception("flywheel: sample part handling failed")
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        else:
            channel.basic_ack(delivery_tag=method.delivery_tag)
            self.received += 1
            self.last_error = None
        return 1

    def run(self, callback: Callable[[Envelope], Any], stop: threading.Event) -> None:
        backoff = 0.5
        while not stop.is_set():
            try:
                if not self.poll(callback):
                    stop.wait(0.2)
                backoff = 0.5
            except Exception as exc:  # noqa: BLE001 — брокер недоступен: переподключение с паузой
                self.last_error = type(exc).__name__
                self.close()
                stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
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
        return {"connected": bool(self._connection and self._connection.is_open), "received": self.received,
                "rejected": self.rejected, "errors": self.errors, "last_error": self.last_error}


class MemoryFlywheelPublisher:
    """Отладка без брокера (транспорт memory или log): последние части в памяти, только счётчики."""

    dedicated = True

    def __init__(self, maxlen: int = 64) -> None:
        self.messages: deque[bytes] = deque(maxlen=maxlen)
        self.published = 0

    def publish(self, envelope: Envelope, *, durable: bool = False) -> None:
        if durable or envelope.kind != KIND:
            raise ValueError("Only flywheel samples belong in the flywheel channel")
        self.messages.append(encode(envelope))
        self.published += 1

    def close(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {"connected": True, "published": self.published}


def edge_publisher(engine: Any) -> Any:
    """Канал сэмплов edge. Вызывается в потоке отправителя при первой части: EdgeRunner к этому времени есть.

    Настоящий RabbitPublisher телеметрии → своё соединение с той же конфигурацией. Иначе (FakeBroker в тестах,
    память и лог для отладки) — тот же объект телеметрии: его publish(envelope, durable=False) ничего не шлёт в сеть.
    """
    runner = getattr(engine, "replication", None)
    telemetry = getattr(runner, "telemetry_publisher", None)
    if isinstance(telemetry, RabbitPublisher):
        return FlywheelPublisher(telemetry.config)
    if telemetry is not None and callable(getattr(telemetry, "publish", None)):
        return telemetry
    return MemoryFlywheelPublisher()


class FlywheelReceiver:
    """Шлюз: поток приёма частей из своей очереди → callback (SampleAssembler.add_envelope)."""

    def __init__(self, consumer: Any) -> None:
        self.consumer = consumer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, callback: Callable[[Envelope], Any]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.consumer.run, args=(callback, self._stop),
                                        name="gateway-flywheel", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> bool:
        """True — поток остановлен. Поток не держит SQLite: зависший можно оставить (daemon)."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        if thread.is_alive():
            log.warning("flywheel: gateway sample receiver did not stop in %.1f s", timeout)
            return False
        self._thread = None
        return True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        status = getattr(self.consumer, "status", None)
        return {"running": self.running, **(status() if callable(status) else {})}
