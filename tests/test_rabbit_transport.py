from collections import defaultdict, deque
from types import SimpleNamespace
from uuid import uuid4

import pytest

from guard.contracts import Event
from guard.transport.fake import FakeBroker
from guard.transport.protocol import Envelope, event_payload
from guard.transport.rabbit import RabbitConfig, RabbitConsumer, RabbitPublisher


def event_message(sequence=1):
    producer = str(uuid4())
    return Envelope(
        edge_id="demo", producer_id=producer, run_id=str(uuid4()),
        message_id=f"{producer}:{sequence}", sequence=sequence, sent_at=100,
        kind="event", payload=event_payload(Event(
            id="event", camera_id="camera", site="factory", rule_id="zone", type="zone_intrusion",
            severity="high", started_at=98, confirmed_at=100, bbox=(0, 0, 10, 20),
        )),
    )


class Channel:
    """Broker push model: basic_consume with per-consumer prefetch, deliveries in process_data_events."""

    def __init__(self):
        self.calls = []
        self.messages = defaultdict(deque)
        self.confirmed = None
        self.prefetch = 0
        self.consumers = []
        self.unacked = {}
        self.max_inflight = defaultdict(int)
        self.next_tag = 41

    def __getattr__(self, name):
        if name in {"exchange_declare", "queue_declare", "queue_bind", "confirm_delivery"}:
            return lambda **kwargs: self.calls.append((name, kwargs))
        raise AttributeError(name)

    def basic_qos(self, **kwargs):
        self.calls.append(("basic_qos", kwargs))
        self.prefetch = kwargs["prefetch_count"]

    def basic_consume(self, *, queue, on_message_callback, auto_ack=False):
        self.calls.append(("basic_consume", {"queue": queue, "auto_ack": auto_ack}))
        self.consumers.append((queue, on_message_callback, self.prefetch))

    def _settle(self, name, kwargs):
        self.calls.append((name, kwargs))
        self.unacked.pop(kwargs["delivery_tag"])

    def basic_ack(self, **kwargs):
        self._settle("basic_ack", kwargs)

    def basic_nack(self, **kwargs):
        self._settle("basic_nack", kwargs)

    def basic_reject(self, **kwargs):
        self._settle("basic_reject", kwargs)

    def basic_publish(self, **kwargs):
        self.calls.append(("basic_publish", kwargs))
        return self.confirmed

    def basic_get(self, *, queue, auto_ack):
        raise AssertionError("basic_get costs a VPN round trip per message")

    def deliver(self, hold_acks=False):
        progressed = True
        while progressed:
            progressed = False
            for queue, callback, prefetch in self.consumers:
                inflight = sum(1 for owner in self.unacked.values() if owner == queue)
                if self.messages[queue] and inflight < prefetch:
                    self.next_tag += 1
                    self.unacked[self.next_tag] = queue
                    self.max_inflight[queue] = max(self.max_inflight[queue], inflight + 1)
                    callback(self, SimpleNamespace(delivery_tag=self.next_tag), None, self.messages[queue].popleft())
                    progressed = True


class Connection:
    def __init__(self):
        self.is_open = True
        self.ch = Channel()
        self.waits = []

    def channel(self):
        return self.ch

    def close(self):
        self.is_open = False

    def process_data_events(self, time_limit=0):
        self.waits.append(time_limit)
        self.ch.deliver()


def test_event_publisher_confirms_persistence_and_separates_tpu_queues():
    config = RabbitConfig("amqp://user:secret@localhost/", "demo")
    connection = Connection()
    sender = RabbitPublisher(config, lambda cfg: connection)
    message = event_message()
    sender.publish(message)
    names = [name for name, _ in connection.ch.calls]
    assert names.index("confirm_delivery") < names.index("basic_publish")
    publish = next(args for name, args in connection.ch.calls if name == "basic_publish")
    assert publish["mandatory"] is True
    assert publish["properties"].delivery_mode == 2
    assert publish["properties"].message_id == message.message_id
    assert publish["exchange"] == "guard.edge.v1"
    event_queue = next(args for name, args in connection.ch.calls if name == "queue_declare" and args["queue"] == config.event_queue)
    assert event_queue["durable"] is True
    assert "secret" not in repr(config)
    assert sender.status()["published"] == 1


def test_failed_confirmation_does_not_count_as_published():
    connection = Connection()
    connection.ch.confirmed = False
    sender = RabbitPublisher(RabbitConfig("amqp://localhost/", "demo"), lambda cfg: connection)
    with pytest.raises(ConnectionError):
        sender.publish(event_message())
    assert sender.status() == {"connected": False, "published": 0, "last_error": "ConnectionError"}
    assert not connection.is_open


def test_telemetry_is_bounded_and_expiring():
    config = RabbitConfig("amqp://localhost/", "demo", telemetry_ttl_ms=1234, telemetry_limit=3)
    connection = Connection()
    sender = RabbitPublisher(config, lambda cfg: connection)
    event = event_message()
    message = event.model_copy(update={"kind": "metrics", "payload": {"data": {"backend": "test"}}})
    sender.publish(message, durable=False)
    queue = next(args for name, args in connection.ch.calls if name == "queue_declare" and args["queue"] == config.telemetry_queue)
    assert queue["arguments"] == {"x-message-ttl": 1234, "x-max-length": 3}
    assert queue["durable"] is True
    assert config.telemetry_queue == "guard.gateway.demo.telemetry.v2"
    assert not any(args.get("queue") == "guard.gateway.demo.telemetry" for _, args in connection.ch.calls)
    publish = next(args for name, args in connection.ch.calls if name == "basic_publish")
    assert publish["properties"].expiration == "1234"
    assert publish["properties"].delivery_mode == 1
    with pytest.raises(ValueError):
        sender.publish(event, durable=False)


def test_consumer_acks_only_after_successful_callback():
    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    message = event_message()
    from guard.transport.protocol import encode
    connection.ch.messages[config.event_queue].append(encode(message))
    consumer = RabbitConsumer(config, lambda cfg: connection)

    def commit_before_ack(envelope):
        assert envelope.message_id == message.message_id
        assert not any(name == "basic_ack" for name, _ in connection.ch.calls)

    assert consumer.poll(commit_before_ack) == 1
    assert any(name == "basic_ack" for name, _ in connection.ch.calls)
    assert consumer.status()["received"] == 1


def test_transient_apply_failure_requeues_without_ack():
    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    from guard.transport.protocol import encode
    connection.ch.messages[config.event_queue].append(encode(event_message()))
    consumer = RabbitConsumer(config, lambda cfg: connection)

    def disk_failure(envelope):
        raise OSError("disk full")

    with pytest.raises(OSError):
        consumer.poll(disk_failure)
    assert ("basic_nack", {"delivery_tag": 42, "requeue": True}) in connection.ch.calls
    assert not any(name == "basic_ack" for name, _ in connection.ch.calls)


def test_invalid_message_is_rejected_without_requeue():
    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    connection.ch.messages[config.event_queue].append(b'{"not": "an envelope"}')
    consumer = RabbitConsumer(config, lambda cfg: connection)
    consumer.poll(lambda env: pytest.fail("Malformed message reached handler"))
    assert ("basic_reject", {"delivery_tag": 42, "requeue": False}) in connection.ch.calls
    assert consumer.status()["rejected"] == 1


def telemetry_message(sequence, camera="cam_s1"):
    return event_message(sequence).model_copy(update={"kind": "metrics", "payload": {"data": {"camera": camera}}})


def test_consumer_subscribes_with_bounded_prefetch_instead_of_polling():
    # Through the VPN (RTT 0.3-0.4 s) basic_get delivered ~1 message/s for both
    # queues while the edge sends ~5 telemetry messages/s: previews of all but
    # one camera expired in the broker (metrics tpu-demo-baseline-queue-0130).
    from guard.transport import rabbit

    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    consumer = RabbitConsumer(config, lambda cfg: connection)
    assert consumer.poll(lambda env: None) == 0
    calls = [(name, args) for name, args in connection.ch.calls if name in {"basic_qos", "basic_consume"}]
    assert calls == [
        ("basic_qos", {"prefetch_count": rabbit.EVENT_PREFETCH}),
        ("basic_consume", {"queue": config.event_queue, "auto_ack": False}),
        ("basic_qos", {"prefetch_count": rabbit.TELEMETRY_PREFETCH}),
        ("basic_consume", {"queue": config.telemetry_queue, "auto_ack": False}),
    ]
    # One preview per camera fits into one round trip; durable events with
    # snapshots (~60 KB) keep a small window so telemetry is not starved.
    assert rabbit.TELEMETRY_PREFETCH >= 4
    assert 1 <= rabbit.EVENT_PREFETCH < rabbit.TELEMETRY_PREFETCH
    # Re-polling an open connection must not subscribe again.
    consumer.poll(lambda env: None)
    assert sum(name == "basic_consume" for name, _ in connection.ch.calls) == 2


def test_one_poll_handles_every_ready_message_from_both_queues():
    from guard.transport.protocol import encode

    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    for sequence in range(1, 7):
        connection.ch.messages[config.telemetry_queue].append(encode(telemetry_message(sequence, f"cam_s{sequence % 4 + 1}")))
    connection.ch.messages[config.event_queue].append(encode(event_message()))
    consumer = RabbitConsumer(config, lambda cfg: connection)
    seen = []
    assert consumer.poll(seen.append, time_limit=0.1) == 7
    assert connection.waits == [0.1]
    assert sorted(envelope.kind for envelope in seen) == ["event"] + ["metrics"] * 6
    assert consumer.status()["received"] == 7
    assert sum(name == "basic_ack" for name, _ in connection.ch.calls) == 7
    assert connection.ch.max_inflight[config.telemetry_queue] <= connection.ch.consumers[1][2]


def test_event_in_telemetry_queue_is_rejected():
    from guard.transport.protocol import encode

    config = RabbitConfig("amqp://localhost/", "demo")
    connection = Connection()
    connection.ch.messages[config.telemetry_queue].append(encode(event_message()))
    consumer = RabbitConsumer(config, lambda cfg: connection)
    assert consumer.poll(lambda env: pytest.fail("Event accepted from the telemetry channel")) == 1
    assert ("basic_reject", {"delivery_tag": 42, "requeue": False}) in connection.ch.calls
    assert consumer.status()["rejected"] == 1


def test_run_reconnects_and_resubscribes_after_connection_failure():
    import threading

    from guard.transport.protocol import encode

    config = RabbitConfig("amqp://localhost/", "demo")
    first, second = Connection(), Connection()
    connections = [first, second]
    stop = threading.Event()

    def broken(time_limit=0):
        raise ConnectionError("tunnel dropped")

    first.process_data_events = broken
    second.ch.messages[config.event_queue].append(encode(event_message()))
    seen = []

    def handle(envelope):
        seen.append(envelope)
        stop.set()

    consumer = RabbitConsumer(config, lambda cfg: connections.pop(0))
    thread = threading.Thread(target=consumer.run, args=(handle, stop), daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert [envelope.kind for envelope in seen] == ["event"]
    assert not first.is_open
    assert sum(name == "basic_consume" for name, _ in second.ch.calls) == 2


def test_fake_uncertain_publish_and_lost_ack_are_replayable():
    broker = FakeBroker()
    message = event_message()
    broker.lose_confirm = True
    with pytest.raises(ConnectionError):
        broker.publish(message)
    broker.publish(message)
    assert len(broker.events) == 2
    received = []
    broker.deliver(received.append, duplicate=True)
    broker.deliver(received.append)
    assert [item.message_id for item in received] == [message.message_id] * 2
    broker.online = False
    with pytest.raises(ConnectionError):
        broker.publish(message)
