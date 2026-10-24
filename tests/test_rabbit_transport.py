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
    def __init__(self):
        self.calls = []
        self.messages = defaultdict(deque)
        self.confirmed = None

    def __getattr__(self, name):
        if name in {"exchange_declare", "queue_declare", "queue_bind", "confirm_delivery", "basic_qos", "basic_ack", "basic_nack", "basic_reject"}:
            return lambda **kwargs: self.calls.append((name, kwargs))
        raise AttributeError(name)

    def basic_publish(self, **kwargs):
        self.calls.append(("basic_publish", kwargs))
        return self.confirmed

    def basic_get(self, *, queue, auto_ack):
        assert auto_ack is False
        if not self.messages[queue]:
            return None, None, None
        return SimpleNamespace(delivery_tag=42), None, self.messages[queue].popleft()


class Connection:
    def __init__(self):
        self.is_open = True
        self.ch = Channel()

    def channel(self):
        return self.ch

    def close(self):
        self.is_open = False

    def process_data_events(self, **kwargs):
        pass


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
