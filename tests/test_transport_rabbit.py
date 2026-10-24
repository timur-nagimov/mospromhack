import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from guard.contracts import Event, Severity
from guard.transport import rabbit
from guard.transport.codec import decode, encode
from guard.transport.messages import (
    EventMsg,
    FrameMetaMsg,
    HeartbeatMsg,
    MetricsMsg,
    TransportClosed,
)
from guard.transport.rabbit import DEFAULT_PREFETCH, RabbitSink, RabbitSource, _Outbox, _Packet


def event_message(event_id="event-1", kind="event"):
    return EventMsg(
        kind=kind,
        event=Event(
            id=event_id,
            camera_id="cam-1",
            site="site-1",
            rule_id="rule-1",
            type="ppe_missing",
            severity=Severity.high,
            started_at=100.0,
            confirmed_at=102.0,
            bbox=(10, 20, 100, 200),
        ),
        snapshot_jpeg=b"\xff\xd8\x80\xff\xd9",
    )


def frame_message(index=1):
    return FrameMetaMsg(
        camera_id="cam-1",
        frame_idx=index,
        ts=100.0,
        w=1920,
        h=1080,
        people=[[1, 10, 20, 100, 200, 0]],
    )


def mock_connection():
    connection = Mock(name="connection")
    channel = Mock(name="channel")
    connection.channel.return_value = channel
    channel.basic_publish.return_value = None
    return connection, channel


def assert_topology(channel, max_length=2000):
    channel.exchange_declare.assert_called_once_with(
        exchange="zazerkalye",
        exchange_type="direct",
        durable=False,
        auto_delete=False,
    )
    channel.queue_declare.assert_called_once_with(
        queue="zazerkalye.gateway",
        durable=False,
        exclusive=False,
        auto_delete=False,
        arguments={
            "x-max-length": max_length,
            "x-overflow": "reject-publish",
        },
    )
    assert channel.queue_bind.call_args_list == [
        call(
            queue="zazerkalye.gateway",
            exchange="zazerkalye",
            routing_key=key,
        )
        for key in ("edge.event", "edge.frame_meta", "edge.metrics")
    ]


def deliveries(*messages):
    for tag, message in enumerate(messages, 1):
        body, headers = encode(message)
        yield (
            SimpleNamespace(delivery_tag=tag),
            SimpleNamespace(headers=headers),
            body,
        )
    # Simulate pika's periodic inactivity yields without spinning.
    idle = threading.Event()
    while True:
        idle.wait(0.005)
        yield None, None, None


def test_outbox_evicts_frames_before_metrics_and_never_events():
    outbox = _Outbox(2)
    frame = _Packet.make(frame_message())
    metrics = _Packet.make(MetricsMsg(data={}))
    first = _Packet.make(event_message())
    second = _Packet.make(event_message("event-2", "event_update"))

    assert outbox.put(frame)
    assert outbox.put(metrics)
    assert outbox.put(first)
    assert outbox.dropped == {"frame_meta": 1, "metrics": 0, "heartbeat": 0}
    assert outbox.put(second)
    assert outbox.dropped == {"frame_meta": 1, "metrics": 1, "heartbeat": 0}
    assert not outbox.put(_Packet.make(frame_message(2)))
    assert not outbox.put(_Packet.make(HeartbeatMsg(ts=1)))
    with pytest.raises(TimeoutError):
        outbox.put(_Packet.make(event_message("event-3")), timeout=0)
    assert outbox.pending == 2

    assert outbox.take(0) is first
    # A retry gets exactly the same packet, without freeing another slot.
    assert outbox.take(0) is first
    assert outbox.pending == 2
    outbox.complete(first)
    assert outbox.take(0) is second
    outbox.complete(second)
    outbox.seal()
    assert outbox.finished


def test_new_frame_replaces_oldest_queued_frame():
    outbox = _Outbox(1)
    old = _Packet.make(frame_message(1))
    new = _Packet.make(frame_message(2))
    outbox.put(old)
    assert outbox.put(new)
    assert outbox.take(0) is new
    assert outbox.dropped["frame_meta"] == 1
    outbox.complete(new)


def test_event_waits_until_inflight_slot_is_released():
    outbox = _Outbox(1)
    first = _Packet.make(event_message("first"))
    second = _Packet.make(event_message("second"))
    outbox.put(first)
    assert outbox.take(0) is first
    entered = threading.Event()
    accepted = threading.Event()
    failures = []

    def publish_second():
        entered.set()
        try:
            outbox.put(second, timeout=3)
            accepted.set()
        except Exception as exc:
            failures.append(exc)

    producer = threading.Thread(target=publish_second)
    producer.start()
    try:
        assert entered.wait(3)
        assert not accepted.wait(0.03)
        assert outbox.pending == 1
        outbox.complete(first)
        assert accepted.wait(3)
        assert failures == []
        assert outbox.take(0) is second
        outbox.complete(second)
    finally:
        outbox.seal()
        producer.join(timeout=3)
    assert not producer.is_alive()


def test_seal_wakes_blocked_event_producer_without_discarding_accepted_event():
    outbox = _Outbox(1)
    first = _Packet.make(event_message("first"))
    outbox.put(first)
    entered = threading.Event()
    errors = []

    def publish_second():
        entered.set()
        try:
            outbox.put(_Packet.make(event_message("second")))
        except Exception as exc:
            errors.append(exc)

    producer = threading.Thread(target=publish_second)
    producer.start()
    assert entered.wait(3)
    outbox.seal()
    producer.join(timeout=3)
    assert not producer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TransportClosed)
    assert outbox.take(0) is first
    outbox.complete(first)
    assert outbox.finished


def test_sink_topology_routing_ttl_and_worker_thread(monkeypatch):
    connection, channel = mock_connection()
    factory = Mock(return_value=connection)
    monkeypatch.setattr(rabbit.pika, "BlockingConnection", factory)
    monkeypatch.setenv(
        "GUARD_TRANSPORT_URL", "amqp://alice:secret@broker.example:5679/demo"
    )
    worker_threads = []

    def published(**_kwargs):
        worker_threads.append(threading.get_ident())

    channel.basic_publish.side_effect = published
    messages = [
        event_message(),
        event_message(kind="event_update"),
        frame_message(),
        MetricsMsg(data={"fps": 5}),
        HeartbeatMsg(ts=123),
    ]
    sink = RabbitSink(auto_start=False, max_length=42)
    for message in messages:
        assert sink.publish(message)
    sink.start()
    sink.close(timeout=3)

    assert sink.pending == 0
    factory.assert_called_once()
    params = factory.call_args.args[0]
    assert params.host == "broker.example"
    assert params.port == 5679
    assert params.virtual_host == "demo"
    assert_topology(channel, max_length=42)
    channel.confirm_delivery.assert_called_once_with()
    assert len(channel.basic_publish.call_args_list) == len(messages)
    publications = {
        item.kwargs["properties"].type: item.kwargs
        for item in channel.basic_publish.call_args_list
    }
    assert {
        kind: publication["routing_key"]
        for kind, publication in publications.items()
    } == {
        "event": "edge.event",
        "event_update": "edge.event",
        "frame_meta": "edge.frame_meta",
        "metrics": "edge.metrics",
        "heartbeat": "edge.metrics",
    }
    for message in messages:
        publication = publications[message.kind]
        properties = publication["properties"]
        assert publication["exchange"] == "zazerkalye"
        assert publication["mandatory"] is True
        assert properties.message_id
        assert decode(publication["body"], properties.headers) == message
        if message.kind == "frame_meta":
            assert 0 < int(properties.expiration) <= 10000
        else:
            assert properties.expiration is None
    assert len(set(worker_threads)) == 1
    assert worker_threads[0] != threading.get_ident()
    connection.close.assert_called_once_with()
    with pytest.raises(TransportClosed):
        sink.publish(event_message())


def test_sink_reconnects_and_retries_same_packet_with_backoff(monkeypatch):
    first_connection, first_channel = mock_connection()
    second_connection, second_channel = mock_connection()
    first_channel.basic_publish.side_effect = ConnectionError("lost confirm")
    factory = Mock(
        side_effect=[
            ConnectionError("offline"),
            first_connection,
            second_connection,
        ]
    )
    monkeypatch.setattr(rabbit.pika, "BlockingConnection", factory)
    sink = RabbitSink(
        auto_start=False,
        max_queue_size=1,
        backoff_initial=0.01,
        backoff_max=0.04,
    )
    sink._retry_wait = Mock()
    sink.publish(event_message())
    sink.start()
    sink.close(timeout=3)

    assert factory.call_count == 3
    assert sink._retry_wait.wait.call_args_list == [call(0.01), call(0.02)]
    assert sink.pending == 0
    assert_topology(first_channel)
    assert_topology(second_channel)
    first_publication = first_channel.basic_publish.call_args.kwargs
    retry = second_channel.basic_publish.call_args.kwargs
    assert first_publication["body"] == retry["body"]
    assert (
        first_publication["properties"].message_id
        == retry["properties"].message_id
    )
    first_connection.close.assert_called_once_with()
    second_connection.close.assert_called_once_with()


def test_sink_retries_broker_rejection(monkeypatch):
    first_connection, first_channel = mock_connection()
    second_connection, second_channel = mock_connection()
    first_channel.basic_publish.return_value = False
    monkeypatch.setattr(
        rabbit.pika,
        "BlockingConnection",
        Mock(side_effect=[first_connection, second_connection]),
    )
    sink = RabbitSink(backoff_initial=0.001, backoff_max=0.002)
    sink.publish(event_message())
    sink.close(timeout=3)
    assert sink.pending == 0
    assert first_channel.basic_publish.call_count == 1
    assert second_channel.basic_publish.call_count == 1


def test_stale_local_frames_expire_without_connecting(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(rabbit.pika, "BlockingConnection", factory)
    sink = RabbitSink(auto_start=False)
    packet = replace(
        _Packet.make(frame_message()),
        enqueued_at=time.monotonic() - 11,
    )
    sink._outbox.put(packet)
    sink.close(timeout=3)
    factory.assert_not_called()
    assert sink.pending == 0
    assert sink.dropped["frame_meta"] == 1


def test_source_reconnects_declares_and_acknowledges_after_callback(monkeypatch):
    first_connection, first_channel = mock_connection()
    second_connection, second_channel = mock_connection()
    first_channel.consume.side_effect = ConnectionError("stream interrupted")
    message = event_message()
    second_channel.consume.return_value = deliveries(message)
    factory = Mock(
        side_effect=[
            ConnectionError("offline"),
            first_connection,
            second_connection,
        ]
    )
    monkeypatch.setattr(rabbit.pika, "BlockingConnection", factory)
    acknowledged = threading.Event()
    second_channel.basic_ack.side_effect = lambda **_kwargs: acknowledged.set()
    received = []
    callback_threads = []

    def callback(item):
        assert second_channel.basic_ack.call_count == 0
        received.append(item)
        callback_threads.append(threading.get_ident())

    source = RabbitSource(backoff_initial=0.001, backoff_max=0.004)
    source.on_message(callback)
    try:
        assert acknowledged.wait(3)
    finally:
        source.close(timeout=3)

    assert factory.call_count == 3
    assert received == [message]
    assert callback_threads[0] != threading.get_ident()
    assert_topology(first_channel)
    assert_topology(second_channel)
    second_channel.basic_qos.assert_called_once_with(prefetch_count=DEFAULT_PREFETCH)
    second_channel.consume.assert_called_once_with(
        queue="zazerkalye.gateway",
        auto_ack=False,
        inactivity_timeout=0.2,
    )
    second_channel.basic_ack.assert_called_once_with(delivery_tag=1)
    second_channel.basic_nack.assert_not_called()
    first_connection.close.assert_called_once_with()
    second_connection.close.assert_called_once_with()


def test_callback_failure_requeues_then_reconnects(monkeypatch):
    first_connection, first_channel = mock_connection()
    second_connection, second_channel = mock_connection()
    message = event_message()
    first_channel.consume.return_value = deliveries(message)
    second_channel.consume.return_value = deliveries(message)
    monkeypatch.setattr(
        rabbit.pika,
        "BlockingConnection",
        Mock(side_effect=[first_connection, second_connection]),
    )
    acknowledged = threading.Event()
    second_channel.basic_ack.side_effect = lambda **_kwargs: acknowledged.set()
    received = []

    def callback(item):
        received.append(item)
        if len(received) == 1:
            raise RuntimeError("journal temporarily unavailable")

    source = RabbitSource(
        callback,
        backoff_initial=0.001,
        backoff_max=0.004,
    )
    try:
        assert acknowledged.wait(3)
    finally:
        source.close(timeout=3)

    assert received == [message, message]
    first_channel.basic_ack.assert_not_called()
    first_channel.basic_nack.assert_called_once_with(
        delivery_tag=1, requeue=True
    )
    second_channel.basic_ack.assert_called_once_with(delivery_tag=1)


def test_source_rejects_malformed_message_and_continues(monkeypatch):
    connection, channel = mock_connection()
    monkeypatch.setattr(
        rabbit.pika, "BlockingConnection", Mock(return_value=connection)
    )
    message = MetricsMsg(data={"fps": 5})

    def stream():
        yield (
            SimpleNamespace(delivery_tag=99),
            SimpleNamespace(headers={}),
            b"broken",
        )
        yield from deliveries(message)

    channel.consume.return_value = stream()
    acknowledged = threading.Event()
    channel.basic_ack.side_effect = lambda **_kwargs: acknowledged.set()
    received = []
    source = RabbitSource(received.append)
    try:
        assert acknowledged.wait(3)
    finally:
        source.close(timeout=3)

    assert received == [message]
    channel.basic_reject.assert_called_once_with(
        delivery_tag=99, requeue=False
    )
    channel.basic_ack.assert_called_once_with(delivery_tag=1)


def test_close_timeout_keeps_unconfirmed_event_for_later_drain(monkeypatch):
    connection, channel = mock_connection()
    entered = threading.Event()
    release = threading.Event()

    def publish(**_kwargs):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test publication was not released")

    channel.basic_publish.side_effect = publish
    monkeypatch.setattr(
        rabbit.pika, "BlockingConnection", Mock(return_value=connection)
    )
    sink = RabbitSink()
    sink.publish(event_message())
    try:
        assert entered.wait(3)
        with pytest.raises(TimeoutError, match="still draining"):
            sink.close(timeout=0)
        assert sink.pending == 1
        with pytest.raises(TransportClosed):
            sink.publish(event_message("another"))
    finally:
        release.set()
        sink.close(timeout=3)
    assert sink.pending == 0


def test_default_url_and_idle_source_shutdown(monkeypatch):
    monkeypatch.delenv("GUARD_TRANSPORT_URL", raising=False)
    connection, channel = mock_connection()
    channel.consume.return_value = deliveries()
    factory = Mock(return_value=connection)
    monkeypatch.setattr(rabbit.pika, "BlockingConnection", factory)
    connected = threading.Event()
    channel.basic_qos.side_effect = lambda **_kwargs: connected.set()
    source = RabbitSource(lambda _message: None)
    try:
        assert connected.wait(3)
    finally:
        source.close(timeout=3)
    params = factory.call_args.args[0]
    assert params.host == "127.0.0.1"
    assert params.port == 5672
    assert params.virtual_host == "/"
    connection.close.assert_called_once_with()
    with pytest.raises(TransportClosed):
        source.start()


def test_source_prefetch_is_configurable_and_validated():
    """prefetch 1 через VPN (RTT 225 мс) — ~4 сообщения/с при ~20/с от edge (docs/EDGE.md §7.1)."""
    assert DEFAULT_PREFETCH >= 50
    with pytest.raises(ValueError):
        RabbitSource(prefetch=0, auto_start=False)
    with pytest.raises(ValueError):
        RabbitSource(prefetch=1.5, auto_start=False)
    source = RabbitSource(prefetch=7, auto_start=False)
    assert source._prefetch == 7
