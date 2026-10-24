import threading

import pytest

from guard.transport import (
    MemorySink,
    MemorySource,
    MemoryTransport,
    MetricsMsg,
    Sink,
    Source,
    TransportClosed,
)


def test_memory_endpoints_and_drain_after_close():
    transport = MemoryTransport()
    sink = MemorySink(transport)
    source = MemorySource(transport)
    assert isinstance(sink, Sink)
    assert isinstance(source, Source)

    first = MetricsMsg(data={"fps": 5})
    second = MetricsMsg(data={"fps": 6})
    assert sink.publish(first)
    assert sink.publish(second.model_dump())
    transport.close()
    assert list(source.iter()) == [first, second]
    assert list(source.iter()) == []
    with pytest.raises(TransportClosed):
        sink.publish(first)


def test_publication_owns_a_deep_copy():
    transport = MemoryTransport()
    model = MetricsMsg(data={"nested": {"values": [1, 2]}})
    transport.publish(model)
    model.data["nested"]["values"].append(3)

    dictionary = {"kind": "metrics", "data": {"nested": [10]}}
    transport.publish(dictionary)
    dictionary["data"]["nested"].append(20)

    transport.close()
    first, second = list(transport.iter())
    assert first.data == {"nested": {"values": [1, 2]}}
    assert second.data == {"nested": [10]}


def test_callback_delivery_and_close_drain():
    transport = MemoryTransport()
    received = []
    callback_threads = []

    def callback(message):
        received.append(message.data["index"])
        callback_threads.append(threading.get_ident())

    transport.on_message(callback)
    with pytest.raises(RuntimeError, match="already running"):
        transport.on_message(callback)
    for index in range(50):
        transport.publish(MetricsMsg(data={"index": index}))
    transport.close()
    assert received == list(range(50))
    assert len(set(callback_threads)) == 1
    assert callback_threads[0] != threading.get_ident()
    with pytest.raises(TransportClosed):
        transport.on_message(callback)


def test_concurrent_producers():
    transport = MemoryTransport()

    def produce(producer):
        for index in range(50):
            transport.publish(
                MetricsMsg(data={"producer": producer, "index": index})
            )

    threads = [
        threading.Thread(target=produce, args=(producer,))
        for producer in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    transport.close()
    messages = list(transport.iter())
    assert len(messages) == 200
    assert {
        (message.data["producer"], message.data["index"])
        for message in messages
    } == {(producer, index) for producer in range(4) for index in range(50)}


def test_close_wakes_waiting_iterator():
    transport = MemoryTransport()
    result = []
    started = threading.Event()

    def consume():
        started.set()
        result.extend(transport.iter())

    thread = threading.Thread(target=consume)
    thread.start()
    assert started.wait(3)
    transport.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result == []


def test_callback_failure_retains_message_and_is_reported():
    transport = MemoryTransport()
    attempted = threading.Event()
    message = MetricsMsg(data={"index": 1})

    def fail(_message):
        attempted.set()
        raise RuntimeError("handler unavailable")

    transport.on_message(fail)
    transport.publish(message)
    assert attempted.wait(3)
    with pytest.raises(RuntimeError, match="remains queued"):
        transport.close()
    assert isinstance(transport.callback_error, RuntimeError)
    assert list(transport.iter()) == [message]
