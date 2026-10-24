"""Transport for local and edge/gateway operation.

Local:
    transport = MemoryTransport()
    transport.on_message(handle_message)
    transport.publish(message)

RabbitMQ:
    source = RabbitSource(callback=handle_message)
    sink = RabbitSink()
    sink.publish(message)

Callbacks receive validated message models. RabbitMQ delivery is at least once:
gateway handlers should upsert events by Event.id and tolerate redelivery.
Connections and channels are used exclusively by their owning worker threads.
"""

from .codec import decode, encode
from .memory import MemorySink, MemorySource, MemoryTransport
from .messages import (
    EventMsg,
    FrameMetaMsg,
    HeartbeatMsg,
    MessageCallback,
    MessageInput,
    MetricsMsg,
    Msg,
    Sink,
    Source,
    TransportClosed,
    parse_message,
)
from .rabbit import RabbitSink, RabbitSource

__all__ = [
    "EventMsg",
    "FrameMetaMsg",
    "HeartbeatMsg",
    "MessageCallback",
    "MessageInput",
    "MetricsMsg",
    "Msg",
    "Sink",
    "Source",
    "TransportClosed",
    "MemoryTransport",
    "MemorySink",
    "MemorySource",
    "RabbitSink",
    "RabbitSource",
    "encode",
    "decode",
    "parse_message",
]
