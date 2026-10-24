"""Durable edge protocol plus explicit compatibility types for upstream callers.

Production replication uses Envelope with RabbitPublisher/RabbitConsumer.
The transient MessageInput transports remain available for legacy integrations;
their binary codec lives in guard.transport.codec, not these core encode/decode.
"""

from guard.transport.protocol import Envelope, decode, encode
from .memory import MemorySink, MemorySource, MemoryTransport
from .messages import (
    EventMsg, FrameMetaMsg, HeartbeatMsg, MessageCallback, MessageInput,
    MetricsMsg, Msg, Sink, Source, TransportClosed, parse_message,
)
from .rabbit import RabbitSink, RabbitSource

__all__ = [
    "Envelope", "decode", "encode", "EventMsg", "FrameMetaMsg", "HeartbeatMsg",
    "MessageCallback", "MessageInput", "MetricsMsg", "Msg", "Sink", "Source",
    "TransportClosed", "parse_message", "MemorySink", "MemorySource",
    "MemoryTransport", "RabbitSink", "RabbitSource",
]
