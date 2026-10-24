"""Validated messages shared by local and RabbitMQ transports."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter

from guard.contracts import Event


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EventMsg(_Message):
    kind: Literal["event", "event_update"] = "event"
    event: Event
    snapshot_jpeg: bytes | None = None


# JSON encodes these tuples as compact arrays. Flags are an opaque integer
# bitmask: interpretation belongs to the producer and overlay renderer.
PersonRow: TypeAlias = tuple[
    StrictInt, StrictInt, StrictInt, StrictInt, StrictInt, StrictInt
]
ObjectRow: TypeAlias = tuple[
    StrictInt, str, StrictInt, StrictInt, StrictInt, StrictInt
]


class FrameMetaMsg(_Message):
    kind: Literal["frame_meta"] = "frame_meta"
    camera_id: str = Field(min_length=1)
    frame_idx: StrictInt = Field(ge=0)
    ts: float
    w: StrictInt = Field(gt=0)
    h: StrictInt = Field(gt=0)
    people: list[PersonRow] = Field(default_factory=list)
    objects: list[ObjectRow] = Field(default_factory=list)
    violators: list[StrictInt] = Field(default_factory=list)


class MetricsMsg(_Message):
    kind: Literal["metrics"] = "metrics"
    data: dict[str, Any]


class HeartbeatMsg(_Message):
    """Application heartbeat, carried on edge.metrics."""

    kind: Literal["heartbeat"] = "heartbeat"
    ts: float = Field(default_factory=time.time)
    node_id: str = "edge"


class FlywheelSampleMsg(_Message):
    """Data-flywheel sample from an edge node (docs/FLYWHEEL.md, «Edge-режим»).

    The raw JPEG travels in small parts to bound head-of-line blocking of
    events on the slow stand → gateway channel. Part 0 carries the metadata;
    the JPEG part is carried outside JSON like an event snapshot.
    """

    kind: Literal["flywheel_sample"] = "flywheel_sample"
    sample_id: str = Field(min_length=1, max_length=64)
    part: StrictInt = Field(ge=0)
    parts: StrictInt = Field(ge=1, le=256)
    sample: dict[str, Any] | None = None
    jpeg: bytes | None = None


Msg: TypeAlias = Annotated[
    EventMsg | FrameMetaMsg | MetricsMsg | HeartbeatMsg | FlywheelSampleMsg,
    Field(discriminator="kind"),
]
MessageInput: TypeAlias = Msg | dict[str, Any]
MessageCallback: TypeAlias = Callable[[Msg], None]

_MESSAGE_ADAPTER = TypeAdapter(Msg)


def parse_message(message: MessageInput) -> Msg:
    """Accept models or the dictionaries currently emitted by EventBus."""

    return _MESSAGE_ADAPTER.validate_python(message)


class TransportClosed(RuntimeError):
    """The transport no longer accepts new messages."""


@runtime_checkable
class Sink(Protocol):
    def publish(self, message: MessageInput) -> bool:
        """Return False only when expendable telemetry was dropped."""
        ...


@runtime_checkable
class Source(Protocol):
    def on_message(self, callback: MessageCallback) -> None:
        """Start one consumer; callbacks run on its worker thread."""
        ...
