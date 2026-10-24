"""Тонкий слой над guard.transport (CX-011): выбор Sink/Source по настройкам и сборка сообщений edge.

Режимы (D-005): edge публикует event (+ снапшот ≤ 60 КБ), frame_meta (рамки int, ~1–3 КБ)
и metrics; gateway их принимает. Транспорт: memory (один процесс, тесты), rabbit
(exchange zazerkalye, очередь zazerkalye.gateway), log (отладка без брокера).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from guard.contracts import Event
from guard.transport import EventMsg, FrameMetaMsg, MemoryTransport, MetricsMsg, Sink

log = logging.getLogger(__name__)

FLAG_VIOLATOR = 1  # бит в PersonRow.flags / ObjectRow не несёт флагов

_shared_memory: MemoryTransport | None = None


def shared_memory_transport() -> MemoryTransport:
    """Общая in-process шина для режима «edge и gateway в одном процессе»."""
    global _shared_memory
    if _shared_memory is None:
        _shared_memory = MemoryTransport()
    return _shared_memory


class LogSink:
    """Отладочный Sink: пишет краткую строку на сообщение в лог."""

    def __init__(self) -> None:
        self.count: dict[str, int] = {}

    def publish(self, message: Any) -> bool:
        kind = message.kind if hasattr(message, "kind") else message.get("kind", "?")
        self.count[kind] = self.count.get(kind, 0) + 1
        if kind in ("event", "event_update"):
            ev = message.event
            log.warning("SINK %s %s %s camera=%s track=%s", kind, ev.id, ev.type, ev.camera_id, ev.track_id)
        return True

    def close(self, timeout: float | None = 5.0) -> None:
        return None


def make_sink(transport: str, url: str | None = None) -> Sink:
    if transport == "memory":
        return shared_memory_transport()
    if transport == "log":
        return LogSink()
    if transport == "rabbit":
        from guard.transport.rabbit import RabbitSink

        return RabbitSink(url)
    raise ValueError(f"unknown transport {transport!r}: memory | rabbit | log")


def make_source(transport: str, url: str | None = None) -> Any:
    """Source без колбэка: gateway вызывает on_message(cb), после чего начинается приём."""
    if transport == "memory":
        return shared_memory_transport()
    if transport == "rabbit":
        from guard.transport.rabbit import RabbitSource

        return RabbitSource(url=url)
    raise ValueError(f"unknown transport {transport!r} for gateway: memory | rabbit")


def compact_jpeg(img: np.ndarray, width: int = 640, quality: int = 70, max_kb: int = 60) -> bytes:
    """Снапшот для медленного обратного канала: ≤ width px, JPEG q≈70, не больше max_kb."""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(2, int(h * width / w))), interpolation=cv2.INTER_AREA)
    q = quality
    while True:
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
        if len(data) <= max_kb * 1024 or q <= 25:
            return data
        q -= 10


def event_msg(kind: str, event: Event, snapshot: bytes | None = None) -> EventMsg:
    return EventMsg(kind=kind, event=event, snapshot_jpeg=snapshot)


def frame_meta_msg(
    camera_id: str, frame_idx: int, ts: float, size: tuple[int, int], people: list, objects: list, violators: set[int]
) -> FrameMetaMsg:
    w, h = size
    return FrameMetaMsg(
        camera_id=camera_id,
        frame_idx=max(0, int(frame_idx)),
        ts=float(ts),
        w=int(w),
        h=int(h),
        people=[
            (int(t.track_id), *(int(round(v)) for v in t.bbox), FLAG_VIOLATOR if t.track_id in violators else 0)
            for t in people
        ],
        objects=[(int(t.track_id), str(t.cls), *(int(round(v)) for v in t.bbox)) for t in objects],
        violators=sorted(int(v) for v in violators),
    )


def metrics_msg(data: dict[str, Any]) -> MetricsMsg:
    return MetricsMsg(data=data)
