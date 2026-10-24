"""Проверка потока до добавления камеры: POST /api/cameras/probe (docs/STREAMS.md, право cameras.manage).

Открытие идёт в пуле потоков и не блокирует цикл событий. Одновременно — не больше PROBE_CONCURRENCY
проверок: каждая держит соединение с камерой и декодер. cv2 и источники импортируются только при вызове:
шлюз (GUARD_MODE=gateway) их не загружает.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from guard.contracts import SourceConfig

PROBE_TIMEOUT_S = 15.0  # как guard.sources.probe.PROBE_TIMEOUT_S
PROBE_CONCURRENCY = 2
_slots = threading.BoundedSemaphore(PROBE_CONCURRENCY)

router = APIRouter()


class ProbeRequest(BaseModel):
    source: SourceConfig
    timeout_s: float | None = Field(None, gt=0, le=60, description="таймаут открытия и первого кадра, с; по умолчанию 15")


class ProbeResponse(BaseModel):
    ok: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    first_frame_ms: float | None = None
    error: str | None = None
    preview_jpeg_b64: str | None = None
    # дополнительно к контракту docs/ROLES.md
    error_code: str | None = None
    type: str | None = None
    stream: str | None = None
    uri: str | None = None
    open_ms: float | None = None
    fps_nominal: float | None = None
    fps_measured: float | None = None
    frames: int = 0
    resolved_via: str | None = None
    hints: list[str] = Field(default_factory=list)


@router.post("/api/cameras/probe", response_model=ProbeResponse)
async def probe_camera_source(body: ProbeRequest, request: Request) -> dict[str, Any]:
    engine = request.app.state.engine
    settings = engine.settings
    if getattr(settings, "mode", "local") == "gateway":
        raise HTTPException(409, "в режиме gateway камеры подключает edge: проверьте поток на edge")
    from guard.pipeline.config import resolve_source_uri
    from guard.sources.probe import probe_source

    source = body.source
    uri = source.uri.strip()
    if source.type in ("file", "images"):
        uri = resolve_source_uri(uri, settings.data_dir)
    if not _slots.acquire(blocking=False):
        raise HTTPException(429, "уже идут другие проверки потока, повторите через несколько секунд")
    try:
        return await run_in_threadpool(probe_source, source, uri, body.timeout_s or PROBE_TIMEOUT_S)
    finally:
        _slots.release()
