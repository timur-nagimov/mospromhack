"""FastAPI-приложение (docs/CONTRACTS.md, «HTTP API»). Запуск: uvicorn guard.api.app:app --port 8000."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from guard.contracts import CameraConfig, Event, EventStatus
from guard.pipeline.errors import CameraError, CameraExists, CameraReadOnly
from guard.pipeline.settings import Settings
from guard.rules.base import available_rules, load_plugins
from guard.rules.geometry import LOCAL_TZ

log = logging.getLogger(__name__)

METRICS_PERIOD_S = 2.0
STATS_GROUPS = {"type", "camera", "zone", "site", "hour", "day", "weekday_hour", "severity", "status"}


class EventPatch(BaseModel):
    status: EventStatus | None = None
    operator_note: str | None = None


def _engine(request: Request) -> Any:
    return request.app.state.engine


def _camera_out(engine: Any, cam: CameraConfig) -> dict[str, Any]:
    """CameraConfig плюс необязательное поле status (рантайм: running, fps, ошибки)."""
    return {**cam.model_dump(mode="json"), "status": engine.camera_status(cam.camera_id)}


def _parse_ts(value: str | None, name: str) -> float | None:
    """from/to: epoch-секунды или ISO 8601 (без зоны — по Москве)."""
    if value is None or value == "":
        return None
    try:
        timestamp = float(value)
    except ValueError:
        pass
    else:
        if not math.isfinite(timestamp):
            raise HTTPException(422, f"{name}: expected finite epoch seconds")
        return timestamp
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, f"{name}: expected epoch seconds or ISO 8601") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.timestamp()


def _split(value: str | None) -> str | list[str] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else parts


# ---------------------------------------------------------------- routes

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    return _engine(request).health()


# ---- камеры ----


@router.get("/api/cameras")
def list_cameras(request: Request) -> list[dict[str, Any]]:
    engine = _engine(request)
    return [_camera_out(engine, c) for c in engine.list_cameras()]


@router.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str, request: Request) -> dict[str, Any]:
    engine = _engine(request)
    cam = engine.get_camera(camera_id)
    if cam is None:
        raise HTTPException(404, f"camera {camera_id} not found")
    return _camera_out(engine, cam)


@router.post("/api/cameras", status_code=201)
def create_camera(cam: CameraConfig, request: Request) -> dict[str, Any]:
    engine = _engine(request)
    try:
        engine.add_camera(cam)
    except CameraReadOnly as exc:
        raise HTTPException(409, str(exc)) from exc
    except CameraExists as exc:
        raise HTTPException(409, f"camera {cam.camera_id} already exists") from exc
    except CameraError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _camera_out(engine, cam)


@router.put("/api/cameras/{camera_id}")
def update_camera(camera_id: str, cam: CameraConfig, request: Request) -> dict[str, Any]:
    engine = _engine(request)
    try:
        engine.update_camera(camera_id, cam)
    except CameraReadOnly as exc:
        raise HTTPException(409, str(exc)) from exc
    except CameraError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"camera {camera_id} not found") from exc
    return _camera_out(engine, cam)


@router.delete("/api/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: str, request: Request) -> Response:
    try:
        _engine(request).delete_camera(camera_id)
    except CameraReadOnly as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"camera {camera_id} not found") from exc
    return Response(status_code=204)


def _worker(request: Request, camera_id: str):
    engine = _engine(request)
    if engine.get_camera(camera_id) is None:
        raise HTTPException(404, f"camera {camera_id} not found")
    worker = engine.workers.get(camera_id)
    if worker is None:
        raise HTTPException(503, f"camera {camera_id} is not running")
    return worker


@router.get("/api/cameras/{camera_id}/frame.jpg")
def camera_frame(camera_id: str, request: Request) -> Response:
    data = _worker(request, camera_id).frame_jpeg()
    if data is None:
        raise HTTPException(503, "no frame yet")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/api/cameras/{camera_id}/stream.mjpg")
async def camera_stream(
    camera_id: str,
    request: Request,
    fps: float = Query(0, ge=0, le=30, description="макс. кадров/с; 0 — из GUARD_STREAM_FPS"),
    mode: str = Query("analyzed", pattern="^(analyzed|live)$"),
) -> StreamingResponse:
    worker = _worker(request, camera_id)
    settings: Settings = _engine(request).settings
    period = 1.0 / (fps or settings.stream_fps)
    boundary = "frame"

    async def gen() -> AsyncIterator[bytes]:
        nonlocal worker
        engine = _engine(request)
        last_key = None
        while not await request.is_disconnected():
            current = engine.workers.get(camera_id)
            if current is None:  # камеру удалили или выключили — поток заканчивается, клиент переподключится
                return
            if current is not worker:  # горячий перезапуск камеры (PUT): переходим на новый воркер, иначе картинка замирает
                worker, last_key = current, None
            item = await run_in_threadpool(worker.overlay_jpeg, mode, settings.stream_max_width, settings.jpeg_quality)
            if item is not None and item[0] != last_key:
                last_key, data = item
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ---- правила ----


@router.get("/api/rules/types")
def rule_types() -> list[dict[str, Any]]:
    load_plugins()  # идемпотентно: реестр мог быть заполнен частично прямым импортом плагина
    return available_rules()


# ---- события ----


@router.get("/api/events")
def list_events(
    request: Request,
    type: str | None = None,
    camera_id: str | None = None,
    zone_id: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    limit: int = Query(50, ge=0, le=1000),
    offset: int = Query(0, ge=0),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    journal = _engine(request).journal
    items, total = journal.query(
        type=_split(type),
        camera_id=_split(camera_id),
        zone_id=_split(zone_id),
        status=_split(status),
        severity=_split(severity),
        ts_from=_parse_ts(from_, "from"),
        ts_to=_parse_ts(to, "to"),
        limit=limit,
        offset=offset,
        order=order,
    )
    return {"items": [e.model_dump(mode="json") for e in items], "total": total}


def _event_or_404(request: Request, event_id: str) -> Event:
    ev = _engine(request).journal.get(event_id)
    if ev is None:
        raise HTTPException(404, f"event {event_id} not found")
    return ev


@router.get("/api/events/{event_id}")
def get_event(event_id: str, request: Request) -> dict[str, Any]:
    return _event_or_404(request, event_id).model_dump(mode="json")


@router.patch("/api/events/{event_id}")
def patch_event(event_id: str, body: EventPatch, request: Request) -> dict[str, Any]:
    _event_or_404(request, event_id)
    fields = {k: getattr(body, k) for k in body.model_fields_set}
    if "status" in fields and fields["status"] is None:
        raise HTTPException(422, "status must not be null")
    updated = _engine(request).patch_event(event_id, **fields)
    if updated is None:
        raise HTTPException(404, f"event {event_id} not found")
    return updated.model_dump(mode="json")


def _media_file(request: Request, rel: str | None, what: str) -> Path:
    path = _engine(request).media.resolve(rel)
    if path is None:
        raise HTTPException(404, f"{what} not available")
    return path


@router.get("/api/events/{event_id}/snapshot.jpg")
def event_snapshot(event_id: str, request: Request) -> FileResponse:
    ev = _event_or_404(request, event_id)
    return FileResponse(_media_file(request, ev.snapshot_path, "snapshot"), media_type="image/jpeg")


@router.get("/api/events/{event_id}/clip.mp4")
def event_clip(event_id: str, request: Request) -> FileResponse:
    ev = _event_or_404(request, event_id)
    return FileResponse(_media_file(request, ev.clip_path, "clip"), media_type="video/mp4")


@router.get("/api/stats")
def stats(
    request: Request,
    group_by: str = Query("type"),
    type: str | None = None,
    camera_id: str | None = None,
    zone_id: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
) -> dict[str, Any]:
    if group_by not in STATS_GROUPS:
        raise HTTPException(422, f"group_by must be one of {sorted(STATS_GROUPS)}")
    items = _engine(request).journal.stats(
        group_by,
        ts_from=_parse_ts(from_, "from"),
        ts_to=_parse_ts(to, "to"),
        type=_split(type),
        camera_id=_split(camera_id),
        zone_id=_split(zone_id),
        status=_split(status),
        severity=_split(severity),
    )
    return {"group_by": group_by, "items": items}


@router.get("/api/metrics")
def metrics(request: Request) -> dict[str, Any]:
    return _engine(request).metrics_snapshot()


@router.get("/api/analytics")
def analytics(
    request: Request,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    employee_id: str | None = Query(None, min_length=1),
) -> dict[str, Any]:
    """Агрегаты выбранного периода из одного согласованного снимка журнала."""
    from guard.employees import UnknownEmployee
    try:
        return _engine(request).journal.analytics(
            ts_from=_parse_ts(from_, "from"), ts_to=_parse_ts(to, "to"), employee_id=employee_id,
        )
    except UnknownEmployee as exc:
        raise HTTPException(404, str(exc)) from exc


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    engine = ws.app.state.engine
    await ws.accept()
    sub = engine.bus.subscribe()

    async def drain_client() -> None:  # входящие не нужны, но так узнаём об отключении
        while True:
            await ws.receive_text()

    reader = asyncio.create_task(drain_client())
    try:
        while not reader.done():
            getter = asyncio.create_task(sub.get())
            done, _ = await asyncio.wait({getter, reader}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                await ws.send_json(getter.result())
            else:
                getter.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        engine.bus.unsubscribe(sub)
        reader.cancel()
        with contextlib.suppress(BaseException):
            await reader


# ---------------------------------------------------------------- app


class SPAStaticFiles(StaticFiles):
    """Статика фронтенда; неизвестные пути (маршруты SPA) отдают index.html."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.startswith(("api/", "ws/")):
                raise
            return await super().get_response("index.html", scope)


async def _metrics_loop(engine: Any) -> None:
    while True:
        await asyncio.sleep(METRICS_PERIOD_S)
        if engine.bus.subscribers:
            try:
                data = await run_in_threadpool(engine.metrics_snapshot)
                engine.bus.publish({"kind": "metrics", "data": data})
            except Exception:  # noqa: BLE001
                log.exception("metrics broadcast failed")


def _create_engine(settings: Settings) -> Any:
    if settings.mode == "gateway":
        if settings.config_sync:  # камеры меняются в UI шлюза и применяются на edge (docs/EDGE.md §8)
            from guard.gateway.config_sync import ConfigSyncGatewayEngine
            return ConfigSyncGatewayEngine(settings)
        from guard.gateway.engine import GatewayEngine
        return GatewayEngine(settings)
    if settings.mode == "edge":
        if settings.config_sync:
            from guard.edge.config_sync import ConfigSyncEdgeEngine
            return ConfigSyncEdgeEngine(settings)
        from guard.edge.runner import EdgeEngine
        return EdgeEngine(settings)
    from guard.pipeline.engine import PipelineEngine
    return PipelineEngine(settings)


def create_app(settings: Settings | None = None, engine: Any | None = None) -> FastAPI:
    settings = settings or (engine.settings if engine else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or _create_engine(settings)
        app.state.engine = eng
        task = None
        failed = False
        try:
            if settings.autostart:
                await run_in_threadpool(eng.start)
            task = asyncio.create_task(_metrics_loop(eng))
            yield
        except BaseException:
            failed = True
            raise
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            try:
                await run_in_threadpool(eng.stop)
            except Exception as exc:
                if not failed:
                    raise
                log.error("Startup cleanup failed: %s", type(exc).__name__)

    app = FastAPI(title="Зазеркалье — видеоаналитика ТБ", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    from guard.api.reports import router as reports_router  # CX-020
    app.include_router(reports_router)
    from guard.api.flywheel import router as flywheel_router  # маховик данных, docs/FLYWHEEL.md
    app.include_router(flywheel_router)
    from guard.api.streams import router as streams_router  # проверка потока, docs/STREAMS.md
    app.include_router(streams_router)
    from guard.api.employees import router as employees_router
    app.include_router(employees_router)
    if settings.frontend_dist.is_dir():
        app.mount("/", SPAStaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return app


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_configure_logging()
app = create_app()
