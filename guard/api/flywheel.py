"""API маховика данных (docs/FLYWHEEL.md, раздел «API»). Подключается в guard/api/app.py.

Хранилище — engine.flywheel_store (local и gateway). Sample — строка таблицы samples с раскрытыми
JSON-полями и frame_url. Экспорт датасета синхронный; задачи (jobs) — подпроцессы ml/flywheel/*.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from guard.auth import audit
from guard.auth.deps import require
from guard.auth.roles import FLYWHEEL_REVIEW, FLYWHEEL_TRAIN
from guard.auth.service import CurrentUser
from guard.flywheel.cards import load_model_cards, models_summary
from guard.flywheel.collector import SampleJob, sample_url
from guard.flywheel.jobs import JobError, JobForbidden, JobManager
from guard.flywheel.store import FlywheelStore, SampleError, new_sample_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/flywheel")


class LabelIn(BaseModel):
    cls: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]
    origin: str | None = None
    conf: float | None = None


class SamplePatch(BaseModel):
    status: Literal["accepted", "rejected"] | None = None
    labels: list[LabelIn] | None = None


class DatasetRequest(BaseModel):
    include_holdout: bool = False
    classes: list[str] | None = None


class JobRequest(BaseModel):
    kind: Literal["autolabel", "export", "train", "gate", "deploy"]
    args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- помощники


def _engine(request: Request) -> Any:
    return request.app.state.engine


def _store(request: Request) -> FlywheelStore:
    store = getattr(_engine(request), "flywheel_store", None)
    if store is None:
        raise HTTPException(503, "flywheel store is not available in this mode")
    return store


def sample_out(sample: dict[str, Any]) -> dict[str, Any]:
    return {**sample, "frame_url": sample_url(sample["id"])}


def _sample_or_404(store: FlywheelStore, sample_id: str) -> dict[str, Any]:
    sample = store.get(sample_id)
    if sample is None:
        raise HTTPException(404, f"sample {sample_id} not found")
    return sample


def _parse_holdout(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    low = value.strip().lower()
    if low in ("1", "true", "yes"):
        return True
    if low in ("0", "false", "no"):
        return False
    raise HTTPException(422, "holdout: expected 0/1 or true/false")


def _jobs(request: Request) -> JobManager:
    manager = getattr(request.app.state, "flywheel_jobs", None)
    if manager is None:
        store = _store(request)
        settings = _engine(request).settings
        manager = JobManager(store.root, allow_train=bool(getattr(settings, "flywheel_allow_train", False)))
        request.app.state.flywheel_jobs = manager
    return manager


def _collector_stats(engine: Any) -> dict[str, Any] | None:
    collector = getattr(engine, "flywheel", None)
    if collector is not None:
        return collector.stats()
    edge = getattr(engine, "edge_metrics", None)  # gateway: сбор идёт на edge, его сводка — в metrics
    if isinstance(edge, dict) and isinstance(edge.get("flywheel"), dict):
        return dict(edge["flywheel"])
    if getattr(engine.settings, "mode", "") == "gateway" and callable(getattr(engine, "metrics_snapshot", None)):
        try:
            data = engine.metrics_snapshot()  # шлюз: свежие metrics edge, устаревшие — без сводки
        except Exception:  # noqa: BLE001
            return None
        if isinstance(data, dict) and not data.get("stale") and isinstance(data.get("flywheel"), dict):
            return dict(data["flywheel"])
    return None


def _receiver_stats(engine: Any) -> dict[str, Any] | None:
    """Шлюз: приём частей сэмплов edge (очередь маховика и сборщик)."""
    assembler = getattr(engine, "flywheel_assembler", None)
    if assembler is None:
        return None
    receiver = getattr(engine, "flywheel_receiver", None)
    return {"assembler": assembler.stats(), "transport": receiver.status() if receiver is not None else None}


# ---------------------------------------------------------------- сводка и сэмплы


@router.get("/summary", dependencies=[require(FLYWHEEL_REVIEW)])
def summary(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    data = _store(request).summary()
    stats = _collector_stats(engine)
    dropped = dict((stats or {}).get("dropped") or {})
    assembler = getattr(engine, "flywheel_assembler", None)
    if assembler is not None:
        for key, value in assembler.stats().get("dropped", {}).items():
            if value:
                dropped[f"gateway_{key}"] = value
    writer = getattr(engine, "flywheel_writer", None)
    collecting = bool(stats and stats.get("collecting"))
    if getattr(engine, "flywheel", None) is None and stats is not None:
        connected = getattr(engine, "edge_connected", None)
        collecting = collecting and bool(connected() if callable(connected) else True)
    data.update(
        {
            "dropped": int(sum(dropped.values())),
            "dropped_detail": dropped,
            "models": models_summary(load_model_cards()),
            "collecting": collecting,
            "collector": stats,
            "writer": writer.stats() if writer is not None else None,
            "receiver": _receiver_stats(engine),
            "allow_train": bool(getattr(engine.settings, "flywheel_allow_train", False)),
            "mode": getattr(engine.settings, "mode", "local"),
        }
    )
    return data


@router.get("/samples", dependencies=[require(FLYWHEEL_REVIEW)])
def list_samples(
    request: Request,
    status: str | None = None,
    source: str | None = None,
    camera_id: str | None = None,
    cls: str | None = None,
    holdout: str | None = None,
    event_id: str | None = None,
    limit: int = Query(50, ge=0, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("-created_at"),
) -> dict[str, Any]:
    store = _store(request)
    try:
        items, total = store.query(
            status=status, source=source, camera_id=camera_id, cls=cls, holdout=_parse_holdout(holdout),
            event_id=event_id, limit=limit, offset=offset, order=order,
        )
    except SampleError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"items": [sample_out(s) for s in items], "total": total}


@router.get("/samples/{sample_id}", dependencies=[require(FLYWHEEL_REVIEW)])
def get_sample(sample_id: str, request: Request) -> dict[str, Any]:
    return sample_out(_sample_or_404(_store(request), sample_id))


@router.get("/samples/{sample_id}/frame.jpg", dependencies=[require(FLYWHEEL_REVIEW)])
def sample_frame(sample_id: str, request: Request) -> FileResponse:
    store = _store(request)
    sample = _sample_or_404(store, sample_id)
    path = store.resolve_frame(sample["frame_path"])
    if path is None:
        raise HTTPException(404, "frame not available")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.patch("/samples/{sample_id}")
def patch_sample(
    sample_id: str, body: SamplePatch, request: Request, user: CurrentUser = require(FLYWHEEL_REVIEW)
) -> dict[str, Any]:
    store = _store(request)
    _sample_or_404(store, sample_id)
    labels = None
    if body.labels is not None:
        labels = [lb.model_dump() for lb in body.labels]
    try:
        updated = store.review(sample_id, status=body.status, labels=labels)
    except SampleError as exc:
        raise HTTPException(422, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, f"sample {sample_id} not found")
    audit.record(request, audit.FLYWHEEL_REVIEW, sample_id, {
        "status": body.status, "labels": None if labels is None else len(labels), "camera_id": updated.get("camera_id"),
    }, user=user)
    return sample_out(updated)


@router.post("/capture/{camera_id}", status_code=201, dependencies=[require(FLYWHEEL_REVIEW)])
def capture(camera_id: str, request: Request) -> dict[str, Any]:
    """Текущий кадр камеры → сэмпл source = operator."""
    engine = _engine(request)
    _store(request)
    cam = engine.get_camera(camera_id)
    if cam is None:
        raise HTTPException(404, f"camera {camera_id} not found")
    writer = getattr(engine, "flywheel_writer", None)
    if writer is None:
        raise HTTPException(503, "flywheel writer is not available")
    collector = getattr(engine, "flywheel", None)
    stored = None
    if collector is not None:
        stored = collector.capture(camera_id, writer, camera_name=cam.name)
    if stored is None:
        stored = _capture_playback(engine, cam, writer)
    if stored is None:
        raise HTTPException(503, f"camera {camera_id} has no frame yet")
    return sample_out(stored)


def _capture_playback(engine: Any, cam: Any, writer: Any) -> dict[str, Any] | None:
    """Gateway: кадр локального проигрывателя и рамки из последнего frame_meta edge."""
    worker = engine.workers.get(cam.camera_id)
    if worker is None:
        return None
    source = getattr(worker, "source", None)
    latest = getattr(source, "latest", None)
    if not callable(latest):
        return None
    pkt = latest()
    if pkt is None:
        return None
    det_dicts: list[dict[str, Any]] = []
    meta_for = getattr(worker, "meta_for", None)
    if callable(meta_for):
        fps = source.native_fps or 25.0
        meta = meta_for(round(pkt.pos_s * fps) if pkt.pos_s is not None else None)
        if meta is not None:
            h, w = pkt.frame.shape[:2]
            sx, sy = w / meta.w, h / meta.h
            # треки edge из frame_meta: уверенности в них нет, как и в оверлее шлюза — conf 1.0
            for tid, x1, y1, x2, y2, _flags in meta.people:
                det_dicts.append({"model_id": "edge_track", "cls": "person", "conf": 1.0,
                                  "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy], "track_id": tid})
            for tid, cls, x1, y1, x2, y2 in meta.objects:
                det_dicts.append({"model_id": "edge_track", "cls": cls, "conf": 1.0,
                                  "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy], "track_id": tid})
    from guard.flywheel.holdout import is_holdout_uri

    sample = {
        "id": new_sample_id(),
        "created_at": time.time(),
        "camera_id": cam.camera_id,
        "source": "operator",
        "reason": "оператор отправил кадр в выборку",
        "holdout": int(is_holdout_uri(cam.source.uri)),
        "meta": {"camera_name": cam.name},
    }
    from guard.flywheel.frames import encode_jpeg

    jpeg, width, height, scale = encode_jpeg(pkt.frame, writer.max_width, writer.quality)
    scaled = [{**d, "bbox": [round(v * scale, 1) for v in d["bbox"]]} for d in det_dicts]
    return writer.write(SampleJob(sample={**sample, "width": width, "height": height}, jpeg=jpeg, det_dicts=scaled))


# ---------------------------------------------------------------- датасеты и модели


@router.post("/datasets", status_code=201)
def create_dataset(
    body: DatasetRequest, request: Request, user: CurrentUser = require(FLYWHEEL_TRAIN)
) -> dict[str, Any]:
    store = _store(request)
    from ml.flywheel.export import ExportError, export_dataset

    try:
        dataset = export_dataset(store, include_holdout=body.include_holdout, classes=body.classes)
    except ExportError as exc:
        raise HTTPException(422, str(exc)) from exc
    audit.record(request, audit.FLYWHEEL_JOB, str(dataset.get("version") or ""), {
        "kind": "dataset", "include_holdout": body.include_holdout, "classes": body.classes,
    }, user=user)
    return dataset


@router.get("/datasets", dependencies=[require(FLYWHEEL_REVIEW)])
def list_datasets(request: Request) -> list[dict[str, Any]]:
    return _store(request).list_datasets()


@router.get("/models", dependencies=[require(FLYWHEEL_REVIEW)])
def list_models() -> list[dict[str, Any]]:
    return load_model_cards()


# ---------------------------------------------------------------- задачи


@router.post("/jobs", status_code=201)
def create_job(body: JobRequest, request: Request, user: CurrentUser = require(FLYWHEEL_TRAIN)) -> dict[str, Any]:
    manager = _jobs(request)
    try:
        job = manager.start(body.kind, body.args)
    except JobForbidden as exc:
        raise HTTPException(403, str(exc)) from exc
    except JobError as exc:
        raise HTTPException(422, str(exc)) from exc
    out = manager.to_dict(job, tail=False)
    audit.record(request, audit.FLYWHEEL_JOB, str(out.get("id") or ""), {
        "kind": body.kind, "args": body.args, "status": out.get("status"),
    }, user=user)
    return out


@router.get("/jobs", dependencies=[require(FLYWHEEL_REVIEW)])
def list_jobs(request: Request) -> list[dict[str, Any]]:
    manager = _jobs(request)
    return [manager.to_dict(j, tail=False) for j in manager.list()]


@router.get("/jobs/{job_id}", dependencies=[require(FLYWHEEL_REVIEW)])
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    manager = _jobs(request)
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return manager.to_dict(job, tail=True)


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse, dependencies=[require(FLYWHEEL_REVIEW)])
def job_log(job_id: str, request: Request) -> str:
    manager = _jobs(request)
    if manager.get(job_id) is None:
        raise HTTPException(404, f"job {job_id} not found")
    return manager.tail(job_id, max_bytes=200_000)
