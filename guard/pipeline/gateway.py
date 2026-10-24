"""Режим gateway (D-005): ноутбук с API/UI, журналом SQLite и клипами; анализ — на edge у TPU.

Из транспорта (guard.transport: RabbitSource / MemoryTransport) приходят:
- event / event_update — событие сохраняется в журнал (upsert по Event.id: доставка at-least-once),
  снапшот с edge пишется в data/media, по WS уходит event / event_update;
- frame_meta — рамки и ID на кадре frame_idx: локальный проигрыватель того же видеофайла
  подтягивается к позиции edge, оверлеи MJPEG рисуются по последним метаданным;
- metrics / heartbeat — метрики edge отдаются в /api/metrics, связь — в /health.
Клип события собирается локально из кольцевого буфера проигрывателя (сдвиг часов edge → gateway
оценивается по frame_meta).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any

import cv2
import numpy as np

from guard.contracts import CameraConfig, Event, Track
from guard.events.media import MediaStore
from guard.flywheel.wiring import attach_gateway_flywheel, start_flywheel, stop_flywheel
from guard.metrics import MetricsRegistry
from guard.pipeline import compat
from guard.pipeline.bus import EventBus
from guard.pipeline.config import load_cameras, resolve_source_uri
from guard.pipeline.engine import PipelineEngine, event_message
from guard.pipeline.settings import Settings
from guard.pipeline.worker import SHORT_LABELS, _scaled_track
from guard.rules.base import load_plugins
from guard.sources import FramePacket, make_source
from guard.sources.video import FileSource
from guard.transport import EventMsg, FrameMetaMsg, HeartbeatMsg, MetricsMsg

log = logging.getLogger(__name__)

EDGE_STALE_S = 10.0  # нет сообщений от edge дольше — связь потеряна
META_FRESH_S = 2.0


def _tracks_from_meta(meta: FrameMetaMsg, sx: float, sy: float) -> tuple[list[Track], list[Track]]:
    people = [
        Track(track_id=tid, cls="person", bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy), conf=1.0, first_seen=meta.ts, last_seen=meta.ts)
        for tid, x1, y1, x2, y2, _flags in meta.people
    ]
    objects = [
        Track(track_id=tid, cls=cls, bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy), conf=1.0, first_seen=meta.ts, last_seen=meta.ts)
        for tid, cls, x1, y1, x2, y2 in meta.objects
    ]
    return people, objects


class PlaybackWorker:
    """Локальное воспроизведение видео камеры, синхронное с frame_meta от edge."""

    def __init__(self, cfg: CameraConfig, engine: GatewayEngine) -> None:
        self.cfg = cfg
        self.engine = engine
        self.metrics = engine.metrics.reset(cfg.camera_id)
        self.metrics.set_period(cfg.sample_fps)
        self.errors: dict[str, str] = {}
        uri = cfg.source.uri
        if cfg.source.type in ("file", "images"):
            uri = resolve_source_uri(uri, engine.settings.data_dir)
        self.source = make_source(cfg.source, uri, cfg.camera_id, on_frame=self._on_frame, fps=cfg.sample_fps)
        self._metas: deque[tuple[FrameMetaMsg, float]] = deque(maxlen=128)
        self._lock = threading.Lock()
        self.ts_offset: float | None = None  # локальное время показа кадра − ts кадра на edge
        self.last_meta_mono: float | None = None
        self.meta_count = 0
        self._jpeg_cache: dict[str, tuple[Any, bytes]] = {}

    # ---- жизненный цикл (интерфейс CameraWorker для API) ----

    def start(self) -> None:
        self.source.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.source.stop(timeout=timeout)

    @property
    def running(self) -> bool:
        return self.source.running

    def _on_frame(self, pkt: FramePacket) -> None:
        self.metrics.on_source_frame()
        if self.engine.settings.clips_enabled:
            self.engine.media.push_frame(self.cfg.camera_id, pkt.frame, pkt.ts)

    # ---- метаданные с edge ----

    def on_meta(self, msg: FrameMetaMsg) -> None:
        now = time.time()
        with self._lock:
            self._metas.append((msg, now))
            offset = now - msg.ts
            self.ts_offset = offset if self.ts_offset is None else 0.9 * self.ts_offset + 0.1 * offset
        self.last_meta_mono = time.monotonic()
        self.meta_count += 1
        self.metrics.record_frame({}, capture_ts=msg.ts, done_ts=now)  # fps_processed и лаг edge → gateway
        if isinstance(self.source, FileSource):
            self.source.sync(msg.frame_idx)

    def meta_for(self, frame_idx: int | None) -> FrameMetaMsg | None:
        """Метаданные для показываемого кадра: последние с frame_idx ≤ показанного (в пределах 2 с)."""
        with self._lock:
            metas = list(self._metas)
        if not metas:
            return None
        fps = self.source.native_fps or 25.0
        if frame_idx is not None:
            best = None
            for meta, _ in metas:
                if meta.frame_idx <= frame_idx and frame_idx - meta.frame_idx <= 2 * fps:
                    if best is None or meta.frame_idx > best.frame_idx:
                        best = meta
            if best is not None:
                return best
        meta, recv = metas[-1]
        return meta if time.time() - recv < META_FRESH_S else None

    # ---- картинки ----

    def frame_jpeg(self, quality: int = 85) -> bytes | None:
        pkt = self.source.latest()
        if pkt is None:
            return None
        key = ("raw", pkt.seq)
        cached = self._jpeg_cache.get("raw")
        if cached and cached[0] == key:
            return cached[1]
        data = cv2.imencode(".jpg", pkt.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self._jpeg_cache["raw"] = (key, data)
        return data

    def overlay_jpeg(self, mode: str = "analyzed", max_width: int = 1280, quality: int = 80) -> tuple[Any, bytes] | None:
        pkt = self.source.latest()
        if pkt is None:
            return None
        fps = self.source.native_fps or 25.0
        local_idx = round(pkt.pos_s * fps) if pkt.pos_s is not None else None
        meta = self.meta_for(local_idx)
        key = ("g", pkt.seq, meta.frame_idx if meta else -1)
        cached = self._jpeg_cache.get("overlay")
        if cached and cached[0] == key:
            return cached
        frame = pkt.frame
        h, w = frame.shape[:2]
        s = min(1.0, max_width / w)
        img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else frame.copy()
        people: list[Track] = []
        objects: list[Track] = []
        highlight: dict[int, str] = {}
        if meta is not None:
            people, objects = _tracks_from_meta(meta, w / meta.w, h / meta.h)
            if s < 1.0:
                people = [_scaled_track(t, s) for t in people]
                objects = [_scaled_track(t, s) for t in objects]
            labels = self.engine.track_labels(self.cfg.camera_id)
            highlight = {int(tid): labels.get(int(tid), "ALERT") for tid in meta.violators}
        img = compat.safe_draw_overlay(img, people, objects, self.cfg.zones, highlight)
        self._status_bar(img)
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        self._jpeg_cache["overlay"] = (key, data)
        return key, data

    def _status_bar(self, img: np.ndarray) -> None:
        age = time.monotonic() - self.last_meta_mono if self.last_meta_mono else None
        edge = self.engine.edge_camera_metrics(self.cfg.camera_id)
        text = (
            f"{self.cfg.camera_id} [gateway]  edge proc {edge.get('fps_processed', 0.0):.1f} fps  "
            + (f"meta age {age * 1000:.0f} ms" if age is not None else "no edge meta")
        )
        scale = max(0.45, img.shape[1] / 1800)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (0, img.shape[0] - th - base - 8), (tw + 12, img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, text, (6, img.shape[0] - base - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def status(self) -> dict[str, Any]:
        local = self.metrics.snapshot()
        edge = self.engine.edge_camera_metrics(self.cfg.camera_id)
        age = time.monotonic() - self.last_meta_mono if self.last_meta_mono else None
        return {
            "running": self.running,
            "source": self.source.status(),
            "mode": "gateway",
            "fps_in": edge.get("fps_in", local["fps_in"]),
            "fps_processed": edge.get("fps_processed", local["fps_processed"]),
            "lag_s": edge.get("lag_s", 0.0),
            "playback_fps": local["fps_in"],
            "meta_fps": local["fps_processed"],
            "meta_age_s": round(age, 3) if age is not None else None,
            "errors": dict(self.errors),
        }


class GatewayEngine(PipelineEngine):
    """Тот же интерфейс, что у PipelineEngine (для API), но без детекторов: данные приходят с edge."""

    def __init__(self, settings: Settings, source: Any | None = None, journal: Any | None = None) -> None:
        load_plugins()
        self.settings = settings
        self.sink = None
        self.sink_errors = 0
        self.backend = None
        self.models = {}
        self.metrics = MetricsRegistry()
        self.bus = EventBus()
        self.media = MediaStore(
            settings.data_dir,
            settings.media_dir,
            pre_s=settings.clip_pre_s,
            post_s=settings.clip_post_s,
            clip_fps=settings.clip_fps,
            max_width=settings.clip_max_width,
            budget_mb=settings.clip_buffer_mb,
        )
        self.events = None
        self.journal = journal if journal is not None else self._make_journal()
        self.cameras: dict[str, CameraConfig] = {}
        self.workers: dict[str, PlaybackWorker] = {}  # type: ignore[assignment]
        self._detectors = {}
        self._schedulers = {}
        self._lock = threading.RLock()
        self._det_lock = threading.Lock()
        self.started_at = time.time()
        self.events_created = 0
        self.source = source
        self.edge_metrics: dict[str, Any] = {}
        self.edge_seen_mono: float | None = None
        self.edge_node: str | None = None
        self.messages: dict[str, int] = defaultdict(int)
        self.message_errors = 0
        self._labels: dict[str, dict[int, str]] = defaultdict(dict)
        attach_gateway_flywheel(self)  # маховик: сэмплы edge → своё хранилище (docs/FLYWHEEL.md)

    # ---- жизненный цикл ----

    def start(self) -> None:
        self.media.start()
        start_flywheel(self)
        try:
            cams = load_cameras(self.settings.cameras_config)
        except Exception:  # noqa: BLE001
            log.exception("cannot load %s", self.settings.cameras_config)
            cams = []
        with self._lock:
            for cam in cams:
                self.cameras[cam.camera_id] = cam
                if cam.enabled:
                    self._start_worker(cam)
        if self.source is not None:
            self.source.on_message(self.on_message)
        log.info("gateway started: %d cameras, transport source %s", len(cams), type(self.source).__name__)

    def stop(self) -> None:
        if self.source is not None:
            try:
                self.source.close(timeout=5.0)
            except Exception:  # noqa: BLE001
                log.exception("transport source close failed")
        with self._lock:
            for cid in list(self.workers):
                worker = self.workers.pop(cid)
                worker.stop()
        self.media.stop()
        stop_flywheel(self)
        close = getattr(self.journal, "close", None)
        if callable(close):
            close()

    def _start_worker(self, cam: CameraConfig) -> None:
        worker = PlaybackWorker(cam, self)
        self.workers[cam.camera_id] = worker
        worker.start()

    def get_detector(self, model_id: str):  # noqa: ANN201
        raise KeyError("gateway has no detectors: analysis runs on edge")

    # ---- приём сообщений ----

    def on_message(self, msg: Any) -> None:
        """Колбэк транспорта (поток потребителя). Не бросает: иначе MemoryTransport остановит приём."""
        try:
            kind = getattr(msg, "kind", "?")
            self.messages[kind] += 1
            self.edge_seen_mono = time.monotonic()
            if isinstance(msg, EventMsg):
                self._on_event_msg(msg)
            elif isinstance(msg, FrameMetaMsg):
                worker = self.workers.get(msg.camera_id)
                if worker is not None:
                    worker.on_meta(msg)
            elif isinstance(msg, MetricsMsg):
                self.edge_metrics = dict(msg.data)
            elif isinstance(msg, HeartbeatMsg):
                self.edge_node = msg.node_id
        except Exception:  # noqa: BLE001
            self.message_errors += 1
            log.exception("gateway: failed to handle %s message", getattr(msg, "kind", "?"))

    def _on_event_msg(self, msg: EventMsg) -> None:
        ev: Event = msg.event.model_copy(deep=True)
        existing = self.journal.get(ev.id)
        if msg.kind == "event" or existing is None:
            if existing is not None:
                return  # повторная доставка
            if msg.snapshot_jpeg:
                try:
                    ev.snapshot_path = self.media.save_snapshot_bytes(ev, msg.snapshot_jpeg)
                except Exception:  # noqa: BLE001
                    log.exception("gateway: snapshot save failed for %s", ev.id)
                    ev.snapshot_path = None
            else:
                ev.snapshot_path = None
            ev.clip_path = None
            ev.details = {**ev.details, "gateway_received_lag_s": round(time.time() - ev.confirmed_at, 3)}
            self.journal.add(ev)
            self.events_created += 1
            if ev.track_id is not None:
                self._labels[ev.camera_id][int(ev.track_id)] = SHORT_LABELS.get(ev.type, ev.type[:8].upper())
            self.bus.publish(event_message("event", ev))
            log.warning("EVENT(edge) %s %s camera=%s track=%s", ev.type, ev.id, ev.camera_id, ev.track_id)
            self._schedule_local_clip(ev)
            if msg.kind == "event":
                return
        updated = self._journal_update(ev.id, ended_at=ev.ended_at)
        if updated is not None:
            self.bus.publish(event_message("event_update", updated))

    def _schedule_local_clip(self, ev: Event) -> None:
        worker = self.workers.get(ev.camera_id)
        if not self.settings.clips_enabled or worker is None:
            return
        off = worker.ts_offset or 0.0
        local = ev.model_copy(update={"started_at": ev.started_at + off, "confirmed_at": ev.confirmed_at + off})
        if not self.media.schedule_clip(local, on_done=lambda _e, rel: self._on_clip(ev, rel)):
            self._on_clip(ev, None)

    def track_labels(self, camera_id: str) -> dict[int, str]:
        return dict(self._labels.get(camera_id, {}))

    # ---- сводки ----

    def edge_connected(self) -> bool:
        return self.edge_seen_mono is not None and time.monotonic() - self.edge_seen_mono < EDGE_STALE_S

    def edge_camera_metrics(self, camera_id: str) -> dict[str, Any]:
        return dict((self.edge_metrics.get("cameras") or {}).get(camera_id) or {})

    def _edge_info(self) -> dict[str, Any]:
        age = time.monotonic() - self.edge_seen_mono if self.edge_seen_mono else None
        return {
            "connected": self.edge_connected(),
            "node_id": self.edge_node,
            "last_seen_s": round(age, 2) if age is not None else None,
            "messages": dict(self.messages),
            "errors": self.message_errors,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.edge_connected() else "degraded",
            "mode": "gateway",
            "backend": self.edge_metrics.get("backend", "remote"),
            "cameras_running": sum(1 for w in self.workers.values() if w.running),
            "cameras_total": len(self.cameras),
            "uptime_s": round(time.time() - self.started_at, 1),
            "edge": self._edge_info(),
        }

    def camera_status(self, camera_id: str) -> dict[str, Any]:
        worker = self.workers.get(camera_id)
        if worker is None:
            return {"running": False, "fps_in": 0.0, "fps_processed": 0.0, "lag_s": 0.0, "errors": {}}
        return worker.status()

    def metrics_snapshot(self) -> dict[str, Any]:
        local = self.metrics.snapshot()
        cams: dict[str, Any] = {}
        for cid in self.cameras:
            edge = self.edge_camera_metrics(cid)
            loc = local.get(cid, {})
            cams[cid] = {
                "fps_in": edge.get("fps_in", loc.get("fps_in", 0.0)),
                "fps_processed": edge.get("fps_processed", loc.get("fps_processed", 0.0)),
                "lag_s": edge.get("lag_s", 0.0),
                "timings_ms": edge.get("timings_ms", {}),
                "gateway": {
                    "playback_fps": loc.get("fps_in", 0.0),
                    "meta_fps": loc.get("fps_processed", 0.0),
                    "meta_lag_s": loc.get("lag_s", 0.0),
                },
            }
        inference = self.edge_metrics.get("inference") or {"backend": "remote", "device": None, "utilization": None, "models": {}}
        return {
            "backend": self.edge_metrics.get("backend", "remote"),
            "mode": "gateway",
            "cameras": cams,
            "inference": inference,
            "media": self.media.stats(),
            "edge": self._edge_info(),
        }


def make_engine(settings: Settings) -> PipelineEngine:
    """Движок по settings.mode: local (по умолчанию) или gateway. Режим edge — python -m guard.edge."""
    if settings.mode == "gateway":
        from guard.pipeline.transport import make_source as make_transport_source

        return GatewayEngine(settings, source=make_transport_source(settings.transport, settings.rabbit_url))
    if settings.mode not in ("local", ""):
        log.warning("GUARD_MODE=%s is not served by the API process; running local", settings.mode)
    return PipelineEngine(settings)
