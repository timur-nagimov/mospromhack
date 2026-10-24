"""Движок: камеры (CRUD + горячий перезапуск), общие детекторы, журнал, медиа, шина WS, метрики."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from guard.contracts import CameraConfig, Event, FrameContext
from guard.inference import get_backend
from guard.inference.base import InferenceBackend
from guard.metrics import MetricsRegistry
from guard.pipeline import compat
from guard.pipeline.bus import EventBus
from guard.pipeline.config import load_cameras, load_models, save_cameras
from guard.pipeline.detector import BatchScheduler, Detector
from guard.pipeline.settings import Settings
from guard.rules.base import available_rules, load_plugins, with_default_severity
from guard.rules.geometry import validate_active_hours
from guard.events.media import MediaStore

if TYPE_CHECKING:
    from guard.pipeline.worker import CameraWorker

log = logging.getLogger(__name__)


class CameraError(ValueError):
    """Некорректная конфигурация камеры (→ HTTP 422)."""


class CameraExists(KeyError):
    """camera_id уже занят (→ HTTP 409)."""


def event_message(kind: str, event: Event) -> dict[str, Any]:
    return {"kind": kind, "event": event.model_dump(mode="json")}


class PipelineEngine:
    def __init__(
        self,
        settings: Settings,
        backend: InferenceBackend | None = None,
        journal: Any | None = None,
        event_manager: Any | None = None,
    ) -> None:
        self.settings = settings
        # base.available_rules()/get_rule() грузят плагины, только пока реестр пуст — грузим явно
        load_plugins()
        self.backend = backend or get_backend(
            settings.backend,
            models_dir=settings.models_dir,
            intra_op_threads=settings.ort_threads,
            max_batch=settings.max_batch,
        )
        self.models = load_models(settings.models_config, settings.models_dir)
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
        self.events = event_manager if event_manager is not None else compat.EventManager()
        self.journal = journal if journal is not None else self._make_journal()
        self.cameras: dict[str, CameraConfig] = {}
        self.workers: dict[str, CameraWorker] = {}
        self._detectors: dict[str, Detector] = {}
        self._schedulers: dict[str, BatchScheduler] = {}
        self._lock = threading.RLock()
        self._det_lock = threading.Lock()
        self.started_at = time.time()
        self.events_created = 0

    def _make_journal(self) -> Any:
        try:
            return compat.Journal(self.settings.db_path)
        except Exception:  # noqa: BLE001 — без журнала сервис всё равно должен оповещать
            log.exception("journal at %s unavailable, using in-memory journal", self.settings.db_path)
            from guard.pipeline._fallbacks import MemoryJournal

            return MemoryJournal()

    # ---------- жизненный цикл ----------

    def start(self) -> None:
        self.media.start()
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
        log.info("pipeline started: %d cameras, backend %s, fallbacks %s", len(cams), self.backend.name, compat.FALLBACKS)

    def stop(self) -> None:
        with self._lock:
            for cid in list(self.workers):
                self._stop_worker(cid)
        for sched in self._schedulers.values():
            sched.stop()
        self.media.stop()
        close = getattr(self.journal, "close", None)
        if callable(close):
            close()
        close = getattr(self.backend, 'close', None)
        if callable(close):
            close()

    # ---------- модели ----------

    def get_detector(self, model_id: str) -> Detector:
        with self._det_lock:
            det = self._detectors.get(model_id)
            if det is not None:
                return det
            spec = self.models.get(model_id)
            if spec is None:
                raise KeyError(f"model {model_id!r} is not in registry {self.settings.models_config}")
            self.backend.load(spec)
            if not spec.classes:
                names = self.backend.class_names(model_id)
                if names:
                    spec = spec.model_copy(update={"classes": names})
            scheduler = None
            if self.settings.batching and self.backend.max_batch_size(model_id) > 1:
                scheduler = BatchScheduler(
                    self.backend, model_id, min(self.settings.max_batch, self.backend.max_batch_size(model_id)),
                    self.settings.batch_timeout_ms / 1000.0,
                )
                self._schedulers[model_id] = scheduler
            det = self._detectors[model_id] = Detector(spec, self.backend, scheduler)
            return det

    # ---------- камеры ----------

    def _start_worker(self, cam: CameraConfig) -> None:
        from guard.pipeline.worker import CameraWorker

        worker = CameraWorker(cam, self)
        self.workers[cam.camera_id] = worker
        worker.start()
        if worker.errors:
            log.warning("camera %s started with errors: %s", cam.camera_id, worker.errors)

    def _stop_worker(self, camera_id: str) -> None:
        worker = self.workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()
        self._close_open_events(camera_id)

    def _close_open_events(self, camera_id: str) -> None:
        """Перед перезапуском камеры закрываем её открытые события и забываем состояние окна."""
        open_events = getattr(self.events, "open_events", None)
        events = list(open_events(camera_id)) if open_events else []
        now = time.time()
        for ev in events:
            updated = self._journal_update(ev.id, ended_at=now)
            if updated is not None:  # UI узнаёт, что событие закрыто (иначе висит «открытым» до перезагрузки списка)
                self.bus.publish(event_message("event_update", updated))
        reset = getattr(self.events, "reset", None)
        if reset:
            reset(camera_id)

    def validate_camera(self, cam: CameraConfig) -> None:
        if not cam.camera_id.strip():
            raise CameraError("camera_id must not be empty")
        if not 0 < cam.sample_fps <= 30:
            raise CameraError("sample_fps must be in (0, 30]")
        zone_ids = [z.zone_id for z in cam.zones]
        if len(set(zone_ids)) != len(zone_ids):
            raise CameraError("zone_id values must be unique")
        for z in cam.zones:
            if len(z.polygon) < 3:
                raise CameraError(f"zone {z.zone_id}: polygon needs at least 3 points")
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in z.polygon):
                raise CameraError(f"zone {z.zone_id}: polygon coordinates must be normalized to 0..1")
            try:
                validate_active_hours(z.active_hours)
            except ValueError as exc:
                raise CameraError(f"zone {z.zone_id}: {exc}") from exc
        known = {r["type"] for r in available_rules()}
        rule_ids = [r.rule_id for r in cam.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise CameraError("rule_id values must be unique within a camera")
        for r in cam.rules:
            if r.type not in known:
                raise CameraError(f"rule {r.rule_id}: unknown type {r.type!r}; available: {sorted(known)}")
            missing = set(r.zones or []) - set(zone_ids)
            if missing:
                raise CameraError(f"rule {r.rule_id}: unknown zones {sorted(missing)}")
            if r.min_duration_s < 0 or not 0 <= r.min_ratio <= 1:
                raise CameraError(f"rule {r.rule_id}: min_duration_s >= 0 and 0 <= min_ratio <= 1 required")

    def _persist(self) -> None:
        save_cameras(self.settings.cameras_config, self.cameras.values())

    def list_cameras(self) -> list[CameraConfig]:
        with self._lock:
            return list(self.cameras.values())

    def get_camera(self, camera_id: str) -> CameraConfig | None:
        with self._lock:
            return self.cameras.get(camera_id)

    def add_camera(self, cam: CameraConfig) -> CameraConfig:
        cam = with_default_severity(cam)
        self.validate_camera(cam)
        with self._lock:
            if cam.camera_id in self.cameras:
                raise CameraExists(cam.camera_id)
            self.cameras[cam.camera_id] = cam
            self._persist()
            if cam.enabled:
                self._start_worker(cam)
        return cam

    def update_camera(self, camera_id: str, cam: CameraConfig) -> CameraConfig:
        if cam.camera_id != camera_id:
            raise CameraError("camera_id in body must match the path")
        cam = with_default_severity(cam)
        self.validate_camera(cam)
        with self._lock:
            if camera_id not in self.cameras:
                raise KeyError(camera_id)
            self._stop_worker(camera_id)
            self.cameras[camera_id] = cam
            self._persist()
            if cam.enabled:
                self._start_worker(cam)  # горячий перезапуск только этой камеры
        return cam

    def delete_camera(self, camera_id: str) -> None:
        with self._lock:
            if camera_id not in self.cameras:
                raise KeyError(camera_id)
            self._stop_worker(camera_id)
            del self.cameras[camera_id]
            self._persist()
        self.media.remove_camera(camera_id)
        self.metrics.remove(camera_id)

    def camera_status(self, camera_id: str) -> dict[str, Any]:
        worker = self.workers.get(camera_id)
        if worker is None:
            return {"running": False, "fps_in": 0.0, "fps_processed": 0.0, "lag_s": 0.0, "errors": {}}
        return worker.status()

    # ---------- события ----------

    def handle_events(
        self, worker: CameraWorker, ctx: FrameContext, new: list[Event], closed: list[Event], t_start: float
    ) -> None:
        """Новое событие: снапшот → журнал → WS → клип в фоне. Закрытое: ended_at → журнал → WS."""
        for ev in new:
            t0 = time.perf_counter()
            try:
                ev.snapshot_path = self.media.save_snapshot(ev, self._render_snapshot(ctx, ev))
            except Exception:  # noqa: BLE001 — без снапшота оповещение всё равно уходит
                log.exception("snapshot for event %s failed", ev.id)
            latency = dict(ev.latency_ms or ctx.timings)
            now = time.perf_counter()
            latency["alert"] = (now - t0) * 1000.0
            latency["total"] = ctx.timings.get("decode", 0.0) + (now - t_start) * 1000.0
            ev.latency_ms = {k: round(float(v), 2) for k, v in latency.items()}
            ev.details = {
                **ev.details,
                "confirm_window_s": round(ev.confirmed_at - ev.started_at, 3),
                "camera_name": worker.cfg.name,
            }
            try:
                self.journal.add(ev)
            except Exception:  # noqa: BLE001
                log.exception("journal add failed for %s", ev.id)
            self.events_created += 1
            self.bus.publish(event_message("event", ev))
            log.warning(
                "EVENT %s %s camera=%s zone=%s track=%s window=%.2fs pipeline=%.0fms",
                ev.type, ev.id, ev.camera_id, ev.zone_id, ev.track_id,
                ev.confirmed_at - ev.started_at, ev.latency_ms.get("total", 0.0),
            )
            if not self.media.schedule_clip(ev, on_done=self._on_clip):
                self._on_clip(ev, None)  # очередь клипов полна — помечаем clip_error, иначе в UI клип «грузится» вечно
        for ev in closed:
            updated = self._journal_update(ev.id, ended_at=ev.ended_at)
            if updated is not None:
                self.bus.publish(event_message("event_update", updated))

    def _render_snapshot(self, ctx: FrameContext, ev: Event):
        img = ctx.frame.copy()
        zones = [z for z in ctx.camera.zones if ev.zone_id is None or z.zone_id == ev.zone_id]
        compat.draw_zones(img, zones)
        label = f"{ev.type} {ev.zone_id or ''} #{ev.track_id if ev.track_id is not None else '-'}"
        out = compat.draw_event_snapshot(img, ev.bbox, label)
        return img if out is None else out

    def _journal_update(self, event_id: str, **fields: Any) -> Event | None:
        try:
            return self.journal.update(event_id, **fields)
        except KeyError:
            return None
        except Exception:  # noqa: BLE001
            log.exception("journal update failed for %s", event_id)
            return None

    def _on_clip(self, ev: Event, rel: str | None) -> None:
        if rel:
            updated = self._journal_update(ev.id, clip_path=rel)
        else:
            current = self.journal.get(ev.id)
            details = {**(current.details if current else ev.details), "clip_error": True}
            updated = self._journal_update(ev.id, details=details)
        if updated is not None:
            self.bus.publish(event_message("event_update", updated))

    def patch_event(self, event_id: str, **fields: Any) -> Event | None:
        if not fields:
            return self.journal.get(event_id)
        updated = self._journal_update(event_id, **fields)
        if updated is not None:
            self.bus.publish(event_message("event_update", updated))
        return updated

    # ---------- сводки ----------

    def _health_base(self) -> dict[str, Any]:
        running = sum(1 for w in self.workers.values() if w.running)
        return {
            "status": "ok",
            "backend": self.backend.name,
            "cameras_running": running,
            "cameras_total": len(self.cameras),
            "uptime_s": round(time.time() - self.started_at, 1),
            "fallbacks": {k: v for k, v in compat.FALLBACKS.items() if v},
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            result = self._health_base()
            workers = [w for w in self.workers.values() if w.running]
            camera_errors = [
                (dict(w.errors), getattr(w, "last", None), bool(getattr(w, "detectors", {})))
                for w in workers
            ]
        stats_error = None
        try:
            stats = dict(self.backend.stats())
        except Exception as exc:  # noqa: BLE001
            stats = {}
            stats_error = f"{type(exc).__name__}: {exc}"[:300]
        active = stats.get("active") or self.backend.name
        fallback = bool(stats.get("fallback", active != self.backend.name))
        circuit_open = bool(stats.get("circuit_open", False))
        last_error = stats.get("last_error") or stats_error
        now = time.monotonic()
        recent_errors = []
        missing_models = []
        for errors, last, has_detectors in camera_errors:
            # worker.errors хранит текущие ошибки, а не их историю.
            # done_mono позволяет исключить старые ошибки при остановившемся источнике.
            done_mono = getattr(last, "done_mono", None)
            recent = done_mono is None or now - done_mono <= 30.0
            infer_errors = [value for key, value in errors.items() if key.startswith("infer:")]
            recent_errors.append(bool(infer_errors) and recent)
            missing_models.append(
                not has_detectors and any(key.startswith("model:") for key in errors)
            )
            if last_error is None:
                if recent and infer_errors:
                    last_error = infer_errors[0]
                elif missing_models[-1]:
                    last_error = next(value for key, value in errors.items() if key.startswith("model:"))
        unavailable = (
            stats.get("available") is False
            or bool(stats.get("closed", False))
            or (circuit_open and not fallback)
            or (bool(missing_models) and all(missing_models))
        )
        degraded = (
            fallback
            or stats_error is not None
            or (bool(recent_errors) and all(recent_errors))
        )
        result["status"] = "error" if unavailable else "degraded" if degraded else "ok"
        result["inference"] = {
            "active": active,
            "circuit_open": circuit_open,
            "last_error": last_error,
        }
        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        cams = self.metrics.snapshot()
        out: dict[str, Any] = {}
        for cid in self.cameras:
            snap = cams.get(cid)
            if snap is None:
                continue
            out[cid] = {
                "fps_in": snap["fps_in"],
                "fps_processed": snap["fps_processed"],
                "lag_s": snap["lag_s"],
                "timings_ms": {k: {"p50": v["p50"], "p95": v["p95"]} for k, v in snap["timings_ms"].items()},
                "frames_processed": snap["frames_processed"],
                "errors": snap["errors"],
            }
        try:
            inference = dict(self.backend.stats())
        except Exception as exc:  # noqa: BLE001
            inference = {"backend": self.backend.name, "error": str(exc)}
        inference.setdefault("backend", self.backend.name)
        inference.setdefault("device", "tpu" if "tpu" in self.backend.name else "cpu")
        inference.setdefault("utilization", None)
        models = inference.setdefault("models", {})
        with self._det_lock:
            detectors = list(self._detectors.items())
        for model_id, det in detectors:
            q = det.queue_stats.summary()
            entry = models.setdefault(model_id, {})
            entry.setdefault("timings_ms", {})["queue"] = {"p50": q["p50"], "p95": q["p95"]}
            entry["batching"] = det.scheduler is not None
        return {"backend": self.backend.name, "cameras": out, "inference": inference, "media": self.media.stats()}
