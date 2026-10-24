"""Edge pipeline with separate durable-event and disposable-telemetry channels."""

from __future__ import annotations

import base64
import io
import logging
import math
import sqlite3
import threading
import time
from typing import Callable

import cv2
from PIL import Image

from guard.events.journal import Journal
from guard.pipeline.engine import PipelineEngine, event_message
from guard.transport.store import IDENTITY_FIELDS
from guard.transport.rabbit import RabbitConfig, RabbitPublisher

log = logging.getLogger(__name__)


def _safe_status(value):
    """Runtime/source errors can embed credentials; publish only generic errors."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "errors" and isinstance(item, dict):
                result[key] = {"processing": "Processing error"} if item else {}
            elif key in {"last_error", "error"}:
                result[key] = "Processing error" if item else None
            elif key == "uri":
                result[key] = "Details hidden"
            else:
                result[key] = _safe_status(item)
        return result
    if isinstance(value, (list, tuple)):  # source.frame_size — кортеж; конверт принимает только JSON-списки
        return [_safe_status(item) for item in value]
    if isinstance(value, str) and "://" in value:
        return "Details hidden"
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class EdgeRunner:
    def __init__(self, engine, *, publisher=None, telemetry_publisher=None, clock: Callable[[], float] = time.monotonic, wall_clock: Callable[[], float] = time.time):
        self.engine = engine
        self.settings = engine.settings
        self.clock = clock
        self.wall_clock = wall_clock
        if publisher is None or telemetry_publisher is None:
            if not self.settings.rabbit_url:
                raise ValueError("GUARD_RABBIT_URL is required")
            config = RabbitConfig(self.settings.rabbit_url, self.settings.edge_id, suffix=self.settings.rabbit_suffix)
            publisher = publisher if publisher is not None else RabbitPublisher(config)
            telemetry_publisher = telemetry_publisher if telemetry_publisher is not None else RabbitPublisher(config)
        self.publisher = publisher
        self.telemetry_publisher = telemetry_publisher
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._next_metrics = float("-inf")
        self._last_preview: dict[str, tuple[float, tuple]] = {}
        self.last_error: str | None = None
        self.telemetry_error: str | None = None
        self.snapshot_error: str | None = None

    def publish_pending_once(self, limit: int = 100) -> int:
        count = 0
        try:
            messages = self.engine.journal.pending_messages(limit=limit)
        except Exception as exc:
            self.last_error = type(exc).__name__
            return 0
        if not messages:
            self.last_error = None
        for message in messages:
            try:
                jpeg = None
                event = message.payload.get("event", {})
                path = self.engine.media.resolve(event.get("snapshot_path"))
                if path is not None:
                    try:
                        if path.stat().st_size <= 1_048_576:
                            jpeg = path.read_bytes()
                            with Image.open(io.BytesIO(jpeg)) as image:
                                if image.format != "JPEG" or len(jpeg) > 1_048_576 or max(image.size) > 4096:
                                    raise ValueError("Invalid snapshot")
                                image.verify()
                        else:
                            self.snapshot_error = "SnapshotTooLarge"
                    except (OSError, ValueError, Image.DecompressionBombError):
                        jpeg = None
                        self.snapshot_error = "SnapshotUnavailable"
                # This freezes the wire payload atomically before first publish,
                # including a missing image, and reuses it after uncertain ACKs.
                try:
                    message = self.engine.journal.prepare_message(message.message_id, snapshot_jpeg=jpeg)
                except ValueError:
                    if jpeg is None:
                        raise
                    # A valid image can still push the combined JSON envelope
                    # over its limit. Freeze the event without that image.
                    self.snapshot_error = "SnapshotRejected"
                    message = self.engine.journal.prepare_message(message.message_id, snapshot_jpeg=None)
                self.publisher.publish(message, durable=True)
                self.engine.journal.mark_published(message.message_id)
            except Exception as exc:
                self.last_error = type(exc).__name__
                break
            count += 1
            self.last_error = None
        return count

    def _preview(self, worker, result):
        if result is None:
            return None
        if not -self.settings.preview_stale_after_s <= self.wall_clock() - result.ts <= self.settings.preview_stale_after_s:
            return None
        height, width = result.frame.shape[:2]
        scale = min(1.0, self.settings.preview_max_width / width, 4096 / height)
        image = result.frame
        if scale < 1:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok or encoded.nbytes > 1_048_576:
            return None

        def track(item):
            return {
                "track_id": item.track_id, "cls": item.cls, "conf": item.conf,
                "bbox": [max(0.0, min(1.0, value / (width if index % 2 == 0 else height))) for index, value in enumerate(item.bbox)],
            }

        return {
            "camera_id": worker.cfg.camera_id, "capture_id": result.capture_id,
            "frame_seq": result.seq, "capture_ts": result.ts, "source_pos_s": result.source_pos_s,
            "width": image.shape[1], "height": image.shape[0], "jpeg_b64": base64.b64encode(encoded).decode("ascii"),
            "people": [track(item) for item in result.people], "objects": [track(item) for item in result.objects],
            "highlight": sorted(result.highlight), "zones": [zone.model_dump(mode="json") for zone in worker.cfg.zones],
        }

    def publish_telemetry_once(self) -> int:
        now = self.clock()
        count = 0
        try:
            # Send catalogue first, and repeat periodically to recover consumers
            # that joined after the initial (disposable) catalogue was lost.
            if now >= self._next_metrics:
                cameras = []
                for camera in self.engine.list_cameras():
                    item = camera.model_dump(mode="json")
                    item["source"]["uri"] = f"edge://{self.settings.edge_id}/{camera.camera_id}"
                    cameras.append(item)
                catalogue = {"cameras": cameras, "statuses": {
                    camera.camera_id: _safe_status(self.engine.camera_status(camera.camera_id))
                    for camera in self.engine.list_cameras()
                }}
                self.telemetry_publisher.publish(self.engine.journal.make_message("cameras", catalogue), durable=False)
                count += 1
                metrics = {"data": _safe_status(self.engine.metrics_snapshot())}
                self.telemetry_publisher.publish(self.engine.journal.make_message("metrics", metrics), durable=False)
                count += 1
                self._next_metrics = now + self.settings.telemetry_interval_s
            if not self.settings.publish_frame_meta:
                self.telemetry_error = None
                return count
            with self.engine._lock:
                workers = list(self.engine.workers.values())
            for worker in workers:
                result = worker.last
                if result is None:
                    continue
                identity = (result.capture_id, result.seq)
                previous = self._last_preview.get(worker.cfg.camera_id)
                if previous is not None and (now - previous[0] < 1 / self.settings.preview_fps or previous[1] == identity):
                    continue
                payload = self._preview(worker, result)
                if payload is None:
                    continue
                self.telemetry_publisher.publish(self.engine.journal.make_message("frame_meta", payload), durable=False)
                self._last_preview[worker.cfg.camera_id] = (now, identity)
                count += 1
            self.telemetry_error = None
        except Exception as exc:
            self.telemetry_error = type(exc).__name__
        return count

    def _run_events(self):
        backoff = 0.5
        try:
            while not self._stop.is_set():
                try:
                    count = self.publish_pending_once()
                except Exception as exc:
                    self.last_error = type(exc).__name__
                    count = 0
                if self.last_error:
                    self._stop.wait(backoff)
                    backoff = min(15.0, backoff * 2)
                else:
                    backoff = 0.5
                    if count < 100:
                        self._stop.wait(0.5)
        finally:
            self.publisher.close()

    def _run_telemetry(self):
        backoff = 0.5
        try:
            while not self._stop.is_set():
                try:
                    count = self.publish_telemetry_once()
                except Exception as exc:
                    self.telemetry_error = type(exc).__name__
                    count = 0
                if self.telemetry_error:
                    self._stop.wait(backoff)
                    backoff = min(15.0, backoff * 2)
                else:
                    if count:
                        backoff = 0.5
                    self._stop.wait(min(1 / self.settings.preview_fps, self.settings.telemetry_interval_s))
        finally:
            self.telemetry_publisher.close()

    def start(self):
        if self._threads:
            return
        for name, target in (("edge-outbox", self._run_events), ("edge-telemetry", self._run_telemetry)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self):
        self._stop.set()
        if not self._threads:
            self.publisher.close()
            self.telemetry_publisher.close()
        for thread in self._threads:
            thread.join(timeout=6)
            if thread.is_alive():
                raise RuntimeError("Edge replication thread did not stop")

    def status(self):
        return {"events": self.publisher.status(), "telemetry": self.telemetry_publisher.status(),
                "last_error": self.last_error, "telemetry_error": self.telemetry_error,
                "snapshot_error": self.snapshot_error}


class EdgeEngine(PipelineEngine):
    def __init__(self, settings, *, publisher=None, telemetry_publisher=None, **kwargs):
        if settings.mode != "edge":
            raise ValueError("EdgeEngine requires edge mode")
        try:
            super().__init__(settings, **kwargs)
            self._persistence_lock = threading.Lock()
            self._pending_persistence = {}
            self.persistence_error = None
            self.run_id = self.journal.replication_stats()["run_id"]
            self.replication = EdgeRunner(self, publisher=publisher, telemetry_publisher=telemetry_publisher)
        except BaseException:
            for resource in (getattr(self, "journal", None), getattr(self, "backend", None)):
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            raise

    def _make_journal(self):
        # Durable event publication must never silently switch to MemoryJournal.
        return Journal(self.settings.db_path, replication_edge_id=self.settings.edge_id)

    def start(self):
        super().start()
        self.replication.start()

    def _before_journal_close(self):
        if self.workers:
            # A failed per-camera cleanup must finish (including manager reset
            # and removal) on the next stop before global resources can close.
            raise RuntimeError("Camera cleanup incomplete; retry stop after recovery")
        self.retry_pending_once()
        self.replication.stop()

    def _ensure_persisted(self, kind, ev):
        current = self.journal.get(ev.id)
        if kind == "new":
            if current is None:
                try:
                    self.journal.add(ev)
                except sqlite3.Error:
                    # A commit can succeed before the caller observes an error.
                    # Unique id plus immutable identity makes its readback safe.
                    current = self.journal.get(ev.id)
                    if current is None:
                        raise
                else:
                    current = self.journal.get(ev.id)
            if any(getattr(current, field) != getattr(ev, field) for field in IDENTITY_FIELDS):
                raise ValueError("Conflicting durable event identity")
            return current
        if current is None:
            raise KeyError(ev.id)
        # EventManager follows the track after confirmation and moves its bbox;
        # durable evidence keeps the confirmation bbox on a closure update.
        if any(getattr(current, field) != getattr(ev, field) for field in IDENTITY_FIELDS if field != "bbox"):
            raise ValueError("Conflicting durable event identity")
        fields = self._lifecycle_fields(ev)
        details = fields.get("details_patch", {})
        if current.ended_at == ev.ended_at and all(current.details.get(key) == value for key, value in details.items()):
            return current
        # Partial machine update never overwrites an operator's decision.
        return self.journal.update(ev.id, **fields)

    def _remember_pending(self, kind, ev):
        key = (ev.id, kind)
        with self._persistence_lock:
            self._pending_persistence.setdefault(key, ev.model_copy(deep=True))
            return key, self._pending_persistence[key]

    def _attempt_pending(self, key, ev):
        try:
            result = self._ensure_persisted(key[1], ev)
        except Exception as exc:
            self.persistence_error = type(exc).__name__
            raise
        with self._persistence_lock:
            self._pending_persistence.pop(key, None)
            if not self._pending_persistence:
                self.persistence_error = None
        return result

    def _persist_retry(self, worker, kind, ev):
        key, prepared = self._remember_pending(kind, ev)
        backoff = 0.05
        while True:
            try:
                return self._attempt_pending(key, prepared)
            except sqlite3.Error:
                # Neither engine nor Journal locks are held during the wait.
                # This camera cannot advance EventManager and lose its revision.
                if worker._stop.wait(backoff):
                    raise RuntimeError("Edge persistence interrupted with pending revision") from None
                backoff = min(2.0, backoff * 2)

    def _persist_new_event(self, worker, ev):
        return self._persist_retry(worker, "new", ev)

    def _prepare_persistence_batch(self, worker, new, closed, reopened=()):
        for kind, events in (("new", new), ("closed", closed), ("reopened", reopened)):
            for ev in events:
                self._remember_pending(kind, ev)

    def _persist_closed_event(self, worker, ev):
        return self._persist_retry(worker, "closed", ev)

    def _persist_capture_closed_event(self, worker, ev):
        # _ensure_persisted is idempotent and clears this revision's retry
        # entry even when an earlier SQLite commit had an uncertain outcome.
        return self._persist_retry(worker, "closed", ev)

    def _persist_reopened_event(self, worker, ev):
        return self._persist_retry(worker, "reopened", ev)

    def retry_pending_once(self, camera_id=None):
        """Bounded shutdown recovery; leave failed revisions visible for retry."""
        for cid, worker in self.workers.items():
            if camera_id is not None and cid != camera_id:
                continue
            thread = getattr(worker, "_thread", None)
            if thread is not None and thread.is_alive():
                raise RuntimeError("Cannot drain persistence while camera worker is alive")
        with self._persistence_lock:
            pending = list(self._pending_persistence.items())
        for key, ev in pending:
            if camera_id is not None and ev.camera_id != camera_id:
                continue
            stored = self._attempt_pending(key, ev)
            if key[1] == "new":
                self._announce_new_event(stored, schedule_clip=False)
            else:
                self.bus.publish(event_message("event_update", stored))

    def _stop_worker(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is not None:
            worker.stop()
            thread = getattr(worker, "_thread", None)
            if thread is not None and thread.is_alive():
                raise RuntimeError("Camera worker did not stop")
        self.retry_pending_once(camera_id)
        self._close_open_events(camera_id)
        self.workers.pop(camera_id, None)
        self._worker_errors.pop(camera_id, None)

    def _close_open_events(self, camera_id):
        open_events = getattr(self.events, "open_events", None)
        for ev in list(open_events(camera_id)) if open_events else []:
            current = self.journal.get(ev.id)
            if current is not None and current.ended_at is not None:
                continue
            key, prepared = self._remember_pending("closed", ev.model_copy(update={"ended_at": time.time()}))
            stored = self._attempt_pending(key, prepared)
            self.bus.publish(event_message("event_update", stored))
        reset = getattr(self.events, "reset", None)
        if reset:
            reset(camera_id)

    def persistence_status(self):
        with self._persistence_lock:
            return {"pending": len(self._pending_persistence), "last_error": self.persistence_error}

    def health(self):
        persistence = self.persistence_status()
        parent = super().health()
        return {**parent, "status": "error" if persistence["pending"] else parent.get("status", "ok"), "mode": "edge", "replication": self.journal.replication_stats(), "transport": self.replication.status(), "persistence": persistence}

    def metrics_snapshot(self):
        return {**super().metrics_snapshot(), "mode": "edge", "replication": self.journal.replication_stats(), "persistence": self.persistence_status()}
