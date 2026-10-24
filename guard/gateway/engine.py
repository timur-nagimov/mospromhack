"""Lightweight gateway: remote events/previews, SQLite and the operator API only."""

from __future__ import annotations

import copy
import hashlib
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from guard.contracts import CameraConfig
from guard.events.journal import Journal
from guard.flywheel.edge import receive_sample
from guard.flywheel.wiring import attach_gateway_flywheel, operator_decision, start_flywheel, stop_flywheel
from guard.gateway.frames import RemoteCameraView
from guard.pipeline.bus import EventBus
from guard.pipeline.errors import CameraReadOnly
from guard.pipeline.settings import Settings
from guard.transport.protocol import Envelope, decode, snapshot_bytes
from guard.transport.rabbit import RabbitConfig, RabbitConsumer


class GatewayMedia:
    """Only gateway-owned media paths are readable through the API."""

    def __init__(self, data_dir: Path, media_dir: Path):
        self.data_dir, self.media_dir = data_dir.resolve(), media_dir.resolve()
        if self.media_dir != self.data_dir and self.data_dir not in self.media_dir.parents:
            raise ValueError("Gateway media directory must be inside data directory")

    def resolve(self, relative: str | None) -> Path | None:
        if not relative or Path(relative).is_absolute() or "\\" in relative or ".." in Path(relative).parts:
            return None
        path = (self.data_dir / relative).resolve()
        if self.media_dir not in path.parents:
            return None
        return path if path.is_file() else None

    def save(self, envelope: Envelope) -> str | None:
        data = snapshot_bytes(envelope)
        if data is None:
            return None
        if len(data) > 1_048_576:
            raise ValueError("Snapshot exceeds limit")
        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if image.format != "JPEG" or not 1 <= width <= 4096 or not 1 <= height <= 4096 or width * height > 16_777_216:
                    raise ValueError("Invalid snapshot dimensions or format")
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid snapshot JPEG") from exc
        # A stale revision can create an unused file, but cannot overwrite the
        # image referenced by a newer committed revision of the event.
        identity = hashlib.sha256((envelope.message_id + ":").encode() + data).hexdigest()
        path = self.media_dir / "remote" / f"{identity}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(f".{identity}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path.relative_to(self.data_dir).as_posix()


class GatewayEngine:
    def __init__(
        self, settings: Settings, *, journal: Journal | None = None, consumer=None,
        clock: Callable[[], float] = time.monotonic, wall_clock: Callable[[], float] = time.time,
    ):
        if settings.mode != "gateway":
            raise ValueError("GatewayEngine requires gateway mode")
        self.settings = settings
        self._clock, self._wall_clock = clock, wall_clock
        self.media = GatewayMedia(settings.data_dir, settings.media_dir)
        self.journal = journal if journal is not None else Journal(settings.db_path)
        try:
            if consumer is None:
                if not settings.rabbit_url:
                    raise ValueError("GUARD_RABBIT_URL is required")
                consumer = RabbitConsumer(RabbitConfig(settings.rabbit_url, settings.edge_id, suffix=settings.rabbit_suffix))
        except BaseException:
            self.journal.close()
            raise
        self.consumer = consumer
        self.bus = EventBus()
        self.cameras: dict[str, CameraConfig] = {}
        self.workers: dict[str, RemoteCameraView] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._producer: str | None = None
        self._run: str | None = None
        self._retired_producers: set[str] = set()
        self._retired_runs: set[tuple[str, str]] = set()
        self._catalogue_sequence = 0
        self._producer_sequence = 0
        self._metrics_sequence = 0
        self._metrics_at: float | None = None
        self._metrics: dict[str, Any] = {}
        self._captures: dict[str, str] = {}
        self._retired_captures: dict[str, set[str]] = {}
        self._closed = False
        attach_gateway_flywheel(self)

    def start(self):
        if self._closed:
            raise RuntimeError("Gateway is closed")
        if self._thread is None:
            start_flywheel(self)
            self._thread = threading.Thread(
                target=self.consumer.run, args=(self.receive, self._stop), name="gateway-replication", daemon=True,
            )
            try:
                self._thread.start()
            except BaseException:
                if not self._thread.is_alive():
                    self._thread = None
                raise

    def stop(self):
        if self._closed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise RuntimeError("Gateway consumer did not stop")
        else:
            try:
                self.consumer.close()
            finally:
                stop_flywheel(self)
                self.journal.close()
                self._closed = True
            return
        stop_flywheel(self)
        self.journal.close()
        self._closed = True

    def receive(self, envelope: Envelope | bytes | str):
        if not isinstance(envelope, Envelope):
            envelope = decode(envelope, expected_edge_id=self.settings.edge_id)
        if envelope.edge_id != self.settings.edge_id:
            raise ValueError("Unexpected edge identity")
        if envelope.kind == "flywheel_sample":
            # Data-flywheel sample parts are not previews: no staleness or run checks (docs/FLYWHEEL.md).
            return receive_sample(self, envelope)
        if envelope.kind in {"event", "event_update"}:
            relative = self.media.save(envelope)
            result = self.journal.apply_remote_event(
                envelope, snapshot_path=relative, expected_edge_id=self.settings.edge_id,
            )
            if result.changed:
                self.bus.publish({"kind": "event" if result.created else "event_update", "event": result.event.model_dump(mode="json")})
            return result
        # Durable history is accepted independently from preview authority. In
        # particular, old replayed events must never reactivate an old stream.
        with self._lock:
            age = self._wall_clock() - envelope.sent_at
            if not -self.settings.preview_stale_after_s <= age <= self.settings.preview_stale_after_s:
                return
            if envelope.kind == "cameras":
                self._catalogue(envelope)
                return
            if (envelope.producer_id, envelope.run_id) != (self._producer, self._run):
                return
            self._producer_sequence = max(self._producer_sequence, envelope.sequence)
            if envelope.kind == "metrics":
                if envelope.sequence <= self._metrics_sequence:
                    return
                self._metrics = copy.deepcopy(envelope.payload["data"])
                self._metrics_at = self._clock()
                self._metrics_sequence = envelope.sequence
            elif envelope.kind == "frame_meta":
                self._frame(envelope)

    def _catalogue(self, envelope: Envelope):
        producer, run = envelope.producer_id, envelope.run_id
        if producer in self._retired_producers or (producer, run) in self._retired_runs:
            return
        changing = (producer, run) != (self._producer, self._run)
        if producer == self._producer:
            if envelope.sequence <= (self._producer_sequence if changing else self._catalogue_sequence):
                return
        cameras = [CameraConfig.model_validate(item) for item in envelope.payload["cameras"]]
        if len({camera.camera_id for camera in cameras}) != len(cameras):
            raise ValueError("Duplicate remote camera")
        for camera in cameras:
            if camera.source.uri != f"edge://{self.settings.edge_id}/{camera.camera_id}":
                raise ValueError("Remote camera URI must be sanitized")
        if changing:
            if self._producer is not None:
                if producer != self._producer:
                    self._retired_producers.add(self._producer)
                self._retired_runs.add((self._producer, self._run))
            for worker in self.workers.values():
                worker.invalidate()
            self.workers = {}
            self._metrics, self._metrics_at = {}, None
            self._metrics_sequence = 0
            self._captures, self._retired_captures = {}, {}
        self._producer, self._run = producer, run
        self._catalogue_sequence = envelope.sequence
        self._producer_sequence = max(self._producer_sequence, envelope.sequence) if not changing else envelope.sequence
        self.cameras = {camera.camera_id: camera for camera in cameras}
        self.workers = {
            camera_id: self.workers.get(camera_id) or RemoteCameraView(camera_id, self.settings.preview_stale_after_s, self._clock, wall_clock=self._wall_clock)
            for camera_id in self.cameras
        }

    def _frame(self, envelope: Envelope):
        camera_id = envelope.payload["camera_id"]
        worker = self.workers.get(camera_id)
        if worker is None:
            return
        capture = envelope.payload["capture_id"]
        retired = self._retired_captures.setdefault(camera_id, set())
        if capture in retired:
            return
        if worker.update(envelope.payload, sequence=envelope.sequence, run_id=envelope.run_id, producer_id=envelope.producer_id):
            previous = self._captures.get(camera_id)
            if previous is not None and previous != capture:
                retired.add(previous)
            self._captures[camera_id] = capture

    def list_cameras(self):
        with self._lock:
            return list(self.cameras.values())

    def get_camera(self, camera_id):
        with self._lock:
            return self.cameras.get(camera_id)

    def camera_status(self, camera_id):
        with self._lock:
            worker = self.workers.get(camera_id)
            return worker.status() if worker else {"running": False, "errors": {"edge": "Waiting for edge catalogue"}}

    def add_camera(self, camera):
        raise CameraReadOnly("Configure cameras on the edge")

    def update_camera(self, camera_id, camera):
        raise CameraReadOnly("Configure cameras on the edge")

    def delete_camera(self, camera_id):
        raise CameraReadOnly("Configure cameras on the edge")

    def patch_event(self, event_id, **fields):
        try:
            event = self.journal.update(event_id, **fields)
        except KeyError:
            return None
        if "status" in fields:
            operator_decision(self, event_id, fields["status"])
        self.bus.publish({"kind": "event_update", "event": event.model_dump(mode="json")})
        return event

    def health(self):
        with self._lock:
            return {
                "status": "ok", "mode": "gateway", "backend": "remote-edge", "edge_id": self.settings.edge_id,
                "cameras_running": sum(worker.running for worker in self.workers.values()),
                "cameras_total": len(self.cameras), "transport": self.consumer.status(),
            }

    def metrics_snapshot(self):
        with self._lock:
            fresh = self._metrics_at is not None and self._clock() - self._metrics_at <= self.settings.preview_stale_after_s
            if fresh:
                data = copy.deepcopy(self._metrics)
            else:
                data = {"backend": "remote-edge", "cameras": {}, "inference": {}}
            data["mode"] = "gateway"
            data["transport"] = self.consumer.status()
            data["stale"] = not fresh
            return data
