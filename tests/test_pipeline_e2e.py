"""CPU integration: synthetic source/inference, otherwise the real application.

Frames are processed explicitly, so no camera threads or wall-clock sleeps are
needed. Media jobs finish as source progress reaches the post-event boundary;
stop() joins the real writers before checking their asynchronous clip update.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import anyio
import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, RuleConfig, SourceConfig, Zone
from guard.events.journal import Journal
from guard.events.manager import EventManager
from guard.inference.base import InferenceBackend
from guard.pipeline import compat
from guard.pipeline.detector import Detector
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.settings import Settings
from guard.pipeline.worker import CameraWorker
from guard.rules.plugins.zone_intrusion import ZoneIntrusionRule
from guard.sources.base import FrameSource
from guard.tracking import Tracker


INSIDE = (44, 24, 84, 96)
OUTSIDE = (2, 24, 22, 96)
FRAME_SIZE = 128


class SyntheticBackend(InferenceBackend):
    """Replace inference only: Detector still runs real letterbox/YOLO decoding."""

    name = "synthetic-cpu"

    def __init__(self):
        self.boxes = []
        self.loads = []
        self.calls = 0

    def load(self, spec):
        assert spec.model_id == "det"
        assert spec.classes == ["person"]
        self.loads.append(spec.model_id)

    def infer(self, model_id, batch):
        assert model_id == "det"
        assert batch.shape == (1, 3, FRAME_SIZE, FRAME_SIZE)
        assert batch.dtype == np.float32
        assert batch.flags.c_contiguous
        # The real preprocessor must swap BGR to RGB and normalize pixels.
        np.testing.assert_allclose(batch[0, :, 0, 0], np.array([56, 34, 12]) / 255, rtol=1e-6)
        self.calls += 1
        raw = np.empty((1, 5, len(self.boxes)), dtype=np.float32)
        for index, (left, top, right, bottom) in enumerate(self.boxes):
            raw[0, :, index] = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top, 0.95]
        return [raw]

    def stats(self):
        return {"backend": self.name, "models": {"det": {"calls": self.calls, "errors": 0}}}


class ManualSource(FrameSource):
    """Use real source publication/callbacks, but never start a source thread."""

    kind = "synthetic"

    def start(self):
        raise AssertionError("E2E tests must not start camera threads")

    def emit(self, frame, ts):
        self.connected = True
        self._publish(frame, decode_ms=0.25, ts=ts)
        return self.latest()


@dataclass
class Pipeline:
    engine: PipelineEngine
    worker: CameraWorker
    backend: SyntheticBackend
    base_ts: float
    barrier_seq: int = 0

    def frame(self, seconds, boxes):
        self.backend.boxes = list(boxes)
        frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), (12, 34, 56), dtype=np.uint8)
        for left, top, right, bottom in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (90, 120, 150), -1)
        packet = self.worker.source.emit(frame, self.base_ts + seconds)
        context = self.worker.process(packet)
        assert self.worker.errors == {}
        return context

    def messages(self, ws):
        """Drain through a FIFO bus marker; no sleeps or empty-queue races."""
        self.barrier_seq += 1
        marker = {"kind": "e2e_barrier", "sequence": self.barrier_seq}
        self.engine.bus.publish(marker)
        messages = []
        while True:
            # Starlette's public receive_json has no timeout. Read the same
            # real ASGI stream with a bounded wait, so a broken WS fails fast.
            async def receive():
                with anyio.fail_after(3):
                    return await ws._send_rx.receive()

            envelope = ws.portal.call(receive)
            assert envelope["type"] == "websocket.send", envelope
            message = json.loads(envelope["text"])
            if message == marker:
                return messages
            if message["kind"] != "metrics":
                messages.append(message)

    def confirm(self, client, ws):
        contexts = [self.frame(second, [INSIDE]) for second in range(3)]
        assert contexts[0].people == []  # real tracker's min_hits=2
        assert len(contexts[1].people) == len(contexts[2].people) == 1
        assert client.get("/api/events").json() == {"items": [], "total": 0}
        assert self.messages(ws) == []
        context = self.frame(3, [INSIDE])
        messages = self.messages(ws)
        assert len(messages) == 1
        assert messages[0]["kind"] == "event"
        event = messages[0]["event"]
        assert event["track_id"] == context.people[0].track_id
        assert event["started_at"] == self.base_ts + 1
        assert event["confirmed_at"] == self.base_ts + 3
        assert event["details"]["confirm_window_s"] == 2
        return event

    def complete_clip(self, client, ws, event_id):
        """Join the real encoder after post frames, then audit its sole update."""
        self.engine.media.stop()
        assert self.engine.media.clips_written == 1
        assert self.engine.media.clip_errors == 0
        ready = client.get(f"/api/events/{event_id}").json()
        assert ready["clip_path"].endswith(".mp4")
        assert self.messages(ws) == [{"kind": "event_update", "event": ready}]
        return ready


def event_page(client):
    response = client.get("/api/events", params={"type": "zone_intrusion"})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models",
        cameras_config=tmp_path / "cameras.yaml", models_config=tmp_path / "models.yaml",
        db_path=tmp_path / "data" / "guard.db", media_dir=tmp_path / "data" / "media",
        frontend_dist=tmp_path / "no-frontend", autostart=False, batching=False,
        tracker_min_hits=2, tracker_buffer_s=1,
        clip_pre_s=1, clip_post_s=2, clip_fps=2, clip_max_width=FRAME_SIZE, clip_buffer_mb=1,
    )
    settings.models_config.write_text(
        "models:\n  det:\n    onnx_path: unused-synthetic-model\n    input_size: [128, 128]\n    classes: [person]\n",
        encoding="utf-8",
    )
    cfg = CameraConfig(
        camera_id="synthetic-camera", name="Тестовая камера", site="Тестовый цех",
        source=SourceConfig(type="images", uri=str(tmp_path / "synthetic-input")), sample_fps=1,
        zones=[Zone(
            zone_id="danger", name="Опасная зона", kind="danger",
            polygon=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.9), (0.25, 0.9)],
        )],
        rules=[RuleConfig(
            rule_id="intrusion", type="zone_intrusion", zones=["danger"],
            min_duration_s=2, min_ratio=1, close_after_s=2, cooldown_s=30,
        )],
    )

    def make_source(config, uri, camera_id, *, on_frame, fps):
        assert fps == cfg.sample_fps
        return ManualSource(config, uri, camera_id, on_frame=on_frame)

    monkeypatch.setattr("guard.pipeline.worker.make_source", make_source)
    backend = SyntheticBackend()
    engine = PipelineEngine(settings, backend=backend)
    engine.validate_camera(cfg)
    engine.cameras[cfg.camera_id] = cfg
    worker = CameraWorker(cfg, engine)
    engine.workers[cfg.camera_id] = worker
    assert isinstance(worker.detectors["det"], Detector)
    assert isinstance(worker.person_tracker, Tracker)
    assert isinstance(worker.rules[0], ZoneIntrusionRule)
    assert isinstance(engine.events, EventManager)
    assert isinstance(engine.journal, Journal)
    assert not any(compat.FALLBACKS.values())
    assert backend.loads == ["det"]

    # Future timestamps prevent a wall timeout; source progress readies the clip.
    system = Pipeline(engine, worker, backend, time.time() + 86400)
    with TestClient(create_app(settings, engine)) as client:
        with client.websocket_connect("/ws/events") as ws:
            assert ws.portal.call(lambda: engine.bus.subscribers) == 1
            yield system, client, ws
    assert engine.bus.subscribers == 0
    assert not worker.running


def test_worker_event_journal_http_websocket_and_real_media(pipeline):
    system, client, ws = pipeline
    engine = system.engine
    event = system.confirm(client, ws)
    event_id = event["id"]
    assert event["camera_id"] == "synthetic-camera"
    assert event["zone_id"] == "danger"
    assert event["type"] == "zone_intrusion"
    assert event["site"] == "Тестовый цех"
    assert event["status"] == "new"
    assert event["details"]["foot_xy"] == [64, 96]
    assert event["details"]["camera_name"] == "Тестовая камера"
    assert event["latency_ms"]["decode"] == 0.25
    assert all(value >= 0 for value in event["latency_ms"].values())
    assert event_page(client) == {"items": [event], "total": 1}
    assert client.get(f"/api/events/{event_id}").json() == event
    assert engine.events_created == 1

    snapshot = client.get(f"/api/events/{event_id}/snapshot.jpg")
    assert snapshot.status_code == 200
    assert snapshot.headers["content-type"] == "image/jpeg"
    assert snapshot.content.startswith(b"\xff\xd8")
    image = cv2.imdecode(np.frombuffer(snapshot.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert snapshot.content == engine.media.resolve(event["snapshot_path"]).read_bytes()
    assert client.get(f"/api/events/{event_id}/clip.mp4").status_code == 404

    for second in (4, 5, 6):
        context = system.frame(second, [INSIDE])
        assert context.people[0].track_id == event["track_id"]
    # Streaming completion is independent of incident closure and may publish
    # at t5. Join before the FIFO marker so this is not a thread scheduling race.
    clip_ready = system.complete_clip(client, ws, event_id)
    assert clip_ready["ended_at"] is None
    assert clip_ready["status"] == "new" and clip_ready["operator_note"] is None
    assert event_page(client) == {"items": [clip_ready], "total": 1}

    patched = client.patch(f"/api/events/{event_id}", json={
        "status": "confirmed", "operator_note": "Проверено оператором",
    })
    assert patched.status_code == 200
    assert patched.json()["status"] == "confirmed"
    assert system.messages(ws) == [{"kind": "event_update", "event": patched.json()}]
    assert engine.journal.get(event_id).operator_note == "Проверено оператором"

    system.frame(7, [])
    system.frame(8, [])
    assert system.messages(ws) == []
    system.frame(9, [])
    closed = client.get(f"/api/events/{event_id}").json()
    assert closed["ended_at"] == system.base_ts + 6
    assert closed["status"] == "confirmed"
    assert closed["operator_note"] == "Проверено оператором"
    assert system.messages(ws) == [{"kind": "event_update", "event": closed}]
    system.frame(10, [])
    assert system.messages(ws) == []
    assert engine.events.open_events() == []

    # Repeated stop is harmless; the clip update was already audited while open.
    engine.media.stop()
    assert engine.media.clips_written == 1
    assert engine.media.clip_errors == 0
    ready = client.get(f"/api/events/{event_id}").json()
    assert ready["clip_path"].endswith(".mp4")
    assert ready["ended_at"] == closed["ended_at"]
    assert ready["operator_note"] == closed["operator_note"]
    assert system.messages(ws) == []
    clip_path = engine.media.resolve(ready["clip_path"])
    clip = client.get(f"/api/events/{event_id}/clip.mp4")
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/mp4"
    assert clip.content == clip_path.read_bytes()
    assert b"ftyp" in clip.content[:32]
    frames = imageio_ffmpeg.read_frames(str(clip_path))
    try:
        metadata = next(frames)
        assert metadata["codec"] == "h264"
        assert metadata["size"] == (FRAME_SIZE, FRAME_SIZE)
        assert metadata["fps"] == pytest.approx(engine.settings.clip_fps)
        assert len(next(frames)) == FRAME_SIZE * FRAME_SIZE * 3
        decoded_frames = 1 + sum(1 for _ in frames)
        source_span = event["confirmed_at"] + engine.settings.clip_post_s - (event["started_at"] - engine.settings.clip_pre_s)
        assert decoded_frames == round(source_span * engine.settings.clip_fps) + 1
        assert metadata["duration"] == pytest.approx(decoded_frames / engine.settings.clip_fps, abs=.1)
    finally:
        frames.close()
    assert event_page(client) == {"items": [ready], "total": 1}
    assert system.backend.calls == 11
    metrics = client.get("/api/metrics").json()
    assert metrics["cameras"]["synthetic-camera"]["frames_processed"] == 11
    assert metrics["cameras"]["synthetic-camera"]["errors"] == 0
    # Independent readback also proves the final update was committed to SQLite.
    with Journal(engine.settings.db_path) as readback:
        assert readback.get(event_id).model_dump(mode="json") == ready
        assert readback.query(limit=0)[1] == 1


@pytest.mark.parametrize("scenario", ["outside-zone", "no-detections", "transient-entry"])
def test_worker_does_not_report_nonviolations_or_short_entries(pipeline, scenario):
    system, client, ws = pipeline
    for second in range(8):
        boxes = [OUTSIDE] if scenario == "outside-zone" else []
        if scenario == "transient-entry" and second < 2:
            boxes = [INSIDE]
        context = system.frame(second, boxes)
        if scenario == "outside-zone" and second > 0:
            assert len(context.people) == 1
    assert system.backend.calls == 8
    assert event_page(client) == {"items": [], "total": 0}
    assert system.messages(ws) == []
    assert system.engine.events_created == 0
    assert system.engine.events.open_events() == []
    assert system.engine.media.clips_written == 0
    assert not list(system.engine.settings.media_dir.rglob("*.jpg"))


def test_worker_track_id_change_keeps_one_event_until_violation_closes(pipeline):
    system, client, ws = pipeline
    original = system.confirm(client, ws)
    system.frame(4, [])
    assert system.frame(5, [INSIDE]).people == []  # old ID expired; new track is tentative
    clip_ready = system.complete_clip(client, ws, original["id"])
    assert clip_ready["ended_at"] is None
    changed = system.frame(6, [INSIDE]).people[0]
    assert changed.track_id != original["track_id"]
    reopened = client.get(f"/api/events/{original['id']}").json()
    assert reopened["ended_at"] is None
    assert reopened["track_id"] == original["track_id"]
    assert reopened["started_at"] == original["started_at"] and reopened["confirmed_at"] == original["confirmed_at"]
    assert reopened["details"]["reopened_count"] == 1
    assert reopened["details"]["active_duration_s"] == 2
    assert reopened["clip_path"] == clip_ready["clip_path"]
    assert system.messages(ws) == [{"kind": "event_update", "event": reopened}]
    assert event_page(client)["total"] == 1
    assert system.engine.events_created == 1
    assert [event.id for event in system.engine.events.open_events()] == [original["id"]]
    for second in (7, 8, 9):
        system.frame(second, [])
    closed = client.get(f"/api/events/{original['id']}").json()
    assert closed["ended_at"] == system.base_ts + 6
    assert system.messages(ws) == [{"kind": "event_update", "event": closed}]
    assert system.engine.events_created == 1
