"""New-rule acceptance through real worker, SQLite, HTTP and ASGI WebSocket."""

import json

import anyio
import numpy as np
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, RuleConfig, Zone
from guard.events.journal import Journal
from guard.events.manager import EventManager
from guard.inference.base import InferenceBackend
from guard.pipeline import compat
from guard.pipeline.detector import Detector
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import make_settings
from guard.pipeline.worker import CameraWorker
from guard.rules.plugins.zone_dwell import ZoneDwellRule
from guard.tracking import Tracker


class SyntheticInference(InferenceBackend):
    """Only model execution is fake; preprocessing and decoding remain real."""

    name = "synthetic"

    def __init__(self):
        self.present = True
        self.calls = 0
        self.loads = []

    def load(self, spec):
        assert spec.model_id == "det" and spec.classes == ["person"]
        self.loads.append(spec.model_id)

    def infer(self, model_id, batch):
        assert model_id == "det" and batch.shape == (1, 3, 128, 128)
        assert batch.dtype == np.float32
        np.testing.assert_allclose(batch[0, :, 0, 0], np.array([56, 34, 12]) / 255, rtol=1e-6)
        self.calls += 1
        # cx/cy/w/h/person score for a fixed box [44, 24, 84, 104].
        raw = np.array([64, 64, 40, 80, .95], np.float32).reshape(1, 5, 1)
        return [raw if self.present else raw[:, :, :0]]

    def stats(self):
        return {"backend": self.name, "models": {"det": {"calls": self.calls, "errors": 0}}}


class NoMedia:
    def __init__(self, *args, **kwargs):
        pass

    def save_snapshot(self, event, frame):
        return None

    def stats(self):
        return {}

    def stop(self):
        pass


def drain_notifications(engine, ws, sequence):
    """A real bus FIFO marker proves no extra notifications, with a total bound."""
    marker = {"kind": "integration_barrier", "sequence": sequence}
    engine.bus.publish(marker)

    async def receive_through_marker():
        messages = []
        # TestClient.receive_json has no deadline; read its same ASGI stream.
        with anyio.fail_after(3):
            while True:
                envelope = await ws._send_rx.receive()
                assert envelope["type"] == "websocket.send", envelope
                message = json.loads(envelope["text"])
                if message == marker:
                    return messages
                if message["kind"] != "metrics":
                    messages.append(message)

    return ws.portal.call(receive_through_marker)


def test_zone_dwell_worker_sqlite_http_ws_operator_and_capture_boundary(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, autostart=False, clips_enabled=False, batching=False,
                             tracker_min_hits=2, tracker_buffer_s=1, edge_id="synthetic-edge")
    settings.models_config.write_text(
        "models:\n  det:\n    onnx_path: unused.onnx\n    input_size: [128, 128]\n    classes: [person]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("guard.pipeline.engine.MediaStore", NoMedia)
    backend = SyntheticInference()
    engine = PipelineEngine(settings, backend=backend)
    camera = CameraConfig(
        camera_id="dwell-camera", name="Synthetic zone dwell", site="Test site",
        source={"type": "images", "uri": str(tmp_path / "not-opened")}, sample_fps=2,
        zones=[Zone(zone_id="work", name="Work zone", polygon=[(.1, .1), (.9, .1), (.9, .9), (.1, .9)])],
        rules=[RuleConfig(rule_id="dwell", type="zone_dwell", zones=["work"],
                          params={"dwell_s": 1, "max_gap_s": 1},
                          min_duration_s=1, min_ratio=1, close_after_s=2, cooldown_s=30)],
    )
    engine.validate_camera(camera)
    engine.cameras[camera.camera_id] = camera
    worker = CameraWorker(camera, engine)
    engine.workers[camera.camera_id] = worker
    assert isinstance(worker.detectors["det"], Detector)
    assert isinstance(worker.person_tracker, Tracker)
    assert isinstance(worker.rules[0], ZoneDwellRule)
    assert isinstance(engine.events, EventManager) and isinstance(engine.journal, Journal)
    assert not any(compat.FALLBACKS.values())
    base_ts = 1_700_000_000.
    frame = np.full((128, 128, 3), (12, 34, 56), np.uint8)

    def process(second, capture="first-capture", present=True):
        backend.present = present
        worker.source.capture_id = capture
        # Actual source publication invokes the real on_frame callback. Neither
        # source nor analysis threads are started; every selected frame is explicit.
        worker.source._publish(frame, .25, ts=base_ts + second, pos_s=second)
        context = worker.process(worker.source.latest())
        assert worker.errors == {}
        assert context.model_observations == {"det": "observed"}
        assert context.rule_observations == {"dwell": "observed"}
        status = client.get(f"/api/cameras/{camera.camera_id}").json()["status"]
        assert status["model_observations"] == context.model_observations
        assert status["rule_observations"] == context.rule_observations
        assert status["capture_id"] == capture and not status["capture_boundary_pending"]
        return context

    def readback(event_id):
        response = client.get(f"/api/events/{event_id}")
        assert response.status_code == 200
        event = response.json()
        assert engine.journal.get(event_id).model_dump(mode="json") == event
        with Journal(settings.db_path) as independent:
            assert independent.get(event_id).model_dump(mode="json") == event
        return event

    with TestClient(create_app(settings, engine)) as client:
        with client.websocket_connect("/ws/events") as ws:
            assert ws.portal.call(lambda: engine.bus.subscribers) == 1
            assert client.get("/api/events").json() == {"items": [], "total": 0}
            # Real min_hits=2, then dwell=1s, then the EventManager window=1s.
            contexts = [process(second) for second in (0, .5, 1, 1.5, 2)]
            assert contexts[0].people == []
            assert drain_notifications(engine, ws, 1) == []
            confirmed = process(2.5)
            messages = drain_notifications(engine, ws, 2)
            assert len(messages) == 1 and messages[0]["kind"] == "event"
            first = messages[0]["event"]
            first_id = first["id"]
            assert first == readback(first_id)
            assert first["type"] == "zone_dwell" and first["rule_id"] == "dwell"
            assert first["camera_id"] == camera.camera_id and first["zone_id"] == "work"
            assert first["track_id"] == confirmed.people[0].track_id
            assert first["started_at"] == base_ts + 1.5 and first["confirmed_at"] == base_ts + 2.5
            assert first["details"]["observed_inside_since"] == base_ts + .5
            assert first["details"]["dwell_s"] == 2 and first["details"]["dwell_limit_s"] == 1
            assert first["details"]["evidence"] == "fresh_observed_person_inside_zone"
            assert first["details"]["origin"]["capture_id"] == "first-capture"
            assert first["details"]["origin"]["frame_seq"] == 6
            assert first["details"]["camera_name"] == camera.name

            patch = client.patch(f"/api/events/{first_id}", json={
                "status": "confirmed", "operator_note": "Checked by operator",
            })
            assert patch.status_code == 200
            assert drain_notifications(engine, ws, 3) == [{"kind": "event_update", "event": patch.json()}]
            assert readback(first_id) == patch.json()
            process(3)
            process(3.5, present=False)  # Successful empty model result remains observed.
            assert drain_notifications(engine, ws, 4) == []
            old_tracker, old_rule = worker.person_tracker, worker.rules[0]

            context = process(4, "second-capture")
            closed = readback(first_id)
            assert drain_notifications(engine, ws, 5) == [{"kind": "event_update", "event": closed}]
            assert closed["ended_at"] == base_ts + 3  # Last violation, not boundary time.
            assert closed["status"] == "confirmed" and closed["operator_note"] == "Checked by operator"
            assert closed["details"]["origin"] == first["details"]["origin"]
            assert worker.person_tracker is not old_tracker and worker.rules[0] is not old_rule
            assert context.people == []  # Fresh min_hits despite identical box position.
            fresh = process(4.5, "second-capture")
            assert fresh.people[0].track_id == first["track_id"]  # Numeric ID may be reused.
            assert fresh.people[0].first_seen == base_ts + 4
            for second in (5, 5.5, 6):
                process(second, "second-capture")
            assert drain_notifications(engine, ws, 6) == []
            assert client.get("/api/events").json() == {"items": [closed], "total": 1}
            process(6.5, "second-capture")
            messages = drain_notifications(engine, ws, 7)
            assert len(messages) == 1 and messages[0]["kind"] == "event"
            second = messages[0]["event"]
            assert second["id"] != first_id and second == readback(second["id"])
            assert second["started_at"] == base_ts + 5.5 and second["confirmed_at"] == base_ts + 6.5
            assert second["details"]["observed_inside_since"] == base_ts + 4.5
            assert second["details"]["origin"]["capture_id"] == "second-capture"
            assert second["status"] == "new" and second["operator_note"] is None
            assert "reopened_count" not in second["details"]
            process(7, "second-capture")
            for stamp in (7.5, 8, 8.5, 9):
                process(stamp, "second-capture", present=False)
            assert drain_notifications(engine, ws, 8) == []
            process(9.5, "second-capture", present=False)
            second_closed = readback(second["id"])
            assert second_closed["ended_at"] == base_ts + 7
            assert drain_notifications(engine, ws, 9) == [{"kind": "event_update", "event": second_closed}]
            process(10, "second-capture", present=False)
            assert drain_notifications(engine, ws, 10) == []
            assert readback(first_id) == closed
            assert engine.events_created == 2 and engine.events.open_events() == []
            expected = {first_id: closed, second["id"]: second_closed}
            page = client.get("/api/events", params={"type": "zone_dwell"}).json()
            assert page["total"] == 2 and {item["id"]: item for item in page["items"]} == expected
            with Journal(settings.db_path) as independent:
                rows, total = independent.query(limit=10)
                assert total == 2 and {event.id: event.model_dump(mode="json") for event in rows} == expected
            metrics = client.get("/api/metrics").json()["cameras"][camera.camera_id]
            assert metrics["frames_processed"] == backend.calls == worker.frame_idx == 21
            assert metrics["errors"] == 0 and backend.loads == ["det"]
    assert engine.bus.subscribers == 0 and not worker.running and not worker.source.running
