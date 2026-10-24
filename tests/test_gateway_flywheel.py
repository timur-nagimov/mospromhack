"""Actual lightweight gateway wiring, operator labels and retryable shutdown."""

import io
import sqlite3
import subprocess
import sys
import threading
import time
from uuid import uuid4
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from guard.api.app import create_app
from guard.contracts import CameraConfig, Event
from guard.flywheel.collector import SampleJob
from guard.gateway import engine as gateway_module
from guard.gateway.engine import GatewayEngine
from guard.pipeline.settings import Settings
from guard.transport.protocol import Envelope


class Consumer:
    def __init__(self, release=None):
        self.ready = threading.Event()
        self.release = release
        self.runs = 0
        self.closes = 0

    def run(self, callback, stop):
        self.runs += 1
        self.ready.set()
        if self.release is None:
            stop.wait()
        else:
            self.release.wait(5)
        self.close()

    def close(self):
        self.closes += 1

    def status(self):
        return {"connected": False}


def settings(root, *, autostart=True):
    return Settings(
        data_dir=root / "data", models_dir=root / "NO_MODELS", cameras_config=root / "NO_CAMERAS",
        models_config=root / "NO_REGISTRY", db_path=root / "data/events.db", media_dir=root / "data/media",
        frontend_dist=root / "NO_FRONTEND", mode="gateway", edge_id="test-edge", autostart=autostart,
    )


def sample_job(event_id=None):
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(stream, format="JPEG")
    return SampleJob(
        sample={"camera_id": "cam", "source": "event" if event_id else "operator", "event_id": event_id,
                "width": 32, "height": 24}, jpeg=stream.getvalue(),
    )


def test_actual_gateway_api_store_capture_503_and_durable_operator_labels(tmp_path, monkeypatch):
    owner = GatewayEngine(settings(tmp_path), consumer=Consumer())
    producer, run = str(uuid4()), str(uuid4())

    def message(kind, payload, seq):
        return Envelope(edge_id="test-edge", producer_id=producer, run_id=run, sequence=seq,
                        message_id=f"{producer}:{seq}", kind=kind, sent_at=time.time(), payload=payload)

    with TestClient(create_app(owner.settings, owner)) as client:
        assert owner.consumer.ready.wait(2)
        writer_thread, consumer_thread = owner.flywheel_writer._thread, owner._thread
        owner.start()
        assert owner._thread is consumer_thread and owner.flywheel_writer._thread is writer_thread
        assert owner.consumer.runs == 1 and owner.flywheel_writer.running
        # Samples arrive from the edge in parts; a fake consumer gets no AMQP sample receiver.
        assert owner.flywheel is None and owner.flywheel_assembler is not None
        assert owner.flywheel_receiver is None
        summary = client.get("/api/flywheel/summary")
        assert summary.status_code == 200 and summary.json()["total"] == 0
        assert summary.json()["mode"] == "gateway" and not summary.json()["collecting"]
        assert client.get("/api/flywheel/samples").json()["total"] == 0

        camera = CameraConfig(camera_id="cam", name="Remote", site="Test",
                              source={"type": "rtsp", "uri": "edge://test-edge/cam"})
        owner.receive(message("cameras", {"cameras": [camera.model_dump(mode="json")], "statuses": {}}, 1))
        assert not callable(getattr(getattr(owner.workers["cam"], "source", None), "latest", None))
        capture = client.post("/api/flywheel/capture/cam")
        assert capture.status_code == 503 and "no frame yet" in capture.json()["detail"]
        assert client.post("/api/flywheel/capture/missing").status_code == 404
        assert owner.flywheel_store.count() == 0

        event = Event(id="event-1", camera_id="cam", site="Test", rule_id="zone", type="zone_intrusion",
                      severity="high", started_at=100, confirmed_at=102, bbox=(1, 2, 20, 23))
        owner.receive(message("event", {"event": event.model_dump(mode="json")}, 2))
        stored = owner.flywheel_writer.write(sample_job(event.id))
        assert client.get(f"/api/flywheel/samples/{stored['id']}").json()["operator_label"] is None
        original_decision = gateway_module.operator_decision
        decisions = []

        def decision_after_commit(engine, event_id, status):
            # The hook must never label an uncommitted event decision.
            assert engine.journal.get(event_id).status == status
            decisions.append(status)
            original_decision(engine, event_id, status)

        monkeypatch.setattr(gateway_module, "operator_decision", decision_after_commit)
        for status, label in [("confirmed", "positive"), ("false", "hard_negative")]:
            patched = client.patch(f"/api/events/{event.id}", json={"status": status})
            assert patched.status_code == 200 and patched.json()["status"] == status
            sample = client.get(f"/api/flywheel/samples/{stored['id']}").json()
            assert sample["operator_label"] == label
        assert len(decisions) == 2
        assert client.patch(f"/api/events/{event.id}", json={"operator_note": "kept"}).status_code == 200
        assert len(decisions) == 2  # Note-only PATCH does not relabel samples.
        late = owner.flywheel_writer.write(sample_job(event.id))
        assert late["operator_label"] == "hard_negative"
        assert owner.patch_event("missing", status="confirmed") is None and len(decisions) == 2

        def fail_write(*args, **kwargs):
            raise sqlite3.OperationalError("injected uncommitted decision")

        with monkeypatch.context() as patch:
            patch.setattr(owner.journal, "update", fail_write)
            with pytest.raises(sqlite3.OperationalError):
                owner.patch_event(event.id, status="confirmed")
        assert len(decisions) == 2 and owner.journal.get(event.id).status == "false"
        assert owner.flywheel_store.get(stored["id"])["operator_label"] == "hard_negative"
    assert owner._closed and not consumer_thread.is_alive() and not writer_thread.is_alive()
    owner.stop()  # Idempotent after both background threads exited.


@pytest.mark.parametrize("blocked", ["consumer", "writer"])
def test_gateway_stop_timeout_retains_threads_and_sqlite_until_retry(tmp_path, monkeypatch, blocked):
    release, writing = threading.Event(), threading.Event()
    consumer = Consumer(release if blocked == "consumer" else None)
    owner = GatewayEngine(settings(tmp_path), consumer=consumer)
    original_write = owner.flywheel_writer.write

    def blocked_write(job):
        writing.set()
        assert release.wait(5)
        return original_write(job)

    if blocked == "writer":
        monkeypatch.setattr(owner.flywheel_writer, "write", blocked_write)
    owner.start()
    writer_thread, consumer_thread = owner.flywheel_writer._thread, owner._thread
    try:
        assert consumer.ready.wait(2)
        if blocked == "writer":
            assert owner.flywheel_writer.put(sample_job()) and writing.wait(2)
        target = consumer_thread if blocked == "consumer" else writer_thread
        join = target.join
        with monkeypatch.context() as patch:
            patch.setattr(target, "join", lambda timeout=None: join(0))
            with pytest.raises(RuntimeError, match="consumer did not stop|writer did not stop"):
                owner.stop()
            assert not owner._closed and target.is_alive()
            assert owner.flywheel_writer._thread is writer_thread
            assert owner.journal.query()[1] == 0 and owner.flywheel_store.count() == 0
        release.set()
        target.join(2)
        owner.stop()
        assert owner._closed and not writer_thread.is_alive() and not consumer_thread.is_alive()
        assert owner.flywheel_writer._thread is None
        with pytest.raises(sqlite3.ProgrammingError):
            owner.journal.query()
        with pytest.raises(sqlite3.ProgrammingError):
            owner.flywheel_store.count()
    finally:
        release.set()
        owner.stop()


def test_gateway_with_unavailable_flywheel_store_still_serves_events(tmp_path, monkeypatch):
    def fail_store(root):
        raise OSError("injected unavailable flywheel directory")

    monkeypatch.setattr("guard.flywheel.wiring.FlywheelStore", fail_store)
    owner = GatewayEngine(settings(tmp_path), consumer=Consumer())
    with TestClient(create_app(owner.settings, owner)) as client:
        assert owner.consumer.ready.wait(2)
        assert owner.flywheel_store is None and owner.flywheel_writer is None
        assert client.get("/api/flywheel/summary").status_code == 503
        assert client.get("/api/events").json() == {"items": [], "total": 0}
    assert owner._closed


@pytest.mark.parametrize("failing_thread", ["gateway-replication", "flywheel-writer"])
def test_gateway_thread_start_failure_still_closes_writer_and_sqlite(tmp_path, monkeypatch, failing_thread):
    owner = GatewayEngine(settings(tmp_path), consumer=Consumer())
    original_start = threading.Thread.start

    def fail_replication_start(thread):
        if thread.name == failing_thread:
            raise RuntimeError("injected thread start failure")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_replication_start)
    with pytest.raises(RuntimeError, match="injected thread start failure"):
        with TestClient(create_app(owner.settings, owner)):
            pytest.fail("Startup failure must not serve requests")
    assert owner._closed and owner._thread is None and not owner.flywheel_writer.running
    assert owner.consumer.closes == 1
    with pytest.raises(sqlite3.ProgrammingError):
        owner.journal.query()
    with pytest.raises(sqlite3.ProgrammingError):
        owner.flywheel_store.count()


def test_local_pipeline_keeps_backend_and_journal_open_until_flywheel_stop_succeeds(tmp_path, monkeypatch):
    from guard.pipeline import engine as pipeline_module
    from guard.pipeline.testing import FakeBackend, make_settings

    backend = FakeBackend()
    backend.close = Mock()
    owner = pipeline_module.PipelineEngine(make_settings(tmp_path, autostart=False), backend=backend)
    journal_close = Mock(wraps=owner.journal.close)
    before_close = Mock(wraps=owner._before_journal_close)
    monkeypatch.setattr(owner.journal, "close", journal_close)
    monkeypatch.setattr(owner, "_before_journal_close", before_close)
    real_stop = pipeline_module.stop_flywheel
    attempts = []

    def stop_once_busy(engine):
        attempts.append(engine)
        if len(attempts) == 1:
            raise RuntimeError("injected busy flywheel writer")
        real_stop(engine)

    monkeypatch.setattr(pipeline_module, "stop_flywheel", stop_once_busy)
    try:
        with pytest.raises(RuntimeError, match="busy flywheel writer"):
            owner.stop()
        assert not owner._stopped and owner.journal.query()[1] == owner.flywheel_store.count() == 0
        backend.close.assert_not_called()
        journal_close.assert_not_called()
        before_close.assert_not_called()
        owner.stop()
        assert owner._stopped and len(attempts) == 2
        backend.close.assert_called_once()
        journal_close.assert_called_once()
        before_close.assert_called_once()
        owner.stop()
        assert len(attempts) == 2
    finally:
        owner.stop()


def test_gateway_flywheel_import_and_api_do_not_require_opencv_inference_or_sources(tmp_path):
    script = r'''
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('guard.inference', 'guard.sources', 'guard.pipeline.engine', 'guard.pipeline.worker', 'cv2', 'onnxruntime', 'linq')):
            raise AssertionError('forbidden lightweight gateway import: ' + fullname)
sys.meta_path.insert(0, Block())
import guard.pipeline.settings as configuration
configuration.load_dotenv_guard = lambda *args: None
from guard.gateway.engine import GatewayEngine
from guard.api.app import create_app
from fastapi.testclient import TestClient
class Consumer:
    def run(self, callback, stop): stop.wait()
    def close(self): pass
    def status(self): return {'connected': False}
root = Path(sys.argv[1])
settings = configuration.Settings(data_dir=root/'data', models_dir=root/'NO_MODELS', cameras_config=root/'NO_CAMERAS', models_config=root/'NO_REGISTRY', db_path=root/'data/events.db', media_dir=root/'data/media', frontend_dist=root/'NO_FRONTEND', mode='gateway', edge_id='edge', autostart=True)
owner = GatewayEngine(settings, consumer=Consumer())
with TestClient(create_app(settings, owner)) as client:
    assert client.get('/health').json()['mode'] == 'gateway'
    assert client.get('/api/flywheel/summary').json()['total'] == 0
    assert client.get('/api/flywheel/samples').json()['total'] == 0
assert owner._closed
assert not settings.models_dir.exists() and not settings.cameras_config.exists()
assert not any(name.startswith(('cv2', 'onnxruntime', 'guard.sources', 'guard.inference')) for name in sys.modules)
print('gateway-flywheel-lightweight-ok')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gateway-flywheel-lightweight-ok" in result.stdout
