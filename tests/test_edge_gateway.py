"""Real SQLite/API/WS integration with an explicitly injected offline broker."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sqlite3
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import anyio
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, Event, SourceConfig
from guard.edge.runner import EdgeEngine
from guard.gateway.engine import GatewayEngine
from guard.inference.base import InferenceBackend
from guard.pipeline.settings import Settings
from guard.sources.base import FramePacket, FrameSource
from guard.transport.fake import FakeBroker
from guard.transport.protocol import MAX_MESSAGE_BYTES, Envelope, decode, encode, event_payload


def settings(root, mode="edge", **overrides):
    return Settings(
        data_dir=root / "data", models_dir=root / "no-models", cameras_config=root / "cameras.yaml",
        models_config=root / "no-models.yaml", db_path=root / "data" / "guard.db", media_dir=root / "data" / "media",
        frontend_dist=root / "no-frontend", mode=mode, edge_id="edge-1" if mode != "local" else None,
        rabbit_url="amqp://user:hidden-password@localhost/", autostart=False, **overrides,
    )


class NoInference(InferenceBackend):
    name = "offline"

    def __init__(self):
        self.closed = 0

    def load(self, spec):
        raise AssertionError("No model may be loaded in these tests")

    def infer(self, model_id, batch):
        raise AssertionError("No inference may run in these tests")

    def stats(self):
        return {"backend": "offline", "models": {}}

    def close(self):
        self.closed += 1


def jpeg(color="red"):
    out = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(out, format="JPEG")
    return out.getvalue()


def event(**overrides):
    return Event(**{
        "id": "violation-1", "camera_id": "cam-1", "site": "Factory", "zone_id": "danger",
        "rule_id": "zone", "type": "zone_intrusion", "severity": "high",
        "started_at": 100, "confirmed_at": 102, "bbox": (1, 2, 20, 23),
        **overrides,
    })


def drain(engine, ws):
    marker = {"kind": "test_barrier", "id": str(uuid4())}
    engine.bus.publish(marker)
    result = []
    while True:
        async def receive():
            with anyio.fail_after(3):
                return await ws._send_rx.receive()
        message = ws.portal.call(receive)
        assert message["type"] == "websocket.send"
        item = json.loads(message["text"])
        if item == marker:
            return result
        if item["kind"] != "metrics":
            result.append(item)


def test_outbox_uncertain_publish_restart_gateway_api_ws_snapshot_and_operator_status(tmp_path):
    broker = FakeBroker()
    edge_settings = settings(tmp_path / "edge")
    edge = EdgeEngine(edge_settings, backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    gateway_settings = settings(tmp_path / "gateway", "gateway")
    gateway = GatewayEngine(gateway_settings, consumer=broker)
    original = event()
    image_path = edge_settings.media_dir / "source.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(jpeg("red"))
    original.snapshot_path = image_path.relative_to(edge_settings.data_dir).as_posix()
    edge.journal.add(original)
    assert edge.journal.replication_stats()["pending"] == 1
    broker.online = False
    assert edge.replication.publish_pending_once() == 0
    assert edge.journal.replication_stats()["pending"] == 1
    broker.online, broker.lose_confirm = True, True
    assert edge.replication.publish_pending_once() == 0
    first_wire = broker.events[0]
    assert edge.journal.replication_stats()["pending"] == 1
    producer = edge.journal.replication_stats()["producer_id"]
    # Restart after an uncertain confirm and mutate the source image: frozen
    # wire content must survive both, or gateway receipt fingerprints reject it.
    edge.stop()
    image_path.write_bytes(jpeg("blue"))
    edge = EdgeEngine(edge_settings, backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    assert edge.journal.replication_stats()["producer_id"] == producer
    assert edge.replication.publish_pending_once() == 1
    assert broker.events[1] == first_wire
    assert edge.journal.replication_stats()["pending"] == 0
    with TestClient(create_app(gateway_settings, gateway)) as client:
        with client.websocket_connect("/ws/events") as ws:
            assert ws.portal.call(lambda: gateway.bus.subscribers) == 1
            broker.deliver(gateway.receive, duplicate=True)
            messages = drain(gateway, ws)
            assert len(messages) == 1 and messages[0]["kind"] == "event"
            stored = messages[0]["event"]
            assert stored["status"] == "new" and stored["operator_note"] is None
            assert stored["snapshot_path"] != original.snapshot_path
            snapshot = client.get("/api/events/violation-1/snapshot.jpg")
            assert snapshot.status_code == 200 and snapshot.content == jpeg("red")
            assert gateway.media.resolve("../source.jpg") is None
            assert gateway.media.resolve(str(image_path)) is None
            patch = client.patch("/api/events/violation-1", json={"status": "confirmed", "operator_note": "Checked"})
            assert patch.status_code == 200
            assert drain(gateway, ws) == [{"kind": "event_update", "event": patch.json()}]
            while broker.events:
                broker.deliver(gateway.receive)
            assert drain(gateway, ws) == []
            edge.journal.update(original.id, ended_at=105, clip_path="/edge/private/clip.mp4")
            assert edge.replication.publish_pending_once() == 1
            broker.deliver(gateway.receive)
            closed = client.get("/api/events/violation-1").json()
            assert closed["ended_at"] == 105 and closed["clip_path"] is None
            assert closed["status"] == "confirmed" and closed["operator_note"] == "Checked"
            assert drain(gateway, ws) == [{"kind": "event_update", "event": closed}]
            assert client.get("/api/events/violation-1/snapshot.jpg").content == jpeg("blue")
            gateway.receive(decode(first_wire))
            assert drain(gateway, ws) == []
            assert client.get("/api/events/violation-1/snapshot.jpg").content == jpeg("blue")
            assert client.get("/api/events").json()["total"] == 1
            assert client.get("/api/events/violation-1/clip.mp4").status_code == 404
    # Gateway restart must retain receipts and local operator decisions.
    restarted = GatewayEngine(gateway_settings, consumer=broker)
    with TestClient(create_app(gateway_settings, restarted)) as client:
        assert not restarted.receive(decode(first_wire)).changed
        assert client.get("/api/events").json()["total"] == 1
        assert client.get("/api/events/violation-1").json() == closed
    edge.stop()


@pytest.mark.parametrize("image", [b"broken JPEG", b"x" * (1_048_576 + 1)])
def test_bad_or_oversize_snapshot_does_not_block_durable_event(tmp_path, image):
    broker = FakeBroker()
    cfg = settings(tmp_path)
    edge = EdgeEngine(cfg, backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    path = cfg.media_dir / "bad.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(image)
    edge.journal.add(event(snapshot_path=path.relative_to(cfg.data_dir).as_posix()))
    assert edge.replication.publish_pending_once() == 1
    assert edge.journal.replication_stats()["pending"] == 0
    assert "snapshot_jpeg" not in decode(broker.events[0]).payload
    assert edge.replication.status()["snapshot_error"]
    edge.stop()


def envelope(kind, payload, *, producer=None, run=None, sequence=1, sent_at=100):
    producer = producer or str(uuid4())
    return Envelope(
        edge_id="edge-1", producer_id=producer, run_id=run or str(uuid4()), sequence=sequence,
        message_id=f"{producer}:{sequence}", kind=kind, sent_at=float(sent_at), payload=payload,
    )


def catalogue():
    camera = CameraConfig(camera_id="cam-1", name="Remote", site="Factory", source=SourceConfig(type="rtsp", uri="edge://edge-1/cam-1"))
    return {"cameras": [camera.model_dump(mode="json")], "statuses": {"cam-1": {"running": True}}}


def preview(capture=None, frame_seq=1, color="red"):
    return {
        "camera_id": "cam-1", "capture_id": capture or str(uuid4()), "frame_seq": frame_seq,
        "capture_ts": 100.0, "source_pos_s": 1.0, "width": 32, "height": 24,
        "jpeg_b64": base64.b64encode(jpeg(color)).decode(), "people": [], "objects": [], "highlight": [], "zones": [],
    }


def test_gateway_orders_catalogue_run_capture_and_stale_frames_and_denies_camera_crud(tmp_path):
    clock = [100.0]
    cfg = settings(tmp_path, "gateway")
    gateway = GatewayEngine(cfg, consumer=FakeBroker(), clock=lambda: clock[0], wall_clock=lambda: clock[0])
    producer, run = str(uuid4()), str(uuid4())
    early = envelope("frame_meta", preview(), producer=producer, run=run)
    gateway.receive(early)
    assert gateway.workers == {}
    gateway.receive(envelope("cameras", catalogue(), producer=producer, run=run, sequence=2))
    first = preview(frame_seq=10)
    gateway.receive(envelope("frame_meta", first, producer=producer, run=run, sequence=3))
    assert gateway.workers["cam-1"].frame_jpeg() == jpeg("red")
    gateway.receive(envelope("frame_meta", preview(color="blue"), producer=producer, run=run, sequence=2))
    assert gateway.workers["cam-1"].frame_jpeg() == jpeg("red")
    second = preview(frame_seq=1, color="blue")  # file loop / reconnect
    gateway.receive(envelope("frame_meta", second, producer=producer, run=run, sequence=4))
    gateway.receive(envelope("frame_meta", first, producer=producer, run=run, sequence=5))
    assert gateway.workers["cam-1"].frame_jpeg() == jpeg("blue")
    new_run = str(uuid4())
    gateway.receive(envelope("frame_meta", preview(color="green"), producer=producer, run=new_run, sequence=6))
    assert gateway.workers["cam-1"].frame_jpeg() == jpeg("blue")
    gateway.receive(envelope("cameras", catalogue(), producer=producer, run=new_run, sequence=7))
    assert gateway.workers["cam-1"].frame_jpeg() is None
    gateway.receive(envelope("frame_meta", preview(color="green"), producer=producer, run=new_run, sequence=8))
    gateway.receive(envelope("cameras", catalogue(), producer=producer, run=run, sequence=99))
    assert gateway.workers["cam-1"].frame_jpeg() == jpeg("green")
    new_producer = str(uuid4())
    gateway.receive(envelope("cameras", catalogue(), producer=new_producer, run=new_run, sequence=1))
    gateway.receive(envelope("frame_meta", preview(color="yellow"), producer=new_producer, run=new_run, sequence=2))
    gateway.receive(envelope("cameras", catalogue(), producer=producer, run=run, sequence=100))
    gateway.receive(envelope("metrics", {"data": {"backend": "remote", "cameras": {}, "inference": {}}}, producer=new_producer, run=new_run, sequence=3))
    with TestClient(create_app(cfg, gateway)) as client:
        assert client.get("/api/cameras/cam-1/frame.jpg").content == jpeg("yellow")
        assert client.get("/api/metrics").json()["stale"] is False
        camera = catalogue()["cameras"][0]
        assert client.post("/api/cameras", json=camera).status_code == 409
        assert client.put("/api/cameras/cam-1", json=camera).status_code == 409
        assert client.delete("/api/cameras/cam-1").status_code == 409
        clock[0] += 11
        assert client.get("/api/cameras/cam-1/frame.jpg").status_code == 503
        assert client.get("/api/cameras").json()[0]["status"]["stale"] is True
        assert client.get("/api/metrics").json()["stale"] is True
        # Replay after timeout must not mark a stale producer as alive again.
        gateway.receive(envelope("frame_meta", preview(), producer=new_producer, run=new_run, sequence=10, sent_at=100))
        assert gateway.workers["cam-1"].frame_jpeg() is None


def test_edge_telemetry_is_bounded_sanitized_and_contains_exact_analysis_frame(tmp_path):
    import numpy as np
    from guard.pipeline.worker import AnalysisResult
    broker = FakeBroker()
    cfg = settings(tmp_path)
    edge = EdgeEngine(cfg, backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    camera = CameraConfig(camera_id="cam-1", name="Camera", site="Factory", source=SourceConfig(type="rtsp", uri="rtsp://user:secret@camera/"))
    edge.cameras[camera.camera_id] = camera
    result = AnalysisResult(seq=1, ts=time.time(), frame=np.zeros((720, 1280, 3), np.uint8), people=[], objects=[], highlight=set(), active_zones=set(), capture_id=str(uuid4()), source_pos_s=2.0)
    edge.workers[camera.camera_id] = SimpleNamespace(
        cfg=camera, last=result, status=lambda: {"running": True, "errors": {"source": "rtsp://user:secret@camera/"}},
        stop=lambda: None,
    )
    clock = [0.0]
    edge.replication.clock = lambda: clock[0]
    assert edge.replication.publish_telemetry_once() == 3
    decoded = [decode(item) for item in broker.telemetry]
    assert [item.kind for item in decoded] == ["cameras", "metrics", "frame_meta"]
    assert decoded[0].payload["cameras"][0]["source"]["uri"] == "edge://edge-1/cam-1"
    assert b"secret" not in b"".join(broker.telemetry)
    frame = decoded[-1].payload
    assert frame["capture_id"] == result.capture_id and frame["frame_seq"] == 1
    assert frame["source_pos_s"] == 2 and frame["width"] == 480 and frame["height"] == 270
    with Image.open(io.BytesIO(base64.b64decode(frame["jpeg_b64"]))) as image:
        assert image.size == (480, 270)
    assert edge.replication.publish_telemetry_once() == 0
    def failing_stats():
        raise ValueError("amqp://user:secret@host/")
    edge.backend.stats = failing_stats
    for _ in range(50):
        clock[0] += 2
        edge.replication.publish_telemetry_once()
    assert len(broker.telemetry) <= 32 and not broker.events
    assert b"secret" not in b"".join(broker.telemetry)
    metrics = [decode(item) for item in broker.telemetry if decode(item).kind == "metrics"]
    assert metrics[-1].payload["data"]["inference"]["error"] == "Processing error"
    edge.stop()


def test_replication_threads_stop_and_backend_resources_close(tmp_path):
    class Probe(FakeBroker):
        def __init__(self):
            super().__init__()
            self.confirmed = threading.Event()
        def publish(self, message, *, durable=True):
            super().publish(message, durable=durable)
            if durable:
                self.confirmed.set()
    broker = Probe()
    backend = NoInference()
    edge = EdgeEngine(settings(tmp_path), backend=backend, publisher=broker, telemetry_publisher=broker)
    edge.journal.add(event())
    edge.replication.start()
    assert broker.confirmed.wait(3)
    edge.stop()
    assert not any(thread.is_alive() for thread in edge.replication._threads)
    assert backend.closed == 1
    edge.stop()
    assert backend.closed == 1


def test_startup_failure_cleanup_preserves_primary_error(tmp_path):
    cfg = settings(tmp_path, "local")
    cfg.autostart = True
    stops = []
    def fail_start():
        raise RuntimeError("primary startup failure")
    def fail_stop():
        stops.append(True)
        raise ValueError("secondary cleanup failure")
    engine = SimpleNamespace(settings=cfg, start=fail_start, stop=fail_stop)
    with pytest.raises(RuntimeError, match="primary startup failure"):
        with TestClient(create_app(cfg, engine)):
            pass
    assert stops == [True]


def test_gateway_starts_with_inference_sources_and_models_unavailable(tmp_path):
    script = r'''
import importlib.abc
import json
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("guard.inference", "guard.sources", "guard.pipeline.engine", "guard.pipeline.worker", "cv2", "onnxruntime", "linq")):
            raise AssertionError("forbidden gateway import: " + fullname)
sys.meta_path.insert(0, Block())
from guard.pipeline.settings import Settings
from guard.api.app import create_app
import guard.gateway.engine as module
from fastapi.testclient import TestClient
class Consumer:
    def run(self, callback, stop):
        stop.wait()
    def close(self):
        pass
    def status(self):
        return {"connected": False}
module.RabbitConsumer = lambda config: Consumer()
root = Path(sys.argv[1])
settings = Settings(data_dir=root/'data', models_dir=root/'NO_MODELS', cameras_config=root/'NO_CAMERAS', models_config=root/'NO_MODELS_YAML', db_path=root/'data/guard.db', media_dir=root/'data/media', frontend_dist=root/'NO_FRONTEND', mode='gateway', edge_id='edge-1', rabbit_url='amqp://user:secret@localhost/', autostart=True)
with TestClient(create_app(settings)) as client:
    assert client.get('/health').json()['mode'] == 'gateway'
    assert client.get('/api/events').json()['total'] == 0
    assert client.get('/api/analytics').json()['types'] == []
    assert client.get('/api/rules/types').status_code == 200
assert not settings.models_dir.exists() and not settings.models_config.exists()
print('gateway-lightweight-ok')
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "gateway-lightweight-ok" in process.stdout


def test_settings_mode_validation_and_secret_safe_repr(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    assert "hidden-password" not in repr(cfg)
    for mode in ("bad-mode", "EDGE"):
        with pytest.raises(ValueError):
            replace(cfg, mode=mode)
    with pytest.raises(ValueError):
        replace(cfg, edge_id=None)
    with pytest.raises(ValueError):
        replace(cfg, preview_fps=2)
    monkeypatch.setenv("GUARD_MODE", "gateway")
    monkeypatch.setenv("GUARD_EDGE_ID", "edge-1")
    monkeypatch.setenv("GUARD_RABBIT_URL", "amqp://user:hidden-password@localhost/")
    parsed = Settings.from_env()
    assert parsed.mode == "gateway" and parsed.edge_id == "edge-1"
    assert "hidden-password" not in repr(parsed)


def test_source_capture_identity_rotates_without_resetting_global_sequence():
    import numpy as np
    source = FrameSource(SourceConfig(type="images", uri="unused"), "unused")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    source._publish(frame, 0, ts=100, pos_s=0)
    first = source.latest()
    source._publish(frame, 0, ts=101, pos_s=1)
    assert source.latest().capture_id == first.capture_id
    source._new_capture()
    source._publish(frame, 0, ts=102, pos_s=0)
    assert source.latest().capture_id != first.capture_id
    assert source.latest().seq == 3 and source.latest().pos_s == 0
    assert FramePacket(frame, 100, 1, 0).capture_id


def test_image_folder_loop_rotates_capture_identity_without_changing_sampling(tmp_path):
    from guard.sources.images import ImageFolderSource
    (tmp_path / "first.jpg").write_bytes(jpeg("red"))
    (tmp_path / "second.jpg").write_bytes(jpeg("blue"))
    packets = []
    def on_frame(packet):
        packets.append(packet)
        if len(packets) == 4:
            source._stop.set()
    source = ImageFolderSource(
        SourceConfig(type="images", uri=str(tmp_path), loop=True, realtime=False),
        str(tmp_path), on_frame=on_frame,
    )
    source._run()
    assert [packet.seq for packet in packets] == [1, 2, 3, 4]
    assert packets[0].capture_id == packets[1].capture_id
    assert packets[2].capture_id == packets[3].capture_id != packets[0].capture_id


def test_sender_stop_timeout_keeps_sqlite_and_backend_open_for_retry(tmp_path, monkeypatch):
    broker = FakeBroker()
    backend = NoInference()
    edge = EdgeEngine(settings(tmp_path), backend=backend, publisher=broker, telemetry_publisher=broker)
    attempts = []
    def stop():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("Sender still alive")
    monkeypatch.setattr(edge.replication, "stop", stop)
    with pytest.raises(RuntimeError, match="Sender still alive"):
        edge.stop()
    assert edge.journal.query(limit=0) == ([], 0)
    assert backend.closed == 0
    edge.stop()
    assert attempts == [True, True]
    assert backend.closed == 1


def test_edge_cannot_fall_back_to_memory_when_sqlite_is_unavailable(tmp_path):
    cfg = settings(tmp_path)
    cfg.db_path.mkdir(parents=True)
    backend = NoInference()
    with pytest.raises(sqlite3.OperationalError, match="database"):
        EdgeEngine(cfg, backend=backend, publisher=FakeBroker(), telemetry_publisher=FakeBroker())
    assert backend.closed == 1


def test_reconnect_does_not_refresh_an_old_analysis_frame(tmp_path):
    import numpy as np
    from guard.pipeline.worker import AnalysisResult
    broker = FakeBroker()
    edge = EdgeEngine(settings(tmp_path / "edge"), backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    camera = CameraConfig(camera_id="cam-1", name="Camera", site="Factory", source=SourceConfig(type="images", uri="unused"))
    edge.cameras[camera.camera_id] = camera
    edge.workers[camera.camera_id] = SimpleNamespace(
        cfg=camera, status=lambda: {"running": True}, stop=lambda: None,
        last=AnalysisResult(seq=1, ts=100, frame=np.zeros((24, 32, 3), np.uint8), people=[], objects=[], highlight=set(), active_zones=set(), capture_id=str(uuid4())),
    )
    now = [100.0]
    edge.replication.wall_clock = lambda: now[0]
    edge.replication.clock = lambda: now[0]
    broker.online = False
    assert edge.replication.publish_telemetry_once() == 0
    broker.online = True
    now[0] = 200.0
    edge.replication.publish_telemetry_once()
    assert all(decode(item).kind != "frame_meta" for item in broker.telemetry)
    gateway = GatewayEngine(settings(tmp_path / "gateway", "gateway"), consumer=broker, clock=lambda: now[0], wall_clock=lambda: now[0])
    producer, run = str(uuid4()), str(uuid4())
    gateway.receive(envelope("cameras", catalogue(), producer=producer, run=run, sent_at=200))
    gateway.receive(envelope("frame_meta", preview(), producer=producer, run=run, sequence=2, sent_at=200))
    assert gateway.workers["cam-1"].frame_jpeg() is None
    gateway.stop()
    edge.stop()


def test_transient_outbox_read_error_is_visible_and_sender_recovers_without_restart(tmp_path, monkeypatch):
    class Probe(FakeBroker):
        def __init__(self):
            super().__init__()
            self.confirmed = threading.Event()
        def publish(self, message, *, durable=True):
            super().publish(message, durable=durable)
            if durable:
                self.confirmed.set()
    broker = Probe()
    edge = EdgeEngine(settings(tmp_path), backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    edge.journal.add(event())
    original = edge.journal.pending_messages
    failed, retry_entered, allow_retry = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def pending(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            failed.set()
            raise sqlite3.OperationalError("temporary read failure")
        if len(calls) == 2:
            retry_entered.set()
            assert allow_retry.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(edge.journal, "pending_messages", pending)
    edge.replication.start()
    try:
        assert failed.wait(3) and retry_entered.wait(3)
        assert edge.replication.last_error == "OperationalError"
        assert all(thread.is_alive() for thread in edge.replication._threads)
        allow_retry.set()
        assert broker.confirmed.wait(3)
    finally:
        allow_retry.set()
        edge.replication.stop()
    assert edge.replication.last_error is None
    assert edge.journal.replication_stats()["pending"] == 0
    edge.stop()


@pytest.mark.parametrize("channel,full_batch", [("events", False), ("telemetry", False), ("events", True)])
def test_sender_backoff_is_capped_resets_on_success_and_uses_interruptible_wait(tmp_path, channel, full_batch):
    class Broker(FakeBroker):
        def publish(self, message, *, durable=True):
            if full_batch and durable and len(self.events) == 100:
                raise ConnectionError("Connection lost after a full confirmed batch")
            super().publish(message, durable=durable)
    broker = Broker()
    edge = EdgeEngine(settings(tmp_path), backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    for index in range(101 if full_batch else 1):
        edge.journal.add(event(id=f"event-{index}"))
    broker.online = False
    class Stop:
        stopped = False
        delays = []
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
        def wait(self, delay):
            self.delays.append(delay)
            if len(self.delays) == (2 if full_batch else 7):
                broker.online = True
            if len(self.delays) == (3 if full_batch else 8):
                self.stopped = True
            return self.stopped
    stop = Stop()
    edge.replication._stop = stop
    getattr(edge.replication, "_run_" + channel)()
    if full_batch:
        # Error, error, 100 positive confirms, error: the last error must start
        # a fresh retry sequence, without introducing a delay after the batch.
        assert stop.delays == [0.5, 1, 0.5]
        assert len(broker.events) == 100
        assert edge.journal.replication_stats()["pending"] == 1
    else:
        assert stop.delays == [0.5, 1, 2, 4, 8, 15, 15, 0.5 if channel == "events" else 1]
    if channel == "events" and not full_batch:
        assert edge.journal.replication_stats()["pending"] == 0
    elif channel == "telemetry":
        assert len(broker.telemetry) == 2
    edge.stop()


def test_local_backend_and_journal_close_despite_independent_worker_cleanup_error(tmp_path):
    from guard.pipeline.engine import PipelineEngine
    backend = NoInference()
    engine = PipelineEngine(settings(tmp_path, "local"), backend=backend)
    def fail():
        raise RuntimeError("worker cleanup failure")
    engine.workers["cam-1"] = SimpleNamespace(stop=fail)
    with pytest.raises(RuntimeError, match="worker cleanup failure"):
        engine.stop()
    assert backend.closed == 1
    with pytest.raises(sqlite3.ProgrammingError):
        engine.journal.query()


def test_gateway_doctor_does_not_require_models_or_camera_configuration(tmp_path):
    from guard.doctor import run_checks
    cfg = settings(tmp_path, "gateway")
    report = run_checks(cfg)
    assert report["ok"]
    assert report["required_models"] == []
    assert not cfg.models_config.exists() and not cfg.cameras_config.exists()
    assert not cfg.data_dir.exists()
    invalid = run_checks(replace(cfg, rabbit_url="http://user:hidden-password@bad"))
    assert not invalid["ok"]
    assert any(item["check"] == "transport:config" and item["status"] == "error" for item in invalid["checks"])
    assert "hidden-password" not in json.dumps(invalid)


def test_snapshot_is_omitted_if_combined_envelope_would_exceed_limit(tmp_path):
    broker = FakeBroker()
    cfg = settings(tmp_path)
    edge = EdgeEngine(cfg, backend=NoInference(), publisher=broker, telemetry_publisher=broker)
    path = cfg.media_dir / "valid.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(jpeg())
    probe = edge.journal.make_message("event", event_payload(event(details={"note": ""})))
    note = "x" * (MAX_MESSAGE_BYTES - len(encode(probe)) - 200)
    edge.journal.add(event(details={"note": note}, snapshot_path=path.relative_to(cfg.data_dir).as_posix()))
    assert edge.replication.publish_pending_once() == 1
    wire = decode(broker.events[0])
    assert wire.payload["event"]["details"]["note"] == note
    assert "snapshot_jpeg" not in wire.payload
    assert edge.replication.snapshot_error == "SnapshotRejected"
    edge.stop()
