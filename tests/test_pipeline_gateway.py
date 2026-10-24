"""Legacy MemoryTransport compatibility and the production durable edge CLI."""

import time

import numpy as np
import yaml
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, RuleConfig, SourceConfig, Zone
from guard.pipeline._fallbacks import MemoryJournal
from guard.pipeline.config import save_cameras
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.gateway import GatewayEngine
from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person, write_images
from guard.pipeline.transport import compact_jpeg
from guard.pipeline.worker import CameraWorker
from guard.sources import FramePacket
from guard.transport import MemoryTransport


def _camera(tmp_path) -> CameraConfig:
    return CameraConfig(
        camera_id="cam1",
        name="Цех",
        site="Цех 1",
        source=SourceConfig(type="images", uri=str(write_images(tmp_path / "imgs", n=2))),
        sample_fps=5,
        zones=[Zone(zone_id="z1", name="Опасная", polygon=[(0.4, 0.5), (0.9, 0.5), (0.9, 1.0), (0.4, 1.0)])],
        rules=[RuleConfig(rule_id="s1", type="zone_intrusion", zones=["z1"], min_duration_s=1.0, close_after_s=1.0)],
    )


def _wait(fn, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return bool(fn())


def test_compact_snapshot_fits_slow_channel():
    import cv2

    rng = np.random.default_rng(0)
    base = np.tile(np.linspace(0, 255, 1920, dtype=np.uint8)[None, :, None], (1080, 1, 3))
    img = np.clip(base.astype(np.int16) + rng.integers(-40, 40, base.shape), 0, 255).astype(np.uint8)
    data = compact_jpeg(img, width=640, quality=70, max_kb=60)
    assert len(data) <= 60 * 1024
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)


def test_edge_to_gateway_event_meta_metrics(tmp_path):
    transport = MemoryTransport()
    cam = _camera(tmp_path)

    es = make_settings(tmp_path / "edge", autostart=False, mode="edge", clips_enabled=False)
    edge = PipelineEngine(es, backend=FakeBackend(), journal=MemoryJournal(), sink=transport)
    edge.cameras[cam.camera_id] = cam
    worker = CameraWorker(cam, edge)
    worker.detectors = {"det": ScriptedDetector(lambda i, f: [person((20 + i * 12, 60, 60 + i * 12, 200))] if i < 30 else [])}

    gs = make_settings(tmp_path / "gw", mode="gateway")
    save_cameras(gs.cameras_config, [cam])
    gw = GatewayEngine(gs, source=transport)
    app = create_app(gs, engine=gw)
    with TestClient(app) as client:
        frame = np.zeros((240, 320, 3), np.uint8)
        t0 = time.time() - 15
        for i in range(45):
            worker.process(FramePacket(frame=frame, ts=t0 + i * 0.2, seq=i + 1, decode_ms=1.0, pos_s=i * 0.2,
                                       capture_id=worker.source.capture_id))
        edge.publish_metrics()

        assert _wait(lambda: gw.journal.query()[1] == 1)
        ev = gw.journal.query()[0][0]
        assert ev.type == "zone_intrusion" and ev.camera_id == "cam1"
        assert gw.media.resolve(ev.snapshot_path) is not None  # снапшот пришёл с edge
        assert _wait(lambda: (gw.journal.get(ev.id).ended_at is not None))  # event_update
        pw = gw.workers["cam1"]
        assert pw.meta_count == 45
        assert _wait(lambda: pw.overlay_jpeg() is not None)
        assert _wait(lambda: gw.metrics_snapshot()["inference"].get("backend") == "fake")

        health = client.get("/health").json()
        assert health["mode"] == "gateway" and health["edge"]["connected"] is True
        listed = client.get("/api/events").json()
        assert listed["total"] == 1 and listed["items"][0]["id"] == ev.id
        assert client.get(f"/api/events/{ev.id}/snapshot.jpg").status_code == 200
        metrics = client.get("/api/metrics").json()
        assert metrics["mode"] == "gateway" and "cam1" in metrics["cameras"]
    edge.stop()


def test_gateway_ignores_redelivery_and_bad_messages(tmp_path):
    transport = MemoryTransport()
    gs = make_settings(tmp_path, mode="gateway")
    save_cameras(gs.cameras_config, [])
    gw = GatewayEngine(gs, source=transport)
    gw.start()
    try:
        from guard.contracts import Event, Severity
        from guard.transport import EventMsg

        ev = Event(id="e1", camera_id="x", site="s", rule_id="r", type="zone_intrusion", severity=Severity.high,
                   started_at=1.0, confirmed_at=2.0, bbox=(0, 0, 1, 1))
        transport.publish(EventMsg(kind="event", event=ev))
        transport.publish(EventMsg(kind="event", event=ev))  # at-least-once: повтор
        assert _wait(lambda: gw.messages["event"] == 2)
        assert gw.journal.query()[1] == 1
        gw.on_message(object())  # неизвестное сообщение не роняет приём
        assert gw.messages["?"] == 1 and gw.message_errors == 0
    finally:
        gw.stop()


def test_edge_cli_smoke(tmp_path, monkeypatch):
    from guard import edge as edge_cli

    folder = write_images(tmp_path / "imgs", n=2)
    cams = tmp_path / "cams.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [{
        "camera_id": "c", "name": "c", "site": "s", "source": {"type": "images", "uri": str(folder)}, "sample_fps": 5, "rules": [],
    }]}), encoding="utf-8")
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(edge_cli.Settings, "from_env", lambda: make_settings(tmp_path / "edge"))
    monkeypatch.setattr(edge_cli, "load_dotenv_guard", lambda: None)
    monkeypatch.setattr(edge_cli, "disable_ort_spinning", lambda: False)
    real_build = edge_cli.build
    built = []

    def build(args):
        result = real_build(args, backend=FakeBackend())
        built.append(result[1])
        return result

    monkeypatch.setattr(edge_cli, "build", build)
    args = edge_cli.parse_args(["--config", str(cams), "--transport", "log", "--duration", "1.0", "--metrics-period", "0.3"])
    settings, engine = edge_cli.build(args)
    from guard.edge.runner import EdgeEngine, EdgeRunner
    from guard.events.journal import Journal
    assert settings.mode == "edge" and not settings.clips_enabled and engine.sink is None
    assert isinstance(engine, EdgeEngine) and isinstance(engine.replication, EdgeRunner)
    assert isinstance(engine.journal, Journal)
    engine.stop()
    assert edge_cli.run(args) == 0
    assert all(item._stopped for item in built)
    from guard.transport.protocol import decode
    messages = [decode(wire) for wire in built[-1].replication.telemetry_publisher.telemetry]
    assert {message.kind for message in messages} >= {"cameras", "metrics"}
