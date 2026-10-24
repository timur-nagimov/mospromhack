"""Маховик в режиме edge → gateway: части flywheel_sample, ограничение трафика, сборка на шлюзе, фейковый транспорт."""

import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from guard.contracts import CameraConfig, EventStatus, RuleConfig, SourceConfig, Zone
from guard.edge import GatewayQueueGuard
from guard.flywheel.collector import SampleJob
from guard.flywheel.edge import EdgeSender, SampleAssembler, TokenBucket
from guard.pipeline._fallbacks import MemoryJournal
from guard.pipeline.config import save_cameras
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.gateway import GatewayEngine
from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person, write_images
from guard.pipeline.worker import CameraWorker
from guard.sources import FramePacket
from guard.transport import FlywheelSampleMsg, MemoryTransport, decode, encode
from guard.transport.rabbit import _Outbox, _Packet


def _noise_frame(w=640, h=360, seed=0):
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(0, 255, w, dtype=np.uint8)[None, :, None], (h, 1, 3))
    return np.clip(base.astype(np.int16) + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)


class ListSink:
    def __init__(self, accept=True):
        self.messages = []
        self.accept = accept

    def publish(self, msg, **_):
        if self.accept is False or (callable(self.accept) and not self.accept(msg)):
            return False
        self.messages.append(msg)
        return True


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _job(source="uncertain", frame=None, sid=None):
    sample = {"camera_id": "cam1", "source": source, "created_at": 1.0, "reason": "r", "holdout": 1,
              "meta": {"focus_bbox": [100.0, 50.0, 300.0, 350.0]}}
    if sid:
        sample["id"] = sid
    det = person((100, 50, 300, 350), conf=0.42)
    return SampleJob(sample=sample, frame=frame if frame is not None else _noise_frame(1280, 720),
                     detections={"det": [det]}, tracks=[(7, "person", (100.0, 50.0, 300.0, 350.0))])


def test_codec_roundtrip_keeps_jpeg_outside_json():
    msg = FlywheelSampleMsg(sample_id="abc", part=0, parts=2, sample={"id": "abc", "camera_id": "c", "width": 640},
                            jpeg=b"\xff\xd8" + b"x" * 100)
    body, headers = encode(msg)
    assert headers["snapshot_length"] == 102 and b"xxxx" not in body[: 4 + headers["json_length"]]
    assert decode(body, headers) == msg
    empty = FlywheelSampleMsg(sample_id="abc", part=1, parts=2)
    assert decode(*encode(empty)).jpeg is None
    import json
    import struct

    raw = json.dumps({"kind": "flywheel_sample", "sample_id": "a", "part": 0, "parts": 1, "jpeg": "AAAA"}).encode()
    with pytest.raises(ValueError):
        decode(struct.pack("!I", len(raw)) + raw, {"guard_version": 1, "json_length": len(raw), "snapshot_length": -1})


def test_rabbit_outbox_counts_flywheel_drops():
    outbox = _Outbox(1)
    part = _Packet.make(FlywheelSampleMsg(sample_id="a", part=0, parts=1, jpeg=b"x"))
    assert outbox.put(part)
    newer = _Packet.make(FlywheelSampleMsg(sample_id="b", part=0, parts=1, jpeg=b"y"))
    assert outbox.put(newer)  # старая часть вытеснена, а не KeyError в счётчике
    assert outbox.dropped["flywheel_sample"] == 1 and outbox.dropped["frame_meta"] == 0
    assert outbox.take(0) is newer


def test_sender_splits_into_small_parts_and_assembler_restores_sample():
    sink = ListSink()
    clock = FakeClock()
    sender = EdgeSender(sink, kbps_per_camera=4.0, cameras=4, clock=clock, sleep=clock.sleep)
    assert sender.send(_job(sid="s1"))
    parts = sink.messages
    assert len(parts) >= 2 and all(len(m.jpeg) <= 8 * 1024 for m in parts)
    assert parts[0].sample is not None and all(m.sample is None for m in parts[1:])
    meta = parts[0].sample
    assert (meta["width"], meta["height"]) == (960, 540)  # ≤ 960 px на edge
    assert meta["detections"][0]["bbox"] == [75.0, 37.5, 225.0, 262.5] and meta["detections"][0]["track_id"] == 7
    assert meta["meta"]["focus_bbox"] == [75.0, 37.5, 225.0, 262.5]
    total = sum(len(m.jpeg) for m in parts)
    assert total <= 60 * 1024
    written = []
    asm = SampleAssembler(SimpleNamespace(put=lambda job: written.append(job) or True))
    for m in reversed(parts):  # порядок доставки не важен
        asm.add(m)
    assert len(written) == 1 and asm.stats()["completed"] == 1
    job = written[0]
    assert job.sample["id"] == "s1" and job.jpeg == b"".join(m.jpeg for m in parts)
    assert cv2.imdecode(np.frombuffer(job.jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (540, 960, 3)
    for m in parts:  # повторная доставка не даёт второй записи
        asm.add(m)
    assert len(written) == 1


def test_traffic_stays_under_budget():
    sink = ListSink()
    clock = FakeClock()
    sender = EdgeSender(sink, kbps_per_camera=4.0, cameras=1, clock=clock, sleep=clock.sleep)
    for i in range(6):
        assert sender.send(_job(sid=f"s{i}", frame=_noise_frame(seed=i)))
    sent = sum(len(m.jpeg) + 256 for m in sink.messages)
    burst = 2 * 8 * 1024
    assert clock.t >= (sent - burst) / (4 * 1024) - 1e-6  # средняя скорость не выше 4 КБ/с на камеру
    assert sent / max(clock.t, 1e-6) <= 10 * 1024  # контракт: прибавка не больше 10 КБ/с на камеру


def test_token_bucket():
    clock = FakeClock()
    bucket = TokenBucket(lambda: 1000.0, burst=500.0, clock=clock)
    assert bucket.wait_time(400) == 0
    bucket.take(400)
    assert bucket.wait_time(400) == pytest.approx(0.3)
    clock.t += 0.3
    assert bucket.wait_time(400) == 0


def test_event_samples_go_first_and_evict_regular_ones():
    sender = EdgeSender(ListSink(), queue_size=2)
    assert sender.put(_job("uncertain", sid="u1")) and sender.put(_job("flicker", sid="f1"))
    assert not sender.put(_job("uncertain", sid="u2"))  # обычный при полном буфере — отброшен
    assert sender.put(_job("event", sid="e1"))  # событие вытесняет самый старый обычный
    order = [sender._buffer.get(0.01).sample["id"] for _ in range(2)]
    assert order == ["e1", "f1"] and sender.stats()["evicted"] == 1 and sender.stats()["dropped_queue"] == 1


def test_held_channel_drops_whole_sample():
    sink = ListSink(accept=False)
    sender = EdgeSender(sink, kbps_per_camera=1000.0)
    assert sender.send(_job(sid="x")) is False
    assert sender.stats()["dropped_channel"] == 1 and sender.stats()["sent_parts"] == 0


def test_assembler_forgets_incomplete_samples():
    clock = FakeClock()
    written = []
    asm = SampleAssembler(SimpleNamespace(put=lambda job: written.append(job) or True), ttl_s=10.0, clock=clock)
    asm.add(FlywheelSampleMsg(sample_id="a", part=0, parts=2, sample={"camera_id": "c"}, jpeg=b"1"))
    clock.t = 20.0
    asm.add(FlywheelSampleMsg(sample_id="b", part=0, parts=1, sample={"camera_id": "c"}, jpeg=b"2"))
    assert asm.stats()["dropped"]["incomplete"] == 1 and len(written) == 1


def test_gateway_guard_pauses_flywheel_with_frame_meta():
    guard = GatewayQueueGuard("amqp://x", backlog_limit=10)
    guard.update(consumers=1, messages=50)
    assert not guard.allows("flywheel_sample") and not guard.allows("frame_meta") and guard.allows("event")
    guard.update(consumers=1, messages=2)
    assert guard.allows("flywheel_sample")
    guard.update(consumers=0, messages=0)
    assert not guard.allows("flywheel_sample")


def _wait(fn, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    return fn()


def test_edge_to_gateway_over_fake_transport(tmp_path):
    transport = MemoryTransport()
    cam = CameraConfig(
        camera_id="cam1", name="Цех", site="Цех 1",
        source=SourceConfig(type="images", uri=str(write_images(tmp_path / "imgs", n=2))),
        sample_fps=5,
        zones=[Zone(zone_id="z1", name="Опасная", polygon=[(0.4, 0.5), (0.9, 0.5), (0.9, 1.0), (0.4, 1.0)])],
        rules=[RuleConfig(rule_id="s1", type="zone_intrusion", zones=["z1"], min_duration_s=1.0, close_after_s=1.0)],
    )
    es = make_settings(tmp_path / "edge", autostart=False, mode="edge", clips_enabled=False, flywheel_edge_kbps=64.0)
    edge = PipelineEngine(es, backend=FakeBackend(), journal=MemoryJournal(), sink=transport)
    assert edge.flywheel_store is None and edge.flywheel is not None  # на edge журнала и хранилища нет
    edge.cameras[cam.camera_id] = cam
    edge.flywheel.output.start()
    worker = CameraWorker(cam, edge)
    worker.detectors = {"det": ScriptedDetector(lambda i, f: [person((200, 60, 240, 200))] if i < 20 else [])}

    gs = make_settings(tmp_path / "gw", mode="gateway")
    save_cameras(gs.cameras_config, [cam])
    gw = GatewayEngine(gs, source=transport)
    gw.start()
    try:
        frame = np.tile(np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], (240, 1, 3))  # гладкий кадр
        t0 = time.time() - 15
        for i in range(30):
            worker.process(FramePacket(frame=frame, ts=t0 + i * 0.2, seq=i + 1, decode_ms=1.0, pos_s=i * 0.2))
        ev = _wait(lambda: gw.journal.query()[0])[0]
        store = gw.flywheel_store
        sample = _wait(lambda: store.query(event_id=ev.id)[0])[0]
        assert sample["source"] == "event" and sample["camera_id"] == "cam1" and sample["holdout"] == 0
        assert (sample["width"], sample["height"]) == (320, 240)
        img = cv2.imread(str(store.resolve_frame(sample["frame_path"])))
        assert float(np.abs(img.astype(int) - frame.astype(int)).mean()) < 3.0  # сырой кадр без рамок, JPEG q80
        assert sample["detections"][0]["cls"] == "person"
        gw.patch_event(ev.id, status=EventStatus.false)
        assert store.get(sample["id"])["operator_label"] == "hard_negative"
        stats = edge.flywheel.stats()
        assert stats["output"]["sent_samples"] >= 1 and stats["output"]["rate_limit_kbps"] == 64.0
        assert edge.metrics_snapshot()["flywheel"]["submitted"]["event"] >= 1
    finally:
        edge.stop()
        gw.stop()
