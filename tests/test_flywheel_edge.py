"""Маховик edge → шлюз по протоколу edge → шлюз: конверты flywheel_sample, ограничение трафика, сборка на шлюзе.

Без сети: брокер — FakeBroker или фейковый канал pika.
"""

import base64
import io
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from guard.api.app import create_app
from guard.contracts import CameraConfig, EventStatus, RuleConfig, SourceConfig, Zone
from guard.edge.runner import EdgeEngine
from guard.flywheel import amqp
from guard.flywheel.collector import SampleJob
from guard.flywheel.edge import EdgeSender, SampleAssembler, TokenBucket
from guard.gateway.engine import GatewayEngine
from guard.pipeline._fallbacks import MemoryJournal
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person, write_images
from guard.pipeline.worker import CameraWorker
from guard.sources import FramePacket
from guard.transport.fake import FakeBroker
from guard.transport.protocol import Envelope, decode, encode
from guard.transport.rabbit import RabbitConfig

PRODUCER, RUN = str(uuid4()), str(uuid4())


class Messages:
    """make_message журнала edge: сквозная sequence без SQLite."""

    def __init__(self, edge_id="edge-1"):
        self.edge_id = edge_id
        self.sequence = 0

    def __call__(self, kind, payload):
        self.sequence += 1
        return Envelope(edge_id=self.edge_id, producer_id=PRODUCER, run_id=RUN, sequence=self.sequence,
                        message_id=f"{PRODUCER}:{self.sequence}", kind=kind, sent_at=time.time(), payload=payload)


class ListPublisher:
    def __init__(self, fail=None):
        self.envelopes = []
        self.fail = fail

    def publish(self, envelope, *, durable=True):
        assert durable is False
        if self.fail is not None:
            raise self.fail
        self.envelopes.append(envelope)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _noise_frame(w=640, h=360, seed=0):
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(0, 255, w, dtype=np.uint8)[None, :, None], (h, 1, 3))
    return np.clip(base.astype(np.int16) + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)


def _job(source="uncertain", frame=None, sid=None, **extra):
    sample = {"camera_id": "cam1", "source": source, "created_at": 1.0, "reason": "r", "holdout": 1,
              "event_id": None, "meta": {"focus_bbox": [100.0, 50.0, 300.0, 350.0]}, **extra}
    if sid:
        sample["id"] = sid
    det = person((100, 50, 300, 350), conf=0.42)
    return SampleJob(sample=sample, frame=frame if frame is not None else _noise_frame(1280, 720),
                     detections={"det": [det]}, tracks=[(7, "person", (100.0, 50.0, 300.0, 350.0))])


def _sender(publisher=None, clock=None, **kwargs):
    clock = clock or FakeClock()
    return EdgeSender(publisher if publisher is not None else ListPublisher(), Messages(),
                      clock=clock, sleep=clock.sleep, **kwargs)


def _jpeg(color="red", size=(64, 48)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="JPEG")
    return out.getvalue()


def _part(sample_id="s1", part=0, parts=1, sample=None, chunk=b"\xff\xd8xx\xff\xd9", seq=1):
    if part == 0 and sample is None:
        sample = {"id": sample_id, "camera_id": "cam1", "source": "uncertain", "width": 64, "height": 48}
    return Messages()("flywheel_sample", {"sample_id": sample_id, "part": part, "parts": parts, "sample": sample,
                                          "jpeg_b64": base64.b64encode(chunk).decode()})


def _writer():
    written = []
    return written, SimpleNamespace(put=lambda job: written.append(job) or True)


# ---------------------------------------------------------------- протокол


def test_protocol_accepts_parts_and_rejects_paths_and_foreign_fields():
    env = _part()
    assert decode(encode(env)) == env
    good = env.payload
    bad_payloads = [
        {**good, "extra": 1},
        {**good, "sample_id": "../x"},
        {**good, "part": 1},  # part ≥ parts
        {**good, "part": 1, "parts": 2},  # метаданные только в части 0
        {**good, "sample": None},
        {**good, "sample": {**good["sample"], "frame_path": "/etc/passwd"}},
        {**good, "sample": {**good["sample"], "status": "accepted"}},
        {**good, "sample": {**good["sample"], "camera_id": "../../etc"}},
        {**good, "sample": {**good["sample"], "id": "other"}},
        {**good, "sample": {**good["sample"], "width": 0}},
        {**good, "jpeg_b64": "not base64!"},
        {**good, "jpeg_b64": ""},
        {**good, "jpeg_b64": base64.b64encode(b"x" * (64 * 1024 + 1)).decode()},
        {**good, "parts": 65},
    ]
    for payload in bad_payloads:
        with pytest.raises(ValueError):
            Messages()("flywheel_sample", payload)


# ---------------------------------------------------------------- edge: отправитель


def test_sender_splits_into_small_parts_and_assembler_restores_sample():
    publisher = ListPublisher()
    sender = _sender(publisher, kbps_per_camera=4.0, cameras=4)
    assert sender.send(_job(sid="s1", status="accepted", frame_path="/tmp/x.jpg"))
    parts = publisher.envelopes
    chunks = [base64.b64decode(env.payload["jpeg_b64"]) for env in parts]
    assert len(parts) >= 2 and all(len(c) <= 8 * 1024 for c in chunks)
    assert all(env.kind == "flywheel_sample" and decode(encode(env)) == env for env in parts)
    assert [env.sequence for env in parts] == list(range(1, len(parts) + 1))
    meta = parts[0].payload["sample"]
    assert all(env.payload["sample"] is None for env in parts[1:])
    assert "status" not in meta and "frame_path" not in meta  # edge не задаёт статус и путь
    assert (meta["width"], meta["height"]) == (960, 540)  # ≤ 960 px на edge
    assert meta["detections"][0]["bbox"] == [75.0, 37.5, 225.0, 262.5] and meta["detections"][0]["track_id"] == 7
    assert meta["meta"]["focus_bbox"] == [75.0, 37.5, 225.0, 262.5]
    assert sum(len(c) for c in chunks) <= 60 * 1024
    assert sender.stats()["sent_kb"] == round(sum(len(encode(e)) for e in parts) / 1024, 1)
    written, writer = _writer()
    asm = SampleAssembler(writer)
    for env in reversed(parts):  # порядок доставки не важен
        asm.add_envelope(decode(encode(env)))
    assert len(written) == 1 and asm.stats()["completed"] == 1 and asm.stats()["pending"] == 0
    job = written[0]
    assert job.sample["id"] == "s1" and job.jpeg == b"".join(chunks)
    assert "status" not in job.sample and "operator_label" not in job.sample and "frame_path" not in job.sample
    assert (job.sample["width"], job.sample["height"]) == (960, 540) and job.sample["holdout"] == 1
    assert cv2.imdecode(np.frombuffer(job.jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (540, 960, 3)
    for env in parts:  # повторная доставка не даёт второй записи
        asm.add_envelope(env)
    assert len(written) == 1


def test_traffic_stays_under_budget():
    clock = FakeClock()
    publisher = ListPublisher()
    sender = _sender(publisher, clock=clock, kbps_per_camera=4.0, cameras=1)
    for i in range(6):
        assert sender.send(_job(sid=f"s{i}", frame=_noise_frame(seed=i)))
    sent = sum(len(encode(env)) for env in publisher.envelopes)  # байты конверта: base64 и метаданные тоже
    burst = 3 * 8 * 1024
    assert sent > 20 * 1024
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
    bucket.take(400)
    clock.t += 10
    assert bucket.wait_time(5000) == 0  # часть больше ведра ждёт полного ведра, а не вечность
    bucket.take(5000)
    assert bucket.wait_time(100) == pytest.approx(4.6)  # долг возвращается по скорости ведра


def test_event_samples_go_first_and_evict_regular_ones():
    sender = _sender(queue_size=2)
    assert sender.put(_job("uncertain", sid="u1")) and sender.put(_job("flicker", sid="f1"))
    assert not sender.put(_job("uncertain", sid="u2"))  # обычный при полном буфере — отброшен
    assert sender.put(_job("event", sid="e1"))  # событие вытесняет самый старый обычный
    order = [sender._buffer.get(0.01).sample["id"] for _ in range(2)]
    assert order == ["e1", "f1"] and sender.stats()["evicted"] == 1 and sender.stats()["dropped_queue"] == 1


def test_publish_failure_drops_whole_sample_and_pauses():
    publisher = ListPublisher(fail=ConnectionError("broker down"))
    sender = EdgeSender(publisher, Messages(), kbps_per_camera=1000.0)
    sender.start()
    try:
        assert sender.put(_job(sid="x"))
        deadline = time.time() + 5
        while sender.stats()["errors"] == 0 and time.time() < deadline:
            time.sleep(0.02)
        stats = sender.stats()
        assert stats["errors"] == 1 and stats["dropped_channel"] == 1 and stats["sent_parts"] == 0
        assert sender.last_error == "ConnectionError"
    finally:
        sender.stop()


def test_publisher_factory_is_resolved_in_sender_thread_once():
    created = []

    def factory():
        created.append(threading.current_thread().name)
        return ListPublisher()

    sender = EdgeSender(factory, Messages(), kbps_per_camera=1000.0)
    assert created == []  # движок edge ещё не построен: соединения нет
    sender.start()
    try:
        assert sender.put(_job(sid="a")) and sender.put(_job(sid="b"))
        deadline = time.time() + 5
        while sender.stats()["sent_samples"] < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert sender.stats()["sent_samples"] == 2 and created == ["flywheel-edge-sender"]
    finally:
        sender.stop()


# ---------------------------------------------------------------- шлюз: сборщик


def test_assembler_forgets_incomplete_and_rejects_bad_samples():
    clock = FakeClock()
    written, writer = _writer()
    asm = SampleAssembler(writer, ttl_s=10.0, clock=clock, max_pending=2)
    jpeg = _jpeg()
    asm.add_envelope(_part("a", 0, 2, chunk=jpeg[:10]))
    clock.t = 20.0
    assert asm.add_envelope(_part("b", 0, 1, chunk=jpeg))
    assert asm.stats()["dropped"]["incomplete"] == 1 and len(written) == 1
    assert (written[0].sample["width"], written[0].sample["height"]) == (64, 48)  # размеры — из JPEG
    # не JPEG и неизвестный source — в invalid, без записи
    assert not asm.add_envelope(_part("c", 0, 1, chunk=b"\xff\xd8garbage\xff\xd9"))
    sample = {"camera_id": "cam1", "source": "bogus", "width": 64, "height": 48}
    assert not asm.add_envelope(_part("d", 0, 1, sample=sample, chunk=jpeg))
    # другое число частей у того же sample_id — сэмпл бросается
    asm.add_envelope(_part("e", 0, 3, chunk=jpeg[:10]))
    assert not asm.add_envelope(_part("e", 1, 2, chunk=jpeg[10:20]))
    assert asm.stats()["dropped"]["invalid"] == 3 and len(written) == 1
    # лимит незавершённых: вытесняется самый старый
    for sid in ("f", "g", "h"):
        asm.add_envelope(_part(sid, 0, 2, chunk=jpeg[:10]))
    assert asm.stats()["pending"] == 2 and asm.stats()["dropped"]["overflow"] == 1


# ---------------------------------------------------------------- AMQP: своя очередь в topology v2


class FakeChannel:
    def __init__(self, queue=()):
        self.calls = []
        self.queue = list(queue)
        self.acked, self.rejected = [], []

    def exchange_declare(self, **kw):
        self.calls.append(("exchange", kw))

    def queue_declare(self, **kw):
        self.calls.append(("queue", kw))

    def queue_bind(self, **kw):
        self.calls.append(("bind", kw))

    def confirm_delivery(self):
        self.calls.append(("confirm", {}))

    def basic_qos(self, **kw):
        self.calls.append(("qos", kw))

    def basic_publish(self, **kw):
        self.calls.append(("publish", kw))

    def basic_get(self, queue, auto_ack):
        assert queue == "guard.gateway.edge-1.flywheel.v1" and auto_ack is False
        if not self.queue:
            return None, None, None
        return SimpleNamespace(delivery_tag=len(self.queue)), None, self.queue.pop(0)

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_reject(self, delivery_tag, requeue):
        assert requeue is False
        self.rejected.append(delivery_tag)


class FakeConnection:
    def __init__(self, channel):
        self._channel = channel
        self.is_open = True

    def channel(self):
        return self._channel

    def process_data_events(self, time_limit=0):
        pass

    def close(self):
        self.is_open = False


def test_flywheel_queue_publisher_and_consumer():
    config = RabbitConfig("amqp://guest:hidden@localhost/", "edge-1")
    channel = FakeChannel()
    publisher = amqp.FlywheelPublisher(config, connection_factory=lambda cfg: FakeConnection(channel))
    publisher.publish(_part())
    kinds = [name for name, _ in channel.calls]
    assert kinds == ["exchange", "queue", "bind", "confirm", "publish"]
    queue = dict(channel.calls)["queue"]
    assert queue["queue"] == "guard.gateway.edge-1.flywheel.v1" and queue["durable"] is True
    assert queue["arguments"] == {"x-message-ttl": 120000, "x-max-length": 256, "x-max-length-bytes": 4194304}
    publish = dict(channel.calls)["publish"]
    assert publish["exchange"] == "guard.edge.v1" and publish["routing_key"] == "edge-1.flywheel"
    assert publish["properties"].delivery_mode == 1 and publish["properties"].expiration == "120000"
    with pytest.raises(ValueError):
        publisher.publish(_part(), durable=True)
    with pytest.raises(ValueError):
        publisher.publish(Messages()("metrics", {"data": {}}))
    assert "hidden" not in repr(config) and publisher.status()["published"] == 1

    wrong_edge = Messages("edge-2")("flywheel_sample", _part().payload)
    metrics = Messages()("metrics", {"data": {}})
    channel = FakeChannel([encode(_part("ok")), b"{", encode(wrong_edge), encode(metrics), encode(_part("boom"))])
    consumer = amqp.FlywheelConsumer(config, connection_factory=lambda cfg: FakeConnection(channel))
    seen = []

    def callback(envelope):
        if envelope.payload["sample_id"] == "boom":
            raise RuntimeError("writer failed")
        seen.append(envelope.payload["sample_id"])

    assert sum(consumer.poll(callback) for _ in range(6)) == 5
    assert seen == ["ok"] and len(channel.acked) == 1 and len(channel.rejected) == 4
    status = consumer.status()
    assert status["received"] == 1 and status["rejected"] == 3 and status["errors"] == 1
    assert ("qos", {"prefetch_count": 1}) in channel.calls


def test_edge_publisher_choice():
    class Runner:
        telemetry_publisher = None

    from guard.transport.rabbit import RabbitPublisher

    engine = SimpleNamespace(replication=Runner())
    assert isinstance(amqp.edge_publisher(engine), amqp.MemoryFlywheelPublisher)
    broker = FakeBroker()
    engine.replication.telemetry_publisher = broker
    assert amqp.edge_publisher(engine) is broker  # тесты и отладка: без сети
    engine.replication.telemetry_publisher = RabbitPublisher(RabbitConfig("amqp://guest:x@localhost/", "edge-1"))
    publisher = amqp.edge_publisher(engine)
    assert isinstance(publisher, amqp.FlywheelPublisher) and publisher.config.edge_id == "edge-1"
    assert publisher.status()["connected"] is False  # соединение — только при первой части


# ---------------------------------------------------------------- edge → шлюз целиком


def _wait(fn, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    return fn()


def _camera(tmp_path):
    return CameraConfig(
        camera_id="cam1", name="Цех", site="Цех 1",
        source=SourceConfig(type="images", uri=str(write_images(tmp_path / "imgs", n=2))),
        sample_fps=5,
        zones=[Zone(zone_id="z1", name="Опасная", polygon=[(0.4, 0.5), (0.9, 0.5), (0.9, 1.0), (0.4, 1.0)])],
        rules=[RuleConfig(rule_id="s1", type="zone_intrusion", zones=["z1"], min_duration_s=1.0, close_after_s=1.0)],
    )


def test_edge_engine_to_gateway_over_fake_broker(tmp_path):
    broker = FakeBroker()
    cam = _camera(tmp_path)
    url = "amqp://guest:hidden@localhost/"
    es = make_settings(tmp_path / "edge", autostart=False, mode="edge", clips_enabled=False,
                       flywheel_edge_kbps=64.0, rabbit_url=url)
    edge = EdgeEngine(es, backend=FakeBackend(), publisher=broker, telemetry_publisher=broker)
    assert edge.flywheel_store is None and edge.flywheel is not None  # на edge хранилища нет
    edge.cameras[cam.camera_id] = cam
    edge.flywheel.start()
    worker = CameraWorker(cam, edge)
    edge.workers[cam.camera_id] = worker
    worker.detectors = {"det": ScriptedDetector(lambda i, f: [person((200, 60, 240, 200))] if i < 20 else [])}

    gs = make_settings(tmp_path / "gw", autostart=False, mode="gateway", rabbit_url=url)
    gateway = GatewayEngine(gs, consumer=broker)
    assert gateway.flywheel_receiver is None  # FakeBroker — части приходят через receive
    gateway.flywheel_writer.start()
    try:
        frame = np.tile(np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], (240, 1, 3))  # гладкий кадр
        t0 = time.time() - 15
        for i in range(30):
            worker.process(FramePacket(frame=frame, ts=t0 + i * 0.2, seq=i + 1, decode_ms=1.0,
                                       capture_id=worker.source.capture_id))
        assert edge.replication.publish_pending_once() >= 1  # событие — durable outbox, как раньше
        assert _wait(lambda: edge.flywheel.output.stats()["sent_samples"] >= 1)
        assert edge.replication.publish_telemetry_once() >= 2  # каталог и metrics со сводкой маховика
        while broker.events:
            broker.deliver(gateway.receive)
        ev = gateway.journal.query()[0][0]
        while broker.telemetry:
            broker.deliver(gateway.receive, durable=False)
        store = gateway.flywheel_store
        sample = _wait(lambda: store.query(event_id=ev.id)[0])[0]
        assert sample["source"] == "event" and sample["camera_id"] == "cam1" and sample["holdout"] == 0
        assert sample["status"] == "new" and sample["operator_label"] is None
        assert (sample["width"], sample["height"]) == (320, 240)
        img = cv2.imread(str(store.resolve_frame(sample["frame_path"])))
        assert float(np.abs(img.astype(int) - frame.astype(int)).mean()) < 3.0  # сырой кадр без рамок, JPEG q80
        assert sample["detections"][0]["cls"] == "person" and sample["meta"]["event_type"] == "zone_intrusion"
        gateway.patch_event(ev.id, status=EventStatus.false)
        assert store.get(sample["id"])["operator_label"] == "hard_negative"
        stats = edge.flywheel.stats()
        assert stats["output"]["sent_samples"] >= 1 and stats["output"]["rate_limit_kbps"] == 64.0
        assert edge.metrics_snapshot()["flywheel"]["submitted"]["event"] >= 1
        with TestClient(create_app(gs, gateway)) as client:
            summary = client.get("/api/flywheel/summary").json()
            assert summary["mode"] == "gateway" and summary["total"] >= 1
            assert summary["receiver"]["assembler"]["completed"] >= 1
            assert summary["collector"]["submitted"]["event"] >= 1 and summary["collecting"] is True
            frame_resp = client.get(f"/api/flywheel/samples/{sample['id']}/frame.jpg")
            assert frame_resp.status_code == 200 and frame_resp.content[:2] == b"\xff\xd8"
    finally:
        edge.flywheel.stop()
        worker.stop()
        edge.workers.clear()
        edge.stop()
        gateway.stop()


def test_legacy_sink_engine_does_not_collect_on_edge(tmp_path):
    es = make_settings(tmp_path, autostart=False, mode="edge", clips_enabled=False)
    edge = PipelineEngine(es, backend=FakeBackend(), journal=MemoryJournal(), sink=FakeBroker())
    try:
        assert edge.flywheel is None and edge.flywheel_store is None
    finally:
        edge.stop()
