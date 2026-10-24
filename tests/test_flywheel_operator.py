"""Маховик в конвейере: сэмпл event с сырым кадром, перенос решения оператора в operator_label."""

import time

import cv2
import numpy as np

from guard.contracts import CameraConfig, EventStatus, RuleConfig, SourceConfig, Zone
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person, write_images
from guard.pipeline.worker import CameraWorker
from guard.sources import FramePacket


def _setup(tmp_path, start_writer=True):
    settings = make_settings(tmp_path, autostart=False)
    engine = PipelineEngine(settings, backend=FakeBackend())
    cam = CameraConfig(
        camera_id="cam1", name="Цех", site="Цех 1",
        source=SourceConfig(type="images", uri=str(write_images(tmp_path / "imgs", n=2))),
        sample_fps=5,
        zones=[Zone(zone_id="z1", name="Опасная", polygon=[(0.4, 0.5), (0.9, 0.5), (0.9, 1.0), (0.4, 1.0)])],
        rules=[RuleConfig(rule_id="s1", type="zone_intrusion", zones=["z1"], min_duration_s=1.0, close_after_s=1.0)],
    )
    engine.cameras[cam.camera_id] = cam
    worker = CameraWorker(cam, engine)
    worker.detectors = {"det": ScriptedDetector(lambda i, f: [person((200, 60, 240, 200))])}  # стоит в зоне
    if start_writer:
        engine.flywheel_writer.start()
    return engine, worker


def _frame():
    """Сырой кадр с узнаваемым узором: оверлеи снапшота его бы испортили."""
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :, 1] = np.tile(np.linspace(0, 200, 320, dtype=np.uint8), (240, 1))
    return img


def _wait(fn, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    return fn()


def _run_frames(worker, n=20):
    frame = _frame()
    t0 = time.time() - 20
    for i in range(n):
        worker.process(FramePacket(frame=frame, ts=t0 + i * 0.2, seq=i + 1, decode_ms=1.0,
                                   capture_id=worker.source.capture_id))
    return frame


def test_event_sample_has_raw_frame_and_all_detections(tmp_path):
    engine, worker = _setup(tmp_path)
    published = []
    engine.bus.publish = published.append
    try:
        raw = _run_frames(worker)
        events, total = engine.journal.query()
        assert total == 1
        ev = events[0]
        store = engine.flywheel_store
        items = _wait(lambda: store.query(source="event")[0])
        assert len(items) == 1
        sample = items[0]
        assert sample["event_id"] == ev.id and sample["camera_id"] == "cam1" and sample["holdout"] == 0
        assert (sample["width"], sample["height"]) == (320, 240)
        assert sample["detections"] == [{"model_id": "det", "cls": "person", "conf": 0.9,
                                         "bbox": [200.0, 60.0, 240.0, 200.0], "track_id": ev.track_id}]
        assert sample["meta"]["event_type"] == "zone_intrusion" and sample["meta"]["event_bbox"] == list(ev.bbox)
        img = cv2.imread(str(store.resolve_frame(sample["frame_path"])))
        assert img.shape == raw.shape
        assert float(np.abs(img.astype(int) - raw.astype(int)).mean()) < 2.0  # без рамок и зон (только JPEG)
        snap = cv2.imread(str(engine.media.resolve(ev.snapshot_path)))
        assert float(np.abs(snap.astype(int) - raw.astype(int)).mean()) > 2.0  # а снапшот — с оверлеями
        ws = [m for m in published if m.get("kind") == "flywheel_sample"]
        assert ws and ws[0]["type"] == "flywheel_sample" and ws[0]["sample"]["event_id"] == ev.id
        assert ws[0]["sample"]["frame_url"] == f"/api/flywheel/samples/{sample['id']}/frame.jpg"
    finally:
        engine.stop()


def test_operator_decision_moves_to_event_sample(tmp_path):
    engine, worker = _setup(tmp_path)
    try:
        _run_frames(worker)
        ev = engine.journal.query()[0][0]
        store = engine.flywheel_store
        sample = _wait(lambda: store.query(event_id=ev.id)[0])[0]
        assert sample["operator_label"] is None
        for status, label in ((EventStatus.false, "hard_negative"), (EventStatus.confirmed, "positive"),
                              (EventStatus.forwarded, "forwarded"), (EventStatus.new, None)):
            engine.patch_event(ev.id, status=status)
            assert store.get(sample["id"])["operator_label"] == label
        engine.patch_event(ev.id, operator_note="без статуса")  # заметка не трогает метку
        assert store.get(sample["id"])["operator_label"] is None
    finally:
        engine.stop()


def test_operator_decision_before_sample_is_written(tmp_path):
    """Оператор успел нажать «ложное» раньше, чем фон записал сэмпл: метка берётся из журнала при записи."""
    engine, worker = _setup(tmp_path, start_writer=False)
    try:
        _run_frames(worker)
        ev = engine.journal.query()[0][0]
        engine.patch_event(ev.id, status=EventStatus.false)
        assert engine.flywheel_store.query(event_id=ev.id)[1] == 0  # фон ещё не писал
        engine.flywheel_writer.start()
        sample = _wait(lambda: engine.flywheel_store.query(event_id=ev.id)[0])[0]
        assert sample["operator_label"] == "hard_negative"
    finally:
        engine.stop()


def test_flywheel_can_be_disabled(tmp_path):
    settings = make_settings(tmp_path, autostart=False, flywheel=False)
    engine = PipelineEngine(settings, backend=FakeBackend())
    try:
        assert engine.flywheel is not None and engine.flywheel.enabled is False
        assert engine.flywheel_store is not None  # проверка и экспорт работают и без сбора
    finally:
        engine.stop()
