"""Маховик не задерживает оповещение: запись в фоне, в горячем пути только постановка в очередь."""

import statistics
import threading
import time

import numpy as np

from guard.contracts import CameraConfig, Event, FrameContext, RuleConfig, Severity, SourceConfig
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings


def _engine(tmp_path, **overrides):
    settings = make_settings(tmp_path, autostart=False, clips_enabled=False, **overrides)
    return PipelineEngine(settings, backend=FakeBackend())


def _cam():
    return CameraConfig(camera_id="cam1", name="Цех", site="Цех 1", source=SourceConfig(type="file", uri="x.mp4"),
                        rules=[RuleConfig(rule_id="r", type="zone_intrusion")])


def _event(i):
    return Event(id=f"ev{i}", camera_id="cam1", site="Цех 1", rule_id="r", type="zone_intrusion",
                 severity=Severity.high, track_id=i, started_at=100.0 + i, confirmed_at=102.0 + i, bbox=(10, 10, 60, 160))


class _Worker:
    def __init__(self, cam):
        self.cfg = cam


def _ctx(cam, frame):
    return FrameContext(camera=cam, ts=time.time(), frame_idx=0, frame=frame, detections={"det": []},
                        timings={"decode": 1.0})


def test_slow_store_does_not_block_handle_events(tmp_path):
    engine = _engine(tmp_path)
    store = engine.flywheel_store
    gate = threading.Event()
    original = store.write_frame

    def slow_write(rel, data):  # диск «завис» на 0.5 с
        gate.wait(0.5)
        return original(rel, data)

    store.write_frame = slow_write
    engine.flywheel_writer.start()
    cam = _cam()
    frame = np.zeros((720, 1280, 3), np.uint8)
    try:
        t0 = time.perf_counter()
        engine.handle_events(_Worker(cam), _ctx(cam, frame), [_event(1)], [], time.perf_counter())
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.25  # снапшот и журнал — штатно; маховик ждать не заставляет
        assert engine.journal.get("ev1") is not None  # событие уже в журнале
        assert store.count() == 0  # сэмпл ещё пишется в фоне
        deadline = time.time() + 5
        while store.count() == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert store.count(sources=["event"]) == 1
    finally:
        gate.set()
        engine.stop()


def test_queue_overflow_drops_samples_without_waiting(tmp_path):
    engine = _engine(tmp_path)  # писатель не запущен: очередь только наполняется
    cam = _cam()
    frame = np.zeros((72, 128, 3), np.uint8)
    try:
        times = []
        for i in range(40):
            t0 = time.perf_counter()
            engine.flywheel.on_event(_ctx(cam, frame), _event(i))
            times.append(time.perf_counter() - t0)
        stats = engine.flywheel.stats()
        assert stats["submitted"]["event"] == 16 and stats["dropped"]["queue"] == 24
        assert max(times) < 0.05
    finally:
        engine.stop()


def test_handle_events_overhead_is_small(tmp_path):
    """Замер: handle_events с маховиком и без, кадр 1280×720, 30 событий (без клипов)."""
    cam = _cam()
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    def measure(engine):
        engine.flywheel_writer.start() if engine.flywheel_writer is not None else None
        out = []
        for i in range(30):
            t0 = time.perf_counter()
            engine.handle_events(_Worker(cam), _ctx(cam, frame), [_event(i)], [], time.perf_counter())
            out.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(out)

    on = _engine(tmp_path / "on")
    off = _engine(tmp_path / "off", flywheel=False)
    try:
        p50_on, p50_off = measure(on), measure(off)
    finally:
        on.stop()
        off.stop()
    # постановка в очередь — микросекунды; разница — шум снапшота (JPEG 1280×720 и запись файла)
    assert p50_on < p50_off + 15.0, (p50_on, p50_off)
