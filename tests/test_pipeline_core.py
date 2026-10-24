"""Конвейер: источники, кольцевой буфер, клипы H.264, метрики, бэкенды, сквозной S1 на синтетике."""

import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from guard.contracts import TIMING_KEYS, CameraConfig, Event, ModelSpec, RuleConfig, Severity, SourceConfig, Zone
from guard.events.media import FrameRing, MediaStore, write_clip
from guard.inference import get_backend, normalize_backend_name
from guard.metrics import MetricsRegistry, RateMeter, RollingStats
from guard.pipeline import _fallbacks as fb
from guard.pipeline.config import load_cameras, load_models, resolve_source_uri, save_cameras
from guard.pipeline.detector import BatchScheduler, Detector
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person, write_images
from guard.pipeline.worker import CameraWorker
from guard.sources import FramePacket, make_source


def ffprobe(path: Path) -> str:
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return out.stderr


# ---------- источники ----------


def _make_video(path: Path, n: int = 40, fps: float = 20.0) -> Path:
    frames = []
    for i in range(n):
        img = np.zeros((120, 160, 3), np.uint8)
        cv2.putText(img, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        frames.append(img)
    write_clip(frames, path, fps)
    return path


def test_file_source_realtime_keeps_only_latest(tmp_path):
    video = _make_video(tmp_path / "v.mp4", n=40, fps=20)
    seen = []
    src = make_source(SourceConfig(type="file", uri=str(video), loop=True), str(video), "c", on_frame=lambda p: seen.append(p.seq))
    src.start()
    try:
        pkt = src.wait_newer(0, timeout=5)
        assert pkt is not None and pkt.frame.shape == (120, 160, 3)
        time.sleep(0.5)
        latest = src.latest()
        assert latest.seq >= 5  # темп ~20 к/с, а не «как можно быстрее»
        assert latest.seq <= 20
        assert abs(latest.ts - time.time()) < 0.5  # слот содержит свежий кадр, без очереди
        assert seen == list(range(1, len(seen) + 1))  # колбэк получает каждый кадр
    finally:
        src.stop()
    assert src.status()["native_fps"] == pytest.approx(20, abs=0.5)


def test_file_source_loops(tmp_path):
    video = _make_video(tmp_path / "short.mp4", n=6, fps=30)
    src = make_source(SourceConfig(type="file", uri=str(video), loop=True), str(video), "c")
    src.start()
    try:
        time.sleep(0.8)
        assert src.latest().seq > 6  # пошёл второй круг
    finally:
        src.stop()


def test_images_source_paces_by_fps(tmp_path):
    folder = write_images(tmp_path / "imgs", n=3)
    src = make_source(SourceConfig(type="images", uri=str(folder)), str(folder), "c", fps=10)
    src.start()
    try:
        time.sleep(0.55)
        n = src.latest().seq
        assert 4 <= n <= 8
    finally:
        src.stop()


def test_resolve_source_uri(tmp_path):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    (data / "videos" / "a.mp4").write_bytes(b"x")
    assert resolve_source_uri("videos/a.mp4", data) == str(data / "videos" / "a.mp4")
    assert resolve_source_uri("data/videos/a.mp4", data) == str(data / "videos" / "a.mp4")
    assert resolve_source_uri("/abs/x.mp4", data) == "/abs/x.mp4"


# ---------- буфер и клипы ----------


def test_ring_decimates_downscales_and_caps():
    ring = FrameRing(seconds=100, fps=10, max_width=640, max_frames=150)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    accepted = sum(ring.push(frame, i / 60.0) for i in range(60 * 30))  # 30 с при 60 к/с
    assert 290 <= accepted <= 310  # ~10 к/с
    assert len(ring) == 150  # жёсткий потолок по числу кадров
    assert ring.snapshot()[-1][1].shape == (360, 640, 3)


def test_ring_budget_downscales_further():
    ring = FrameRing(seconds=100, fps=10, max_width=640, budget_bytes=20 * 640 * 360 * 3, max_frames=150)
    frame = np.zeros((360, 640, 3), np.uint8)
    for i in range(100):
        ring.push(frame, i * 0.1)
    assert ring.downscaled and ring.max_width < 640
    assert ring.stats()["mb"] <= 20 * 640 * 360 * 3 / 2**20 + 0.01


def test_write_clip_is_browser_h264(tmp_path):
    frames = [np.full((241, 321, 3), i * 10 % 255, np.uint8) for i in range(15)]  # нечётный размер
    path = tmp_path / "c.mp4"
    write_clip(frames, path, fps=10)
    info = ffprobe(path)
    assert "h264" in info and "yuv420p" in info and "320x240" in info


def test_media_store_streaming_clip_and_snapshot(tmp_path):
    media = MediaStore(tmp_path, pre_s=1.0, post_s=0.5, clip_fps=10, max_width=320)
    now = time.time()
    for i in range(25):  # 2.5 с кадров до события
        media.push_frame("cam", np.full((240, 320, 3), i * 8, np.uint8), now - 2.5 + i * 0.1)
    ev = Event(
        id="e1", camera_id="cam", site="s", rule_id="r", type="zone_intrusion", severity=Severity.high,
        started_at=now - 1.0, confirmed_at=now, bbox=(1, 2, 3, 4),
    )
    rel = media.save_snapshot(ev, np.zeros((240, 320, 3), np.uint8))
    assert media.resolve(rel) is not None
    done = []
    media.schedule_clip(ev, on_done=lambda e, r: done.append(r))
    for i in range(1, 8):  # кадры после события приходят «вживую»
        time.sleep(0.1)
        media.push_frame("cam", np.full((240, 320, 3), 200, np.uint8), now + i * 0.1)
    deadline = time.time() + 10
    while not done and time.time() < deadline:
        time.sleep(0.05)
    media.stop()
    assert done and done[0]
    path = media.resolve(done[0])
    info = ffprobe(path)
    assert "h264" in info and "yuv420p" in info
    assert media.resolve("../../etc/passwd") is None


# ---------- метрики ----------


def test_rolling_stats_and_rate():
    st = RollingStats()
    for v in range(1, 101):
        st.add(v)
    s = st.summary()
    assert s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05)
    rm = RateMeter(window_s=5)
    for i in range(50):
        rm.tick(100 + i * 0.1)
    assert rm.rate(now=104.9) == pytest.approx(10, rel=0.05)
    assert rm.rate(now=200) == 0.0


def test_camera_metrics_snapshot_keys():
    reg = MetricsRegistry()
    cm = reg.camera("c")
    cm.record_frame({k: 1.0 for k in TIMING_KEYS}, capture_ts=time.time() - 0.05)
    snap = reg.snapshot()["c"]
    assert set(snap["timings_ms"]) == set(TIMING_KEYS)
    assert 0.0 <= snap["lag_s"] < 1.0


# ---------- инференс ----------


def test_get_backend_names_and_lazy_tpu():
    assert normalize_backend_name("ONNX") == "onnx-cpu"
    be = get_backend("onnx-cpu", models_dir="/nonexistent", unknown_kwarg=1)
    assert be.name == "onnx-cpu"
    with pytest.raises(ValueError):
        get_backend("nope")


def test_onnx_backend_real_model_if_available():
    from guard.pipeline.settings import Settings

    s = Settings.from_env()
    specs = load_models(s.models_config, s.models_dir)
    if "det" not in specs or not (s.models_dir / specs["det"].onnx_path).exists():
        pytest.skip("det model not available")
    be = get_backend("onnx-cpu", models_dir=s.models_dir, intra_op_threads=2)
    be.load(specs["det"])
    det = Detector(specs["det"], be)
    timings: dict[str, float] = {}
    dets = det.detect(np.zeros((480, 640, 3), np.uint8), timings)
    assert dets == [] and timings["infer"] > 0
    assert be.class_names("det")[0] == "person"
    st = be.stats()
    assert st["device"] == "cpu" and "det" in st["models"] and 0 <= st["utilization"] <= 1


def test_batch_scheduler_merges_cameras():
    be = FakeBackend(max_batch=4)
    spec = ModelSpec(model_id="det", onnx_path="x", classes=["person"] * 80)
    be.load(spec)
    sched = BatchScheduler(be, "det", max_batch=4, timeout_s=0.05)
    try:
        futs = [sched.submit(np.zeros((1, 3, 640, 640), np.float32)) for _ in range(3)]
        results = [f.result(timeout=2) for f in futs]
    finally:
        sched.stop()
    assert be.calls == 1  # три кадра — один вызов infer
    assert all(r[0][0].shape[0] == 1 and r[1]["batch"] == 3 for r in results)


# ---------- заглушки (запасной вариант модулей Codex) ----------


def test_fallback_letterbox_decode_roundtrip():
    spec = ModelSpec(model_id="det", onnx_path="x", classes=["person", "car"], conf_threshold=0.3)
    img = np.zeros((480, 640, 3), np.uint8)
    tensor, meta = fb.letterbox(img, (640, 640))
    assert tensor.shape == (1, 3, 640, 640) and meta.pad_y == 80
    out = np.zeros((1, 6, 100), np.float32)
    out[0, :4, 0] = [320, 320, 100, 200]  # cx, cy, w, h во входе модели
    out[0, 4, 0] = 0.9
    out[0, :4, 1] = [322, 321, 100, 200]  # дубль — NMS
    out[0, 4, 1] = 0.8
    dets = fb.decode_yolov8([out], meta, spec)
    assert len(dets) == 1 and dets[0].cls == "person"
    assert dets[0].bbox == pytest.approx((270, 140, 370, 340), abs=1)


def test_fallback_tracker_and_event_manager_one_event():
    trk = fb.IouTracker(classes={"person"}, high_thresh=0.4, new_track_thresh=0.45, min_hits=2)
    cam = CameraConfig(
        camera_id="c", name="c", site="s", source=SourceConfig(type="file", uri="x"),
        rules=[RuleConfig(rule_id="r", type="zone_intrusion", min_duration_s=2.0)],
    )
    em = fb.SimpleEventManager()
    from guard.contracts import Violation

    created = []
    for i in range(40):  # 8 с при 5 к/с
        ts = 100 + i * 0.2
        tracks = trk.update([person((100 + i, 100, 150 + i, 300))], ts)
        viol = [Violation("r", "zone_intrusion", Severity.high, t.track_id, None, t.bbox) for t in tracks]
        new, _ = em.process(cam, ts, viol)
        created += new
    assert len(created) == 1
    assert created[0].confirmed_at - created[0].started_at == pytest.approx(2.0, abs=0.25)


# ---------- конфиги ----------


def test_cameras_yaml_roundtrip(tmp_path):
    cam = CameraConfig(
        camera_id="c1", name="Склад", site="Склад", source=SourceConfig(type="file", uri="videos/a.mp4"),
        zones=[Zone(zone_id="z", name="Зона", polygon=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)], active_hours=["08:00-20:00"])],
        rules=[RuleConfig(rule_id="r", type="zone_intrusion", zones=["z"])],
    )
    path = tmp_path / "cams.yaml"
    save_cameras(path, [cam])
    assert load_cameras(path) == [cam]
    assert "Склад" in path.read_text(encoding="utf-8")


def test_repo_cameras_yaml_is_valid():
    from guard.pipeline.settings import CODE_ROOT

    cams = load_cameras(CODE_ROOT / "configs" / "cameras.yaml")
    assert 2 <= len(cams) <= 4
    assert any(r.type == "zone_intrusion" for c in cams for r in c.rules)


# ---------- сквозной S1 без моделей ----------


def test_worker_end_to_end_single_s1_event(tmp_path):
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

    def script(i, frame):  # идёт слева направо: опора входит в зону на ~10-м кадре, потом уходит
        x = 20 + i * 12 if i < 30 else 600
        return [person((x, 60, x + 40, 200))]

    worker.detectors = {"det": ScriptedDetector(script)}
    frame = np.zeros((240, 320, 3), np.uint8)
    t0 = time.time() - 20
    published = []
    engine.bus.publish = published.append  # перехват WS
    for i in range(50):
        ts = t0 + i * 0.2
        engine.media.push_frame("cam1", frame, ts)
        worker.process(FramePacket(frame=frame, ts=ts, seq=i + 1, decode_ms=1.0))
    events, total = engine.journal.query(camera_id="cam1")
    assert total == 1
    ev = events[0]
    assert ev.type == "zone_intrusion" and ev.zone_id == "z1" and ev.severity == Severity.high
    assert ev.confirmed_at - ev.started_at == pytest.approx(1.0, abs=0.25)
    assert set(TIMING_KEYS) <= set(ev.latency_ms)
    assert engine.media.resolve(ev.snapshot_path) is not None
    kinds = [m["kind"] for m in published]
    assert kinds.count("event") == 1 and "event_update" in kinds  # закрытие
    snap = engine.metrics.snapshot()["cam1"]
    assert snap["frames_processed"] == 50
    jpeg = worker.overlay_jpeg()
    assert jpeg is not None and jpeg[1][:2] == b"\xff\xd8"
    engine.stop()
