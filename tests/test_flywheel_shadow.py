"""Маховик: сверка TPU с CPU (disagreement) — только на linq-tpu, в своём потоке, по целевым классам правил."""

from types import SimpleNamespace

import numpy as np

from guard.contracts import CameraConfig, Detection, FrameContext, RuleConfig, SourceConfig
from guard.flywheel.collector import FlywheelCollector
from guard.flywheel.shadow import ShadowChecker, make_shadow
from guard.rules.base import get_rule


def det(cls, conf, bbox):
    return Detection(model="det", cls=cls, cls_id=0, conf=conf, bbox=bbox)


class Out:
    def __init__(self):
        self.jobs = []

    def put(self, job):
        self.jobs.append(job)
        return True


class CpuDetector:
    def __init__(self, dets, threshold=0.25, output_format="yolov8"):
        self.dets = dets
        self.spec = SimpleNamespace(conf_threshold=threshold, output_format=output_format)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.dets)


def _setup(cpu_dets):
    cam = CameraConfig(camera_id="cam1", name="Цех", site="Цех 1", source=SourceConfig(type="file", uri="x.mp4"),
                       rules=[RuleConfig(rule_id="z", type="zone_intrusion")])
    worker = SimpleNamespace(cfg=cam, rules=[get_rule("zone_intrusion")(cam.rules[0], cam)], detectors={})
    cpu = CpuDetector(cpu_dets)
    shadow = ShadowChecker(lambda model_id: cpu if model_id == "det" else None)
    out = Out()
    col = FlywheelCollector(out, holdout=frozenset(), shadow=shadow, shadow_period_s=30.0)
    return cam, worker, cpu, shadow, out, col


def _ctx(cam, ts, dets):
    return FrameContext(camera=cam, ts=ts, frame_idx=0, frame=np.zeros((36, 64, 3), np.uint8), detections={"det": dets})


def test_mismatch_gives_disagreement_sample_with_both_sides():
    tpu = [det("person", 0.80, (10, 10, 60, 160)), det("chair", 0.9, (200, 10, 260, 90))]
    cpu = [det("person", 0.82, (12, 10, 62, 160)), det("person", 0.71, (300, 10, 350, 160))]
    cam, worker, _, shadow, out, col = _setup(cpu)
    ctx = _ctx(cam, 100.0, tpu)
    assert shadow.check(col, worker, ctx, {"det": tpu}) is True
    assert len(out.jobs) == 1
    sample = out.jobs[0].sample
    assert sample["source"] == "disagreement" and "только на CPU" in sample["reason"]
    assert sample["meta"]["mismatches"] == [
        {"model_id": "det", "cls": "person", "conf": 0.71, "bbox": [300, 10, 350, 160], "only_on": "cpu"}
    ]  # chair — не целевой класс правила зоны, не сравнивается
    assert set(out.jobs[0].detections) == {"det", "det_cpu"}
    assert shadow.stats()["mismatches"] == 1


def test_agreement_gives_nothing_and_low_conf_is_ignored():
    tpu = [det("person", 0.40, (10, 10, 60, 160))]
    cpu = [det("person", 0.46, (11, 10, 61, 160)), det("person", 0.30, (300, 10, 350, 160))]  # 0.30 < 0.25 + 0.10
    cam, worker, _, shadow, out, col = _setup(cpu)
    assert shadow.check(col, worker, _ctx(cam, 1.0, tpu), {"det": tpu}) is False
    assert out.jobs == [] and shadow.stats()["checks"] == 1


def test_collector_submits_one_check_per_period_and_runs_in_thread():
    tpu = [det("person", 0.8, (10, 10, 60, 160))]
    cam, worker, cpu, shadow, out, col = _setup([])  # CPU не видит человека → расхождение
    col.start()
    try:
        for i in range(100):  # 20 с при 5 к/с — одна сверка
            col.observe(worker, _ctx(cam, 1000.0 + i * 0.2, tpu))
        import time

        deadline = time.time() + 3
        while time.time() < deadline and not out.jobs:
            time.sleep(0.02)
        assert cpu.calls == 1 and len([j for j in out.jobs if j.sample["source"] == "disagreement"]) == 1
        assert "только на TPU" in out.jobs[0].sample["reason"]
    finally:
        col.stop()


def test_full_queue_skips_without_waiting():
    cam, worker, _, shadow, _, col = _setup([])
    ctx = _ctx(cam, 1.0, [])
    assert shadow.submit(col, worker, ctx) and shadow.submit(col, worker, ctx)
    assert not shadow.submit(col, worker, ctx) and shadow.stats()["skipped"] == 1


def test_shadow_only_on_tpu_backend():
    engine = SimpleNamespace(backend=SimpleNamespace(name="onnx-cpu"))
    assert make_shadow(engine) is None
