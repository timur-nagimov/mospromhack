"""Люди для трекера из det и pose (ds-falls): лежачего YOLOv8n теряет, pose-модель находит.

Проверяем: без флага всё как раньше; с флагом добавляются только люди pose, которых нет у det;
один человек не даёт двух треков; флаг одинаково работает в CameraWorker и в run_offline.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, RuleConfig, SourceConfig
from guard.metrics import MetricsRegistry
from guard.pipeline import compat
from guard.pipeline import settings as pipeline_settings
from guard.pipeline.settings import Settings
from guard.pipeline.testing import make_settings
from guard.pipeline.worker import CameraWorker, person_detections
from guard.rules.base import build_rules, load_plugins
from guard.sources import FramePacket
from ml.eval import run_offline as offline


@pytest.fixture(autouse=True)
def isolate_guard_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("GUARD_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pipeline_settings, "load_dotenv_guard", lambda *args, **kwargs: None)


def _det(model: str, bbox, cls: str = "person", conf: float = 0.9) -> Detection:
    keypoints = np.zeros((17, 3), np.float32) if model == "pose" else None
    return Detection(model=model, cls=cls, cls_id=0, conf=conf, bbox=tuple(float(v) for v in bbox),
                     keypoints=keypoints)


STANDING = (100, 20, 140, 200)
LYING = (200, 150, 380, 200)


# ---------- выбор людей ----------


def test_without_fill_people_come_from_det_only():
    dets = {"det": [_det("det", STANDING), _det("det", (0, 0, 30, 20), "car")], "pose": [_det("pose", LYING)]}
    assert person_detections(dets) == [dets["det"][0]]


def test_without_det_people_come_from_pose():
    dets = {"pose": [_det("pose", LYING)]}
    assert person_detections(dets) == dets["pose"]
    assert person_detections(dets, pose_fill=True) == dets["pose"]


def test_fill_adds_pose_person_that_det_missed():
    standing, lying = _det("det", STANDING), _det("pose", LYING)
    people = person_detections({"det": [standing], "pose": [_det("pose", STANDING), lying]}, pose_fill=True)
    assert people == [standing, lying]
    assert people[1].keypoints is not None  # точки pose остаются у добавленного человека


def test_fill_skips_pose_person_overlapping_det_person():
    shifted = (110, 30, 150, 210)  # тот же человек, рамка pose сдвинута: IoU ≈ 0.6
    dets = {"det": [_det("det", STANDING)], "pose": [_det("pose", shifted)]}
    assert person_detections(dets, pose_fill=True) == dets["det"]


def test_fill_skips_pose_box_mostly_inside_det_box():
    inner = (105, 20, 135, 90)  # верх тела внутри рамки det: IoU 0.29, покрытие 1.0
    dets = {"det": [_det("det", STANDING)], "pose": [_det("pose", inner)]}
    assert person_detections(dets, pose_fill=True) == dets["det"]


def test_fill_skips_det_box_mostly_inside_pose_box():
    # det видит только часть лежачего, pose — всего человека: это один человек
    dets = {"det": [_det("det", (200, 150, 260, 200))], "pose": [_det("pose", LYING)]}
    assert person_detections(dets, pose_fill=True) == dets["det"]


def test_fill_keeps_pose_person_next_to_standing_one():
    # лежит у ног стоящего: рамки пересекаются, но это разные люди
    at_feet = (60, 170, 260, 215)
    dets = {"det": [_det("det", STANDING)], "pose": [_det("pose", at_feet)]}
    assert person_detections(dets, pose_fill=True) == [dets["det"][0], dets["pose"][0]]


def test_fill_ignores_non_person_boxes_of_det():
    dets = {"det": [_det("det", LYING, "car")], "pose": [_det("pose", LYING)]}
    assert person_detections(dets, pose_fill=True) == dets["pose"]


def test_fill_skips_invalid_pose_boxes():
    dets = {"det": [], "pose": [_det("pose", (10, 10, 5, 50)), _det("pose", (0, 0, float("nan"), 9))]}
    assert person_detections(dets, pose_fill=True) == []


def test_fill_without_pose_model_is_det_only():
    dets = {"det": [_det("det", STANDING)]}
    assert person_detections(dets, pose_fill=True) == dets["det"]


# ---------- настройка ----------


def test_setting_is_off_by_default_and_read_from_env(monkeypatch):
    assert Settings.from_env().person_pose_fill is False
    monkeypatch.setenv("GUARD_PERSON_POSE_FILL", "1")
    assert Settings.from_env().person_pose_fill is True


# ---------- трекер: один человек — один трек ----------


def _tracks(frames, pose_fill):
    tracker = compat.Tracker(classes={"person"}, high_thresh=0.4, low_thresh=0.1, match_iou=0.25,
                             new_track_thresh=0.45, track_buffer_s=3.0, min_hits=2)
    ids = set()
    per_frame = []
    for index, dets in enumerate(frames):
        people = tracker.update(person_detections(dets, pose_fill=pose_fill), index * 0.2)
        per_frame.append(len(people))
        ids |= {track.track_id for track in people}
    return ids, per_frame


def test_same_person_seen_by_both_models_gives_one_track():
    frames = [{"det": [_det("det", STANDING)], "pose": [_det("pose", (103 + i % 3, 22, 143, 203))]} for i in range(30)]
    ids, per_frame = _tracks(frames, pose_fill=True)
    assert len(ids) == 1 and max(per_frame) == 1


def test_pose_keeps_track_alive_when_det_loses_lying_person():
    frames = []
    for i in range(40):
        det = [_det("det", LYING, conf=0.6)] if i < 10 or i % 4 == 0 else []  # после падения det видит урывками
        frames.append({"det": det, "pose": [_det("pose", (198, 149, 381, 201), conf=0.7)]})
    ids_fill, per_frame_fill = _tracks(frames, pose_fill=True)
    assert len(ids_fill) == 1 and min(per_frame_fill[2:]) == 1
    ids_det, per_frame_det = _tracks(frames, pose_fill=False)
    assert min(per_frame_det[2:]) == 0  # без pose трек пропадает


# ---------- CameraWorker и run_offline берут людей одинаково ----------


class _Scripted:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def detect(self, frame, timings=None, **kwargs):
        out = self.fn(self.calls)
        self.calls += 1
        return out


def _fall_camera(uri: str) -> CameraConfig:
    return CameraConfig(
        camera_id="cam-fall", name="Цех", site="Тест",
        source=SourceConfig(type="file", uri=uri, loop=False, realtime=False),
        rules=[RuleConfig(rule_id="fall", type="fall_immobile", min_duration_s=1.0, min_ratio=0.5,
                          params={"min_lying_s": 1.0, "immobile_window_s": 1.5})],
    )


def _only_pose_sees_lying():
    # det не видит лежачего вовсе, pose видит его уверенно и неподвижным
    return {"det": _Scripted(lambda i: []), "pose": _Scripted(lambda i: [_det("pose", LYING, conf=0.8)])}


def test_worker_takes_people_from_pose_when_flag_is_on(tmp_path):
    load_plugins()
    cfg = _fall_camera("never-opened")
    results = {}
    for flag in (False, True):
        detectors = _only_pose_sees_lying()
        engine = SimpleNamespace(
            settings=make_settings(tmp_path, tracker_min_hits=1, person_pose_fill=flag),
            metrics=MetricsRegistry(), get_detector=detectors.__getitem__, events=compat.EventManager(),
            run_id="unit", on_analysis=lambda *args: None, handle_events=lambda *args, **kwargs: None,
            close_capture=lambda *args: None)
        worker = CameraWorker(cfg, engine)
        ctx = None
        for index in range(3):
            ts = index * 0.2
            ctx = worker.process(FramePacket(np.zeros((240, 400, 3), np.uint8), ts, index + 1, 0, capture_id="one"))
        results[flag] = len(ctx.people)
    assert results == {False: 0, True: 1}


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "lying.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (400, 240))
    try:
        assert writer.isOpened()
        for index in range(60):
            writer.write(np.full((240, 400, 3), 40 + index, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_offline_fall_from_pose_person_only_with_flag(video, tmp_path):
    camera = _fall_camera(str(video))
    scenario = offline.Scenario(id="s", video=video.name, camera=camera)
    off = offline.run_video(scenario, _only_pose_sees_lying(), rules=build_rules(camera),
                            settings=make_settings(tmp_path))
    on = offline.run_video(scenario, _only_pose_sees_lying(), rules=build_rules(camera),
                           settings=make_settings(tmp_path, person_pose_fill=True))
    assert off.events == []
    assert [event.type for event in on.events] == ["fall_immobile"]
