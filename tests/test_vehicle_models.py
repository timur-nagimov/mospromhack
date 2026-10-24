"""Погрузчик из отдельной модели: параметр models правила vehicle_proximity и техника из всех моделей кадра."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, RuleConfig, SourceConfig
from guard.pipeline import settings as pipeline_settings
from guard.pipeline.settings import Settings
from guard.pipeline.worker import OBJECT_CLASSES, object_detections
from guard.rules.base import available_rules, build_rules, load_plugins
from guard.rules.plugins.vehicle_proximity import VehicleProximityRule
from ml.eval import run_offline as offline


@pytest.fixture(autouse=True)
def isolate_guard_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("GUARD_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pipeline_settings, "load_dotenv_guard", lambda *args, **kwargs: None)


def _camera(params: dict, video: str = "synthetic.avi") -> CameraConfig:
    return CameraConfig(
        camera_id="cam-forklift",
        name="Склад",
        site="Тест",
        source=SourceConfig(type="file", uri=video, loop=False, realtime=False),
        rules=[RuleConfig(rule_id="vehicle", type="vehicle_proximity", params=params,
                          min_duration_s=1.0, min_ratio=0.7)],
    )


def _det(model: str, cls: str, bbox, conf: float = 0.9) -> Detection:
    return Detection(model=model, cls=cls, cls_id=0, conf=conf, bbox=tuple(float(v) for v in bbox))


# ---------- параметр models ----------


def test_default_rule_needs_only_det():
    rule = VehicleProximityRule(RuleConfig(rule_id="v", type="vehicle_proximity"), _camera({}))
    assert rule.required_models == frozenset({"det"})


def test_models_param_adds_forklift_model_on_instance():
    rule = VehicleProximityRule(
        RuleConfig(rule_id="v", type="vehicle_proximity", params={"models": ["det", "forklift"]}), _camera({}))
    assert rule.required_models == frozenset({"det", "forklift"})
    # класс не меняется: другие камеры и API типов правил видят прежнее
    assert VehicleProximityRule.required_models == frozenset({"det"})
    load_plugins()
    rules = {r["type"]: r for r in available_rules()}
    assert rules["vehicle_proximity"]["required_models"] == ["det"]


def test_det_is_always_required_for_people():
    rule = VehicleProximityRule(
        RuleConfig(rule_id="v", type="vehicle_proximity", params={"models": ["forklift"]}), _camera({}))
    assert rule.required_models == frozenset({"det", "forklift"})


def test_models_param_is_in_schema():
    schema = VehicleProximityRule.params_schema["properties"]["models"]
    assert schema["default"] == ["det"]
    assert schema["items"]["type"] == "string"


# ---------- техника из всех моделей ----------


def test_object_detections_merges_models_in_model_id_order():
    dets = {
        "forklift": [_det("forklift", "forklift", (0, 0, 10, 10)), _det("forklift", "person", (0, 0, 5, 9))],
        "det": [_det("det", "truck", (20, 0, 40, 10)), _det("det", "person", (50, 0, 60, 30)),
                _det("det", "cup", (1, 1, 2, 2))],
    }
    out = object_detections(dets)
    assert [(d.model, d.cls) for d in out] == [("det", "truck"), ("forklift", "forklift")]


def test_object_detections_det_only_is_unchanged():
    det = [_det("det", c, (0, 0, 10, 10)) for c in ("person", "truck", "bus", "car", "cup")]
    assert object_detections({"det": det}) == [d for d in det if d.cls in OBJECT_CLASSES]


def test_forklift_class_is_an_object_class():
    assert "forklift" in OBJECT_CLASSES


# ---------- офлайн-конвейер: погрузчик из своей модели даёт событие ----------


class _Seq:
    """Фейковый детектор: i-й вызов — i-й анализируемый кадр."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls = 0

    def detect(self, frame, timings):
        out = self.fn(self.calls)
        self.calls += 1
        return out


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (128, 96))
    try:
        assert writer.isOpened()
        for index in range(40):
            writer.write(np.full((96, 128, 3), 30 + index, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _detectors():
    person = _Seq(lambda i: [_det("det", "person", (60, 20, 80, 90))])
    # погрузчик едет к человеку: 2 пикс. за кадр анализа (5 к/с) — 0.2 высоты рамки в секунду
    forklift = _Seq(lambda i: [_det("forklift", "forklift", (0 + 2 * i, 40, 40 + 2 * i, 90))])
    return {"det": person, "forklift": forklift}


def test_forklift_model_gives_vehicle_event(video):
    camera = _camera({"min_vehicle_speed_bh": 0.05, "models": ["det", "forklift"]}, str(video))
    rules = build_rules(camera)
    scenario = offline.Scenario(id="s", video=video.name, camera=camera)
    result = offline.run_video(scenario, _detectors(), rules=rules, settings=Settings.from_env())
    assert [e.type for e in result.events] == ["vehicle_proximity"]
    assert result.events[0].details["vehicle_track_id"] is not None


def test_without_forklift_model_there_is_no_vehicle(video):
    camera = _camera({"min_vehicle_speed_bh": 0.05}, str(video))
    rules = build_rules(camera)
    scenario = offline.Scenario(id="s", video=video.name, camera=camera)
    detectors = _detectors()
    result = offline.run_video(scenario, detectors, rules=rules, settings=Settings.from_env())
    assert result.events == []
    assert detectors["forklift"].calls == 0  # модель не нужна правилу — не вызывается


def test_missing_forklift_detector_is_an_error(video):
    camera = _camera({"models": ["det", "forklift"]}, str(video))
    scenario = offline.Scenario(id="s", video=video.name, camera=camera)
    with pytest.raises(ValueError, match="forklift"):
        offline.run_video(scenario, {"det": _detectors()["det"]}, rules=build_rules(camera), settings=Settings.from_env())
