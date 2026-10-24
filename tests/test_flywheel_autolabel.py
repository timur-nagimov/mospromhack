"""Авторазметка учителем без тяжёлых моделей: отбор сэмплов, маппинг классов, слияние, статус auto, пропуск учителя."""

import cv2
import numpy as np
import pytest

from guard.flywheel.store import FlywheelStore
from ml.flywheel import autolabel as al


@pytest.fixture()
def store(tmp_path):
    s = FlywheelStore(tmp_path / "fw")
    for sid in ("n1", "n2", "acc", "op", "held"):
        s.write_frame(f"frames/cam1/{sid}.jpg", cv2.imencode(".jpg", np.full((120, 160, 3), 80, np.uint8))[1].tobytes())
        s.add({"id": sid, "camera_id": "cam1", "source": "event", "width": 160, "height": 120,
               "created_at": {"n1": 5, "n2": 4, "acc": 3, "op": 2, "held": 1}[sid], "holdout": 1 if sid == "held" else 0})
    s.update("acc", status="accepted")
    s.update("op", labels=[{"cls": "person", "bbox": [0, 0, 10, 10], "origin": "operator"}])
    yield s
    s.close()


def test_class_maps_and_merge():
    assert al.map_class("coco", "truck") == "truck" and al.map_class("coco", "bowl") is None
    assert al.map_class("world", "hard hat") == "helmet" and al.map_class("world", "forklift") == "forklift"
    assert al.map_class("hardhat", "NO-Hardhat") == "no_helmet" and al.map_class("hardhat", "Hardhat") == "helmet"
    merged = al.merge_labels([
        {"cls": "person", "bbox": [0, 0, 100, 200], "conf": 0.6, "teacher": "coco"},
        {"cls": "person", "bbox": [2, 0, 102, 200], "conf": 0.9, "teacher": "world"},
        {"cls": "helmet", "bbox": [2, 0, 102, 200], "conf": 0.5, "teacher": "world"},
    ])
    assert [(m["cls"], m["conf"]) for m in merged] == [("person", 0.9), ("helmet", 0.5)]


def test_selection_skips_reviewed_and_operator_labels(store):
    args = al.parse_args(["--limit", "10"])
    assert [s["id"] for s in al.select_samples(store, args)] == ["n1", "n2", "held"]
    skip = al.parse_args(["--limit", "10", "--skip-holdout"])
    assert [s["id"] for s in al.select_samples(store, skip)] == ["n1", "n2"]
    over = al.parse_args(["--ids", "acc,op", "--overwrite"])
    assert {s["id"] for s in al.select_samples(store, over)} == {"acc", "op"}
    assert [s["id"] for s in al.select_samples(store, al.parse_args(["--limit", "1"]))] == ["n1"]


def test_autolabel_writes_teacher_labels_and_skips_missing_teacher(store, tmp_path, monkeypatch):
    monkeypatch.setattr(al, "teacher_available", lambda t: (t != "world", "нет пакета clip" if t == "world" else ""))
    monkeypatch.setattr(al, "load_teacher", lambda t: object())

    def fake_run(teacher, model, image, *, conf, imgsz, device):
        if teacher == "coco":
            return [{"cls": "person", "bbox": [10.0, 10.0, 60.0, 110.0], "origin": "teacher", "conf": 0.88, "teacher": "coco"}]
        return [{"cls": "no_helmet", "bbox": [20.0, 10.0, 40.0, 30.0], "origin": "teacher", "conf": 0.61, "teacher": "hardhat"}]

    monkeypatch.setattr(al, "run_teacher", fake_run)
    args = al.parse_args(["--limit", "2", "--device", "cpu", "--viz", str(tmp_path / "viz")])
    report = al.autolabel(store, args)
    assert report["labeled"] == 2 and report["labels_by_class"] == {"person": 2, "no_helmet": 2}
    assert report["teachers"]["world"]["skipped"] == "нет пакета clip" and "boxes" in report["teachers"]["coco"]
    s = store.get("n1")
    assert s["status"] == "auto" and {lb["origin"] for lb in s["labels"]} == {"teacher"}
    assert {lb["cls"] for lb in s["labels"]} == {"person", "no_helmet"}
    assert s["meta"]["autolabel"]["teachers"] == ["coco", "hardhat"]
    assert store.get("acc")["status"] == "accepted" and store.get("op")["status"] == "new"  # не тронуты
    assert len(list((tmp_path / "viz").glob("*.jpg"))) == 2


def test_no_teachers_available(store, monkeypatch):
    monkeypatch.setattr(al, "teacher_available", lambda t: (False, "нет весов"))
    report = al.autolabel(store, al.parse_args(["--limit", "1"]))
    assert report["labeled"] == 0 and report["error"] == "ни один учитель недоступен"
    assert store.get("n1")["status"] == "new"


def test_world_needs_clip(monkeypatch, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(al, "weights_path", lambda t: weights)
    monkeypatch.setattr(al.importlib.util, "find_spec", lambda name: None)
    ok, why = al.teacher_available("world")
    assert not ok and "clip" in why
    assert al.teacher_available("coco") == (True, "")
