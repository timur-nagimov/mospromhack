"""Гейт маховика: правило решения, сводка событий, mAP50 на синтетике, CLI без моделей."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.flywheel import gate


def _events(tp=2, fp=0, fn=2, dup=0, fa=0.0, by_type=None):
    total = tp + fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "duplicates": dup,
        "precision": tp / total if total else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_alarms_per_hour": fa,
        "by_type": by_type or {},
    }


def test_better_recall_same_fp_promotes():
    ok, reasons = gate.decide({"events": _events(tp=3, fn=1), "map50": 0.8}, {"events": _events(), "map50": None})
    assert ok
    assert any("mAP не применим" in r for r in reasons)
    assert not any(r.startswith("НЕ ПРОЙДЕНО") for r in reasons)


def test_equal_metrics_promote():
    ok, _ = gate.decide({"events": _events(), "map50": 0.5}, {"events": _events(), "map50": 0.5})
    assert ok


@pytest.mark.parametrize(
    "cand, why",
    [
        (_events(tp=1, fn=3), "recall"),
        (_events(tp=3, fp=1, fn=1, fa=6.7), "precision"),
        (_events(tp=2, fp=0, fn=2, fa=1.0), "ложных тревог"),
        (_events(dup=1), "дублей"),
    ],
)
def test_any_worse_event_metric_rejects(cand, why):
    ok, reasons = gate.decide({"events": cand, "map50": 0.9}, {"events": _events(), "map50": 0.9})
    assert not ok
    assert any(r.startswith("НЕ ПРОЙДЕНО") and why in r for r in reasons)


def test_map_drop_within_tolerance_passes_and_beyond_fails():
    base = {"events": _events(), "map50": 0.80}
    assert gate.decide({"events": _events(), "map50": 0.791}, base)[0]
    ok, reasons = gate.decide({"events": _events(), "map50": 0.785}, base)
    assert not ok
    assert any("mAP50" in r and r.startswith("НЕ ПРОЙДЕНО") for r in reasons)


def test_missing_candidate_map_rejects():
    ok, _ = gate.decide({"events": _events(), "map50": None}, {"events": _events(), "map50": None})
    assert not ok


def test_precision_without_predictions_compares_fp():
    empty = _events(tp=0, fp=0, fn=4)
    ok, _ = gate.decide({"events": empty, "map50": 0.5}, {"events": empty, "map50": None})
    assert ok


def test_by_type_changes_are_explained():
    cand = _events(tp=3, fn=1, by_type={"vehicle_proximity": {"tp": 3, "fp": 0, "fn": 1, "duplicates": 0}})
    base = _events(by_type={"vehicle_proximity": {"tp": 2, "fp": 0, "fn": 2, "duplicates": 0}})
    _, reasons = gate.decide({"events": cand, "map50": 0.7}, {"events": base, "map50": None})
    assert "vehicle_proximity: 3/4 против 2/4, FP 0 против 0" in reasons


def _report(tp, fp, fn, videos=("s4_01.mp4", "s4_02.mp4")):
    metrics = {"tp": tp, "fp": fp, "fn": fn, "duplicates": 0, "ignored_uncertain": 0,
               "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn),
               "latency_median_s": 1.234, "latency_p95_s": 2.0, "false_alarms_per_hour": 0.0}
    return {"overall": metrics, "by_type": {"vehicle_proximity": metrics}, "videos": list(videos)}


def test_event_summary_is_compact():
    s = gate.event_summary(_report(3, 0, 1))
    assert s["videos"] == ["s4_01", "s4_02"]
    assert s["latency_p50_s"] == 1.23
    assert s["by_type"]["vehicle_proximity"] == {"tp": 3, "fp": 0, "fn": 1, "duplicates": 0}


def test_parse_model_and_subset_args():
    assert gate.parse_model_arg("ker=models/f.onnx:forklift,person") == (
        "ker", Path("models/f.onnx"), ["forklift", "person"])
    assert gate.parse_subset("forklift_rf=a/b.txt") == ("forklift_rf", Path("a/b.txt"))


def test_label_path_follows_ultralytics_layout():
    assert gate.label_path(Path("/d/images/test/x.jpg")) == Path("/d/labels/test/x.txt")


def _tiny_dataset(tmp_path: Path) -> Path:
    import cv2

    for split in ("test",):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    img = np.zeros((100, 200, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "images/test/a.jpg"), img)
    cv2.imwrite(str(tmp_path / "images/test/b.jpg"), img)
    # a: погрузчик 20..60 × 10..50 и человек; b: пусто (негатив)
    (tmp_path / "labels/test/a.txt").write_text("1 0.2 0.3 0.2 0.4\n0 0.8 0.5 0.1 0.6\n")
    (tmp_path / "labels/test/b.txt").write_text("")
    (tmp_path / "data.yaml").write_text(f"path: {tmp_path}\ntest: images/test\nnames:\n  0: person\n  1: forklift\n")
    return tmp_path / "data.yaml"


def test_dataset_map_with_fake_predictions(tmp_path, monkeypatch):
    data = _tiny_dataset(tmp_path)
    images, names = gate.dataset_images(data, "test")
    assert names == ["person", "forklift"] and len(images) == 2
    a, b = (str(p) for p in images)

    def fake_predict(onnx, classes, imgs, conf=0.001, iou=0.6):
        return {
            "forklift": [(a, 0.9, np.array([20.0, 10.0, 60.0, 50.0])),   # TP
                         (b, 0.3, np.array([0.0, 0.0, 50.0, 50.0]))],     # FP на негативе, ниже
            "person": [(a, 0.8, np.array([150.0, 20.0, 170.0, 80.0]))],
        }

    monkeypatch.setattr(gate, "predict_onnx", fake_predict)
    res = gate.dataset_map(Path("x.onnx"), ["person", "forklift"], images, names)
    assert res["per_class"]["forklift"]["ap50"] == pytest.approx(1.0, abs=0.01)
    assert res["per_class"]["forklift"]["n_gt"] == 1
    assert res["per_class"]["person"]["ap50"] == pytest.approx(1.0, abs=0.01)
    assert res["map50"] == pytest.approx(1.0, abs=0.01)
    # модель без класса набора: класс не входит в mAP
    res2 = gate.dataset_map(Path("x.onnx"), ["forklift"], images, names)
    assert set(res2["per_class"]) == {"forklift"}


def test_cli_writes_gate_to_card(tmp_path, monkeypatch):
    card = {"model_id": "forklift", "version": "forklift-vt", "parent": "yolov8n-coco",
            "created_at": "2026-10-24T15:00:00+0300", "task": "detect", "classes": ["person", "forklift"],
            "onnx": "forklift_vt-480.onnx", "status": "candidate"}
    card_file = tmp_path / "forklift-vt.json"
    card_file.write_text(json.dumps(card))
    (tmp_path / "cand.json").write_text(json.dumps(_report(3, 0, 1)))
    (tmp_path / "base.json").write_text(json.dumps(_report(2, 0, 2)))
    data = _tiny_dataset(tmp_path / "ds")
    monkeypatch.setattr(gate, "dataset_map", lambda *a, **k: {"images": 2, "map50": 0.8, "per_class": {}})
    saved = {}
    monkeypatch.setattr(gate, "save_card", lambda c: saved.update(c) or tmp_path / "out.json")
    code = gate.main(["--candidate", str(card_file), "--onnx", str(tmp_path / "x.onnx"), "--data", str(data),
                      "--events-candidate", str(tmp_path / "cand.json"),
                      "--events-baseline", str(tmp_path / "base.json"),
                      "--baseline-name", "det (yolov8n COCO)", "--write"])
    assert code == 0
    assert saved["gate"]["promote"] is True
    assert saved["status"] == "candidate"
    assert saved["holdout"]["events"]["tp"] == 3
    assert saved["baseline"]["version"] == "det (yolov8n COCO)"
    assert saved["baseline"]["holdout"]["events"]["tp"] == 2


def test_cli_rejects_and_marks_card(tmp_path, monkeypatch):
    card = {"model_id": "forklift", "version": "forklift-vt", "parent": "p", "created_at": "x",
            "task": "detect", "classes": ["person", "forklift"], "status": "candidate"}
    card_file = tmp_path / "forklift-vt.json"
    card_file.write_text(json.dumps(card))
    (tmp_path / "cand.json").write_text(json.dumps(_report(2, 1, 2)))
    (tmp_path / "base.json").write_text(json.dumps(_report(2, 0, 2)))
    data = _tiny_dataset(tmp_path / "ds")
    monkeypatch.setattr(gate, "dataset_map", lambda *a, **k: {"images": 2, "map50": 0.8, "per_class": {}})
    saved = {}
    monkeypatch.setattr(gate, "save_card", lambda c: saved.update(c) or tmp_path / "out.json")
    code = gate.main(["--candidate", str(card_file), "--onnx", str(tmp_path / "x.onnx"), "--data", str(data),
                      "--events-candidate", str(tmp_path / "cand.json"),
                      "--events-baseline", str(tmp_path / "base.json"), "--write"])
    assert code == 1
    assert saved["status"] == "rejected"
    assert saved["gate"]["promote"] is False


def test_cli_refuses_events_on_different_videos(tmp_path, monkeypatch):
    card = {"model_id": "m", "version": "v", "parent": "p", "created_at": "x", "task": "detect",
            "classes": ["person"], "status": "candidate"}
    card_file = tmp_path / "v.json"
    card_file.write_text(json.dumps(card))
    (tmp_path / "cand.json").write_text(json.dumps(_report(2, 0, 2, videos=("s4_01.mp4",))))
    (tmp_path / "base.json").write_text(json.dumps(_report(2, 0, 2)))
    code = gate.main(["--candidate", str(card_file), "--onnx", "x.onnx",
                      "--events-candidate", str(tmp_path / "cand.json"),
                      "--events-baseline", str(tmp_path / "base.json")])
    assert code == 2


def test_parser_requires_candidate():
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args([])
