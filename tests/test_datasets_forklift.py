"""Набор погрузчика: конвертация разметки, разбиение, дедуп и сверка с роликами оценки. Без сети."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from ml.datasets import forklift as ds


def _img(path, seed, size=(64, 48)):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    return path


def test_yolo_box_is_clipped_and_normalized():
    cx, cy, w, h = ds._yolo_box(-10, 10, 60, 40, 100, 50)
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.6, 0.5, 0.8))


def test_hash_split_is_deterministic_and_roughly_80_10_10():
    splits = [ds._hash_split(f"coco:{i}") for i in range(2000)]
    assert splits == [ds._hash_split(f"coco:{i}") for i in range(2000)]
    share = {s: splits.count(s) / len(splits) for s in ds.SPLITS}
    assert 0.75 < share["train"] < 0.85 and 0.06 < share["test"] < 0.14


def test_read_coco_json_keeps_person_and_forklift_drops_vehicles(tmp_path):
    _img(tmp_path / "a.jpg", 1, (100, 50))
    (tmp_path / "_annotations.coco.json").write_text(json.dumps({
        "categories": [{"id": 0, "name": "forklift"}, {"id": 1, "name": "person"}, {"id": 2, "name": "truck"}],
        "images": [{"id": 7, "file_name": "a.jpg", "width": 100, "height": 50},
                   {"id": 8, "file_name": "missing.jpg", "width": 100, "height": 50}],
        "annotations": [{"image_id": 7, "category_id": 0, "bbox": [10, 5, 20, 10]},
                        {"image_id": 7, "category_id": 1, "bbox": [50, 0, 10, 40]},
                        {"image_id": 7, "category_id": 2, "bbox": [0, 0, 5, 5]}],
    }))
    (s,) = ds.read_coco_json("forklift_rf", tmp_path, "test")
    assert s.split == "test"
    assert [b[0] for b in s.boxes] == [1, 0]  # forklift → 1, person → 0
    assert s.boxes[0][1:] == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert s.dropped == {"truck": 1}


def test_read_yolo_dir_maps_names(tmp_path):
    _img(tmp_path / "train" / "images" / "x.jpg", 2)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "x.txt").write_text("4 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.1 0.1\n17 0.3 0.3 0.1 0.1\n")
    names = ["barcode", "car", "box", "fire", "forklift", "c", "g", "h", "l", "lp", "person", "q", "r", "v", "s",
             "t", "tl", "truck", "van", "pallet"]
    (s,) = ds.read_yolo_dir("logistics_rf", tmp_path, {"train": "train"}, names,
                            {"forklift": "forklift", "person": "person"})
    assert [b[0] for b in s.boxes] == [1, 0]
    assert s.dropped == {"truck": 1}


def test_dedup_prefers_test_and_drops_holdout_matches(tmp_path):
    a = _img(tmp_path / "a.jpg", 10)
    a_copy = tmp_path / "a_copy.jpg"
    a_copy.write_bytes(a.read_bytes())
    b = _img(tmp_path / "b.jpg", 11)
    c = _img(tmp_path / "c.jpg", 12)
    samples = [
        ds.Sample("forklift_rf", "train", "g1", a_copy),
        ds.Sample("forklift_rf", "test", "g2", a),
        ds.Sample("coco_vehicles", "train", "g3", b),
        ds.Sample("coco_vehicles", "val", "g4", c),
    ]
    holdout = np.array([ds.dhash(c)], dtype=np.uint64)  # c совпадает с кадром ролика оценки
    kept, stats = ds.dedup(samples, holdout)
    assert {(s.image.name, s.split) for s in kept} == {("a.jpg", "test"), ("b.jpg", "train")}
    assert stats["dropped_duplicates"] == {"forklift_rf/train": 1}
    assert stats["dropped_holdout_match"] == {"coco_vehicles": 1}


def test_cap_groups_limits_frames_per_video(tmp_path):
    samples = [ds.Sample("excavators_rf100", "train", "excavators_rf100:v1", tmp_path / f"{i}.jpg")
               for i in range(10)]
    samples.append(ds.Sample("coco_vehicles", "train", "coco:1", tmp_path / "c.jpg"))
    out = ds.cap_groups(samples, 3, seed=1)
    assert sum(s.source == "excavators_rf100" for s in out) == 3
    assert sum(s.source == "coco_vehicles" for s in out) == 1


def test_imagenet_frames_need_teacher_boxes_and_stay_in_train(tmp_path):
    raw = tmp_path / "raw"
    for sid in ("forklift_rf", "construction_rf"):
        for sub in ("train", "valid", "test"):
            (raw / sid / sub).mkdir(parents=True)
            (raw / sid / sub / "_annotations.coco.json").write_text(
                json.dumps({"categories": [], "images": [], "annotations": []}))
    root = raw / "logistics_rf" / "Logistics.v2i.yolov8"
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [forklift, person]\n")
    with_box = _img(raw / "imagenet_forklift" / "images" / "train_00001.jpg", 3)
    _img(raw / "imagenet_forklift" / "images" / "train_00002.jpg", 4)
    teacher_people = {str(with_box): [[0.5, 0.5, 0.1, 0.3]]}
    teacher_forklift = {str(with_box): [[0.3, 0.6, 0.4, 0.5]]}
    samples = ds.load_sources(raw, teacher_people, teacher_forklift)
    assert [(s.source, s.split, s.image.name) for s in samples] == [("imagenet_forklift", "train", "train_00001.jpg")]
    assert sorted(b[0] for b in samples[0].boxes) == [0, 1]
    assert samples[0].teacher == 1


def test_teacher_person_does_not_duplicate_ground_truth():
    gt = (0.5, 0.5, 0.2, 0.4)
    assert ds._iou_cxcywh(gt, gt) == pytest.approx(1.0)
    assert ds._iou_cxcywh(gt, (0.9, 0.9, 0.1, 0.1)) == 0.0


def test_forklift_caption_pattern():
    assert ds.FORKLIFT_WORDS.search("A man drives a fork lift in a warehouse")
    assert ds.FORKLIFT_WORDS.search("forklift carrying boxes")
    assert ds.FORKLIFT_WORDS.search("Fork-lift near a truck")
    assert not ds.FORKLIFT_WORDS.search("a fork on a table next to a truck")
