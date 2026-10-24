"""Набор для дообучения pose на лежачих (ml/datasets/falls.py): тест детекции и повтор COCO не пересекаются."""

from __future__ import annotations

import pytest

from ml.datasets import falls


def _series(boxes):
    return falls.Series(source="cauca", name="S08_FallForward", person="cauca:S08", kind="fall",
                        activity="FallForward", split="test", boxes=boxes)


def test_pick_test_frames_keeps_all_lying_and_every_second_upright():
    boxes = {
        1: [[0, 0.5, 0.5, 0.2, 0.6]],
        5: [[0, 0.5, 0.5, 0.2, 0.6]],
        9: [[0, 0.5, 0.5, 0.2, 0.6]],
        13: [[1, 0.5, 0.7, 0.6, 0.2]],
        17: [[1, 0.5, 0.7, 0.6, 0.2]],
        21: [],  # человека нет или кадр не размечен
        25: [[0, 0.5, 0.5, 0.2, 0.6]],
    }
    assert falls.pick_test_frames(_series(boxes), upright_every=2) == {1: False, 9: False, 13: True, 17: True}
    assert falls.pick_test_frames(_series(boxes), upright_every=1) == {
        1: False, 5: False, 9: False, 13: True, 17: True, 25: False,
    }


@pytest.mark.parametrize("limit, test", [(3, 2), (5, 0), (6, 4)])
def test_coco_replay_and_test_do_not_overlap(limit, test):
    ids = list(range(100, 106))
    replay, held_out = falls.coco_split_ids(ids, limit, test)
    assert replay == ids[:limit]
    assert not set(replay) & set(held_out)
    assert held_out == ids[limit : limit + test]


def test_recall_views_cover_models_and_det_pose_union():
    from guard.contracts import Detection

    def det(model, cls, box, conf):
        return Detection(model=model, cls=cls, cls_id=0, conf=conf, bbox=box)

    standing = det("det", "person", (10, 10, 30, 80), 0.3)
    car = det("det", "car", (100, 50, 160, 90), 0.9)
    lying = det("pose", "person", (40, 60, 120, 85), 0.5)
    same = det("pose", "person", (11, 10, 31, 80), 0.6)
    found = {"det": [standing, car], "pose": [lying, same]}
    low = falls.recall_views(found, [("det", "pose")], 0.25)
    assert low == {"det": [standing], "pose": [lying, same], "det+pose": [standing, lying]}
    high = falls.recall_views(found, [("det", "pose")], 0.4)
    assert high == {"det": [], "pose": [lying, same], "det+pose": [lying, same]}


def test_coco_label_lines_skip_tiny_boxes_and_hide_unlabeled_points():
    points = [0.0] * 51
    points[0:3] = [50.0, 40.0, 2]  # нос размечен
    lines = falls.coco_label_lines(
        [{"bbox": [40, 20, 20, 60], "keypoints": points}, {"bbox": [0, 0, 3, 30], "keypoints": points}], 100, 100
    )
    assert len(lines) == 1
    values = lines[0].split()
    assert values[:5] == ["0", "0.500000", "0.500000", "0.200000", "0.600000"]
    assert values[5:8] == ["0.500000", "0.400000", "2"]
    assert values[8:11] == ["0.000000", "0.000000", "0"]
