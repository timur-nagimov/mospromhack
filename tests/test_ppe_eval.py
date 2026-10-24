"""PPE detection metrics and report contracts, without loading neural models."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml import eval_ppe


BOX = np.array([0.0, 0.0, 10.0, 10.0])
OTHER_BOX = np.array([20.0, 20.0, 30.0, 30.0])


@pytest.mark.parametrize("preds,boxes,expected_ap,expected_p,expected_r", [
    ([("val/a.jpg", 0.9, BOX)], [BOX], 1.0, 1.0, 1.0),
    # Only recall grid points 0.00 through 0.50 are reached: 51 / 101.
    ([("val/a.jpg", 0.9, BOX)], [BOX, OTHER_BOX], 51 / 101, 1.0, 0.5),
    ([], [BOX], 0.0, 0.0, 0.0),
    ([("val/a.jpg", 0.9, OTHER_BOX)], [BOX], 0.0, 0.0, 0.0),
    ([("val/a.jpg", 0.9, OTHER_BOX), ("val/a.jpg", 0.8, BOX)], [BOX], 0.5, 0.5, 1.0),
    ([("val/a.jpg", 0.9, BOX), ("val/a.jpg", 0.8, BOX)], [BOX], 1.0, 0.5, 1.0),
    (
        [("val/a.jpg", 0.9, BOX), ("val/a.jpg", 0.8, BOX), ("val/a.jpg", 0.7, OTHER_BOX)],
        [BOX, OTHER_BOX], (51 + 50 * 2 / 3) / 101, 2 / 3, 1.0,
    ),
    ([("test/a.jpg", 0.9, BOX)], [BOX], 0.0, 0.0, 0.0),
])
def test_coco_101_point_ap_and_one_to_one_matching(preds, boxes, expected_ap, expected_p, expected_r):
    result = eval_ppe.ap_for_class(preds, {"val/a.jpg": np.array(boxes)}, 0.5, 0.35)
    assert result == pytest.approx((expected_ap, expected_p, expected_r, len(boxes)))


def test_precision_envelope_uses_later_better_precision():
    # FP, TP, TP: the precision envelope is 2/3 at every recall grid point.
    preds = [("empty.jpg", 0.9, BOX), ("a.jpg", 0.8, BOX), ("a.jpg", 0.7, OTHER_BOX)]
    ap, precision, recall, _ = eval_ppe.ap_for_class(preds, {"a.jpg": np.array([BOX, OTHER_BOX])}, 0.5, 0.35)
    assert (ap, precision, recall) == pytest.approx((2 / 3, 2 / 3, 1.0))


@pytest.mark.parametrize("gts", [{}, {"a.jpg": np.empty((0, 4))}])
def test_no_ground_truth_is_unscored(gts):
    assert eval_ppe.ap_for_class([("a.jpg", 0.9, BOX)], gts, 0.5, 0.35) == (None, None, None, 0)


def test_operating_threshold_is_inclusive_and_does_not_change_ap():
    preds = [("a.jpg", 0.35, BOX), ("a.jpg", 0.8, OTHER_BOX)]
    gts = {"a.jpg": np.array([BOX])}
    at_boundary = eval_ppe.ap_for_class(preds, gts, 0.5, 0.35)
    above_boundary = eval_ppe.ap_for_class(preds, gts, 0.5, 0.36)
    no_predictions = eval_ppe.ap_for_class(preds, gts, 0.5, 0.9)
    assert at_boundary == pytest.approx((0.5, 0.5, 1.0, 1))
    assert above_boundary == pytest.approx((0.5, 0.0, 0.0, 1))
    assert no_predictions == pytest.approx((0.5, 0.0, 0.0, 1))


def test_equal_confidence_preserves_prediction_order():
    gts = {"a.jpg": np.array([BOX])}
    preds = [("a.jpg", 0.9, OTHER_BOX), ("a.jpg", 0.9, BOX)]
    assert eval_ppe.ap_for_class(preds, gts, 0.5, 0.35)[0] == pytest.approx(0.5)


def test_iou_threshold_includes_exact_boundary():
    pred = [("a.jpg", 0.9, np.array([0.0, 0.0, 20.0, 10.0]))]
    gts = {"a.jpg": np.array([BOX])}
    assert eval_ppe.ap_for_class(pred, gts, 0.5, 0.35)[0] == 1.0
    assert eval_ppe.ap_for_class(pred, gts, 0.5001, 0.35)[0] == 0.0
    assert eval_ppe.iou_matrix(np.zeros((1, 4)), np.zeros((1, 4))).item() == 0.0


def make_dataset(data: Path, *, nested: bool = False) -> None:
    for split, center in [("val", 0.2), ("test", 0.8)]:
        relative = Path(split) / ("nested/a.jpg" if nested else "a.jpg")
        image = data / "images" / relative
        label = (data / "labels" / relative).with_suffix(".txt")
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        assert cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
        label.write_text("\n".join(f"{c} {center} {center} 0.2 0.2" for c in range(3)))
    (data / "data.yaml").write_text("names: [person, Hardhat, Safety Vest]\n")


@pytest.mark.parametrize("nested", [False, True])
def test_load_gt_keeps_split_and_nested_directory_in_image_keys(tmp_path, nested):
    make_dataset(tmp_path, nested=nested)
    images, gts = eval_ppe.load_gt(tmp_path, ["val", "test"], {0: "person", 1: "Hardhat", 2: "Safety Vest"})
    relative = "nested/a.jpg" if nested else "a.jpg"
    assert len(images) == 2
    assert set(gts["person"]) == {f"val/{relative}", f"test/{relative}"}
    np.testing.assert_allclose(gts["person"][f"val/{relative}"], [[10, 10, 30, 30]])
    np.testing.assert_allclose(gts["person"][f"test/{relative}"], [[70, 70, 90, 90]])
    assert eval_ppe.image_key(tmp_path, images[0].parent / ".." / images[0].parent.name / images[0].name) == f"val/{relative}"


def core_gts():
    return {c: {"a.jpg": np.array([BOX])} for c in eval_ppe.CORE}


def test_common_core_denominator_ranks_full_ppe_above_person_only():
    gts = core_gts()
    person_only = eval_ppe.evaluate_predictions({"person": [("a.jpg", 0.9, BOX)]}, gts, ["person"], 0.35)
    full_ppe = eval_ppe.evaluate_predictions(
        {c: [("a.jpg", 0.9, BOX)] for c in ["person", "helmet"]}, gts, eval_ppe.CORE, 0.35,
    )
    assert person_only[eval_ppe.CORE_AP50] == 0.333
    assert full_ppe[eval_ppe.CORE_AP50] == 0.667
    assert person_only[eval_ppe.CORE_AP50] < full_ppe[eval_ppe.CORE_AP50]
    assert person_only["helmet"] == {"ap50": 0.0, "p": 0.0, "r": 0.0, "n_gt": 1}
    assert person_only["coverage"]["per_class"]["helmet"] == {
        "model_supported": False, "n_gt": 1, "status": "unsupported",
    }
    assert full_ppe["coverage"]["per_class"]["vest"]["status"] == "evaluated"
    assert person_only["coverage"]["core"] == {
        "classes_with_gt": list(eval_ppe.CORE), "classes_without_gt": [], "denominator": 3,
        "supported_classes": ["person"], "unsupported_classes": ["helmet", "vest"], "supported_fraction": 1 / 3,
    }


def test_missing_gt_is_distinct_from_unsupported_model_class():
    gts = {"person": {"a.jpg": np.array([BOX])}, "helmet": {"a.jpg": np.array([BOX])}}
    row = eval_ppe.evaluate_predictions({"person": [("a.jpg", 0.9, BOX)]}, gts, ["person", "vest"], 0.35)
    assert row[eval_ppe.CORE_AP50] == 0.5
    assert row["vest"] is None
    assert row["coverage"]["core"]["denominator"] == 2
    assert row["coverage"]["core"]["classes_without_gt"] == ["vest"]
    assert row["coverage"]["core"]["unsupported_classes"] == ["helmet"]
    assert row["coverage"]["per_class"]["vest"] == {"model_supported": True, "n_gt": 0, "status": "no_ground_truth"}
    assert row["coverage"]["per_class"]["no_vest"] == {"model_supported": False, "n_gt": 0, "status": "no_ground_truth"}


def test_empty_core_ground_truth_stays_unscored():
    row = eval_ppe.evaluate_predictions({}, {}, ["person"], 0.35)
    assert row[eval_ppe.CORE_AP50] is None
    assert row["coverage"]["core"]["denominator"] == 0
    assert row["coverage"]["core"]["supported_fraction"] is None
    assert row["coverage"]["classes_without_gt"] == list(eval_ppe.NORM)


def test_no_helmet_diagnostic_ap_uses_separate_iou_threshold():
    gts = {"no_helmet": {"a.jpg": np.array([BOX])}}
    preds = {"no_helmet": [("a.jpg", 0.9, np.array([0.0, 0.0, 25.0, 10.0]))]}
    row = eval_ppe.evaluate_predictions(preds, gts, [eval_ppe.normalize("NO-Hardhat")], 0.35)
    assert row["no_helmet"]["ap50"] == 0.0
    assert row["no_helmet@iou0.3"]["ap"] == 1.0
    assert row[eval_ppe.CORE_AP50] is None


def test_cli_uses_canonical_prediction_keys_and_writes_coverage(monkeypatch, tmp_path, capsys):
    data = tmp_path / "dataset"
    make_dataset(data)
    calls = []

    class Tensor:
        def __init__(self, values):
            self.values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        names = {0: "person", 1: "Hardhat", 2: "Safety Vest"}

        def __init__(self, path):
            assert path == "local.pt"

        def predict(self, images, **kwargs):
            calls.append(kwargs)
            # Deliberately return results in another order; names collide.
            for image in reversed(images):
                path = Path(image)
                box = [10, 10, 30, 30] if path.parent.name == "val" else [70, 70, 90, 90]
                yield types.SimpleNamespace(
                    path=str(path.parent / ".." / path.parent.name / path.name),
                    boxes=types.SimpleNamespace(xyxy=Tensor([box] * 3), conf=Tensor([0.9] * 3), cls=Tensor([0, 1, 2])),
                )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Model))
    monkeypatch.chdir(tmp_path)
    assert eval_ppe.main(["--data", "dataset", "--splits", "val", "test", "--model", "ppe=local.pt", "--json", "out.json"]) == 0
    row = json.loads((tmp_path / "out.json").read_text())["ppe"]
    for c in eval_ppe.CORE:
        assert row[c] == {"ap50": 1.0, "p": 1.0, "r": 1.0, "n_gt": 2}
    assert row[eval_ppe.CORE_AP50] == 1.0
    assert row["ap_method"] == "coco_101_point"
    assert row["coverage"]["core"]["denominator"] == 3
    assert row["coverage"]["core"]["supported_fraction"] == 1.0
    assert calls[0]["device"] == "cpu" and calls[0]["conf"] == 0.001
    assert "core coverage" in capsys.readouterr().out
