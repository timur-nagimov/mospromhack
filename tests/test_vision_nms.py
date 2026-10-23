import numpy as np
import pytest

from guard.vision.nms import batched_nms, nms


def test_nms_removes_duplicates_and_orders_by_score():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]],
        dtype=np.float32,
    )
    original = boxes.copy()
    scores = np.array([0.8, 0.9, 0.7])
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), [1, 2])
    np.testing.assert_array_equal(boxes, original)


def test_class_aware_nms_keeps_overlapping_different_classes():
    boxes = np.array([[0, 0, 10, 10]] * 4)
    scores = np.array([0.8, 0.9, 0.7, 0.6])
    classes = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(
        batched_nms(boxes, scores, classes, 0.5), [1, 0]
    )


def test_equal_scores_preserve_original_order_across_classes():
    boxes = np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
    )
    scores = np.ones(3)
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), [0, 1, 2])
    np.testing.assert_array_equal(
        batched_nms(boxes, scores, np.array([2, 0, 1]), 0.5),
        [0, 1, 2],
    )


def test_iou_threshold_is_strict_and_uses_continuous_coordinates():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
    scores = np.array([0.9, 0.8])
    np.testing.assert_array_equal(nms(boxes, scores, 1 / 3), [0, 1])
    np.testing.assert_array_equal(nms(boxes, scores, 0.32), [0])


def test_zero_area_boxes_do_not_produce_nan():
    boxes = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 2, 2]])
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(nms(boxes, [0.9, 0.8, 0.7], 0), [0, 1, 2])


def test_empty_inputs():
    for result in (
        nms([], [], 0.5),
        batched_nms(np.empty((0, 4)), np.empty(0), np.empty(0), 0.5),
    ):
        assert result.shape == (0,)
        assert result.dtype == np.dtype(np.intp)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        nms(np.empty((0, 4)), np.empty(0), threshold)


def test_invalid_shapes_and_nonfinite_values():
    with pytest.raises(ValueError):
        nms(np.zeros((2, 3)), np.zeros(2), 0.5)
    with pytest.raises(ValueError):
        nms(np.zeros((2, 4)), np.zeros(1), 0.5)
    with pytest.raises(ValueError):
        nms(np.zeros((1, 4)), [np.nan], 0.5)
    with pytest.raises(ValueError):
        batched_nms(np.zeros((2, 4)), np.ones(2), [0], 0.5)
