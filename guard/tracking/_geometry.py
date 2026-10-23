"""Geometry and threshold-aware linear assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from guard.contracts import BBox, Detection


def valid_detection(det: Detection) -> bool:
    box = np.asarray(det.bbox, dtype=float)
    return bool(
        box.shape == (4,)
        and np.all(np.isfinite(box))
        and box[2] > box[0]
        and box[3] > box[1]
        and np.isfinite(det.conf)
        and 0.0 <= det.conf <= 1.0
    )


def iou_matrix(left: list[BBox], right: list[BBox]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=float)
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(hi - lo, 0.0), axis=2)
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0.0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0.0), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def match_boxes(
    left: list[BBox],
    right: list[BBox],
    threshold: float,
    allowed: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Maximize valid match count, then total IoU.

    Apply gating before assignment. The penalty exceeds the total possible
    valid-edge cost, so invalid edges cannot displace a larger valid matching.
    """
    overlaps = iou_matrix(left, right)
    if overlaps.size == 0:
        return []
    valid = overlaps >= threshold
    if allowed is not None:
        valid &= allowed
    penalty = float(min(overlaps.shape) + 1)
    costs = np.where(valid, 1.0 - overlaps, penalty)
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if valid[row, column]
    ]
