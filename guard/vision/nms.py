"""Deterministic numpy NMS for continuous xyxy coordinates."""

from __future__ import annotations

import numpy as np


def _validate(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError("scores must have shape (N,)")
    if not np.isfinite(iou_thr) or not 0 <= iou_thr <= 1:
        raise ValueError("iou_thr must be finite and between 0 and 1")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("boxes and scores must be finite")
    return boxes, scores


def nms(
    boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float
) -> np.ndarray:
    """Return original indices sorted by descending score.

    Suppress boxes with IoU strictly greater than iou_thr. Equal scores
    preserve input order. Zero-area boxes have zero IoU. Pixel coordinates
    are continuous: area uses width * height, without an extra pixel.
    """
    boxes, scores = _validate(boxes_xyxy, scores, iou_thr)
    order = np.argsort(-scores, kind="stable")
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    areas = sizes[:, 0] * sizes[:, 1]
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        if not rest.size:
            break
        intersection_sizes = np.maximum(
            np.minimum(boxes[current, 2:], boxes[rest, 2:])
            - np.maximum(boxes[current, :2], boxes[rest, :2]),
            0,
        )
        intersection = intersection_sizes[:, 0] * intersection_sizes[:, 1]
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= iou_thr]

    return np.asarray(kept, dtype=np.intp)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float,
) -> np.ndarray:
    """Apply NMS independently per class and return score-sorted indices."""
    boxes, scores = _validate(boxes, scores, iou_thr)
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(classes) != len(boxes):
        raise ValueError("classes must have shape (N,)")

    kept: list[int] = []
    for cls in np.unique(classes):
        indices = np.flatnonzero(classes == cls)
        local_keep = nms(boxes[indices], scores[indices], iou_thr)
        kept.extend(indices[local_keep].tolist())

    result = np.asarray(kept, dtype=np.intp)
    # Original index breaks score ties consistently across different classes.
    return result[np.lexsort((result, -scores[result]))]
