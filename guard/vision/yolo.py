"""Decode raw, concatenated YOLOv8 detect/pose outputs using numpy."""

from __future__ import annotations

import numpy as np

from guard.contracts import Detection, ModelSpec
from guard.vision.nms import batched_nms
from guard.vision.preprocess import LetterboxMeta


def _channels(spec: ModelSpec, pose: bool) -> int:
    if pose:
        return 56
    if not spec.classes:
        raise ValueError("YOLOv8 detection requires non-empty spec.classes")
    return 4 + len(spec.classes)


def _prediction_rows(outputs: list[np.ndarray], channels: int) -> np.ndarray:
    """Normalize one raw output to (batch, candidates, channels).

    Square, inherently ambiguous layouts use channels-first precedence.
    Separate feature maps and exports with embedded NMS are unsupported.
    """
    if len(outputs) != 1:
        raise ValueError("expected one concatenated YOLOv8 output tensor")
    prediction = np.asarray(outputs[0])
    if prediction.ndim != 3:
        raise ValueError("YOLOv8 output must have shape (B, C, N) or (B, N, C)")
    if prediction.shape[1] == channels:
        return prediction.transpose(0, 2, 1)
    if prediction.shape[2] == channels:
        return prediction
    raise ValueError(
        f"expected {channels} output channels, received shape {prediction.shape}"
    )


def _restore_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] = np.clip(
        (restored[:, [0, 2]] - meta.pad_x) / meta.scale, 0, meta.orig_w
    )
    restored[:, [1, 3]] = np.clip(
        (restored[:, [1, 3]] - meta.pad_y) / meta.scale, 0, meta.orig_h
    )
    return restored


def _decode_rows(
    rows: np.ndarray,
    meta: LetterboxMeta,
    spec: ModelSpec,
    pose: bool,
) -> list[Detection]:
    if not np.isfinite(meta.scale) or meta.scale <= 0:
        raise ValueError("letterbox scale must be finite and positive")
    if meta.orig_w <= 0 or meta.orig_h <= 0:
        raise ValueError("original image dimensions must be positive")
    if not np.isfinite(spec.conf_threshold) or not 0 <= spec.conf_threshold <= 1:
        raise ValueError("conf_threshold must be finite and between 0 and 1")
    if not np.isfinite(spec.iou_threshold) or not 0 <= spec.iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and between 0 and 1")
    if len(rows) == 0:
        return []

    rows = np.asarray(rows, dtype=np.float32)
    # Invalid candidates must not influence argmax or suppress valid boxes.
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 2:4] > 0).all(axis=1)
    rows = rows[valid]
    if len(rows) == 0:
        return []

    if pose:
        class_ids = np.zeros(len(rows), dtype=np.intp)
        scores = rows[:, 4]
    else:
        probabilities = rows[:, 4:]
        class_ids = probabilities.argmax(axis=1)
        scores = probabilities[np.arange(len(rows)), class_ids]

    accepted = scores >= spec.conf_threshold
    rows = rows[accepted]
    scores = scores[accepted]
    class_ids = class_ids[accepted]
    if len(rows) == 0:
        return []

    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
    boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    keep = batched_nms(boxes, scores, class_ids, spec.iou_threshold)
    restored = _restore_boxes(boxes[keep], meta)

    detections: list[Detection] = []
    for index, bbox in zip(keep, restored):
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        keypoints = None
        if pose:
            keypoints = rows[index, 5:].reshape(17, 3).copy()
            keypoints[:, 0] = np.clip(
                (keypoints[:, 0] - meta.pad_x) / meta.scale, 0, meta.orig_w
            )
            keypoints[:, 1] = np.clip(
                (keypoints[:, 1] - meta.pad_y) / meta.scale, 0, meta.orig_h
            )
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                model=spec.model_id,
                cls="person" if pose else spec.classes[class_id],
                cls_id=class_id,
                conf=float(scores[index]),
                bbox=tuple(float(value) for value in bbox),
                keypoints=keypoints,
            )
        )
    return detections


def decode_yolov8(
    outputs: list[np.ndarray],
    meta: LetterboxMeta,
    spec: ModelSpec,
) -> list[Detection]:
    """Decode one image: cx, cy, w, h followed by class probabilities.

    Uses the highest-scoring class per candidate, an inclusive confidence
    threshold, and class-aware NMS before coordinate restoration/clipping.
    Coordinates are pixels in the letterboxed input, not normalized values.
    Neither sigmoid nor objectness multiplication is applied.
    """
    if not outputs:
        return []
    rows = _prediction_rows(outputs, _channels(spec, pose=False))
    if rows.shape[0] != 1:
        raise ValueError("single-image decoder requires batch size 1")
    return _decode_rows(rows[0], meta, spec, pose=False)


def decode_yolov8_pose(
    outputs: list[np.ndarray],
    meta: LetterboxMeta,
    spec: ModelSpec,
) -> list[Detection]:
    """Decode 4 box + 1 person probability + 17 * (x, y, confidence).

    Supports both (1, 56, N) and (1, N, 56). Keypoint confidence is preserved;
    coordinates are restored and clipped to the original image bounds.
    """
    if not outputs:
        return []
    rows = _prediction_rows(outputs, _channels(spec, pose=True))
    if rows.shape[0] != 1:
        raise ValueError("single-image decoder requires batch size 1")
    return _decode_rows(rows[0], meta, spec, pose=True)


def decode_batch(
    outputs: list[np.ndarray],
    metas: list[LetterboxMeta],
    spec: ModelSpec,
) -> list[list[Detection]]:
    """Decode each batch item using its own letterbox metadata."""
    if spec.task == "classify" or spec.output_format == "raw":
        raise ValueError("decode_batch supports raw YOLOv8 detect/pose heads only")
    if not outputs:
        return [[] for _ in metas]
    pose = spec.task == "pose" or spec.output_format == "yolov8-pose"
    rows = _prediction_rows(outputs, _channels(spec, pose))
    if rows.shape[0] != len(metas):
        raise ValueError("output batch size must equal the number of metadata entries")
    return [
        _decode_rows(item, meta, spec, pose)
        for item, meta in zip(rows, metas)
    ]
