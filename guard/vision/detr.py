"""Explicit raw HF D-FINE focal-loss decoder, using only numpy.

The output contract is ordered [logits (B,Q,C), pred_boxes (B,Q,4)].
This is not inferred from tensor size and does not use YOLO NMS.
"""

from __future__ import annotations

import numpy as np

from guard.contracts import Detection, ModelSpec
from guard.vision.preprocess import DirectResizeMeta


def validate_dfine_spec(spec: ModelSpec) -> None:
    if spec.output_format != "dfine" or spec.task != "detect" or spec.input_dtype != "float32":
        raise ValueError("D-FINE requires explicit dfine detect/float32 ModelSpec")
    if not spec.classes or len(set(spec.classes)) != len(spec.classes) or any(not name for name in spec.classes):
        raise ValueError("D-FINE requires explicit unique non-empty class names")
    if not np.isfinite(spec.conf_threshold) or not 0 <= spec.conf_threshold <= 1:
        raise ValueError("D-FINE conf_threshold must be finite and between 0 and 1")


def validate_dfine_outputs(outputs: list[np.ndarray], class_count: int, *, batch_size: int,
                           queries: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reject contract/finite-value failures rather than returning empty results."""
    if len(outputs) != 2:
        raise ValueError("D-FINE requires two ordered outputs: logits, pred_boxes")
    logits, boxes = (np.asarray(value) for value in outputs)
    if (logits.ndim != 3 or boxes.ndim != 3 or logits.shape[0] != batch_size
            or logits.shape[1] <= 0 or logits.shape[2] != class_count
            or boxes.shape != (*logits.shape[:2], 4)
            or (queries is not None and logits.shape[1] != queries)):
        raise ValueError("D-FINE output shape mismatch: expected logits (B,Q,C), pred_boxes (B,Q,4)")
    if logits.dtype != np.float32 or boxes.dtype != np.float32:
        raise ValueError("D-FINE outputs must be float32")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise ValueError("Non-finite D-FINE model output")
    return logits, boxes


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_negative = np.exp(logits[~positive])
    scores[~positive] = exp_negative / (1 + exp_negative)
    return scores


def decode_dfine_batch(outputs: list[np.ndarray], metas: list[DirectResizeMeta],
                       spec: ModelSpec) -> list[list[Detection]]:
    """Focal sigmoid; global top-Q query/class pairs; strict score > threshold.

    Equal-score pairs use flattened index order for deterministic ties. Multiple
    classes from one query are retained, with no NMS. Normalized cxcywh boxes
    are scaled to each original image and clipped to its bounds.
    """
    validate_dfine_spec(spec)
    logits, boxes = validate_dfine_outputs(outputs, len(spec.classes), batch_size=len(metas))
    results = []
    for item_logits, item_boxes, meta in zip(logits, boxes, metas):
        if (not isinstance(meta, DirectResizeMeta) or not np.isfinite([meta.orig_w, meta.orig_h]).all()
                or meta.orig_w <= 0 or meta.orig_h <= 0):
            raise ValueError("D-FINE requires positive original dimensions from direct resize")
        scores = _sigmoid(item_logits).ravel()
        # Sorting all Q*C pairs is small (300*80) and gives reproducible ties.
        indices = np.argsort(-scores, kind="stable")[:len(item_logits)]
        indices = indices[scores[indices] > spec.conf_threshold]
        # FDR's affine refinements are in normalized units, but need not lie
        # within [0,1]. Clip only after scaling; finite off-frame queries are valid.
        selected = item_boxes[indices // len(spec.classes)].astype(np.float64)
        xyxy = np.concatenate((selected[:, :2] - selected[:, 2:] / 2,
                               selected[:, :2] + selected[:, 2:] / 2), axis=1)
        xyxy *= np.array([meta.orig_w, meta.orig_h, meta.orig_w, meta.orig_h], dtype=np.float64)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, meta.orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, meta.orig_h)
        detections = []
        for index, box in zip(indices, xyxy):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            class_id = int(index % len(spec.classes))
            detections.append(Detection(model=spec.model_id, cls=spec.classes[class_id], cls_id=class_id,
                                        conf=float(scores[index]), bbox=tuple(float(value) for value in box)))
        results.append(detections)
    return results


def decode_dfine(outputs: list[np.ndarray], meta: DirectResizeMeta, spec: ModelSpec) -> list[Detection]:
    return decode_dfine_batch(outputs, [meta], spec)[0]
