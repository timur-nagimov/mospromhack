"""Минимальный пре- и постпроцессинг YOLOv8 для проверок в ml/.

Боевой декодер живёт в guard/vision/; этот модуль нужен только для
ml/check_onnx.py и сверки ONNX ↔ TPU. Формат выходов — как у экспорта Ultralytics без NMS:
  detect: (1, 4 + nc, N) — cx, cy, w, h в пикселях входа, затем sigmoid-скоры классов;
  pose:   (1, 4 + 1 + 17*3, N) — рамка, скор person, ключевые точки (x, y, conf).
"""

from __future__ import annotations

import cv2
import numpy as np

from guard.contracts import Detection, ModelSpec

PAD_VALUE = 114

# COCO-17: пары точек скелета (индексы с 0)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """BGR-кадр → тензор NCHW float32 RGB 0..1 с паддингом 114 по центру (как Ultralytics)."""
    h0, w0 = img.shape[:2]
    th, tw = size
    r = min(th / h0, tw / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (tw - nw) / 2, (th - nh) / 2
    if (nw, nh) != (w0, h0):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), {"ratio": r, "pad": (left, top), "orig_shape": (h0, w0)}


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def _scale_back(xy: np.ndarray, meta: dict) -> np.ndarray:
    px, py = meta["pad"]
    r = meta["ratio"]
    out = xy.copy()
    out[..., 0] = (out[..., 0] - px) / r
    out[..., 1] = (out[..., 1] - py) / r
    return out


def decode(
    outputs: list[np.ndarray],
    meta: dict,
    spec: ModelSpec,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int = 300,
) -> list[Detection]:
    conf = spec.conf_threshold if conf is None else conf
    iou = spec.iou_threshold if iou is None else iou
    pred = outputs[0][0].T  # (N, C)
    is_pose = spec.output_format == "yolov8-pose"
    nc = 1 if is_pose else pred.shape[1] - 4
    cls_scores = pred[:, 4 : 4 + nc]
    cls_id = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(pred)), cls_id]
    m = score >= conf
    pred, cls_id, score = pred[m], cls_id[m], score[m]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    # NMS по классам через сдвиг координат на класс
    keep = nms(boxes + cls_id[:, None] * 7680.0, score, iou)[:max_det]
    boxes, cls_id, score, pred = boxes[keep], cls_id[keep], score[keep], pred[keep]
    h0, w0 = meta["orig_shape"]
    xy = _scale_back(boxes.reshape(-1, 2, 2), meta).reshape(-1, 4)
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w0)
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h0)
    kpts = None
    if is_pose:
        kpts = pred[:, 5:].reshape(-1, 17, 3).copy()
        kpts[..., :2] = _scale_back(kpts[..., :2], meta)
    dets = []
    for i in range(len(xy)):
        c = int(cls_id[i])
        name = spec.classes[c] if c < len(spec.classes) else str(c)
        dets.append(
            Detection(
                model=spec.model_id,
                cls=name,
                cls_id=c,
                conf=float(score[i]),
                bbox=tuple(float(v) for v in xy[i]),
                keypoints=None if kpts is None else kpts[i],
            )
        )
    return dets


_COLORS = {
    "person": (255, 160, 0),
    "helmet": (0, 200, 0),
    "vest": (0, 220, 120),
    "respirator": (0, 180, 180),
    "no_helmet": (0, 0, 255),
    "no_vest": (0, 60, 255),
    "no_respirator": (60, 0, 255),
    "fire": (0, 80, 255),
    "smoke": (160, 160, 160),
}


def color_for(name: str, cid: int) -> tuple[int, int, int]:
    if name in _COLORS:
        return _COLORS[name]
    rng = np.random.default_rng(cid * 7919 + 17)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def draw(img: np.ndarray, dets: list[Detection], kpt_conf: float = 0.5) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(out.shape[:2]) / 400))
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        col = color_for(d.cls, d.cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        label = f"{d.cls} {d.conf:.2f}"
        fs = 0.4 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        yt = max(y1, th + 4)
        cv2.rectangle(out, (x1, yt - th - 4), (x1 + tw + 2, yt), col, -1)
        cv2.putText(out, label, (x1 + 1, yt - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t // 2))
        if d.keypoints is not None:
            k = d.keypoints
            for a, b in SKELETON:
                if k[a, 2] >= kpt_conf and k[b, 2] >= kpt_conf:
                    cv2.line(out, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), (255, 255, 0), t)
            for x, y, c in k:
                if c >= kpt_conf:
                    cv2.circle(out, (int(x), int(y)), t + 2, (0, 255, 255), -1)
    return out
