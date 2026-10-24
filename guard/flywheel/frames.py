"""Кадр и детекции сэмпла: JPEG без оверлеев и детекции в пикселях сохранённого кадра.

Всё здесь вызывается в фоновом потоке (писатель или отправитель edge), не в горячем пути кадра.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from guard.contracts import Detection
from guard.flywheel.select import box_iou

TrackRow = tuple[int, str, tuple[float, float, float, float]]  # track_id, cls, bbox
TRACK_MATCH_IOU = 0.5


def encode_jpeg(
    frame: np.ndarray, max_width: int = 1280, quality: int = 90, max_kb: int | None = None
) -> tuple[bytes, int, int, float]:
    """JPEG кадра шириной не больше max_width. Возвращает (данные, ширина, высота, масштаб)."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w) if max_width and w > max_width else 1.0
    img = frame
    if scale < 1.0:
        img = cv2.resize(frame, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    q = int(quality)
    while True:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise ValueError("JPEG encoding failed")
        data = buf.tobytes()
        if max_kb is None or len(data) <= max_kb * 1024 or q <= 40:
            return data, int(img.shape[1]), int(img.shape[0]), scale
        q -= 10


def decode_size(jpeg: bytes) -> tuple[int, int]:
    """Ширина и высота JPEG без полной распаковки не нужны часто — распаковываем."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a JPEG")
    return int(img.shape[1]), int(img.shape[0])


def snapshot_tracks(people: Iterable[Any], objects: Iterable[Any]) -> list[TrackRow]:
    """Снимок треков кадра: трекер меняет объекты на следующих кадрах (дёшево, в горячем пути)."""
    rows: list[TrackRow] = []
    for t in (*people, *objects):
        rows.append((int(t.track_id), str(t.cls), tuple(float(v) for v in t.bbox)))  # type: ignore[arg-type]
    return rows


def _match_track(det: Detection, tracks: Sequence[TrackRow]) -> int | None:
    best_id, best_iou = None, TRACK_MATCH_IOU
    for tid, cls, bbox in tracks:
        if cls != det.cls:
            continue
        iou = box_iou(det.bbox, bbox)
        if iou >= best_iou:
            best_id, best_iou = tid, iou
    return best_id


def detections_to_dicts(
    detections: Mapping[str, Iterable[Detection]], tracks: Sequence[TrackRow] = (), scale: float = 1.0
) -> list[dict[str, Any]]:
    """Детекции всех моделей кадра → [{model_id, cls, conf, bbox, track_id}] в пикселях сохранённого кадра."""
    out: list[dict[str, Any]] = []
    for model_id, dets in detections.items():
        for det in dets:
            x1, y1, x2, y2 = (float(v) * scale for v in det.bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                {
                    "model_id": str(model_id),
                    "cls": str(det.cls),
                    "conf": round(float(det.conf), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "track_id": _match_track(det, tracks),
                }
            )
    return out


def scale_box(box: Iterable[float], scale: float) -> list[float]:
    return [round(float(v) * scale, 1) for v in box]
