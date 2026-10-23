"""Временные замены модулей Codex, пока их нет в ветке.

Конвейер выбирает реализацию в guard/pipeline/compat.py через try/except ImportError:
- guard.vision (CX-001)        → letterbox, decode_yolov8, decode_yolov8_pose, draw_overlay, draw_event_snapshot;
- guard.tracking (CX-002)      → IouTracker (вместо ByteTrack Tracker), assign_attributes;
- guard.events.manager (CX-003) → SimpleEventManager;
- guard.events.journal (CX-004) → MemoryJournal (в памяти, без SQLite).

Интерфейсы повторяют задания CX-001…CX-004, чтобы после слияния замены отключились сами.
"""

from __future__ import annotations

import csv
import threading
import uuid
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from guard.contracts import (
    BBox,
    CameraConfig,
    Detection,
    Event,
    EventStatus,
    ModelSpec,
    RuleConfig,
    Track,
    Violation,
    Zone,
)
from guard.rules.geometry import bbox_iou, foot_point

# ======================================================================
# CX-001: препроцессинг и декодирование YOLOv8
# ======================================================================


@dataclass(slots=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def letterbox(img_bgr: np.ndarray, size: tuple[int, int] = (640, 640), dtype: str = "float32") -> tuple[np.ndarray, LetterboxMeta]:
    """BGR-кадр → тензор NCHW (1, 3, h, w) RGB; float32 в [0, 1]; паддинг 114 как в ultralytics."""
    h0, w0 = img_bgr.shape[:2]
    th, tw = int(size[0]), int(size[1])
    r = min(th / h0, tw / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = img_bgr if (nw, nh) == (w0, h0) else cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (tw - nw) / 2.0, (th - nh) / 2.0
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    bottom, right = th - nh - top, tw - nw - left
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if dtype == "uint8":
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))[None]
    else:
        tensor = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, swapRB=True)  # NCHW float32 RGB
    return tensor, LetterboxMeta(scale=r, pad_x=float(left), pad_y=float(top), orig_w=w0, orig_h=h0)


def make_batch(tensors: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(tensors), axis=0)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Жадный NMS на numpy → индексы оставленных рамок по убыванию score."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-aware NMS: сдвигаем рамки разных классов, чтобы они не пересекались."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    offset = classes.astype(np.float32)[:, None] * (float(boxes.max()) + 1.0)
    return nms(boxes + offset, scores, iou_thr)


def _as_rows(out: np.ndarray, n_attrs: int | None) -> np.ndarray:
    """(1, A, N) или (1, N, A) → (N, A). A = 4 + nc (det) или 56 (pose)."""
    arr = out[0] if out.ndim == 3 else out
    if n_attrs is not None:
        if arr.shape[0] == n_attrs and arr.shape[1] != n_attrs:
            return arr.T
        if arr.shape[1] == n_attrs:
            return arr
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def _unletterbox(xyxy: np.ndarray, meta: Any) -> np.ndarray:
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, meta.orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, meta.orig_h)
    return out


def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def decode_yolov8(outputs: list[np.ndarray], meta: Any, spec: ModelSpec, max_det: int = 300) -> list[Detection]:
    if not outputs or outputs[0].size == 0:
        return []
    nc = len(spec.classes) or None
    rows = _as_rows(np.asarray(outputs[0]), 4 + nc if nc else None)
    if rows.shape[1] <= 4:
        return []
    cls_scores = rows[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    conf = cls_scores[np.arange(len(rows)), cls_ids]
    mask = conf >= spec.conf_threshold
    if not mask.any():
        return []
    boxes = _cxcywh_to_xyxy(rows[mask, :4].astype(np.float32))
    conf, cls_ids = conf[mask], cls_ids[mask]
    keep = batched_nms(boxes, conf, cls_ids, spec.iou_threshold)[:max_det]
    boxes = _unletterbox(boxes[keep], meta)
    dets: list[Detection] = []
    for b, c, k in zip(boxes, conf[keep], cls_ids[keep]):
        k = int(k)
        name = spec.classes[k] if k < len(spec.classes) else str(k)
        dets.append(Detection(model=spec.model_id, cls=name, cls_id=k, conf=float(c), bbox=tuple(float(v) for v in b)))
    return dets


def decode_yolov8_pose(outputs: list[np.ndarray], meta: Any, spec: ModelSpec, max_det: int = 100) -> list[Detection]:
    if not outputs or outputs[0].size == 0:
        return []
    rows = _as_rows(np.asarray(outputs[0]), 56)
    if rows.shape[1] < 56:
        return []
    conf = rows[:, 4]
    mask = conf >= spec.conf_threshold
    if not mask.any():
        return []
    rows, conf = rows[mask], conf[mask]
    boxes = _cxcywh_to_xyxy(rows[:, :4].astype(np.float32))
    keep = nms(boxes, conf, spec.iou_threshold)[:max_det]
    boxes = _unletterbox(boxes[keep], meta)
    kpts = rows[keep, 5:56].reshape(-1, 17, 3).astype(np.float32).copy()
    kpts[:, :, 0] = ((kpts[:, :, 0] - meta.pad_x) / meta.scale).clip(0, meta.orig_w)
    kpts[:, :, 1] = ((kpts[:, :, 1] - meta.pad_y) / meta.scale).clip(0, meta.orig_h)
    return [
        Detection(model=spec.model_id, cls="person", cls_id=0, conf=float(c), bbox=tuple(float(v) for v in b), keypoints=k)
        for b, c, k in zip(boxes, conf[keep], kpts)
    ]


def decode_batch(outputs: list[np.ndarray], metas: Sequence[Any], spec: ModelSpec) -> list[list[Detection]]:
    decode = decode_yolov8_pose if spec.output_format == "yolov8-pose" else decode_yolov8
    return [decode([o[i : i + 1] for o in outputs], meta, spec) for i, meta in enumerate(metas)]


# ======================================================================
# CX-001: отрисовка (кириллицу в cv2.putText не используем)
# ======================================================================

_GREEN, _ORANGE, _RED, _WHITE = (60, 200, 60), (0, 165, 255), (40, 40, 230), (255, 255, 255)
_ZONE_COLORS = {
    "danger": (40, 40, 230),
    "ppe_required": (0, 200, 255),
    "no_smoking": (200, 80, 200),
    "smoking_area": (120, 200, 120),
    "vehicle_path": (255, 160, 0),
    "height": (0, 120, 255),
    "fence": (180, 180, 180),
}
_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, _WHITE, 1, cv2.LINE_AA)


def draw_zones(frame: np.ndarray, zones: Iterable[Zone], active_ids: set[str] | None = None, alpha: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    zones = list(zones)
    if not zones:
        return frame
    overlay = frame.copy()
    for z in zones:
        pts = np.array([[x * w, y * h] for x, y in z.polygon], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(overlay, [pts], _ZONE_COLORS.get(z.kind, _RED))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    scale = max(0.45, w / 1600)
    for z in zones:
        pts = np.array([[x * w, y * h] for x, y in z.polygon], dtype=np.int32)
        if len(pts) < 3:
            continue
        color = _ZONE_COLORS.get(z.kind, _RED)
        inactive = active_ids is not None and z.zone_id not in active_ids
        cv2.polylines(frame, [pts], True, color, 1 if inactive else 2, cv2.LINE_AA)
        tag = f"{z.zone_id} [{z.kind}]" + (" off" if inactive else "")
        _label(frame, tag, (int(pts[:, 0].min()), int(pts[:, 1].min())), color, scale)
    return frame


def draw_overlay(
    frame: np.ndarray,
    people: Sequence[Track],
    objects: Sequence[Track],
    zones: Sequence[Zone],
    events_highlight: Iterable[int] | None = None,
) -> np.ndarray:
    """Зоны, рамки, ID треков; нарушители (track_id из events_highlight) — красным."""
    highlight = set(events_highlight or ())
    h, w = frame.shape[:2]
    scale = max(0.45, w / 1600)
    thick = max(2, w // 640)
    draw_zones(frame, zones)
    for t in objects:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _ORANGE, thick)
        _label(frame, f"{t.cls} #{t.track_id}", (x1, y1), _ORANGE, scale)
    for t in people:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        bad = t.track_id in highlight
        color = _RED if bad else _GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick + (1 if bad else 0))
        attrs = []
        for key, short in (("helmet", "H"), ("vest", "V")):
            val = t.attributes.get(key)
            if val is not None:
                attrs.append(f"{short}{'+' if val else '-'}")
        text = f"#{t.track_id} {t.conf:.2f}" + (" " + " ".join(attrs) if attrs else "") + (" ALERT" if bad else "")
        _label(frame, text, (x1, y1), color, scale)
        fx, fy = foot_point(t.bbox)
        cv2.circle(frame, (int(fx), int(fy)), thick + 2, color, -1)
        if t.keypoints is not None:
            kp = np.asarray(t.keypoints)
            for a, b in _SKELETON:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), color, 1, cv2.LINE_AA)
    return frame


def draw_event_snapshot(frame: np.ndarray, bbox: BBox, label: str) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    thick = max(3, w // 400)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _RED, thick)
    _label(frame, label, (x1, y1), _RED, max(0.55, w / 1300))
    return frame


# ======================================================================
# CX-002: трекинг (IoU-трекер вместо ByteTrack) и атрибуты
# ======================================================================


class IouTracker:
    """Простой трекер: прогноз по постоянной скорости + жадное сопоставление по IoU.

    Конструктор совместим с guard.tracking.Tracker (CX-002). update() возвращает
    только подтверждённые (hits >= min_hits) треки, обновлённые на этом кадре.
    """

    def __init__(
        self,
        classes: set[str] | None = None,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_iou: float = 0.3,
        new_track_thresh: float = 0.6,
        track_buffer_s: float = 3.0,
        min_hits: int = 2,
        **_: Any,
    ) -> None:
        self.classes = set(classes) if classes else None
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.new_track_thresh = new_track_thresh
        self.track_buffer_s = track_buffer_s
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _predict(t: Track, ts: float) -> BBox:
        dt = max(0.0, min(ts - t.last_seen, 1.0))
        vx, vy = t.velocity
        x1, y1, x2, y2 = t.bbox
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    def _match(self, tracks: list[Track], dets: list[Detection], ts: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        pairs = []
        preds = [self._predict(t, ts) for t in tracks]
        for ti, pb in enumerate(preds):
            for di, d in enumerate(dets):
                if tracks[ti].cls != d.cls:
                    continue
                iou = bbox_iou(pb, d.bbox)
                if iou >= self.match_iou:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        matches = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        return matches, set(range(len(tracks))) - used_t, set(range(len(dets))) - used_d

    def _apply(self, t: Track, d: Detection, ts: float) -> None:
        old_foot = foot_point(t.bbox)
        dt = ts - t.last_seen
        t.bbox, t.conf = tuple(float(v) for v in d.bbox), float(d.conf)
        t.hits += 1
        new_foot = foot_point(t.bbox)
        h = t.bbox[3] - t.bbox[1]
        if dt > 1e-3:
            vx, vy = (new_foot[0] - old_foot[0]) / dt, (new_foot[1] - old_foot[1]) / dt
            a = 0.5
            t.velocity = (a * vx + (1 - a) * t.velocity[0], a * vy + (1 - a) * t.velocity[1])
        t.last_seen = ts
        t.history.append((ts, new_foot[0], new_foot[1], h))
        hs = [item[3] for item in t.history] or [h]
        mean_h = max(float(np.mean(hs[-10:])), 1.0)
        t.speed_bh = float(np.hypot(*t.velocity) / mean_h)
        if d.keypoints is not None:
            t.keypoints = d.keypoints

    def update(self, dets: list[Detection], ts: float) -> list[Track]:
        dets = [d for d in dets if (self.classes is None or d.cls in self.classes) and d.conf >= self.low_thresh]
        high = [d for d in dets if d.conf >= self.high_thresh]
        low = [d for d in dets if d.conf < self.high_thresh]
        # 1) высокая уверенность — ко всем живым трекам
        matches, free_t, free_d = self._match(self._tracks, high, ts)
        updated: set[int] = set()
        for ti, di in matches:
            self._apply(self._tracks[ti], high[di], ts)
            updated.add(ti)
        # 2) низкая уверенность — к оставшимся трекам (как в ByteTrack)
        rest_idx = sorted(free_t)
        rest = [self._tracks[i] for i in rest_idx]
        matches2, _, _ = self._match(rest, low, ts)
        for ri, di in matches2:
            self._apply(rest[ri], low[di], ts)
            updated.add(rest_idx[ri])
        # 3) новые треки из уверенных несопоставленных детекций
        for di in sorted(free_d):
            d = high[di]
            if d.conf < self.new_track_thresh:
                continue
            fx, fy = foot_point(d.bbox)
            t = Track(
                track_id=self._next_id,
                cls=d.cls,
                bbox=tuple(float(v) for v in d.bbox),
                conf=float(d.conf),
                first_seen=ts,
                last_seen=ts,
                keypoints=d.keypoints,
            )
            t.history.append((ts, fx, fy, d.bbox[3] - d.bbox[1]))
            self._next_id += 1
            self._tracks.append(t)
            updated.add(len(self._tracks) - 1)
        # 4) удаляем потерянные дольше track_buffer_s
        active = [self._tracks[i] for i in sorted(updated)]
        self._tracks = [t for t in self._tracks if ts - t.last_seen <= self.track_buffer_s]
        return [t for t in active if t.hits >= self.min_hits]


def assign_attributes(people: list[Track], dets: dict[str, list[Detection]], ts: float) -> None:
    """Минимальная версия: keypoints из модели pose по IoU ≥ 0.5; СИЗ — None (нет данных)."""
    pose = dets.get("pose") or []
    for t in people:
        t.attributes.setdefault("helmet", None)
        t.attributes.setdefault("vest", None)
        t.attributes.setdefault("pose_state", "unknown")
        best, best_iou = None, 0.5
        for d in pose:
            iou = bbox_iou(t.bbox, d.bbox)
            if iou >= best_iou:
                best, best_iou = d, iou
        if best is not None and best.keypoints is not None:
            t.keypoints = best.keypoints


# ======================================================================
# CX-003: подтверждение нарушений и подавление повторов
# ======================================================================


@dataclass
class _Candidate:
    key: tuple
    rule_id: str
    first_ts: float
    last_ts: float
    hits: deque = field(default_factory=deque)  # ts нарушающих кадров
    event_id: str | None = None


@dataclass
class _Tracked:
    event: Event
    rule: RuleConfig
    last_violation_ts: float
    open: bool = True


class SimpleEventManager:
    """Окно подтверждения, одно событие на нарушение, дедуп смены ID по IoU, закрытие по тишине."""

    IOU_DEDUP = 0.3
    GRID = 8

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, deque[float]] = defaultdict(deque)
        self._cands: dict[tuple, _Candidate] = {}
        self._events: dict[str, _Tracked] = {}  # открытые и недавно закрытые (для cooldown)

    @staticmethod
    def _rule(camera: CameraConfig, rule_id: str) -> RuleConfig:
        for r in camera.rules:
            if r.rule_id == rule_id:
                return r
        return RuleConfig(rule_id=rule_id, type="unknown")

    def _key(self, camera: CameraConfig, v: Violation, frame_wh: tuple[float, float] | None) -> tuple:
        if v.track_id is not None:
            return (camera.camera_id, v.rule_id, int(v.track_id))
        cx, cy = (v.bbox[0] + v.bbox[2]) / 2, (v.bbox[1] + v.bbox[3]) / 2
        w, h = frame_wh or (1920.0, 1080.0)
        cell = (min(self.GRID - 1, int(cx / max(w, 1) * self.GRID)), min(self.GRID - 1, int(cy / max(h, 1) * self.GRID)))
        return (camera.camera_id, v.rule_id, v.zone_id, cell)

    def process(
        self,
        camera: CameraConfig,
        ts: float,
        violations: list[Violation],
        timings: dict[str, float] | None = None,
        frame_wh: tuple[float, float] | None = None,
    ) -> tuple[list[Event], list[Event]]:
        cam_id = camera.camera_id
        new: list[Event] = []
        closed: list[Event] = []
        with self._lock:
            frames = self._frames[cam_id]
            frames.append(ts)
            while frames and frames[0] < ts - 120.0:
                frames.popleft()
            seen: set[tuple] = set()
            extended_now: set[str] = set()  # события, уже продлённые на этом кадре своим ключом
            pending: list[tuple[_Candidate, Violation, RuleConfig]] = []
            # Проход 1: продлеваем свои события, остальным ведём серию
            for v in violations:
                key = self._key(camera, v, frame_wh)
                if key in seen:
                    continue
                seen.add(key)
                rule = self._rule(camera, v.rule_id)
                cand = self._cands.get(key)
                tracked = self._events.get(cand.event_id) if cand is not None and cand.event_id else None
                if tracked is not None and (tracked.open or ts - tracked.last_violation_ts <= rule.cooldown_s):
                    cand.last_ts = ts
                    self._extend(tracked, v, ts)
                    extended_now.add(tracked.event.id)
                    continue
                if cand is not None and (cand.event_id is not None or ts - cand.last_ts > rule.min_duration_s):
                    cand = None  # серия прервалась или событие уже забыто — начинаем заново
                if cand is None:
                    cand = self._cands[key] = _Candidate(key=key, rule_id=v.rule_id, first_ts=ts, last_ts=ts)
                cand.hits.append(ts)
                cand.last_ts = ts
                while cand.hits and cand.hits[0] < ts - max(rule.min_duration_s, 0.0) - 1e-6:
                    cand.hits.popleft()
                pending.append((cand, v, rule))
            # Проход 2: смена ID трека (дедуп по IoU) или подтверждение нового события
            for cand, v, rule in pending:
                dup = self._find_duplicate(cam_id, v, ts, rule, exclude=extended_now)
                if dup is not None:
                    cand.event_id = dup.event.id
                    self._extend(dup, v, ts)
                    extended_now.add(dup.event.id)
                    continue
                if ts - cand.first_ts + 1e-6 < rule.min_duration_s:
                    continue
                window = [f for f in frames if f >= ts - rule.min_duration_s - 1e-6]
                ratio = len(cand.hits) / max(len(window), 1)
                if ratio + 1e-9 < rule.min_ratio:
                    continue
                event = Event(
                    id=uuid.uuid4().hex,
                    camera_id=cam_id,
                    site=camera.site,
                    zone_id=v.zone_id,
                    rule_id=v.rule_id,
                    type=v.type,
                    severity=v.severity,
                    track_id=v.track_id,
                    started_at=cand.first_ts,
                    confirmed_at=ts,
                    bbox=tuple(float(x) for x in v.bbox),
                    details=dict(v.details),
                    latency_ms=dict(timings or {}),
                )
                self._events[event.id] = _Tracked(event=event, rule=rule, last_violation_ts=ts)
                cand.event_id = event.id
                extended_now.add(event.id)
                new.append(event)
            # закрытие и уборка
            for ev_id, tr in list(self._events.items()):
                if tr.event.camera_id != cam_id:
                    continue
                if tr.open and ts - tr.last_violation_ts > tr.rule.close_after_s:
                    tr.open = False
                    tr.event.ended_at = tr.last_violation_ts
                    closed.append(tr.event)
                if not tr.open and ts - tr.last_violation_ts > tr.rule.cooldown_s:
                    del self._events[ev_id]
            for key, cand in list(self._cands.items()):
                if key[0] != cam_id:
                    continue
                rule = self._rule(camera, cand.rule_id)
                horizon = max(rule.min_duration_s, rule.close_after_s, rule.cooldown_s if cand.event_id else 0.0)
                if ts - cand.last_ts > horizon:
                    del self._cands[key]
        return new, closed

    def _extend(self, tracked: _Tracked, v: Violation, ts: float) -> None:
        tracked.last_violation_ts = ts
        tracked.event.bbox = tuple(float(x) for x in v.bbox)
        if not tracked.open:  # нарушение возобновилось в пределах cooldown — то же событие
            tracked.open = True
            tracked.event.ended_at = None

    def _find_duplicate(
        self, cam_id: str, v: Violation, ts: float, rule: RuleConfig, exclude: set[str] | None = None
    ) -> _Tracked | None:
        best, best_iou = None, self.IOU_DEDUP
        for tr in self._events.values():
            ev = tr.event
            if ev.camera_id != cam_id or ev.rule_id != v.rule_id or (exclude and ev.id in exclude):
                continue
            if not tr.open and ts - tr.last_violation_ts > rule.cooldown_s:
                continue
            iou = bbox_iou(ev.bbox, v.bbox)
            if iou >= best_iou:
                best, best_iou = tr, iou
        return best

    def open_events(self, camera_id: str | None = None) -> list[Event]:
        with self._lock:
            return [t.event for t in self._events.values() if t.open and (camera_id is None or t.event.camera_id == camera_id)]

    def reset(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(camera_id, None)
            for key in [k for k in self._cands if k[0] == camera_id]:
                del self._cands[key]
            for ev_id in [e for e, t in self._events.items() if t.event.camera_id == camera_id]:
                del self._events[ev_id]


# ======================================================================
# CX-004: журнал событий (в памяти)
# ======================================================================

_UPDATABLE = {"status", "operator_note", "ended_at", "snapshot_path", "clip_path", "details"}
_GROUP_FIELDS = {"type": "type", "camera": "camera_id", "zone": "zone_id", "severity": "severity", "status": "status"}


def _as_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return {getattr(v, "value", v) for v in value}
    return {getattr(value, "value", value)}


class MemoryJournal:
    """Журнал в памяти с интерфейсом guard.events.journal.Journal (CX-004)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._events: dict[str, Event] = {}

    def add(self, event: Event) -> None:
        with self._lock:
            self._events[event.id] = event.model_copy(deep=True)

    def get(self, event_id: str) -> Event | None:
        with self._lock:
            ev = self._events.get(event_id)
            return ev.model_copy(deep=True) if ev else None

    def update(self, event_id: str, **fields: Any) -> Event:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"fields not updatable: {sorted(bad)}")
        with self._lock:
            ev = self._events.get(event_id)
            if ev is None:
                raise KeyError(event_id)
            data = ev.model_dump()
            data.update(fields)
            new = Event.model_validate(data)
            self._events[event_id] = new
            return new.model_copy(deep=True)

    def _filter(
        self,
        type: Any = None,
        camera_id: Any = None,
        zone_id: Any = None,
        status: Any = None,
        severity: Any = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[Event]:
        f_type, f_cam, f_zone = _as_set(type), _as_set(camera_id), _as_set(zone_id)
        f_status, f_sev = _as_set(status), _as_set(severity)
        with self._lock:
            items = list(self._events.values())
        out = []
        for ev in items:
            if f_type and ev.type not in f_type:
                continue
            if f_cam and ev.camera_id not in f_cam:
                continue
            if f_zone and ev.zone_id not in f_zone:
                continue
            if f_status and ev.status.value not in f_status:
                continue
            if f_sev and ev.severity.value not in f_sev:
                continue
            if ts_from is not None and ev.confirmed_at < ts_from:
                continue
            if ts_to is not None and ev.confirmed_at > ts_to:
                continue
            out.append(ev)
        return out

    def query(
        self,
        type: Any = None,
        camera_id: Any = None,
        zone_id: Any = None,
        status: Any = None,
        severity: Any = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
    ) -> tuple[list[Event], int]:
        items = self._filter(type, camera_id, zone_id, status, severity, ts_from, ts_to)
        items.sort(key=lambda e: e.confirmed_at, reverse=(order != "asc"))
        page = items[offset : offset + limit] if limit else items[offset:]
        return [e.model_copy(deep=True) for e in page], len(items)

    def stats(
        self, group_by: str, ts_from: float | None = None, ts_to: float | None = None, tz: str = "Europe/Moscow", **filters: Any
    ) -> list[dict[str, Any]]:
        items = self._filter(ts_from=ts_from, ts_to=ts_to, **filters)
        zone = ZoneInfo(tz)
        counts: dict[Any, int] = defaultdict(int)
        for ev in items:
            if group_by in _GROUP_FIELDS:
                key = getattr(ev, _GROUP_FIELDS[group_by])
                key = getattr(key, "value", key)
            elif group_by == "hour":
                key = datetime.fromtimestamp(ev.confirmed_at, zone).hour
            elif group_by == "day":
                key = datetime.fromtimestamp(ev.confirmed_at, zone).strftime("%Y-%m-%d")
            else:
                raise ValueError(f"unsupported group_by {group_by!r}")
            counts[key] += 1
        ordered = sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else ""))
        return [{"key": k, "count": c} for k, c in ordered]

    def export_csv(self, path: str | Path, **filters: Any) -> Path:
        items, _ = self.query(limit=0, order="asc", **filters)
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["ID", "Камера", "Участок", "Зона", "Тип", "Важность", "Начало", "Подтверждено", "Окончание", "Статус", "Комментарий"])
            for e in items:
                w.writerow([e.id, e.camera_id, e.site, e.zone_id or "", e.type, e.severity.value, e.started_at, e.confirmed_at, e.ended_at or "", e.status.value, e.operator_note or ""])
        return path

    def close(self) -> None:
        return None


__all__ = [
    "EventStatus",
    "IouTracker",
    "LetterboxMeta",
    "MemoryJournal",
    "SimpleEventManager",
    "assign_attributes",
    "batched_nms",
    "decode_batch",
    "decode_yolov8",
    "decode_yolov8_pose",
    "draw_event_snapshot",
    "draw_overlay",
    "draw_zones",
    "letterbox",
    "make_batch",
    "nms",
]
