"""OpenCV overlays. All public functions return a copy of the BGR frame."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import cv2
import numpy as np

from guard.contracts import BBox, Detection, Event, Track, Violation, Zone

_RED = (0, 0, 255)
_GREEN = (60, 210, 60)
_ORANGE = (0, 180, 255)
_ZONE_COLOR = (210, 120, 40)

# COCO's 17-keypoint skeleton, with zero-based indices.
_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def _copy_frame(frame: np.ndarray) -> np.ndarray:
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
        or frame.dtype != np.uint8
    ):
        raise ValueError("frame must be a non-empty uint8 BGR array")
    return frame.copy()


def _ascii(text: object) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


def _label(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    text = _ascii(text)
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    frame_h, frame_w = frame.shape[:2]
    x = int(np.clip(anchor[0], 0, max(0, frame_w - width - 6)))
    top = max(0, anchor[1] - height - baseline - 6)
    bottom = min(frame_h - 1, top + height + baseline + 6)
    cv2.rectangle(
        frame, (x, top), (min(frame_w - 1, x + width + 6), bottom), color, -1
    )
    cv2.putText(
        frame,
        text,
        (x + 3, min(frame_h - 1, top + height + 3)),
        font,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_box(
    frame: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    coordinates = np.asarray(bbox, dtype=np.float64)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        return
    height, width = frame.shape[:2]
    if (
        coordinates[2] < 0 or coordinates[3] < 0
        or coordinates[0] >= width or coordinates[1] >= height
    ):
        return
    coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, width - 1)
    coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, height - 1)
    x1, y1, x2, y2 = map(int, np.rint(coordinates))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    _label(frame, label, (x1, y1), color)


def _draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    if keypoints is None:
        return
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (17, 3):
        return
    height, width = frame.shape[:2]
    visible = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= 0.5)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    pixels = {
        index: (
            min(width - 1, int(round(points[index, 0]))),
            min(height - 1, int(round(points[index, 1]))),
        )
        for index in np.flatnonzero(visible)
    }
    for first, second in _SKELETON:
        if first in pixels and second in pixels:
            cv2.line(frame, pixels[first], pixels[second], color, 2, cv2.LINE_AA)
    for point in pixels.values():
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def _track_label(item: Track | Detection) -> str:
    parts = [_ascii(item.cls)]
    if isinstance(item, Track):
        parts.append(f"#{item.track_id}")
        for name in ("helmet", "vest", "respirator", "phone"):
            if name not in item.attributes:
                continue
            value = item.attributes[name]
            if value is None:
                state = "?"
            elif isinstance(value, (bool, np.bool_)):
                state = "yes" if value else "no"
            else:
                state = _ascii(value)
            parts.append(f"{name}:{state}")
        if "pose_state" in item.attributes:
            parts.append(f"pose:{_ascii(item.attributes['pose_state'])}")
    return " ".join(parts)


def draw_overlay(
    frame: np.ndarray,
    people: Iterable[Track | Detection],
    objects: Iterable[Track | Detection],
    zones: Iterable[Zone],
    events_highlight: Iterable[int | Event | Violation] | None = None,
) -> np.ndarray:
    """Draw tracks, PPE attributes, zones, skeletons, and violations.

    events_highlight accepts track IDs or Event/Violation objects. Objects
    with track_id=None are highlighted directly using their event bbox.
    Zone labels use zone_id; arbitrary non-ASCII label characters become '?'.
    """
    result = _copy_frame(frame)
    height, width = result.shape[:2]

    for zone in zones:
        polygon = np.asarray(zone.polygon, dtype=np.float64)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
            or not np.isfinite(polygon).all()
        ):
            continue
        pixels = np.rint(
            np.clip(polygon, 0, 1) * np.array([width - 1, height - 1])
        ).astype(np.int32)
        overlay = result.copy()
        cv2.fillPoly(overlay, [pixels], _ZONE_COLOR)
        cv2.addWeighted(overlay, 0.2, result, 0.8, 0, dst=result)
        cv2.polylines(result, [pixels], True, _ZONE_COLOR, 1, cv2.LINE_AA)
        _label(
            result,
            f"zone:{_ascii(zone.zone_id)}",
            (int(pixels[0, 0]), int(pixels[0, 1])),
            _ZONE_COLOR,
        )

    highlighted_ids: set[int] = set()
    spatial_events: list[Event | Violation] = []
    if events_highlight is not None:
        for event in events_highlight:
            if isinstance(event, Integral):
                highlighted_ids.add(int(event))
            elif isinstance(event, (Event, Violation)):
                if event.track_id is None:
                    spatial_events.append(event)
                else:
                    highlighted_ids.add(event.track_id)
            else:
                raise TypeError("highlights must contain track IDs, Events or Violations")

    # Draw highlighted boxes last so their red borders stay visible.
    highlighted_items: list[Track | Detection] = []
    for items, color in ((people, _GREEN), (objects, _ORANGE)):
        for item in items:
            highlighted = (
                isinstance(item, Track) and item.track_id in highlighted_ids
            )
            if highlighted:
                highlighted_items.append(item)
                continue
            _draw_pose(result, item.keypoints, color)
            _draw_box(result, item.bbox, _track_label(item), color)

    for item in highlighted_items:
        _draw_pose(result, item.keypoints, _RED)
        _draw_box(result, item.bbox, _track_label(item), _RED, thickness=3)
    for event in spatial_events:
        _draw_box(result, event.bbox, _ascii(event.type), _RED, thickness=3)
    return result


def draw_event_snapshot(
    frame: np.ndarray, bbox: BBox, label: str
) -> np.ndarray:
    """Return a snapshot with a red event box and an ASCII-safe label."""
    result = _copy_frame(frame)
    _draw_box(result, bbox, label, _RED, thickness=3)
    return result
