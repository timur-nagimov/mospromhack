"""Геометрия правил: зоны (нормированные полигоны), точка опоры, активность зоны по времени.

Соглашения (docs/CONTRACTS.md):
- полигоны зон заданы в координатах 0..1 относительно кадра;
- «человек в зоне» проверяется по точке опоры — низ-центр рамки;
- active_hours: ["07:55-12:00", "22:00-06:00"] по местному времени (Europe/Moscow);
  None или пустой список — зона активна всегда.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from guard.contracts import BBox, Zone

log = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Europe/Moscow")
_RANGE_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")

Point = tuple[float, float]


# ---------- рамки ----------


def foot_point(bbox: BBox) -> Point:
    """Точка опоры: низ-центр рамки."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------- полигоны ----------


def to_pixels(polygon: Iterable[Sequence[float]], width: int, height: int) -> list[Point]:
    """Нормированный полигон → пиксели кадра width×height."""
    return [(float(x) * width, float(y) * height) for x, y in polygon]


def to_normalized(polygon: Iterable[Sequence[float]], width: int, height: int) -> list[Point]:
    return [(float(x) / width, float(y) / height) for x, y in polygon]


def _on_segment(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    (px, py), (ax, ay), (bx, by) = p, a, b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, abs(bx - ax) + abs(by - ay)):
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Точка внутри полигона или на его границе (ray casting). Вырожденный полигон — False."""
    n = len(polygon)
    if n < 3:
        return False
    px, py = float(point[0]), float(point[1])
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if _on_segment((px, py), (xi, yi), (xj, yj)):
            return True
        if (yi > py) != (yj > py):
            x_cross = xi + (py - yi) * (xj - xi) / (yj - yi)
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def zone_polygon_px(zone: Zone, width: int, height: int) -> list[Point]:
    return to_pixels(zone.polygon, width, height)


def point_in_zone(point: Sequence[float], zone: Zone, width: int, height: int) -> bool:
    return point_in_polygon(point, zone_polygon_px(zone, width, height))


# ---------- время активности ----------


@lru_cache(maxsize=256)
def parse_time_range(text: str) -> tuple[int, int]:
    """'07:55-12:00' → (475, 720) минут от полуночи. Допускается 24:00 как конец суток."""
    m = _RANGE_RE.match(text)
    if not m:
        raise ValueError(f"bad active_hours range {text!r}, expected 'HH:MM-HH:MM'")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    for h, mi in ((h1, m1), (h2, m2)):
        if not (0 <= h <= 24 and 0 <= mi < 60) or (h == 24 and mi != 0):
            raise ValueError(f"bad time in active_hours range {text!r}")
    return h1 * 60 + m1, h2 * 60 + m2


def validate_active_hours(ranges: Iterable[str] | None) -> None:
    for r in ranges or ():
        parse_time_range(r)


def _minute_of_day(ts: float | datetime | None, tz: ZoneInfo) -> float:
    if ts is None:
        dt = datetime.now(tz)
    elif isinstance(ts, datetime):
        dt = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=tz)
    else:
        dt = datetime.fromtimestamp(float(ts), tz)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def in_time_ranges(ranges: Iterable[str] | None, ts: float | datetime | None = None, tz: ZoneInfo = LOCAL_TZ) -> bool:
    """True, если момент ts попадает в любой из диапазонов (диапазоны через полночь поддерживаются)."""
    ranges = list(ranges or [])
    if not ranges:
        return True
    minute = _minute_of_day(ts, tz)
    for r in ranges:
        try:
            start, end = parse_time_range(r)
        except ValueError:
            log.warning("ignoring malformed active_hours %r: zone treated as active", r)
            return True  # безопасность важнее: битое расписание не должно выключать зону
        if start == end:
            return True  # «круглые сутки»
        if start < end:
            if start <= minute < end:
                return True
        elif minute >= start or minute < end:  # через полночь: 22:00-06:00
            return True
    return False


def is_zone_active(zone: Zone, ts: float | datetime | None = None, tz: ZoneInfo = LOCAL_TZ) -> bool:
    return in_time_ranges(zone.active_hours, ts, tz)
