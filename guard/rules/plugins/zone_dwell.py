"""Observed continuous dwell in one explicitly selected active zone."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import isfinite
from numbers import Integral, Real
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Track, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import LOCAL_TZ, foot_point, is_zone_active, point_in_zone, validate_active_hours


def _number(value: Any, name: str, *, minimum: float = 0, positive: bool = False,
            maximum: float | None = None, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if result < minimum or (positive and result == minimum) or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    if integer and not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return result


@register_rule
class ZoneDwellRule(RulePlugin):
    type: ClassVar[str] = "zone_dwell"
    title: ClassVar[str] = "Длительное пребывание в зоне"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    zone_schema: ClassVar[dict[str, Any]] = {"type": "array", "minItems": 1, "maxItems": 1}
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object", "title": "Длительное пребывание в зоне", "required": ["dwell_s"],
        "properties": {
            "dwell_s": {"type": "number", "exclusiveMinimum": 0,
                        "title": "Лимит непрерывного наблюдаемого пребывания, с"},
            "max_gap_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 60, "default": 1.0,
                          "title": "Максимальный интервал между наблюдениями, с"},
            "min_conf": {"type": "number", "minimum": 0, "maximum": 1, "default": .35,
                         "title": "Мин. уверенность детекции человека"},
            "min_box_h_px": {"type": "integer", "minimum": 0, "default": 40,
                             "title": "Мин. высота рамки человека, px"},
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        if set(params) - self.params_schema["properties"].keys():
            raise ValueError("Unknown zone_dwell parameter")
        if "dwell_s" not in params:
            raise ValueError("zone_dwell requires params.dwell_s > 0")
        self.dwell_s = _number(params["dwell_s"], "dwell_s", positive=True)
        self.max_gap_s = _number(params.get("max_gap_s", 1.), "max_gap_s", positive=True, maximum=60)
        self.min_conf = _number(params.get("min_conf", .35), "min_conf", maximum=1)
        self.min_box_h_px = _number(params.get("min_box_h_px", 40), "min_box_h_px", integer=True)
        if not cfg.zones or len(cfg.zones) != 1:
            raise ValueError("zone_dwell requires exactly one explicit RuleConfig.zones entry")
        selected = [zone for zone in camera.zones if zone.zone_id == cfg.zones[0]]
        if len(selected) != 1:
            raise ValueError("zone_dwell zone must exist exactly once in the camera")
        self.zone = selected[0].model_copy(deep=True)
        points = self.zone.polygon
        if (len(points) < 3 or len(set(points)) < 3
                or any(not isfinite(value) or not 0 <= value <= 1 for point in points for value in point)
                or abs(sum(x * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * y
                           for i, (x, y) in enumerate(points))) == 0):
            raise ValueError("zone_dwell needs a finite nondegenerate normalized polygon")
        validate_active_hours(self.zone.active_hours)
        self._camera_id = camera.camera_id
        self._inside_since: dict[tuple[int, float], float] = {}
        self._last_ts: float | None = None
        self._last_size: tuple[int, int] | None = None

    def _clear(self) -> None:
        self._inside_since.clear()
        self._last_ts = None
        self._last_size = None

    def _key(self, track: Track, ts: float, width: int, height: int) -> tuple[int, float] | None:
        try:
            if isinstance(track.track_id, bool) or not isinstance(track.track_id, Integral) or track.track_id < 0:
                return None
            conf = _number(track.conf, "track.conf", maximum=1)
            first = _number(track.first_seen, "track.first_seen", minimum=-float("inf"))
            last = _number(track.last_seen, "track.last_seen", minimum=-float("inf"))
            box = tuple(float(value) for value in track.bbox)
            if (track.cls != "person" or conf < self.min_conf or first > ts or last != ts or len(box) != 4
                    or not all(isfinite(value) for value in box)):
                return None
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and y2 - y1 >= self.min_box_h_px):
                return None
            if not point_in_zone(foot_point(box), self.zone, width, height):
                return None
            return int(track.track_id), first
        except (TypeError, ValueError, OverflowError):
            return None

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if ctx.camera.camera_id != self._camera_id:
            return []  # A foreign camera cannot advance or reset this instance.
        status = self.observation_status(ctx)
        ctx.rule_observations[self.cfg.rule_id] = status
        if status != "observed":
            self._clear()
            return []
        try:
            ts = _number(ctx.ts, "capture timestamp", minimum=-float("inf"))
            datetime.fromtimestamp(ts, LOCAL_TZ)  # Validate epoch range, even for all-day zones.
            size = ctx.size
            if min(size) <= 0 or (self._last_ts is not None and ts <= self._last_ts):
                raise ValueError("Non-increasing capture clock or invalid frame size")
        except (TypeError, ValueError, OverflowError, OSError, AttributeError, IndexError):
            ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            self._clear()
            return []
        if (self._last_ts is not None and ts - self._last_ts > self.max_gap_s) or self._last_size != size:
            self._inside_since.clear()
        self._last_ts, self._last_size = ts, size
        if not is_zone_active(self.zone, ts, LOCAL_TZ):
            self._inside_since.clear()
            return []
        counts = Counter(track.track_id for track in ctx.people if isinstance(track.track_id, Integral))
        fresh: dict[tuple[int, float], float] = {}
        out = []
        for track in ctx.people:
            if not isinstance(track.track_id, Integral) or counts[track.track_id] != 1:
                continue
            key = self._key(track, ts, *size)
            if key is None:
                continue
            entered = fresh[key] = self._inside_since.get(key, ts)
            elapsed = ts - entered
            if elapsed >= self.dwell_s:
                fx, fy = foot_point(track.bbox)
                out.append(self.violation(ctx, track_id=int(track.track_id), zone_id=self.zone.zone_id,
                    bbox=tuple(float(value) for value in track.bbox), score=float(track.conf), details={
                        "zone_name": self.zone.name, "zone_kind": self.zone.kind,
                        "dwell_s": elapsed, "dwell_limit_s": self.dwell_s,
                        "observed_inside_since": entered, "max_gap_s": self.max_gap_s,
                        "foot_xy": [float(fx), float(fy)], "evidence": "fresh_observed_person_inside_zone",
                    }))
        # No lifetime history: memory is bounded by this frame's unique eligible people.
        self._inside_since = fresh
        return out
