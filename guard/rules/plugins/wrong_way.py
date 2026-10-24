"""Observed motion against a configured image-plane direction in one zone."""

from __future__ import annotations

from collections import deque
from math import acos, cos, degrees, hypot, isfinite, radians
from numbers import Real
from typing import Any, ClassVar, NamedTuple

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Track, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import foot_point, is_zone_active, point_in_zone, validate_active_hours


class _Point(NamedTuple):
    ts: float
    x: float
    y: float
    height: float


@register_rule
class WrongWayRule(RulePlugin):
    type: ClassVar[str] = "wrong_way"
    title: ClassVar[str] = "Движение против разрешённого направления"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    zone_schema: ClassVar[dict[str, Any]] = {"type": "array", "minItems": 1, "maxItems": 1}
    MAX_TRACKS = 256
    MAX_POINTS = 512
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["direction"],
        "additionalProperties": False,
        "properties": {
            "direction": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "title": "Разрешённое направление: две разные точки [x, y] в 0..1",
                "items": {"type": "array", "minItems": 2, "maxItems": 2,
                          "items": {"type": "number", "minimum": 0, "maximum": 1}},
            },
            "window_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": 2},
            "min_observed_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": .8},
            "max_gap_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": .5},
            "min_conf": {"type": "number", "minimum": 0, "maximum": 1, "default": .35},
            "min_box_h_px": {"type": "number", "exclusiveMinimum": 0, "default": 40},
            "min_displacement_bh": {"type": "number", "exclusiveMinimum": 0, "default": .25},
            "min_path_bh": {"type": "number", "exclusiveMinimum": 0, "default": .3},
            "max_angle_deg": {"type": "number", "minimum": 0, "exclusiveMaximum": 90, "default": 30},
            "min_progress_ratio": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": .8},
        },
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        if set(cfg.params) - self.params_schema["properties"].keys():
            raise ValueError("wrong_way: unknown params")
        if cfg.zones is None or len(cfg.zones) != 1:
            raise ValueError("wrong_way requires exactly one explicitly selected zone")
        zones = [zone for zone in camera.zones if zone.zone_id == cfg.zones[0]]
        if len(zones) != 1:
            raise ValueError("wrong_way selected zone must exist exactly once")
        self.zone = zones[0].model_copy(deep=True)
        polygon = self.zone.polygon
        if len(polygon) < 3 or any(not isfinite(x) or not 0 <= x <= 1 for point in polygon for x in point):
            raise ValueError("wrong_way zone must be a finite normalized polygon")
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))
        if abs(area) <= 1e-12:
            raise ValueError("wrong_way zone must have nonzero area")
        validate_active_hours(self.zone.active_hours)
        direction = cfg.params.get("direction")
        if not isinstance(direction, (list, tuple)) or len(direction) != 2:
            raise ValueError("wrong_way requires two normalized direction points")
        for point in direction:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) or not 0 <= x <= 1
                for x in point
            ):
                raise ValueError("wrong_way direction points must be finite coordinates in [0, 1]")
        self.direction = tuple(tuple(float(x) for x in point) for point in direction)
        if self.direction[0] == self.direction[1]:
            raise ValueError("wrong_way direction points must differ")
        for name, schema in self.params_schema["properties"].items():
            if name == "direction":
                continue
            value = cfg.params.get(name, schema["default"])
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"wrong_way {name} must be a finite number")
            if ("minimum" in schema and value < schema["minimum"] or
                "maximum" in schema and value > schema["maximum"] or
                "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"] or
                "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]):
                raise ValueError(f"wrong_way {name} is outside the permitted range")
            setattr(self, name, float(value))
        if self.min_observed_s > self.window_s or self.max_gap_s > self.window_s:
            raise ValueError("wrong_way min_observed_s and max_gap_s must not exceed window_s")
        self._cos_angle = cos(radians(self.max_angle_deg))
        self._history: dict[tuple[int, float], deque[_Point]] = {}
        self._last_ts: float | None = None
        self._size: tuple[int, int] | None = None

    def _fresh_point(self, track: Track, ctx: FrameContext) -> tuple[_Point | None, bool]:
        if (track.cls != "person" or len(track.bbox) != 4 or
            not all(isfinite(x) for x in (*track.bbox, track.conf, track.first_seen, track.last_seen)) or
            not self.min_conf <= track.conf <= 1 or track.first_seen > ctx.ts or track.last_seen != ctx.ts):
            return None, False
        x1, y1, x2, y2 = track.bbox
        height = y2 - y1
        if x2 <= x1 or height < self.min_box_h_px:
            return None, False
        x, y = foot_point(track.bbox)
        if not point_in_zone((x, y), self.zone, *ctx.size):
            return None, True
        return _Point(float(ctx.ts), x, y, height), True

    def _advance(self, key: tuple[int, float], point: _Point, axis: tuple[float, float]) -> dict | None:
        history = self._history.setdefault(key, deque(maxlen=self.MAX_POINTS))
        if history:
            previous = history[-1]
            dx, dy = point.x - previous.x, point.y - previous.y
            step = hypot(dx, dy)
            against = -(dx * axis[0] + dy * axis[1])
            # A past window alone cannot emit once current motion stops or turns.
            # Even a small contradictory step restarts this conservative proof.
            if (point.ts - previous.ts > self.max_gap_s or step <= 1e-9 or
                against <= 0 or against / step < self._cos_angle - 1e-12):
                history.clear()
        history.append(point)
        while history and point.ts - history[0].ts > self.window_s + 1e-9:
            history.popleft()
        if len(history) < 3 or point.ts - history[0].ts + 1e-9 < self.min_observed_s:
            return None
        first = history[0]
        dx, dy = point.x - first.x, point.y - first.y
        displacement = hypot(dx, dy)
        projection = -(dx * axis[0] + dy * axis[1])
        path = sum(hypot(b.x - a.x, b.y - a.y) for a, b in zip(history, list(history)[1:]))
        mean_height = sum(p.height for p in history) / len(history)
        if displacement <= 0 or path <= 0:
            return None
        progress = projection / path
        if (displacement / mean_height < self.min_displacement_bh or
            path / mean_height < self.min_path_bh or progress < self.min_progress_ratio or
            projection / displacement < self._cos_angle - 1e-12):
            return None
        return {
            "direction": [list(p) for p in self.direction],
            "allowed_unit_vector_px": list(axis),
            "frame_size_px": list(self._size),
            "positions": [{"ts": p.ts, "foot_px": [p.x, p.y], "box_height_px": p.height} for p in history],
            "observed_s": point.ts - first.ts,
            "mean_box_height_px": mean_height,
            "net_displacement_bh": displacement / mean_height,
            "net_opposite_projection_bh": projection / mean_height,
            "path_bh": path / mean_height,
            "progress_ratio": progress,
            "opposite_angle_deg": degrees(acos(max(-1., min(1., projection / displacement)))),
            "distance_unit": "mean_bbox_height",
            "clock": "capture_timestamp_seconds",
            "thresholds": {name: getattr(self, name) for name in self.params_schema["properties"] if name != "direction"},
        }

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        status = self.observation_status(ctx)
        ctx.rule_observations[self.cfg.rule_id] = status
        if status != "observed":
            self._history.clear()
            self._last_ts = None
            return []
        if (ctx.camera.camera_id != self.camera.camera_id or not isfinite(ctx.ts) or
            ctx.frame.ndim < 2 or min(ctx.size) <= 0):
            self._history.clear()
            self._last_ts = None
            ctx.rule_observations[self.cfg.rule_id] = "error"
            return []
        if self._last_ts is not None and ctx.ts <= self._last_ts:
            self._history.clear()
            self._last_ts = ctx.ts
            ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            return []
        if ctx.size != self._size:
            self._history.clear()
            self._size = ctx.size
        self._last_ts = ctx.ts
        try:
            active = is_zone_active(self.zone, ctx.ts)
        except (OverflowError, OSError, ValueError):
            active = False
        if not active or len(ctx.people) > self.MAX_TRACKS:
            self._history.clear()
            ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            return []
        # Duplicate IDs are not two independent observations of the same actor.
        if len({t.track_id for t in ctx.people}) != len(ctx.people):
            self._history.clear()
            ctx.rule_observations[self.cfg.rule_id] = "error"
            return []
        width, height = ctx.size
        start, end = self.direction
        vx, vy = (end[0] - start[0]) * width, (end[1] - start[1]) * height
        length = hypot(vx, vy)
        axis = vx / length, vy / length
        fresh = set()
        out = []
        for track in ctx.people:
            point, observable = self._fresh_point(track, ctx)
            if not observable:
                ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            if point is None:
                continue
            key = (track.track_id, float(track.first_seen))
            fresh.add(key)
            details = self._advance(key, point, axis)
            if details is not None:
                out.append(self.violation(ctx, track_id=track.track_id, zone_id=self.zone.zone_id,
                    bbox=tuple(float(v) for v in track.bbox), score=float(track.conf), details=details))
        self._history = {key: history for key, history in self._history.items() if key in fresh}
        return out
