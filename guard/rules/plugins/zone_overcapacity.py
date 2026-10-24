"""Count fresh, distinct observed people in one explicitly configured zone."""

from __future__ import annotations

from math import isfinite
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import bbox_iou, foot_point, is_zone_active, point_in_polygon, zone_polygon_px


@register_rule
class ZoneOvercapacityRule(RulePlugin):
    type: ClassVar[str] = "zone_overcapacity"
    title: ClassVar[str] = "Превышение числа людей в зоне"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    zone_schema: ClassVar[dict[str, Any]] = {"type": "array", "minItems": 1, "maxItems": 1}
    default_severity: ClassVar[Severity] = Severity.medium
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object", "title": title,
        "properties": {
            "capacity": {"type": "integer", "minimum": 1, "title": "Допустимое число людей"},
            "min_conf": {"type": "number", "minimum": 0, "maximum": 1, "default": .35},
            "min_box_h_px": {"type": "number", "minimum": 0, "default": 40},
            "min_track_hits": {"type": "integer", "minimum": 1, "default": 3},
        },
        "required": ["capacity"], "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        if cfg.zones is None or len(cfg.zones) != 1:
            raise ValueError("zone_overcapacity requires exactly one explicit zone")
        found = [zone for zone in camera.zones if zone.zone_id == cfg.zones[0]]
        if len(found) != 1:
            raise ValueError("zone_overcapacity requires one existing unique zone")
        self.zone = found[0]
        polygon = self.zone.polygon
        if (len(polygon) < 3 or any(not isfinite(v) or not 0 <= v <= 1 for point in polygon for v in point)
                or abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) <= 1e-12):
            raise ValueError("zone_overcapacity requires a finite nondegenerate normalized polygon")
        p = cfg.params
        if set(p) - self.params_schema["properties"].keys():
            raise ValueError("Unknown zone_overcapacity parameter")
        self.capacity = p.get("capacity")
        self.min_hits = p.get("min_track_hits", 3)
        for name, value in (("capacity", self.capacity), ("min_track_hits", self.min_hits)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, default in (("min_conf", .35), ("min_box_h_px", 40)):
            value = p.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, float(value))
        if self.min_conf > 1:
            raise ValueError("min_conf must not exceed 1")
        self._last_ts: float | None = None

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if ctx.camera.camera_id != self.camera.camera_id:
            return []
        status = self.observation_status(ctx)
        ctx.rule_observations[self.cfg.rule_id] = status
        if not isfinite(ctx.ts) or (self._last_ts is not None and ctx.ts <= self._last_ts):
            if status == "observed":
                ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            return []
        self._last_ts = ctx.ts
        w, h = ctx.size
        if status != "observed" or w <= 0 or h <= 0 or not is_zone_active(self.zone, ctx.ts):
            if status == "observed":
                ctx.rule_observations[self.cfg.rule_id] = "not_observable"
            return []
        polygon = zone_polygon_px(self.zone, w, h)
        eligible = {}
        for track in ctx.people:
            if track.cls != "person" or track.last_seen != ctx.ts or track.hits < self.min_hits:
                continue
            try:
                x1, y1, x2, y2 = track.bbox
                if (not all(isfinite(v) for v in (*track.bbox, track.conf)) or track.conf < self.min_conf
                        or x2 <= x1 or y2 <= y1 or y2 - y1 < self.min_box_h_px):
                    continue
            except (TypeError, ValueError):
                continue
            if point_in_polygon(foot_point(track.bbox), polygon):
                previous = eligible.get(track.track_id)
                if previous is None or track.conf > previous.conf:
                    eligible[track.track_id] = track
        # Suppress near-identical boxes conservatively. This is still an estimate:
        # distinct IDs alone do not prove distinct physical people.
        people, suppressed = [], []
        for track in sorted(eligible.values(), key=lambda item: (-item.conf, item.track_id)):
            if any(bbox_iou(track.bbox, kept.bbox) >= .98 for kept in people):
                suppressed.append(track.track_id)
            else:
                people.append(track)
        if len(people) <= self.capacity:
            return []
        xs, ys = zip(*polygon)
        return [self.violation(ctx, track_id=None, zone_id=self.zone.zone_id,
            bbox=(min(xs), min(ys), max(xs), max(ys)), score=min(track.conf for track in people),
            details={"zone_name": self.zone.name, "observed_count": len(people), "capacity": self.capacity,
                     "participant_track_ids": sorted(track.track_id for track in people),
                     "suppressed_near_duplicate_track_ids": sorted(suppressed),
                     "count_basis": "fresh observed person tracks; near-identical IoU>=0.98 boxes counted once",
                     "coverage": "observed track estimate; people may be missed or duplicated; below-limit count does not prove safety"})]
