"""S4 «Опасное сближение с движущейся техникой»."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from guard.contracts import (
    BBox,
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Violation,
)
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_height,
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)


def _valid_bbox(box: BBox) -> bool:
    return (
        all(math.isfinite(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _distance_to_bbox(point: tuple[float, float], box: BBox) -> float:
    """Евклидово расстояние до прямоугольника; внутри него — ноль."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


@register_rule
class VehicleProximityRule(RulePlugin):
    type: ClassVar[str] = "vehicle_proximity"
    title: ClassVar[str] = "Опасное сближение с техникой"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    default_severity: ClassVar[Severity] = Severity.critical
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Опасное сближение с техникой",
        "properties": {
            "vehicle_classes": {
                "type": "array",
                "title": "Классы техники",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
                "default": ["car", "truck", "bus", "forklift", "motorcycle"],
            },
            "min_vehicle_speed_bh": {
                "type": "number",
                "title": "Мин. скорость техники, высот рамки в секунду",
                "minimum": 0,
                "default": 0.15,
            },
            "distance_factor": {
                "type": "number",
                "title": "Опасное расстояние, долей высоты человека",
                "minimum": 0,
                "default": 0.5,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.vehicle_classes = frozenset(cfg.params.get(
            "vehicle_classes", ["car", "truck", "bus", "forklift", "motorcycle"]
        ))
        self.min_vehicle_speed_bh = float(
            cfg.params.get("min_vehicle_speed_bh", 0.15)
        )
        self.distance_factor = float(cfg.params.get("distance_factor", 0.5))
        self._zone_limited = bool(cfg.zones)
        wanted = set(cfg.zones or ())
        self.zones = [
            z for z in camera.zones
            if z.zone_id in wanted and z.kind == "vehicle_path"
        ]
        # RuleConfig имеет общий default high; явный выбор пользователя сохраняем.
        self.severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        w, h = ctx.size
        active = [
            (zone, zone_polygon_px(zone, w, h))
            for zone in self.zones
            if is_zone_active(zone, ctx.ts)
        ]
        if self._zone_limited and not active:
            return []

        # speed_bh уже рассчитан трекером по движению точки опоры.
        vehicles = [
            track for track in ctx.objects
            if track.cls in self.vehicle_classes
            and _valid_bbox(track.bbox)
            and math.isfinite(track.speed_bh)
            and track.speed_bh > 0.0
            and track.speed_bh >= self.min_vehicle_speed_bh
        ]
        if not vehicles:
            return []

        out: list[Violation] = []
        for person in ctx.people:
            if not _valid_bbox(person.bbox):
                continue
            point = foot_point(person.bbox)
            zone_id = next(
                (
                    zone.zone_id for zone, polygon in active
                    if point_in_polygon(point, polygon)
                ),
                None,
            )
            if self._zone_limited and zone_id is None:
                continue

            limit = self.distance_factor * bbox_height(person.bbox)
            candidates = []
            for vehicle in vehicles:
                distance = _distance_to_bbox(point, vehicle.bbox)
                if distance < limit:
                    candidates.append((distance, vehicle))
            if not candidates:
                continue

            # Одно нарушение на человека; ближайшая техника объясняет причину.
            distance, vehicle = min(
                candidates, key=lambda item: (item[0], item[1].track_id)
            )
            out.append(self.violation(
                ctx,
                severity=self.severity,
                track_id=person.track_id,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in person.bbox),
                score=float(min(person.conf, vehicle.conf)),
                details={
                    "vehicle_track_id": vehicle.track_id,
                    "distance_px": float(distance),
                },
            ))
        return out
