"""S6 «Использование телефона»: явный атрибут телефона у человека."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_height,
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)


@register_rule
class PhoneUsageRule(RulePlugin):
    type: ClassVar[str] = "phone_usage"
    title: ClassVar[str] = "Использование телефона"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "pose"})
    default_severity: ClassVar[Severity] = Severity.medium
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Использование телефона",
        "properties": {
            "only_when_moving": {
                "type": "boolean",
                "title": "Только при перемещении",
                "default": False,
            },
            "min_speed_bh": {
                "type": "number",
                "title": "Мин. скорость, высот тела в секунду",
                "minimum": 0.0,
                "default": 0.3,
            },
            "min_box_h_px": {
                "type": "integer",
                "title": "Мин. высота рамки человека, px",
                "minimum": 0,
                "default": 80,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.only_when_moving = bool(cfg.params.get("only_when_moving", False))
        self.min_speed_bh = float(cfg.params.get("min_speed_bh", 0.3))
        self.min_box_h_px = float(cfg.params.get("min_box_h_px", 80))
        # RuleConfig имеет общий default high; учитываем явный выбор пользователя.
        self.severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )
        self._zone_limited = bool(cfg.zones)
        wanted = set(cfg.zones or ())
        self.zones = [zone for zone in camera.zones if zone.zone_id in wanted]

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        w, h = ctx.size
        active = [
            (zone, zone_polygon_px(zone, w, h))
            for zone in self.zones
            if is_zone_active(zone, ctx.ts)
        ]
        if self._zone_limited and not active:
            return []

        out: list[Violation] = []
        for track in ctx.people:
            if track.attributes.get("phone") is not True:
                continue
            box = track.bbox
            if (
                not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
                or bbox_height(box) < self.min_box_h_px
            ):
                continue
            if self.only_when_moving and (
                not math.isfinite(track.speed_bh)
                or track.speed_bh < self.min_speed_bh
            ):
                continue

            point = foot_point(box)
            zone_id = next(
                (
                    zone.zone_id
                    for zone, polygon in active
                    if point_in_polygon(point, polygon)
                ),
                None,
            )
            if self._zone_limited and zone_id is None:
                continue

            out.append(self.violation(
                ctx,
                severity=self.severity,
                track_id=track.track_id,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in box),
                score=float(track.conf),
                details={"phone": True},
            ))
        return out
