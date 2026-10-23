"""S2 «Нет СИЗ»: только явно отсутствующие средства защиты."""

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
class PpeMissingRule(RulePlugin):
    type: ClassVar[str] = "ppe_missing"
    title: ClassVar[str] = "Нет обязательных СИЗ"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "ppe"})
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Нет обязательных СИЗ",
        "properties": {
            "required": {
                "type": "array",
                "title": "Обязательные средства индивидуальной защиты",
                "items": {
                    "type": "string",
                    "enum": ["helmet", "vest", "respirator"],
                },
                "uniqueItems": True,
                "default": ["helmet", "vest"],
            },
            "min_box_h_px": {
                "type": "integer",
                "title": "Мин. высота рамки человека, px",
                "minimum": 0,
                "default": 60,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.required = tuple(dict.fromkeys(
            cfg.params.get("required", ["helmet", "vest"])
        ))
        self.min_box_h_px = float(cfg.params.get("min_box_h_px", 60))
        if cfg.zones:
            wanted = set(cfg.zones)
            self.zones = [z for z in camera.zones if z.zone_id in wanted]
        else:
            self.zones = [z for z in camera.zones if z.kind == "ppe_required"]
        # Не расширяем область проверки при неизвестных ID или неактивных зонах.
        self._zone_limited = bool(cfg.zones) or bool(self.zones)

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
            box = track.bbox
            if (
                not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
                or bbox_height(box) < self.min_box_h_px
            ):
                continue

            missing = [
                name for name in self.required
                if track.attributes.get(name) is False
            ]
            if not missing:
                continue

            point = foot_point(box)
            zone_id = next(
                (
                    zone.zone_id for zone, polygon in active
                    if point_in_polygon(point, polygon)
                ),
                None,
            )
            if self._zone_limited and zone_id is None:
                continue

            out.append(self.violation(
                ctx,
                track_id=track.track_id,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in box),
                score=float(track.conf),
                details={"missing": missing},
            ))
        return out
