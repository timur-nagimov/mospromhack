"""S7 «Курение и открытый огонь»: огонь и опционально дым по центру рамки."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_center,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)


@register_rule
class SmokingFireRule(RulePlugin):
    type: ClassVar[str] = "smoking_fire"
    title: ClassVar[str] = "Курение и открытый огонь"
    required_models: ClassVar[frozenset[str]] = frozenset({"fire"})
    default_severity: ClassVar[Severity] = Severity.critical
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Курение и открытый огонь",
        "properties": {
            "include_smoke": {
                "type": "boolean",
                "title": "Учитывать детекции дыма",
                "default": False,
            },
            "min_conf": {
                "type": "number",
                "title": "Мин. уверенность детекции",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
            },
            "min_area_px": {
                "type": "number",
                "title": "Мин. площадь рамки, квадратных пикселей",
                "minimum": 0.0,
                "default": 400,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.include_smoke = bool(cfg.params.get("include_smoke", False))
        self.min_conf = float(cfg.params.get("min_conf", 0.5))
        self.min_area_px = float(cfg.params.get("min_area_px", 400))
        self.severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )
        if cfg.zones:
            wanted = set(cfg.zones)
            self.zones = [
                zone for zone in camera.zones if zone.zone_id in wanted
            ]
        else:
            self.zones = [
                zone for zone in camera.zones if zone.kind == "no_smoking"
            ]
        # Не расширяем область при неизвестных ID или неактивных зонах.
        self._zone_limited = bool(cfg.zones) or bool(self.zones)
        self.smoking_areas = [
            zone for zone in camera.zones if zone.kind == "smoking_area"
        ]

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        w, h = ctx.size
        active = [
            (zone, zone_polygon_px(zone, w, h))
            for zone in self.zones
            if is_zone_active(zone, ctx.ts)
        ]
        if self._zone_limited and not active:
            return []

        # Исключения smoking_area относятся к режиму проверки всего кадра.
        excluded = [
            zone_polygon_px(zone, w, h)
            for zone in self.smoking_areas
            if not self._zone_limited and is_zone_active(zone, ctx.ts)
        ]
        classes = {"fire", "smoke"} if self.include_smoke else {"fire"}
        out: list[Violation] = []
        for detection in ctx.detections.get("fire", []):
            if (
                detection.cls not in classes
                or not math.isfinite(detection.conf)
                or detection.conf < self.min_conf
            ):
                continue

            box = detection.bbox
            if (
                not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if not math.isfinite(area) or area < self.min_area_px:
                continue

            point = bbox_center(box)
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
            if any(point_in_polygon(point, polygon) for polygon in excluded):
                continue

            out.append(self.violation(
                ctx,
                severity=self.severity,
                track_id=None,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in box),
                score=float(detection.conf),
                details={
                    "cls": detection.cls,
                    "conf": float(detection.conf),
                },
            ))
        return out
