"""S5 «Бег»: скорость точки опоры в высотах тела в секунду."""

from __future__ import annotations

from math import isfinite
from typing import Any, ClassVar

from guard.contracts import (
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Violation,
    Zone,
)
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_height,
    foot_point,
    is_zone_active,
    point_in_zone,
)


@register_rule
class RunningRule(RulePlugin):
    type: ClassVar[str] = "running"
    title: ClassVar[str] = "Бег"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    default_severity: ClassVar[Severity] = Severity.medium
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Бег",
        "properties": {
            "min_speed_bh": {
                "type": "number",
                "title": "Мин. скорость, высот тела в секунду",
                "minimum": 0.0,
                "default": 1.8,
            },
            "min_box_h_px": {
                "type": "integer",
                "title": "Мин. высота рамки человека, px",
                "minimum": 0,
                "default": 50,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.min_speed_bh = float(cfg.params.get("min_speed_bh", 1.8))
        self.min_box_h_px = float(cfg.params.get("min_box_h_px", 50))
        self._severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )

        # По контракту None означает весь кадр; [] не выбирает ни одной зоны.
        self.zones: list[Zone] | None = None
        if cfg.zones is not None:
            wanted = set(cfg.zones)
            self.zones = [
                zone for zone in camera.zones if zone.zone_id in wanted
            ]

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts):
            return []

        active_zones = (
            None
            if self.zones is None
            else [
                zone for zone in self.zones if is_zone_active(zone, ctx.ts)
            ]
        )
        if active_zones == []:
            return []

        width, height = ctx.size
        out: list[Violation] = []
        for track in ctx.people:
            age = ctx.ts - track.first_seen
            if not isfinite(age) or age < 1.0:
                continue
            if (
                not isfinite(track.speed_bh)
                or track.speed_bh < self.min_speed_bh
            ):
                continue
            if not all(isfinite(value) for value in track.bbox):
                continue
            box_height = bbox_height(track.bbox)
            if (
                track.bbox[2] <= track.bbox[0]
                or box_height <= 0
                or box_height < self.min_box_h_px
            ):
                continue

            zone_id = None
            if active_zones is not None:
                point = foot_point(track.bbox)
                zone_id = next(
                    (
                        zone.zone_id
                        for zone in active_zones
                        if point_in_zone(point, zone, width, height)
                    ),
                    None,
                )
                if zone_id is None:
                    continue

            out.append(
                self.violation(
                    ctx,
                    severity=self._severity,
                    track_id=track.track_id,
                    zone_id=zone_id,
                    bbox=tuple(float(value) for value in track.bbox),
                    score=float(track.conf),
                    details={"speed_bh": float(track.speed_bh)},
                )
            )
        return out
