"""S1 «Вход в опасную зону»: точка опоры человека внутри активной зоны."""

from __future__ import annotations

from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation, Zone
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import bbox_height, foot_point, is_zone_active, point_in_polygon, zone_polygon_px


@register_rule
class ZoneIntrusionRule(RulePlugin):
    type: ClassVar[str] = "zone_intrusion"
    title: ClassVar[str] = "Вход в опасную зону"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Вход в опасную зону",
        "properties": {
            "min_conf": {
                "type": "number",
                "title": "Мин. уверенность детекции человека",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.35,
            },
            "min_box_h_px": {
                "type": "integer",
                "title": "Мин. высота рамки человека, px",
                "minimum": 0,
                "default": 24,
            },
            "zone_kinds": {
                "type": "array",
                "title": "Типы зон, если зоны правила не заданы",
                "items": {
                    "type": "string",
                    "enum": ["danger", "ppe_required", "no_smoking", "smoking_area", "vehicle_path", "height", "fence"],
                },
                "default": ["danger"],
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        p = cfg.params
        self.min_conf = float(p.get("min_conf", 0.35))
        self.min_box_h_px = float(p.get("min_box_h_px", 24))
        kinds = set(p.get("zone_kinds") or ["danger"])
        if cfg.zones:
            wanted = set(cfg.zones)
            self.zones: list[Zone] = [z for z in camera.zones if z.zone_id in wanted]
        else:
            self.zones = [z for z in camera.zones if z.kind in kinds]
        self._px_cache: dict[tuple[str, int, int], list[tuple[float, float]]] = {}

    def _polygon(self, zone: Zone, w: int, h: int) -> list[tuple[float, float]]:
        key = (zone.zone_id, w, h)
        poly = self._px_cache.get(key)
        if poly is None:
            poly = self._px_cache[key] = zone_polygon_px(zone, w, h)
        return poly

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        active = [z for z in self.zones if is_zone_active(z, ctx.ts)]
        if not active or not ctx.people:
            return []
        w, h = ctx.size
        out: list[Violation] = []
        for track in ctx.people:
            if track.conf < self.min_conf or bbox_height(track.bbox) < self.min_box_h_px:
                continue
            fx, fy = foot_point(track.bbox)
            for zone in active:
                if point_in_polygon((fx, fy), self._polygon(zone, w, h)):
                    out.append(
                        self.violation(
                            ctx,
                            track_id=track.track_id,
                            zone_id=zone.zone_id,
                            bbox=tuple(float(v) for v in track.bbox),
                            score=float(track.conf),
                            details={
                                "zone_name": zone.name,
                                "zone_kind": zone.kind,
                                "foot_xy": [round(fx, 1), round(fy, 1)],
                            },
                        )
                    )
                    break  # одно нарушение на человека: ключ события (камера, правило, трек)
        return out
