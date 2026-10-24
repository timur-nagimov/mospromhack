"""S9: отсутствие подтверждённой привязи в зоне высоты.

По умолчанию правило срабатывает только при attributes["harness"] is False.
Отсутствующий или неизвестный атрибут не является доказательством нарушения.

Без модели привязи mode="strict" работает как «контроль доступа на высоту»:
любой человек без явно подтверждённой привязи вызывает нарушение high.
Это не распознавание страховочной системы и не проверка её крепления.
Зоны выбираются явно через cfg.zones либо по kind="height".
Подтверждение событий и дедупликацию выполняет EventManager.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)


@register_rule
class HeightNoHarnessRule(RulePlugin):
    type: ClassVar[str] = "height_no_harness"
    title: ClassVar[str] = "Работа на высоте без привязи"
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Работа на высоте без привязи",
        "description": (
            "Без модели привязи strict — контроль доступа на высоту, "
            "а не распознавание отсутствия или правильности крепления привязи."
        ),
        "properties": {
            "mode": {
                "type": "string",
                "title": "Поведение при неизвестном состоянии привязи",
                "enum": ["off", "strict"],
                "default": "off",
                "description": (
                    "off: срабатывать только при harness is False. "
                    "strict: любой человек в выбранной активной зоне без "
                    "harness is True считается нарушителем high. Без модели "
                    "привязи это контроль доступа на высоту."
                ),
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        self.mode = cfg.params.get("mode", "off")
        if self.mode not in ("off", "strict"):
            raise ValueError("mode must be 'off' or 'strict'")
        if cfg.zones is not None:
            wanted = set(cfg.zones)
            self.zones = [zone for zone in camera.zones if zone.zone_id in wanted]
        else:
            self.zones = [zone for zone in camera.zones if zone.kind == "height"]

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts):
            return []

        width, height = ctx.size
        active = []
        for zone in self.zones:
            if not is_zone_active(zone, ctx.ts):
                continue
            polygon = zone_polygon_px(zone, width, height)
            if len(polygon) >= 3 and all(
                isfinite(x) and isfinite(y) for x, y in polygon
            ):
                active.append((zone, polygon))

        out: list[Violation] = []
        for track in ctx.people:
            box = track.bbox
            if (
                track.cls != "person"
                or not all(isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
                or not isfinite(track.conf)
            ):
                continue

            harness = track.attributes.get("harness")
            if harness is True:
                continue
            if harness is not False and self.mode == "off":
                continue

            foot = foot_point(box)
            for zone, polygon in active:
                if not point_in_polygon(foot, polygon):
                    continue
                explicit_missing = harness is False
                details: dict[str, Any] = {
                    "condition": (
                        "harness_missing" if explicit_missing else "harness_unconfirmed"
                    ),
                    "evidence": "attribute" if explicit_missing else "access_control",
                    "mode": self.mode,
                    "zone_name": zone.name,
                    "zone_kind": zone.kind,
                }
                if explicit_missing:
                    details["missing"] = ["harness"]
                out.append(self.violation(
                    ctx,
                    severity=Severity.high,
                    track_id=track.track_id,
                    zone_id=zone.zone_id,
                    bbox=tuple(float(value) for value in box),
                    score=float(track.conf),
                    details=details,
                ))
                break
        return out
