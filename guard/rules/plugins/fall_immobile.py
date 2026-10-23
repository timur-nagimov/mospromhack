"""S3 «Падение и неподвижность».

Длительность известной позы лежания берётся только из lying_since.
При unknown запасной признак описывает текущую рамку; длительность
лежания неизвестна (lying_s=None), подтверждение ведёт EventManager.
Плагин не изменяет треки и не хранит состояние подтверждения.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any, ClassVar

from guard.contracts import (
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Track,
    Violation,
)
from guard.rules.base import RulePlugin, register_rule


@register_rule
class FallImmobileRule(RulePlugin):
    type: ClassVar[str] = "fall_immobile"
    title: ClassVar[str] = "Падение и неподвижность"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "pose"})
    default_severity: ClassVar[Severity] = Severity.critical
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Падение и неподвижность",
        "properties": {
            "min_lying_s": {
                "type": "number",
                "title": "Мин. длительность позы лёжа, с",
                "minimum": 0.0,
                "default": 2.0,
            },
            "immobile_window_s": {
                "type": "number",
                "title": "Окно проверки неподвижности, с",
                "exclusiveMinimum": 0.0,
                "default": 3.0,
            },
            "max_shift_bh": {
                "type": "number",
                "title": "Макс. смещение, высот тела",
                "exclusiveMinimum": 0.0,
                "default": 0.3,
            },
            "allow_bbox_fallback": {
                "type": "boolean",
                "title": "Использовать форму рамки при неизвестной позе",
                "default": True,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        self.min_lying_s = float(params.get("min_lying_s", 2.0))
        self.immobile_window_s = float(params.get("immobile_window_s", 3.0))
        self.max_shift_bh = float(params.get("max_shift_bh", 0.3))
        self.allow_bbox_fallback = bool(params.get("allow_bbox_fallback", True))
        # RuleConfig имеет общий default high; явно заданный high сохраняем.
        self._severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )

    def _shift_bh(self, track: Track, ts: float) -> float | None:
        """Диаметр траектории в окне, делённый на среднюю высоту рамки.

        Проверяем все пары, чтобы движение с возвратом не считалось
        неподвижностью. Нужны хотя бы две разновременные точки.
        Используем доступную часть окна, без дополнительного таймера.
        """
        samples = [
            sample
            for sample in track.history
            if ts - self.immobile_window_s <= sample[0] <= ts
        ]
        if len(samples) < 2:
            return None
        if any(
            not all(isfinite(value) for value in sample) or sample[3] <= 0
            for sample in samples
        ):
            return None
        if max(sample[0] for sample in samples) <= min(
            sample[0] for sample in samples
        ):
            return None

        mean_height = sum(sample[3] for sample in samples) / len(samples)
        displacement = max(
            hypot(right[1] - left[1], right[2] - left[2])
            for index, left in enumerate(samples)
            for right in samples[index + 1 :]
        )
        shift = displacement / mean_height
        return float(shift) if isfinite(shift) else None

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts):
            return []

        out: list[Violation] = []
        for track in ctx.people:
            if not all(isfinite(value) for value in track.bbox):
                continue
            x1, y1, x2, y2 = track.bbox
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue

            pose_state = track.attributes.get("pose_state", "unknown")
            lying_s: float | None = None
            if pose_state == "lying":
                since = track.attributes.get("lying_since")
                if since is None:
                    continue
                lying_s = float(ctx.ts - since)
                if (
                    not isfinite(lying_s)
                    or lying_s < 0
                    or lying_s < self.min_lying_s
                ):
                    continue
            elif (
                pose_state == "unknown"
                and self.allow_bbox_fallback
                and width / height >= 1.3
            ):
                # Возраст трека не доказывает длительность лежания.
                # В history нет ширины рамки или прошлых состояний позы.
                lying_s = None
            else:
                continue

            shift_bh = self._shift_bh(track, ctx.ts)
            if shift_bh is None or shift_bh >= self.max_shift_bh:
                continue

            out.append(
                self.violation(
                    ctx,
                    severity=self._severity,
                    track_id=track.track_id,
                    zone_id=None,
                    bbox=tuple(float(value) for value in track.bbox),
                    score=float(track.conf),
                    details={"lying_s": lying_s, "shift_bh": shift_bh},
                )
            )
        return out
