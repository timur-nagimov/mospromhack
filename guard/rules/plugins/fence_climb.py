"""S8: перелезание через ограждение и пролезание под препятствием.

Медианная высота берётся из доступной истории трека. Запасной признак
при keypoints=None использует непрерывное пребывание в зоне по history.
Подтверждение и дедупликацию выполняет EventManager; треки не изменяются.
"""

from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any, ClassVar

import numpy as np

from guard.contracts import (
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Track,
    Violation,
    Zone,
)
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_height,
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)


@register_rule
class FenceClimbRule(RulePlugin):
    type: ClassVar[str] = "fence_climb"
    title: ClassVar[str] = "Перелезание через ограждение или пролезание под препятствием"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "pose"})
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Перелезание и пролезание",
        "properties": {
            "kp_conf": {
                "type": "number",
                "title": "Мин. уверенность ключевой точки",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.4,
            },
            "lift_bh": {
                "type": "number",
                "title": "Мин. подъём над низом зоны, высот тела",
                "minimum": 0.0,
                "default": 0.15,
            },
            "crouch_ratio": {
                "type": "number",
                "title": "Порог отношения высоты рамки к медианной высоте",
                "exclusiveMinimum": 0.0,
                "maximum": 1.0,
                "default": 0.6,
            },
            "max_dwell_s": {
                "type": "number",
                "title": "Допустимое время в зоне без данных позы, с",
                "minimum": 0.0,
                "default": 2.0,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        self.kp_conf = float(params.get("kp_conf", 0.4))
        self.lift_bh = float(params.get("lift_bh", 0.15))
        self.crouch_ratio = float(params.get("crouch_ratio", 0.6))
        self.max_dwell_s = float(params.get("max_dwell_s", 2.0))

        if not isfinite(self.kp_conf) or not 0 <= self.kp_conf <= 1:
            raise ValueError("kp_conf must be finite and in [0, 1]")
        if not isfinite(self.lift_bh) or self.lift_bh < 0:
            raise ValueError("lift_bh must be finite and nonnegative")
        if not isfinite(self.crouch_ratio) or not 0 < self.crouch_ratio <= 1:
            raise ValueError("crouch_ratio must be finite and in (0, 1]")
        if not isfinite(self.max_dwell_s) or self.max_dwell_s < 0:
            raise ValueError("max_dwell_s must be finite and nonnegative")

        if cfg.zones:
            wanted = set(cfg.zones)
            self.zones: list[Zone] = [
                zone for zone in camera.zones if zone.zone_id in wanted
            ]
        else:
            self.zones = [
                zone for zone in camera.zones if zone.kind == "fence"
            ]

    def _keypoints(
        self, track: Track
    ) -> dict[int, tuple[float, float]] | None:
        """None означает отсутствие pose; пустой словарь — ненадёжный pose."""
        if track.keypoints is None:
            return None
        try:
            points = np.asarray(track.keypoints, dtype=float)
        except (TypeError, ValueError):
            return {}
        if points.shape != (17, 3):
            return {}
        return {
            index: (float(row[0]), float(row[1]))
            for index, row in enumerate(points)
            if np.all(np.isfinite(row)) and row[2] >= self.kp_conf
        }

    @staticmethod
    def _median_height(track: Track, ts: float) -> float | None:
        heights = [
            float(height)
            for sample_ts, _, _, height in track.history
            if isfinite(sample_ts)
            and sample_ts <= ts
            and isfinite(height)
            and height > 0
        ]
        return float(median(heights)) if heights else None

    @staticmethod
    def _dwell_s(
        track: Track,
        ts: float,
        polygon: list[tuple[float, float]],
    ) -> float:
        """Возраст текущего непрерывного пребывания, без first_seen.

        Текущая точка опоры уже проверена вызывающим кодом. Выход из зоны
        или недостоверная запись прерывает историю пребывания.
        """
        since = ts
        for sample_ts, x, y, _ in reversed(track.history):
            if not isfinite(sample_ts):
                break
            if sample_ts > ts:
                continue
            if (
                not isfinite(x)
                or not isfinite(y)
                or not point_in_polygon((x, y), polygon)
            ):
                break
            since = sample_ts
        return max(0.0, float(ts - since))

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts) or not ctx.people:
            return []

        width, height = ctx.size
        active = []
        for zone in self.zones:
            if not is_zone_active(zone, ctx.ts):
                continue
            polygon = zone_polygon_px(zone, width, height)
            if len(polygon) < 3 or not all(
                isfinite(x) and isfinite(y) for x, y in polygon
            ):
                continue
            bottom = max(y for _, y in polygon)
            midpoint = (min(y for _, y in polygon) + bottom) / 2.0
            active.append((zone, polygon, bottom, midpoint))

        out: list[Violation] = []
        for track in ctx.people:
            if not all(isfinite(value) for value in track.bbox):
                continue
            body_height = bbox_height(track.bbox)
            if (
                body_height <= 0
                or track.bbox[2] <= track.bbox[0]
                or not isfinite(track.conf)
            ):
                continue

            foot = foot_point(track.bbox)
            points = self._keypoints(track)
            median_height = self._median_height(track, ctx.ts)

            for zone, polygon, bottom, midpoint in active:
                details: dict[str, Any] | None = None
                lift = (bottom - foot[1]) / body_height
                if points and lift > self.lift_bh:
                    for index in (13, 14, 15, 16):
                        point = points.get(index)
                        if point is not None and point_in_polygon(point, polygon):
                            details = {
                                "condition": "climb",
                                "keypoint_index": index,
                                "lift_bh": float(lift),
                            }
                            break

                inside = point_in_polygon(foot, polygon)
                if details is None and inside:
                    if (
                        median_height is not None
                        and body_height < self.crouch_ratio * median_height
                    ):
                        details = {
                            "condition": "crawl",
                            "evidence": "height_ratio",
                            "height_ratio": float(body_height / median_height),
                            "median_height_px": median_height,
                        }
                    else:
                        nose = points.get(0) if points else None
                        if nose is not None and nose[1] > midpoint:
                            details = {
                                "condition": "crawl",
                                "evidence": "head_below_midpoint",
                                "nose_y": nose[1],
                                "polygon_mid_y": midpoint,
                            }

                    if details is None and points is None:
                        dwell_s = self._dwell_s(track, ctx.ts, polygon)
                        if dwell_s > self.max_dwell_s:
                            details = {
                                "condition": "dwell",
                                "dwell_s": dwell_s,
                            }

                if details is None:
                    continue

                details.update(
                    zone_name=zone.name,
                    zone_kind=zone.kind,
                    foot_xy=[float(foot[0]), float(foot[1])],
                )
                out.append(
                    self.violation(
                        ctx,
                        track_id=track.track_id,
                        zone_id=zone.zone_id,
                        bbox=tuple(float(value) for value in track.bbox),
                        score=float(track.conf),
                        details=details,
                    )
                )
                # Одно нарушение на человека за кадр, как у zone_intrusion.
                break

        return out
